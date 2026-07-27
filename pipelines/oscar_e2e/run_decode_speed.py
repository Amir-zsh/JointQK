#!/usr/bin/env python3
"""Serving-throughput sweep driver: arms x (input_len, batch_size) grid.

Reusable harness for the OSCAR throughput reproductions. An *arm* is one served
configuration (bf16 / OSCAR-INT2 / vq2 / ...); a *cell* is one (arm, input_len,
batch_size) point. Both panels of the paper's Figure 4 are the same measurement
with a different grid, so they share one code path:

    left  panel: batch_size fixed at 1, input_len varies (30k / 60k / 100k)
    right panel: input_len fixed at 100k, batch_size varies (1 / 8 / 32)

MEASUREMENT
    All cells are measured by sglang's own ``bench_serving`` with the
    ``generated-shared-prefix`` dataset -- the same tool and workload shape the
    paper's numbers come from. We deliberately do NOT hand-roll a client: a
    bespoke measurement path is one more place for our own bugs, and the headline
    metric (output token throughput) is what the paper plots.

PREFILL MUST NOT DOMINATE
    Two mechanisms, both checked rather than assumed:
      1. ``generated-shared-prefix`` gives every concurrent request the same long
         system prompt, and ``bench_serving``'s warmup request sends the real
         first prompt (``input_requests[0]``) without flushing the cache
         afterwards. So the shared prefix is resident before the measured run.
         This is also what makes BS=32 at 100k fit at all: 32 x 100k = 3.2M
         tokens would exceed the pool, but a shared prefix is stored once.
      2. Every cell computes
             prefill_share = TTFT / (TTFT + (ngen-1) * TPOT)
         and FAILS above ``MAX_PREFILL_SHARE``. If the prefix were not reused the
         cell degenerates into a prefill benchmark, and we would rather the run
         refuse than quietly report the wrong quantity.

CONCURRENCY
    Server boots are parallelised across GPUs (not timing-sensitive). Measurement
    is serialized by default: boards share a power/thermal budget and the client
    is timed host-side, so co-tenancy can perturb exactly what we measure.
    ``--concurrency-check`` quantifies that cost; ``--parallel-measure`` opts in.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONTAINER = "oscar-ab"
BOOT_TIMEOUT_S = 1800
GPU_FREE_MIB = 2000

# A warm shared prefix should make prefill a rounding error in per-request wall
# time. Above this share the number is not measuring decode.
MAX_PREFILL_SHARE = 0.20

_BENCH_FIELDS = {
    "output_tok_s": "Output token throughput (tok/s):",
    "total_tok_s": "Total token throughput (tok/s):",
    "mean_ttft_ms": "Mean TTFT (ms):",
    "median_ttft_ms": "Median TTFT (ms):",
    "mean_tpot_ms": "Mean TPOT (ms):",
    "median_tpot_ms": "Median TPOT (ms):",
    "duration_s": "Benchmark duration (s):",
    "request_throughput": "Request throughput (req/s):",
}


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gpu_used_mib(gpu: int) -> int:
    r = sh(["nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits", "-i", str(gpu)])
    try:
        return int(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 10**9


def wait_gpu_free(gpu: int, timeout_s: int = 300) -> bool:
    """A fixed sleep is not a guarantee: sglang children outlive pkill and
    freeing tens of GB of CUDA is not instant. Poll instead."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if gpu_used_mib(gpu) < GPU_FREE_MIB:
            return True
        time.sleep(5)
    return False


def cell_id(point: dict) -> str:
    base = f"in{point['input_len']}_bs{point['batch_size']}"
    return base + (f"_g{point['groups']}" if "groups" in point else "")


class Server:
    """One served arm, pinned to one GPU."""

    def __init__(self, container, arm, gpu, port, serve, log_dir):
        self.container, self.arm, self.gpu, self.port = container, arm, gpu, port
        self.serve = serve
        # ONE path for both sides: docker-compose maps the repo at an identical
        # absolute path inside the container (${REPO_ROOT}:${REPO_ROOT}), so the
        # server writes and we poll the same file. Pointing the container at
        # /scratch and polling logs/ silently waits on an empty file forever.
        # Include the GPU: the concurrency check runs one arm on every GPU.
        self.log = log_dir / f"serve_{arm['label']}_gpu{gpu}.log"

    def _kill(self):
        sh(["docker", "exec", self.container, "bash", "-lc",
            f"pkill -f '[l]aunch_server.*--port {self.port}'"])

    def start(self):
        self._kill()
        wait_gpu_free(self.gpu)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        if self.log.exists():
            self.log.unlink()
        self.log.touch()
        env = ["-e", f"RADIX_CACHE={self.serve.get('radix_cache', 1)}"]
        if "max_tokens" in self.serve:
            env += ["-e", f"MAX_TOKENS={self.serve['max_tokens']}"]
        # Prefix reuse in the mixed-KV pool only commits the FIRST prefill chunk:
        # measured max cached = chunked_prefill_size - RECENT_TOKENS regardless of
        # prompt length (7936 = 8192-256 at the default, 29728 of a 30k prompt at
        # 32768). So the chunk size must exceed the shared prefix or the paper's
        # "full prefix-cache hit" protocol silently degrades to a re-prefill.
        if self.serve.get("extra"):
            env += ["-e", f"SERVE_EXTRA={self.serve['extra']}"]
        # OSCAR's own eval driver (rotation/eval_oscar_gpqa.sh) serves with fa3
        # prefill + triton decode; match it so serving-side numbers are comparable.
        if self.serve.get("prefill_backend"):
            env += ["-e", f"PREFILL_BACKEND={self.serve['prefill_backend']}"]
        for k, v in self.arm.get("env", {}).items():
            env += ["-e", f"{k}={v}"]
        cmd = (
            f"cd {REPO} && bash pipelines/oscar_e2e/serve_oscar.sh "
            f"--gpu {self.gpu} --port {self.port} "
            f"--ctx {self.serve.get('ctx', 131072)} "
            f"--mem-frac {self.serve.get('mem_frac', 0.85)} "
            f"{self.arm.get('flags', '')} >> {self.log} 2>&1"
        )
        sh(["docker", "exec", "-d", *env, self.container, "bash", "-lc", cmd])
        return self

    def wait_ready(self, timeout_s: int = BOOT_TIMEOUT_S) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            txt = self.log.read_text(errors="ignore") if self.log.exists() else ""
            if "The server is fired up" in txt:
                return True
            for bad in ("CUDA out of memory", "Received sigquit", "Traceback"):
                if bad in txt:
                    print(f"  [{self.arm['label']} gpu{self.gpu}] BOOT FAILED ({bad})",
                          flush=True)
                    return False
            time.sleep(5)
        print(f"  [{self.arm['label']} gpu{self.gpu}] BOOT TIMEOUT after {timeout_s}s",
              flush=True)
        return False

    def provenance(self) -> dict:
        """Record what actually loaded rather than trusting the flags we passed.
        A control arm that silently ran the wrong mode has already burned us
        once; the giveaway was in this log, not in the flags."""
        txt = self.log.read_text(errors="ignore") if self.log.exists() else ""

        def grab(pat):
            m = re.search(pat, txt)
            return m.group(1) if m else None

        return {
            "kv_cache_dtype": grab(r"kv_cache_dtype='([^']*)'"),
            "num_kv_splits": grab(r"triton_attention_num_kv_splits=([^,]*)"),
            "max_total_num_tokens": grab(r"max_total_num_tokens=(\d+)"),
            "vq_codebook_loaded": "vq2: loaded codebook" in txt,
            "radix_cache_disabled": "--disable-radix-cache" in txt,
        }

    def alive(self) -> bool:
        """Is the server still up? Cheap /health probe plus a log scan, so one
        crash does not masquerade as a run of independent cell failures."""
        r = sh(["docker", "exec", self.container, "bash", "-lc",
                f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 10 "
                f"http://127.0.0.1:{self.port}/health"])
        if r.stdout.strip() == "200":
            return True
        txt = self.log.read_text(errors="ignore") if self.log.exists() else ""
        for bad in ("CUDA out of memory", "SIGQUIT", "Received sigquit"):
            if bad in txt:
                print(f"  [{self.arm['label']} gpu{self.gpu}] SERVER DIED ({bad}) "
                      f"-- skipping this arm's remaining cells", flush=True)
                return False
        return False

    def stop(self):
        self._kill()
        wait_gpu_free(self.gpu)


def measure(server: Server, point: dict, cfg: dict, out_dir: Path) -> dict:
    """One (input_len, batch_size) cell via sglang bench_serving."""
    inp, bs = point["input_len"], point["batch_size"]
    cell = out_dir / server.arm["label"] / cell_id(point)
    cell.mkdir(parents=True, exist_ok=True)
    ngen = cfg.get("ngen", 1000)
    base = (
        f"cd {REPO} && PYTHONPATH={REPO}/vendor/OSCAR-vq/sglang-research/python "
        f"$OSCAR_PYTHON -m sglang.bench_serving --backend sglang "
        f"--host 127.0.0.1 --port {server.port} "
    )
    # Two workloads, and the choice decides WHAT the benchmark measures:
    #
    #  shared_prefix -- all BS requests share one long system prompt. The context
    #      is stored ONCE regardless of precision, so KV capacity stops binding
    #      and the run measures decode efficiency only. This erases the memory
    #      advantage INT2 exists to provide.
    #  replay -- BS DISTINCT prompts, the whole set run twice; the second run is
    #      measured, so every request hits its own cached KV ("immediate replay
    #      after warmup", the protocol named in the paper's Figure 6 caption).
    #      The pool must hold BS x input_len, so capacity binds hard: at 32x100k
    #      = 3.2M tokens BF16 (~374k pool) holds ~3 while INT2 (~2.4M) holds ~23.
    #      That capacity gap is what Figure 4 (right) is actually showing.
    if cfg.get("workload", "shared_prefix") == "replay":
        cmd = (base + f"--dataset-name random --random-input-len {inp} "
               f"--random-output-len {ngen} --random-range-ratio 1.0 "
               f"--num-prompts {bs} --max-concurrency {bs} "
               f"--seed {cfg.get('seed', 1)} --warmup-requests 0")
        sh(["docker", "exec", server.container, "bash", "-lc", cmd])  # populate
        r = sh(["docker", "exec", server.container, "bash", "-lc", cmd])  # measured
    else:
        groups = int(point.get("groups", cfg.get("gsp_num_groups", 1)))
        groups = max(1, min(groups, bs))
        per_group = max(1, bs // groups)
        cmd = (base + f"--dataset-name generated-shared-prefix "
               f"--gsp-num-groups {groups} --gsp-prompts-per-group {per_group} "
               f"--gsp-system-prompt-len {inp} "
               f"--gsp-question-len {cfg.get('question_len', 128)} "
               f"--gsp-output-len {ngen} "
               f"--num-prompts {groups * per_group} --max-concurrency {bs} "
               f"--warmup-requests {cfg.get('warmup_requests', 1)}")
        r = sh(["docker", "exec", server.container, "bash", "-lc", cmd])
    text = r.stdout + r.stderr
    (cell / "bench_serving.log").write_text(text)

    m = {"label": server.arm["label"], "input_len": inp, "batch_size": bs,
         "ngen": ngen, "ok": False}
    for key, needle in _BENCH_FIELDS.items():
        for line in text.splitlines():
            if needle in line:
                try:
                    m[key] = float(line.split(needle)[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
                break

    # Two throughput notions, both kept, because they differ in how they treat
    # prefill and the paper uses one of each:
    #   output_tok_s  = total_output / benchmark_duration -> INCLUDES prefill in
    #                   the denominator. This is "job-level throughput" (Fig 4
    #                   right) and is what a serving operator sees.
    #   decode_tok_s  = 1000 / TPOT, and sglang defines
    #                   TPOT = (latency - ttft) / (out_len - 1), so a request's
    #                   own prefill is EXCLUDED. This is "decode throughput"
    #                   (Fig 4 left) and isolates kernel efficiency.
    # With a warm shared prefix the two should nearly coincide; a gap between
    # them means prefill is still intruding, which is why both are reported.
    if m.get("mean_tpot_ms"):
        m["decode_tok_s"] = 1000.0 / m["mean_tpot_ms"]
    if "output_tok_s" in m and m.get("mean_tpot_ms"):
        decode_ms = (ngen - 1) * m["mean_tpot_ms"]
        ttft = m.get("mean_ttft_ms", 0.0)
        m["prefill_share"] = ttft / (ttft + decode_ms) if (ttft + decode_ms) else 1.0
        # `ok` means the measurement itself succeeded. Whether a high prefill
        # share INVALIDATES the cell depends on the metric, so that judgement
        # lives in cell_ok() at aggregation time: decode_tok_s is prefill-free by
        # construction (TPOT subtracts the request's own TTFT), while
        # output_tok_s divides by a duration that contains prefill.
        m["ok"] = True
        m["prefill_ok"] = m["prefill_share"] <= MAX_PREFILL_SHARE
    else:
        m["error"] = "bench_serving produced no throughput line"

    (cell / "metrics.json").write_text(json.dumps(m, indent=2))
    status = "" if m["ok"] else f"  FAILED ({m.get('error')})"
    print(f"  {server.arm['label']:<12} in={inp:>6} bs={bs:<3} "
          f"{m.get('output_tok_s', float('nan')):8.1f} out tok/s  "
          f"TTFT={m.get('mean_ttft_ms', float('nan')):8.0f}ms  "
          f"TPOT={m.get('mean_tpot_ms', float('nan')):6.2f}ms  "
          f"prefill={m.get('prefill_share', float('nan')):.3f}{status}", flush=True)
    return m


def aggregate(out_dir: Path, cfg: dict) -> dict:
    """Throughput and speedup-vs-reference tables, the form Figure 4 reports."""
    grid = cfg["grid"]
    labels = [a["label"] for a in cfg["arms"]]
    ref = cfg.get("reference")
    metric = cfg.get("metric", "output_tok_s")
    # Always carry both throughput notions plus prefill_share, so a reader can
    # see whether the prefill-inclusive and prefill-free numbers agree.
    tracked = ["output_tok_s", "decode_tok_s", "prefill_share"]

    def cell_ok(c: dict) -> bool:
        """Is this cell usable FOR THE CONFIGURED METRIC?

        output_tok_s divides by a wall-clock duration that includes prefill, so a
        prefill-heavy cell is not measuring serving throughput and is rejected.
        decode_tok_s is 1000/TPOT and sglang's TPOT subtracts the request's own
        TTFT, so prefill cannot leak into it at bs=1 -- rejecting those cells
        would throw away valid decode measurements (which it originally did).
        """
        if not c.get("ok") or c.get(metric) is None:
            return False
        # Only enforce the prefill gate where we EXPECT a full cache hit: the
        # shared-prefix protocol with a single group. Anywhere capacity can bind
        # (replay, or >1 distinct prefix) a high prefill share is the RESULT --
        # the arm is thrashing because its pool cannot hold the working set --
        # and rejecting it discards exactly the cells that carry the finding.
        # This gate previously hid the entire groups sweep.
        capacity_bound = (cfg.get("workload") == "replay"
                          or any(int(g.get("groups", 1)) > 1 for g in cfg["grid"]))
        if metric == "output_tok_s" and not capacity_bound:
            return c.get("prefill_share", 1.0) <= MAX_PREFILL_SHARE
        # Under the replay workload a HIGH prefill share is the result, not a
        # measurement fault: an arm whose pool cannot hold BS x input_len must
        # re-prefill, and job-level throughput is meant to include exactly that
        # cost. Gating it here would reject the very cell that demonstrates the
        # capacity advantage (bf16 holds ~3 of 32 distinct 100k prompts).
        return True

    def collect(key):
        out = {}
        for lab in labels:
            out[lab] = {}
            for p in grid:
                f = out_dir / lab / cell_id(p) / "metrics.json"
                c = json.loads(f.read_text()) if f.exists() else {}
                out[lab][cell_id(p)] = c.get(key) if cell_ok(c) else None
        return out

    tables = {k: collect(k) for k in tracked}
    vals = tables[metric]

    summary = {"study": cfg.get("study"), "metric": metric, "ngen": cfg.get("ngen"),
               "grid": grid, "reference": ref, "speedup_vs_reference": {}, **tables}
    if ref:
        for lab in labels:
            summary["speedup_vs_reference"][lab] = {
                cell_id(p): (round(vals[lab][cell_id(p)] / vals[ref][cell_id(p)], 3)
                             if vals[lab][cell_id(p)] and vals.get(ref, {}).get(cell_id(p))
                             else None)
                for p in grid
            }

    (out_dir / "decode_speed_summary.json").write_text(json.dumps(summary, indent=2))

    w = max(len(l) for l in labels) + 2
    cols = [cell_id(p) for p in grid]
    print(f"\n=== {cfg.get('study')} — {metric}, ngen={cfg.get('ngen')} ===")
    for title, table in (
        ("output_tok_s (incl prefill)", tables["output_tok_s"]),
        ("decode_tok_s (excl prefill)", tables["decode_tok_s"]),
        ("prefill_share", tables["prefill_share"]),
        (f"speedup vs {ref} [{metric}]", summary["speedup_vs_reference"]),
    ):
        if not table:
            continue
        print(f"\n{title:<{w}}" + "".join(f"{c:>18}" for c in cols))
        for lab in labels:
            print(f"{lab:<{w}}" + "".join(
                f"{table[lab][c]:18.2f}" if table[lab][c] is not None else f"{'--':>18}"
                for c in cols))
    return summary


def concurrency_check(cfg, args, out_dir, log_dir):
    """Does co-tenancy perturb the measurement? Run one arm alone, then the same
    arm on every GPU at once, and report the inflation. Settles whether
    --parallel-measure is safe instead of guessing either way."""
    arm, point = cfg["arms"][0], cfg["grid"][len(cfg["grid"]) // 2]
    print(f"[concurrency-check] arm={arm['label']} cell={cell_id(point)} gpus={args.gpus}")
    servers = [Server(args.container, arm, g, args.base_port + i, cfg["serve"], log_dir).start()
               for i, g in enumerate(args.gpus)]
    ready = [s for s in servers if s.wait_ready()]
    if not ready:
        print("[concurrency-check] no server booted"); return
    solo = measure(ready[0], point, cfg, out_dir / "_concurrency" / "solo")
    with ThreadPoolExecutor(max_workers=len(ready)) as ex:
        res = list(ex.map(
            lambda s: measure(s, point, cfg, out_dir / "_concurrency" / f"par_gpu{s.gpu}"),
            ready))
    for s in servers:
        s.stop()
    par = [r["output_tok_s"] for r in res if r.get("ok")]
    if solo.get("ok") and par:
        med = statistics.median(par)
        ratio = solo["output_tok_s"] / med
        print(f"[concurrency-check] solo {solo['output_tok_s']:.1f} tok/s | "
              f"{len(par)}-way parallel median {med:.1f} tok/s | "
              f"slowdown {ratio:.3f}x")
        print("[concurrency-check] parallel measurement is safe if ~1.00; "
              "above ~1.02 -> serialize.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",")], default=[0])
    ap.add_argument("--base-port", type=int, default=31000)
    ap.add_argument("--container", default=DEFAULT_CONTAINER)
    ap.add_argument("--shard-cells", action="store_true",
                    help="split each arm's cells across spare GPUs (boots the arm "
                         "more than once) so parallelism is not capped by arm count")
    ap.add_argument("--parallel-measure", action="store_true",
                    help="measure arms concurrently; run --concurrency-check first")
    ap.add_argument("--concurrency-check", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    out_dir = args.out or (REPO / "artifacts/oscar_e2e" / cfg["study"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = REPO / "logs" / cfg["study"]
    log_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = REPO / "logs" / f"{cfg['study']}.heartbeat"

    if args.aggregate_only:
        aggregate(out_dir, cfg); return 0
    if args.concurrency_check:
        concurrency_check(cfg, args, out_dir, log_dir); return 0

    shutil.copy(args.config, out_dir / "config.json")
    arms, gpus, grid = cfg["arms"], args.gpus, cfg["grid"]

    # A GPU hosts one server, so naive scheduling is bounded by ARM count: 3 arms
    # leaves 5 of 8 GPUs idle no matter how many cells there are. Sharding splits
    # each arm's cells across several GPUs (booting that arm more than once), so
    # wall time falls to roughly one boot + one cell instead of one boot + all
    # cells. Safe here because the concurrency check measured 0.983x -- i.e. no
    # cross-GPU perturbation of the metric.
    tasks: list[tuple[dict, list[dict]]] = []
    if args.shard_cells and len(arms) < len(gpus):
        per_arm = max(1, min(len(grid), len(gpus) // len(arms)))
        for a in arms:
            step = -(-len(grid) // per_arm)  # ceil
            for k in range(0, len(grid), step):
                tasks.append((a, grid[k:k + step]))
    else:
        tasks = [(a, grid) for a in arms]

    print(f"[decode-speed] study={cfg['study']} arms={len(arms)} cells/arm={len(grid)} "
          f"gpus={gpus} tasks={len(tasks)} "
          f"measure={'parallel' if args.parallel_measure else 'serialized'}")

    for i in range(0, len(tasks), len(gpus)):
        wave = tasks[i:i + len(gpus)]
        servers = [Server(args.container, a, gpus[j], args.base_port + j,
                          cfg["serve"], log_dir).start()
                   for j, (a, _) in enumerate(wave)]
        points_for = {id(s): pts for s, (_, pts) in zip(servers, wave)}
        ready = [s for s in servers if s.wait_ready()]
        for s in ready:
            prov = s.provenance()
            (out_dir / s.arm["label"]).mkdir(parents=True, exist_ok=True)
            (out_dir / s.arm["label"] / "provenance.json").write_text(json.dumps(prov, indent=2))
            print(f"  [{s.arm['label']}] gpu={s.gpu} {prov}", flush=True)

        def run_arm(s):
            for p in points_for[id(s)]:
                measure(s, p, cfg, out_dir)
                heartbeat.touch()

        if args.parallel_measure:
            with ThreadPoolExecutor(max_workers=len(ready)) as ex:
                list(ex.map(run_arm, ready))
        else:
            for s in ready:
                run_arm(s)
        for s in servers:
            s.stop()

    aggregate(out_dir, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
