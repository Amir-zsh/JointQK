"""Kernel-level timeline of one server's decode, per arm.

The vq2_cuda arm measured 1.6-8.3x end-to-end, which cannot be the kernel: a
liveness probe that makes the kernel 1.9x slower per call left throughput
unchanged. So something else in the server responds to SGLANG_VQ2_CUDA=1, and
guessing has not found it. This asks the server directly.

sglang exposes the torch profiler over HTTP (`/start_profile`, `/stop_profile`).
This boots one server per arm with the harness's own Server class, warms a
prompt, profiles a short decode, and prints the top CUDA kernels by total time.
Comparing the two lists shows exactly which kernels run and where the step time
goes -- including whether the vq2 stage-1 kernel appears at all.

  python pipelines/throughput/kernel_study/profile_server_decode.py --arm vq2_s8
  python pipelines/throughput/kernel_study/profile_server_decode.py --arm vq2_cuda
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "pipelines/throughput"))

from run_throughput import Server  # noqa: E402


def top_kernels(trace_dir: Path, limit: int = 18) -> list[tuple[str, float, int]]:
    files = sorted(trace_dir.glob("*.trace.json*"), key=lambda f: f.stat().st_mtime)
    if not files:
        return []
    f = files[-1]
    raw = gzip.open(f, "rt").read() if f.suffix == ".gz" else f.read_text()
    ev = json.loads(raw).get("traceEvents", [])
    tot: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    for e in ev:
        # cat="kernel" is the device-side execution, which is what we want;
        # cpu_op would double-count launch overhead.
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            tot[e["name"]] += e.get("dur", 0)
            cnt[e["name"]] += 1
    return sorted(((k, v / 1e3, cnt[k]) for k, v in tot.items()),
                  key=lambda x: -x[1])[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO / "pipelines/throughput/configs/qwen3_8b.json")
    ap.add_argument("--ctx", type=int, default=30000)
    ap.add_argument("--out", type=int, default=64)
    ap.add_argument("--port", type=int, default=32500)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=1,
                    help="concurrent requests; the kernel-vs-end-to-end\n"
                         "discrepancy only appears at large batch")
    ap.add_argument("--max-reqs", type=int, default=None)
    ap.add_argument("--graph-bs", type=int, default=None)
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    # The vq2_cuda arm is intentionally absent from the shipped configs (it is
    # not validated end-to-end); synthesise it here from vq2_s8 so this probe
    # does not depend on the config carrying an arm we do not want charted.
    arms = {x["label"]: x for x in cfg["arms"]}
    if a.arm == "vq2_cuda":
        arm = json.loads(json.dumps(arms["vq2_s8"]))
        arm["label"] = "vq2_cuda"
        arm["env"]["SGLANG_VQ2_CUDA"] = "1"
    else:
        arm = arms[a.arm]
    trace_dir = REPO / "logs/decode_profile" / f"{a.arm}_bs{a.bs}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for old in trace_dir.glob("*.trace.json*"):
        old.unlink()

    log_dir = REPO / "logs/decode_profile"
    srv = Server(arm, cfg, [a.gpu], a.port, log_dir,
                 max_reqs=a.max_reqs or max(4, a.bs),
                 graph_bs=a.graph_bs or max(4, a.bs), tag="prof")
    srv.arm = dict(arm)
    srv.arm.setdefault("env", {})
    srv.arm["env"] = dict(arm.get("env", {}))
    srv.arm["env"]["SGLANG_TORCH_PROFILER_DIR"] = str(trace_dir)
    srv.start(cfg["serve"]["mem_frac"])
    ok, why = srv.wait_ready()
    if not ok:
        print(f"server failed: {why}")
        return 1
    base = f"http://127.0.0.1:{a.port}"
    try:
        from concurrent.futures import ThreadPoolExecutor
        # Distinct prompts, as the benchmark does -- a shared prefix would let
        # the radix cache collapse the batch and change what is measured.
        def body_for(r, n):
            ids = [(1234 + r * 7919 + i) % 100000 for i in range(a.ctx)]
            return {"input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": n,
                                        "ignore_eos": True}}

        def fire(n):
            with ThreadPoolExecutor(max_workers=a.bs) as ex:
                list(ex.map(lambda r: requests.post(f"{base}/generate",
                                                    json=body_for(r, n), timeout=2400),
                            range(a.bs)))

        fire(1)                                        # warm every prefix
        requests.post(f"{base}/start_profile", json={"num_steps": None}, timeout=60)
        t0 = time.time()
        fire(a.out)
        wall = time.time() - t0
        requests.post(f"{base}/stop_profile", timeout=120)
        time.sleep(12)          # the profiler writes the trace asynchronously
    finally:
        srv.stop()

    ks = top_kernels(trace_dir)
    print(f"\n=== {a.arm}: bs={a.bs}, {a.out} steps in {wall:.2f}s "
          f"({a.bs * a.out / wall:.1f} tok/s aggregate) ===")
    if not ks:
        print(f"no trace written under {trace_dir}")
        return 1
    print(f"{'kernel':<78} {'ms':>9} {'calls':>7} {'us/call':>9}")
    for name, ms, n in ks:
        print(f"{name[:78]:<78} {ms:>9.2f} {n:>7} {ms * 1e3 / max(n, 1):>9.1f}")
    print(f"{'TOTAL (top listed)':<78} {sum(k[1] for k in ks):>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
