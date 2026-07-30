#!/usr/bin/env python3
"""Do the quant arms still generate coherent text after the SWA allocation fix?

No throughput gate can answer this. SWA slot aliasing -- two full-tier ids
mapping to one live slot -- corrupts only the sliding-window layers, which shows
up as degraded text, not as a crash, a retraction or a failed cache report. And
the throughput harness cannot answer it either: its codebook and rotations are
random by construction, so its output is meaningless whether or not the pool is
correct.

So this boots with the REAL gpt-oss artifacts (OSCAR rotations, and the real VQ
codebook for vq2), generates on real prompts, and prints the text for reading.

What counts as a pass: FLUENT text. OSCAR int2 on gpt-oss scores 0.22-0.25 on
MATH-500 against bf16's 0.91, a quantization-quality problem that predates the
SWA work entirely -- so wrong-but-fluent answers are expected and are NOT a
failure here. Word salad, repetition loops, or wrong-language output would
indicate aliasing.

The prompts deliberately span the flush boundary: SGLANG_MIXED_KV_RECENT_TOKENS
is 256 by default, so a long prompt forces tokens through the HP->quant flush
path where free_swa now runs, which a short prompt would never exercise.

  python pipelines/throughput/check_swa_coherence.py --arms bf16,oscar_int2,vq2 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
CONTAINER = "oscar-ab"
ROT = "artifacts/oscar_gptoss20b_mxfp4/rotations_gpqa198"
VQ_CB = "artifacts/oscar_gptoss20b/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt"

PROMPTS = [
    ("short", "Explain in three sentences why the sky appears blue."),
    # Long enough to push tokens past hp_recent_tokens (256) into the quant
    # tier, so the flush + free_swa path is exercised before the answer.
    ("long-recall",
     "Here is a list of facts.\n"
     + "\n".join(f"Fact {i}: the code word for city number {i} is {w}."
                 for i, w in enumerate(
                     ["alpha", "bravo", "cinnamon", "delta", "echo", "foxglove",
                      "granite", "harbor", "indigo", "juniper", "kestrel",
                      "lantern", "marigold", "nutmeg", "obsidian", "pewter",
                      "quartz", "rosemary", "saffron", "thistle", "umber",
                      "verbena", "walnut", "xenon", "yarrow", "zephyr"], 1))
     + "\n\nQuestion: what is the code word for city number 7? "
       "Answer with the word, then explain how you found it."),
]

ARMS = {
    "bf16": {"flags": "--bf16", "env": {}},
    "oscar_int2": {"flags": "", "env": {}},
    "vq2": {"flags": f"--vq2 --vq-codebook {VQ_CB}",
            "env": {"SGLANG_VQ_FP8_FMT": "e5m2", "SGLANG_VQ_OPT_QMAP": "1",
                    "SGLANG_VQ_OPT_KMAP": "1", "SGLANG_VQ_OPT_FLUSH": "1",
                    "SGLANG_VQ_OPT_PREFILL": "1"}},
}


def sh(cmd: list[str]) -> None:
    subprocess.run(cmd, check=False)


def boot(arm: str, gpu: int, port: int, log: Path, swa_pin: str) -> None:
    # Port-scoped, matching kill_own() in repro_chunk_prefix_cache.py and the
    # teardown in run_throughput.py. An unscoped pkill tears down every server in
    # the container, so this script could not share the box with any other run --
    # it silently killed a concurrent MATH-500 pair, and a launcher waiting on it
    # deadlocked. The [l] bracket keeps the pattern from matching its own shell.
    sh(["docker", "exec", CONTAINER, "bash", "-lc",
        f"pkill -9 -f '[l]aunch_server.*--port {port}'"])
    time.sleep(6)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    env = {
        "MODEL": "openai/gpt-oss-20b", "MAX_TOKENS": "auto", "RADIX_CACHE": "1",
        "MAX_REQS": "8", "CUDA_GRAPH_BS": "8", "PREFILL_BACKEND": "triton",
        "TP": "1", "ROT_DIR": str(REPO / ROT), "KV_SPLITS": "8",
        "ABSORB_V_ROT": "0", "QUANT_GROUP_SIZE": "0",
        "SERVE_EXTRA": ("--enable-cache-report --chunked-prefill-size 8192 "
                        "--moe-runner-backend triton_kernel"),
    }
    if arm != "bf16":
        env["SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS"] = "1"
        env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
        # Opt-in probe: assert no HP-recent slot id ever lands in the radix tree
        # (they are per-req_pool_idx and alias across requests). This script is
        # the natural carrier -- gpt-oss + RADIX_CACHE=1 + mixed KV is exactly
        # the combination the check needs, and the chunked-prefill repro is not
        # (it serves Qwen, which has no SWA tier at all).
        if os.environ.get("SGLANG_CHECK_MIXED_KV_SWA"):
            env["SGLANG_CHECK_MIXED_KV_SWA"] = os.environ["SGLANG_CHECK_MIXED_KV_SWA"]
        if swa_pin:
            env["SGLANG_SWA_POOL_TOKENS"] = swa_pin
    env.update(ARMS[arm]["env"])
    eargs = []
    for k, v in env.items():
        eargs += ["-e", f"{k}={v}"]
    cmd = (f"cd {REPO} && bash pipelines/oscar_e2e/serve_oscar.sh --gpu {gpu} "
           f"--port {port} --ctx 32768 --mem-frac 0.85 "
           f"{ARMS[arm]['flags']} >> {log} 2>&1")
    sh(["docker", "exec", "-d", *eargs, CONTAINER, "bash", "-lc", cmd])


def wait_ready(log: Path, port: int, timeout: int = 900) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        t = log.read_text(errors="ignore")
        if "The server is fired up" in t:
            return True
        if "Traceback" in t and "Ignore import error" not in t.split("Traceback")[-1][:200]:
            return False
        time.sleep(4)
    return False


def generate(port: int, prompt: str, n: int) -> str:
    r = requests.post(f"http://127.0.0.1:{port}/generate", timeout=600, json={
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": n},
    })
    r.raise_for_status()
    d = r.json()
    return (d[0] if isinstance(d, list) else d).get("text", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bf16,oscar_int2,vq2")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=41000)
    ap.add_argument("--tokens", type=int, default=140)
    ap.add_argument("--swa-pin", default="1048576")
    a = ap.parse_args()

    out: dict[str, dict] = {}
    for arm in [s for s in a.arms.split(",") if s]:
        log = REPO / "logs/swa_coherence" / f"serve_{arm}.log"
        print(f"\n{'=' * 72}\n{arm}\n{'=' * 72}", flush=True)
        boot(arm, a.gpu, a.port, log, "" if arm == "bf16" else a.swa_pin)
        if not wait_ready(log, a.port):
            print(f"  BOOT FAILED -- see {log}", flush=True)
            out[arm] = {"boot": "failed"}
            continue
        out[arm] = {"boot": "ok"}
        for tag, p in PROMPTS:
            try:
                txt = generate(a.port, p, a.tokens)
            except Exception as e:
                txt = f"<request failed: {type(e).__name__}: {e}>"
            out[arm][tag] = txt
            print(f"\n--- {arm} / {tag} ---\n{txt.strip()[:900]}\n", flush=True)
        # The assertion added to the runtime checker fires on negative SWA
        # occupancy; surface it here rather than leaving it in the serve log.
        t = log.read_text(errors="ignore")
        neg = t.count("swa token usage: -")
        bad = "SWA accounting invariant violated" in t
        print(f"  [{arm}] negative-SWA log lines: {neg} | invariant assertion "
              f"fired: {bad}", flush=True)
        out[arm]["negative_swa_lines"] = neg
        out[arm]["invariant_violated"] = bad

    # Port-scoped, matching kill_own() in repro_chunk_prefix_cache.py and the
    # teardown in run_throughput.py. An unscoped pkill tears down every server in
    # the container, so this script could not share the box with any other run --
    # it silently killed a concurrent MATH-500 pair, and a launcher waiting on it
    # deadlocked. The [l] bracket keeps the pattern from matching its own shell.
    sh(["docker", "exec", CONTAINER, "bash", "-lc",
        f"pkill -9 -f '[l]aunch_server.*--port {a.port}'"])
    dst = REPO / "artifacts/throughput/swa_coherence.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
