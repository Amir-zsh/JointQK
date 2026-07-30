#!/usr/bin/env python3
"""Minimal reproduction: mixed-KV prefix reuse is capped at the FIRST prefill chunk.

THE CLAIM UNDER TEST
    Send an identical prompt twice to a warm server with the radix (prefix) cache
    enabled. The second request should be served almost entirely from cache, i.e.
    ``#cached-token ~= prompt_len``. On OSCAR's mixed-KV (INT2 + BF16 sink/recent)
    pool it instead caps at ``chunked_prefill_size - hp_recent_tokens``, no matter
    how long the prompt is. BF16 on the same server flags does not cap.

WHY (suspected root cause, which this repro is designed to confirm or refute)
    ``mem_cache/common.py`` folds two different constraints into one variable,
    ``req.mixed_kv_quant_slack_cutoff_len``, using a running ``min``:
      A (:543) non-final-chunk HP-recent tail bound -- purely positional, GROWS
        with seq_len, no persistent ownership behind it;
      B (:563, :590) partial-quant-page ownership -- backed by accumulating
        ``mixed_kv_quant_slack_indices``, so persisting IS correct.
    Because A is min-accumulated it latches at chunk 1's value and never advances.

CELLS
    Three cells isolate the variable. If the hypothesis holds:
      int2 / chunk 8192   -> capped at ~7936      (BUG)
      int2 / chunk 32768  -> ~full prompt          (prompt fits one chunk, so the
                                                    non-final branch never runs)
      bf16 / chunk 8192   -> ~full prompt          (control: not mixed-KV)

Usage:
    python pipelines/oscar_e2e/repro_chunk_prefix_cache.py [--gpu 0] [--prompt-len 30000]
Exit code 0 = bug reproduced (or, after a fix, all cells cache fully -- see the
verdict line, which states which of the two it is).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTAINER = "oscar-ab"
RECENT_TOKENS = 256          # SGLANG_MIXED_KV_RECENT_TOKENS as set by serve_oscar.sh
FULL_CACHE_FRACTION = 0.90   # "cached ~= prompt" tolerance


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def kill_own(port):
    """Kill only OUR server. An unscoped pkill lets two concurrent invocations
    (e.g. fix-on and fix-off on different GPUs) tear down each other's servers."""
    sh(["docker", "exec", CONTAINER, "bash", "-lc",
        f"pkill -f '[l]aunch_server.*--port {port}'"])


def wait_gpu_free(gpu, timeout_s=300):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = sh(["nvidia-smi", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits", "-i", str(gpu)])
        try:
            if int(r.stdout.strip().splitlines()[0]) < 2000:
                return True
        except (ValueError, IndexError):
            pass
        time.sleep(5)
    return False


def run_cell(gpu, port, mode, chunk, prompt_len, extra_env=None):
    """Boot one server, send the same prompt twice, return (cached, ttft_cold, ttft_warm)."""
    # Port-tagged: two concurrent invocations (fix-on / fix-off) otherwise
    # write the SAME file and each parses the other's #cached-token lines.
    log = REPO / "logs" / f"repro_chunk_{mode}_{chunk}_p{port}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        log.unlink()
    log.touch()

    kill_own(port)
    wait_gpu_free(gpu)

    env = ["-e", "RADIX_CACHE=1", "-e", "MAX_TOKENS=600000",
           "-e", f"SERVE_EXTRA=--chunked-prefill-size {chunk} --enable-cache-report"]
    # Forward the strict-mem-check level when the caller sets it, so this repro
    # can double as the carrier for the (temporary) HP-recent-in-tree assertion.
    for passthru in ("SGLANG_CHECK_MIXED_KV_SWA",):
        if os.environ.get(passthru):
            env += ["-e", f"{passthru}={os.environ[passthru]}"]
    for k, v in (extra_env or {}).items():
        env += ["-e", f"{k}={v}"]
    flags = "--bf16" if mode == "bf16" else ""
    sh(["docker", "exec", "-d", *env, CONTAINER, "bash", "-lc",
        f"cd {REPO} && bash pipelines/oscar_e2e/serve_oscar.sh --gpu {gpu} "
        f"--port {port} --ctx 131072 --mem-frac 0.82 {flags} >> {log} 2>&1"])

    for _ in range(300):
        txt = log.read_text(errors="ignore")
        if "The server is fired up" in txt:
            break
        if "CUDA out of memory" in txt or "Received sigquit" in txt:
            return None, None, None
        time.sleep(5)
    else:
        return None, None, None

    rng = random.Random(4242)
    ids = [rng.randint(10, 100000) for _ in range(prompt_len)]

    def send():
        t0 = time.time()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=json.dumps({"input_ids": ids,
                             "sampling_params": {"max_new_tokens": 4, "temperature": 0}}).encode(),
            headers={"Content-Type": "application/json"})
        json.load(urllib.request.urlopen(req, timeout=1800))
        return (time.time() - t0) * 1000

    ttft_cold = send()          # populates the cache
    mark = len(log.read_text(errors="ignore"))
    ttft_warm = send()          # should be served from cache
    tail = log.read_text(errors="ignore")[mark:]

    cached = max((int(x) for x in re.findall(r"#cached-token: (\d+)", tail)), default=0)
    kill_own(port)
    wait_gpu_free(gpu)
    return cached, ttft_cold, ttft_warm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=31950)
    ap.add_argument("--prompt-len", type=int, default=30000)
    ap.add_argument("--fix", action="store_true",
                    help="run with SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS=1")
    a = ap.parse_args()
    L = a.prompt_len
    extra = {"SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS": "1"} if a.fix else {}

    cells = [
        ("int2", 8192,  f"{8192 - RECENT_TOKENS} if capped, ~{L} if correct"),
        ("int2", 32768, f"~{L} (prompt fits one chunk)"),
        ("bf16", 8192,  f"~{L} (control: not mixed-KV)"),
    ]
    print(f"prompt_len={L}  recent_tokens={RECENT_TOKENS}  "
          f"fix={'ON' if a.fix else 'OFF'}\n")
    print(f"{'mode':6}{'chunk':>8}{'cached':>10}{'cold ms':>10}{'warm ms':>10}   expectation")
    print("-" * 78)

    results = {}
    for mode, chunk, expect in cells:
        cached, cold, warm = run_cell(a.gpu, a.port, mode, chunk, L, extra)
        results[(mode, chunk)] = cached
        if cached is None:
            print(f"{mode:6}{chunk:>8}{'BOOT FAIL':>10}")
            continue
        print(f"{mode:6}{chunk:>8}{cached:>10}{cold:>10.0f}{warm:>10.0f}   {expect}")

    print()
    capped = results.get(("int2", 8192))
    big = results.get(("int2", 32768))
    ctrl = results.get(("bf16", 8192))
    full = FULL_CACHE_FRACTION * L
    if capped is None or big is None or ctrl is None:
        print("VERDICT: INCONCLUSIVE (a cell failed to boot)")
        return 2
    if capped < full and big >= full and ctrl >= full:
        print(f"VERDICT: BUG REPRODUCED — int2 caches only {capped} of {L} at the "
              f"default chunk, while the same prompt caches {big} with a larger "
              f"chunk and {ctrl} under bf16. Cap tracks chunk size, not prompt "
              f"length, and is mixed-KV-specific.")
        return 0
    if capped >= full and big >= full and ctrl >= full:
        print(f"VERDICT: NO CAP — all three cells cache >= {full:.0f} of {L}. "
              f"This is the expected result AFTER the fix.")
        return 0
    print(f"VERDICT: UNEXPECTED PATTERN — int2/8192={capped}, int2/32768={big}, "
          f"bf16/8192={ctrl}. Hypothesis does not hold as stated.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
