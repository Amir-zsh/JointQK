#!/usr/bin/env python3
"""Correctness check for SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS on real long-context content.

The fix makes MORE of a long prompt be served from the radix (prefix) cache. The
hazard it could introduce is therefore specific: cached KV that is stale or
misattributed would produce wrong answers on the cache-hit path. A short-prompt
output diff cannot detect this at all -- a short prompt never exceeds one prefill
chunk, so the changed code never runs.

So this sends real NIAH-64K rows TWICE against a radix-cache-enabled server:
  cold  -- populates the cache (prefix reuse irrelevant)
  warm  -- served largely from cache; this is the path the fix widens
and checks three things:
  1. warm answer == cold answer                      (cache path is self-consistent)
  2. accuracy(warm) == accuracy(cold)                (cache path is still correct)
  3. cached-token count on warm requests             (the fix actually engaged)
Run with and without --fix to compare.

Usage:
  python pipelines/oscar_e2e/verify_chunk_cutoff_fix.py [--rows 24] [--fix] [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTAINER = "oscar-ab"
ROWS = REPO / "artifacts/prompt_rows/niah_65536_qwen.jsonl"
CB = "third_party/samuel_vq/codebooks/vqa_G4_strat_flat_ptn_gpqacc64k_fp8.pt"


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def kill_own(port):
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


def boot(gpu, port, mode, fix, log):
    kill_own(port)
    wait_gpu_free(gpu)
    if log.exists():
        log.unlink()
    log.touch()
    env = ["-e", "RADIX_CACHE=1", "-e", "MAX_TOKENS=600000",
           "-e", "SERVE_EXTRA=--enable-cache-report"]   # DEFAULT chunk size: the fix
    if fix:                                            # must work without the 102400
        env += ["-e", "SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS=1"]   # deviation.
    flags = "" if mode == "int2" else f"--vq2 --vq-codebook {CB}"
    sh(["docker", "exec", "-d", *env, CONTAINER, "bash", "-lc",
        f"cd {REPO} && bash pipelines/oscar_e2e/serve_oscar.sh --gpu {gpu} "
        f"--port {port} --ctx 131072 --mem-frac 0.82 {flags} >> {log} 2>&1"])
    for _ in range(360):
        t = log.read_text(errors="ignore")
        if "The server is fired up" in t:
            return True
        if "CUDA out of memory" in t or "Received sigquit" in t:
            return False
        time.sleep(5)
    return False


def generate(port, row):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps({
            "text": row["prompt"] + row.get("answer_prefix", ""),
            "sampling_params": {"max_new_tokens": int(row.get("max_new_tokens", 128)),
                                "temperature": 0},
        }).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))["text"]


def hit(text, answers):
    return any(str(a).lower() in (text or "").lower() for a in answers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=31970)
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--mode", default="int2", choices=["int2", "vq2"])
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in ROWS.read_text().splitlines()[: a.rows]]
    log = REPO / "logs" / f"verify_cutoff_{a.mode}_{'on' if a.fix else 'off'}.log"
    if not boot(a.gpu, a.port, a.mode, a.fix, log):
        print("BOOT FAILED"); return 2

    same = cold_ok = warm_ok = 0
    cached_tokens = []
    for i, row in enumerate(rows):
        cold = generate(a.port, row)
        mark = len(log.read_text(errors="ignore"))
        warm = generate(a.port, row)
        tail = log.read_text(errors="ignore")[mark:]
        c = max((int(x) for x in re.findall(r"#cached-token: (\d+)", tail)), default=0)
        cached_tokens.append(c)
        same += (cold == warm)
        cold_ok += hit(cold, row["answer"])
        warm_ok += hit(warm, row["answer"])
        print(f"  row {i:3d} {row['task']:18s} cached={c:6d} "
              f"cold={'OK' if hit(cold, row['answer']) else '..'} "
              f"warm={'OK' if hit(warm, row['answer']) else '..'} "
              f"identical={'yes' if cold == warm else 'NO'}", flush=True)

    kill_own(a.port); wait_gpu_free(a.gpu)
    n = len(rows)
    med = sorted(cached_tokens)[n // 2] if n else 0
    print(f"\nmode={a.mode} fix={'ON' if a.fix else 'OFF'} rows={n}")
    print(f"  median cached tokens on warm request : {med}")
    print(f"  warm output identical to cold        : {same}/{n}")
    print(f"  accuracy cold / warm                 : {cold_ok}/{n}  {warm_ok}/{n}")
    ok = (same == n) and (warm_ok == cold_ok)
    print("  VERDICT:", "PASS — cache path is self-consistent and accuracy is unchanged"
          if ok else "FAIL — cache path changed outputs or accuracy")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
