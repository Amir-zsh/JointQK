#!/usr/bin/env python3
"""Drive exported prompt rows through a local SGLang server and score them
with our registry scorers.

Pattern follows the OSCAR authors' rotation/_eval_runner/run_simple_eval.py
sampler (OpenAI-compatible host), but hits the NATIVE /generate endpoint with
the final prompt TEXT from pipelines/eval/export_prompt_rows.py — the rows
were templated identically to our harness, so re-templating via the chat
endpoint would double-wrap them. Greedy (temperature 0), max_new_tokens per
row. Output dir mirrors a bench cell: predictions.csv + metrics.json, so
bootstrap_pairs.py and the aggregators consume it unchanged.

    .venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
        --rows artifacts/prompt_rows/ruler_32768_qwen.jsonl \
        --port 30800 --out artifacts/oscar_e2e/int2/ruler_32768
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "kvpress"))


def generate(base, rec, timeout, retries=3):
    payload = {
        "text": rec["prompt"],
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": int(rec["max_new_tokens"]),
        },
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{base}/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["text"]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"[client] retry rid={rec['rid']} after {exc}", flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from kvq.benchmarks.evaluate_registry import SCORER_REGISTRY

    recs = [json.loads(l) for l in Path(args.rows).open()]
    base = f"http://{args.host}:{args.port}"
    info = requests.get(f"{base}/get_server_info", timeout=30).json()
    t0 = time.time()
    print(f"[client] {len(recs)} rows -> {base} "
          f"(kv_cache_dtype={info.get('kv_cache_dtype', '?')})", flush=True)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "io_log.jsonl").open("w")

    done = [0]
    def run_one(rec):
        text = generate(base, rec, args.timeout)
        log.write(json.dumps({"rid": rec["rid"], "response": text}) + "\n")
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"[client] {done[0]}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
        return text

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        preds = list(ex.map(run_one, recs))
    log.close()

    df = pd.DataFrame(recs).drop(columns=["prompt"])
    df["predicted_answer"] = preds
    df["compression_ratio"] = None
    df.to_csv(out / "predictions.csv", index=False)

    dataset = recs[0]["dataset"]
    metrics = SCORER_REGISTRY[dataset](df)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "server_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"[client] metrics: {json.dumps(metrics)[:300]}", flush=True)
    print(f"[client] wrote {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
