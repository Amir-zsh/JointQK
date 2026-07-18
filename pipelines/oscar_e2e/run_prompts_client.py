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


def generate(base, rec, timeout, sampling, retries=3):
    params = {
        "max_new_tokens": int(rec["max_new_tokens"]),
        **sampling,
    }
    payload = {"text": rec["prompt"], "sampling_params": params}
    for attempt in range(retries):
        try:
            r = requests.post(f"{base}/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            out = r.json()
            meta = out.get("meta_info", {})
            return out["text"], {
                "completion_tokens": meta.get("completion_tokens"),
                "finish_reason": (meta.get("finish_reason") or {}).get("type"),
            }
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
    # Sampling: defaults follow the OSCAR authors' long-horizon eval driver
    # (temperature 1.0, top_p 0.95, top_k 40) when --samples > 1; the
    # single-sample default stays greedy for the existing NIAH/LB cells.
    ap.add_argument("--samples", type=int, default=1,
                    help="K independent samples per row (acc@K)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: 0.0 when --samples 1, else 1.0")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    temperature = args.temperature
    if temperature is None:
        temperature = 0.0 if args.samples == 1 else 1.0
    sampling = {"temperature": temperature}
    if temperature > 0:
        sampling.update({"top_p": args.top_p, "top_k": args.top_k})

    from kvq.benchmarks.evaluate_registry import SCORER_REGISTRY

    recs = [json.loads(l) for l in Path(args.rows).open()]
    if args.samples > 1:
        recs = [dict(r, sample_k=k) for k in range(args.samples) for r in recs]
    else:
        recs = [dict(r, sample_k=0) for r in recs]

    # Resume: a prior interrupted run leaves io_log.jsonl without
    # metrics.json. Reload its completed (rid, sample_k) pairs (last entry
    # wins across client retries), skip them, and append to the same log.
    prior = {}
    out_dir = REPO / args.out
    log_path = out_dir / "io_log.jsonl"
    if log_path.exists() and not (out_dir / "metrics.json").exists():
        for line in log_path.open():
            try:
                e = json.loads(line)
                prior[(e["rid"], e.get("sample_k", 0))] = e
            except (json.JSONDecodeError, KeyError):
                continue  # torn tail line from the interrupted run
        if prior:
            print(f"[client] resume: {len(prior)} completed pairs found, "
                  f"{len(recs) - len(prior)} to run", flush=True)
    base = f"http://{args.host}:{args.port}"
    info = requests.get(f"{base}/get_server_info", timeout=30).json()
    t0 = time.time()
    print(f"[client] {len(recs)} rows -> {base} "
          f"(kv_cache_dtype={info.get('kv_cache_dtype', '?')})", flush=True)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "io_log.jsonl").open("a" if prior else "w")

    done = [0]
    metas = {}
    def run_one(rec):
        key = (rec["rid"], rec["sample_k"])
        if key in prior:
            e = prior[key]
            metas[key] = {"completion_tokens": e.get("completion_tokens"),
                          "finish_reason": e.get("finish_reason")}
            return e["response"]
        text, meta = generate(base, rec, args.timeout, sampling)
        metas[key] = meta
        log.write(json.dumps({"rid": rec["rid"], "sample_k": rec["sample_k"],
                              "response": text, **meta}) + "\n")
        log.flush()
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
    df["completion_tokens"] = [metas[(r["rid"], r["sample_k"])]["completion_tokens"] for r in recs]
    df["finish_reason"] = [metas[(r["rid"], r["sample_k"])]["finish_reason"] for r in recs]
    df.to_csv(out / "predictions.csv", index=False)

    dataset = recs[0]["dataset"]
    scorer = SCORER_REGISTRY[dataset]
    if args.samples > 1:
        per_k = [scorer(df[df.sample_k == k].reset_index(drop=True))
                 for k in range(args.samples)]
        metrics = {
            "samples": args.samples,
            "sampling": {**sampling, "top_p": args.top_p, "top_k": args.top_k},
            "per_k": per_k,
            "accuracy_avg_at_k": sum(m["accuracy"] for m in per_k) / len(per_k),
            "cap_hit_rate": float((df.finish_reason == "length").mean()),
            "mean_completion_tokens": float(df.completion_tokens.mean()),
            "total": int((df.sample_k == 0).sum()),
        }
    else:
        metrics = scorer(df)
        metrics["cap_hit_rate"] = float((df.finish_reason == "length").mean())
        metrics["mean_completion_tokens"] = float(df.completion_tokens.mean())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "server_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"[client] metrics: {json.dumps(metrics)[:300]}", flush=True)
    print(f"[client] wrote {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
