#!/usr/bin/env python3
"""Score still-running eval cells from their live io_log.jsonl, per completed seed,
so we can watch the running mean converge before metrics.json is written.

Usage: score_running.py <dataset> <rows.jsonl> <out_dir>[:target] [<out_dir>[:target] ...]
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import statistics as st
import pandas as pd
sys.path.insert(0, "/vault/amir/efficient-llm/teamily-project")
from kvq.benchmarks.evaluate_registry import SCORER_REGISTRY

dataset, rows_path = sys.argv[1], sys.argv[2]
scorer = SCORER_REGISTRY[dataset]
rows = {json.loads(l)["rid"]: json.loads(l) for l in open(rows_path)}
N = len(rows)

for spec in sys.argv[3:]:
    out_dir, _, target = spec.partition(":")
    try:
        io = [json.loads(l) for l in open(f"{out_dir}/io_log.jsonl") if l.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        # tolerate a torn tail line from the live writer
        io = []
        for l in open(f"{out_dir}/io_log.jsonl"):
            try: io.append(json.loads(l))
            except json.JSONDecodeError: pass
    by_seed = {}
    for e in io:
        by_seed.setdefault(e.get("sample_k", 0), {})[e["rid"]] = e["response"]
    name = "/".join(out_dir.rstrip("/").split("/")[-2:])
    full = []
    parts = []
    for k in sorted(by_seed):
        recs = [dict(rows[r], sample_k=k, predicted_answer=t) for r, t in by_seed[k].items()]
        for x in recs: x.pop("prompt", None)
        acc = scorer(pd.DataFrame(recs))["accuracy"] * 100
        if len(by_seed[k]) >= N:
            full.append(acc)
        else:
            parts.append(f"seed{k} {len(by_seed[k])}/{N}={acc:.1f}")
    mu = f"{st.mean(full):.2f}+/-{st.stdev(full):.2f}" if len(full) > 1 else (f"{full[0]:.2f}" if full else "--")
    print(f"{name:<34} mean={mu} over {len(full)} seed(s)  {' '.join(parts)}  | OSCAR={target or '?'}")
