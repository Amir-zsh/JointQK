#!/usr/bin/env python3
"""Merge sharded eval cells (<cell>__s0, __s1, ...) into the parent cell dir
and score the union — the read-side companion of the GPU-pool sharding.

    .venv/bin/python pipelines/oscar_e2e/merge_shards.py \
        --root artifacts/oscar_llama31_8b/grid [--force]

For every cell that has shard dirs and no (or --force) merged metrics.json:
concatenates the shards' predictions.csv (they carry disjoint rid sets by
construction), verifies no rid/sample_k overlap and full shard coverage,
re-scores with the same registry the client uses, and writes predictions.csv
+ metrics.json into the parent cell dir. Shard dirs are left in place.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "kvpress"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from kvq.benchmarks.evaluate_registry import SCORER_REGISTRY
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "code_scorers", REPO / "pipelines" / "eval" / "code_scorers.py")
    _cs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cs)
    scorer_for = lambda ds: _cs.CODE_SCORERS.get(ds) or SCORER_REGISTRY[ds]

    root = REPO / args.root
    shards = defaultdict(list)
    for d in sorted(root.glob("*/*__s[0-9]*")):
        m = re.match(r"(.+)__s(\d+)$", d.name)
        if m and (d / "metrics.json").exists():
            shards[d.parent / m.group(1)].append(d)

    for cell, dirs in sorted(shards.items()):
        out_metrics = cell / "metrics.json"
        if out_metrics.exists() and not args.force:
            continue
        expected = len(list(cell.parent.glob(f"{cell.name}__s[0-9]*")))
        if len(dirs) < expected:
            print(f"[merge] {cell.relative_to(REPO)}: {len(dirs)}/{expected} "
                  f"shards done — skipping")
            continue
        df = pd.concat(
            [pd.read_csv(d / "predictions.csv") for d in sorted(dirs)],
            ignore_index=True,
        )
        # CSV round-trip stringifies list-typed columns (NIAH answers are
        # lists); the ruler scorer then iterates the STRING character-wise
        # ("['1679215']" -> 7/11 chars match -> 63.64%-style garbage).
        # Parse list-like strings back to real lists before scoring.
        import ast
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str)
                if len(sample) and sample.str.startswith("[").all():
                    df[col] = df[col].map(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )
        dup = df.duplicated(subset=["rid", "sample_k"]).sum()
        assert dup == 0, f"{cell}: {dup} duplicate (rid, sample_k) rows"
        cell.mkdir(parents=True, exist_ok=True)
        df.to_csv(cell / "predictions.csv", index=False)

        dataset = df["dataset"].iloc[0]
        scorer = scorer_for(dataset)
        ks = sorted(df["sample_k"].unique())
        if len(ks) > 1:
            per_k = [scorer(df[df.sample_k == k].reset_index(drop=True))
                     for k in ks]
            accs = [m["accuracy"] for m in per_k]
            metrics = {
                "samples": len(ks), "per_k": per_k,
                "accuracy_avg_at_k": sum(accs) / len(accs),
                "accuracy_mean": sum(accs) / len(accs),
                "accuracy_std": (pd.Series(accs).std(ddof=1) if len(accs) > 1 else 0.0),
                "cap_hit_rate": float((df.finish_reason == "length").mean()),
                "mean_completion_tokens": float(df.completion_tokens.mean()),
                "total": int((df.sample_k == ks[0]).sum()),
                "merged_from_shards": len(dirs),
            }
        else:
            metrics = scorer(df)
            metrics["cap_hit_rate"] = float((df.finish_reason == "length").mean())
            metrics["mean_completion_tokens"] = float(df.completion_tokens.mean())
            metrics["merged_from_shards"] = len(dirs)
        out_metrics.write_text(json.dumps(metrics, indent=2, default=float))
        print(f"[merge] {cell.relative_to(REPO)}: {len(dirs)} shards -> "
              f"{json.dumps(metrics)[:160]}")


if __name__ == "__main__":
    main()
