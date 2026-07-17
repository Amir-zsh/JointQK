#!/usr/bin/env python3
"""Aggregate acc@K long-horizon cells (plan11 C1): per-arm avg@K / pass@K
with row-bootstrap CIs, and paired deltas vs an anchor arm (same rows, same
sample index space).

    .venv/bin/python pipelines/eval/aggregate_acck.py \
        --cells bf16=artifacts/oscar_e2e/lh/bf16_gpqa int2=... vq2=... \
        --anchor bf16 --out artifacts/oscar_e2e/lh/gpqa_acck_summary.json

Each cell dir is a run_prompts_client.py output (predictions.csv with
sample_k + finish_reason columns). Row-level 0/1 comes from the task's
score_rows (math-verify for math tasks, simple-evals letter match for gpqa).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "kvpress"))


def row_scorer(dataset):
    if dataset == "gpqa":
        from kvq.benchmarks.gpqa_adapter import score_rows
        return score_rows
    if dataset in ("math500", "aime25"):
        from kvq.benchmarks.math_verify_scorer import score_rows
        return score_rows
    raise SystemExit(f"no row-level scorer for dataset {dataset!r}")


def load_cell(path):
    df = pd.read_csv(path / "predictions.csv")
    dataset = df["dataset"].iloc[0]
    score = row_scorer(dataset)
    correct = np.array(score(df), dtype=float)
    piv = (
        pd.DataFrame({
            "rid": df["rid"], "k": df["sample_k"], "c": correct,
        })
        .pivot(index="rid", columns="k", values="c")
        .sort_index()
    )
    cap = (
        pd.DataFrame({"rid": df["rid"], "k": df["sample_k"],
                      "cap": (df["finish_reason"] == "length").astype(float)})
        .pivot(index="rid", columns="k", values="cap")
        .sort_index()
    )
    return dataset, piv, cap, df


def boot_ci(values, n_boot, rng, stat):
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = stat(values[idx])
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="name=path pairs (path = client out dir)")
    ap.add_argument("--anchor", default="bf16")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cells = {}
    dataset = None
    for spec in args.cells:
        name, path = spec.split("=", 1)
        ds, piv, cap, _df = load_cell(REPO / path)
        dataset = dataset or ds
        assert ds == dataset, f"mixed datasets: {ds} vs {dataset}"
        cells[name] = (piv, cap)

    rids = None
    for name, (piv, _cap) in cells.items():
        rids = piv.index if rids is None else rids.intersection(piv.index)
    summary = {"dataset": dataset, "n_rows": int(len(rids)),
               "anchor": args.anchor, "arms": {}, "contrasts": {}}

    mats = {}
    for name, (piv, cap) in cells.items():
        m = piv.loc[rids].to_numpy()  # [rows, K]
        mats[name] = m
        avg_k = m.mean(axis=1)  # per-row avg over K
        summary["arms"][name] = {
            "avg_at_k": float(avg_k.mean()),
            "avg_at_k_ci95": boot_ci(avg_k, args.n_boot, rng,
                                     lambda x: x.mean(axis=1)),
            "pass_at_k": float(m.max(axis=1).mean()),
            "per_k": [float(v) for v in m.mean(axis=0)],
            "cap_hit_rate": float(cap.loc[rids].to_numpy().mean()),
            "K": int(m.shape[1]),
        }

    anchor = mats.get(args.anchor)
    for name, m in mats.items():
        if name == args.anchor or anchor is None:
            continue
        d = m.mean(axis=1) - anchor.mean(axis=1)  # paired per-row delta
        summary["contrasts"][f"{name}-{args.anchor}"] = {
            "delta_avg_at_k": float(d.mean()),
            "delta_ci95": boot_ci(d, args.n_boot, rng, lambda x: x.mean(axis=1)),
        }
    # vq2 vs int2 difference-of-deltas reduces to the direct paired contrast
    # (the shared anchor cancels row-wise).
    if "vq2" in mats and "int2" in mats:
        d = mats["vq2"].mean(axis=1) - mats["int2"].mean(axis=1)
        summary["contrasts"]["vq2-int2"] = {
            "delta_avg_at_k": float(d.mean()),
            "delta_ci95": boot_ci(d, args.n_boot, rng, lambda x: x.mean(axis=1)),
        }

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
