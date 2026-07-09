#!/usr/bin/env python3
"""Row-paired bootstrap for page_quant F1 contrasts (theory-critic rec).

The decisive contrasts sit near the 0.5-pp noise floor; paired differencing
on shared eval rows resolves far below it. Joins two methods' predictions.csv
per task on `_id`, computes per-row LongBench scores with the official
scorers, and bootstraps the mean difference (per task and the 3-task mean).

    python pipelines/page_quant/bootstrap_pairs.py \
        --a pgq_ea@1.0 --b ecu@1.0
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(REPO / "vendor/kvpress/evaluation"))

import argparse  # noqa: E402
import ast  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmarks.longbench.calculate_metrics import dataset2metric  # noqa: E402

BENCH = REPO / "artifacts/bench_pgq/llama31_8b"
CELL_GLOB = "pgq__{kind}__b{rate}__*__{task}"
TASKS = ["lcc", "musique", "2wikimqa"]
NBOOT = 10000
SEED = 20260707


def cell_predictions(method: str, task: str, bench: Path = BENCH,
                     cell_glob: str = CELL_GLOB) -> pd.DataFrame:
    kind, rate = method.split("@")
    hits = sorted(bench.glob(cell_glob.format(kind=kind, rate=rate,
                                              task=task)))
    hits = [h for h in hits if "__f0" not in h.name]
    assert len(hits) == 1, f"{method}/{task}: {len(hits)} cell dirs"
    csvs = [p for p in hits[0].glob("*/predictions.csv")
            if "fraction" not in p.parent.name]
    assert len(csvs) == 1, f"{method}/{task}: {len(csvs)} full-fraction csvs"
    return pd.read_csv(csvs[0])


def row_scores(df: pd.DataFrame, task: str) -> pd.Series:
    metric = dataset2metric[task]
    out = {}
    for _, row in df.iterrows():
        pred = str(row["predicted_answer"]) if pd.notna(
            row["predicted_answer"]) else ""
        answers = ast.literal_eval(row["answers"]) \
            if isinstance(row["answers"], str) else row["answers"]
        ac = row.get("all_classes")
        all_classes = ast.literal_eval(ac) if isinstance(ac, str) else None
        if task in ("trec", "triviaqa", "samsum", "lsht"):
            pred = pred.lstrip().split("\n")[0]
        s = 0.0
        for gt in answers:
            s = max(s, metric(pred.lstrip(), gt, all_classes=all_classes))
        out[row["_id"]] = 100.0 * s
    return pd.Series(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="e.g. pgq_ea@1.0")
    ap.add_argument("--b", required=True, help="e.g. ecu@1.0")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--bench-root", default=str(BENCH),
                    help="cell tree (eviction: artifacts/bench_evict/"
                         "llama31_8b)")
    ap.add_argument("--cell-glob", default=CELL_GLOB,
                    help="cell-dir glob with {kind}/{rate}/{task} slots "
                         "(eviction: 'evict__{kind}__r{rate}__{task}'; "
                         "the method arg stays '<kind>@<rate>')")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    bench = Path(args.bench_root)

    per_task_delta = {}
    boots = {}
    for task in args.tasks:
        sa = row_scores(cell_predictions(args.a, task, bench,
                                         args.cell_glob), task)
        sb = row_scores(cell_predictions(args.b, task, bench,
                                         args.cell_glob), task)
        ids = sa.index.intersection(sb.index)
        assert len(ids) == len(sa) == len(sb), \
            f"{task}: row sets differ ({len(sa)} vs {len(sb)}, join {len(ids)})"
        d = (sa[ids] - sb[ids]).to_numpy()
        idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
        bm = d[idx].mean(1)
        per_task_delta[task] = (float(d.mean()), float(np.percentile(bm, 2.5)),
                                float(np.percentile(bm, 97.5)))
        boots[task] = bm
        lo, hi = per_task_delta[task][1], per_task_delta[task][2]
        sig = "SIG" if lo > 0 or hi < 0 else "tie"
        print(f"{task:12s} delta={d.mean():+6.2f}  95% CI [{lo:+6.2f}, "
              f"{hi:+6.2f}]  n={len(d)}  {sig}")

    mean_boot = np.mean([boots[t] for t in args.tasks], axis=0)
    mdelta = float(np.mean([per_task_delta[t][0] for t in args.tasks]))
    lo, hi = np.percentile(mean_boot, [2.5, 97.5])
    sig = "SIG" if lo > 0 or hi < 0 else "tie"
    print(f"{'3-task mean':12s} delta={mdelta:+6.2f}  95% CI [{lo:+6.2f}, "
          f"{hi:+6.2f}]  {sig}   ({args.a} minus {args.b})")


if __name__ == "__main__":
    main()
