#!/usr/bin/env python3
"""pgq5 Stage-B row-paired bootstraps (plan5 pre-registered contrasts).

Wraps bootstrap_pairs' scoring for the Qwen cell layouts: pgq cells live
under artifacts/bench_pgq/qwen3_8b (attempt subdirs possible), reference
cells (full_precision / turboquant_k2_v2 / _v3) under artifacts/bench/
qwen3_8b with the May-sweep naming. Emits pgq5_bootstraps.json.

    python pipelines/page_quant/pgq5_bootstrap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipelines.page_quant.bootstrap_pairs import row_scores  # noqa: E402
from kvq.io import save_json  # noqa: E402

PGQ_ROOT = REPO / "artifacts/bench_pgq/qwen3_8b"
REF_ROOT = REPO / "artifacts/bench/qwen3_8b"
TASKS = ["lcc", "musique", "2wikimqa", "qasper", "hotpotqa"]
NBOOT = 10000
SEED = 20260710
CONTRASTS = [
    ("pgq_proflmrw_rdo@2", "full_precision"),
    ("pgq_proflmrw_rdo@2", "turboquant_k2_v2"),
    ("pgq_proflm_rdo@2", "turboquant_k2_v2"),
    ("pgq_proflmrw_rdo@2", "pgq_proflm_rdo@2"),
    ("pgq_proflmrw_rdo@2", "turboquant_k2_v3"),
]


def find_csv(method: str, task: str) -> pd.DataFrame:
    if "@" in method:                          # pgq cell
        kind, rate = method.split("@")
        hits = sorted(PGQ_ROOT.glob(f"pgq__{kind}__b{rate}__*__{task}"))
    else:                                      # May-sweep reference cell
        hits = sorted(REF_ROOT.glob(f"{method}_{task}"))
    assert len(hits) == 1, f"{method}/{task}: {len(hits)} cell dirs"
    csvs = sorted(hits[0].rglob("predictions.csv"),
                  key=lambda p: len(p.parts))
    assert csvs, f"{method}/{task}: no predictions.csv"
    return pd.read_csv(csvs[-1])               # deepest = latest attempt


def main() -> None:
    rng = np.random.default_rng(SEED)
    scores: dict[tuple, pd.Series] = {}

    def get(method, task):
        if (method, task) not in scores:
            scores[(method, task)] = row_scores(find_csv(method, task), task)
        return scores[(method, task)]

    out = {}
    for a, b in CONTRASTS:
        per_task = {}
        boots = []
        for task in TASKS:
            sa, sb = get(a, task), get(b, task)
            ids = sa.index.intersection(sb.index)
            # pgq cells exclude compact8 train rows; refs did too — sets
            # must match exactly or the pairing is broken
            assert len(ids) == len(sa) == len(sb), \
                f"{a} vs {b} / {task}: rows {len(sa)}/{len(sb)}/join {len(ids)}"
            d = (sa[ids] - sb[ids]).to_numpy()
            idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
            bm = d[idx].mean(1)
            per_task[task] = dict(
                delta=float(d.mean()),
                ci=[float(np.percentile(bm, 2.5)),
                    float(np.percentile(bm, 97.5))],
                n=int(len(d)))
            boots.append(bm)
        mean_boot = np.mean(boots, axis=0)
        lo, hi = np.percentile(mean_boot, [2.5, 97.5])
        pooled = dict(
            delta=float(np.mean([per_task[t]["delta"] for t in TASKS])),
            ci=[float(lo), float(hi)])
        nonlcc = np.mean([b for t, b in zip(TASKS, boots) if t != "lcc"],
                         axis=0)
        nl_lo, nl_hi = np.percentile(nonlcc, [2.5, 97.5])
        out[f"{a}__minus__{b}"] = dict(
            per_task=per_task, pooled5=pooled,
            nonlcc4=dict(
                delta=float(np.mean([per_task[t]["delta"]
                                     for t in TASKS if t != "lcc"])),
                ci=[float(nl_lo), float(nl_hi)]))
        sig = "SIG" if lo > 0 or hi < 0 else "tie"
        print(f"{a} - {b}: pooled5 {pooled['delta']:+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}] {sig}")

    save_json(REPO / "artifacts/page_quant2/pgq5_bootstraps.json", out)
    print("-> artifacts/page_quant2/pgq5_bootstraps.json")


if __name__ == "__main__":
    main()
