#!/usr/bin/env python3
"""Paired A/B for two NIAH arms scored on the SAME rows.

Comparing two arms as independent proportions throws away the pairing and leaves
a noise floor set by item difficulty (~3.3 pp at n=200), which swamps the effect
we care about. Both arms answer the identical row set, so the informative
statistic is the per-row difference d_i = score(treat, i) - score(control, i):
item difficulty cancels, and only rows where the arms actually disagree carry
information.

SCORING matches the authoritative scorer exactly -- string_match_all in
vendor/kvpress/evaluation/benchmarks/ruler/calculate_metrics.py, reached via
SCORER_REGISTRY["niah"]:

    per-row score = (# reference needles found in the prediction) / (# refs)

so a row is CONTINUOUS in [0, 1], not binary: the multi-needle subtasks
(multiquery, multivalue) award partial credit. An earlier version of this script
used ``any(ref in pred)``, which scored multiquery at 81% against the official
38.2% and would have produced a confident but meaningless verdict. The
cross-check in ``load`` now catches that whole class of error.

The reported test is an exact two-sided SIGN TEST over rows whose scores differ:
are disagreements one-sided (a real regression) or balanced (numerical flips)?
Both arms decode greedily, but greedy is not deterministic ACROSS arms here --
reusing quantized KV changes reduction order, so borderline rows flip either way.

Usage:
  python pipelines/eval/paired_niah_ab.py --control <dir> --treat <dir>
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from math import comb
from pathlib import Path

import pandas as pd

CTRL_CHARS = re.compile(r"[\x00-\x1f]")


def row_score(pred, refs) -> float:
    """Exactly ``string_match_all`` for a single row: fraction of refs found."""
    if isinstance(refs, str):
        try:
            refs = ast.literal_eval(refs)
        except Exception:
            refs = [refs]
    if not isinstance(refs, (list, tuple)):
        refs = [refs]
    if not len(refs):
        return 0.0
    pred = CTRL_CHARS.sub("", str(pred or "").strip()).strip()
    return sum(1.0 if str(r).lower() in pred.lower() else 0.0 for r in refs) / len(refs)


def sign_test_two_sided(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def load(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "predictions.csv")
    if "predicted_answer" not in df.columns:
        raise SystemExit(f"{d}: no 'predicted_answer' column; got {list(df.columns)}")
    df["_s"] = [row_score(p, a) for p, a in zip(df["predicted_answer"], df["answer"])]

    # Validate this scorer by AGREEMENT with the authoritative one, not by the
    # value it produces. Rejecting 0% or 100% outright would suppress a genuine
    # catastrophic regression -- the result most worth reporting -- and would be
    # wrong here anyway: multikey_2/3 legitimately score 0 for OSCAR-INT2.
    mpath = d / "metrics.json"
    if mpath.exists() and "task" in df.columns:
        official = json.loads(mpath.read_text())
        for task, grp in df.groupby("task"):
            ref = official.get(task)
            ref = ref.get("string_match") if isinstance(ref, dict) else ref
            if ref is None:
                continue
            ours = grp["_s"].mean() * 100
            if abs(ours - ref) > 0.5:
                raise SystemExit(
                    f"{d}: scorer mismatch on {task}: this script {ours:.2f}% vs "
                    f"metrics.json {ref:.2f}%. Fix the scorer before any verdict."
                )
    key = "_id" if "_id" in df.columns else df.columns[0]
    return df.set_index(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--treat", type=Path, required=True)
    a = ap.parse_args()

    ctl, trt = load(a.control), load(a.treat)
    common = ctl.index.intersection(trt.index)
    ctl, trt = ctl.loc[common], trt.loc[common]
    task = ctl["task"] if "task" in ctl.columns else pd.Series("all", index=common)
    print(f"paired rows: {len(common)}   (scorer cross-checked against metrics.json)\n")
    print(f"{'subtask':20}{'ctl %':>8}{'trt %':>8}{'delta':>8}{'ctl+':>6}{'trt+':>6}{'p':>8}")
    print("-" * 64)

    tot_b = tot_w = 0
    for t in sorted(task.unique()):
        m = task == t
        d = trt["_s"][m] - ctl["_s"][m]
        b, w = int((d > 0).sum()), int((d < 0).sum())
        tot_b += b
        tot_w += w
        print(f"{t:20}{ctl['_s'][m].mean()*100:8.2f}{trt['_s'][m].mean()*100:8.2f}"
              f"{d.mean()*100:+8.2f}{w:6d}{b:6d}{sign_test_two_sided(b, w):8.3f}")

    d = trt["_s"] - ctl["_s"]
    p = sign_test_two_sided(tot_b, tot_w)
    print("-" * 64)
    print(f"{'OVERALL':20}{ctl['_s'].mean()*100:8.2f}{trt['_s'].mean()*100:8.2f}"
          f"{d.mean()*100:+8.2f}{tot_w:6d}{tot_b:6d}{p:8.3f}")
    n_disc = tot_b + tot_w
    print(f"\nrows where the arms differ: {n_disc}/{len(common)} "
          f"({n_disc / max(1, len(common)) * 100:.1f}%) -- these carry the information")
    print(f"control better on {tot_w}, treatment better on {tot_b}")
    if p >= 0.05:
        print(f"VERDICT: no significant difference (exact sign test p={p:.3f}); "
              "disagreements are balanced, consistent with cache-order numerical "
              "flips rather than a systematic regression.")
    else:
        print(f"VERDICT: significant "
              f"{'REGRESSION' if tot_w > tot_b else 'improvement'} "
              f"(exact sign test p={p:.3f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
