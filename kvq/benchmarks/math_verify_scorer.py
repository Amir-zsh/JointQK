"""math-verify (HF) scorer for math benchmarks (aime25, math500).

Replaces the vendored regex scorers as the registered scorer: symbolic
equivalence via math_verify.parse/verify instead of exact string match on the
\\boxed{} payload (so 1/2 == 0.5 == \\frac{1}{2}). The vendored regex accuracy
is still computed and reported as `accuracy_boxed_exact` for cross-checking;
`accuracy` (the headline key every aggregator reads) is math-verify's.

Per-row results are exposed via `score_rows` so row-paired bootstraps can
consume identical rows across harnesses (our workers and the SGLang client).
"""
from __future__ import annotations

import pandas as pd
from math_verify import parse, verify


def _mv_correct(pred_answer, true_answer) -> bool:
    # Gold answers in these datasets are bare latex ("\\left( 3, \\frac..."),
    # which parse() can MISparse to a bare number while still returning a
    # truthy result (tuples come back as their first scalar), so the wrapped
    # "$...$" parse must always be tried too — not only as a fallback on
    # empty. Accept if either gold reading verifies.
    try:
        pred = parse(str(pred_answer))
        if not pred:
            return False
        for gold in (parse(f"${true_answer}$"), parse(str(true_answer))):
            if gold and verify(gold, pred):
                return True
        return False
    except Exception:
        return False


def _boxed_exact(pred_answer, true_answer) -> bool:
    try:
        boxed = str(str(pred_answer).split("boxed{")[-1].split("}")[0])
    except IndexError:
        return False
    return boxed == str(true_answer)


def score_rows(df: pd.DataFrame) -> list[bool]:
    return [_mv_correct(row["predicted_answer"], row["answer"])
            for _idx, row in df.iterrows()]


def calculate_metrics(df: pd.DataFrame) -> dict:
    rows = score_rows(df)
    correct = sum(rows)
    boxed_exact = sum(_boxed_exact(row["predicted_answer"], row["answer"])
                      for _idx, row in df.iterrows())
    answered = sum("boxed{" in str(row["predicted_answer"])
                   for _idx, row in df.iterrows())
    return {
        "correct": int(correct),
        "answered": int(answered),
        "accuracy": correct / len(df),
        "accuracy_boxed_exact": boxed_exact / len(df),
        "scorer": "math_verify",
        "total": len(df),
    }
