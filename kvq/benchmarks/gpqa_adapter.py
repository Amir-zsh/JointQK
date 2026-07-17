"""GPQA-diamond adapter + scorer (simple-evals protocol, per the OSCAR
authors' eval driver in vendor/OSCAR/rotation/_eval_runner/).

Loader: the simple-evals CSV (cached locally under artifacts/prompt_rows/),
choice permutation with random.Random(seed) drawn sequentially per row —
byte-matching dump_gpqa_prompts.py — formatted with their multichoice
template. Emits the same column schema the exporter/client expect
(context/question/answer/answer_prefix/max_new_tokens).

Scorer: last `Answer: X` match (last, not first, so thinking traces that
restate the format mid-reasoning don't clip the final answer), per-row 0/1
via `score_rows` for acc@K aggregation.
"""
from __future__ import annotations

import pathlib
import random
import re

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
_CSV_CACHE = _REPO_ROOT / "artifacts" / "prompt_rows" / "gpqa_{variant}.csv"

# simple-evals QUERY_TEMPLATE_MULTICHOICE / ANSWER_PATTERN_MULTICHOICE.
QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([A-D])"


def load_gpqa_df(variant: str = "diamond", seed: int = 0,
                 max_new_tokens: int = 8192) -> pd.DataFrame:
    cache = pathlib.Path(str(_CSV_CACHE).format(variant=variant))
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(_CSV_URL.format(variant=variant)).to_csv(cache, index=False)
    raw = pd.read_csv(cache)

    rng = random.Random(seed)
    rows = []
    letters = "ABCD"
    for _idx, r in raw.iterrows():
        perm = rng.sample(range(4), 4)
        choices = [
            r["Correct Answer"], r["Incorrect Answer 1"],
            r["Incorrect Answer 2"], r["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in perm]
        question = QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3],
            Question=r["Question"],
        )
        rows.append({
            "context": "",
            "question": question,
            "answer": letters[perm.index(0)],
            "answer_prefix": "",
            "max_new_tokens": max_new_tokens,
        })
    return pd.DataFrame(rows)


def extract_answer(text: str):
    matches = re.findall(ANSWER_PATTERN, str(text))
    return matches[-1].upper() if matches else None


def score_rows(df: pd.DataFrame) -> list[bool]:
    return [extract_answer(row["predicted_answer"]) == str(row["answer"])
            for _idx, row in df.iterrows()]


def calculate_metrics(df: pd.DataFrame) -> dict:
    rows = score_rows(df)
    extracted = sum(extract_answer(row["predicted_answer"]) is not None
                    for _idx, row in df.iterrows())
    return {
        "correct": int(sum(rows)),
        "accuracy": sum(rows) / len(df),
        "extraction_rate": extracted / len(df),
        "scorer": "gpqa_simple_evals",
        "total": len(df),
    }
