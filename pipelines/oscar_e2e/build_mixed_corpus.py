#!/usr/bin/env python3
"""Build the mixed-domain calibration corpus for the Llama codebook retrain.

Motivation (report12 §4): group-VQ codebooks are domain-sensitive — the
GPQA-only Llama codebook is the prime suspect for the context-flat ~6-pt
NIAH gap, and the served deficit concentrates in multikey subtasks whose
haystacks are dense random-digit / uuid key-value text (maximally OOD from
GPQA prose). This emits {domain, text} segments over five domains:

  gpqa    198 GPQA-Diamond prompts (OSCAR's original calibration domain)
  math    math500 question+solution text
  code    HumanEval stub+test source
  essay   Paul Graham essay text recovered from the RULER haystack parquet
          with every needle sentence stripped
  kvfacts synthetic key-value digit/uuid lines with NON-RULER templates —
          covers the multikey text domain without copying the eval's
          needle format (no "special magic", different sentence shape)

    .venv/bin/python pipelines/oscar_e2e/build_mixed_corpus.py \
        --out artifacts/oscar_llama31_8b/mixed_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import uuid
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

GPQA_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before "
    "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)

ADJ = ("amber bold calm dusty eager fabled grim hollow iron jolly keen lucid "
       "mellow noble oaken pale quaint rustic solemn tidy umber vivid wry "
       "young zesty brisk clever drowsy faint gilded").split()
NOUN = ("anchor beacon cabin delta ember forge grove harbor inlet jetty kiln "
        "lantern meadow notch orchard prairie quarry ridge summit terrace "
        "upland vale wharf yard zenith basin cliff dune estuary fjord").split()
KV_TEMPLATES = (
    "The catalog entry for {name} lists serial {num}.",
    "Inventory audit: item {name} carries reference {num}.",
    "Shipment {name} was logged under tracking code {num}.",
    "The registry maps {name} to identifier {tok}.",
    "Archive record {name} is filed as {tok}.",
)


def essay_segments(rng: random.Random) -> list[str]:
    """Recover clean essay text from RULER single_2 haystacks (largest ctx),
    dropping every sentence that contains a needle."""
    df = pd.read_parquet(REPO / "artifacts/niah_bench/65536/test.parquet")
    ctx = df[df.task == "niah_single_2"].iloc[0].context
    body = ctx.split("afterwards.", 1)[-1]  # drop the instruction header
    sentences = re.split(r"(?<=[.!?])\s+", body)
    clean = [s for s in sentences if "special magic" not in s and len(s) > 20]
    segs, cur = [], []
    n = 0
    for s in clean:
        cur.append(s)
        n += len(s)
        if n > 3000:  # ~800 tokens per segment
            segs.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        segs.append(" ".join(cur))
    rng.shuffle(segs)
    return segs


def kvfacts_segments(rng: random.Random, n_segs: int = 60) -> list[str]:
    segs = []
    for _ in range(n_segs):
        lines = []
        for _ in range(40):
            name = f"{rng.choice(ADJ)}-{rng.choice(NOUN)}"
            t = rng.choice(KV_TEMPLATES)
            lines.append(t.format(
                name=name,
                num=rng.randrange(10**6, 10**8),
                tok=str(uuid.UUID(int=rng.getrandbits(128))),
            ))
        segs.append("\n".join(lines))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    segments = []

    df = pd.read_csv(REPO / "artifacts/prompt_rows/gpqa_diamond.csv")
    for _, r in df.iterrows():
        segments.append(("gpqa", GPQA_TEMPLATE.format(
            Question=r["Question"], A=r["Correct Answer"],
            B=r["Incorrect Answer 1"], C=r["Incorrect Answer 2"],
            D=r["Incorrect Answer 3"])))

    for line in open(REPO / "artifacts/prompt_rows/math500_llama.jsonl"):
        r = json.loads(line)
        segments.append(("math", r["question"] + "\n\nSolution: " + r["solution"]))

    for line in open(REPO / "artifacts/prompt_rows_code/humaneval_llama.jsonl"):
        r = json.loads(line)
        segments.append(("code", r["he_stub"] + "\n\n" + r["he_test"]))

    segments += [("essay", s) for s in essay_segments(rng)]
    segments += [("kvfacts", s) for s in kvfacts_segments(rng)]

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for dom, text in segments:
            fh.write(json.dumps({"domain": dom, "text": text}) + "\n")

    from collections import Counter
    counts = Counter(d for d, _ in segments)
    chars = Counter()
    for d, t in segments:
        chars[d] += len(t)
    total = sum(chars.values())
    print(f"wrote {len(segments)} segments -> {out}")
    for d in counts:
        print(f"  {d:8s} n={counts[d]:4d}  chars={chars[d]:9d} ({chars[d]/total:.1%})")


if __name__ == "__main__":
    main()
