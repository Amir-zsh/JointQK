#!/usr/bin/env python3
"""Build the sharded job queue for gpu_pool_runner.sh.

Splits each remaining cell's rows into shards (separate jsonl files +
__sN out dirs), skips cells whose merged or monolithic metrics.json already
exists, and writes an arm-major TSV so workers' arm-affinity claiming keeps
servers alive across consecutive jobs.

    .venv/bin/python pipelines/oscar_e2e/build_pool_queue.py \
        --out logs/pool_queue.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

LLAMA_ARMS = ["bf16", "int2", "vq2", "quarot", "naive"]
SAMP = "--samples 5 --temperature 0.6 --top-p 0.9 --top-k -1"

# (cell, rows_file, n_shards, extra_args)
LLAMA_CELLS = [
    # niah_16384 intentionally absent: in flight monolithically at the
    # monolith->pool transition (re-add if an arm's cell failed).
    ("niah_32768", "artifacts/prompt_rows/niah_32768_llama.jsonl", 4, "--threads 4"),
    ("niah_65536", "artifacts/prompt_rows/niah_65536_llama.jsonl", 8, "--threads 2"),
    ("gpqa", "artifacts/prompt_rows/gpqa_diamond_llama.jsonl", 2, f"{SAMP} --threads 24"),
    ("math500", "artifacts/prompt_rows/math500_llama.jsonl", 4, f"{SAMP} --threads 24"),
    ("aime25", "artifacts/prompt_rows/aime25_llama.jsonl", 1, f"{SAMP} --threads 12"),
    ("humaneval", "artifacts/prompt_rows_code/humaneval_llama.jsonl", 2, f"{SAMP} --threads 24"),
]
# Qwen extras: VQ-V 64K NIAH + vq2 8K/16K completeness cells.
QWEN_JOBS = [
    ("qwen-vqv", "vqv_niah_65536", "artifacts/prompt_rows/niah_65536_qwen.jsonl", 8, "--threads 2"),
    ("qwen-vq2", "vq2_niah_8192", "artifacts/prompt_rows/niah_8192_qwen.jsonl", 2, "--threads 12"),
    ("qwen-vq2", "vq2_niah_16384", "artifacts/prompt_rows/niah_16384_qwen.jsonl", 2, "--threads 8"),
]
QWEN_OUT = "artifacts/oscar_e2e/lh/pool"
LLAMA_OUT = "artifacts/oscar_llama31_8b/grid"


def shard_rows(rows_path: Path, n: int) -> list[Path]:
    lines = rows_path.read_text().splitlines()
    outs = []
    base = rows_path.with_suffix("")
    for i in range(n):
        p = Path(f"{base}__s{i}of{n}.jsonl")
        if not p.exists():
            p.write_text("\n".join(lines[i::n]) + "\n")
        outs.append(p)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    jobs = []

    def add(arm, cell_dir, cell, rows_file, n_shards, extra):
        cell_path = REPO / cell_dir / cell
        if (cell_path / "metrics.json").exists():
            return  # already complete (monolithic or merged)
        rows = REPO / rows_file
        if not rows.exists():
            print(f"[queue] MISSING rows {rows_file} — skipped")
            return
        if n_shards == 1:
            jobs.append((arm, f"{cell_dir}/{cell}", rows_file, extra))
            return
        for i, sp in enumerate(shard_rows(rows, n_shards)):
            out = f"{cell_dir}/{cell}__s{i}"
            if (REPO / out / "metrics.json").exists():
                continue
            jobs.append((arm, out, str(sp.relative_to(REPO)), extra))

    for arm in LLAMA_ARMS:
        for cell, rows, n, extra in LLAMA_CELLS:
            add(arm, f"{LLAMA_OUT}/{arm}", cell, rows, n, extra)
    for arm, cell, rows, n, extra in QWEN_JOBS:
        add(arm, QWEN_OUT, cell, rows, n, extra)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for arm, od, rows, extra in jobs:
            fh.write(f"TODO\t{arm}\t{od}\t{rows}\t{extra}\n")
    print(f"[queue] wrote {len(jobs)} jobs -> {out}")
    from collections import Counter
    print("[queue] per arm:", dict(Counter(j[0] for j in jobs)))


if __name__ == "__main__":
    main()
