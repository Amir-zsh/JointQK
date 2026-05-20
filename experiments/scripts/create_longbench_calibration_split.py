#!/usr/bin/env python3
"""Create a reproducible LongBench calibration/train-test split manifest.

The manifest records row IDs for the calibration examples so downstream
evaluation can exclude them. It also records held-out rows for offline
calibration validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.data.base import load_and_filter
from experiments.data.kvpress_adapter import build_kvpress_dataset_spec


DEFAULT_TASKS = (
    "qasper",
    "hotpotqa",
    "musique",
    "qmsum",
    "multi_news",
    "triviaqa",
    "passage_retrieval_en",
    "repobench-p",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="longbench")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=60)
    parser.add_argument("--train-per-task", type=int, default=50)
    parser.add_argument("--test-per-task", type=int, default=10)
    parser.add_argument("--min-tokens", type=int, default=4000)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument(
        "--output-dir",
        default="artifacts/calibration_splits/longbench_compact8_60_seed20260504_4k16k",
    )
    return parser.parse_args()


def row_record(example: Any, split: str, split_index: int) -> dict[str, Any]:
    prompt_hash = hashlib.sha256(example.prompt_text.encode("utf-8")).hexdigest()
    return {
        "split": split,
        "split_index": split_index,
        "dataset": example.dataset,
        "config": example.config,
        "task": example.metadata.get("task") or example.config,
        "row_index": int(example.row_index),
        "prompt_length": int(example.prompt_length),
        "prompt_sha256": prompt_hash,
        "metadata": example.metadata,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["split", "split_index", "dataset", "config", "task", "row_index", "prompt_length", "prompt_sha256"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.train_per_task + args.test_per_task != args.samples_per_task:
        raise ValueError("--train-per-task + --test-per-task must equal --samples-per-task")

    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    spec = build_kvpress_dataset_spec(
        name=args.dataset,
        config_names=tasks,
        metadata_fields=("task", "answers", "length", "all_classes"),
    )

    selected = load_and_filter(
        spec,
        tokenizer=tokenizer,
        num_examples_per_config=args.samples_per_task,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        config_names=tasks,
    )

    by_task: dict[str, list[Any]] = defaultdict(list)
    for example in selected:
        by_task[example.config].append(example)

    missing = []
    for task in tasks:
        count = len(by_task.get(task, []))
        if count != args.samples_per_task:
            missing.append(f"{task}: selected {count}/{args.samples_per_task}")
    if missing:
        raise RuntimeError(
            "Not enough examples passed the token filter. "
            "Adjust the length band or task set.\n" + "\n".join(missing)
        )

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    summary_rows: list[str] = []

    for task in tasks:
        examples = by_task[task]
        train_examples = examples[: args.train_per_task]
        test_examples = examples[args.train_per_task :]
        train_rows.extend(row_record(example, "train", i) for i, example in enumerate(train_examples))
        test_rows.extend(row_record(example, "test", i) for i, example in enumerate(test_examples))

        lengths = [int(example.prompt_length) for example in examples]
        train_ids = [int(example.row_index) for example in train_examples]
        test_ids = [int(example.row_index) for example in test_examples]
        summary_rows.append(
            f"| `{task}` | {len(examples)} | {min(lengths)} | {sum(lengths) / len(lengths):.1f} | "
            f"{max(lengths)} | `{train_ids}` | `{test_ids}` |"
        )

    all_rows = train_rows + test_rows
    manifest = {
        "purpose": "Final JointQK calibration split. Training rows must be excluded from downstream evaluation.",
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "tasks": list(tasks),
            "samples_per_task": args.samples_per_task,
            "train_per_task": args.train_per_task,
            "test_per_task": args.test_per_task,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "num_rows": len(all_rows),
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "train_row_ids_by_task": {
            task: [row["row_index"] for row in train_rows if row["config"] == task] for task in tasks
        },
        "test_row_ids_by_task": {
            task: [row["row_index"] for row in test_rows if row["config"] == task] for task in tasks
        },
        "rows": all_rows,
    }

    write_json(output_dir / "manifest.json", manifest)
    write_jsonl(output_dir / "train_rows.jsonl", train_rows)
    write_jsonl(output_dir / "test_rows.jsonl", test_rows)
    write_csv(output_dir / "rows.csv", all_rows)

    lines = [
        "# LongBench Calibration Split",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Model/tokenizer: `{args.model}`",
        f"- Seed: `{args.seed}`",
        f"- Token filter: `[{args.min_tokens}, {args.max_tokens}]`",
        f"- Tasks: `{', '.join(tasks)}`",
        f"- Split: `{args.train_per_task}` train + `{args.test_per_task}` test per task",
        "",
        "Training row IDs are calibration examples and should be removed from end-to-end evaluation.",
        "",
        "| Task | Selected | Min tokens | Mean tokens | Max tokens | Train row IDs | Test row IDs |",
        "|---|---:|---:|---:|---:|---|---|",
        *summary_rows,
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))
    print(f"Wrote split manifest to {output_dir / 'manifest.json'}")
    print(f"Train rows: {len(train_rows)}; test rows: {len(test_rows)}")


if __name__ == "__main__":
    main()
