#!/usr/bin/env python3
"""Add messages_sha256 to an existing calibration split manifest.

The original manifest's prompt_sha256 was tokenizer-locked (computed under one
model's chat template). Adding messages_sha256 (a tokenizer-independent hash of
the canonical messages list) makes the same manifest usable across models.

Updates: manifest.json + train_rows.jsonl + test_rows.jsonl + rows.csv (best
effort — only files that exist are touched).

Usage:
    .venv/bin/python experiments/stage1/scripts/migrate_manifest_messages_hash.py \
        --split-dir artifacts/stage1/calibration_splits/longbench_compact8_60_seed20260504_2k32k
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.stage1.calibration.common import messages_hash
from experiments.stage1.data.kvpress_adapter import build_kvpress_dataset_spec
from datasets import load_dataset
from experiments.stage1.benchmarks.evaluate_registry import DATASET_REGISTRY


def ts() -> str:
    return time.strftime("%H:%M:%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, required=True,
                   help="Calibration split directory containing manifest.json (and friends).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and report hashes without writing files.")
    return p.parse_args()


def task_to_messages_map(tasks: list[str]) -> dict[tuple[str, int], list[dict]]:
    """Pre-load each task's row → messages map by streaming the LongBench dataset.

    Returns {(config, row_index): messages_list}.
    """
    spec = build_kvpress_dataset_spec(
        name="longbench",
        config_names=tuple(tasks),
        metadata_fields=("task", "answers", "length", "all_classes"),
    )
    out: dict[tuple[str, int], list[dict]] = {}
    for task in tasks:
        print(f"[{ts()}] loading LongBench '{task}'...", flush=True)
        ds = load_dataset(DATASET_REGISTRY["longbench"], data_dir=task, split="test")
        df = ds.to_pandas()
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            messages = spec.row_to_messages(row)
            out[(task, i)] = messages
        print(f"[{ts()}]   loaded {len(df)} rows for {task}", flush=True)
    return out


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir if args.split_dir.is_absolute() else (REPO / args.split_dir)
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    tasks = sorted({r["config"] for r in manifest["rows"]})
    print(f"[{ts()}] split_dir={split_dir}")
    print(f"[{ts()}] tasks ({len(tasks)}): {tasks}")
    print(f"[{ts()}] rows: {len(manifest['rows'])}")

    msg_map = task_to_messages_map(tasks)

    # Compute messages_sha256 per row.
    n_added = 0
    n_already = 0
    for r in manifest["rows"]:
        key = (r["config"], int(r["row_index"]))
        messages = msg_map.get(key)
        if messages is None:
            raise KeyError(f"row not found in dataset: {key}")
        mhash = messages_hash(messages)
        if "messages_sha256" in r and r["messages_sha256"] == mhash:
            n_already += 1
            continue
        r["messages_sha256"] = mhash
        n_added += 1
    print(f"[{ts()}] computed: added/updated={n_added}, already-correct={n_already}")

    if args.dry_run:
        print(f"[{ts()}] DRY-RUN — not writing.")
        return

    # Backup + overwrite manifest.json
    backup = manifest_path.with_suffix(manifest_path.suffix + f".pre_messages_hash.{int(time.time())}")
    shutil.copy2(manifest_path, backup)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{ts()}] wrote {manifest_path} (backup at {backup.name})")

    # Update train_rows.jsonl + test_rows.jsonl if present.
    by_id = {(r["config"], int(r["row_index"]), r["split"]): r["messages_sha256"]
             for r in manifest["rows"]}
    for jsonl_name, split_name in [("train_rows.jsonl", "train"), ("test_rows.jsonl", "test")]:
        p = split_dir / jsonl_name
        if not p.exists():
            continue
        backup = p.with_suffix(p.suffix + f".pre_messages_hash.{int(time.time())}")
        shutil.copy2(p, backup)
        new_lines = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            mh = by_id.get((row["config"], int(row["row_index"]), row.get("split", split_name)))
            if mh is None:
                # Fall back to lookup without explicit split.
                mh = next((v for (c, i, s), v in by_id.items()
                           if c == row["config"] and int(i) == int(row["row_index"])), None)
            if mh is not None:
                row["messages_sha256"] = mh
            new_lines.append(json.dumps(row))
        p.write_text("\n".join(new_lines) + "\n")
        print(f"[{ts()}] wrote {p} (backup {backup.name})")

    # Update rows.csv if present (add messages_sha256 column).
    rows_csv = split_dir / "rows.csv"
    if rows_csv.exists():
        backup = rows_csv.with_suffix(rows_csv.suffix + f".pre_messages_hash.{int(time.time())}")
        shutil.copy2(rows_csv, backup)
        with rows_csv.open("r") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = list(reader.fieldnames or [])
        for r in existing_rows:
            mh = by_id.get((r["config"], int(r["row_index"]), r.get("split")))
            if mh is None:
                mh = next((v for (c, i, _), v in by_id.items()
                           if c == r["config"] and int(i) == int(r["row_index"])), None)
            r["messages_sha256"] = mh or ""
        new_fields = existing_fields + (["messages_sha256"] if "messages_sha256" not in existing_fields else [])
        with rows_csv.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writeheader()
            writer.writerows(existing_rows)
        print(f"[{ts()}] wrote {rows_csv} (backup {backup.name})")

    print(f"[{ts()}] done.")


if __name__ == "__main__":
    main()
