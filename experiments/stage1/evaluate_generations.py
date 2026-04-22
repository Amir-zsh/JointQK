from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.stage1.benchmarks.evaluate_registry import SCORER_REGISTRY
from experiments.stage1.toolkit import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated outputs from collect_query_stats.py using the "
            "vendored kvpress benchmark scorers."
        )
    )
    parser.add_argument(
        "--stats_dir",
        required=True,
        help="Directory produced by collect_query_stats.py (contains manifest.json and examples/).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to <stats_dir>/evaluation.",
    )
    return parser.parse_args()


def _extract_generated_text(entry: dict[str, Any], stats_dir: Path) -> str:
    if "generated_text" in entry:
        return str(entry["generated_text"])
    example_path = stats_dir / entry["file"]
    payload = torch.load(example_path, map_location="cpu")
    return str(payload.get("generated_text", ""))


def _build_dataframe(entries: list[dict[str, Any]], stats_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        metadata = entry.get("metadata") or {}
        rows.append({**metadata, "predicted_answer": _extract_generated_text(entry, stats_dir)})
    return pd.DataFrame(rows)


def _write_summary_markdown(
    path: Path,
    dataset_name: str,
    per_config_results: dict[str, Any],
    per_config_count: dict[str, int],
) -> None:
    lines = [f"# Evaluation: {dataset_name}", ""]
    for config in sorted(per_config_results):
        result = per_config_results[config]
        lines.append(f"## {config} (n={per_config_count[config]})")
        lines.append("")
        if isinstance(result, dict):
            lines.append("| Metric | Value |")
            lines.append("| --- | ---: |")
            for key in sorted(result):
                value = result[key]
                if isinstance(value, (int, float)):
                    lines.append(f"| {key} | {value:.4f} |")
                else:
                    lines.append(f"| {key} | {value} |")
        else:
            lines.append(f"score: `{result}`")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    stats_dir = Path(args.stats_dir)
    output_dir = ensure_dir(args.output_dir or (stats_dir / "evaluation"))

    manifest = json.loads((stats_dir / "manifest.json").read_text())
    dataset_name = manifest["dataset"]
    if dataset_name not in SCORER_REGISTRY:
        raise KeyError(
            f"No scorer registered for dataset '{dataset_name}'. "
            f"Available: {sorted(SCORER_REGISTRY)}"
        )
    scorer = SCORER_REGISTRY[dataset_name]

    entries_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest["examples"]:
        entries_by_config[entry["config"]].append(entry)

    per_config_results: dict[str, Any] = {}
    per_config_count: dict[str, int] = {}
    per_example: list[dict[str, Any]] = []

    for config, entries in entries_by_config.items():
        df = _build_dataframe(entries, stats_dir)
        per_config_results[config] = scorer(df)
        per_config_count[config] = len(entries)

        for entry, prediction in zip(entries, df["predicted_answer"].tolist()):
            metadata = entry.get("metadata") or {}
            per_example.append(
                {
                    "dataset": entry["dataset"],
                    "config": entry["config"],
                    "row_index": entry["row_index"],
                    "prediction": prediction,
                    "answers": metadata.get("answers"),
                    "answer": metadata.get("answer"),
                    "task": metadata.get("task"),
                }
            )

    save_json(
        output_dir / "evaluation.json",
        {
            "dataset": dataset_name,
            "per_config": per_config_results,
            "per_config_count": per_config_count,
            "per_example": per_example,
        },
    )
    _write_summary_markdown(output_dir / "summary.md", dataset_name, per_config_results, per_config_count)


if __name__ == "__main__":
    main()
