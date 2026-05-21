#!/usr/bin/env python3
"""Aggregate Phase 1D K-basis stability JSONL rows into JSON + Markdown."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def summarize(values: list[float]) -> dict[str, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
        "min": float(t.min()),
        "p10": float(torch.quantile(t, 0.10)),
        "p50": float(torch.quantile(t, 0.50)),
        "p90": float(torch.quantile(t, 0.90)),
        "max": float(t.max()),
        "n": int(t.numel()),
    }


def metric_sort_key(name: str) -> tuple[int, int]:
    if m := re.match(r"subspace_overlap_r(\d+)$", name):
        return (0, int(m.group(1)))
    if m := re.match(r"log2_regret_k(\d+)$", name):
        return (1, int(m.group(1)))
    if m := re.match(r"bit_l1_k(\d+)$", name):
        return (2, int(m.group(1)))
    return (9, 0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", nargs="+", required=True, help="JSONL row files to merge.")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    for path_s in args.rows:
        path = Path(path_s)
        if not path.exists():
            raise SystemExit(f"Missing rows file: {path}")
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("No Phase 1D rows found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = output_dir / "phase1d_k_basis_stability_rows.jsonl"
    with combined_rows.open("w") as f:
        for row in sorted(rows, key=lambda r: (r["source"], r["n_label"], r["rep"], r["eval"])):
            f.write(json.dumps(row, sort_keys=True) + "\n")

    metric_names = sorted(
        [k for k in rows[0].keys() if k.startswith(("subspace_overlap_r", "log2_regret_k", "bit_l1_k"))],
        key=metric_sort_key,
    )
    regret_metrics = [m for m in metric_names if m.startswith("log2_regret_k")]
    overlap_r64 = "subspace_overlap_r64"
    if overlap_r64 not in metric_names:
        overlap_r64 = next((m for m in metric_names if m.startswith("subspace_overlap_r")), metric_names[0])

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], int(row["n_label"]), row["eval"])].append(row)

    summary: dict[str, Any] = {"n_rows": len(rows), "groups": {}}
    for key, group_rows in sorted(grouped.items()):
        source, n_label, eval_name = key
        out_key = f"{source}|n={n_label}|eval={eval_name}"
        summary["groups"][out_key] = {
            metric: summarize([float(r[metric]) for r in group_rows])
            for metric in metric_names
        }
        summary["groups"][out_key]["n_trials"] = len(group_rows)

    summary_path = output_dir / "phase1d_k_basis_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    header = (
        "| source | n label | eval | "
        + overlap_r64.replace("subspace_overlap_", "overlap ")
        + " mean | "
        + " | ".join(m.replace("log2_regret_", "regret ") + " mean" for m in regret_metrics)
        + " | trials |"
    )
    align = "|---|---:|---|---:|" + "".join("---:|" for _ in regret_metrics) + "---:|"
    lines = [
        "# Phase 1D K-Basis Stability Summary",
        "",
        "Lower `log2_regret_k*` is better. Zero means the subset-calibrated basis/allocation",
        "matches the full eval-reference basis/allocation under the closed-form K distortion metric.",
        "",
        header,
        align,
    ]
    for key in sorted(summary["groups"]):
        source, n_part, eval_part = key.split("|")
        n_label = n_part.split("=", 1)[1]
        eval_name = eval_part.split("=", 1)[1]
        g = summary["groups"][key]
        regret_values = " | ".join(f"{g[m]['mean']:.4f}" for m in regret_metrics)
        lines.append(
            f"| {source} | {n_label} | {eval_name} | "
            f"{g[overlap_r64]['mean']:.4f} | {regret_values} | {g['n_trials']} |"
        )
    md_path = output_dir / "phase1d_k_basis_stability_summary.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Merged rows:   {combined_rows}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote table:   {md_path}")


if __name__ == "__main__":
    main()
