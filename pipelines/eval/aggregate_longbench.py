#!/usr/bin/env python3
"""Aggregate Phase 7 LongBench full-sweep results into a single summary JSON.

Walks artifacts/downstream/<model>/<config>_<task>/.../metrics.json,
collects per-(task, method, k_bits, model) F1 scores, and emits a structured JSON.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CONFIG_RE = re.compile(
    r"^(?P<method>full_precision|jointqk_k\d+_v\d+|turboquant_k\d+_v\d+|kivi_int4)_(?P<task>.+)$"
)


def find_metric(run_dir: Path) -> float | None:
    # kvpress evaluate.py writes results to numbered subdirs (1/, 2/, ...) — pick the LATEST.
    candidates = sorted(run_dir.glob("**/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    metrics = json.loads(candidates[0].read_text())
    if isinstance(metrics, (int, float)):
        return float(metrics)
    if isinstance(metrics, dict):
        for v in metrics.values():
            if isinstance(v, (int, float)):
                return float(v)
    return None


def collect_model(model_dir: Path) -> dict:
    """Return {task: {config_name: F1}}."""
    out: dict = {}
    for sub in sorted(model_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = CONFIG_RE.match(sub.name)
        if not m:
            continue
        method = m.group("method")
        task = m.group("task")
        f1 = find_metric(sub)
        if f1 is None:
            continue
        out.setdefault(task, {})[method] = f1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-dir", required=True)
    p.add_argument("--llama-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    summary = {
        "qwen3_8b": collect_model(Path(args.qwen_dir)),
        "llama31_8b": collect_model(Path(args.llama_dir)),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")

    # Print per-model retention summary
    for model, by_task in summary.items():
        print(f"\n=== {model} ===")
        if not by_task:
            print("  (no results)")
            continue
        # Print one row per task: full | jointqk@k=2/3/4 | turboquant@k=2/3/4 | kivi
        all_methods = sorted({m for d in by_task.values() for m in d})
        print(f"{'task':<20s}  " + "  ".join(f"{m:>22s}" for m in all_methods))
        for task in sorted(by_task):
            row = f"{task:<20s}  "
            for m in all_methods:
                v = by_task[task].get(m)
                row += f"{v:>22.2f}" if v is not None else f"{'-':>22s}"
                row += "  "
            print(row)


if __name__ == "__main__":
    main()
