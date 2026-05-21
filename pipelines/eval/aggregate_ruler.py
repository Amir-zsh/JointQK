#!/usr/bin/env python3
"""Aggregate Phase 7 RULER NIAH results into summary JSON."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import json
import re
from pathlib import Path

CONFIG_RE = re.compile(
    r"^ruler_(?P<method>full|jointqk_k\d+|turboquant_k\d+|kivi)_(?P<ctx>\d+)$"
)


def find_metric(run_dir: Path) -> float | None:
    candidates = list(run_dir.glob("**/metrics.json"))
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


def collect(model_dir: Path) -> dict:
    out: dict = {}
    for sub in sorted(model_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = CONFIG_RE.match(sub.name)
        if not m:
            continue
        ctx = int(m.group("ctx"))
        method = m.group("method")
        f1 = find_metric(sub)
        if f1 is None:
            continue
        out.setdefault(ctx, {})[method] = f1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-dir", required=True)
    p.add_argument("--llama-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    summary = {
        "qwen3_8b": collect(Path(args.qwen_dir)),
        "llama31_8b": collect(Path(args.llama_dir)),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")

    for model, by_ctx in summary.items():
        print(f"\n=== {model} ===")
        if not by_ctx:
            print("  (no results)")
            continue
        all_methods = sorted({m for d in by_ctx.values() for m in d})
        print(f"{'ctx':<7s}  " + "  ".join(f"{m:>20s}" for m in all_methods))
        for ctx in sorted(by_ctx):
            row = f"{ctx:<7d}  "
            for m in all_methods:
                v = by_ctx[ctx].get(m)
                row += f"{v:>20.2f}" if v is not None else f"{'-':>20s}"
                row += "  "
            print(row)


if __name__ == "__main__":
    main()
