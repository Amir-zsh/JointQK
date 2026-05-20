#!/usr/bin/env python3
"""Aggregate Phase 6 decode-scope ablation: write decode_decision.txt with WINNER=A or B.

Decision rule: if max |Mode B − Mode A| across (task × kb) cells < 2 pp → WINNER=B; else A.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-decision", required=True)
    p.add_argument("--threshold-pp", type=float, default=2.0,
                   help="If max(|B-A|) < threshold_pp percentage points, B wins")
    args = p.parse_args()

    root = Path(args.input_dir)
    cells_a = {}  # (task, kb) -> F1 mode A
    cells_b = {}
    for task in ("qasper", "narrativeqa"):
        for kb in (2, 3, 4):
            for tag, store in [("modeA", cells_a), ("modeB", cells_b)]:
                d = root / f"{task}_{tag}_k{kb}"
                f = find_metric(d)
                if f is not None:
                    store[(task, kb)] = f

    print(f"{'cell':<18s}  modeA   modeB   |B-A|")
    max_diff = 0.0
    for task in ("qasper", "narrativeqa"):
        for kb in (2, 3, 4):
            a = cells_a.get((task, kb))
            b = cells_b.get((task, kb))
            if a is None or b is None:
                print(f"{task} k={kb:<5d}  MISSING")
                continue
            diff = abs(b - a)
            max_diff = max(max_diff, diff)
            print(f"{task} k={kb:<5d}  {a:6.2f}  {b:6.2f}  {diff:5.2f}")

    print(f"\nmax |B-A| = {max_diff:.2f} pp")
    winner = "B" if max_diff < args.threshold_pp else "A"
    print(f"WINNER: {winner}")

    out = Path(args.output_decision)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"WINNER={winner}\nMAX_DIFF_PP={max_diff:.2f}\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
