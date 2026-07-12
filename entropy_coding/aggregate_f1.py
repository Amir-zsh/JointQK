#!/usr/bin/env python3
"""Aggregate LongBench F1 cells (metrics.json) into a method x task table.

Walks <root>/<method>/**/metrics.json (a metrics.json holds one scalar F1 for the
cell's task, matching the worker's output layout) and prints an ordered table plus
the layer-0-excluded-style mean across the 3 tasks. Usage:
  python aggregate_f1.py <bench_root> [--order m1 m2 ...]
"""
import argparse, json, sys
from pathlib import Path

TASKS = ["lcc", "2wikimqa", "musique"]


def find_metrics(method_dir: Path):
    """Return {task: f1} for a method dir by locating each task's metrics.json."""
    out = {}
    for mj in method_dir.rglob("metrics.json"):
        # cell dir name encodes the task: longbench__<task>__...
        name = "__".join(mj.parts)
        for t in TASKS:
            if f"__{t}__" in mj.parent.name or f"/{t}/" in str(mj) or f"__{t}__" in name:
                try:
                    v = json.load(open(mj))
                    out[t] = float(v) if not isinstance(v, dict) else float(v.get("score", v.get("f1", "nan")))
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--order", nargs="*", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    methods = sorted([d.name for d in root.iterdir() if d.is_dir()]) if root.exists() else []
    if args.order:
        methods = [m for m in args.order if m in methods] + [m for m in methods if m not in args.order]

    rows = []
    for m in methods:
        r = find_metrics(root / m)
        mean3 = (sum(r.get(t, float("nan")) for t in TASKS) / 3) if all(t in r for t in TASKS) else float("nan")
        rows.append((m, r, mean3))

    w = max((len(m) for m, _, _ in rows), default=10)
    print(f"{'method':<{w}} | " + " | ".join(f"{t:>9}" for t in TASKS) + " |  mean")
    print("-" * (w + 4 + 12 * (len(TASKS) + 1)))
    for m, r, mean3 in rows:
        cells = " | ".join(f"{r.get(t, float('nan')):9.2f}" for t in TASKS)
        print(f"{m:<{w}} | {cells} | {mean3:6.2f}")


if __name__ == "__main__":
    main()
