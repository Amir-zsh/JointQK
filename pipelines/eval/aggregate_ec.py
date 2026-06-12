#!/usr/bin/env python3
"""Aggregate the EC k2/v2 bench cells against the v7 Llama baselines.

Walks artifacts/bench_ec/llama31_8b/ (EC cells) and the kept v7 grid at
artifacts/stage1/downstream_v7/llama31_8b/ (full_precision, turboquant_k2_v2,
jointqk_k2_v2, kivi_int2), pulls each cell's F1 (latest-mtime metrics.json,
same rule as aggregate_longbench), annotates EC variants with their held-out
coded rate from the bundle, and prints the comparison table.

    python pipelines/eval/aggregate_ec.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
EC_BENCH = REPO / "artifacts/bench_ec/llama31_8b"
BASELINES = REPO / "artifacts/stage1/downstream_v7/llama31_8b"
EC_BUNDLES = REPO / "artifacts/ec/llama31_8b"
OUT = REPO / "artifacts/ec/llama31_8b/bench_summary.json"

TASKS = ["lcc", "musique", "2wikimqa"]
BASELINE_METHODS = ["full_precision", "turboquant_k2_v2", "jointqk_k2_v2", "kivi_int2"]


def cell_f1(run_dir: Path) -> float | None:
    cands = sorted(run_dir.glob("**/metrics.json"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not cands:
        return None
    m = json.loads(cands[0].read_text())
    if isinstance(m, (int, float)):
        return float(m)
    if isinstance(m, dict):
        for v in m.values():
            if isinstance(v, (int, float)):
                return float(v)
    return None


def main() -> None:
    rows: dict[str, dict] = {}
    for m in BASELINE_METHODS:
        rows[m] = {"f1": {}, "rate": 2.125 if m.startswith("turboquant") else None}
        for t in TASKS:
            d = BASELINES / f"{m}_{t}"
            if d.exists():
                rows[m]["f1"][t] = cell_f1(d)

    for cell in sorted(EC_BENCH.glob("ec_*_k2_v2_*")) if EC_BENCH.exists() else []:
        label = cell.name  # ec_{basis}_dz{dz}_k2_v2_{task}
        head, task = label.rsplit("_k2_v2_", 1)
        rows.setdefault(head, {"f1": {}, "rate": None})
        rows[head]["f1"][task] = cell_f1(cell)

    for blob_path in sorted(EC_BUNDLES.glob("ec_bundle__*.pt")):
        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        head = f"ec_{blob['basis']}_dz{blob['dz']:g}"
        if head in rows:
            rows[head]["rate"] = blob["achieved_rate_heldout_pooled"]
            rows[head]["rate_per_task"] = blob["achieved_rate_per_task"]

    width = max(len(m) for m in rows) + 2
    print(f"{'method':<{width}}" + "".join(f"{t:>10}" for t in TASKS)
          + f"{'mean':>8} {'K rate b/c':>11}")
    summary = {}
    for m, r in rows.items():
        f1s = [r["f1"].get(t) for t in TASKS]
        mean = sum(v for v in f1s if v is not None) / max(1, sum(v is not None for v in f1s))
        rate = r.get("rate")
        rate_s = f"{rate:.3f}" if isinstance(rate, float) else ("2 (grid)" if m.startswith(("jointqk", "kivi")) else "fp16" if m == "full_precision" else "-")
        print(f"{m:<{width}}" + "".join(
            f"{(f'{v:.2f}' if v is not None else '—'):>10}" for v in f1s)
            + f"{mean:8.2f} {rate_s:>11}")
        summary[m] = {"f1": r["f1"], "mean_f1": mean, "rate": rate,
                      "rate_per_task": r.get("rate_per_task")}

    tq = summary.get("turboquant_k2_v2", {}).get("f1", {})
    print("\nDelta vs turboquant_k2_v2 (positive = beats TQ):")
    for m, s in summary.items():
        if m.startswith("ec_"):
            ds = {t: (s["f1"].get(t) - tq.get(t)) for t in TASKS
                  if s["f1"].get(t) is not None and tq.get(t) is not None}
            wins = sum(v > 0 for v in ds.values())
            print(f"  {m}: " + " ".join(f"{t}:{v:+.2f}" for t, v in ds.items())
                  + f"  ({wins}/{len(ds)} tasks won)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
