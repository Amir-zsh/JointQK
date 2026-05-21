#!/usr/bin/env python3
"""Aggregate Phase 1AB sweep: pick V method + V bits lock, K bit floor.

Inputs:
- artifacts/v_bases/sweep/full_precision/<longbench__qasper__...>/metrics.json
- artifacts/v_bases/sweep/{vonly_<method>_v<b>, konly_k<b>}/<...>/metrics.json

Decisions:
- V lock: smallest v_bits with rel_F1 >= threshold across all V methods; within
  the same bit budget, choose the best measured rel_F1.
- K floor: smallest k_bits with rel_F1 >= threshold for k_method=r_sym_waterfill.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

V_METHOD_PRIORITY = {
    "v_turboquant": 4,
    "v_eigen_waterfill": 3,
    "v_eigen_uniform": 2,
    "v_random": 1,
}


def find_metric(run_dir: Path) -> float | None:
    """LongBench eval writes metrics.json in a subdir named by params; load the first."""
    candidates = list(run_dir.glob("**/metrics.json"))
    if not candidates:
        return None
    metrics = json.loads(candidates[0].read_text())
    # LongBench scorer returns a single number; fall through if dict
    if isinstance(metrics, (int, float)):
        return float(metrics)
    if isinstance(metrics, dict):
        # take first numeric value
        for v in metrics.values():
            if isinstance(v, (int, float)):
                return float(v)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Phase 1AB sweep root (sweep/)")
    p.add_argument("--output-v-lock", required=True)
    p.add_argument("--output-k-floor", required=True)
    p.add_argument("--threshold", type=float, default=0.97, help="rel_F1 acceptability cutoff")
    args = p.parse_args()

    root = Path(args.input_dir)
    full_dir = root / "full_precision"
    full_f1 = find_metric(full_dir)
    if full_f1 is None:
        raise SystemExit(f"No full-precision metrics found in {full_dir}")

    print(f"Full-precision F1: {full_f1:.4f}")

    # V-only cells
    print("\nV-only sweep (rel_F1 vs full):")
    print(f"{'method':<22s}  v=2     v=3     v=4")
    v_cells = {}  # (method, vb) -> rel_F1
    for vmethod in ("v_random", "v_eigen_uniform", "v_eigen_waterfill", "v_turboquant"):
        row = f"{vmethod:<22s}  "
        for vb in (2, 3, 4):
            d = root / f"vonly_{vmethod}_v{vb}"
            f = find_metric(d)
            if f is None:
                row += f"{'MISS':<7s} "
                continue
            rel = f / full_f1 if full_f1 else 0.0
            v_cells[(vmethod, vb)] = rel
            row += f"{rel:.4f}  "
        print(row)

    # K-only cells
    print("\nK-only sweep (rel_F1 vs full):")
    print(f"{'k_bits':<7s}  rel_F1")
    k_cells = {}
    for kb in (2, 3, 4):
        d = root / f"konly_k{kb}"
        f = find_metric(d)
        if f is None:
            print(f"{kb:<7d}  MISS")
            continue
        rel = f / full_f1 if full_f1 else 0.0
        k_cells[kb] = rel
        print(f"{kb:<7d}  {rel:.4f}")

    # V lock decision
    acceptable_v = [(vm, vb, rel) for (vm, vb), rel in v_cells.items() if rel >= args.threshold]
    if acceptable_v:
        # Sort: smallest vb first, then best measured rel_F1. Priority is only
        # a deterministic fallback for exact metric ties.
        acceptable_v.sort(key=lambda t: (t[1], -t[2], -V_METHOD_PRIORITY[t[0]]))
        winner_vm, winner_vb, winner_rel = acceptable_v[0]
    else:
        # No cell meets threshold — pick the best by rel_F1
        all_v = sorted(v_cells.items(), key=lambda kv: -kv[1])
        if not all_v:
            raise SystemExit("No V-only metrics available; cannot decide V lock.")
        ((winner_vm, winner_vb), winner_rel) = all_v[0]
        print(f"\nWARNING: no cell met threshold {args.threshold}; "
              f"falling back to best cell (rel={winner_rel:.4f})")

    print(f"\nV LOCK: V_METHOD={winner_vm}, V_BITS={winner_vb}, REL_F1={winner_rel:.4f}")
    Path(args.output_v_lock).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_v_lock).write_text(
        f"V_METHOD={winner_vm}\nV_BITS={winner_vb}\nV_REL_F1_AT_LOCK={winner_rel:.4f}\n"
    )

    # K floor decision
    acceptable_k = [(kb, rel) for kb, rel in k_cells.items() if rel >= args.threshold]
    if acceptable_k:
        acceptable_k.sort(key=lambda t: t[0])
        floor_kb, floor_rel = acceptable_k[0]
    else:
        all_k = sorted(k_cells.items(), key=lambda kv: -kv[1])
        floor_kb, floor_rel = all_k[0] if all_k else (None, None)
        print(f"\nWARNING: no K cell met threshold {args.threshold}")

    print(f"K FLOOR: K_FLOOR={floor_kb}, REL_F1_AT_FLOOR={floor_rel:.4f}" if floor_kb else "K FLOOR: undefined")
    Path(args.output_k_floor).write_text(
        f"K_FLOOR={floor_kb}\nK_REL_F1_AT_FLOOR={floor_rel:.4f}\n" if floor_kb else "K_FLOOR=undefined\n"
    )


if __name__ == "__main__":
    main()
