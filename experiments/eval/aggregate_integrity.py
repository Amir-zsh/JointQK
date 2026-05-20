"""Aggregate Tier 1 attention-parity CSVs into a comparison summary.

Reads CSV(s) from experiments/logs/integrity/tier1/*.csv and prints:
1. Direct vs wrapper parity (Tier 1.1 sanity): rows from direct_turboquant
   and turboquant for the same (model, K, V, ctx) should match within
   tolerance.
2. JointQK vs TurboQuant comparison (Tier 1.3): for each (K, ctx), compare
   cos sim and top-1 / top-5.

Tolerance for parity: cos diff < 1e-3, top1/top5 within 2pp.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_csv(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({**r,
                         "k_bits": int(r["k_bits"]),
                         "v_bits": int(r["v_bits"]),
                         "ctx": int(r["ctx"]),
                         "cos": float(r["cos"]),
                         "top1_pct": float(r["top1_pct"]),
                         "top5_pct": float(r["top5_pct"])})
    return rows


def parity_check(rows, model_filter):
    """Compare direct_turboquant vs turboquant for the same config."""
    print(f"\n=== Direct vs Wrapper parity (model={model_filter}) ===")
    direct = {(r["k_bits"], r["v_bits"], r["ctx"]): r
              for r in rows
              if r["model"] == model_filter and r["press"] == "direct_turboquant"}
    wrapper = {(r["k_bits"], r["v_bits"], r["ctx"]): r
               for r in rows
               if r["model"] == model_filter and r["press"] == "turboquant"}

    fails = 0
    print(f"{'K':>3s} {'V':>3s} {'ctx':>5s}   "
          f"{'cos_d':>8s} {'cos_w':>8s} {'cos_diff':>10s}   "
          f"{'t1_d':>6s} {'t1_w':>6s} {'t1_diff':>8s}")
    for key in sorted(direct.keys() & wrapper.keys()):
        d = direct[key]
        w = wrapper[key]
        cos_diff = abs(d["cos"] - w["cos"])
        t1_diff = abs(d["top1_pct"] - w["top1_pct"])
        ok = (cos_diff < 1e-3) and (t1_diff < 2.0)
        flag = "" if ok else "  ! FAIL"
        if not ok:
            fails += 1
        print(f"{key[0]:>3d} {key[1]:>3d} {key[2]:>5d}   "
              f"{d['cos']:>8.6f} {w['cos']:>8.6f} {cos_diff:>10.2e}   "
              f"{d['top1_pct']:>6.2f} {w['top1_pct']:>6.2f} {t1_diff:>8.2f}{flag}")
    if fails == 0:
        print(f"  → ALL {len(direct.keys() & wrapper.keys())} cells WITHIN tolerance")
    else:
        print(f"  → {fails} cells FAILED parity")
    return fails


def head_to_head(rows, model_filter):
    """JointQK vs TurboQuant attention-score comparison."""
    print(f"\n=== JointQK vs TurboQuant (model={model_filter}) ===")
    print(f"{'method':>30s} {'K':>3s} {'V':>3s} {'ctx':>5s}   "
          f"{'cos':>8s} {'top1_pct':>8s} {'top5_pct':>8s}")

    keep = [r for r in rows if r["model"] == model_filter and
            r["press"] in ("turboquant", "jointqk")]

    # group rows by (K, V, ctx, press, layer0_full)
    keep.sort(key=lambda r: (r["k_bits"], r["ctx"], r["press"], r["layer0_full"]))
    for r in keep:
        method = r["press"]
        if method == "jointqk":
            method += f"(L0_full={r['layer0_full']})"
        print(f"{method:>30s} {r['k_bits']:>3d} {r['v_bits']:>3d} {r['ctx']:>5d}   "
              f"{r['cos']:>8.6f} {r['top1_pct']:>8.2f} {r['top5_pct']:>8.2f}")

    # Pairwise comparison at each (K, ctx) — JointQK best vs TurboQuant
    print(f"\n  Comparison at K=2 (most discriminating):")
    for ctx in sorted({r["ctx"] for r in keep if r["k_bits"] == 2}):
        tq = next((r for r in keep if r["press"] == "turboquant" and r["k_bits"] == 2 and r["ctx"] == ctx), None)
        # Pick best JointQK variant
        jq_T = next((r for r in keep if r["press"] == "jointqk" and r["layer0_full"] == "True" and r["k_bits"] == 2 and r["ctx"] == ctx), None)
        jq_F = next((r for r in keep if r["press"] == "jointqk" and r["layer0_full"] == "False" and r["k_bits"] == 2 and r["ctx"] == ctx), None)
        if not (tq and jq_T):
            continue
        t1_lead_T = jq_T["top1_pct"] - tq["top1_pct"]
        line = (f"    ctx={ctx}: TurboQuant top1={tq['top1_pct']:.1f}%   "
                f"JointQK_L0Full=True top1={jq_T['top1_pct']:.1f}% (Δ={t1_lead_T:+.1f})")
        if jq_F:
            t1_lead_F = jq_F["top1_pct"] - tq["top1_pct"]
            line += f"   JointQK_L0Full=False top1={jq_F['top1_pct']:.1f}% (Δ={t1_lead_F:+.1f})"
        print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="append", required=True,
                   help="CSV file(s) to load (repeat for multiple)")
    args = p.parse_args()

    all_rows = []
    for csv_path in args.csv:
        rows = load_csv(Path(csv_path))
        print(f"Loaded {len(rows)} rows from {csv_path}")
        all_rows.extend(rows)

    models = sorted({r["model"] for r in all_rows})
    print(f"\nModels in CSVs: {models}")

    fails = 0
    for m in models:
        if any(r["press"] == "direct_turboquant" for r in all_rows if r["model"] == m):
            fails += parity_check(all_rows, m)
        if any(r["press"] == "jointqk" for r in all_rows if r["model"] == m):
            head_to_head(all_rows, m)

    print(f"\n=== Summary ===  parity failures: {fails}")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
