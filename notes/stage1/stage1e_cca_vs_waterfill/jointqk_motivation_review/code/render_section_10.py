#!/usr/bin/env python
"""Render §10 markdown tables (and §8.2 / §5 numbers) from empirical_results.json.

Usage:
    ./.venv/bin/python notes/stage1/stage1e_cca_vs_waterfill/jointqk_motivation_review/code/render_section_10.py

Prints markdown to stdout. Manually copy-paste into jointqk_motivation.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = THIS_DIR.parent / "results" / "empirical_results.json"


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    args = ap.parse_args()

    with open(args.results) as f:
        r = json.load(f)

    print("# Markdown blocks for jointqk_motivation.md\n")

    # --- §10.1 table -----------------------------------------------------
    print("## §10.1 — Geomean prediction table\n")
    g = r["section_10_1_geomean_predictions"]
    floor = g["hadamard_floor"]
    print("| Basis | $(\\prod w_j)^{1/d}$ (Q-side) | $(\\prod \\sigma_j^2)^{1/d}$ (K-side) | product (geomean) |")
    print("|---|---:|---:|---:|")
    print(f"| Hadamard floor $(\\det \\Sigma_\\cdot)^{{1/d}}$ | — | — | **{fmt(floor['per_head_floor_l0excl_mean'])}** ← unreachable ($[\\Sigma_Q, \\Sigma_K] \\neq 0$) |")
    for label, key in [
        ("**$R_{\\text{sym}}$**", "R_sym"),
        ("$V_Q$ (Q-only)", "V_Q"),
        ("$V_K$ (K-only)", "V_K"),
        ("$V_h$ (orthogonal CCA)", "V_h"),
        ("Random orth (TurboQuant proxy)", "random_orth_TQ_proxy"),
    ]:
        if key not in g: continue
        d = g[key]
        print(f"| {label} | {fmt(d['Q_side_geomean__l0excl'])} | {fmt(d['K_side_geomean__l0excl'])} | **{fmt(d['product_geomean__l0excl'])}** |")
    print()

    # --- §8.2 headroom ---------------------------------------------------
    h = r["section_8_2_headroom"]
    print("## §8.2 — Headroom\n")
    print(f"R_sym geomean / Hadamard floor = {fmt(h['R_sym_geomean'])} / {fmt(h['hadamard_floor'])} ≈ **{fmt(h['R_sym_over_floor'], 2)}×**\n")
    print(f"V_Q  geomean / Hadamard floor = {fmt(h['V_Q_geomean'])} / {fmt(h['hadamard_floor'])} ≈ **{fmt(h['V_Q_over_floor'], 2)}×**\n")

    # --- §10.2 predicted ratios -----------------------------------------
    pr = r["section_10_2_predicted_ratios"]
    print("## §10.2 — Predicted ratios\n")
    print("| Comparison | Predicted (geomean) |")
    print("|---|---:|")
    for label, key in [
        ("$R_{\\text{sym}} / V_Q$", "R_sym_over_V_Q"),
        ("$V_Q / \\text{TurboQuant proxy}$", "V_Q_over_TQ_proxy"),
        ("$R_{\\text{sym}} / \\text{TurboQuant proxy}$", "R_sym_over_TQ_proxy"),
    ]:
        print(f"| {label} | {fmt(pr[key])} |")
    print()

    # --- §10.2 predicted vs measured ------------------------------------
    if "section_10_2_predicted_vs_measured" in r:
        pvm = r["section_10_2_predicted_vs_measured"]
        print("## §10.2 — Predicted vs measured (real quantization at b=3)\n")
        print("| Comparison | Predicted (geomean) | Measured (geo distortion) | Agreement |")
        print("|---|---:|---:|---:|")
        for label, key in [
            ("$R_{\\text{sym}} / V_Q$", "R_sym_over_V_Q"),
            ("$V_Q / \\text{V3}$ (real)", "V_Q_over_TQ_real"),
            ("$R_{\\text{sym}} / \\text{V3}$ (real)", "R_sym_over_TQ_real"),
        ]:
            if key not in pvm: continue
            d = pvm[key]
            ag = abs(d["predicted_geomean"] - d["measured_geo_distortion"]) / d["predicted_geomean"] * 100
            print(f"| {label} | {fmt(d['predicted_geomean'])} | {fmt(d['measured_geo_distortion'])} | within {ag:.1f}% |")
        print()

    # --- §10.2 per-layer Pearson ----------------------------------------
    if "section_10_2_per_layer_pearson" in r:
        plp = r["section_10_2_per_layer_pearson"]
        print("## §10.2 — Per-layer Pearson (predicted geomean vs measured geo distortion)\n")
        for m, p in plp.items():
            print(f"- {m}: **{fmt(p, 3)}**")
        print()

    # --- §10.3 top-1 + geo distortion -----------------------------------
    if "phase_B_real_quantization" in r:
        pb = r["phase_B_real_quantization"]
        print(f"## §10.3 — Real quantization at b_avg = {pb['b_avg']}, r = {pb['rank']} (n = {pb['n_test_files']} test files)\n")
        print(f"Test files: {', '.join(pb['test_files_used'])}\n")
        # Sort by descending top-1
        methods = sorted(pb["methods"].items(), key=lambda kv: -kv[1]["top1_l0excl_mean"])
        # Predicted geomean lookup
        pred_lookup = {
            "v_truncate": r["section_10_1_geomean_predictions"].get("V_Q", {}).get("product_geomean__l0excl"),
            "v_waterfill": r["section_10_1_geomean_predictions"].get("V_Q", {}).get("product_geomean__l0excl"),
            "cca_orth_uniform": r["section_10_1_geomean_predictions"].get("V_h", {}).get("product_geomean__l0excl"),
            "cca_orth_waterfill": r["section_10_1_geomean_predictions"].get("V_h", {}).get("product_geomean__l0excl"),
            "r_sym_uniform": r["section_10_1_geomean_predictions"].get("R_sym", {}).get("product_geomean__l0excl"),
            "r_sym_waterfill": r["section_10_1_geomean_predictions"].get("R_sym", {}).get("product_geomean__l0excl"),
            "v3": r["section_10_1_geomean_predictions"].get("random_orth_TQ_proxy", {}).get("product_geomean__l0excl"),
        }
        print("| Method | top-1 prefill | geo distortion | predicted geomean |")
        print("|---|---:|---:|---:|")
        for m, d in methods:
            pg = pred_lookup.get(m)
            pg_str = f"{fmt(pg, 2)}" if pg is not None else "—"
            print(f"| `{m}` | **{fmt(d['top1_l0excl_mean'])}** | **{fmt(d['geo_l0excl_mean'])}** | {pg_str} |")
        print()

    # --- §10.4 cross-task spread ----------------------------------------
    if "section_10_4_cross_task_predicted_spread" in r:
        ct = r["section_10_4_cross_task_predicted_spread"]
        print("## §10.4 — Cross-task spread of predicted geomean (per LongBench task, test split)\n")
        print("| Method | per-task min | per-task max | spread (relative) |")
        print("|---|---:|---:|---:|")
        for label, key in [
            ("$R_{\\text{sym}}$", "R_sym"),
            ("$V_Q$", "V_Q"),
            ("$V_h$", "V_h"),
            ("$V_K$", "V_K"),
            ("Random orth", "random_orth_TQ_proxy"),
        ]:
            if key not in ct: continue
            d = ct[key]
            print(f"| {label} | {fmt(d['min'])} | {fmt(d['max'])} | {d['spread_relative']*100:.2f}% |")
        print()

    # --- §5 active-set --------------------------------------------------
    if "section_5_active_set_R_sym" in r:
        a = r["section_5_active_set_R_sym"]
        print("## §5 — Active-set sizes for $R_{\\text{sym}}$ water-fill on test moments\n")
        print("```")
        for b_avg, stats in sorted(a.items(), key=lambda kv: int(kv[0])):
            print(f"b_avg={b_avg}: active count over 288 heads — min={stats['min']}, median={stats['median']}, max={stats['max']}")
        print("```\n")

    # --- §8 M-indefinite -------------------------------------------------
    if "section_8_M_indefinite" in r:
        m = r["section_8_M_indefinite"]
        print("## §8 — M = ½(Σ_Q Σ_K + Σ_K Σ_Q) eigenvalue audit (train moments)\n")
        print("```")
        print(f"min eigenvalue across all (head, j):  {m['min_eigenvalue_overall']:.3e}")
        print(f"max eigenvalue:                       {m['max_eigenvalue_overall']:.3e}")
        print(f"# negative eigvals:                   {m['n_negative_eigvals']} / {m['n_total_eigvals']} ({m['frac_negative']*100:.2f}%)")
        print(f"# heads with ≥1 negative eigval:      {m['n_heads_with_negative']} / {m['n_heads_total']}")
        print(f"per-head min/max ratio:               [{m['min_max_ratio_min']:.3f}, {m['min_max_ratio_max']:.3f}]")
        print("```\n")


if __name__ == "__main__":
    main()
