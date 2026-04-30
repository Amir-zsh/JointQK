"""Generate E2 result/takeaway charts for the Stage 1E report.

Reads `metrics_e1_e2.json` (post-F8) and the E3 summaries for sim-vs-reality comparison.

Writes:
    artifacts/stage1/cca_vs_waterfill_study/report_charts/
        e2_headline_at_b3.png            — bar chart of log2(D/D_v3) per method at b_avg=3
        e2_bit_budget_lines.png          — per-method log2(D/D_v3) vs b_avg
        e2_per_layer_heatmap.png         — log2(D/D_v3) heatmap (layer × kv_head) for 4 headline methods
        e2_f8_before_after.png           — pre vs post F8 fix at b_avg=3
        e2_sim_vs_real.png               — sim log2(D/D_v3) vs E3 real log2(geo/geo_v3) at b_avg=3
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "artifacts/stage1/cca_vs_waterfill_study"
OUT = BASE / "report_charts"
OUT.mkdir(parents=True, exist_ok=True)


METHOD_LABELS = {
    "v3": "v3 (random rotation, uniform bits)",
    "v_waterfill": "v_waterfill (V basis, water-fill on λ·σ²(V))",
    "v_truncate_r64": "v_truncate r=64 (V basis, top-r uniform)",
    "v_truncate_r96": "v_truncate r=96",
    "v_truncate_r48": "v_truncate r=48",
    "v_truncate_r32": "v_truncate r=32",
    "cca_waterfill": "cca_waterfill (CCA basis, water-fill on (P_K_inv)^T Σ_Q P_K_inv ·σ²(CCA))",
    "cca_uniform_r64": "cca_uniform r=64 (CCA basis, top-r uniform)",
    "cca_uniform_r96": "cca_uniform r=96",
    "cca_uniform_r48": "cca_uniform r=48",
    "cca_uniform_r32": "cca_uniform r=32",
}

# Color by (basis, allocation):
COLORS = {
    "v3": "#888888",            # gray (baseline)
    "v_waterfill": "#228833",   # green (best class)
    "v_truncate_r64": "#66CC99",
    "v_truncate_r96": "#99DDBB",
    "v_truncate_r48": "#33AA77",
    "v_truncate_r32": "#117755",
    "cca_waterfill": "#4477AA", # blue
    "cca_uniform_r64": "#88AACC",
    "cca_uniform_r96": "#BBCCDD",
    "cca_uniform_r48": "#5588BB",
    "cca_uniform_r32": "#33669A",
}


def load_e2_metrics() -> dict:
    with open(BASE / "metrics_e1_e2.json") as f:
        return json.load(f)


def load_e3_summary(b_avg: float) -> dict | None:
    """Return E3 summary for given b_avg, or None if missing."""
    p = BASE / f"e3/e3_b{int(b_avg)}_r64_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def plot_headline_at_b3(m: dict) -> None:
    """Bar chart of log2(D_method/D_v3) at b_avg=3, sorted descending (best first)."""
    sim = m["simulation"]["b3.0"]
    rows = []
    for method, payload in sim.items():
        if method == "v3":
            continue
        lr = payload.get("log2_D_over_Dv3_l0excl_mean")
        if lr is not None:
            rows.append((method, lr, payload.get("frac_better_than_v3_l0excl", 0.0)))
    rows.sort(key=lambda x: x[1])  # ascending = best first

    fig, ax = plt.subplots(figsize=(11, 6.5))
    methods = [r[0] for r in rows]
    log_ratios = [r[1] for r in rows]
    fracs = [r[2] for r in rows]
    bar_colors = [COLORS.get(m, "#999999") for m in methods]
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, log_ratios, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m for m in methods], fontsize=9)
    ax.invert_yaxis()  # best at top
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="V3 baseline")
    ax.set_xlabel("log₂(D_method / D_v3)  ←  better          worse  →")
    ax.set_title("E2 simulation @ b_avg=3, layer-0-excluded (post-F8)")
    # Annotate fraction better-than-v3
    for i, (lr, frac) in enumerate(zip(log_ratios, fracs)):
        # Place label near bar
        x_text = lr + (0.07 if lr < 0 else -0.07)
        ha = "left" if lr < 0 else "right"
        color = "white" if abs(lr) > 1.0 else "black"
        ax.text(
            lr / 2, i, f"{frac*100:.0f}% pairs < V3",
            ha="center", va="center", fontsize=8, color=color
        )
    ax.set_xlim(-4, 3)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "e2_headline_at_b3.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_bit_budget_lines(m: dict) -> None:
    """Per-method log2(D/D_v3) as a function of b_avg."""
    fig, ax = plt.subplots(figsize=(10, 6))
    headline_methods = ["v_waterfill", "cca_waterfill", "v_truncate_r64", "cca_uniform_r64"]
    for method in headline_methods:
        xs, ys = [], []
        for b_avg in [2.0, 3.0, 4.0]:
            sim = m["simulation"].get(f"b{b_avg}", {})
            payload = sim.get(method)
            if payload and "log2_D_over_Dv3_l0excl_mean" in payload:
                xs.append(b_avg)
                ys.append(payload["log2_D_over_Dv3_l0excl_mean"])
        if xs:
            ax.plot(xs, ys, "o-", color=COLORS.get(method, "#999"), linewidth=2.0,
                    markersize=8, label=method)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="V3 baseline")
    ax.set_xlabel("$b_{avg}$ (bits per coordinate)")
    ax.set_ylabel("log₂(D_method / D_v3)  (l0excl mean)")
    ax.set_title("Bit-budget sensitivity: predicted log-ratio per method")
    ax.set_xticks([2.0, 3.0, 4.0])
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "e2_bit_budget_lines.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_layer_heatmap(m: dict) -> None:
    """log2(D_method/D_v3) heatmap per (layer, kv_head) for 4 headline methods at b_avg=3."""
    sim_b3 = m["simulation"]["b3.0"]
    n_layers = m["n_layers"]
    n_kv_heads = m["n_kv_heads"]
    methods = ["v_waterfill", "cca_waterfill", "v_truncate_r64", "cca_uniform_r64"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)
    D_v3 = np.array(sim_b3["v3"]["D"]).reshape(n_layers, n_kv_heads)
    # Choose symmetric color range from data
    vmaxes = []
    for method in methods:
        if method in sim_b3:
            D_m = np.array(sim_b3[method]["D"]).reshape(n_layers, n_kv_heads)
            log_ratio = np.log2(np.maximum(D_m, 1e-30) / np.maximum(D_v3, 1e-30))
            vmaxes.append(np.percentile(np.abs(log_ratio[1:]), 95))  # exclude layer 0 from scale
    vmax = max(vmaxes) if vmaxes else 5.0
    for ax, method in zip(axes, methods):
        if method not in sim_b3:
            ax.text(0.5, 0.5, f"{method}\nnot available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        D_m = np.array(sim_b3[method]["D"]).reshape(n_layers, n_kv_heads)
        log_ratio = np.log2(np.maximum(D_m, 1e-30) / np.maximum(D_v3, 1e-30))
        im = ax.imshow(log_ratio, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("kv_head")
        if ax is axes[0]:
            ax.set_ylabel("layer")
        median_l0excl = np.median(log_ratio[1:])
        ax.set_title(f"{method}\nl0excl median = {median_l0excl:+.2f} log₂")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-(layer, kv_head) log₂(D_method / D_v3) at b_avg=3 (blue = beats V3, red = loses)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e2_per_layer_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_f8_before_after(m: dict) -> None:
    """Compare buggy (pre-F8) vs corrected (post-F8) simulation at b_avg=3.

    Pre-F8 numbers are hardcoded from the original buggy simulation output (recorded
    in the conversation history and fixes_to_apply.md).
    """
    sim_b3 = m["simulation"]["b3.0"]
    methods = ["v_waterfill", "cca_waterfill", "cca_uniform_r96", "cca_uniform_r64",
               "cca_uniform_r48", "cca_uniform_r32", "v_truncate_r96", "v_truncate_r64",
               "v_truncate_r48", "v_truncate_r32"]

    # Pre-F8 numbers (from original buggy simulation):
    pre_f8 = {
        "v_waterfill": -3.449,
        "cca_waterfill": -8.400,
        "cca_uniform_r96": -7.024,
        "cca_uniform_r64": -4.560,
        "cca_uniform_r48": -3.429,
        "cca_uniform_r32": -2.437,
        "v_truncate_r96": +0.061,
        "v_truncate_r64": +0.647,
        "v_truncate_r48": +1.167,
        "v_truncate_r32": +1.799,
    }
    post_f8 = {}
    for method in methods:
        payload = sim_b3.get(method, {})
        post_f8[method] = payload.get("log2_D_over_Dv3_l0excl_mean", float("nan"))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y_pos = np.arange(len(methods))
    width = 0.4
    pre_vals = [pre_f8[m] for m in methods]
    post_vals = [post_f8[m] for m in methods]
    bars1 = ax.barh(y_pos - width/2, pre_vals, width, color="#CC6677", alpha=0.85,
                    label="pre-F8 (buggy `weights = ρ²`)", edgecolor="white")
    bars2 = ax.barh(y_pos + width/2, post_vals, width, color="#4477AA", alpha=0.85,
                    label="post-F8 (trace formula)", edgecolor="white")
    # Mark large shifts
    for i, m_name in enumerate(methods):
        shift = post_f8[m_name] - pre_f8[m_name]
        if abs(shift) > 1.0:
            x = max(pre_f8[m_name], post_f8[m_name]) + 0.3
            ax.annotate(f"Δ={shift:+.1f}", xy=(x, i), fontsize=8, color="black",
                        va="center", ha="left", fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="V3 baseline")
    ax.set_xlabel("log₂(D_method / D_v3)  (l0excl mean)")
    ax.set_title("F8 fix: per-method shift in simulation prediction at b_avg=3")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "e2_f8_before_after.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_sim_vs_real(m: dict) -> None:
    """Scatter sim log2(D_method/D_v3) vs E3 real log2(geo/geo_v3) at b_avg=3."""
    e3 = load_e3_summary(3.0)
    if e3 is None:
        print("  (skipping e2_sim_vs_real.png — no E3 summary at b_avg=3)")
        return
    sim_b3 = m["simulation"]["b3.0"]
    e3_agg = e3.get("aggregated", {})
    geo_v3 = e3_agg.get("v3", {}).get("geometry_distortion", {}).get("l0excl_mean")
    if geo_v3 is None:
        print("  (skipping e2_sim_vs_real.png — V3 geo_dist missing)")
        return
    methods = ["v_waterfill", "cca_waterfill", "v_truncate", "cca_uniform"]
    rows = []
    for method in methods:
        if method not in e3_agg:
            continue
        geo_m = e3_agg[method].get("geometry_distortion", {}).get("l0excl_mean")
        if geo_m is None or geo_m <= 0:
            continue
        real_lr = float(np.log2(geo_m / geo_v3))
        # Match to sim. cca_uniform → use cca_uniform_r64; v_truncate → v_truncate_r64.
        sim_key = method
        if method == "v_truncate":
            sim_key = "v_truncate_r64"
        if method == "cca_uniform":
            sim_key = "cca_uniform_r64"
        sim_payload = sim_b3.get(sim_key, {})
        sim_lr = sim_payload.get("log2_D_over_Dv3_l0excl_mean")
        if sim_lr is None:
            continue
        rows.append((method, sim_lr, real_lr))

    fig, ax = plt.subplots(figsize=(8, 7))
    for method, sim_lr, real_lr in rows:
        c = COLORS.get(method, COLORS.get(method + "_r64", "#666"))
        ax.scatter(sim_lr, real_lr, s=200, c=c, edgecolors="black", linewidth=1.5,
                   label=method, zorder=10)
        # label slightly to the right
        label = method
        ax.annotate(label, xy=(sim_lr, real_lr), xytext=(sim_lr + 0.15, real_lr - 0.05),
                    fontsize=10, ha="left", va="top")
    # Perfect-agreement diagonal
    lims = [min(min(r[1] for r in rows), min(r[2] for r in rows)) - 0.5,
            max(max(r[1] for r in rows), max(r[2] for r in rows)) + 0.5]
    ax.plot(lims, lims, "k--", linewidth=1.0, alpha=0.5, label="sim = reality")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("E2 simulation log₂(D / D_v3)  (l0excl)")
    ax.set_ylabel("E3 real-quantization log₂(geo / geo_v3)  (l0excl)")
    ax.set_title("Sim vs real E3 at b_avg=3, post-F8/post-F11")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "e2_sim_vs_real.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    m = load_e2_metrics()
    print(f"Loaded metrics_e1_e2.json (n_layers={m['n_layers']}, n_kv_heads={m['n_kv_heads']}, head_dim={m['head_dim']})")
    print(f"Output dir: {OUT}")
    plot_headline_at_b3(m)
    print("  wrote e2_headline_at_b3.png")
    plot_bit_budget_lines(m)
    print("  wrote e2_bit_budget_lines.png")
    plot_per_layer_heatmap(m)
    print("  wrote e2_per_layer_heatmap.png")
    plot_f8_before_after(m)
    print("  wrote e2_f8_before_after.png")
    plot_sim_vs_real(m)
    print("  wrote e2_sim_vs_real.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
