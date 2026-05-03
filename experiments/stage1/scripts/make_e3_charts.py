"""Generate E3 real-quantization charts for the Stage 1E review.

Reads the persisted E3 summaries/rows and corrected E2 simulation metrics.

Writes:
    artifacts/stage1/cca_vs_waterfill_study/report_charts/
        e3_top1_b3.png
        e3_bit_budget_sensitivity.png
        e3_sim_vs_real_geo.png
        e3_top1_heatmap_b3.png
        e3_smoke_test.png
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "artifacts/stage1/cca_vs_waterfill_study"
E3_DIR = BASE / "e3"
OUT = BASE / "report_charts"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "v_waterfill", "v3", "v_truncate",
    "cca_waterfill", "cca_uniform",
    "cca_orth_waterfill", "cca_orth_uniform",
    "r_sym_waterfill", "r_sym_uniform",
]
METHOD_LABELS = {
    "v3": "V3",
    "v_waterfill": "V + water-fill",
    "v_truncate": "V truncate r=64",
    "cca_waterfill": "CCA + water-fill",
    "cca_uniform": "CCA uniform r=64",
    "cca_orth_waterfill": "CCA (V_h orth) + water-fill",
    "cca_orth_uniform": "CCA (V_h orth) uniform r=64",
    "r_sym_waterfill": "R_sym + water-fill",
    "r_sym_uniform": "R_sym uniform r=64",
}
COLORS = {
    "v3": "#777777",
    "v_waterfill": "#228833",
    "v_truncate": "#66CC99",
    "cca_waterfill": "#4477AA",
    "cca_uniform": "#88AACC",
    "cca_orth_waterfill": "#0066AA",
    "cca_orth_uniform": "#3388BB",
    "r_sym_waterfill": "#AA3366",
    "r_sym_uniform": "#CC6688",
}
SIM_METHOD = {
    "v3": "v3",
    "v_waterfill": "v_waterfill",
    "v_truncate": "v_truncate_r64",
    "cca_waterfill": "cca_waterfill",
    "cca_uniform": "cca_uniform_r64",
}


def load_summary(b_avg: int) -> dict:
    with open(E3_DIR / f"e3_b{b_avg}_r64_summary.json") as f:
        return json.load(f)


def load_rows(b_avg: int) -> list[dict]:
    return torch.load(
        E3_DIR / f"e3_b{b_avg}_r64_rows.pt",
        map_location="cpu",
        weights_only=False,
    )["rows"]


def l0excl(summary: dict, method: str, metric: str) -> float:
    return float(summary["aggregated"][method][metric]["l0excl_mean"])


def plot_top1_b3() -> None:
    summary = load_summary(3)
    rows = []
    for method in METHODS:
        agg = summary["aggregated"][method]
        ci = agg.get("bootstrap_ci", {})
        rows.append(
            {
                "method": method,
                "top1": l0excl(summary, method, "top1_prefill"),
                "top5": l0excl(summary, method, "top5_prefill"),
                "geo": l0excl(summary, method, "geometry_distortion"),
                "ci_lo": float(ci.get("lo95", float("nan"))),
                "ci_hi": float(ci.get("hi95", float("nan"))),
            }
        )
    rows.sort(key=lambda r: r["top1"], reverse=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(rows))
    vals = np.array([r["top1"] for r in rows])
    colors = [COLORS[r["method"]] for r in rows]
    bars = ax.barh(y, vals, color=colors, alpha=0.9, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[r["method"]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.9)
    ax.set_xlabel("Prefill top-1 retention, layer-0 excluded")
    ax.set_title("E3 real quantization @ b_avg=3: attention top-1 retention")
    ax.grid(axis="x", alpha=0.25)
    for bar, r in zip(bars, rows):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{r['top1']:.3f}",
            va="center",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(OUT / "e3_top1_b3.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_bit_budget_sensitivity() -> None:
    summaries = {b: load_summary(b) for b in [2, 3, 4]}
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))

    for method in METHODS:
        xs = [2, 3, 4]
        top1 = [l0excl(summaries[b], method, "top1_prefill") for b in xs]
        geo = [l0excl(summaries[b], method, "geometry_distortion") for b in xs]
        axes[0].plot(xs, top1, "o-", color=COLORS[method], linewidth=2.2, label=METHOD_LABELS[method])
        axes[1].plot(xs, geo, "o-", color=COLORS[method], linewidth=2.2, label=METHOD_LABELS[method])

    axes[0].set_title("Top-1 retention")
    axes[0].set_ylabel("Prefill top-1, layer-0 excluded")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_title("Geometry distortion")
    axes[1].set_ylabel("Q-weighted geometry distortion, layer-0 excluded")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("b_avg (bits per coordinate)")
        ax.set_xticks([2, 3, 4])
        ax.grid(alpha=0.28)
    # Single shared legend below both panels so it doesn't cover any data lines.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("E3 bit-budget sensitivity")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(OUT / "e3_bit_budget_sensitivity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_sim_vs_real_geo() -> None:
    with open(BASE / "metrics_e1_e2.json") as f:
        e2 = json.load(f)
    e3 = load_summary(3)
    sim_b3 = e2["simulation"]["b3.0"]
    real_geo_v3 = l0excl(e3, "v3", "geometry_distortion")
    rows = []
    for method in METHODS:
        if method not in SIM_METHOD:
            # New methods (cca_orth_*, r_sym_*) lack closed-form E2 simulation rows.
            continue
        sim_name = SIM_METHOD[method]
        sim_payload = sim_b3[sim_name]
        if method == "v3":
            sim_lr = 0.0
            real_lr = 0.0
        else:
            sim_lr = float(sim_payload["log2_D_over_Dv3_l0excl_mean"])
            real_geo = l0excl(e3, method, "geometry_distortion")
            real_lr = math.log2(real_geo / real_geo_v3)
        rows.append((method, sim_lr, real_lr))

    xs = np.array([r[1] for r in rows])
    ys = np.array([r[2] for r in rows])
    lim_min = min(xs.min(), ys.min()) - 0.35
    lim_max = max(xs.max(), ys.max()) + 0.35

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="black", linewidth=1.0, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.35)
    for method, x, y in rows:
        edge = "white"
        linewidth = 1.0
        ax.scatter(x, y, s=120, color=COLORS[method], edgecolor=edge, linewidth=linewidth, zorder=3)
        label = METHOD_LABELS[method]
        ax.annotate(label, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=8.5)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("E2 simulation log2(D / D_v3), b_avg=3")
    ax.set_ylabel("E3 real log2(geo / geo_v3), b_avg=3")
    ax.set_title("Simulation vs real quantization: geometry distortion (post-F11 E3)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "e3_sim_vs_real_geo.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_top1_heatmap_b3() -> None:
    rows = load_rows(3)
    n_layers = 36
    n_heads = 8
    fig, axes = plt.subplots(1, len(METHODS), figsize=(18, 5.6), sharey=True)
    for ax, method in zip(axes, METHODS):
        mat = np.full((n_layers, n_heads), np.nan)
        counts = np.zeros((n_layers, n_heads), dtype=np.int64)
        for row in rows:
            if row["method"] != method:
                continue
            layer = int(row["layer"])
            head = int(row["kv_head"])
            if np.isnan(mat[layer, head]):
                mat[layer, head] = 0.0
            mat[layer, head] += float(row["top1_prefill"])
            counts[layer, head] += 1
        mat = mat / np.maximum(counts, 1)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{METHOD_LABELS[method]}\nmean={np.nanmean(mat[1:]):.3f}")
        ax.set_xlabel("kv_head")
        if ax is axes[0]:
            ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("E3 top-1 retention by (layer, kv_head), b_avg=3", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "e3_top1_heatmap_b3.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_smoke_test() -> None:
    with open(E3_DIR / "e3_b3_r64_smoke_b16.json") as f:
        smoke = json.load(f)
    # Only plot methods present in the smoke file (new methods may be absent until merge runs).
    methods = [m for m in METHODS if m in smoke]
    labels = [METHOD_LABELS[m] for m in methods]
    geo = [smoke[m]["geometry_distortion"] for m in methods]
    top1 = [smoke[m]["top1_prefill"] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(methods))
    axes[0].bar(x, geo, color=colors, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Geometry distortion, log scale")
    axes[0].set_title("Quasi-full-precision distortion (b=8)")
    axes[1].bar(x, top1, color=colors, edgecolor="white")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].set_ylabel("Prefill top-1")
    axes[1].set_title("Quasi-full-precision top-1 (b=8)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("E3 full-precision smoke test on first example/layer/head")
    fig.tight_layout()
    fig.savefig(OUT / "e3_smoke_test.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    plot_top1_b3()
    plot_bit_budget_sensitivity()
    plot_sim_vs_real_geo()
    plot_top1_heatmap_b3()
    plot_smoke_test()
    print(f"Wrote E3 charts to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
