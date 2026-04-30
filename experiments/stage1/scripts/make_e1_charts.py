"""Generate E1 result/takeaway charts for the Stage 1E report.

Reads cca_stats.pt + metrics_e1_e2.json, writes:
    artifacts/stage1/cca_vs_waterfill_study/report_charts/
        e1_layer0_anomaly.png
        e1_r95_distribution.png
        e1_per_layer_r95.png
        e1_cumulative_energy.png
        e1_top_rho_heatmap.png
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


def load_data() -> dict:
    cca = torch.load(BASE / "cca_stats.pt", map_location="cpu", weights_only=False)
    with open(BASE / "metrics_e1_e2.json") as f:
        m = json.load(f)
    rho = cca["rho"]  # (n_layers, n_kv_heads, d)
    return {"rho": rho.numpy(), "metrics": m, "n_layers": int(cca["n_layers"]),
            "n_kv_heads": int(cca["n_kv_heads"]), "head_dim": int(cca["head_dim"])}


def cumulative_energy(rho_arr: np.ndarray) -> np.ndarray:
    """rho_arr shape (..., d) → cumulative-energy curves of same shape."""
    rho_sq = rho_arr ** 2
    cum = np.cumsum(rho_sq, axis=-1)
    total = cum[..., -1:].clip(min=1e-30)
    return cum / total


def r_at_threshold(cum_energy: np.ndarray, threshold: float) -> np.ndarray:
    """For each row, smallest r (1-indexed) such that cum_energy[..., r-1] >= threshold."""
    above = cum_energy >= threshold
    # argmax returns first True; +1 to convert from 0-index to "rank used"
    return above.argmax(axis=-1) + 1


def plot_layer0_anomaly(d: dict) -> None:
    rho = d["rho"]
    n_layers, n_heads, dim = rho.shape
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: median spectrum, layer 0 vs layers 1..n-1
    layer0_rho = rho[0].reshape(-1, dim)
    rest_rho = rho[1:].reshape(-1, dim)
    x = np.arange(dim)
    axes[0].fill_between(x, np.percentile(rest_rho, 10, axis=0),
                         np.percentile(rest_rho, 90, axis=0),
                         color="#4477AA", alpha=0.25, label="layers 1–35: 10–90% range")
    axes[0].plot(x, np.median(rest_rho, axis=0), color="#4477AA", linewidth=2.0,
                 label=f"layers 1–{n_layers-1}: median (n={rest_rho.shape[0]})")
    axes[0].fill_between(x, np.percentile(layer0_rho, 10, axis=0),
                         np.percentile(layer0_rho, 90, axis=0),
                         color="#CC6677", alpha=0.25, label="layer 0: 10–90% range")
    axes[0].plot(x, np.median(layer0_rho, axis=0), color="#CC6677", linewidth=2.0,
                 label=f"layer 0: median (n={layer0_rho.shape[0]})")
    axes[0].set_xlabel("rank index $i$")
    axes[0].set_ylabel("canonical correlation $\\rho_i$")
    axes[0].set_title("Layer 0 has a steeper, sink-dominated spectrum")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: cumulative energy, same split
    cum_layer0 = cumulative_energy(layer0_rho)
    cum_rest = cumulative_energy(rest_rho)
    axes[1].fill_between(x, np.percentile(cum_rest, 10, axis=0),
                         np.percentile(cum_rest, 90, axis=0),
                         color="#4477AA", alpha=0.25)
    axes[1].plot(x, np.median(cum_rest, axis=0), color="#4477AA", linewidth=2.0,
                 label=f"layers 1–{n_layers-1}: median")
    axes[1].fill_between(x, np.percentile(cum_layer0, 10, axis=0),
                         np.percentile(cum_layer0, 90, axis=0),
                         color="#CC6677", alpha=0.25)
    axes[1].plot(x, np.median(cum_layer0, axis=0), color="#CC6677", linewidth=2.0,
                 label="layer 0: median")
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                    label="95% energy threshold")
    axes[1].set_xlabel("rank cutoff $r$")
    axes[1].set_ylabel("cumulative canonical-correlation energy")
    axes[1].set_title("Layer 0 needs fewer directions to capture 95% of energy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "e1_layer0_anomaly.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_r95_distribution(d: dict) -> None:
    rho = d["rho"]
    n_layers = d["n_layers"]
    cum = cumulative_energy(rho.reshape(-1, rho.shape[-1]))
    r95 = r_at_threshold(cum, 0.95)
    r95 = r95.reshape(n_layers, -1)
    layer0_r95 = r95[0]
    rest_r95 = r95[1:].flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.arange(0, d["head_dim"] + 4, 4)

    # Panel 1: raw counts
    axes[0].hist(rest_r95, bins=bins, color="#4477AA", alpha=0.75,
                 label=f"layers 1–{n_layers-1} (n={rest_r95.size})", edgecolor="white")
    axes[0].hist(layer0_r95, bins=bins, color="#CC6677", alpha=0.85,
                 label=f"layer 0 (n={layer0_r95.size})", edgecolor="white")
    axes[0].axvline(np.median(rest_r95), color="#4477AA", linestyle="--", linewidth=2.0,
                    label=f"median (layers 1–{n_layers-1}) = {int(np.median(rest_r95))}")
    axes[0].axvline(np.median(layer0_r95), color="#CC6677", linestyle="--", linewidth=2.0,
                    label=f"median (layer 0) = {int(np.median(layer0_r95))}")
    axes[0].set_xlabel("$r_{95}$")
    axes[0].set_ylabel("count of (layer, kv_head) pairs")
    axes[0].set_title("Raw counts (layer 0 has fewer pairs by design)")
    axes[0].set_xlim(0, d["head_dim"] + 2)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)

    # Panel 2: density-normalized — fair shape comparison since each group integrates to 1
    axes[1].hist(rest_r95, bins=bins, density=True, color="#4477AA", alpha=0.75,
                 label=f"layers 1–{n_layers-1}", edgecolor="white")
    axes[1].hist(layer0_r95, bins=bins, density=True, color="#CC6677", alpha=0.85,
                 label="layer 0", edgecolor="white")
    axes[1].axvline(np.median(rest_r95), color="#4477AA", linestyle="--", linewidth=2.0)
    axes[1].axvline(np.median(layer0_r95), color="#CC6677", linestyle="--", linewidth=2.0)
    axes[1].set_xlabel("$r_{95}$")
    axes[1].set_ylabel("density (each group integrates to 1)")
    axes[1].set_title("Density-normalized (fair shape comparison)")
    axes[1].set_xlim(0, d["head_dim"] + 2)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Distribution of $r_{{95}}$ across all {rho.size // d['head_dim']} (layer, kv_head) pairs",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "e1_r95_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_layer_r95(d: dict) -> None:
    rho = d["rho"]
    n_layers, n_heads, dim = rho.shape
    cum = cumulative_energy(rho.reshape(-1, dim))
    r95_flat = r_at_threshold(cum, 0.95)
    r95 = r95_flat.reshape(n_layers, n_heads)
    median = np.median(r95, axis=-1)
    p10 = np.percentile(r95, 10, axis=-1)
    p90 = np.percentile(r95, 90, axis=-1)

    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(n_layers)
    ax.fill_between(layers, p10, p90, color="#4477AA", alpha=0.3, label="10–90% across kv heads")
    ax.plot(layers, median, color="#4477AA", linewidth=2.2, marker="o", markersize=4,
            label="median across kv heads")
    ax.axhline(64, color="#228833", linestyle=":", linewidth=1.5, label="r = 64 (E3 default cutoff)")
    ax.axhline(np.median(median[1:]), color="black", linestyle="--", linewidth=1.0,
               label=f"layers 1–{n_layers-1} median of medians = {int(np.median(median[1:]))}")
    ax.set_xlabel("layer index")
    ax.set_ylabel("$r_{95}$")
    ax.set_title("Per-layer $r_{95}$: layer 0 is the low-rank outlier (steeper decay)")
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(0, dim + 4)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "e1_per_layer_r95.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_energy_overall(d: dict) -> None:
    rho = d["rho"]
    n_layers, n_heads, dim = rho.shape
    cum = cumulative_energy(rho.reshape(-1, dim))
    cum_rest = cumulative_energy(rho[1:].reshape(-1, dim))
    x = np.arange(dim)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Light gray: each individual head
    for row in cum_rest:
        ax.plot(x, row, color="lightgray", alpha=0.15, linewidth=0.5)
    ax.plot(x, np.median(cum_rest, axis=0), color="#4477AA", linewidth=2.5,
            label=f"median across (layers 1–{n_layers-1}, kv_heads), n={cum_rest.shape[0]}")
    # Annotate energy thresholds at common ranks
    for r_target in [16, 32, 64, 96]:
        med = np.median(cum_rest[:, r_target - 1]) * 100
        ax.scatter([r_target], [med / 100], s=60, color="#CC6677", zorder=5)
        ax.annotate(f"r={r_target}: {med:.1f}%",
                    xy=(r_target, med / 100), xytext=(r_target + 3, med / 100 - 0.05),
                    fontsize=9, color="#CC6677",
                    arrowprops=dict(arrowstyle="->", color="#CC6677", alpha=0.5))
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
               label="95% energy")
    ax.set_xlabel("rank cutoff $r$")
    ax.set_ylabel("cumulative canonical-correlation energy")
    ax.set_title("Energy captured vs. rank — moderate decay (no sharp cliff)")
    ax.set_xlim(0, dim)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "e1_cumulative_energy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_top_rho_heatmap(d: dict) -> None:
    rho = d["rho"]  # (n_layers, n_kv_heads, dim)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, rank, title in zip(
        axes,
        [0, 8, 32],
        ["$\\rho_1$ (top canonical)", "$\\rho_9$ (rank 9)", "$\\rho_{33}$ (rank 33)"],
    ):
        im = ax.imshow(rho[..., rank], aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=1.0)
        ax.set_xlabel("kv_head")
        ax.set_ylabel("layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Canonical correlation at selected ranks", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e1_top_rho_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    d = load_data()
    print(f"Loaded ρ tensor: shape {d['rho'].shape}")
    print(f"Output dir: {OUT}")
    plot_layer0_anomaly(d)
    print("  wrote e1_layer0_anomaly.png")
    plot_r95_distribution(d)
    print("  wrote e1_r95_distribution.png")
    plot_per_layer_r95(d)
    print("  wrote e1_per_layer_r95.png")
    plot_cumulative_energy_overall(d)
    print("  wrote e1_cumulative_energy.png")
    plot_top_rho_heatmap(d)
    print("  wrote e1_top_rho_heatmap.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
