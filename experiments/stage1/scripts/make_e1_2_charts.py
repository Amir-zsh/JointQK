"""Generate E1-2 result/takeaway charts for the Stage 1E report.

Reads `distribution_diagnostics/distribution_stats.pt` and `metrics_e1_2.json`, writes
into `report_charts/`. Chart titles / annotations are populated from numeric stats so a
later "I assumed direction X but data says Y" inversion (like in the E1 charts) can't happen.

Reuse with: python -m experiments.stage1.scripts.make_e1_2_charts
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
DIST_DIR = BASE / "distribution_diagnostics"
OUT = BASE / "report_charts"
OUT.mkdir(parents=True, exist_ok=True)


COLOR_PER_TASK = {
    "qasper_e": "#4477AA",
    "hotpotqa_e": "#CC6677",
    "passage_retrieval_en_e": "#117733",
}
TASK_LABELS = {
    "qasper_e": "qasper",
    "hotpotqa_e": "hotpotqa",
    "passage_retrieval_en_e": "passage_retrieval_en",
}


def _load() -> dict:
    stats = torch.load(DIST_DIR / "distribution_stats.pt", map_location="cpu", weights_only=False)
    with open(DIST_DIR / "metrics_e1_2.json") as f:
        m = json.load(f)
    return {"stats": stats, "metrics": m}


def _per_metric_eigvals(stats: dict, metric: str, cfg: str) -> np.ndarray:
    """Returns (n_layers, n_kv_heads, d) eigvalues, descending, normalized by λ_1 (per-head)."""
    key = f"{metric}/{cfg}"
    mt = stats["metric_tensors"][key]
    evs = mt["eigvals"].numpy()  # (n_layers, n_kv_heads, d)
    top = evs[..., 0:1].clip(min=1e-30)
    return evs / top


def _per_metric_cumulative(stats: dict, metric: str, cfg: str) -> np.ndarray:
    key = f"{metric}/{cfg}"
    mt = stats["metric_tensors"][key]
    evs = mt["eigvals"].numpy()
    cum = np.cumsum(evs, axis=-1)
    return cum / cum[..., -1:].clip(min=1e-30)


def _per_metric_r95(stats: dict, metric: str, cfg: str) -> np.ndarray:
    key = f"{metric}/{cfg}"
    return stats["metric_tensors"][key]["r95"].numpy()


def plot_marginal_cumulative_energy(data: dict) -> None:
    """One row per metric (Q_prefill, Q_decode, K), per-task curves with reference annotations."""
    stats = data["stats"]
    metrics = ["Q_prefill", "Q_decode", "K"]
    configs = stats["configs"]
    head_dim = stats["head_dim"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    target_ranks = [16, 32, 64, 96]
    for ax, metric in zip(axes, metrics):
        for cfg in configs:
            cum = _per_metric_cumulative(stats, metric, cfg)
            cum_l0excl = cum[1:].reshape(-1, head_dim)
            x = np.arange(head_dim)
            color = COLOR_PER_TASK[cfg]
            label = f"{TASK_LABELS[cfg]} (n_tok={stats['n_tokens'][(cfg, metric.split('_')[-1].lower()) if metric != 'K' else (cfg, 'prefill')]})"
            ax.fill_between(
                x, np.percentile(cum_l0excl, 10, axis=0), np.percentile(cum_l0excl, 90, axis=0),
                color=color, alpha=0.18,
            )
            ax.plot(x, np.median(cum_l0excl, axis=0), color=color, linewidth=2.0,
                    linestyle=("--" if cfg in {"hotpotqa_e"} and metric == "Q_decode" else "-"),
                    label=label)
        ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        for r in target_ranks:
            for cfg in configs:
                cum = _per_metric_cumulative(stats, metric, cfg)
                med = np.median(cum[1:].reshape(-1, head_dim)[:, r - 1])
                ax.scatter([r], [med], color=COLOR_PER_TASK[cfg], s=20, zorder=5, alpha=0.7)
        ax.set_xlabel("rank cutoff $r$")
        ax.set_title(f"{metric}: cumulative eigenvalue energy")
        ax.set_xlim(0, head_dim)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("cumulative energy (layer-0-excluded)")
    fig.suptitle("Marginal eigenvalue spectrum of Q (prefill / decode) and K, per task", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e12_marginal_cumulative_energy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_r95_per_task_per_metric(data: dict) -> None:
    """Density-normalized r95 distribution per metric × task (so unequal sample sizes are fair)."""
    stats = data["stats"]
    metrics = ["Q_prefill", "Q_decode", "K"]
    configs = stats["configs"]
    head_dim = stats["head_dim"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    bins = np.arange(0, head_dim + 4, 4)
    for ax, metric in zip(axes, metrics):
        for cfg in configs:
            r95 = _per_metric_r95(stats, metric, cfg)
            r95_l0excl = r95[1:].flatten()
            color = COLOR_PER_TASK[cfg]
            ax.hist(r95_l0excl, bins=bins, density=True, color=color, alpha=0.45,
                    label=f"{TASK_LABELS[cfg]} (median={int(np.median(r95_l0excl))})", edgecolor="white")
            ax.axvline(np.median(r95_l0excl), color=color, linestyle="--", linewidth=1.5)
        ax.set_xlabel("$r_{95}$")
        ax.set_title(f"{metric}: $r_{{95}}$ distribution (l0excl, density-normalized)")
        ax.set_xlim(0, head_dim + 2)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Marginal $r_{95}$ across (layer 1+, kv_head) per task — wider bars = lower-rank distribution",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e12_marginal_r95_per_task.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_layer_r95(data: dict) -> None:
    """Per-layer median r95 with task colors; separate panel per metric."""
    stats = data["stats"]
    metrics = ["Q_prefill", "Q_decode", "K"]
    configs = stats["configs"]
    n_layers = stats["n_layers"]
    head_dim = stats["head_dim"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, metric in zip(axes, metrics):
        for cfg in configs:
            r95 = _per_metric_r95(stats, metric, cfg)
            median = np.median(r95, axis=-1)
            p10 = np.percentile(r95, 10, axis=-1)
            p90 = np.percentile(r95, 90, axis=-1)
            color = COLOR_PER_TASK[cfg]
            ax.fill_between(np.arange(n_layers), p10, p90, color=color, alpha=0.18)
            ax.plot(np.arange(n_layers), median, color=color, linewidth=1.8, marker="o", markersize=3,
                    label=TASK_LABELS[cfg])
        ax.axhline(64, color="#228833", linestyle=":", linewidth=1.0, alpha=0.7, label="r=64")
        ax.set_xlabel("layer index")
        ax.set_title(f"{metric}: per-layer $r_{{95}}$")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(0, head_dim + 2)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("$r_{95}$")
    fig.tight_layout()
    fig.savefig(OUT / "e12_per_layer_r95.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_layer0_anomaly(data: dict) -> None:
    """Layer-0 vs layers-1+ comparison per metric, pooled across tasks."""
    stats = data["stats"]
    metrics = ["Q_prefill", "Q_decode", "K"]
    head_dim = stats["head_dim"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, metric in zip(axes, metrics):
        # Pool across tasks (for K we only have one phase; for Q_decode it's noisy but still informative).
        layer0 = []
        rest = []
        for cfg in stats["configs"]:
            cum = _per_metric_cumulative(stats, metric, cfg)  # (n_layers, n_kv_heads, d)
            layer0.append(cum[0].reshape(-1, head_dim))
            rest.append(cum[1:].reshape(-1, head_dim))
        layer0 = np.concatenate(layer0, axis=0)
        rest = np.concatenate(rest, axis=0)
        x = np.arange(head_dim)
        ax.fill_between(x, np.percentile(rest, 10, axis=0), np.percentile(rest, 90, axis=0),
                        color="#4477AA", alpha=0.25)
        ax.plot(x, np.median(rest, axis=0), color="#4477AA", linewidth=2.0,
                label=f"layers 1+ (n={rest.shape[0]})")
        ax.fill_between(x, np.percentile(layer0, 10, axis=0), np.percentile(layer0, 90, axis=0),
                        color="#CC6677", alpha=0.25)
        ax.plot(x, np.median(layer0, axis=0), color="#CC6677", linewidth=2.0,
                label=f"layer 0 (n={layer0.shape[0]})")
        ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("rank cutoff $r$")
        ax.set_title(f"{metric}: cumulative-energy comparison")
        ax.set_xlim(0, head_dim)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("cumulative energy")
    fig.suptitle("Layer 0 vs layers 1+: marginal-rank comparison per metric (pooled across tasks)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e12_layer0_anomaly_qk.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_phase_distance_heatmap(data: dict) -> None:
    """Per-task heatmap (layer × kv_head) of prefill-vs-decode Bures distance."""
    stats = data["stats"]
    configs = stats["configs"]
    low_conf = set(stats["low_confidence_tasks"])
    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5.5))
    if len(configs) == 1:
        axes = [axes]
    for ax, cfg in zip(axes, configs):
        d = stats["phase_distances"][cfg].numpy()
        flagged = f"{cfg}/decode" in low_conf
        im = ax.imshow(d, aspect="auto", cmap="viridis")
        ax.set_xlabel("kv_head")
        ax.set_ylabel("layer")
        med = float(np.median(d[1:]))
        title = f"{TASK_LABELS[cfg]} — l0excl median d_Bures = {med:.2f}"
        if flagged:
            title += "  ⚠ low-confidence"
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Within-task phase Bures distance: $d(\\Sigma_Q^{\\rm prefill}, \\Sigma_Q^{\\rm decode})$ per (layer, kv_head)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e12_phase_distance_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_task_distance_heatmap(data: dict) -> None:
    """Average pairwise cross-task distance per (layer, kv_head), one panel per metric."""
    stats = data["stats"]
    metrics = ["Q_prefill", "Q_decode", "K"]
    n_layers = stats["n_layers"]
    n_kv_heads = stats["n_kv_heads"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5.5))
    for ax, metric in zip(axes, metrics):
        # Average over all (cfg_a, cfg_b) pairs for this metric.
        accumulator = torch.zeros(n_layers, n_kv_heads)
        count = 0
        for k, d in stats["cross_task_distances"].items():
            if k.startswith(f"{metric}/"):
                accumulator = accumulator + d
                count += 1
        if count > 0:
            avg = (accumulator / count).numpy()
        else:
            avg = np.zeros((n_layers, n_kv_heads))
        im = ax.imshow(avg, aspect="auto", cmap="viridis")
        ax.set_xlabel("kv_head")
        ax.set_ylabel("layer")
        med = float(np.median(avg[1:]))
        ax.set_title(f"{metric}: l0excl median = {med:.2f}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Cross-task Bures distance (averaged over all task pairs)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e12_task_distance_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_subspace_overlap(data: dict) -> None:
    """Top-r subspace overlap as r grows; one curve per (split, metric)."""
    stats = data["stats"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    metrics = ["Q_prefill", "Q_decode", "K"]
    rs = sorted({int(k.rsplit("/r", 1)[-1]) for k in stats["subspace_overlaps"].keys()})
    for ax, metric in zip(axes, metrics):
        # Cross-task pairs.
        pair_keys = [k for k in stats["subspace_overlaps"].keys() if k.startswith(f"{metric}/")]
        per_r_values = {r: [] for r in rs}
        for k in pair_keys:
            r = int(k.rsplit("/r", 1)[-1])
            ov = stats["subspace_overlaps"][k]
            ov_l0excl = ov[1:].numpy().flatten()
            per_r_values[r].extend(ov_l0excl.tolist())
        median_curve = [float(np.median(per_r_values[r])) for r in rs]
        p10_curve = [float(np.percentile(per_r_values[r], 10)) for r in rs]
        p90_curve = [float(np.percentile(per_r_values[r], 90)) for r in rs]
        ax.fill_between(rs, p10_curve, p90_curve, color="#4477AA", alpha=0.2, label="cross-task 10-90%")
        ax.plot(rs, median_curve, color="#4477AA", linewidth=2.2, marker="o",
                label="cross-task median")
        # Phase pairs (only for Q metrics).
        if metric in {"Q_prefill", "Q_decode"}:
            phase_pairs = [k for k in stats["phase_subspace_overlaps"].keys()]
            phase_per_r = {r: [] for r in rs}
            for cfg in stats["configs"]:
                for r in rs:
                    ov = stats["phase_subspace_overlaps"][(cfg, r)] if (cfg, r) in stats["phase_subspace_overlaps"] else None
                    if ov is None:
                        # Stored as flat key after _cpu_tree
                        ov = stats["phase_subspace_overlaps"].get(f"{cfg}/r{r}")
                    if ov is None:
                        continue
                    phase_per_r[r].extend(ov[1:].numpy().flatten().tolist())
            if any(len(v) > 0 for v in phase_per_r.values()):
                phase_med = [float(np.median(phase_per_r[r])) if phase_per_r[r] else float("nan") for r in rs]
                phase_p10 = [float(np.percentile(phase_per_r[r], 10)) if phase_per_r[r] else float("nan") for r in rs]
                phase_p90 = [float(np.percentile(phase_per_r[r], 90)) if phase_per_r[r] else float("nan") for r in rs]
                ax.fill_between(rs, phase_p10, phase_p90, color="#CC6677", alpha=0.2, label="phase 10-90%")
                ax.plot(rs, phase_med, color="#CC6677", linewidth=2.2, marker="s",
                        label="phase median (within-task prefill vs decode)")
        ax.set_xlabel("rank $r$")
        ax.set_title(f"{metric}: top-$r$ subspace overlap")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("subspace overlap (l0excl)")
    fig.suptitle("Top-r subspace overlap across distribution splits — closer to 1 = more agreement",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e12_subspace_overlap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_distance_vs_e5_gap(data: dict) -> None:
    """Per (layer, kv_head) scatter: x = phase Bures distance, y = E5 decode-prefill top1 gap."""
    stats = data["stats"]
    e3_path = BASE / "e3" / "e3_b3_r64_summary.json"
    if not e3_path.exists():
        return
    with open(e3_path) as f:
        e3 = json.load(f)
    if e3.get("query_phase") not in {"both", "decode"}:
        return
    methods = list(e3["aggregated"].keys())
    # The summary is per-layer; we want per (layer, kv_head). Per-layer is the granularity we have.
    fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 5), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    n_layers = stats["n_layers"]
    n_kv_heads = stats["n_kv_heads"]
    # Aggregate phase distance per layer (mean across kv heads), per task, then average across tasks.
    avg_phase_dist_per_layer = torch.zeros(n_layers)
    for cfg in stats["configs"]:
        if f"{cfg}/decode" in stats["low_confidence_tasks"]:
            continue
        d = stats["phase_distances"][cfg]
        avg_phase_dist_per_layer = avg_phase_dist_per_layer + d.mean(dim=-1)
    n_used = len(stats["configs"]) - sum(1 for c in stats["configs"] if f"{c}/decode" in stats["low_confidence_tasks"])
    avg_phase_dist_per_layer = (avg_phase_dist_per_layer / max(1, n_used)).numpy()
    for ax, method in zip(axes, methods):
        agg = e3["aggregated"][method]
        if "top1_decode" not in agg or "top1_prefill" not in agg:
            ax.set_visible(False)
            continue
        d_top1 = np.array(agg["top1_decode"]["per_layer"]) - np.array(agg["top1_prefill"]["per_layer"])
        # Layer-0 excluded
        x = avg_phase_dist_per_layer[1:]
        y = d_top1[1:]
        # Clip to finite range
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        ax.scatter(x, y, s=24, alpha=0.7)
        if x.size >= 2:
            corr = float(np.corrcoef(x, y)[0, 1])
            ax.set_title(f"{method}: corr={corr:+.2f}")
        else:
            ax.set_title(method)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("avg phase Bures distance per layer")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("E5 decode minus prefill top-1 gap")
    fig.suptitle("Does distributional phase shift predict the decode-vs-prefill top-1 gap? (l0excl)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e12_distance_vs_e5_gap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    data = _load()
    print(f"Loaded {len(data['stats']['configs'])} configs: {data['stats']['configs']}")
    print(f"Output dir: {OUT}")
    plot_marginal_cumulative_energy(data)
    print("  wrote e12_marginal_cumulative_energy.png")
    plot_r95_per_task_per_metric(data)
    print("  wrote e12_marginal_r95_per_task.png")
    plot_per_layer_r95(data)
    print("  wrote e12_per_layer_r95.png")
    plot_layer0_anomaly(data)
    print("  wrote e12_layer0_anomaly_qk.png")
    plot_phase_distance_heatmap(data)
    print("  wrote e12_phase_distance_heatmap.png")
    plot_task_distance_heatmap(data)
    print("  wrote e12_task_distance_heatmap.png")
    plot_subspace_overlap(data)
    print("  wrote e12_subspace_overlap.png")
    plot_distance_vs_e5_gap(data)
    print("  wrote e12_distance_vs_e5_gap.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
