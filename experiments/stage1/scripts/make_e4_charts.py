"""Generate E4 generalization charts (cross-task + within-task LOO) for the Stage 1E review.

Reads the post-F1/post-F11 canonical summaries in:
    artifacts/stage1/cca_vs_waterfill_study/e4a/
    artifacts/stage1/cca_vs_waterfill_study/e4b/

Writes to:
    artifacts/stage1/cca_vs_waterfill_study/report_charts/
        e4_cross_task_heatmap_top1.png   3 x 3 calib x eval matrix per method (top-1)
        e4_cross_task_heatmap_geo.png    3 x 3 calib x eval matrix per method (geometry)
        e4_loo_fold_top1.png             per-fold top-1 across 24 folds, colored by config
        e4_loo_variance.png              std dev across folds, per (method, config)
        e4_in_domain_vs_e3.png           E4a in-domain diagonal vs E3 (calib all 24)
        e4_pre_f11_vs_post_f11.png       cca_waterfill pre-F11 vs post-F11 across folds
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
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
E4A_DIR = BASE / "e4a"
E4B_DIR = BASE / "e4b"
OUT = BASE / "report_charts"
OUT.mkdir(parents=True, exist_ok=True)


METHODS = [
    "v3", "v_truncate", "v_waterfill",
    "cca_uniform", "cca_waterfill",
    "cca_orth_uniform", "cca_orth_waterfill",
    "r_sym_uniform", "r_sym_waterfill",
]
METHOD_LABELS = {
    "v3": "V3",
    "v_truncate": "V truncate r=64",
    "v_waterfill": "V + water-fill",
    "cca_uniform": "CCA uniform r=64",
    "cca_waterfill": "CCA + water-fill",
    "cca_orth_uniform": "CCA (V_h orth) uniform r=64",
    "cca_orth_waterfill": "CCA (V_h orth) + water-fill",
    "r_sym_uniform": "R_sym uniform r=64",
    "r_sym_waterfill": "R_sym + water-fill",
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
CONFIGS = ["qasper", "hotpotqa", "passage_retrieval_en"]
CONFIG_LABELS = {
    "qasper": "qasper",
    "hotpotqa": "hotpotqa",
    "passage_retrieval_en": "passage\nretrieval_en",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def headline(summary: dict, method: str, metric: str) -> float:
    return float(summary["aggregated"][method][metric]["l0excl_mean"])


# ---------- E4a cross-task per-(eval-config, method) means ----------


def cross_task_matrix(metric: str) -> np.ndarray:
    """Returns matrix[calib_idx, eval_idx, method_idx] of layer-0-excluded means."""
    out = np.full((len(CONFIGS), len(CONFIGS), len(METHODS)), np.nan)
    for ci, src in enumerate(CONFIGS):
        path = E4A_DIR / f"e4a_calib_{src}_b3_r64_rows.pt"
        rows = torch.load(path, map_location="cpu", weights_only=False)["rows"]
        bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in rows:
            if r["layer"] == 0:
                continue
            cfg = r["config"].removesuffix("_e")
            bucket[(cfg, r["method"])].append(float(r[metric]))
        for ei, eval_cfg in enumerate(CONFIGS):
            for mi, m in enumerate(METHODS):
                vals = bucket.get((eval_cfg, m), [])
                if vals:
                    out[ci, ei, mi] = float(np.mean(vals))
    return out


# ---------- charts ----------


def chart_cross_task_heatmap(metric: str, file_suffix: str, cmap_name: str, fmt: str, title_metric: str) -> Path:
    matrix = cross_task_matrix(metric)
    n_methods = len(METHODS)
    n_cols = 3
    n_rows = math.ceil(n_methods / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.0 * n_rows))
    axes_flat = axes.flatten() if n_rows > 1 else axes
    vmax = np.nanmax(matrix)
    vmin = np.nanmin(matrix)
    # Log color scale for metrics that span multiple orders of magnitude
    # (geometry distortion ranges from ~0.05 to ~62 across methods).
    use_log = metric == "geometry_distortion" and vmax / max(vmin, 1e-12) > 50
    if use_log:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    else:
        norm = None
    last_im = None
    for mi, m in enumerate(METHODS):
        ax = axes_flat[mi]
        data = matrix[:, :, mi]
        if use_log:
            im = ax.imshow(data, cmap=cmap_name, norm=norm, aspect="auto")
        else:
            im = ax.imshow(data, cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
        last_im = im
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_yticks(range(len(CONFIGS)))
        # Bottom-row panels show x ticks; left-column panels show y ticks; rest hide for clarity.
        row_idx, col_idx = mi // n_cols, mi % n_cols
        is_bottom_row = row_idx == n_rows - 1 or mi >= n_methods - n_cols
        is_left_col = col_idx == 0
        if is_bottom_row:
            ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS], rotation=20, fontsize=10)
            ax.set_xlabel("evaluation config", fontsize=10)
        else:
            ax.set_xticklabels([])
        if is_left_col:
            ax.set_yticklabels([CONFIG_LABELS[c] for c in CONFIGS], fontsize=10)
            ax.set_ylabel("calibration source", fontsize=10)
        else:
            ax.set_yticklabels([])
        for ci in range(len(CONFIGS)):
            for ei in range(len(CONFIGS)):
                v = data[ci, ei]
                if not np.isnan(v):
                    diag = ci == ei
                    if use_log:
                        # Threshold on log scale.
                        logmid = 0.5 * (math.log10(max(vmin, 1e-3)) + math.log10(vmax))
                        is_dark = math.log10(max(v, 1e-12)) < logmid
                    else:
                        is_dark = v < (vmin + vmax) / 2
                    ax.text(ei, ci, fmt.format(v), ha="center", va="center",
                            fontsize=11, color="black" if is_dark else "white",
                            fontweight="bold" if diag else "normal")
        ax.set_title(METHOD_LABELS[m], fontsize=12)
    # Hide any leftover empty panels (e.g. if n_methods < n_rows * n_cols).
    for extra in range(n_methods, n_rows * n_cols):
        axes_flat[extra].set_visible(False)
    fig.suptitle(
        f"E4a cross-task {title_metric} (b_avg=3, layer-0 excluded). Bold = in-domain.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    if last_im is not None:
        cbar_ax = fig.add_axes([0.945, 0.10, 0.012, 0.78])
        fig.colorbar(last_im, cax=cbar_ax)
    out_path = OUT / f"e4_cross_task_heatmap_{file_suffix}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_loo_fold_top1() -> Path:
    fold_data: dict[str, dict[int, float]] = defaultdict(dict)
    fold_config: dict[int, str] = {}
    for sf in sorted(E4B_DIR.glob("e4b_*_summary.json")):
        if "smoke" in sf.name:
            continue
        s = load(sf)
        idx = int(s["loo_index"])
        fold_config[idx] = s["loo_config"]
        for m in METHODS:
            fold_data[m][idx] = headline(s, m, "top1_prefill")

    folds = sorted(fold_config.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(folds))
    for m in METHODS:
        ys = [fold_data[m][i] for i in folds]
        ax.plot(x, ys, marker="o", label=METHOD_LABELS[m], color=COLORS[m], linewidth=1.5, markersize=4)

    # shade per-config bands
    cfg_starts = {}
    for cfg in CONFIGS:
        idxs = [i for i in folds if fold_config[i] == cfg]
        if idxs:
            cfg_starts[cfg] = (min(idxs), max(idxs))
    band_colors = {"qasper": "#FFE4B5", "hotpotqa": "#E0FFE0", "passage_retrieval_en": "#E0E8FF"}
    for cfg, (lo, hi) in cfg_starts.items():
        ax.axvspan(folds.index(lo) - 0.5, folds.index(hi) + 0.5, color=band_colors[cfg], alpha=0.4, zorder=0)
        mid = (folds.index(lo) + folds.index(hi)) / 2.0
        ax.text(mid, ax.get_ylim()[1] * 0.98, cfg, ha="center", va="top", fontsize=9, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=8)
    ax.set_xlabel("LOO held-out example index")
    ax.set_ylabel("layer-0-excluded top-1 retention")
    ax.set_title("E4b LOO: top-1 across 24 folds (b_avg=3, layer-0 excluded)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=5, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT / "e4_loo_fold_top1.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_loo_variance() -> Path:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for sf in sorted(E4B_DIR.glob("e4b_*_summary.json")):
        if "smoke" in sf.name:
            continue
        s = load(sf)
        by_config[s["loo_config"]].append(s)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.18
    x = np.arange(len(METHODS))
    for ci, cfg in enumerate(CONFIGS):
        runs = by_config.get(cfg, [])
        sds = []
        for m in METHODS:
            vals = [headline(r, m, "top1_prefill") for r in runs]
            sds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        ax.bar(x + (ci - 1) * width, sds, width, label=cfg, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right")
    ax.set_ylabel("std dev of top-1 across 8 LOO folds")
    ax.set_title("E4b: top-1 std dev across LOO folds, by config (b_avg=3, layer-0 excluded)")
    ax.legend(title="LOO config", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = OUT / "e4_loo_variance.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_in_domain_vs_e3() -> Path:
    """E4a per-config in-domain top-1 (calib=eval) vs E3 baseline (calib all 24)."""
    e3 = load(E3_DIR / "e3_b3_r64_summary.json")
    e3_top1 = {m: headline(e3, m, "top1_prefill") for m in METHODS}

    matrix = cross_task_matrix("top1_prefill")  # [calib, eval, method]
    in_domain_per_config = {}
    for ci, cfg in enumerate(CONFIGS):
        in_domain_per_config[cfg] = {METHODS[mi]: float(matrix[ci, ci, mi]) for mi in range(len(METHODS))}

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    x = np.arange(len(METHODS))
    cfg_offset = {cfg: (ci - 1.5) * width for ci, cfg in enumerate(CONFIGS)}
    for ci, cfg in enumerate(CONFIGS):
        ys = [in_domain_per_config[cfg][m] for m in METHODS]
        ax.bar(x + cfg_offset[cfg], ys, width, label=f"E4a in-domain, calib={cfg}", alpha=0.85)
    ys_e3 = [e3_top1[m] for m in METHODS]
    ax.bar(x + 1.5 * width, ys_e3, width, label="E3 (calib all 24)", alpha=0.85, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right")
    ax.set_ylabel("layer-0-excluded top-1 retention")
    ax.set_title("E4a in-domain top-1 vs E3 baseline (b_avg=3)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = OUT / "e4_in_domain_vs_e3.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_pre_post_f11() -> Path:
    """cca_waterfill pre-F11 vs post-F11 top-1 across 24 LOO folds."""
    pre, post, idxs, cfgs = [], [], [], []
    for sf in sorted(E4B_DIR.glob("e4b_*_summary.json")):
        if "smoke" in sf.name:
            continue
        post_s = load(sf)
        pre_p = sf.with_name(sf.name + ".pre_f11")
        if not pre_p.exists():
            continue
        pre_s = json.loads(pre_p.read_text())
        idxs.append(int(post_s["loo_index"]))
        cfgs.append(post_s["loo_config"])
        pre.append(headline(pre_s, "cca_waterfill", "top1_prefill"))
        post.append(headline(post_s, "cca_waterfill", "top1_prefill"))

    order = sorted(range(len(idxs)), key=lambda i: idxs[i])
    idxs = [idxs[i] for i in order]
    cfgs = [cfgs[i] for i in order]
    pre = [pre[i] for i in order]
    post = [post[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(idxs))
    ax.bar(x - 0.2, pre, 0.4, label="pre-F11 (rho^2 allocation)", color="#999")
    ax.bar(x + 0.2, post, 0.4, label="post-F11 (trace-formula)", color=COLORS["cca_waterfill"])
    ax.set_xticks(x)
    ax.set_xticklabels(idxs, fontsize=8)
    ax.set_xlabel("LOO held-out example index")
    ax.set_ylabel("layer-0-excluded top-1 retention")
    ax.set_title("E4b cca_waterfill: pre-F11 vs post-F11 top-1, per fold (b_avg=3)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = OUT / "e4_pre_f11_vs_post_f11.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    paths = [
        chart_cross_task_heatmap("top1_prefill", "top1", "viridis", "{:.3f}", "top-1 retention"),
        chart_cross_task_heatmap("geometry_distortion", "geo", "viridis_r", "{:.3f}", "geometry distortion (lower is better)"),
        chart_loo_fold_top1(),
        chart_loo_variance(),
        chart_in_domain_vs_e3(),
        chart_pre_post_f11(),
    ]
    for p in paths:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
