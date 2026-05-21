#!/usr/bin/env python3
"""Generate Phase 1D K-basis calibration stability charts.

Reads:
    artifacts/v_bases/k_basis_stability/
        phase1d_k_basis_stability_summary.json

Writes:
    artifacts/v_bases/k_basis_stability/figures/
        phase1d_pooled_eval_regret_k3_vs_n.png
        phase1d_pooled_eval_overlap_r64_vs_n.png
        phase1d_cross_task_regret_k3_n4.png
        phase1d_pooled_stratified_regret_by_eval.png
        phase1d_leave_one_out_regret_k3_vs_n.png
        README.md
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO / "artifacts/v_bases/k_basis_stability/phase1d_k_basis_stability_summary.json"
DEFAULT_OUT = REPO / "artifacts/v_bases/k_basis_stability/figures"

SOURCE_ORDER = ["qasper_e", "hotpotqa_e", "passage_retrieval_en_e", "pooled_stratified"]
TASK_EVAL_ORDER = ["qasper_e", "hotpotqa_e", "passage_retrieval_en_e", "pooled"]
N_ORDER = [1, 2, 4, 8]
COLORS = {
    "qasper_e": "#4477AA",
    "hotpotqa_e": "#228833",
    "passage_retrieval_en_e": "#CC6677",
    "pooled_stratified": "#AA3377",
}
LABELS = {
    "qasper_e": "Qasper",
    "hotpotqa_e": "HotpotQA",
    "passage_retrieval_en_e": "Passage retrieval",
    "pooled_stratified": "Pooled stratified",
    "pooled": "Pooled",
}


def load_summary(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def group_key(source: str, n_label: int, eval_name: str) -> str:
    return f"{source}|n={n_label}|eval={eval_name}"


def split_group_key(key: str) -> tuple[str, int, str]:
    source, n_part, eval_part = key.split("|")
    return source, int(n_part.split("=", 1)[1]), eval_part.split("=", 1)[1]


def available_n(summary: dict[str, Any]) -> list[int]:
    return sorted({split_group_key(key)[1] for key in summary["groups"]})


def ordered_sources(summary: dict[str, Any], *, include_loo: bool = False) -> list[str]:
    present = {split_group_key(key)[0] for key in summary["groups"]}
    base = [s for s in SOURCE_ORDER if s in present]
    loo = sorted(s for s in present if s.startswith("loo_excl_"))
    other = sorted(present - set(base) - set(loo))
    return base + (loo if include_loo else []) + other


def ordered_evals(summary: dict[str, Any]) -> list[str]:
    present = {split_group_key(key)[2] for key in summary["groups"]}
    return [e for e in TASK_EVAL_ORDER if e in present] + sorted(present - set(TASK_EVAL_ORDER))


def metric(summary: dict[str, Any], source: str, n_label: int, eval_name: str, name: str) -> dict[str, float] | None:
    group = summary["groups"].get(group_key(source, n_label, eval_name))
    if group is None:
        return None
    return group[name]


def source_label(source: str) -> str:
    if source.startswith("loo_excl_"):
        return f"LOO excl {source_label(source.removeprefix('loo_excl_'))}"
    return LABELS.get(source, source)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.28, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_pooled_eval_regret(summary: dict[str, Any], out_dir: Path, k_bits: int) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    metric_name = f"log2_regret_k{k_bits}"
    n_order = available_n(summary)
    for source in ordered_sources(summary):
        xs: list[int] = []
        means: list[float] = []
        p10: list[float] = []
        p90: list[float] = []
        for n_label in n_order:
            m = metric(summary, source, n_label, "pooled", metric_name)
            if m is None:
                continue
            xs.append(n_label)
            means.append(float(m["mean"]))
            p10.append(float(m["p10"]))
            p90.append(float(m["p90"]))
        if not xs:
            continue
        color = COLORS[source]
        ax.plot(xs, means, marker="o", linewidth=2.2, color=color, label=source_label(source))
        if len(xs) > 1:
            ax.fill_between(xs, p10, p90, color=color, alpha=0.14, linewidth=0)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_order)
    ax.set_xticklabels([str(n) for n in n_order])
    ax.set_xlabel("Calibration examples per source label")
    ax.set_ylabel(f"Mean log2 regret at K={k_bits} bits")
    ax.set_title("Pooled evaluation: K-basis regret falls fastest with stratified calibration")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out_dir / f"phase1d_pooled_eval_regret_k{k_bits}_vs_n.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pooled_eval_overlap(summary: dict[str, Any], out_dir: Path, rank: int) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    metric_name = f"subspace_overlap_r{rank}"
    n_order = available_n(summary)
    for source in ordered_sources(summary):
        xs: list[int] = []
        means: list[float] = []
        p10: list[float] = []
        p90: list[float] = []
        for n_label in n_order:
            m = metric(summary, source, n_label, "pooled", metric_name)
            if m is None:
                continue
            xs.append(n_label)
            means.append(float(m["mean"]))
            p10.append(float(m["p10"]))
            p90.append(float(m["p90"]))
        if not xs:
            continue
        color = COLORS[source]
        ax.plot(xs, means, marker="o", linewidth=2.2, color=color, label=source_label(source))
        if len(xs) > 1:
            ax.fill_between(xs, p10, p90, color=color, alpha=0.14, linewidth=0)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_order)
    ax.set_xticklabels([str(n) for n in n_order])
    ax.set_ylim(0.80, 1.01)
    ax.set_xlabel("Calibration examples per source label")
    ax.set_ylabel(f"Mean top-{rank} subspace overlap")
    ax.set_title("Pooled evaluation: top-64 basis overlap stabilizes with pooled task coverage")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    path = out_dir / f"phase1d_pooled_eval_overlap_r{rank}_vs_n.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cross_task_heatmap(summary: dict[str, Any], out_dir: Path, n_label: int, k_bits: int) -> Path:
    sources = ordered_sources(summary, include_loo=True)
    evals = ordered_evals(summary)
    matrix = np.full((len(sources), len(evals)), np.nan, dtype=float)
    metric_name = f"log2_regret_k{k_bits}"
    for si, source in enumerate(sources):
        for ei, eval_name in enumerate(evals):
            m = metric(summary, source, n_label, eval_name, metric_name)
            if m is not None:
                matrix[si, ei] = float(m["mean"])

    fig_h = max(5.6, 0.55 * len(sources) + 2.2)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    vmax = float(np.nanmax(matrix))
    im = ax.imshow(matrix, cmap="YlGnBu_r", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(evals)))
    ax.set_yticks(np.arange(len(sources)))
    ax.set_xticklabels([source_label(x) for x in evals], rotation=20, ha="right")
    ax.set_yticklabels([source_label(x) for x in sources])
    ax.set_xlabel("Evaluation reference")
    ax.set_ylabel("Calibration source")
    ax.set_title(f"Cross-task K-basis regret at n={n_label}, K={k_bits} bits")
    for si in range(matrix.shape[0]):
        for ei in range(matrix.shape[1]):
            value = matrix[si, ei]
            if np.isnan(value):
                continue
            color = "white" if value > vmax * 0.55 else "black"
            ax.text(ei, si, f"{value:.3f}", ha="center", va="center", color=color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Mean log2 regret at K={k_bits} bits")
    fig.tight_layout()
    path = out_dir / f"phase1d_cross_task_regret_k{k_bits}_n{n_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pooled_stratified_by_eval(summary: dict[str, Any], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), sharex=True, sharey=True)
    for ax, k_bits in zip(axes, [2, 3, 4], strict=True):
        metric_name = f"log2_regret_k{k_bits}"
        n_order = available_n(summary)
        for eval_name in ordered_evals(summary):
            xs: list[int] = []
            means: list[float] = []
            p10: list[float] = []
            p90: list[float] = []
            for n_label in n_order:
                m = metric(summary, "pooled_stratified", n_label, eval_name, metric_name)
                if m is None:
                    continue
                xs.append(n_label)
                means.append(float(m["mean"]))
                p10.append(float(m["p10"]))
                p90.append(float(m["p90"]))
            ax.plot(xs, means, marker="o", linewidth=2.0, label=source_label(eval_name))
            if len(xs) > 1:
                ax.fill_between(xs, p10, p90, alpha=0.10, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_order)
        ax.set_xticklabels([str(n) for n in n_order])
        ax.set_title(f"K={k_bits} bits")
        ax.set_xlabel("Examples per task")
        style_axis(ax)
    axes[0].set_ylabel("Mean log2 regret")
    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Pooled-stratified calibration generalizes across eval tasks", fontsize=13)
    fig.tight_layout()
    path = out_dir / "phase1d_pooled_stratified_regret_by_eval.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_leave_one_out(summary: dict[str, Any], out_dir: Path, k_bits: int) -> Path | None:
    loo_sources = [s for s in ordered_sources(summary, include_loo=True) if s.startswith("loo_excl_")]
    if not loo_sources:
        return None
    n_order = available_n(summary)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    metric_name = f"log2_regret_k{k_bits}"
    for source in loo_sources:
        holdout = source.removeprefix("loo_excl_")
        xs: list[int] = []
        means: list[float] = []
        p10: list[float] = []
        p90: list[float] = []
        for n_label in n_order:
            m = metric(summary, source, n_label, holdout, metric_name)
            if m is None:
                continue
            xs.append(n_label)
            means.append(float(m["mean"]))
            p10.append(float(m["p10"]))
            p90.append(float(m["p90"]))
        if not xs:
            continue
        ax.plot(xs, means, marker="o", linewidth=2.2, label=f"Held out: {source_label(holdout)}")
        if len(xs) > 1:
            ax.fill_between(xs, p10, p90, alpha=0.12, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_order)
    ax.set_xticklabels([str(n) for n in n_order])
    ax.set_xlabel("Calibration examples per non-held-out task")
    ax.set_ylabel(f"Held-out mean log2 regret at K={k_bits} bits")
    ax.set_title("Leave-one-task-out calibration: held-out task regret")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out_dir / f"phase1d_leave_one_out_regret_k{k_bits}_vs_n.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_index(paths: list[Path], out_dir: Path) -> Path:
    lines = [
        "# Phase 1D K-Basis Stability Charts",
        "",
        "Generated from `phase1d_k_basis_stability_summary.json`.",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    index = out_dir / "README.md"
    index.write_text("\n".join(lines) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.summary)
    paths = [
        plot_pooled_eval_regret(summary, args.output_dir, k_bits=3),
        plot_pooled_eval_overlap(summary, args.output_dir, rank=64),
        plot_cross_task_heatmap(summary, args.output_dir, n_label=4, k_bits=3),
        plot_pooled_stratified_by_eval(summary, args.output_dir),
    ]
    loo_path = plot_leave_one_out(summary, args.output_dir, k_bits=3)
    if loo_path is not None:
        paths.append(loo_path)
    paths.append(write_index(paths, args.output_dir))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
