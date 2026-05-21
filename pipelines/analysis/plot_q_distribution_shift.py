#!/usr/bin/env python3
"""Generate the 4 charts for the Q distribution shift analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

TASKS_ORDER = ["hotpotqa", "qasper", "qmsum", "multi_news",
               "repobench-p", "musique", "lcc", "2wikimqa"]
IN_CALIB = {"hotpotqa", "qasper", "qmsum", "multi_news", "repobench-p", "musique"}
TASK_COLORS = {
    "hotpotqa": "#1f77b4", "qasper": "#2ca02c", "qmsum": "#9467bd",
    "multi_news": "#7f7f7f", "repobench-p": "#bcbd22", "musique": "#17becf",
    "lcc": "#d62728", "2wikimqa": "#ff7f0e",
}


def chart_prefill_drift(data: dict, out_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    axes = axes.flatten()
    for i, task in enumerate(TASKS_ORDER):
        ax = axes[i]
        td = data["tasks"].get(task, {})
        binned = td.get("prefill_binned", [])
        if not binned:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            xs = [b["position"] for b in binned]
            ys = [b["mean_cos"] for b in binned]
            stds = [b["std_cos"] for b in binned]
            ax.plot(xs, ys, marker="o", color=TASK_COLORS[task], linewidth=1.5)
            ax.fill_between(xs,
                            [y - s for y, s in zip(ys, stds)],
                            [y + s for y, s in zip(ys, stds)],
                            alpha=0.2, color=TASK_COLORS[task])
        ax.set_xlabel("Prefill position (tokens)")
        if i % 4 == 0:
            ax.set_ylabel(f"Top-{data.get('top_k',16)} subspace cosine\n(vs compact8 reference)")
        ax.set_ylim(0.4, 1.02)
        ood_tag = "" if task in IN_CALIB else " (OOD)"
        ax.set_title(f"{task}{ood_tag}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    fig.suptitle("Σ_Q drift across prefill positions vs compact8 calibration reference\n"
                 "(200-token windows, averaged across 5 test prompts/task)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def chart_decode_drift(data: dict, out_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    axes = axes.flatten()
    for i, task in enumerate(TASKS_ORDER):
        ax = axes[i]
        td = data["tasks"].get(task, {})
        bins = td.get("decode_bins", [])
        if not bins:
            ax.text(0.5, 0.5, "no decode data", ha="center", va="center", transform=ax.transAxes)
        else:
            xs = [b["bin"] for b in bins]
            ys = [b["mean_cos"] for b in bins]
            ax.bar(xs, ys, color=TASK_COLORS[task])
            # annotate sample counts
            for j, b in enumerate(bins):
                ax.text(j, ys[j] + 0.005, f"n={b['n_samples']}", ha="center", fontsize=8)
        ax.set_xlabel("Decode step bin")
        if i % 4 == 0:
            ax.set_ylabel(f"Top-{data.get('top_k',16)} subspace cosine")
        ax.set_ylim(0.4, 1.02)
        ood_tag = "" if task in IN_CALIB else " (OOD)"
        ax.set_title(f"{task}{ood_tag}", fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    fig.suptitle("Σ_Q drift in decode-step bins vs compact8 calibration reference",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def chart_combined_trajectory(data: dict, out_path: Path):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    for task in TASKS_ORDER:
        td = data["tasks"].get(task, {})
        binned = td.get("prefill_binned", [])
        if not binned: continue
        max_pos = max(b["position"] for b in binned)
        xs = [b["position"] for b in binned]
        ys = [b["mean_cos"] for b in binned]
        ls = "-" if task in IN_CALIB else "--"
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, linestyle=ls,
                color=TASK_COLORS[task],
                label=f"{task}{'' if task in IN_CALIB else ' (OOD)'}")
        # Append decode bins as separate segment to the right
        bins = td.get("decode_bins", [])
        if bins:
            decode_offset = max_pos + 1000
            xs_d = [decode_offset + i * 600 for i in range(len(bins))]
            ys_d = [b["mean_cos"] for b in bins]
            ax.plot(xs_d, ys_d, marker="s", markersize=7, linewidth=1.5, linestyle=":",
                    color=TASK_COLORS[task])
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Position (prefill tokens, then decode bins to the right)")
    ax.set_ylabel(f"Top-{data.get('top_k',16)} subspace cosine to compact8 ref")
    ax.set_title("Combined prefill → decode Σ_Q drift trajectory per task\n"
                 "(solid lines/circles = prefill; dotted/squares = decode-step bins)")
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def chart_summary_bars(data: dict, out_path: Path, f1_table_path: Path | None = None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(TASKS_ORDER))
    width = 0.35
    prefill_means, decode_means = [], []
    for task in TASKS_ORDER:
        td = data["tasks"].get(task, {})
        pp = td.get("prefill_points", [])
        prefill_means.append(sum(p["cos"] for p in pp) / len(pp) if pp else 0)
        dbins = td.get("decode_bins", [])
        if dbins:
            # sample-weighted average
            tot_n = sum(b["n_samples"] for b in dbins)
            decode_means.append(sum(b["mean_cos"] * b["n_samples"] for b in dbins) / max(1, tot_n))
        else:
            decode_means.append(0)
    bars1 = ax.bar(x - width/2, prefill_means, width, label="Prefill (mean across windows)",
                   color=[TASK_COLORS[t] for t in TASKS_ORDER], alpha=0.85)
    bars2 = ax.bar(x + width/2, decode_means, width, label="Decode (mean across bins, sample-weighted)",
                   color=[TASK_COLORS[t] for t in TASKS_ORDER], alpha=0.4, hatch="//")
    # Annotate bars
    for bar, v in zip(bars1, prefill_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.2f}",
                ha="center", fontsize=8)
    for bar, v in zip(bars2, decode_means):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xticks(x)
    labels = [f"{t}{'' if t in IN_CALIB else '\n(OOD)'}" for t in TASKS_ORDER]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(f"Top-{data.get('top_k',16)} subspace cosine to compact8 ref")
    ax.set_title("Summary: mean Σ_Q drift per task (prefill vs decode)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.05)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path",
        default=str(REPO / "artifacts/q_distribution_shift/per_task_drift.json"))
    parser.add_argument("--out-dir",
        default=str(REPO / "notes/figs/q_drift"))
    args = parser.parse_args()

    data = json.loads(Path(args.in_path).read_text())
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    chart_prefill_drift(data, out_dir / "prefill_drift.png")
    chart_decode_drift(data, out_dir / "decode_drift.png")
    chart_combined_trajectory(data, out_dir / "combined_trajectory.png")
    chart_summary_bars(data, out_dir / "summary_bars.png")


if __name__ == "__main__":
    main()
