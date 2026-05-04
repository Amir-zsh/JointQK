#!/usr/bin/env python3
"""Cross-model chart: top-1 vs b_avg for Qwen3-8B and Llama-3.1-8B side-by-side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ["v3", "v_waterfill", "cca_orth_waterfill", "r_sym_waterfill"]
METHOD_LABELS = {
    "v3": "TurboQuant (V3)",
    "v_waterfill": "Q-Eigen WaterFill",
    "cca_orth_waterfill": "CCA-Orth WaterFill",
    "r_sym_waterfill": "JointQK WaterFill",
}
COLORS = {"v3": "#888888", "v_waterfill": "#1f77b4",
          "cca_orth_waterfill": "#2ca02c", "r_sym_waterfill": "#d62728"}


def load_top1(summary_dir: Path, b_avgs: list[int]) -> dict[str, list[float]]:
    """Returns {method: [top1@b=2, top1@b=3, top1@b=4]}, layer-0 excluded."""
    out: dict[str, list[float]] = {m: [] for m in METHODS}
    for b in b_avgs:
        f = summary_dir / f"e3_b{b}_r64_summary.json"
        if not f.exists():
            for m in METHODS:
                out[m].append(float("nan"))
            continue
        s = json.loads(f.read_text())
        # Stage-1E summary structure:
        #   {"aggregated": {method: {"top1_prefill": {"per_layer": [...], "all_mean": ..., "l0excl_mean": ...}}}}
        agg = s.get("aggregated") or {}
        for m in METHODS:
            v = (agg.get(m) or {}).get("top1_prefill", {}).get("l0excl_mean")
            out[m].append(float("nan") if v is None else float(v))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-dir", default="artifacts/stage1/cca_vs_waterfill_study/e3")
    p.add_argument("--llama-dir", default="artifacts/stage1/cca_vs_waterfill_study/llama31_8b")
    p.add_argument("--out", default="artifacts/stage1/cca_vs_waterfill_study/report_charts/cross_model_b_sensitivity.png")
    args = p.parse_args()

    b_avgs = [2, 3, 4]
    qwen = load_top1(Path(args.qwen_dir), b_avgs)
    llama = load_top1(Path(args.llama_dir), b_avgs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (title, data) in zip(axes, [("Qwen3-8B", qwen), ("Llama-3.1-8B", llama)]):
        for m in METHODS:
            ax.plot(b_avgs, data[m], "o-", label=METHOD_LABELS[m], color=COLORS[m], linewidth=2, markersize=7)
        ax.set_xlabel("b_avg (bits per coordinate)")
        ax.set_xticks(b_avgs)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Top-1 retention (layer-0 excluded)")
    axes[1].legend(loc="lower right", fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
