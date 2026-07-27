#!/usr/bin/env python3
"""Render OSCAR Figure 4 from decode_speed_summary.json files.

Layout follows the paper: LEFT is decode throughput speedup vs BF16 at batch
size 1 across input lengths, one subplot per model; RIGHT is job-level throughput
speedup at a fixed 100k input, one subplot per batch size, models on the x-axis.

Two deliberate differences from the published figure, both annotated on the plot
rather than left for the reader to infer:
  * Saw-INT4 is ABSENT, not substituted -- it is not implemented in this stack,
    so inventing a stand-in baseline would misrepresent the comparison.
  * vq2 is OURS and is not in the paper. It appears only for Qwen3-8B; no vq2
    codebook exists for Qwen3-4B-Thinking-2507.
Published OSCAR values are drawn as hollow markers so reproduction and reference
can be compared at a glance.

Usage:
  python pipelines/eval/plot_fig4.py --out notes/fig4_reproduction.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts/oscar_e2e"

# Paper values, main.pdf p9 (Figure 4). GLM omitted by request.
PAPER = {
    "Qwen3-4B-Thinking-2507": {"left": [1.98, 2.52, 3.08], "right": [3.08, 5.19, 6.17]},
    "Qwen3-8B":               {"left": [1.84, 2.29, 2.88], "right": [2.88, 3.35, 3.44]},
}
MODELS = [
    ("Qwen3-4B-Thinking-2507", "fig4_{}_qwen3_4b_thinking"),
    ("Qwen3-8B", "fig4_{}_qwen3_8b"),
]
STYLE = {
    "bf16": ("#8c8c8c", "BF16"),
    "oscar_int2": ("#1f77b4", "OSCAR-INT2"),
    "vq2_s8": ("#2ca02c", "vq2 (ours)"),
    "vq2_s48": ("#d62728", "vq2 tuned splits (ours)"),
}


def load(study: str) -> dict | None:
    for d in (ART / study, ART / f"{study}_cached", ART / f"{study}_replay"):
        f = d / "decode_speed_summary.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def bars(ax, summary, cols, title, paper_vals):
    if summary is None:
        ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes,
                color="#999")
        ax.set_title(title, fontsize=10)
        return
    sp = summary["speedup_vs_reference"]
    labels = [l for l in STYLE if l in sp]
    w = 0.8 / max(1, len(labels))
    for i, lab in enumerate(labels):
        vals = [sp[lab].get(c) or 0 for c in cols]
        colour, disp = STYLE[lab]
        xs = [x + i * w for x in range(len(cols))]
        ax.bar(xs, vals, w, color=colour, label=disp)
        for x, v in zip(xs, vals):
            if v:
                ax.text(x, v + 0.04, f"{v:.2f}", ha="center", fontsize=7)
    if paper_vals:
        centre = [x + 0.4 - w / 2 for x in range(len(cols))]
        ax.plot(centre, paper_vals, "o", mfc="none", mec="black", ms=7,
                label="OSCAR (paper)")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([x + 0.4 - w / 2 for x in range(len(cols))])
    ax.axhline(1.0, color="#bbb", lw=0.8, ls="--")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "notes/fig4_reproduction.png")
    a = ap.parse_args()

    fig, axes = plt.subplots(2, max(2, len(MODELS)), figsize=(11, 7))
    ctx_lbl = ["30k", "60k", "100k"]
    bs_lbl = ["BS=1", "BS=8", "BS=32"]

    for j, (model, tmpl) in enumerate(MODELS):
        s = load(tmpl.format("left"))
        cols = [f"in{n}_bs1" for n in (30000, 60000, 100000)]
        bars(axes[0][j], s, cols, model, PAPER.get(model, {}).get("left"))
        axes[0][j].set_xticklabels(ctx_lbl)
        axes[0][j].set_xlabel("input length")

        s = load(tmpl.format("right"))
        cols = [f"in100000_bs{b}" for b in (1, 8, 32)]
        bars(axes[1][j], s, cols, model, PAPER.get(model, {}).get("right"))
        axes[1][j].set_xticklabels(bs_lbl)
        axes[1][j].set_xlabel("batch size (100k input)")

    axes[0][0].set_ylabel("decode throughput\nspeedup vs BF16 (x)")
    axes[1][0].set_ylabel("job-level throughput\nspeedup vs BF16 (x)")
    for row in axes:
        for ax in row:
            ax.grid(axis="y", alpha=0.25)
    h, l = axes[0][1].get_legend_handles_labels()
    if not h:
        h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle("OSCAR Figure 4 reproduction — Qwen3 (H100, TP=1)\n"
                 "Saw-INT4 absent (not implemented here); vq2 is ours, not in the paper",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
