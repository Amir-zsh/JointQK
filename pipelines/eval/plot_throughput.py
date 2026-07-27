#!/usr/bin/env python3
"""Render the controlled throughput benchmark from throughput_summary.json.

One row per model, one column per input length. Each panel groups the three
operating points -- bs=1 (equal setting), bs=4 (scaling), bs=max (best
achievable per GPU) -- with one bar per arm.

bs=max bars are annotated with the batch size that arm actually reached, because
the arms do NOT reach the same batch there: a bigger KV pool admits a bigger
batch, and that is the result rather than a confound.

Usage:
  python pipelines/eval/plot_throughput.py --out notes/throughput_benchmark.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts/throughput"

STUDIES = [
    ("Qwen3-8B", "throughput_qwen3_8b"),
    ("Qwen3-4B-Thinking-2507", "throughput_qwen3_4b_thinking"),
]
STYLE = [
    ("bf16", "#8c8c8c", "BF16"),
    ("oscar_int2", "#1f77b4", "OSCAR-INT2"),
    ("vq2_s8", "#2ca02c", "vq2 (splits 8)"),
    ("vq2_s48", "#d62728", "vq2 (splits 48)"),
    ("vq2_cuda", "#9467bd", "vq2 CUDA (shared codebook)"),
]


def load(study: str) -> dict | None:
    f = ART / study / "throughput_summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def cell_for(rows: dict, arm: str, ctx: int, which: str) -> dict | None:
    cands = [m for m in rows.get(arm, {}).values()
             if m.get("ok") and m.get("input_len") == ctx]
    if not cands:
        return None
    if which == "max":
        return max(cands, key=lambda m: m["decode_tok_s"])
    bs = 1 if which == "1" else 4
    hit = [m for m in cands if m["batch_size"] == bs]
    return hit[0] if hit else None


def panel(ax, summary, ctx, show_legend=False):
    rows = summary["rows"]
    groups = [("1", "bs=1"), ("4", "bs=4"), ("max", "bs=max")]
    arms = [(a, c, l) for a, c, l in STYLE if a in rows]
    w = 0.8 / max(1, len(arms))
    for i, (arm, colour, label) in enumerate(arms):
        xs, ys, notes = [], [], []
        for gi, (which, _) in enumerate(groups):
            m = cell_for(rows, arm, ctx, which)
            xs.append(gi + i * w)
            ys.append(m["decode_tok_s"] if m else 0)
            notes.append(m["batch_size"] if (m and which == "max") else None)
        ax.bar(xs, ys, w, color=colour, label=label if show_legend else None)
        for x, y, n in zip(xs, ys, notes):
            if y and n:
                ax.text(x, y, f"b{n}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks([g + 0.4 - w / 2 for g in range(len(groups))])
    ax.set_xticklabels([lbl for _, lbl in groups], fontsize=8)
    ax.set_title(f"{ctx // 1000}k input", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "notes/throughput_benchmark.png")
    a = ap.parse_args()

    loaded = [(name, load(s)) for name, s in STUDIES]
    loaded = [(n, s) for n, s in loaded if s]
    if not loaded:
        print("no throughput_summary.json found -- run the sweep first")
        return 1

    lens = loaded[0][1].get("rows") and sorted(
        {m["input_len"] for cells in loaded[0][1]["rows"].values()
         for m in cells.values() if m.get("input_len")})
    fig, axes = plt.subplots(len(loaded), len(lens),
                             figsize=(3.4 * len(lens), 3.1 * len(loaded)),
                             squeeze=False)
    for r, (name, summary) in enumerate(loaded):
        for c, ctx in enumerate(lens):
            panel(axes[r][c], summary, ctx, show_legend=(r == 0 and c == 0))
        axes[r][0].set_ylabel(f"{name}\naggregate decode tok/s", fontsize=8)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=8)
    fig.suptitle(
        "Decode throughput, prefill excluded — H100 TP=1, no cross-request prefix sharing\n"
        "bs=max is each arm's largest batch that fits its own KV pool (annotated bN)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print(f"wrote {a.out}")

    for name, s in loaded:
        print(f"\n=== {name} — speedup vs {s['reference']} ===")
        for arm, cells in s.get("speedup_best", {}).items():
            if arm == s["reference"]:
                continue
            for cell, d in sorted(cells.items()):
                print(f"  {arm:11s} {cell:>9s}  best {d['tok_s']:>8,.0f} tok/s "
                      f"(bs={d['bs']:>2})  vs {d['ref_tok_s']:>8,.0f} "
                      f"(bs={d['ref_bs']:>2})  = {d['speedup']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
