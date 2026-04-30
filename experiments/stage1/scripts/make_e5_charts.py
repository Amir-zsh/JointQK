"""Generate E5 decode-phase charts for the Stage 1E review.

Reads the post-F11 canonical E3 rows (which contain both prefill and decode
metrics because the runs were launched with `--query-phase both`).

Writes to:
    artifacts/stage1/cca_vs_waterfill_study/report_charts/
        e5_decode_vs_prefill_top1.png      grouped bars per (method, b_avg)
        e5_decode_query_count_hist.png     per-example decode query count
        e5_per_layer_gap.png               decode - prefill top-1 by layer for v_waterfill at b=3
        e5_per_example_decode.png          per-example decode top-1 for each method at b=3
        e5_bit_budget_decode.png           decode top-1 vs b_avg per method
"""

from __future__ import annotations

import json
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
OUT = BASE / "report_charts"
OUT.mkdir(parents=True, exist_ok=True)


METHODS = ["v3", "v_truncate", "v_waterfill", "cca_uniform", "cca_waterfill"]
METHOD_LABELS = {
    "v3": "V3",
    "v_truncate": "V truncate r=64",
    "v_waterfill": "V + water-fill",
    "cca_uniform": "CCA uniform r=64",
    "cca_waterfill": "CCA + water-fill",
}
COLORS = {
    "v3": "#777777",
    "v_waterfill": "#228833",
    "v_truncate": "#66CC99",
    "cca_waterfill": "#4477AA",
    "cca_uniform": "#88AACC",
}
B_AVG_GRID = [2, 3, 4]


def load_rows(b_avg: int) -> list[dict]:
    return torch.load(E3_DIR / f"e3_b{b_avg}_r64_rows.pt", map_location="cpu", weights_only=False)["rows"]


def per_method_means(rows: list[dict], metric_pref: str, metric_dec: str) -> dict[str, dict[str, float]]:
    """Return per-method dict with prefill mean, decode mean (unweighted), decode wmean."""
    out: dict[str, dict[str, float]] = {}
    for m in METHODS:
        pref = []
        dec = []
        wts = []
        for r in rows:
            if r["layer"] == 0 or r["method"] != m:
                continue
            pref.append(r[metric_pref])
            dec.append(r[metric_dec])
            wts.append(r["decode_query_count"])
        total_w = sum(wts) if wts else 1
        out[m] = {
            "pref": statistics.mean(pref) if pref else float("nan"),
            "dec": statistics.mean(dec) if dec else float("nan"),
            "dec_w": (sum(d * w for d, w in zip(dec, wts)) / total_w) if total_w else float("nan"),
            "n_rows": len(pref),
            "total_decode_q": int(total_w),
        }
    return out


# ---------- charts ----------


def chart_decode_vs_prefill_top1() -> Path:
    fig, axes = plt.subplots(1, len(B_AVG_GRID), figsize=(4.5 * len(B_AVG_GRID), 4.5), sharey=True)
    width = 0.35
    x = np.arange(len(METHODS))
    for ai, b in enumerate(B_AVG_GRID):
        ax = axes[ai]
        rows = load_rows(b)
        stats = per_method_means(rows, "top1_prefill", "top1_decode")
        prefs = [stats[m]["pref"] for m in METHODS]
        decs = [stats[m]["dec_w"] for m in METHODS]
        ax.bar(x - width / 2, prefs, width, label="prefill Q", color=[COLORS[m] for m in METHODS], alpha=0.55)
        ax.bar(x + width / 2, decs, width, label="decode Q", color=[COLORS[m] for m in METHODS], edgecolor="black", linewidth=1.0)
        for i, (p, d) in enumerate(zip(prefs, decs)):
            ax.annotate(f"{p:.2f}", xy=(i - width / 2, p), ha="center", va="bottom", fontsize=8)
            ax.annotate(f"{d:.2f}", xy=(i + width / 2, d), ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right", fontsize=9)
        ax.set_title(f"b_avg = {b}")
        if ai == 0:
            ax.set_ylabel("layer-0-excluded top-1 retention")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("E5: decode-phase Q (vs compressed prefill K) is *easier* than prefill-phase Q across all methods")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUT / "e5_decode_vs_prefill_top1.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_decode_query_count_hist() -> Path:
    rows = load_rows(3)
    by_ex = {}
    cfg_by_ex = {}
    for r in rows:
        ex = r["example_index"]
        if ex not in by_ex:
            by_ex[ex] = r["decode_query_count"]
            cfg_by_ex[ex] = r["config"].removesuffix("_e")
    exes = sorted(by_ex)
    counts = [by_ex[e] for e in exes]
    cfgs = [cfg_by_ex[e] for e in exes]
    cfg_color = {"qasper": "#EEAA66", "hotpotqa": "#88AA66", "passage_retrieval_en": "#6688CC"}
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(exes, counts, color=[cfg_color[c] for c in cfgs])
    ax.axhline(64, color="red", linestyle=":", linewidth=1, label="plan target ≥ 64")
    ax.axhline(16, color="orange", linestyle=":", linewidth=1, label="plan minimum ≥ 16")
    ax.set_xticks(exes)
    ax.set_xticklabels(exes, fontsize=8)
    ax.set_xlabel("example index")
    ax.set_ylabel("decode_query_count")
    ax.set_title("E5: decode_query_count per example (N = total_length − prompt_length captured)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cfg_color.values()] + ax.get_legend_handles_labels()[0]
    labels = list(cfg_color.keys()) + ax.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = OUT / "e5_decode_query_count_hist.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_per_layer_gap() -> Path:
    rows = load_rows(3)
    per_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_layer_gap_by_method: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] not in METHODS:
            continue
        per_layer_gap_by_method[r["method"]][r["layer"]].append((r["top1_prefill"], r["top1_decode"]))

    fig, ax = plt.subplots(figsize=(11, 5))
    for m in METHODS:
        layers = sorted(per_layer_gap_by_method[m].keys())
        gaps = []
        for l in layers:
            pairs = per_layer_gap_by_method[m][l]
            pref_mean = sum(p for p, d in pairs) / len(pairs)
            dec_mean = sum(d for p, d in pairs) / len(pairs)
            gaps.append(dec_mean - pref_mean)
        ax.plot(layers, gaps, marker="o", label=METHOD_LABELS[m], color=COLORS[m], linewidth=1.5, markersize=4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("decode top-1 − prefill top-1")
    ax.set_title("E5: per-layer decode−prefill gap at b_avg=3 (positive ⇒ decode easier)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT / "e5_per_layer_gap.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_per_example_decode() -> Path:
    rows = load_rows(3)
    by_ex_method: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cfg_by_ex: dict[int, str] = {}
    cnt_by_ex: dict[int, int] = {}
    for r in rows:
        if r["layer"] == 0:
            continue
        cfg_by_ex[r["example_index"]] = r["config"].removesuffix("_e")
        cnt_by_ex[r["example_index"]] = r["decode_query_count"]
        by_ex_method[r["example_index"]][r["method"]].append(r["top1_decode"])
    exes = sorted(by_ex_method)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(exes))
    for m in METHODS:
        ys = [statistics.mean(by_ex_method[ex][m]) for ex in exes]
        ax.plot(x, ys, marker="o", label=METHOD_LABELS[m], color=COLORS[m], linewidth=1.5, markersize=4)

    cfg_color = {"qasper": "#FFE4B5", "hotpotqa": "#E0FFE0", "passage_retrieval_en": "#E0E8FF"}
    band_starts: dict[str, tuple[int, int]] = {}
    for cfg in ("qasper", "hotpotqa", "passage_retrieval_en"):
        ids = [exes.index(e) for e in exes if cfg_by_ex[e] == cfg]
        if ids:
            band_starts[cfg] = (min(ids), max(ids))
    for cfg, (lo, hi) in band_starts.items():
        ax.axvspan(lo - 0.5, hi + 0.5, color=cfg_color[cfg], alpha=0.4, zorder=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.97, cfg, ha="center", va="top", fontsize=9, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n(dq={cnt_by_ex[e]})" for e in exes], fontsize=7)
    ax.set_xlabel("example index (decode_query_count below)")
    ax.set_ylabel("layer-0-excluded decode top-1")
    ax.set_title("E5: per-example decode top-1 by method (b_avg=3, layer-0 excluded)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=5, fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = OUT / "e5_per_example_decode.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_bit_budget_decode() -> Path:
    """Decode top-1 vs b_avg per method, vs prefill top-1 dashed."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHODS:
        decs = []
        prefs = []
        for b in B_AVG_GRID:
            rows = load_rows(b)
            stats = per_method_means(rows, "top1_prefill", "top1_decode")
            decs.append(stats[m]["dec_w"])
            prefs.append(stats[m]["pref"])
        ax.plot(B_AVG_GRID, decs, marker="o", color=COLORS[m], linewidth=1.5, label=METHOD_LABELS[m])
        ax.plot(B_AVG_GRID, prefs, marker="x", color=COLORS[m], linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(B_AVG_GRID)
    ax.set_xlabel("b_avg")
    ax.set_ylabel("layer-0-excluded top-1 retention")
    ax.set_title("E5: decode (solid, ●) vs prefill (dashed, ×) top-1 across bit budgets")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT / "e5_bit_budget_decode.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    paths = [
        chart_decode_vs_prefill_top1(),
        chart_decode_query_count_hist(),
        chart_per_layer_gap(),
        chart_per_example_decode(),
        chart_bit_budget_decode(),
    ]
    for p in paths:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
