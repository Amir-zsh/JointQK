#!/usr/bin/env python3
"""
Plot histograms of key activations for selected layers.

Loads artifacts with:
  art["k_post"]      # [L, Hkv, T, d]
  art["prompt_length"]

For each selected layer, aggregates all key coordinates across:
  - examples
  - heads
  - tokens up to prompt_length
  - feature dimension

Optionally centers keys token-wise before plotting.

Example:
  python key_histogram_layers.py \
      --idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
      --layers 0 8 16 24 \
      --centered \
      --bins 200
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FULL_DIR = "/mnt/c/JointQK_data/query_stats_longbench_under4k"


def _ensure_environment():
    needed = ("numpy", "torch", "matplotlib")
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"Missing deps: {missing}. Activate the right environment first.")


_ensure_environment()

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for cand in (Path.cwd().resolve(), Path.cwd().resolve().parent, SCRIPT_DIR, SCRIPT_DIR.parent):
    s = str(cand)
    if s not in sys.path:
        sys.path.insert(0, s)

import _bootstrap  # noqa: E402,F401
REPO = Path(_bootstrap.__file__).resolve().parent


def data_root():
    r = REPO / "notebooks" / "data" / FULL_DIR
    if not (r / "manifest.json").exists():
        sys.exit(f"Dataset not found at {r} (expected manifest.json).")
    return r


def load_manifest(root):
    return json.loads((root / "manifest.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--idx",
        type=int,
        nargs="+",
        required=True,
        help="Example indices to evaluate, e.g. 0 1 2 ...",
    )
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layer indices to plot, e.g. 0 8 16 24",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Number of histogram bins",
    )
    ap.add_argument(
        "--centered",
        action="store_true",
        help="Center keys token-wise before plotting",
    )
    ap.add_argument(
        "--x-percentiles",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional x-axis clipping percentiles, e.g. 0.5 99.5",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="key_histograms.png",
        help="Output PNG filename",
    )
    ap.add_argument(
        "--sharex",
        action="store_true",
        help="Share x-axis across all layers",
    )
    args = ap.parse_args()

    root = data_root()
    manifest = load_manifest(root)
    exs = manifest["examples"]
    nex = len(exs)

    for i in args.idx:
        if i < 0 or i >= nex:
            sys.exit(f"Index {i} out of range for dataset of size {nex}.")

    print(f"Dataset size: {nex}")
    print(f"Selected examples: {args.idx}")
    print(f"Selected layers: {args.layers}")
    print(f"Mode: {'centered keys' if args.centered else 'raw keys'}")

    layer_values = {l: [] for l in args.layers}
    layer_meta = {l: {"count": 0} for l in args.layers}

    # Optional sanity check on layer indices using the first artifact
    first_art = torch.load(root / exs[args.idx[0]]["file"], map_location="cpu", weights_only=False)
    n_layers = int(first_art["k_post"].shape[0])
    for l in args.layers:
        if l < 0 or l >= n_layers:
            sys.exit(f"Layer {l} out of range for model with {n_layers} layers.")

    for rank, i in enumerate(args.idx, 1):
        art = torch.load(root / exs[i]["file"], map_location="cpu", weights_only=False)
        T = int(art["prompt_length"])

        k = art["k_post"][:, :, :T, :].float()  # [L, Hkv, T, d]

        if args.centered:
            k = k - k.mean(dim=2, keepdim=True)

        for l in args.layers:
            vals = k[l].reshape(-1).cpu().numpy()
            layer_values[l].append(vals)
            layer_meta[l]["count"] += vals.size

        print(f"  [{rank:02d}/{len(args.idx):02d}] example {i}: T={T}")

    for l in args.layers:
        layer_values[l] = np.concatenate(layer_values[l]).astype(np.float64)

    # Compute stats and optional common x-limits
    stats = {}
    all_vals = []
    for l in args.layers:
        vals = layer_values[l]
        stats[l] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(vals.size),
        }
        all_vals.append(vals)

    if args.x_percentiles is not None:
        low_p, high_p = args.x_percentiles
        if not (0.0 <= low_p < high_p <= 100.0):
            sys.exit("--x-percentiles must satisfy 0 <= LOW < HIGH <= 100")
        concat_vals = np.concatenate(all_vals)
        x_low = float(np.percentile(concat_vals, low_p))
        x_high = float(np.percentile(concat_vals, high_p))
    else:
        x_low = x_high = None

    n = len(args.layers)
    fig_w = max(12, 4 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3), squeeze=False, sharex=args.sharex)
    axes = axes.ravel()

    for ax, l in zip(axes, args.layers):
        vals = layer_values[l]
        s = stats[l]

        ax.hist(vals, bins=args.bins, density=True)
        ax.set_title(
            f"Layer {l} | min={s['min']:.1f} | max={s['max']:.1f}"
        )
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)

        if x_low is not None and x_high is not None:
            ax.set_xlim(x_low, x_high)

    axes[-1].set_xlabel("key value")
    fig.suptitle("Centered keys" if args.centered else "Raw key activations", y=0.995)
    fig.tight_layout()

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nPer-layer summary:")
    for l in args.layers:
        s = stats[l]
        print(
            f"  layer {l:>3}: n={s['n']:,}  mean={s['mean']:.6f}  "
            f"std={s['std']:.6f}  min={s['min']:.6f}  max={s['max']:.6f}"
        )

    print(f"\nSaved plot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()  