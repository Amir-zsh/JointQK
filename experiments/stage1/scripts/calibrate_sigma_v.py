#!/usr/bin/env python3
"""Calibrate Σ_V on a 24-bundle: per-(layer, kv_head) second-moment of V on prefill tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Path to bundle dir (containing manifest.json + examples/)")
    parser.add_argument("--output", required=True, help="Output .pt path; saves dict with key 'sigma_v'")
    parser.add_argument("--prefill-only", action="store_true", default=True,
                        help="Average over prefill tokens only (default True)")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    examples_dir = bundle / "examples"
    if not examples_dir.exists():
        raise SystemExit(f"No examples/ directory under {bundle}")
    example_files = sorted(examples_dir.glob("*.pt"))
    if not example_files:
        raise SystemExit(f"No .pt files in {examples_dir}")

    print(f"Found {len(example_files)} example files in {examples_dir}")

    sum_outer = None  # (n_layers, n_kv_heads, head_dim, head_dim)
    total_count = 0

    for i, ex_file in enumerate(example_files):
        b = torch.load(ex_file, map_location="cpu", weights_only=False)
        v = b["v"].float()  # (n_layers, n_kv_heads, seq_len, head_dim)
        prompt_length = int(b["prompt_length"])
        if args.prefill_only:
            v = v[:, :, :prompt_length, :]
        # Per-(layer, kv_head) outer product accumulated over tokens
        outer = torch.einsum("lhsd,lhse->lhde", v, v)
        if sum_outer is None:
            sum_outer = outer.clone()
        else:
            sum_outer += outer
        total_count += v.shape[2]
        print(f"  [{i+1}/{len(example_files)}] {ex_file.name}: {v.shape[2]} tokens (cum {total_count})")

    sigma_v = sum_outer / total_count  # (n_layers, n_kv_heads, head_dim, head_dim)
    n_layers, n_kv_heads, d, _ = sigma_v.shape

    # Sanity: PSD-ness via min eigenvalue
    eigs = torch.linalg.eigvalsh(sigma_v)  # (n_layers, n_kv_heads, head_dim)
    min_eig = eigs.min().item()
    max_eig = eigs.max().item()
    print(f"sigma_v shape: {tuple(sigma_v.shape)}")
    print(f"eigenvalue range: [{min_eig:.4e}, {max_eig:.4e}]")
    if min_eig < -1e-3:
        print(f"WARNING: min eigenvalue is negative ({min_eig:.4e}); numerical issue?")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sigma_v": sigma_v,
        "metadata": {
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": d,
            "n_examples": len(example_files),
            "total_token_count": total_count,
            "prefill_only": args.prefill_only,
            "bundle": str(bundle),
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
        },
    }, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
