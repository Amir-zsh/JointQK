#!/usr/bin/env python3
"""Calibrate V statistics on a 24-bundle: per-(layer, kv_head) mean μ_V and
centered covariance Cov[V] on prefill tokens.

Saved fields (v_stats.pt):
  - cov_v: (n_layers, n_kv_heads, d, d), centered Cov[V] = E[v v^T] - μ_v μ_v^T
  - mu_v:  (n_layers, n_kv_heads, d),    E[v]
  - sigma_v: (n_layers, n_kv_heads, d, d), uncentered E[v v^T]  (kept for
    backward compat — old consumers fall back to this if mu_v / cov_v missing)

The earlier version saved only `sigma_v` (uncentered), which biased the
eigenbasis toward μ_v's direction. Mean-centering moves Lloyd-Max's zero-mean
unit-Gaussian centroids onto the actual data distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

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

    sum_outer = None     # Σ_t v_t v_t^T per (layer, kv_head), shape (L, H, D, D)
    sum_v = None          # Σ_t v_t per (layer, kv_head), shape (L, H, D)
    total_count = 0

    for i, ex_file in enumerate(example_files):
        b = torch.load(ex_file, map_location="cpu", weights_only=False)
        v = b["v"].float()  # (n_layers, n_kv_heads, seq_len, head_dim)
        prompt_length = int(b["prompt_length"])
        if args.prefill_only:
            v = v[:, :, :prompt_length, :]
        outer = torch.einsum("lhsd,lhse->lhde", v, v)
        sum_v_i = v.sum(dim=2)  # (L, H, D)
        if sum_outer is None:
            sum_outer = outer.clone()
            sum_v = sum_v_i.clone()
        else:
            sum_outer += outer
            sum_v += sum_v_i
        total_count += v.shape[2]
        print(f"  [{i+1}/{len(example_files)}] {ex_file.name}: {v.shape[2]} tokens (cum {total_count})")

    sigma_v = sum_outer / total_count                            # E[v v^T] (uncentered)
    mu_v = sum_v / total_count                                   # E[v]
    cov_v = sigma_v - torch.einsum("lhd,lhe->lhde", mu_v, mu_v)  # centered Cov[v]
    n_layers, n_kv_heads, d, _ = sigma_v.shape

    # PSD sanity on both moments.
    eigs_sigma = torch.linalg.eigvalsh(sigma_v)
    eigs_cov = torch.linalg.eigvalsh(cov_v)
    print(f"sigma_v (uncentered) shape: {tuple(sigma_v.shape)}")
    print(f"  uncentered eigenvalue range: [{eigs_sigma.min().item():.4e}, {eigs_sigma.max().item():.4e}]")
    print(f"  centered Cov[v] eigenvalue range: [{eigs_cov.min().item():.4e}, {eigs_cov.max().item():.4e}]")
    print(f"  ‖μ_v‖² stats per (L,H): min={(mu_v.norm(dim=-1) ** 2).min().item():.3e}, "
          f"median={(mu_v.norm(dim=-1) ** 2).median().item():.3e}, "
          f"max={(mu_v.norm(dim=-1) ** 2).max().item():.3e}")
    if eigs_cov.min().item() < -1e-3:
        print(f"WARNING: centered Cov[v] min eigenvalue is negative ({eigs_cov.min().item():.4e}); numerical issue?")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cov_v": cov_v,             # centered — preferred for new consumers
        "mu_v": mu_v,
        "sigma_v": sigma_v,         # uncentered — kept for backward compat
        "metadata": {
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": d,
            "n_examples": len(example_files),
            "total_token_count": total_count,
            "prefill_only": args.prefill_only,
            "bundle": str(bundle),
            "uncentered_min_eigenvalue": eigs_sigma.min().item(),
            "uncentered_max_eigenvalue": eigs_sigma.max().item(),
            "centered_min_eigenvalue": eigs_cov.min().item(),
            "centered_max_eigenvalue": eigs_cov.max().item(),
            "version": 2,           # bumped from implicit v1 (sigma_v only)
        },
    }, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
