#!/usr/bin/env python3
"""Pre-derive R_sym and V_h bases into an existing cca_stats.pt.

Loads sigma_q/sigma_k/P_K from cca_stats.pt, runs the same _derive_vh_rsym used
inside run_cca_vs_waterfill_study.py, and writes V_h + R_sym back into the file.
Idempotent — running twice overwrites previous keys.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.stage1.run_cca_vs_waterfill_study import _derive_vh_rsym


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cca-stats", required=True, help="Path to cca_stats.pt to update in place")
    parser.add_argument("--eps", type=float, default=1e-6, help="Whitening regularization")
    args = parser.parse_args()

    path = Path(args.cca_stats)
    print(f"Loading {path}")
    stats = torch.load(path, map_location="cpu", weights_only=False)

    needed = ["sigma_q", "sigma_k", "P_K"]
    for k in needed:
        if k not in stats:
            raise SystemExit(f"cca_stats.pt missing key {k!r}; cannot derive newbases")

    print(f"Deriving V_h + R_sym (eps={args.eps}, shape={tuple(stats['sigma_q'].shape)})")
    V_h, R_sym = _derive_vh_rsym(
        sigma_q=stats["sigma_q"],
        sigma_k=stats["sigma_k"],
        P_K=stats["P_K"],
        eps=args.eps,
    )
    print(f"  V_h shape: {tuple(V_h.shape)}")
    print(f"  R_sym shape: {tuple(R_sym.shape)}")

    stats["V_h"] = V_h
    stats["R_sym"] = R_sym

    torch.save(stats, path)
    print(f"Wrote V_h + R_sym back into {path}")


if __name__ == "__main__":
    main()
