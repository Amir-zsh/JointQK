#!/usr/bin/env python3
"""Emit a simulated-quantization bundle for TurboQuant, for the serving stack to apply.

Why a bundle instead of reimplementing the quantizer inside the sglang fork: the served arm must be
the SAME quantizer the kvpress harness uses, or the number means nothing. Rather than write it twice
and hope they agree, the rotations and Lloyd-Max centroids are computed HERE with the vendored
TurboQuant code (`turboquant_pytorch.compressors_v3.MSECompressor`) and shipped as tensors. The fork
then only has to normalise, rotate, pick nearest centroid, and undo -- no constants of its own.

This exists because the accuracy tasks (GPQA / HumanEval / AIME25) are generation-dominated: AIME25's
prompt is ~72 words against ~32K generated tokens, so the kvpress harness (prefill-only compression)
would compress ~0.3% of the KV and report BF16 for every method. The serving stack compresses decode
KV natively but has no 4-bit path at all. Simulating the quantizer in the BF16 write path gives the
exact accuracy of a real 4-bit pool without a new pool dtype, new unpack kernels, or touching the
fused Hadamard writer -- see notes/scope_4bit_k_pool.md (option B).

Rate note: K=4/V=2 is 3.125 BPE (K 128*4+16 = 528 b, V 128*2+16 = 272 b, 800/256). That is
TurboQuant's own recommended asymmetry and satisfies OSCAR's b_K + b_V = 6 for their 3.25 row (their
formula over-charges by 0.125 -- TurboQuant stores one fp16 norm per vector, not a scale+zero per
group). Simulation stores BF16 and saves no memory; only accuracy transfers, which is the point.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vendor"))

from turboquant_pytorch.compressors_v3 import MSECompressor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--k-bits", type=int, default=4)
    ap.add_argument("--v-bits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42, help="matches TurboQuantPress default")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = {
        "head_dim": args.head_dim,
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "seed": args.seed,
        "n_layers": args.layers,
        # Per-layer because MSECompressor's seed is seed + layer_idx*1000 (K) and +500 (V); one
        # rotation per layer, SHARED across kv heads, exactly as the press builds it.
        "k_rotation": torch.empty(args.layers, args.head_dim, args.head_dim),
        "v_rotation": torch.empty(args.layers, args.head_dim, args.head_dim),
        "note": "simulated (fake) quant: dequantized BF16 is stored; no memory saving by design",
    }

    for L in range(args.layers):
        seed_base = args.seed + L * 1000
        kc = MSECompressor(args.head_dim, args.k_bits, seed=seed_base, device="cpu")
        vc = MSECompressor(args.head_dim, args.v_bits, seed=seed_base + 500, device="cpu")
        bundle["k_rotation"][L] = kc.Pi
        bundle["v_rotation"][L] = vc.Pi
        if L == 0:
            # Centroids depend only on (head_dim, bits), so one copy each.
            bundle["k_centroids"] = kc.centroids.clone()
            bundle["v_centroids"] = vc.centroids.clone()
        print(f"  layer {L:2d}: K seed={seed_base} V seed={seed_base + 500}", flush=True)

    # Sanity: orthogonality is what makes (x @ Pi.T) @ Pi an inverse. A non-orthogonal rotation
    # would silently inflate error at every bit-width.
    for name in ("k_rotation", "v_rotation"):
        Pi = bundle[name][0]
        err = (Pi.T @ Pi - torch.eye(args.head_dim)).abs().max().item()
        assert err < 1e-4, f"{name} not orthogonal: {err:.2e}"
    print(f"orthogonality OK | K centroids {bundle['k_centroids'].numel()} levels, "
          f"V {bundle['v_centroids'].numel()} levels")

    kbpe = (args.head_dim * args.k_bits + 16) / args.head_dim
    vbpe = (args.head_dim * args.v_bits + 16) / args.head_dim
    print(f"nominal BPE if actually packed: K {kbpe:.3f} + V {vbpe:.3f} -> {(kbpe + vbpe) / 2:.3f}")

    torch.save(bundle, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
