"""V-side compressor builders for the Phase 1 V method study.

Three V methods, parallel to the K-side (TurboQuant V3 / Q-Eigen / JointQK):
- v_random: random Hadamard rotation + uniform Lloyd–Max bits per coord (TurboQuant V).
- v_eigen_uniform: eigenvectors of Σ_V (orthogonal) + uniform bits.
- v_eigen_waterfill: eigenvectors of Σ_V + reverse water-fill on λ_j(V).

For V the rate-distortion target is plain reconstruction MSE (no Q-weighting),
because the attention output o = Σ_i α_i v_i has linear sensitivity in v_i. So the
weights for water-fill are just the eigenvalues of Σ_V — bits go to coords with
the largest variance.

All three methods produce orthogonal forward maps; inverse = transpose.

Usage:
    from experiments.stage1.toolkit.v_compressor_adapter import build_v_compressor
    comp = build_v_compressor(method="v_eigen_waterfill",
                              sigma_v_for_head=Σ_V[layer, kv_head],
                              head_dim=128, bits=2, seed=42)
    v_recon = comp.roundtrip(v_tensor)   # v_tensor: (..., 128)
"""
from __future__ import annotations

import torch

from experiments.stage1.toolkit.per_coord_quantization import (
    PerCoordCompressor,
    round_bits_to_integer,
)


def _random_hadamard_like(d: int, seed: int) -> torch.Tensor:
    """Random orthogonal matrix (d, d) seeded for reproducibility.

    We use a random rotation derived from QR of a Gaussian matrix. This is
    functionally equivalent to TurboQuant's "random rotation" baseline; we use
    QR rather than a Walsh–Hadamard product because the latter only exists for
    d that is a power of 2 (head_dim = 128 satisfies that, but QR generalizes).
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.float()


def build_v_compressor(
    method: str,
    sigma_v_for_head: torch.Tensor,
    head_dim: int,
    bits: int,
    seed: int = 42,
) -> PerCoordCompressor:
    """Construct a PerCoordCompressor for V-side quantization.

    Args:
        method: one of "v_random", "v_eigen_uniform", "v_eigen_waterfill".
        sigma_v_for_head: (head_dim, head_dim) per-(layer, kv_head) Σ_V.
        head_dim: dimensionality.
        bits: nominal bits per coordinate (will be enforced via uniform alloc
              for v_random / v_eigen_uniform; via water-fill for v_eigen_waterfill).
        seed: seed for v_random.

    Returns:
        PerCoordCompressor with forward_map=basis, inverse_map=basis.T,
        std_per_coord=sqrt(diag of Σ_V in the basis).
    """
    if method == "v_random":
        forward_map = _random_hadamard_like(head_dim, seed)
    elif method.startswith("v_eigen_"):
        # eigenvectors of Σ_V, columns sorted by descending eigenvalue
        eigvals, eigvecs = torch.linalg.eigh(sigma_v_for_head.float())
        # eigh returns ascending; reverse to descending
        idx = torch.argsort(eigvals, descending=True)
        forward_map = eigvecs[:, idx].float()
    else:
        raise ValueError(f"Unknown V method: {method!r}")

    inverse_map = forward_map.transpose(-1, -2)

    # Per-coord variance in the basis: diag(forward.T @ Σ_V @ forward).
    # For eigenbasis this is exactly the eigenvalues; for random rotation it's
    # the rotated diagonal (still positive).
    sv_in_basis = forward_map.transpose(-1, -2) @ sigma_v_for_head.float() @ forward_map
    sigma_v_diag = sv_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)

    total_bits = int(bits) * head_dim

    if method.endswith("_random") or method.endswith("_uniform"):
        # Uniform allocation: every coord gets `bits`.
        bits_int = torch.full((head_dim,), int(bits), dtype=torch.long)
    elif method.endswith("_waterfill"):
        from experiments.stage1.toolkit.metric_transform import water_fill
        # For V, the rate-distortion target is plain MSE: D = Σ_j σ²_j(V) · 2^{-2 b_j}.
        # The water-fill solver expects per-coord "weight × variance"; since the
        # weight here is uniform-1 (no Q-weighting), we pass sigma_v_diag itself.
        wf_input = sigma_v_diag.unsqueeze(0)
        bits_continuous = water_fill(wf_input, total_bits=total_bits).squeeze(0)
        bits_int = round_bits_to_integer(
            bits_continuous.unsqueeze(0), total_bits=total_bits
        ).squeeze(0)
    else:
        raise ValueError(f"Unknown allocation in V method: {method!r}")

    coord_stds = torch.sqrt(sigma_v_diag.float())
    return PerCoordCompressor(
        bits_per_coord=bits_int,
        std_per_coord=coord_stds,
        forward_map=forward_map.float(),
        inverse_map=inverse_map.float(),
    )


V_METHOD_NAMES = ("v_random", "v_eigen_uniform", "v_eigen_waterfill")
