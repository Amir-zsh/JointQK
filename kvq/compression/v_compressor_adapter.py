"""V-side compressor builders for the Phase 1 V method study.

Four V methods, parallel to the K-side (TurboQuant V3 / Q-Eigen / JointQK):
- v_random: random Hadamard rotation + uniform Lloyd–Max bits per coord (TurboQuant V).
- v_eigen_uniform: eigenvectors of Cov[V] (orthogonal) + uniform bits.
- v_eigen_waterfill: eigenvectors of Cov[V] + reverse water-fill on λ_j(Cov[V]).
- v_turboquant: TurboQuant V3's MSE-only value compressor.

For V the rate-distortion target is plain reconstruction MSE (no Q-weighting):
the attention output o = Σ_i α_i v_i has linear sensitivity in v_i. So the
weights for water-fill are the eigenvalues of Cov[V] — bits go to coords with
the largest *centered* variance.

Mean-centering: when `mu_v_for_head` is supplied, the compressor subtracts it
before quantization and adds it back after, so Lloyd-Max's zero-mean
unit-Gaussian centroids align with the actual data distribution. Without
centering, the eigenbasis is biased toward μ_v's direction (rank-1 offset),
and waterfill wastes bits there.

Usage:
    comp = build_v_compressor(method="v_eigen_waterfill",
                              cov_v_for_head=Cov[V[layer, kv_head]],
                              mu_v_for_head=μ_V[layer, kv_head],
                              head_dim=128, bits=2, seed=42)
    v_recon = comp.roundtrip(v_tensor)   # v_tensor: (..., 128)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from kvq.compression.per_coord import (
    PerCoordCompressor,
    round_bits_to_integer,
)

_TQ_DIR = Path(__file__).resolve().parents[2] / "vendor" / "turboquant-pytorch"
if str(_TQ_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_TQ_DIR.parent))

from turboquant_pytorch.compressors_v3 import MSECompressor  # type: ignore  # noqa: E402


class CenteredCompressor:
    """Wraps any roundtrip-capable inner compressor with mean centering.

    On compress: subtract `mu` from input, then call inner.roundtrip on the
    centered tensor. On output: add `mu` back. This makes a compressor whose
    codebooks were built for a zero-mean distribution work correctly on data
    with a real offset μ.
    """

    def __init__(self, inner, mu: torch.Tensor):
        self.inner = inner
        self.mu = mu  # (d,) in original (un-rotated) coordinates

    def to(self, device):
        self.inner.to(device)
        if self.mu.device != torch.device(device):
            self.mu = self.mu.to(device)
        return self

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        if self.mu.device != states.device:
            self.to(states.device)
        # mu is (d,) — broadcasts against states (..., d) without copying.
        centered = states - self.mu
        recon = self.inner.roundtrip(centered)
        return recon + self.mu


class TurboQuantValueCompressor:
    """Adapter exposing TurboQuant V3's value compressor as a roundtrip object.

    MSECompressor expects tensors shaped (B, H, S, D). The JointQK V path calls
    per-head compressors with tensors shaped (..., D), so this adapter flattens
    all leading dimensions to a single sequence axis and reshapes the result.
    """

    def __init__(self, head_dim: int, bits: int, seed: int = 42) -> None:
        self.d = int(head_dim)
        self.bits = int(bits)
        self._compressor = MSECompressor(self.d, self.bits, seed=seed, device="cpu")

    def to(self, device: str | torch.device) -> "TurboQuantValueCompressor":
        device = torch.device(device)
        if self._compressor.Pi.device != device:
            self._compressor.Pi = self._compressor.Pi.to(device)
            self._compressor.centroids = self._compressor.centroids.to(device)
            self._compressor.device = str(device)
        return self

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.d:
            raise ValueError(f"states last-dim {states.shape[-1]} != compressor.d {self.d}")
        if self._compressor.Pi.device != states.device:
            self.to(states.device)
        leading_shape = states.shape[:-1]
        flat = states.reshape(1, 1, -1, self.d)
        compressed = self._compressor.compress(flat)
        recon = self._compressor.decompress(compressed)
        return recon.reshape(leading_shape + (self.d,)).to(states.dtype)


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
    cov_v_for_head: torch.Tensor | None = None,
    head_dim: int = 0,
    bits: int = 0,
    seed: int = 42,
    mu_v_for_head: torch.Tensor | None = None,
    *,
    sigma_v_for_head: torch.Tensor | None = None,  # legacy alias, treated as cov_v
) -> PerCoordCompressor | "TurboQuantValueCompressor" | "CenteredCompressor":
    """Construct a V-side compressor for one (layer, kv_head).

    Args:
        method: one of "v_random", "v_eigen_uniform", "v_eigen_waterfill", "v_turboquant".
        cov_v_for_head: (d, d) centered covariance Cov[V] per (layer, kv_head).
        head_dim: dimensionality of v.
        bits: nominal bits per coordinate.
        seed: seed for v_random's rotation / for v_turboquant.
        mu_v_for_head: (d,) per-(layer, kv_head) mean μ_V. When provided the
            returned compressor subtracts μ before quantising and adds it back
            after — required for Lloyd-Max's zero-mean codebook to align with
            the data. Pass None for legacy (uncentered) behaviour.
        sigma_v_for_head: deprecated alias for cov_v_for_head; treated identically
            (callers passing the uncentered second moment must also provide
            mu_v_for_head=None and accept the legacy biased eigenbasis).

    Returns:
        A compressor with `.to(device)` and `.roundtrip(v) -> v_recon`.
    """
    # Backward-compat: callers that still pass `sigma_v_for_head` keep working.
    if cov_v_for_head is None and sigma_v_for_head is not None:
        cov_v_for_head = sigma_v_for_head

    if method == "v_turboquant":
        comp = TurboQuantValueCompressor(head_dim=head_dim, bits=bits, seed=seed)
        if mu_v_for_head is not None:
            comp = CenteredCompressor(comp, mu_v_for_head.float())
        return comp

    if cov_v_for_head is None:
        raise ValueError(f"{method} requires cov_v_for_head (or sigma_v_for_head)")

    if method == "v_random":
        forward_map = _random_hadamard_like(head_dim, seed)
    elif method.startswith("v_eigen_"):
        # eigenvectors of Cov[V], columns sorted by descending eigenvalue
        eigvals, eigvecs = torch.linalg.eigh(cov_v_for_head.float())
        idx = torch.argsort(eigvals, descending=True)
        forward_map = eigvecs[:, idx].float()
    else:
        raise ValueError(f"Unknown V method: {method!r}")

    inverse_map = forward_map.transpose(-1, -2)

    # Per-coord variance in the basis: diag(forward.T @ Cov[V] @ forward).
    # For Cov[V]'s eigenbasis this is exactly the eigenvalues; for random
    # rotation it's the rotated diagonal (positive for PSD Cov[V]).
    sv_in_basis = forward_map.transpose(-1, -2) @ cov_v_for_head.float() @ forward_map
    sigma_v_diag = sv_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)

    total_bits = int(bits) * head_dim

    if method.endswith("_random") or method.endswith("_uniform"):
        # Uniform allocation: every coord gets `bits`.
        bits_int = torch.full((head_dim,), int(bits), dtype=torch.long)
    elif method.endswith("_waterfill"):
        from kvq.compression.metric_transform import water_fill
        # For V, the rate-distortion target is plain MSE: D = Σ_j σ²_j(V) · 2^{-2 b_j}.
        # The water-fill solver expects per-coord "weight × variance"; since the
        # weight here is uniform-1 (no Q-weighting), we pass sigma_v_diag itself.
        wf_input = sigma_v_diag.unsqueeze(0)
        bits_continuous = water_fill(wf_input, total_bits=total_bits).squeeze(0)
        bits_int = round_bits_to_integer(
            bits_continuous.unsqueeze(0), total_bits=total_bits
        ).squeeze(0)
        # Match the K-side cap (per_coord_quantization.build_jointqk_compressor):
        # unit_gaussian_centroids(b) is a Lloyd–Max solve with 2^b levels, so
        # uncapped saturated allocations (water_fill defaults to max_bits=16)
        # would mean 65,536-level codebooks per coord. Cap at 8 and redistribute.
        MAX_BITS = 8
        excess = (bits_int - MAX_BITS).clamp_min(0).sum().item()
        bits_int = bits_int.clamp_max(MAX_BITS)
        while excess > 0:
            below = (bits_int < MAX_BITS).nonzero(as_tuple=True)[0]
            if below.numel() == 0:
                break
            min_val = bits_int[below].min()
            candidates = below[bits_int[below] == min_val]
            take = min(int(candidates.numel()), int(excess))
            bits_int[candidates[:take]] += 1
            excess -= take
    else:
        raise ValueError(f"Unknown allocation in V method: {method!r}")

    coord_stds = torch.sqrt(sigma_v_diag.float())
    inner = PerCoordCompressor(
        bits_per_coord=bits_int,
        std_per_coord=coord_stds,
        forward_map=forward_map.float(),
        inverse_map=inverse_map.float(),
    )
    if mu_v_for_head is not None:
        return CenteredCompressor(inner, mu_v_for_head.float())
    return inner


V_METHOD_NAMES = ("v_random", "v_eigen_uniform", "v_eigen_waterfill", "v_turboquant")
