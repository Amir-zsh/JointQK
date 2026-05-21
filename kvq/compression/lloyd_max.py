from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


_LLOYD_MAX_CACHE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _gaussian_approx_pdf(x: float, d: int) -> float:
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-(x * x) / (2 * sigma2))


def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (d, bits)
    cached = _LLOYD_MAX_CACHE.get(cache_key)
    if cached is not None:
        centroids, boundaries = cached
        return centroids.clone(), boundaries.clone()

    from scipy import integrate

    n_levels = 2**bits
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        updated = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _ = integrate.quad(lambda x: x * _gaussian_approx_pdf(x, d), a, b)
            denominator, _ = integrate.quad(lambda x: _gaussian_approx_pdf(x, d), a, b)
            updated.append(numerator / denominator if denominator > 1e-15 else centroids[i])
        max_shift = max(abs(updated[i] - centroids[i]) for i in range(n_levels))
        centroids = updated
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32)
    boundaries_tensor = torch.tensor(boundaries, dtype=torch.float32)
    _LLOYD_MAX_CACHE[cache_key] = (centroids_tensor.clone(), boundaries_tensor.clone())
    return centroids_tensor, boundaries_tensor


def generate_rotation_matrix(d: int, seed: int | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    gaussian = torch.randn(d, d, generator=generator)
    q, r = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(0)
    return q.to(device)


class Stage1MSECompressor:
    """
    V3-equivalent single-stage MSE compressor used for the stage-1 study.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str | torch.device = "cpu") -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.device = str(device)
        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.centroids, _ = solve_lloyd_max(head_dim, bits)
        self.centroids = self.centroids.to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict[str, Any]:
        bsz, heads, seq_len, dim = states.shape
        flat = states.reshape(-1, dim).float()
        vec_norms = torch.norm(flat, dim=-1)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        indices_per_byte = 8 // self.bits
        idx_pad = (indices_per_byte - dim % indices_per_byte) % indices_per_byte
        idx_flat = indices.long()
        if idx_pad:
            idx_flat = F.pad(idx_flat, (0, idx_pad))
        n_groups = idx_flat.shape[-1] // indices_per_byte
        idx_powers = torch.tensor(
            [2 ** (self.bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_flat.device,
        )
        idx_bytes = (idx_flat.reshape(flat.shape[0], n_groups, indices_per_byte) * idx_powers).sum(-1).to(torch.uint8)
        return {
            "idx_bytes": idx_bytes.reshape(bsz, heads, seq_len, n_groups),
            "vec_norms": vec_norms.to(torch.float16).reshape(bsz, heads, seq_len),
            "shape": (bsz, heads, seq_len, dim),
            "idx_pad": idx_pad,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict[str, Any]) -> torch.Tensor:
        bsz, heads, seq_len, dim = compressed["shape"]
        idx_bytes = compressed["idx_bytes"].reshape(-1, compressed["idx_bytes"].shape[-1])
        vec_norms = compressed["vec_norms"].reshape(-1, 1).float()
        idx_pad = compressed["idx_pad"]

        indices_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        idx_shifts = torch.tensor(
            [self.bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_bytes.device,
        )
        indices = ((idx_bytes.long().unsqueeze(-1) >> idx_shifts) & mask).reshape(-1, indices_per_byte * idx_bytes.shape[-1])
        if idx_pad:
            indices = indices[:, :dim]

        reconstructed = (self.centroids[indices] @ self.Pi) * vec_norms
        return reconstructed.reshape(bsz, heads, seq_len, dim)

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        return self.decompress(self.compress(states))
