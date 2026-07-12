#!/usr/bin/env python3
"""OSCAR's actual quantization recipe (arXiv:2605.17757, Zhou/Zhuang/Li/Chen/
Song/Athiwaratkun/Wu, Together AI/USyd/UIUC, May 2026), reimplemented as
method #7 for this repo's accuracy (attention-retention) and timing
(decode+attend) harnesses -- run on OUR real K/V data and calibration corpus,
for a fair, apples-to-apples number on OUR metrics, rather than pulling in
OSCAR's actual SGLang-coupled kernels or reproducing their downstream-task
accuracy metric (a different axis; see faithful-report audit notes).

Algorithm (Section 1 of the paper, cross-checked against their public repo's
`sglang-research/python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py`):

  R_K = U_Q . H_Had . P_br
  - U_Q: eigenbasis (descending eigenvalue) of the empirical UNCENTERED,
    GQA-pooled query second moment C_Q = (1/N) sum_n q_n q_n^T. This repo's
    `calib_moments()` already computes exactly this quantity as `sigma_q`.
  - H_Had: normalized Walsh-Hadamard transform.
  - P_br: bit-reversal permutation for balancing quantization groups smaller
    than d. NO-OP at OSCAR's own default group size G=d=128 (their Lemma 1),
    which is what we use here (single per-token scale/zero, matching their
    "scalar" / num_groups==1 kernel path) -- so R_K = U_Q @ H_Had.

  Quantization: per-token dynamic asymmetric affine INT2 (q_max=3): scale =
  (max-min)/3, zero = -min/scale, applied per row after per-token percentile
  clipping (clip_ratio default 0.96, matching their reported c_K). Mixed-
  precision cache: first `sink` tokens and last `recent` tokens stored exact
  (BF16 in their system); OSCAR's own ablation shows collapse to ~0 accuracy
  without this protection band, so it's on by default here, not optional.
"""
from __future__ import annotations

import math

import torch


def _sym(m):
    return 0.5 * (m + m.transpose(-1, -2))


def hadamard_matrix(n: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Normalized Walsh-Hadamard matrix, n a power of two."""
    assert (n & (n - 1)) == 0, f"Hadamard order must be a power of two, got {n}"
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def build_oscar_rotation(sigma_q: torch.Tensor) -> torch.Tensor:
    """sigma_q: (..., d, d) uncentered, GQA-pooled Q second moment (this repo's
    calib_moments() `sigma_q` return is already exactly this). Returns R_K:
    (..., d, d), forward_map convention `transformed = k @ R_K` (R_K orthogonal,
    so inverse_map = R_K.transpose(-1, -2)).

    P_br (bit-reversal permutation) is a no-op at G=d (this module's only
    supported group size, matching OSCAR's own default) -- see Lemma 1 in the
    paper: with one scale/zero per token covering the whole head_dim, there is
    only one "group" to balance, so any fixed permutation of channels is
    equivalent up to relabeling and does not change the quantizer's operating
    point. Included here as a no-op for documentation, not applied.
    """
    d = sigma_q.shape[-1]
    sq = _sym(sigma_q.to(torch.float64))
    lam, U = torch.linalg.eigh(sq)
    order = torch.argsort(lam, dim=-1, descending=True)
    U_Q = torch.gather(U, -1, order.unsqueeze(-2).expand(*U.shape[:-1], -1))
    H_had = hadamard_matrix(d, dtype=torch.float64, device=sigma_q.device)
    R_K = U_Q @ H_had
    return R_K


class OSCARCompressor:
    """OSCAR-style rotation + per-token dynamic INT2 + sink/recent protection.

    forward_map: (d, d) = R_K from build_oscar_rotation. clip_ratio: fraction
    of |rotated coord| kept before clamping to the percentile threshold
    (OSCAR's reported default c_K ~= 0.96). sink/recent: token counts kept
    exact (BF16 in a real system; float32 here, error is added in the
    quantized middle segment only, matching their design).
    """

    # OSCAR's Lloyd-Max INT2 (SGLANG_LLOYD_MAX=1 in the reference kernel
    # oscar_rotation_clip_int2_kv.py): standardize the clipped, rotated row by its
    # own mean/std, bucketize on the fixed Lloyd-Max boundaries, and reconstruct at
    # LM-aligned centroids via the uniform-equivalent (scale, zero). This is the
    # distribution-optimal 2-bit quantizer the paper's headline results use; the
    # min/max path the reference ships as `lloyd_max=False` is its own
    # "backward-compatible" fallback and wastes INT2 resolution on the two extreme
    # channels. Constants copied verbatim from the reference kernel.
    _LM_M = 0.9810652732849121                     # bucket boundary (+/-), 0 in between
    _LM_C0 = -1.5095585584640503
    _LM_SPAN = 2 * 1.5095585584640503
    _LM_RATIO = 1.16

    def __init__(self, forward_map: torch.Tensor, clip_ratio: float = 0.96,
                 sink: int = 64, recent: int = 256, lloyd_max: bool = True):
        self.forward_map = forward_map.float()
        self.inverse_map = forward_map.float().transpose(-1, -2)
        self.clip_ratio = clip_ratio
        self.sink = sink
        self.recent = recent
        self.lloyd_max = lloyd_max

    def to(self, device):
        self.forward_map = self.forward_map.to(device)
        self.inverse_map = self.inverse_map.to(device)
        return self

    def _split(self, T: int):
        s = min(self.sink, T)
        r = min(self.recent, max(T - s, 0))
        return s, T - s - r, r   # (n_sink, n_middle, n_recent)

    def _quantize_middle(self, mid: torch.Tensor):
        """mid: (M, d) -> (idx uint8 (M,d), scale (M,), zero (M,))."""
        r = mid @ self.forward_map
        if self.clip_ratio > 0 and self.clip_ratio < 1:
            d = r.shape[-1]
            k = max(1, min(d - 1, int(round(self.clip_ratio * d))))
            thr = torch.abs(r).sort(dim=-1).values[:, k]
            r = r.clamp(-thr.unsqueeze(-1), thr.unsqueeze(-1))
        if getattr(self, "lloyd_max", True):
            row_mean = r.mean(dim=-1)
            row_std = ((r - row_mean.unsqueeze(-1)).pow(2).mean(dim=-1) + 1e-8).sqrt()
            z = (r - row_mean.unsqueeze(-1)) / row_std.unsqueeze(-1)
            M = self._LM_M
            idx = ((z >= -M).to(torch.uint8) + (z >= 0).to(torch.uint8)
                   + (z >= M).to(torch.uint8))                     # {0,1,2,3}
            # uniform-equivalent (scale, zero): (idx - zero)*scale lands on the
            # LM-aligned centroids  mean + LM_RATIO*std*(idx*LM_SPAN/3 + LM_C0).
            scale = (self._LM_SPAN / 3.0) * self._LM_RATIO * row_std
            zero = (-self._LM_C0) / (self._LM_SPAN / 3.0) - row_mean / scale
            return idx, scale, zero
        rmin = r.min(dim=-1).values
        rmax = r.max(dim=-1).values
        scale = (rmax - rmin).clamp_min(1e-8) / 3.0
        zero = -rmin / scale
        idx = (r / scale.unsqueeze(-1) + zero.unsqueeze(-1) + 0.5).clamp(0, 3).to(torch.uint8)
        return idx, scale, zero

    def _dequantize_middle(self, idx, scale, zero):
        r_hat = (idx.float() - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
        return r_hat @ self.inverse_map

    def encode(self, k: torch.Tensor):
        """k: (T, d) -> compressed state (the stored representation)."""
        T = k.shape[0]
        s, m, r = self._split(T)
        sink_k = k[:s]
        mid_k = k[s:s + m]
        recent_k = k[s + m:]
        idx, scale, zero = self._quantize_middle(mid_k) if m > 0 else (None, None, None)
        return dict(sink=sink_k, recent=recent_k, idx=idx, scale=scale, zero=zero, T=T, s=s, m=m, r=r)

    def decode(self, state) -> torch.Tensor:
        """state from encode() -> k_hat: (T, d)."""
        if state["m"] > 0:
            mid_hat = self._dequantize_middle(state["idx"], state["scale"], state["zero"])
            return torch.cat([state["sink"], mid_hat, state["recent"]], dim=0)
        return torch.cat([state["sink"], state["recent"]], dim=0)

    def roundtrip(self, k: torch.Tensor) -> torch.Tensor:
        # sink/recent are along the sequence axis, so a leading batch dim (the kvpress
        # hook passes (B, S, d)) must be handled per-sequence, not flattened.
        # Compute in float32 (forward_map is float32; a bf16 K would else mismatch the
        # matmul, since unlike the QPCA paths OSCAR slices K without a float re-centering
        # that would promote it) and cast back to the model dtype.
        dt = k.dtype
        kf = k.float()
        if kf.dim() == 3:
            out = torch.stack([self.decode(self.encode(kf[b])) for b in range(kf.shape[0])], 0)
        else:
            out = self.decode(self.encode(kf))
        return out.to(dt)

    def bits_per_coord(self, T: int) -> float:
        """True BPE at sequence length T: 2-bit payload for the middle segment
        + BF16 sink/recent (16 bits/coord for those tokens) + BF16 scale/zero
        metadata (2 x 16 bits per middle token, amortized over d coords),
        matching OSCAR's own BPE accounting convention."""
        s, m, r = self._split(T)
        d = self.forward_map.shape[-1]
        payload_bits = m * d * 2 + (s + r) * d * 16
        meta_bits = m * 2 * 16
        return (payload_bits + meta_bits) / (T * d)
