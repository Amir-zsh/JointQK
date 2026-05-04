"""KIVI K/V quantization routines.

KIVI baseline (Liu et al. 2024): per-channel asymmetric int4 quantization for K
(group_size=128 along channel axis) and per-token asymmetric int4 for V
(group_size=128 along token axis).

Both directions use the standard min-max asymmetric scheme:
    scale = (max - min) / (2^bits - 1)
    zero  = -round(min / scale)
    q     = clamp(round(x/scale) + zero, 0, 2^bits-1)
    x_hat = (q - zero) * scale

For K (per-channel): one (scale, zero) per (batch, head, channel) — each channel
of the head_dim axis quantized independently across the seq_len.

For V (per-token): one (scale, zero) per (batch, head, token) — each token's V
vector quantized as a single group across head_dim.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def kivi_quantize_keys(keys: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """Per-channel asymmetric int{bits} quantization of K, returns reconstruction.

    keys: (B, H, S, D). Per-(B, H, channel) min/max along seq dim.
    group_size: along the channel (D) axis. With group_size=D this collapses to
    one quantizer per channel.
    """
    B, H, S, D = keys.shape
    if D % group_size != 0:
        raise ValueError(f"head_dim={D} not divisible by group_size={group_size}")
    n_groups = D // group_size
    qmax = (1 << bits) - 1

    # Reshape: (B, H, S, n_groups, group_size)
    k = keys.reshape(B, H, S, n_groups, group_size).float()

    # min/max per (B, H, group_member, group), reduced over S → shape (B, H, 1, n_groups, group_size)
    kmin = k.min(dim=2, keepdim=True).values
    kmax = k.max(dim=2, keepdim=True).values
    scale = (kmax - kmin).clamp_min(1e-8) / qmax
    zero = (-kmin / scale).round().clamp(0, qmax)

    q = ((k / scale) + zero).round().clamp(0, qmax)
    recon = (q - zero) * scale

    return recon.reshape(B, H, S, D).to(keys.dtype)


@torch.no_grad()
def kivi_quantize_values(values: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """Per-token asymmetric int{bits} quantization of V, returns reconstruction.

    values: (B, H, S, D). Per-(B, H, S, group) min/max over the head_dim axis.
    group_size: along the head_dim (D) axis. With group_size=D, each token's full
    V vector is one group.
    """
    B, H, S, D = values.shape
    if D % group_size != 0:
        raise ValueError(f"head_dim={D} not divisible by group_size={group_size}")
    n_groups = D // group_size
    qmax = (1 << bits) - 1

    v = values.reshape(B, H, S, n_groups, group_size).float()
    vmin = v.min(dim=-1, keepdim=True).values
    vmax = v.max(dim=-1, keepdim=True).values
    scale = (vmax - vmin).clamp_min(1e-8) / qmax
    zero = (-vmin / scale).round().clamp(0, qmax)

    q = ((v / scale) + zero).round().clamp(0, qmax)
    recon = (q - zero) * scale

    return recon.reshape(B, H, S, D).to(values.dtype)
