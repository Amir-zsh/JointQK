"""KIVI K/V quantization routines.

This module mirrors the official jy-yuan/KIVI quantization math while keeping
the local experiment API simple: each function returns the reconstructed tensor
rather than packed integer codes.

Official KIVI uses min-subtraction asymmetric quantization:
    scale = (max - min) / (2^bits - 1)
    q     = clamp(round((x - min) / scale), 0, 2^bits-1)
    x_hat = q * scale + min

For K, `group_size` is along the sequence axis. Each token block gets a separate
per-channel quantizer, matching KIVI's per-channel K-cache quantization. If the
sequence length is not divisible by `group_size`, the divisible prefix is
quantized and the tail is left full precision, matching KIVI's residual-cache
semantics.

For V, `group_size` is along the head_dim axis. Each token gets one or more
groups across its value vector, matching KIVI's per-token V-cache quantization.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def kivi_quantize_keys(keys: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """Per-channel asymmetric int{bits} quantization of K, returns reconstruction.

    keys: (B, H, S, D). KIVI groups along S and reduces over each token block,
    producing one min/max pair per (B, H, sequence_group, channel).
    """
    B, H, S, D = keys.shape
    if bits not in (2, 4, 8):
        raise ValueError(f"KIVI supports bits in {{2, 4, 8}}, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    quantized_len = (S // group_size) * group_size
    if quantized_len == 0:
        return keys

    prefix = keys[:, :, :quantized_len, :].contiguous().to(torch.float16)
    n_groups = quantized_len // group_size
    qmax = (1 << bits) - 1

    # Official KIVI shape: (B, H, num_seq_groups, group_size, D).
    k = prefix.view(B, H, n_groups, group_size, D)
    kmin = k.min(dim=-2, keepdim=True).values
    kmax = k.max(dim=-2, keepdim=True).values
    scale = (kmax - kmin) / qmax

    q = ((k - kmin) / scale).clamp(0, qmax).round().to(torch.int32)
    recon_prefix = (q.to(torch.float16) * scale + kmin).view(B, H, quantized_len, D)

    if quantized_len == S:
        return recon_prefix.to(keys.dtype)

    return torch.cat([recon_prefix.to(keys.dtype), keys[:, :, quantized_len:, :]], dim=2)


@torch.no_grad()
def kivi_quantize_values(values: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """Per-token asymmetric int{bits} quantization of V, returns reconstruction.

    values: (B, H, S, D). KIVI groups along D and reduces over each value-vector
    group, producing one min/max pair per (B, H, S, head_dim_group).
    """
    B, H, S, D = values.shape
    if bits not in (2, 4, 8):
        raise ValueError(f"KIVI supports bits in {{2, 4, 8}}, got {bits}")
    if D % group_size != 0:
        raise ValueError(f"head_dim={D} not divisible by group_size={group_size}")
    n_groups = D // group_size
    qmax = (1 << bits) - 1

    v = values.contiguous().to(torch.float16).view(B, H, S, n_groups, group_size)
    vmin = v.min(dim=-1, keepdim=True).values
    vmax = v.max(dim=-1, keepdim=True).values
    scale = (vmax - vmin) / qmax

    q = ((v - vmin) / scale).clamp(0, qmax).round().to(torch.int32)
    recon = q.to(torch.float16) * scale + vmin

    return recon.view(B, H, S, D).to(values.dtype)
