"""pgq3 family (f): OSCAR-emulation baseline (published-SOTA anchor).

Single-rung, per-token quantization in the given transform domain (the fit
supplies OSCAR's own Hadamard rotation with mu = 0, NOT qpca_unc — this arm
reproduces the published method on our protocol, it does not adopt our
basis): clip each token's coords at its clip_q-quantile of |r|, then
per-(token, group) asymmetric min-max INT-w with fp16 scale + fp16 zero
point, both charged against the rate (32 bits per group per token). Codec
honesty: reconstruction uses the fp16-ROUNDED scale/zero the decoder reads.

Faithful-emulation notes: no exact zero level is guaranteed (min-max grids
span [min, max] as published), and sinks get no special rung — per-token
scales already isolate sink outliers from their neighbours. Rate per token
is fixed at w*d + 32*(d/group_size): with one group per token and d=128
that is w + 0.25 b/c, the honest ~2.25 b/c INT2 point of amendment 2.

Uniform mode only (single rung — the page budget is diagnostic telemetry,
not an allocator input).
"""
from __future__ import annotations

import torch

from kvq.compression.page_quant import HEADER_BITS, _PagedBase

SCALE_ZERO_BITS = 32  # fp16 scale + fp16 zero per (token, group)


class OscarArmCompressor(_PagedBase):

    def __init__(self, forward_map, inverse_map, mu, mu_q, width,
                 group_size, clip_q, b_page, ptok=64):
        self.forward_map = forward_map.float()
        self.inverse_map = inverse_map.float()
        self.mu = mu.float()
        self.mu_q = mu_q.float()
        self.width = int(width)
        self.group_size = int(group_size)
        self.clip_q = float(clip_q)
        d = forward_map.shape[0]
        if d % self.group_size:
            raise ValueError(f"group_size {group_size} does not divide {d}")
        self.n_groups = d // self.group_size
        self.n_rungs = 1
        self.rate_token = float(self.width * d
                                + SCALE_ZERO_BITS * self.n_groups)
        self.b_page = float(b_page)
        self.ptok = int(ptok)
        self.mode = "uniform"
        self.device = mu.device
        self.reset_stats()

    def _side_bits(self, ntok):
        return HEADER_BITS

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]

        r = (k - self.mu) @ self.forward_map
        clip = torch.quantile(r.abs(), self.clip_q, dim=1, keepdim=True)
        rc = r.clamp(-clip, clip)

        g = rc.reshape(T, self.n_groups, self.group_size)
        zero = g.min(2, keepdim=True).values.to(torch.float16).float()
        levels = (1 << self.width) - 1
        scale = ((g.max(2, keepdim=True).values - zero) / levels) \
            .to(torch.float16).float()
        q = torch.round((g - zero) / scale.clamp_min(1e-12)).clamp(0, levels)
        r_hat = torch.where(scale > 0, q * scale + zero, zero).reshape(T, d)

        P = (T + self.ptok - 1) // self.ptok
        self.pages_total += P
        self.pages_overflow += 0
        self.bits_payload += self.rate_token * T
        self.bits_side += HEADER_BITS * P
        self.tokens_total += T
        self.rung_hist[0] += T

        out = r_hat @ self.inverse_map + self.mu
        return out.reshape(shape).to(states.dtype)
