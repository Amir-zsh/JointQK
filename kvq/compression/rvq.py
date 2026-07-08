"""Arm B of the pgq2 study: residual VQ with per-token stage count.

Per token: fp16 norm n_t (16 bits) + unit direction u_t coded by S residual
product-VQ stages (each stage: n_sub subvectors x K_s entries), the per-token
stage count s_t in {0..S} chosen by (omega-)RDO within fixed page budgets.
s=0 evicts (no norm, no payload, k_hat = mu_k). Codebooks are trained on
NORMALIZED directions in the qpca_unc code domain (k-means L2 there == the
Q-weighted objective; the norm absorbs sink scale — raw-domain codebooks
would have no codewords near sink-scale vectors).

A frozen variance-balancing coordinate permutation spreads the eigen-spectrum
across subvectors; codewords are stored in permuted order and the inverse
permutation is applied once at reconstruction — zero extra decode cost,
random access intact (pure table lookups).

Rate per token: R_t(s) = 16 + sum_{j<s} stage_bits[j] for s>0, else 0.
Page sideband: HEADER_BITS + id_bits*ptok (rdo) or HEADER_BITS (uniform).
No entropy coding anywhere.
"""
from __future__ import annotations

import math

import torch

from kvq.compression.page_quant import (
    HEADER_BITS, _PagedBase, _paged_lambda_assign,
)

NORM_BITS = 16


def nearest_codewords(x_sub: torch.Tensor, cb: torch.Tensor):
    """x_sub (n_sub, T, sd), cb (n_sub, K, sd) -> quantized (n_sub, T, sd).
    Batched L2 nearest via bmm."""
    d2 = (x_sub.square().sum(2, keepdim=True)
          - 2.0 * torch.bmm(x_sub, cb.transpose(1, 2))
          + cb.square().sum(2).unsqueeze(1))
    idx = d2.argmin(2)                                    # (n_sub, T)
    return torch.gather(
        cb, 1, idx.unsqueeze(2).expand(-1, -1, cb.shape[2]))


class ResidualVQPagedCompressor(_PagedBase):
    """mode: "uniform" (fixed stage count, no ids) | "rdo"."""

    def __init__(self, forward_map, inverse_map, mu, mu_q,
                 codebooks, perm, b_page, ptok=64, mode="rdo",
                 uniform_stages=None, omega_tau=0.0, omega_clamp_bits=4.0,
                 gate_theta=0.0, head_m_spread=1.0):
        self.forward_map = forward_map.float()
        self.inverse_map = inverse_map.float()
        self.mu = mu.float()
        self.mu_q = mu_q.float()
        # list over stages of (n_sub, K_s, sub_dim) float32
        self.codebooks = [c.float() for c in codebooks]
        self.n_stages = len(self.codebooks)
        self.n_rungs = self.n_stages + 1              # rung 0 = eviction
        self.perm = perm.long()
        self.inv_perm = torch.argsort(self.perm)
        self.stage_bits = [int(c.shape[0] * math.ceil(math.log2(c.shape[1])))
                           for c in self.codebooks]
        cum = torch.tensor([0] + list(torch.tensor(self.stage_bits)
                                      .cumsum(0).tolist())).double()
        self.rate_bits = torch.where(
            torch.arange(self.n_rungs) > 0, cum + NORM_BITS,
            torch.zeros(self.n_rungs).double())
        self.b_page = float(b_page)
        self.ptok = int(ptok)
        self.mode = mode
        self.uniform_stages = uniform_stages
        self.omega_tau = float(omega_tau)
        self.omega_clamp_bits = float(omega_clamp_bits)
        self.gate_theta = float(gate_theta)
        self.head_m_spread = float(head_m_spread)
        self.id_bits = max(1, math.ceil(math.log2(max(2, self.n_rungs))))
        self.device = mu.device
        self.reset_stats()

    def _side_bits(self, ntok):
        if self.mode == "uniform":
            return HEADER_BITS
        return HEADER_BITS + self.id_bits * ntok

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]
        dev = k.device

        r = (k - self.mu) @ self.forward_map
        n16 = r.norm(dim=1).to(torch.float16).float()
        u = r / n16.clamp_min(1e-8).unsqueeze(1)

        # Sinks (positions 0-3) bypass the VQ stages entirely: their
        # directions are outliers no codebook trained on 138k bulk tokens
        # covers (G2-gate finding). They get an absolute 8-bit [-1,1] grid
        # on the direction — outlier-proof for any unit vector — at
        # 16 + 8*d bits, charged to page 0. Position-derivable, zero
        # sideband (the pre-registered Arm-A-style sink contingency).
        nsink_g = min(4, T)
        u_sink = torch.round(u[:nsink_g] * 127.0).clamp(-127, 127) / 127.0

        # cumulative RVQ reconstructions in PERMUTED space
        n_sub = self.codebooks[0].shape[0]
        sd = self.codebooks[0].shape[2]
        up = u[:, self.perm].reshape(T, n_sub, sd).permute(1, 0, 2)
        resid = up.clone()
        acc = torch.zeros_like(up)
        uhat_by_stage = []                              # (T, d) unpermuted
        for s in range(self.n_stages):
            q = nearest_codewords(resid.contiguous(), self.codebooks[s])
            acc = acc + q
            resid = resid - q
            uh = acc.permute(1, 0, 2).reshape(T, d)[:, self.inv_perm]
            uhat_by_stage.append(uh)

        R = self.n_rungs
        Dmat = torch.empty(T, R, device=dev)
        Dmat[:, 0] = r.square().sum(1)                  # eviction
        for s in range(self.n_stages):
            r_hat_s = uhat_by_stage[s] * n16.unsqueeze(1)
            Dmat[:, s + 1] = (r - r_hat_s).square().sum(1)
        Rrow = self.rate_bits.to(dev)

        ptok = self.ptok
        P = (T + ptok - 1) // ptok
        pad = P * ptok - T
        ntok = torch.full((P,), float(ptok), dtype=torch.float64, device=dev)
        if pad:
            ntok[-1] = ptok - pad
        budgets = (self.b_page * d * ntok
                   - torch.as_tensor([self._side_bits(int(x))
                                      for x in ntok.tolist()],
                                     dtype=torch.float64, device=dev))

        sink_cost = float(nsink_g) * (NORM_BITS + 8 * d)
        if self.mode == "uniform":
            ri = int(self.uniform_stages)
            assign = torch.full((T,), ri, dtype=torch.long, device=dev)
            nsink = nsink_g
            assign[:nsink] = R - 1     # hist bucket only; recon/rate special
            used_pages = None
        else:
            if self.omega_tau > 0.0:
                m_sig = (k @ self.mu_q) / math.sqrt(d)
                c = self.omega_clamp_bits * math.log(2.0)
                om = torch.exp((self.omega_tau * (m_sig - m_sig.median()))
                               .clamp(-c, c))
                if self.gate_theta > 0.0:
                    om = self._apply_gate(om, m_sig, P, ptok, T)
                Dw = Dmat * om.unsqueeze(1)
            else:
                Dw = Dmat
            Dw64 = Dw.double()
            R64 = Rrow.expand(T, R).double().clone()
            if pad:
                z = torch.zeros(pad, R, dtype=torch.float64, device=dev)
                Dw64 = torch.cat([Dw64, z])
                R64 = torch.cat([R64, z.clone()])
            Dp = Dw64.reshape(P, ptok, R).clone()
            Rp = R64.reshape(P, ptok, R).clone()
            nsink = nsink_g
            wmax = R - 1
            budgets = budgets.clone()
            Dp[0, :nsink, :] = 0.0
            Rp[0, :nsink, :] = 0.0
            budgets[0] = budgets[0] - sink_cost
            assign_p = _paged_lambda_assign(Dp, Rp, budgets, ptok)
            assign_p[0, :nsink] = wmax
            assign = assign_p.reshape(-1)[:T]
            used_pages = (Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2)
                          .sum(1), sink_cost)

        self.pages_total += P
        if used_pages is None:
            self.bits_payload += (float(Rrow[assign[nsink_g:]].sum())
                                  + sink_cost)
        else:
            used, sc = used_pages
            self.pages_overflow += int((used > budgets).sum())
            self.bits_payload += float(used.sum()) + sc
        self.bits_side += float(sum(self._side_bits(int(x))
                                    for x in ntok.tolist()))
        self.tokens_total += T
        binc = torch.bincount(assign, minlength=R)
        for ri in range(R):
            self.rung_hist[ri] += int(binc[ri])

        r_hat = torch.zeros_like(r)
        for s in range(self.n_stages):
            msk = assign == (s + 1)
            if msk.any():
                r_hat[msk] = uhat_by_stage[s][msk] * n16[msk].unsqueeze(1)
        r_hat[:nsink_g] = u_sink * n16[:nsink_g].unsqueeze(1)
        out = r_hat @ self.inverse_map + self.mu
        return out.reshape(shape).to(states.dtype)

    def _apply_gate(self, om, m_sig, P, ptok, T):
        pad = P * ptok - T
        m_p = torch.cat([m_sig, torch.zeros(pad, device=m_sig.device)]) \
            if pad else m_sig
        m_p = m_p.reshape(P, ptok)
        spread = (m_p.quantile(0.9, dim=1) - m_p.median(dim=1).values)
        gate_off = spread < self.gate_theta * self.head_m_spread
        om_p = torch.cat([om, torch.ones(pad, device=om.device)]) \
            if pad else om
        om_p = om_p.reshape(P, ptok)
        return torch.where(gate_off.unsqueeze(1), torch.ones_like(om_p),
                           om_p).reshape(-1)[:T]
