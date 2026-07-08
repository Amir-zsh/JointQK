"""Arm A of the pgq2 study: norm + width-adaptive direction codes.

Per token, in the qpca_unc code domain (code-SE == Q-weighted key MSE):
store the fp16 norm n_t = ||r_t|| (16 bits — reconstruction uses the
fp16-ROUNDED value, codec honesty) plus the unit direction u_t coded
per-coordinate with Lloyd-Max codebooks at a per-token rung. Rungs are
per-coord bit PROFILES: uniform widths w*ones(d) and, optionally,
water-filled fractional-rate profiles (still fixed-width fields — decode
offsets derive from the rung id + frozen profile; random access intact).
Rung 0 is eviction: no norm, no payload, k_hat = mu_k.

The per-token norm is the representation fix for both proxy-invisible
failures of the scalar study (norm inflation; sink clipping — sink scale
lives in the norm, the direction is always unit). Positions 0-3 are still
forced to the max rung: norms make sinks representable, forcing makes them
unstrippable under tight-budget RDO.

Rate accounting per page (charged): HEADER_BITS + id_bits*ptok (rdo modes)
or HEADER_BITS only (uniform mode), plus per-token R_t = 16 + profile.sum()
for rung > 0, R_t = 0 for rung 0. No entropy coding anywhere.
"""
from __future__ import annotations

import math

import torch

from kvq.compression.page_quant import (
    HEADER_BITS, _PagedBase, _paged_lambda_assign,
)
from kvq.compression.per_coord import unit_gaussian_centroids

NORM_BITS = 16


def build_rung_codebooks(profiles: torch.Tensor,
                         coord_std_dir: torch.Tensor):
    """profiles (R, d) int; coord_std_dir (d,). Returns list over rungs of
    (d, Kmax_r) centroid tensors, +inf padded (shared-argmin trick from
    PerCoordCompressor), scaled by the direction's per-coord std.

    Width 8 is special: an ABSOLUTE [-1, 1] uniform grid, not std-scaled.
    Direction coords are bounded in [-1, 1] by construction (unit vector),
    so this rung is outlier-proof for ANY token — it is the forced sink
    rung (sink *directions* are outliers too: std-scaled codebooks clipped
    them at relerr ~0.6, caught by the G2 gate)."""
    out = []
    for prof in profiles:
        kmax = int(2 ** int(prof.max())) if int(prof.max()) > 0 else 1
        cb = torch.full((prof.shape[0], kmax), float("inf"))
        for j, b in enumerate(prof.tolist()):
            if b <= 0:
                cb[j, 0] = 0.0
            elif b >= 8:
                # mid-tread: 255 levels incl. EXACT zero (a 256-pt linspace
                # has no zero level -> +-1/255 offsets on ~all near-zero
                # coords, amplified in raw-k by the non-orthonormal inverse;
                # the zero-level lesson, third sighting)
                cb[j, :255] = torch.arange(-127, 128).float() / 127.0
            else:
                c = unit_gaussian_centroids(int(b)) * coord_std_dir[j]
                cb[j, : c.numel()] = c.sort().values
        out.append(cb)
    return out


class NormDirectionPagedCompressor(_PagedBase):
    """mode: "uniform" (single rung, no ids) | "rdo" (per-token rung)."""

    def __init__(self, forward_map, inverse_map, mu, mu_q,
                 coord_std_dir, profiles, b_page, ptok=64, mode="rdo",
                 uniform_rung=None, omega_tau=0.0, omega_clamp_bits=4.0,
                 gate_theta=0.0, head_m_spread=1.0):
        self.forward_map = forward_map.float()
        self.inverse_map = inverse_map.float()
        self.mu = mu.float()
        self.mu_q = mu_q.float()
        self.coord_std_dir = coord_std_dir.float().clamp_min(1e-12)
        self.profiles = profiles.long()               # (R, d); rung 0 all-0
        self.n_rungs = int(profiles.shape[0])
        self.codebooks = build_rung_codebooks(self.profiles,
                                              self.coord_std_dir)
        payload = self.profiles.sum(1)                # (R,)
        self.rate_bits = torch.where(payload > 0, payload + NORM_BITS,
                                     torch.zeros_like(payload)).double()
        self.b_page = float(b_page)
        self.ptok = int(ptok)
        self.mode = mode
        self.uniform_rung = uniform_rung
        self.omega_tau = float(omega_tau)
        self.omega_clamp_bits = float(omega_clamp_bits)
        # omega-confidence gate: per page, omega := 1 when the within-page
        # spread of m is below gate_theta * head_m_spread (fit-row spread).
        self.gate_theta = float(gate_theta)
        self.head_m_spread = float(head_m_spread)
        self.id_bits = max(1, math.ceil(math.log2(max(2, self.n_rungs))))
        self.device = mu.device
        self.reset_stats()

    def _side_bits(self, ntok):
        if self.mode == "uniform":
            return HEADER_BITS
        return HEADER_BITS + self.id_bits * ntok

    def _quant_rung(self, u, ri):
        """Nearest-centroid per coord against the rung's +inf-padded
        codebook (PerCoordCompressor pattern), chunked for memory."""
        cb = self.codebooks[ri]                       # (d, K) inf-padded
        out = torch.empty_like(u)
        for s in range(0, u.shape[0], 8192):
            uc = u[s: s + 8192]
            idx = (uc.unsqueeze(2) - cb.unsqueeze(0)).abs().argmin(2)
            out[s: s + 8192] = cb.gather(1, idx.t()).t()
        return out

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]
        dev = k.device

        r = (k - self.mu) @ self.forward_map
        n16 = r.norm(dim=1).to(torch.float16).float()  # decoder's norm
        u = r / n16.clamp_min(1e-8).unsqueeze(1)

        R = self.n_rungs
        Dmat = torch.empty(T, R, device=dev)
        uhat_by_rung = []
        for ri in range(R):
            if int(self.profiles[ri].max()) == 0:
                r_hat_r = torch.zeros_like(r)
                uh = None
            else:
                uh = self._quant_rung(u, ri)
                r_hat_r = uh * n16.unsqueeze(1)
            Dmat[:, ri] = (r - r_hat_r).square().sum(1)
            uhat_by_rung.append(uh)
        Rrow = self.rate_bits.to(dev)                  # (R,)

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

        if self.mode == "uniform":
            ri = int(self.uniform_rung)
            assign = torch.full((T,), ri, dtype=torch.long, device=dev)
            # sinks still forced to max rung
            nsink = min(4, T)
            assign[:nsink] = R - 1
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
            Dp = Dw64.reshape(P, ptok, R)
            Rp = R64.reshape(P, ptok, R)
            nsink = min(4, T)
            wmax = R - 1
            sink_bits = float(Rp[0, :nsink, wmax].sum())
            budgets = budgets.clone()
            Dp = Dp.clone()
            Rp = Rp.clone()
            Dp[0, :nsink, :] = 0.0
            Rp[0, :nsink, :] = 0.0
            budgets[0] = budgets[0] - sink_bits
            assign_p = _paged_lambda_assign(Dp, Rp, budgets, ptok)
            assign_p[0, :nsink] = wmax
            assign = assign_p.reshape(-1)[:T]
            used_pages = (Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2)
                          .sum(1), sink_bits)

        # ---- stats / honest rate ----
        self.pages_total += P
        if used_pages is None:
            payload = float(Rrow[assign].sum())
            self.bits_payload += payload
            self.pages_overflow += 0   # uniform rate is fixed by construction
        else:
            used, sink_bits = used_pages
            self.pages_overflow += int((used > budgets).sum())
            self.bits_payload += float(used.sum()) + sink_bits
        self.bits_side += float(sum(self._side_bits(int(x))
                                    for x in ntok.tolist()))
        self.tokens_total += T
        binc = torch.bincount(assign, minlength=R)
        for ri in range(R):
            self.rung_hist[ri] += int(binc[ri])

        r_hat = torch.zeros_like(r)
        for ri in range(R):
            msk = assign == ri
            if msk.any() and uhat_by_rung[ri] is not None:
                r_hat[msk] = uhat_by_rung[ri][msk] * n16[msk].unsqueeze(1)
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
        om_p = torch.where(gate_off.unsqueeze(1), torch.ones_like(om_p),
                           om_p)
        return om_p.reshape(-1)[:T]
