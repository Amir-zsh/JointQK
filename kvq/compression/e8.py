"""pgq3 Thrust-B family (c): E8 Voronoi product codes.

Per token, in the (optionally Hadamard-mixed) qpca_unc code domain: the d
coords split into d/8 subvectors, each quantized to the E8 lattice scaled by
a single per-head beta, with m in {0, 1, 2} bits/dim per subvector set by a
per-rung water-filled profile. Codebooks are Voronoi codes (Conway-Sloane):
the 2^(8m) lattice points inside the Voronoi region of 2^m*E8, indexed
closed-form both ways — encode is nearest-point + generator-basis mod 2^m,
decode is a table-free linear map + one nearest-point call. Out-of-region
subvectors WRAP under the modulo (overload); Dmat prices the wrapped
reconstruction, so allocation sees overload honestly (snap-aware).

The Hadamard mixer is orthogonal in code space, so code-SE == Q-weighted key
MSE is preserved exactly while subvectors become near-isotropic — one shared
beta per head suffices (OSCAR amendment 1).

The lattice contains 0, so an all-zero profile row is the eviction rung
(k_hat = mu_k). A final fp16 escape rung (16 b/c, exact) is appended
internally: it is the forced sink rung (positions 0-3 are 3-5x beyond q999
and would overload every lattice rung) and is available to the RDO for any
other outlier token.

Rate per token: 8 * profile.sum() bits (lattice rungs; no norm sideband) or
16*d (escape). Page sideband: HEADER_BITS + id_bits*ptok (rdo) or
HEADER_BITS (uniform). No entropy coding anywhere.
"""
from __future__ import annotations

import math

import torch

from kvq.compression.page_quant import (
    HEADER_BITS, _PagedBase, _paged_lambda_assign,
)

_E8_GEN = torch.tensor([
    [2., 0., 0., 0., 0., 0., 0., 0.],
    [-1., 1., 0., 0., 0., 0., 0., 0.],
    [0., -1., 1., 0., 0., 0., 0., 0.],
    [0., 0., -1., 1., 0., 0., 0., 0.],
    [0., 0., 0., -1., 1., 0., 0., 0.],
    [0., 0., 0., 0., -1., 1., 0., 0.],
    [0., 0., 0., 0., 0., -1., 1., 0.],
    [.5, .5, .5, .5, .5, .5, .5, .5]], dtype=torch.float64)
_E8_GEN_INV = torch.linalg.inv(_E8_GEN)


def _d8_nearest(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of D8 (integer vectors with even sum) to x (..., 8)."""
    r = x.round()
    err = x - r
    j = err.abs().argmax(-1, keepdim=True)
    step = torch.where(err.gather(-1, j) >= 0, 1.0, -1.0)
    r_flip = r.scatter(-1, j, r.gather(-1, j) + step)
    odd = (r.sum(-1).long().remainder(2) != 0).unsqueeze(-1)
    return torch.where(odd, r_flip, r)


def e8_nearest(x: torch.Tensor) -> torch.Tensor:
    """Nearest E8 point to x (..., 8). E8 = D8 union (D8 + 1/2)."""
    a = _d8_nearest(x)
    b = _d8_nearest(x - 0.5) + 0.5
    pick_a = ((x - a).square().sum(-1)
              <= (x - b).square().sum(-1)).unsqueeze(-1)
    return torch.where(pick_a, a, b)


def voronoi_roundtrip(y: torch.Tensor, m: int) -> torch.Tensor:
    """Encode y (..., 8) in E8 to the 2^(8m) Voronoi code of 2^m*E8 and
    decode: identity inside the region, modulo wrap outside (overload)."""
    if m <= 0:
        return torch.zeros_like(y)
    gen = _E8_GEN.to(y.device)
    gen_inv = _E8_GEN_INV.to(y.device)
    a = (y.double() @ gen_inv).round()             # integer coords in E8
    y0 = a.remainder(2 ** m) @ gen                 # representative mod 2^m*E8
    y_hat = y0 - (2 ** m) * e8_nearest(y0 / (2 ** m)).double()
    return y_hat.to(y.dtype)


def quant_profile(x_sub: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
    """x_sub (T, d/8, 8) in beta-units -> quantized, per-subvector m.

    Boundary cosets of E8 / 2^m*E8 can decode to a FAR equivalent of
    the nearest lattice point (at m=1 every coset is +-symmetric, so a
    sign can flip across the region) — a wrap costing more than zeroing
    the subvector. Index 0 always decodes to exactly 0, so the encoder
    keeps the better of {wrapped codeword, 0} per subvector: still a
    valid index stream, and rung distortion <= eviction structurally."""
    out = torch.zeros_like(x_sub)
    for m in profile.unique().tolist():
        if m <= 0:
            continue
        cols = profile == m
        x = x_sub[:, cols]
        y_hat = voronoi_roundtrip(e8_nearest(x), int(m))
        keep = ((x - y_hat).square().sum(-1)
                <= x.square().sum(-1)).unsqueeze(-1)
        out[:, cols] = torch.where(keep, y_hat, torch.zeros_like(y_hat))
    return out


class E8PagedCompressor(_PagedBase):
    """mode: "uniform" (single rung, no ids) | "rdo" (per-token rung)."""

    def __init__(self, forward_map, inverse_map, mu, mu_q,
                 profiles, beta, mixer, b_page, ptok=64, mode="rdo",
                 uniform_rung=None, omega_tau=0.0, omega_clamp_bits=4.0):
        self.forward_map = forward_map.float()
        self.inverse_map = inverse_map.float()
        self.mu = mu.float()
        self.mu_q = mu_q.float()
        self.profiles = profiles.long()               # (R_lat, d/8)
        self.beta = float(beta)
        self.mixer = None if mixer is None else mixer.float()
        d = forward_map.shape[0]
        if d % 8:
            raise ValueError(f"head_dim {d} not divisible by 8")
        if int(self.profiles.shape[1]) != d // 8:
            raise ValueError("profile length != d/8")
        # lattice rungs (norm 16b + 8*sum(m); the all-zero profile is a true
        # evict rung at 0 bits) + the fp16 escape/sink rung appended last
        self.n_rungs = int(self.profiles.shape[0]) + 1
        psum = self.profiles.sum(1).double()
        self.rate_bits = torch.cat([
            torch.where(psum > 0, 16.0 + 8.0 * psum,
                        torch.zeros_like(psum)),
            torch.tensor([16.0 * d]).double()])
        self.b_page = float(b_page)
        self.ptok = int(ptok)
        self.mode = mode
        self.uniform_rung = uniform_rung
        self.omega_tau = float(omega_tau)
        self.omega_clamp_bits = float(omega_clamp_bits)
        self.id_bits = max(1, math.ceil(math.log2(max(2, self.n_rungs))))
        self.device = mu.device
        self.reset_stats()

    def _side_bits(self, ntok):
        if self.mode == "uniform":
            return HEADER_BITS
        return HEADER_BITS + self.id_bits * ntok

    def _quant_profile(self, x_sub: torch.Tensor,
                       profile: torch.Tensor) -> torch.Tensor:
        return quant_profile(x_sub, profile)

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]
        dev = k.device

        r = (k - self.mu) @ self.forward_map
        # Norm+direction structure (third gate design): the raw-domain norm
        # is transmitted (16b, in every lattice rung's rate) and the E8
        # code carries only the unit direction — wrap-guard shrinkage then
        # cannot touch the scale (decode: r_hat = n16 * u_hat/||u_hat @ G||,
        # raw normR = 1 by construction). The LS-gain approach is gone: it
        # is a shrinkage estimator and made G3 worse (second gate sweep).
        n16 = (k - self.mu).norm(dim=1).to(torch.float16).float()
        u = r / r.norm(dim=1, keepdim=True).clamp_min(1e-8)
        um = u @ self.mixer if self.mixer is not None else u
        x_sub = (um / self.beta).reshape(T, d // 8, 8)

        def _scaled(uh_m):
            uh = uh_m @ self.mixer.t() if self.mixer is not None else uh_m
            g = (uh @ self.inverse_map).norm(dim=1)
            return uh * (n16 / g.clamp_min(1e-8)).unsqueeze(1)

        R = self.n_rungs
        Dmat = torch.empty(T, R, device=dev)
        rhat_by_rung = []
        for ri in range(R - 1):
            rh = _scaled(self._quant_profile(x_sub, self.profiles[ri])
                         .reshape(T, d) * self.beta)
            rhat_by_rung.append(rh)
            Dmat[:, ri] = (r - rh).square().sum(1)
        r16 = r.to(torch.float16).float()              # escape rung, exact
        rhat_by_rung.append(r16)
        Dmat[:, R - 1] = (r - r16).square().sum(1)
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

        nsink = min(4, T)
        sink_cost = float(nsink) * 16.0 * d            # forced escape rung
        if self.mode == "uniform":
            ri = int(self.uniform_rung)
            assign = torch.full((T,), ri, dtype=torch.long, device=dev)
            assign[:nsink] = R - 1
            used_pages = None
        else:
            if self.omega_tau > 0.0:
                m_sig = (k @ self.mu_q) / math.sqrt(d)
                c = self.omega_clamp_bits * math.log(2.0)
                om = torch.exp((self.omega_tau * (m_sig - m_sig.median()))
                               .clamp(-c, c))
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
            budgets = budgets.clone()
            Dp[0, :nsink, :] = 0.0
            Rp[0, :nsink, :] = 0.0
            budgets[0] = budgets[0] - sink_cost
            assign_p = _paged_lambda_assign(Dp, Rp, budgets, ptok)
            assign_p[0, :nsink] = R - 1
            assign = assign_p.reshape(-1)[:T]
            used_pages = (Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2)
                          .sum(1), sink_cost)

        self.pages_total += P
        if used_pages is None:
            self.bits_payload += (float(Rrow[assign[nsink:]].sum())
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

        r_hat = torch.empty_like(r)
        for ri in range(R):
            msk = assign == ri
            if msk.any():
                r_hat[msk] = rhat_by_rung[ri][msk]
        out = r_hat @ self.inverse_map + self.mu
        return out.reshape(shape).to(states.dtype)
