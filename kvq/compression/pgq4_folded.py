"""pgq4: folded-scalar paged compressors (decode-resident kernel tier).

One class covers families A/B/C of plan4 (notes/page_quant/studies/plan4.md):
rungs are per-coord WIDTH PROFILES (R, d) over the ladder {0, 2, 3, 4}.
Family A uses constant rows ([0]*d, [2]*d, ...), family B water-filled /
prefix-truncation profiles with width changes only at 32-coord blocks,
family C is the page-rung allocation mode of the same codec.

The kernel identity this codec is built around: with r = (k - mu) @ F and
k_hat = (s * i) @ G + mu (i integer codes, s calibration-static per-coord
scales), the decode score is

    q . k_hat = ((q @ G^T) * s) . i + q . mu

and q.mu is constant per head -> cancels in softmax. Basis AND scales fold
into the query once per decode step; per-token metadata is the rung id
(+ optional fp16 gain). Widths are uniform per rung-segment, so each rung
maps to one OSCAR-style stage-1 launch (constexpr unpack width).

Grids:
  uniform  symmetric mid-tread INT, levels {-lim..lim}*s_j, lim = 2^(w-1)-1,
           s_j = alpha_w * code_std_j (alpha_w fit per head at fit time).
           Integer-dot capable.
  lm       Lloyd-Max Gaussian LUT: levels = cents_w * code_std_j with the
           near-zero centroid forced to exact 0. Dequant-then-fp32 only
           (OSCAR's per-centroid-in-graph cautionary tale).

Sinks (positions 0-3, start_pos == 0 only) bypass every rung to an absolute
8-bit per-coord grid (rate 8*d [+16 gain]) — position-derivable, zero
sideband, page 0 pays. Narrow grids clip sinks 3-5x past q999 and poison the
softmax denominator globally (pgq1 lesson).

Gain variant: transmit g_t = fp16(||k - mu||) / ||r_hat @ G|| per token
(computed at encode; decode is one scalar multiply per token). Raw-domain
norm ratio == 1 by construction through the non-orthogonal inverse.

Rate accounting (charged against every page's budget): HEADER_BITS/page +
id_bits*ntok rung ids; payload sum_j w_j (+16 gain) per token. Calibration
constants (scales, alphas, LM cents, sink grid) are off-page like every
method's codebooks.
"""
from __future__ import annotations

import math

import torch

from kvq.compression.page_quant import (
    HEADER_BITS, _PagedBase, _paged_lambda_assign,
)

PGQ4_BUNDLE_VERSION = 5
WIDTH_LADDER = (0, 2, 3, 4)
SINK_BITS = 8                     # absolute sink-escape grid, bits/coord
SINK_LIM = 127
GAIN_BITS = 16


def uniform_codes(r: torch.Tensor, scale: torch.Tensor, w: int):
    """Symmetric mid-tread saturating INT-w grid; scale (d,) or (1, d).
    Returns (codes uint8 in [0, 2^w-2] with code = idx + lim, dequant).
    Dequant is idx * s — the single source of truth for uniform_quant."""
    lim = (1 << (w - 1)) - 1
    s = scale.unsqueeze(0) if scale.dim() == 1 else scale
    idx = torch.round(r / s).clamp_(-lim, lim)
    return (idx + lim).to(torch.uint8), idx * s


def uniform_quant(r: torch.Tensor, scale: torch.Tensor, w: int) -> torch.Tensor:
    return uniform_codes(r, scale, w)[1]


def lm_codes(r: torch.Tensor, cents: torch.Tensor, code_std: torch.Tensor):
    """Per-coord Lloyd-Max LUT: cents (2^w,) unit levels (zero forced),
    scaled by code_std (d,) — or (T, d) for per-row scale maps (pgq8).
    Nearest-level via bucketize on the unit axis.
    Returns (LUT positions uint8, dequant = cents[pos] * code_std)."""
    code_std = (code_std.unsqueeze(0) if code_std.dim() == 1 else code_std)
    y = r / code_std.clamp_min(1e-12)
    pos = torch.bucketize(y, cents).clamp_max_(cents.numel() - 1)
    left = (pos - 1).clamp_min(0)
    pick_left = (y - cents[left]) < (cents[pos] - y)
    pos = torch.where(pick_left & (pos > 0), left, pos)
    return pos.to(torch.uint8), cents[pos] * code_std


def lm_quant(r: torch.Tensor, cents: torch.Tensor,
             code_std: torch.Tensor) -> torch.Tensor:
    return lm_codes(r, cents, code_std)[1]


class FoldedScalarPagedCompressor(_PagedBase):
    """Profile-rung paged codec: per-page RDO (or page-rung / uniform) over
    fixed-width scalar grids in a calibrated basis, scales foldable into q."""

    supports_start_pos = True

    def __init__(self, forward_map, inverse_map, mu, mu_q, code_std,
                 profiles, alphas, sink_scale, b_page,
                 grid="uniform", lm_cents=None, gain=False,
                 ptok=64, mode="rdo", uniform_rung=None,
                 omega_tau=0.0, omega_clamp_bits=4.0,
                 force_recent_pages=0, width_ladder=WIDTH_LADDER):
        self.forward_map = forward_map.float()
        self.inverse_map = inverse_map.float()
        self.mu = mu.float()
        self.mu_q = mu_q.float()
        self.code_std = code_std.float().clamp_min(1e-12)
        self.width_ladder = tuple(int(w) for w in width_ladder)
        self.widths_pos = [w for w in self.width_ladder if w > 0]
        self.profiles = profiles.long()               # (R, d) widths
        self.n_rungs = int(profiles.shape[0])
        # alphas: (n_widths>0,) scalars aligned to WIDTH_LADDER[1:]
        self.alphas = alphas.float()
        self.sink_scale = sink_scale.float().clamp_min(1e-12)
        self.grid = grid
        self.lm_cents = ([c.float() for c in lm_cents]
                         if lm_cents is not None else None)
        self.gain = bool(gain)
        self.b_page = float(b_page)
        self.ptok = int(ptok)
        self.mode = mode
        self.uniform_rung = uniform_rung
        self.omega_tau = float(omega_tau)
        self.omega_clamp_bits = float(omega_clamp_bits)
        self.force_recent_pages = int(force_recent_pages)
        self.id_bits = max(1, math.ceil(math.log2(max(2, self.n_rungs))))
        # per-rung payload bits/token (gain charged on every non-evict rung)
        rates = self.profiles.sum(1).double()
        if self.gain:
            rates = rates + GAIN_BITS * (self.profiles.sum(1) > 0).double()
        self.rung_rate = rates                         # (R,)
        self.device = mu.device
        self.reset_stats()

    def _side_bits(self, ntok):
        if self.mode == "uniform":
            return HEADER_BITS
        if self.mode == "pagerung":
            return HEADER_BITS + self.id_bits          # one id per page
        return HEADER_BITS + self.id_bits * ntok

    def _quant_width(self, r, w, wi):
        return self._codes_width(r, w, wi)[1]

    def _codes_width(self, r, w, wi):
        """(LUT codes uint8, dequant) for width w; dequant is what
        _quant_width always produced (codes share the same snap ops)."""
        if w == 0:
            return (torch.zeros(r.shape, dtype=torch.uint8, device=r.device),
                    torch.zeros_like(r))
        if self.grid == "lm" and w < self.width_ladder[-1]:
            return lm_codes(r, self.lm_cents[wi], self.code_std)
        return uniform_codes(r, self.alphas[wi] * self.code_std, w)

    def _apply_gain(self, k, r_hat):
        """k_hat = g * (r_hat @ G) + mu with g = fp16(||k-mu||)/||r_hat@G||.
        Evicted tokens (r_hat == 0) reconstruct mu exactly (no gain sent)."""
        y = r_hat @ self.inverse_map
        n16 = (k - self.mu).norm(dim=1).to(torch.float16).float()
        g = n16 / y.norm(dim=1).clamp_min(1e-8)
        g = torch.where(r_hat.abs().sum(1) > 0, g, torch.zeros_like(g))
        return g.unsqueeze(1) * y + self.mu

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor,
                  start_pos: int = 0, emit: dict | None = None) -> torch.Tensor:
        """emit (additive, pgq7 K1): pass a dict to receive the packed-format
        source data — assign (T,), per-coord LUT codes (T, d) uint8 under each
        token's assigned profile, sink_codes (nsink, d) uint8, nsink, and the
        exact pre-gain r_hat (the bit-identity reference for pack/unpack)."""
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]
        dev = k.device
        r = (k - self.mu) @ self.forward_map
        R = self.n_rungs

        # per-width quantizations + per-coord squared errors, shared by rungs
        widths_pos = self.widths_pos
        want_codes = emit is not None
        codes_by_w = {}
        dq_by_w = {0: torch.zeros_like(r)}
        err2_by_w = {0: r.square()}
        for wi, w in enumerate(widths_pos):
            if want_codes:
                codes_by_w[w], dq = self._codes_width(r, w, wi)
            else:
                dq = self._quant_width(r, w, wi)
            dq_by_w[w] = dq
            err2_by_w[w] = (r - dq).square()

        Dmat = torch.empty(T, R, device=dev)
        for ri in range(R):
            row = self.profiles[ri]
            acc = torch.zeros(T, device=dev)
            for w in self.width_ladder:
                cols = (row == w)
                if cols.any():
                    acc += err2_by_w[w][:, cols].sum(1)
            Dmat[:, ri] = acc
        Rmat = self.rung_rate.to(dev).unsqueeze(0).expand(T, R)

        nsink = min(4, T) if start_pos == 0 else 0
        sink_bits = nsink * (SINK_BITS * d + (GAIN_BITS if self.gain else 0))

        if self.mode == "uniform":
            assign = torch.full((T,), int(self.uniform_rung),
                                dtype=torch.long, device=dev)
            P = (T + self.ptok - 1) // self.ptok
            payload = float(Rmat[0, int(self.uniform_rung)]) * (T - nsink) \
                + sink_bits
            side = self._side_bits(0) * P
            overflow = 0
        else:
            if self.omega_tau > 0.0:
                m_sig = (k @ self.mu_q) / math.sqrt(d)
                c = self.omega_clamp_bits * math.log(2.0)
                om = torch.exp((self.omega_tau * (m_sig - m_sig.median()))
                               .clamp(-c, c))
                Dw = Dmat * om.unsqueeze(1)
            else:
                Dw = Dmat
            ptok = self.ptok
            P = (T + ptok - 1) // ptok
            pad = P * ptok - T
            ntok = torch.full((P,), float(ptok), dtype=torch.float64,
                              device=dev)
            if pad:
                ntok[-1] = ptok - pad
            Dw64 = Dw.double()
            R64 = Rmat.double().clone()
            if pad:
                z = torch.zeros(pad, R, dtype=torch.float64, device=dev)
                Dw64 = torch.cat([Dw64, z])
                R64 = torch.cat([R64, z.clone()])

            if self.mode == "pagerung":
                Dpg = Dw64.reshape(P, ptok, R).sum(1)          # (P, R)
                Rpg = R64.reshape(P, ptok, R).sum(1)
                budget_tot = torch.tensor(
                    [self.b_page * d * T - self._side_bits(0) * P
                     - sink_bits],
                    dtype=torch.float64, device=dev)
                top = R - 1
                Dpg, Rpg = Dpg.clone(), Rpg.clone()
                # sink page forced to the top rung (kernel-trivial variant)
                pg0_bits = float(Rpg[0, top]) \
                    - nsink * float(self.rung_rate[top])
                budget_tot -= pg0_bits
                Dpg[0] = 0.0
                Rpg[0] = 0.0
                assign_pg = _paged_lambda_assign(
                    Dpg.unsqueeze(0), Rpg.unsqueeze(0), budget_tot, P
                ).squeeze(0)
                assign_pg[0] = top
                used_free = float(
                    Rpg.gather(1, assign_pg.unsqueeze(1)).sum())
                overflow = int(used_free > float(budget_tot))
                assign = assign_pg.unsqueeze(1).expand(P, ptok) \
                    .reshape(-1)[:T]
                payload = used_free + pg0_bits + sink_bits
            else:
                budgets = (self.b_page * d * ntok
                           - torch.as_tensor(
                               [self._side_bits(int(n))
                                for n in ntok.tolist()],
                               dtype=torch.float64, device=dev))
                Dp = Dw64.reshape(P, ptok, R).clone()
                Rp = R64.reshape(P, ptok, R).clone()
                budgets = budgets.clone()
                if nsink:
                    Dp[0, :nsink] = 0.0
                    Rp[0, :nsink] = 0.0
                    budgets[0] -= sink_bits
                top = R - 1
                nforce = 0
                forced_bits = 0.0
                if start_pos == 0 and self.force_recent_pages > 0 and P > 1:
                    nforce = min(self.force_recent_pages, P - 1)
                    for p in range(P - nforce, P):
                        # padded rows carry zero rates, so this is exact for
                        # a partial last page
                        cost = float(Rp[p, :, top].sum())
                        forced_bits += cost
                        budgets[p] -= cost
                        Dp[p] = 0.0
                        Rp[p] = 0.0
                assign_p = _paged_lambda_assign(Dp, Rp, budgets, ptok)
                if nsink:
                    assign_p[0, :nsink] = top
                if nforce:
                    assign_p[P - nforce:] = top

                # greedy refinement: spend leftover bits on best D-drop/bit
                used = Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2).sum(1)
                left = budgets - used
                for _ in range(8):
                    cur_d = Dp.gather(2, assign_p.unsqueeze(2)).squeeze(2)
                    cur_r = Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2)
                    gain_m = cur_d.unsqueeze(2) - Dp
                    cost = Rp - cur_r.unsqueeze(2)
                    ok = (cost > 0) & (cost <= left.view(P, 1, 1)) \
                        & (gain_m > 0)
                    score = torch.where(ok, gain_m / cost,
                                        torch.zeros_like(gain_m))
                    best = score.view(P, -1).argmax(1)
                    best_score = score.view(P, -1).gather(
                        1, best.unsqueeze(1)).squeeze(1)
                    pi = torch.nonzero(best_score > 0).squeeze(1)
                    if pi.numel() == 0:
                        break
                    bt = (best[pi] // R).long()
                    bw = (best[pi] % R).long()
                    cur_w = assign_p[pi, bt]
                    ar = torch.arange(pi.numel(), device=dev)
                    old_r = Rp[pi, bt][ar, cur_w]
                    new_r = Rp[pi, bt][ar, bw]
                    assign_p[pi, bt] = bw
                    left[pi] -= (new_r - old_r)

                assign = assign_p.reshape(-1)[:T]
                used = Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2).sum(1)
                # forced pages exceed their per-page budget by design; the
                # overshoot is charged in the honest rate, not as overflow
                n_over = P - nforce if nforce else P
                overflow = int((used[:n_over] > budgets[:n_over]).sum())
                payload = float(used.sum()) + sink_bits + forced_bits
            side = float(sum(self._side_bits(int(n))
                             for n in ([ptok] * (P - 1)
                                       + [ptok - pad if pad else ptok])))

        # reconstruction
        r_hat = torch.zeros_like(r)
        codes = (torch.zeros(T, d, dtype=torch.uint8, device=dev)
                 if want_codes else None)
        for ri in range(R):
            msk = assign == ri
            if not msk.any():
                continue
            row = self.profiles[ri]
            block = torch.zeros(int(msk.sum()), d, device=dev)
            cblk = (torch.zeros(int(msk.sum()), d, dtype=torch.uint8,
                                device=dev) if want_codes else None)
            for w in widths_pos:
                cols = (row == w)
                if cols.any():
                    block[:, cols] = dq_by_w[w][msk][:, cols]
                    if want_codes:
                        cblk[:, cols] = codes_by_w[w][msk][:, cols]
            r_hat[msk] = block
            if want_codes:
                codes[msk] = cblk
        sink_codes = None
        if nsink:
            sidx = (torch.round(r[:nsink] / self.sink_scale)
                    .clamp_(-SINK_LIM, SINK_LIM))
            r_hat[:nsink] = sidx * self.sink_scale
            if want_codes:
                sink_codes = (sidx + SINK_LIM).to(torch.uint8)
                codes[:nsink] = 0        # sink rows live in sink_codes only

        self.pages_total += int(P)
        self.pages_overflow += int(overflow)
        self.bits_payload += float(payload)
        self.bits_side += float(side)
        self.tokens_total += T
        binc = torch.bincount(assign[nsink:], minlength=R)
        for ri in range(R):
            self.rung_hist[ri] += int(binc[ri])

        if want_codes:
            emit.update(assign=assign, codes=codes, nsink=nsink,
                        sink_codes=sink_codes, r_hat=r_hat)

        if self.gain:
            out = self._apply_gain(k, r_hat)
        else:
            out = r_hat @ self.inverse_map + self.mu
        return out.reshape(shape).to(states.dtype)
