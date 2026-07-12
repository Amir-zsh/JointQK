"""pgq8: page-level token-axis DCT codec (plan8).

Per interior full page, the 64 token rows of the code matrix (qpca_unc
space) are replaced by DCT-II coefficient rows Y = D @ R before the width-
rung RDO; decode inverts with R_hat = D^T @ Y_hat. Orthogonality preserves
code-space SE == Q-weighted logit error exactly, so every distortion the
allocator sees keeps its meaning; the fold identity survives as
z_page = D^T (Y_hat q~^T).

Identity (untransformed) pages: page 0 (sink escape lives there),
rw-forced recency pages, the partial last page, and all decode chunks
(start_pos > 0, Mode-B' ring) — per-token protections stay per-token.

Grids are the pgq4 LM / covering-uniform grids scaled by a per-
(coefficient-row, coord) std map `dct_std` (fit-pool refit; basis,
profiles, alphas, ladder all frozen from the pgq4 bundle). Rate semantics
are byte-identical to pgq4 (header + 3-bit id per row), so honest rates
are directly comparable. v1 scope: mode="rdo", no gain, no omega.
"""
from __future__ import annotations

import math

import torch

from kvq.compression.page_quant import _paged_lambda_assign
from kvq.compression.pgq4_folded import (
    SINK_BITS, SINK_LIM, FoldedScalarPagedCompressor, lm_codes, uniform_codes,
)


def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II, rows = coefficients (row 0 = scaled page mean)."""
    t = torch.arange(n).float()
    m = torch.cos(math.pi * (2 * t.unsqueeze(0) + 1) * t.unsqueeze(1) / (2 * n))
    m = m * math.sqrt(2.0 / n)
    m[0] *= 1.0 / math.sqrt(2.0)
    return m


class PageDCTCompressor(FoldedScalarPagedCompressor):
    """Token-axis DCT rows through the pgq4 width-rung page RDO."""

    def __init__(self, dct_std=None, **kw):
        super().__init__(**kw)
        assert self.mode == "rdo", "pgq8 v1 supports mode='rdo' only"
        assert not self.gain, "pgq8 v1 does not support the gain variant"
        assert self.omega_tau == 0.0, "pgq8 v1 does not support omega"
        assert dct_std is not None and dct_std.shape[0] == self.ptok
        self.dct_std = dct_std.float().clamp_min(1e-12)      # (ptok, d)
        self.dct_m = dct_matrix(self.ptok)

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor,
                  start_pos: int = 0, emit: dict | None = None) -> torch.Tensor:
        """emit (pgq7 golden vectors): assign, coefficient-row LUT codes,
        sink_codes, tmask (per full page: transformed?), nfull, y_hat (the
        coefficient-domain bit-identity reference; pack/unpack must
        reproduce it exactly), sigma (the (T, d) scale map the codes were
        quantized against), and the post-inverse r_hat."""
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]
        dev = k.device
        r = (k - self.mu) @ self.forward_map
        R = self.n_rungs
        ptok = self.ptok
        P = (T + ptok - 1) // ptok
        pad = P * ptok - T
        nfull = P - (1 if pad else 0)

        nsink = min(4, T) if start_pos == 0 else 0
        nforce = 0
        if start_pos == 0 and self.force_recent_pages > 0 and P > 1:
            nforce = min(self.force_recent_pages, P - 1)

        # transform-eligible pages: prefill, full, not page 0, not rw-forced
        tmask = torch.zeros(max(nfull, 1), dtype=torch.bool, device=dev)
        if start_pos == 0 and nfull > 0:
            tmask[:nfull] = True
            tmask[0] = False                                  # sink page
            if nforce:
                tmask[max(0, P - nforce):] = False

        y = r
        sigma = self.code_std.unsqueeze(0).expand(T, d)
        if tmask.any():
            y = r.clone()
            sigma = sigma.clone()
            yp = y[: nfull * ptok].reshape(nfull, ptok, d)
            yp[tmask[:nfull]] = torch.einsum(
                "st,ptd->psd", self.dct_m,
                r[: nfull * ptok].reshape(nfull, ptok, d)[tmask[:nfull]])
            sp = sigma[: nfull * ptok].reshape(nfull, ptok, d)
            sp[tmask[:nfull]] = self.dct_std

        # per-width quantizations on the (mixed) rows with per-row scales
        widths_pos = self.widths_pos
        want_codes = emit is not None
        codes_by_w = {}
        dq_by_w = {0: torch.zeros_like(y)}
        err2_by_w = {0: y.square()}
        for wi, w in enumerate(widths_pos):
            if self.grid == "lm" and w < self.width_ladder[-1]:
                cw, dq = lm_codes(y, self.lm_cents[wi], sigma)
            else:
                cw, dq = uniform_codes(y, self.alphas[wi] * sigma, w)
            if want_codes:
                codes_by_w[w] = cw
            dq_by_w[w] = dq
            err2_by_w[w] = (y - dq).square()

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

        sink_bits = nsink * SINK_BITS * d

        ntok = torch.full((P,), float(ptok), dtype=torch.float64, device=dev)
        if pad:
            ntok[-1] = ptok - pad
        Dw64, R64 = Dmat.double(), Rmat.double().clone()
        if pad:
            z = torch.zeros(pad, R, dtype=torch.float64, device=dev)
            Dw64 = torch.cat([Dw64, z])
            R64 = torch.cat([R64, z.clone()])
        budgets = (self.b_page * d * ntok
                   - torch.as_tensor([self._side_bits(int(n))
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
        forced_bits = 0.0
        if nforce:
            for p in range(P - nforce, P):
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

        used = Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2).sum(1)
        left = budgets - used
        for _ in range(8):
            cur_d = Dp.gather(2, assign_p.unsqueeze(2)).squeeze(2)
            cur_r = Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2)
            gain_m = cur_d.unsqueeze(2) - Dp
            cost = Rp - cur_r.unsqueeze(2)
            ok = (cost > 0) & (cost <= left.view(P, 1, 1)) & (gain_m > 0)
            score = torch.where(ok, gain_m / cost, torch.zeros_like(gain_m))
            best = score.view(P, -1).argmax(1)
            best_score = score.view(P, -1).gather(1, best.unsqueeze(1)).squeeze(1)
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
        n_over = P - nforce if nforce else P
        overflow = int((used[:n_over] > budgets[:n_over]).sum())
        payload = float(used.sum()) + sink_bits + forced_bits
        side = float(sum(self._side_bits(int(n)) for n in ntok.tolist()))

        # reconstruction in the mixed-row domain
        y_hat = torch.zeros_like(y)
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
            y_hat[msk] = block
            if want_codes:
                codes[msk] = cblk
        sink_codes = None
        if nsink:
            sidx = (torch.round(r[:nsink] / self.sink_scale)
                    .clamp_(-SINK_LIM, SINK_LIM))
            y_hat[:nsink] = sidx * self.sink_scale
            if want_codes:
                sink_codes = (sidx + SINK_LIM).to(torch.uint8)
                codes[:nsink] = 0        # sink rows live in sink_codes only
        if want_codes:
            emit.update(assign=assign, codes=codes, nsink=nsink,
                        sink_codes=sink_codes, y_hat=y_hat.clone(),
                        sigma=sigma, tmask=tmask.clone(), nfull=nfull)

        # invert the transform on transformed pages (y_hat was cloned into
        # emit above — this in-place pass must not touch the golden copy)
        r_hat = y_hat
        if tmask.any():
            rp = r_hat[: nfull * ptok].reshape(nfull, ptok, d)
            rp[tmask[:nfull]] = torch.einsum(
                "st,psd->ptd", self.dct_m, rp[tmask[:nfull]])
        if want_codes:
            emit["r_hat"] = r_hat

        self.pages_total += int(P)
        self.pages_overflow += int(overflow)
        self.bits_payload += float(payload)
        self.bits_side += float(side)
        self.tokens_total += T
        binc = torch.bincount(assign[nsink:], minlength=R)
        for ri in range(R):
            self.rung_hist[ri] += int(binc[ri])

        out = r_hat @ self.inverse_map + self.mu
        return out.reshape(shape).to(states.dtype)
