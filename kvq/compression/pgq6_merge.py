"""pgq6: merged-page codec — n→m token merging as a rung of the paged RDO.

Extends the pgq4 profile-rung format (plan6): each page may store m < ptok
rows, where contiguous position runs are merged into centroids. The bias form
(k̄, v̄, β = log c) is mathematically identical to replicating the centroid c
times at the member positions — softmax over c identical keys equals one key
with a +log c logit — so `roundtrip` returns same-shape K̂ with centroids
replicated and every existing harness (screening, bench worker, Mode-A/B')
works unchanged. Deployment stores m rows + a 6-bit count per row (the
contiguous-run map IS the counts) + the page's merge-class id; the bias folds
into a (d+1)-th key coordinate, so FlashAttention needs no bias support.

Fixed-byte pages are preserved: every page keeps the SAME bit budget
(b_page·d·ntok); merging frees rows so the per-row width RDO spends the freed
bits on wider grids for the surviving rows (or, in deployment, the page drops
to a power-of-2 size class — M ∈ {64, 32, 16} → {S, S/2, S/4}). Merge level
is chosen PER PAGE by comparing the achieved RDO distortion of each level
under the identical budget (the levels are nested snapshots of one adjacent
Ward merge, so comparisons are consistent).

Clustering is adjacent-only (contiguous runs) Ward on the rotated codes r:
in the qpca_unc basis, unweighted r-space distance IS Σ_Q-weighted raw-key
distance (the basis whitens by query energy), so the merge objective matches
the RDO's code-space MSE convention. Exact per-row distortion decomposition
(cross term vanishes over a cluster):

    Σ_{i∈S} ||r_i − Q(r̄)||² = Σ_{i∈S} ||r_i − r̄||²  +  c·||r̄ − Q(r̄)||²
                              (spread, width-independent)   (quantization)

Sink rows (positions 0-3) keep the absolute escape and page 0 is forced
unmerged; force_recent_pages stay unmerged at the top width rung. All
signals are phase-symmetric (calibration + position only).
"""
from __future__ import annotations

import math

import torch

from kvq.compression.page_quant import HEADER_BITS, _paged_lambda_assign
from kvq.compression.pgq4_folded import (
    GAIN_BITS, SINK_BITS, SINK_LIM, FoldedScalarPagedCompressor,
)

PGQ6_COUNT_BITS = 6          # run length c-1 per stored row (c <= 64)
PGQ6_MCHOICE_BITS = 2        # page merge-class id
MERGE_LEVELS = (64, 32, 16)  # power-of-2 page size classes {S, S/2, S/4}
_INF = float("inf")


class MergedPageCompressor(FoldedScalarPagedCompressor):
    """Per-page (merge level × per-row width rung) RDO over fixed-byte pages.

    v1 contract: mode='rdo' only, gain unsupported, omega weighting
    unsupported (protection arms are a later ablation) — asserted, not
    silently ignored.
    """

    supports_start_pos = True

    def __init__(self, *args, merge_levels=None, **kw):
        super().__init__(*args, **kw)
        assert self.mode == "rdo", "pgq6 v1 is RDO-only"
        assert not self.gain, "pgq6 v1 has no gain variant"
        assert self.omega_tau == 0.0, "pgq6 v1 is omega-free (plan6)"
        if merge_levels is None:
            # power-of-2 size classes {S, S/2, S/4} scaled to the page size
            merge_levels = (self.ptok, self.ptok // 2, self.ptok // 4)
        self.merge_levels = tuple(sorted((int(m) for m in merge_levels),
                                         reverse=True))
        assert self.merge_levels[0] == self.ptok, \
            "merge_levels must include the unmerged level (= ptok)"
        self.reset_stats()

    def reset_stats(self):
        super().reset_stats()
        # tokens living in pages of each merge level, aligned to merge_levels
        self.merge_hist = [0] * len(getattr(self, "merge_levels", ()))
        self.rows_total = 0                    # stored rows (m per page)

    # ---- clustering ------------------------------------------------------

    def _merge_hierarchy(self, rp, valid, protect=None):
        """Adjacent Ward merge, vectorized across pages. rp (P, n, d),
        valid (P, n). protect (P, n) bool: protected slots never merge (the
        screening ORACLE's hook — production passes None; plan6 signal
        policy). Returns {M: (labels, cnt, ssum)} nested snapshots; labels
        map each token slot to its run-head slot."""
        P, n, d = rp.shape
        dev = rp.device
        cnt = valid.to(rp.dtype)
        ssum = rp * valid.unsqueeze(2)
        lab = torch.arange(n, device=dev).expand(P, n).contiguous()
        idx = torch.arange(n, device=dev)
        nxt = torch.where(
            (idx < n - 1).expand(P, n)
            & valid.roll(-1, dims=1) & valid,
            (idx + 1).expand(P, n),
            torch.full((P, n), -1, dtype=torch.long, device=dev)).clone()

        # NOTE: every step must stay sync-free (no .item()/nonzero/python
        # branches on tensor values) — a data-dependent sync here costs
        # 48 x 248 (l,h) round-trips per prompt and was a 25x bench
        # slowdown. Pages that cannot merge execute masked no-ops.
        pidx = torch.arange(P, device=rp.device)
        snaps = {}
        steps_done = 0
        for M in self.merge_levels:
            while steps_done < n - M:
                mean = ssum / cnt.clamp_min(1e-12).unsqueeze(2)
                j = nxt.clamp_min(0)
                cj = cnt.gather(1, j)
                mj = mean.gather(1, j.unsqueeze(2).expand(-1, -1, d))
                w = cnt * cj / (cnt + cj).clamp_min(1e-12)
                diff2 = (mean - mj).square().sum(2)
                ok = (nxt >= 0) & (cnt > 0) & (cj > 0)
                if protect is not None:
                    pj = protect.gather(1, j)
                    ok = ok & ~protect & ~pj
                cost = torch.where(ok, w * diff2,
                                   torch.full_like(diff2, _INF))
                a = cost.argmin(1)
                can = torch.isfinite(cost.gather(1, a.unsqueeze(1))
                                     .squeeze(1))
                b = nxt[pidx, a].clamp_min(0)
                canf = can.to(rp.dtype)
                ssum[pidx, a] += ssum[pidx, b] * canf.unsqueeze(1)
                cnt[pidx, a] += cnt[pidx, b] * canf
                cnt[pidx, b] = cnt[pidx, b] * (1.0 - canf)
                relabel = (lab == b.unsqueeze(1)) & can.unsqueeze(1)
                lab = torch.where(relabel, a.unsqueeze(1), lab)
                nxt[pidx, a] = torch.where(can, nxt[pidx, b], nxt[pidx, a])
                steps_done += 1
            snaps[M] = (lab.clone(), cnt.clone(), ssum.clone())
        return snaps

    # ---- roundtrip -------------------------------------------------------

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor, start_pos: int = 0,
                  token_weights: torch.Tensor | None = None,
                  protect_mask: torch.Tensor | None = None) -> torch.Tensor:
        # token_weights / protect_mask are SCREENING-ORACLE hooks (observed
        # attention upper bound, plan6 sec.2) — the production path passes
        # neither and is bit-identical to the phase-symmetric format.
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]
        dev = k.device
        r = (k - self.mu) @ self.forward_map
        R = self.n_rungs
        n = self.ptok
        P = (T + n - 1) // n
        pad = P * n - T

        rp = torch.cat([r, torch.zeros(pad, d, device=dev)]) \
            .reshape(P, n, d)
        valid = torch.arange(P * n, device=dev).reshape(P, n) < T
        ntok = valid.sum(1).double()
        wp = None
        if token_weights is not None:
            wp = torch.cat([token_weights.to(dev).float(),
                            torch.zeros(pad, device=dev)]).reshape(P, n)
        prot = None
        if protect_mask is not None:
            prot = torch.cat([protect_mask.to(dev),
                              torch.zeros(pad, dtype=torch.bool,
                                          device=dev)]).reshape(P, n)

        snaps = self._merge_hierarchy(rp, valid, protect=prot)

        nsink = min(4, T) if start_pos == 0 else 0
        sink_bits = nsink * SINK_BITS * d

        nlev = len(self.merge_levels)
        widths_pos = self.widths_pos
        prof = self.profiles.to(dev)

        # per level: per-row D over rungs, per-row rates, budgets, RDO assign
        lvl = {}
        for li, M in enumerate(self.merge_levels):
            lab, cnt, ssum = snaps[M]
            alive = cnt > 0
            mean = ssum / cnt.clamp_min(1e-12).unsqueeze(2)
            cent_tok = mean.gather(1, lab.unsqueeze(2).expand(-1, -1, d))
            spread_tok = (rp - cent_tok).square().sum(2) \
                * valid.to(rp.dtype)
            if wp is not None:
                spread_tok = spread_tok * wp
            spread_row = torch.zeros(P, n, device=dev) \
                .scatter_add_(1, lab, spread_tok)

            cent = mean.reshape(-1, d)
            err2_by_w = {0: cent.square()}
            dq_by_w = {0: torch.zeros_like(cent)}
            for wi, w in enumerate(widths_pos):
                dq = self._quant_width(cent, w, wi)
                dq_by_w[w] = dq
                err2_by_w[w] = (cent - dq).square()

            Drow = torch.empty(P * n, R, device=dev)
            for ri in range(R):
                row = prof[ri]
                acc = torch.zeros(P * n, device=dev)
                for w in self.width_ladder:
                    cols = row == w
                    if cols.any():
                        acc += err2_by_w[int(w)][:, cols].sum(1)
                Drow[:, ri] = acc
            if wp is None:
                rowmul = cnt
            else:
                rowmul = torch.zeros(P, n, device=dev) \
                    .scatter_add_(1, lab, wp * valid.to(wp.dtype))
            Drow = Drow.reshape(P, n, R) * rowmul.unsqueeze(2) \
                + spread_row.unsqueeze(2)
            Drow = torch.where(alive.unsqueeze(2), Drow,
                               torch.zeros_like(Drow)).double()

            row_rate = self.rung_rate.to(dev)
            if M < n:
                row_rate = row_rate + PGQ6_COUNT_BITS
            Rrow = torch.where(
                alive.unsqueeze(2),
                row_rate.unsqueeze(0).unsqueeze(0).expand(P, n, R),
                torch.zeros(P, n, R, dtype=torch.float64, device=dev))

            m_alive = alive.sum(1).double()
            side = (HEADER_BITS + PGQ6_MCHOICE_BITS
                    + self.id_bits * m_alive)
            budgets = (self.b_page * d * ntok - side).clone()

            Dw = Drow.clone()
            Rw = Rrow.clone()
            if nsink:
                # sink rows are singletons on the forced-unmerged page 0
                if M == n:
                    Dw[0, :nsink] = 0.0
                    Rw[0, :nsink] = 0.0
                    budgets[0] -= sink_bits
            assign = _paged_lambda_assign(Dw, Rw, budgets, n)
            usedD = Dw.gather(2, assign.unsqueeze(2)).squeeze(2).sum(1)
            usedR = Rw.gather(2, assign.unsqueeze(2)).squeeze(2).sum(1)
            lvl[li] = dict(M=M, lab=lab, alive=alive, assign=assign,
                           usedD=usedD, usedR=usedR, budgets=budgets,
                           Dw=Dw, Rw=Rw, side=side,
                           dq_by_w=dq_by_w)

        # ---- per-page merge-level choice (min achieved distortion; ties →
        # the earlier level, i.e. less merged) --------------------------------
        allD = torch.stack([lvl[li]["usedD"] for li in range(nlev)], 1)
        choice = allD.argmin(1)                                   # (P,)
        choice[0] = 0 if nsink else choice[0]     # sink page unmerged
        top = R - 1
        nforce = 0
        if start_pos == 0 and self.force_recent_pages > 0 and P > 1:
            nforce = min(self.force_recent_pages, P - 1)
            choice[P - nforce:] = 0               # forced pages unmerged

        # gather the chosen configuration per page
        pidx = torch.arange(P, device=dev)
        Dp = torch.stack([lvl[li]["Dw"] for li in range(nlev)])[choice, pidx]
        Rp = torch.stack([lvl[li]["Rw"] for li in range(nlev)])[choice, pidx]
        assign_p = torch.stack(
            [lvl[li]["assign"] for li in range(nlev)])[choice, pidx]
        budgets = torch.stack(
            [lvl[li]["budgets"] for li in range(nlev)])[choice, pidx].clone()
        side_p = torch.stack(
            [lvl[li]["side"] for li in range(nlev)])[choice, pidx]

        forced_bits = 0.0
        if nforce:
            for p in range(P - nforce, P):
                alive_p = lvl[0]["alive"][p]
                cost = float(Rp[p, alive_p, top].sum())
                forced_bits += cost
                budgets[p] -= cost
                Dp[p] = 0.0
                Rp[p] = 0.0
                assign_p[p] = top

        # greedy refinement on the chosen configs (parent convention)
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

        used = Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2).sum(1)
        n_over = P - nforce if nforce else P
        overflow = int((used[:n_over] > budgets[:n_over]).sum())
        payload = float(used.sum()) + sink_bits + forced_bits

        # ---- reconstruction: replicate quantized centroids ---------------
        r_hat = torch.zeros(P, n, d, device=dev)
        rows_total = 0
        for li in range(nlev):
            pmask = choice == li
            if not pmask.any():
                continue
            L = lvl[li]
            dq_by_w = L["dq_by_w"]
            lab = L["lab"]
            rows_total += int(L["alive"][pmask].sum())
            rhat_rows = torch.zeros(P, n, d, device=dev)
            asg = assign_p                                   # (P, n)
            for ri in range(R):
                rmask = pmask.unsqueeze(1) & (asg == ri) & L["alive"]
                if not rmask.any():
                    continue
                row = prof[ri]
                flat = rmask.reshape(-1)
                block = torch.zeros(int(flat.sum()), d, device=dev)
                for w in widths_pos:
                    cols = row == w
                    if cols.any():
                        block[:, cols] = dq_by_w[int(w)][flat][:, cols]
                rhat_rows.reshape(-1, d)[flat] = block
            rep = rhat_rows.gather(1, lab.unsqueeze(2).expand(-1, -1, d))
            r_hat[pmask] = rep[pmask]
            self.merge_hist[li] += int(
                (valid & pmask.unsqueeze(1)).sum())

        r_hat = r_hat.reshape(P * n, d)[:T]
        if nsink:
            r_hat[:nsink] = (torch.round(r[:nsink] / self.sink_scale)
                             .clamp_(-SINK_LIM, SINK_LIM) * self.sink_scale)

        self.pages_total += int(P)
        self.pages_overflow += int(overflow)
        self.bits_payload += float(payload)
        self.bits_side += float(side_p.sum())
        self.tokens_total += T
        self.rows_total += rows_total
        # rung_hist counts TOKENS at each width rung (rows weighted by run
        # length) so histograms stay comparable with pgq4
        lab_all = torch.stack(
            [lvl[li]["lab"] for li in range(nlev)])[choice, pidx]
        tok_rung = assign_p.gather(1, lab_all)[valid]
        binc = torch.bincount(tok_rung[nsink:] if start_pos == 0
                              else tok_rung, minlength=R)
        for ri in range(R):
            self.rung_hist[ri] += int(binc[ri])

        out = r_hat @ self.inverse_map + self.mu
        return out.reshape(shape).to(states.dtype)
