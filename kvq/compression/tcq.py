"""pgq3 Thrust-B families (a)/(b)/(e): trellis-coded direction quantization.

Per token, in the qpca_unc code domain (code-SE == Q-weighted key MSE):
fp16 norm n_t (16 bits) + unit direction u_t coded by one of:

  TCQ rungs (family b)    width-w trellis-coded quantization over the d
                          coords: per-coord tables of 2^(w+1) warped
                          Lloyd-Max levels split into 4 subsets (sorted
                          index mod 4), an n_states shift-register trellis
                          (1 branch bit + (w-1) within-subset bits = exactly
                          w bits/coord), Viterbi encode. Tables are fit in
                          the companded domain sign(u)|u|^p and UNWARPED
                          once at fit time, so encode cost is plain squared
                          error against the levels the decoder reproduces
                          (snap-aware by construction).
  1-state control (a)     n_states=1 degenerates to per-coord nearest-level
                          scalar quantization over 2^w warped-LM levels —
                          the compander control isolating the trellis gain.
  sparse-K rungs (e)      top-K |coords| of the direction, each stored as a
                          ceil(log2(d))-bit index + mag_bits absolute-grid
                          magnitude; the other coords decode to exactly 0.
                          Fixed width K*(idx_bits+mag_bits).

Rung 0 evicts (no norm, no payload, k_hat = mu_k). Sinks (positions 0-3)
bypass the ladder with the absolute 8-bit [-1,1] direction grid + fp16 norm
(the pgq2 sink lessons: sink norms AND directions are outliers; norms make
them representable, absolute grids make them codebook-proof). Every table
carries an exact zero level (third-sighting lesson).

Rate per token: R_t = 16 + w*d (TCQ) / 16 + K*(idx_bits+mag_bits) (sparse),
0 for rung 0. Page sideband: HEADER_BITS + id_bits*ptok (rdo) or
HEADER_BITS (uniform). No entropy coding anywhere; decode is table lookups
plus one linear pass over the trellis path bits.
"""
from __future__ import annotations

import math

import torch

from kvq.compression.page_quant import (
    HEADER_BITS, _PagedBase, _paged_lambda_assign,
)

NORM_BITS = 16
N_SUBSETS = 4


def _compand(x: torch.Tensor, p: float) -> torch.Tensor:
    return x.sign() * x.abs().clamp_min(1e-12).pow(p)


def _expand(y: torch.Tensor, p: float) -> torch.Tensor:
    return y.sign() * y.abs().clamp_min(1e-12).pow(1.0 / p)


def fit_warped_lm_tables(u: torch.Tensor, n_levels: int, p: float,
                         iters: int = 15) -> torch.Tensor:
    """Per-coord empirical Lloyd-Max in the companded domain, unwarped.

    u (N, d) unit-direction samples; returns (d, n_levels) sorted level
    tables in the ORIGINAL domain, with the smallest-|level| snapped to
    exactly 0. Adapted from run_pca_leg.py's empirical_lloyd_max, fit on
    c = sign(u)|u|^p so p>1 packs resolution into the tails.
    """
    d = u.shape[1]
    # Batched over coords: (d, N) against (d, V) — bin means of ordered
    # intervals stay sorted, so searchsorted midpoints are valid every iter.
    c = _compand(u, p).double().t().contiguous()          # (d, N)
    qs = torch.linspace(0, 1, n_levels + 2, device=c.device)[1:-1].double()
    cent = torch.quantile(c, qs, dim=1).t().contiguous()  # (d, V)
    ones = torch.ones_like(c)
    tol = 1e-6 * (c.std(dim=1, keepdim=True) + 1e-12)     # (d, 1)
    for _ in range(iters):
        idx = torch.searchsorted((cent[:, 1:] + cent[:, :-1]) / 2, c)
        sums = torch.zeros_like(cent).scatter_add_(1, idx, c)
        cnts = torch.zeros_like(cent).scatter_add_(1, idx, ones)
        new = torch.where(cnts > 0, sums / cnts.clamp_min(1), cent)
        done = (new - cent).abs().amax(dim=1, keepdim=True) < tol
        cent = new
        if bool(done.all()):
            break
    lv = _expand(cent.float(), p).sort(dim=1).values
    lv[torch.arange(d, device=lv.device), lv.abs().argmin(dim=1)] = 0.0
    return lv.cpu()


def _subset_stats(u: torch.Tensor, table: torch.Tensor):
    """Best level per (token, coord, subset). Subset = sorted index mod 4.

    u (T, d), table (d, V) sorted with V % 4 == 0. Returns
    dist_sub (T, d, 4) squared errors and val_sub (T, d, 4) level values.
    """
    T, d = u.shape
    V = table.shape[1]
    diffs = (u.unsqueeze(2) - table.unsqueeze(0)).square()   # (T, d, V)
    diffs = diffs.reshape(T, d, V // N_SUBSETS, N_SUBSETS)
    dist_sub, arg = diffs.min(2)                             # (T, d, 4)
    member = table.reshape(1, d, V // N_SUBSETS, N_SUBSETS).expand(
        T, -1, -1, -1)
    val_sub = member.gather(2, arg.unsqueeze(2)).squeeze(2)
    return dist_sub, val_sub


def _trellis_moves(n_states: int, device):
    """Incoming moves per next-state: next = ((s<<1)|b) % n_states, subset =
    ((s&1)<<1)|b. Returns (NS, 2) prev-state / branch-bit / subset tensors —
    each state has exactly 2 predecessors under the shift register."""
    prev = torch.empty(n_states, 2, dtype=torch.long, device=device)
    bit = torch.empty_like(prev)
    sub = torch.empty_like(prev)
    for ns in range(n_states):
        moves = [(s, b) for s in range(n_states) for b in (0, 1)
                 if ((s << 1) | b) % n_states == ns]
        for i, (s, b) in enumerate(moves[:2]):
            prev[ns, i] = s
            bit[ns, i] = b
            sub[ns, i] = ((s & 1) << 1) | b
    return prev, bit, sub


def tcq_viterbi(u: torch.Tensor, table: torch.Tensor,
                n_states: int) -> torch.Tensor:
    """Best-path trellis quantization of u (T, d) against table (d, V).

    Start state 0 (decoder convention, no extra bits). n_states == 1 is the
    scalar compander control: plain per-coord nearest level.
    """
    T, d = u.shape
    dev = u.device
    if n_states == 1:
        idx = (u.unsqueeze(2) - table.unsqueeze(0)).abs().argmin(2)
        return table.unsqueeze(0).expand(T, -1, -1).gather(
            2, idx.unsqueeze(2)).squeeze(2)

    dist_sub, val_sub = _subset_stats(u, table)
    prev, _, sub = _trellis_moves(n_states, dev)
    alpha = torch.full((T, n_states), torch.inf, device=dev)
    alpha[:, 0] = 0.0
    bp = torch.empty(T, d, n_states, dtype=torch.int8, device=dev)
    for j in range(d):
        cand = alpha[:, prev] + dist_sub[:, j, sub]          # (T, NS, 2)
        alpha, which = cand.min(2)
        bp[:, j] = which.to(torch.int8)
    state = alpha.argmin(1)                                  # (T,)
    ar = torch.arange(T, device=dev)
    vals = torch.empty(T, d, device=dev)
    for j in range(d - 1, -1, -1):
        which = bp[ar, j, state].long()
        vals[:, j] = val_sub[ar, j, sub[state, which]]
        state = prev[state, which]
    return vals


def tcq_greedy(u: torch.Tensor, table: torch.Tensor,
               n_states: int) -> torch.Tensor:
    """Locally-best branch walk over the same trellis/tables — the control
    Viterbi must beat (fit-time sanity gate + unit test)."""
    T, d = u.shape
    dev = u.device
    if n_states == 1:
        return tcq_viterbi(u, table, 1)
    dist_sub, val_sub = _subset_stats(u, table)
    state = torch.zeros(T, dtype=torch.long, device=dev)
    ar = torch.arange(T, device=dev)
    vals = torch.empty(T, d, device=dev)
    for j in range(d):
        sub0 = (state & 1) << 1
        d0 = dist_sub[ar, j, sub0]
        d1 = dist_sub[ar, j, sub0 + 1]
        b = (d1 < d0).long()
        vals[:, j] = val_sub[ar, j, sub0 + b]
        state = ((state << 1) | b) % n_states
    return vals


def sparse_k_quantize(u: torch.Tensor, k_keep: int,
                      mag_bits: int) -> torch.Tensor:
    """Top-k_keep |coords|; magnitudes on the absolute [-1, 1] mid-tread
    grid (2^mag_bits - 1 levels incl. exact zero); the rest decode to 0."""
    lim = (1 << (mag_bits - 1)) - 1
    idx = u.abs().topk(k_keep, dim=1).indices
    q = torch.round(u.gather(1, idx) * lim).clamp(-lim, lim) / lim
    out = torch.zeros_like(u)
    out.scatter_(1, idx, q)
    return out


class TCQPagedCompressor(_PagedBase):
    """mode: "uniform" (single rung, no ids) | "rdo" (per-token rung)."""

    def __init__(self, forward_map, inverse_map, mu, mu_q,
                 tables, tcq_widths, n_states, sparse_ks, mag_bits,
                 b_page, ptok=64, mode="rdo", uniform_rung=None,
                 omega_tau=0.0, omega_clamp_bits=4.0, mixer=None):
        self.forward_map = forward_map.float()
        self.inverse_map = inverse_map.float()
        self.mu = mu.float()
        self.mu_q = mu_q.float()
        # Amendment-1 domain split within one ladder: TCQ rungs code the
        # Hadamard-MIXED direction (tables are fit there), sparse-K rungs
        # and the sink grid code the raw direction (they need compacted
        # energy). The mixer is orthogonal, so both rungs' D live on the
        # same code-SE scale and the Lagrangian comparison stays exact.
        self.mixer = None if mixer is None else mixer.float()
        self.tables = [t.float() for t in tables]     # per width, (d, V_w)
        self.tcq_widths = [int(w) for w in tcq_widths]
        self.n_states = int(n_states)
        self.sparse_ks = [int(k) for k in sparse_ks]
        self.mag_bits = int(mag_bits)
        d = forward_map.shape[0]
        self.idx_bits = max(1, math.ceil(math.log2(d)))
        # rung 0 = evict, then TCQ widths, then sparse-K rungs
        self.n_rungs = 1 + len(self.tcq_widths) + len(self.sparse_ks)
        self.rate_bits = torch.tensor(
            [0.0]
            + [NORM_BITS + w * d for w in self.tcq_widths]
            + [NORM_BITS + k * (self.idx_bits + self.mag_bits)
               for k in self.sparse_ks]).double()
        for w, t in zip(self.tcq_widths, self.tables):
            want = 2 ** (w + 1) if self.n_states > 1 else 2 ** w
            if t.shape != (d, want):
                raise ValueError(
                    f"width-{w} table is {tuple(t.shape)}, want ({d}, {want})")
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

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]
        dev = k.device

        r = (k - self.mu) @ self.forward_map
        # Transmit the RAW-domain norm (second gate sweep: code-domain
        # renorm left raw normR at 0.98-1.05 through the non-orthogonal
        # inverse map). Decoder: k_hat = mu + n16 * w/||w||, w = u_hat @ G.
        # In code domain that is r_hat = n16 * u_hat / ||u_hat @ G|| — one
        # extra per-rung row-norm, priced exactly; raw normR = 1 by
        # construction.
        n16 = (k - self.mu).norm(dim=1).to(torch.float16).float()
        ncode = r.norm(dim=1)
        u = r / ncode.clamp_min(1e-8).unsqueeze(1)

        # sink bypass: absolute 8-bit [-1,1] direction grid + fp16 norm,
        # charged to page 0 (rvq.py pattern; position-derivable, zero
        # sideband)
        nsink = min(4, T)
        u_sink = torch.round(u[:nsink] * 127.0).clamp(-127, 127) / 127.0
        sink_cost = float(nsink) * (NORM_BITS + 8 * d)

        R = self.n_rungs
        Dmat = torch.empty(T, R, device=dev)
        Dmat[:, 0] = r.square().sum(1)                 # eviction
        # Decoded directions come back systematically short (sparse-K zeros
        # most coords; coarse tables underfit tails) — shrunken bulk logits
        # reallocate softmax mass onto the exactly-kept sinks (first pgq3
        # gate sweep). The raw-norm decode above fixes the scale; here each
        # rung's ACTUAL reconstruction r_hat = n16 * u_hat / ||u_hat @ G||
        # is materialized so Dmat prices the exact snap. All-zero rows give
        # r_hat = 0 (priced as eviction; RDO then prefers the cheap rung).
        def _scaled(uh):
            g = (uh @ self.inverse_map).norm(dim=1)
            return uh * (n16 / g.clamp_min(1e-8)).unsqueeze(1)

        um = u @ self.mixer if self.mixer is not None else u
        rhat_by_rung = [None]
        for wi in range(len(self.tcq_widths)):
            uh = tcq_viterbi(um, self.tables[wi], self.n_states)
            if self.mixer is not None:
                uh = uh @ self.mixer.t()               # back to raw direction
            rh = _scaled(uh)
            rhat_by_rung.append(rh)
            Dmat[:, 1 + wi] = (r - rh).square().sum(1)
        for ki, kk in enumerate(self.sparse_ks):
            rh = _scaled(sparse_k_quantize(u, kk, self.mag_bits))
            rhat_by_rung.append(rh)
            Dmat[:, 1 + len(self.tcq_widths) + ki] = \
                (r - rh).square().sum(1)
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

        if self.mode == "uniform":
            ri = int(self.uniform_rung)
            assign = torch.full((T,), ri, dtype=torch.long, device=dev)
            assign[:nsink] = R - 1   # hist bucket only; recon/rate special
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

        r_hat = torch.zeros_like(r)
        for ri in range(1, R):
            msk = assign == ri
            msk[:nsink] = False
            if msk.any():
                r_hat[msk] = rhat_by_rung[ri][msk]
        if nsink:
            g = (u_sink @ self.inverse_map).norm(dim=1)
            r_hat[:nsink] = u_sink * (n16[:nsink]
                                      / g.clamp_min(1e-8)).unsqueeze(1)
        out = r_hat @ self.inverse_map + self.mu
        return out.reshape(shape).to(states.dtype)
