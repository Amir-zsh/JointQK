"""Paged per-token-precision K compressors (page_quant study).

Two families, both operating on fixed-byte pages of PTOK tokens in the
qpca_unc code domain (r = (k - mu) @ F; code-domain SE == Q-weighted key MSE):

  PagedRDOCompressor      deadzone rung ladder + frozen-model entropy coding.
                          Per-token rung chosen by per-page Lagrangian RDO,
                          optionally distortion-weighted by an encoder-side
                          importance omega_t = exp(tau*clamp(mu_q.k/sqrt(d)
                          - median, +-c*ln2)). Decoder needs only the packed
                          rung ids (id_bits/token) — omega costs no sideband.
                          Modes: "rdo" (per-token rungs) and "pagerung" (one
                          rung per page — the fixed-page token-uniform
                          control).

  FixedWidthPagedCompressor  saturating uniform grids at widths {0,1,2,4}
                          bits/coord (b=0 reconstructs mu_k), per-coord scale
                          = alpha_w * code_std. Exact R_t = w*d, greedy
                          refinement fills the page budget. No entropy
                          coding: decode is shift/mask — the fused-kernel
                          target.

Both live in kvq/ (not entropy_coding/) so press disk-cache pickles re-import
cleanly (same reason as ec_roundtrip.py). Allocation runs in fp64 with
first-index-wins argmin so a CPU/GPU pair is deterministic. Distortion is
snap-aware: D uses the frozen-alphabet reconstruction the decoder actually
produces, not the raw rounding (allocation must see OOD clipping).

Rate accounting (charged against every page's budget):
  header 96 bits + id_bits*ptok rung ids [+ 72*N_LANES rANS lane overhead for
  the EC family]. Calibration constants (delta, alphabets, std, alpha) are
  off-page, like every method's codebooks; per-vector data always in-page.
"""
from __future__ import annotations

import math

import torch

from kvq.compression.ec_roundtrip import dz_round, dz_dequant

PGQ_BUNDLE_VERSION = 2
PGQ2_BUNDLE_VERSION = 3
PGQ3_BUNDLE_VERSION = 4
N_LANES = 4
HEADER_BITS = 96
LANE_OVERHEAD_BITS = 72  # per lane: u32 length + u32 final state + slack


def snap_positions(idx: torch.Tensor, support_vals: torch.Tensor,
                   support_lens: torch.Tensor) -> torch.Tensor:
    """Nearest-alphabet positions (tie -> left), per coord.

    idx (N, d) float; support_vals (d, V) ascending, +inf padded;
    support_lens (d,) long. Returns positions (N, d) long. Exact replica of
    _CoordModel.snap / SnappedDeadzoneECCompressor.snap_indices semantics.
    """
    n, d = idx.shape
    cols = idx.t().contiguous()                       # (d, N)
    pos = torch.searchsorted(support_vals, cols)      # (d, N)
    maxpos = (support_lens - 1).clamp_min(0).unsqueeze(1)
    pos = pos.clamp_max_(maxpos.expand_as(pos)).clamp_min_(0)
    left = (pos - 1).clamp_min(0)
    vr = support_vals.gather(1, pos)
    vl = support_vals.gather(1, left)
    choose_left = (idx.t() - vl) <= (vr - idx.t())    # tie -> left
    pos = torch.where(choose_left & (pos > 0), left, pos)
    single = (support_lens <= 1).unsqueeze(1)
    pos = torch.where(single, torch.zeros_like(pos), pos)
    return pos.t().contiguous()


def bit_reversal_perm(n: int) -> torch.Tensor:
    """Index permutation by reversed bit order (n a power of 2)."""
    bits = n.bit_length() - 1
    return torch.tensor([int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)],
                        dtype=torch.long)


def build_hadamard(d: int) -> torch.Tensor:
    """Orthonormal Sylvester-Hadamard with bit-reversed rows (d a power of
    2). Orthogonal in code space, so composing it after qpca_unc preserves
    code-SE == Q-weighted key MSE exactly while mixing coordinate energy
    (OSCAR amendment 1: isotropizes subvectors for E8/TCQ shared scales)."""
    if d & (d - 1):
        raise ValueError(f"d={d} is not a power of 2")
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h[bit_reversal_perm(d)] / math.sqrt(d)


def _paged_lambda_assign(Dw64, R64, budgets, ptok, iters=40):
    """Per-page bisection over lambda. Dw64/R64: (P, ptok, R) fp64 (padded
    tokens carry zero D and zero R so they never affect sums). budgets (P,)
    fp64. Returns (P, ptok) long assignment (first-index-wins argmin)."""
    P = Dw64.shape[0]
    lo = torch.zeros(P, 1, 1, dtype=torch.float64, device=Dw64.device)
    hi = torch.full_like(lo, max(1.0, float(Dw64.max())) * 1e3)
    rmin = R64.argmin(2)
    fits_min = R64.gather(2, rmin.unsqueeze(2)).squeeze(2).sum(1) <= budgets
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = (Dw64 + mid * R64).argmin(2)
        rr = R64.gather(2, a.unsqueeze(2)).squeeze(2).sum(1)
        over = (rr > budgets).view(P, 1, 1)
        lo = torch.where(over, mid, lo)
        hi = torch.where(over, hi, mid)
    a = (Dw64 + hi * R64).argmin(2)
    # pages where even the min-rate choice overflows: take min-rate (counted
    # as overflow by the caller)
    a = torch.where(fits_min.unsqueeze(1), a, rmin)
    return a


class _PagedBase:
    def to(self, device):
        for name, val in list(self.__dict__.items()):
            if torch.is_tensor(val):
                setattr(self, name, val.to(device))
            elif isinstance(val, list) and val and torch.is_tensor(val[0]):
                setattr(self, name, [t.to(device) for t in val])
        self.device = device
        return self

    def _maybe_migrate(self, states):
        if states.device != getattr(self, "device", states.device):
            self.to(states.device)

    def reset_stats(self):
        self.pages_total = 0
        self.pages_overflow = 0
        self.bits_payload = 0.0
        self.bits_side = 0.0
        self.tokens_total = 0
        self.rung_hist = [0] * self.n_rungs


class PagedRDOCompressor(_PagedBase):
    """EC family: deadzone rung ladder, per-page RDO, frozen-model rates."""

    def __init__(self, forward_map, inverse_map, mu, mu_q, delta_base,
                 m_grid, support_vals, support_lens, support_nlp,
                 dz, b_page, ptok=64, mode="rdo",
                 omega_tau=0.0, omega_clamp_bits=4.0):
        self.forward_map = forward_map.float()
        self.inverse_map = inverse_map.float()
        self.mu = mu.float()
        self.mu_q = mu_q.float()
        self.delta_base = float(delta_base)
        self.m_grid = [float(m) for m in m_grid]
        self.n_rungs = len(self.m_grid)
        # lists of (d, V_r) tensors, V_r per rung
        self.support_vals = [v.float() for v in support_vals]
        self.support_lens = [l.long() for l in support_lens]
        self.support_nlp = [p.float() for p in support_nlp]
        self.dz = float(dz)
        self.b_page = float(b_page)
        self.ptok = int(ptok)
        self.mode = mode
        self.omega_tau = float(omega_tau)
        self.omega_clamp_bits = float(omega_clamp_bits)
        self.id_bits = max(1, math.ceil(math.log2(max(2, self.n_rungs))))
        self.device = mu.device
        self.reset_stats()

    def _side_bits(self, ntok):
        return (HEADER_BITS + self.id_bits * ntok
                + N_LANES * LANE_OVERHEAD_BITS)

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]
        dev = k.device

        r = (k - self.mu) @ self.forward_map
        R = self.n_rungs
        Dmat = torch.empty(T, R, device=dev)
        Rmat = torch.empty(T, R, device=dev)
        vals_by_rung = []
        for ri in range(R):
            delta = self.delta_base * self.m_grid[ri]
            idx = dz_round(r, torch.as_tensor(delta, device=dev), self.dz)
            pos = snap_positions(idx, self.support_vals[ri],
                                 self.support_lens[ri])
            v = self.support_vals[ri].gather(
                1, pos.t()).t()                       # snapped index values
            dq = dz_dequant(v, torch.as_tensor(delta, device=dev), self.dz)
            Dmat[:, ri] = (r - dq).square().sum(1)    # snap-aware distortion
            Rmat[:, ri] = self.support_nlp[ri].gather(1, pos.t()).t().sum(1)
            vals_by_rung.append(v)

        ptok = self.ptok
        P = (T + ptok - 1) // ptok
        pad = P * ptok - T
        ntok = torch.full((P,), float(ptok), dtype=torch.float64, device=dev)
        if pad:
            ntok[-1] = ptok - pad
        page_bits = self.b_page * d * ntok
        budgets = page_bits - torch.as_tensor(
            [self._side_bits(int(n)) for n in ntok.tolist()],
            dtype=torch.float64, device=dev)

        if self.mode == "pagerung":
            # one rung per page: finest whose frozen-model page cost fits
            R64 = Rmat.double()
            if pad:
                R64 = torch.cat([R64, torch.zeros(pad, R, device=dev,
                                                  dtype=torch.float64)])
            page_cost = R64.reshape(P, ptok, R).sum(1)          # (P, R)
            fits = page_cost <= budgets.unsqueeze(1)
            first_fit = torch.where(
                fits.any(1), fits.float().argmax(1),
                torch.full((P,), R - 1, dtype=torch.long, device=dev))
            assign_p = first_fit.unsqueeze(1).expand(P, ptok)
            chosen_cost = page_cost.gather(1, first_fit.unsqueeze(1)).squeeze(1)
            overflow = ~fits.any(1)
        else:
            if self.omega_tau > 0.0:
                m_sig = (k @ self.mu_q) / math.sqrt(d)
                c = self.omega_clamp_bits * math.log(2.0)
                om = torch.exp((self.omega_tau * (m_sig - m_sig.median()))
                               .clamp(-c, c))
                Dw = Dmat * om.unsqueeze(1)
            else:
                Dw = Dmat
            Dw64, R64 = Dw.double(), Rmat.double()
            if pad:
                z = torch.zeros(pad, R, dtype=torch.float64, device=dev)
                Dw64 = torch.cat([Dw64, z])
                R64 = torch.cat([R64, z.clone()])
            assign_p = _paged_lambda_assign(
                Dw64.reshape(P, ptok, R), R64.reshape(P, ptok, R),
                budgets, ptok)
            chosen_cost = R64.reshape(P, ptok, R).gather(
                2, assign_p.unsqueeze(2)).squeeze(2).sum(1)
            overflow = chosen_cost > budgets

        assign = assign_p.reshape(-1)[:T]
        # stats + honest rate telemetry
        self.pages_total += P
        self.pages_overflow += int(overflow.sum())
        self.bits_payload += float(chosen_cost.sum())
        self.bits_side += float(sum(self._side_bits(int(n))
                                    for n in ntok.tolist()))
        self.tokens_total += T
        binc = torch.bincount(assign, minlength=R)
        for ri in range(R):
            self.rung_hist[ri] += int(binc[ri])

        r_hat = torch.empty_like(r)
        for ri in range(R):
            msk = assign == ri
            if msk.any():
                delta = torch.as_tensor(self.delta_base * self.m_grid[ri],
                                        device=dev)
                r_hat[msk] = dz_dequant(vals_by_rung[ri][msk], delta, self.dz)
        out = r_hat @ self.inverse_map + self.mu
        return out.reshape(shape).to(states.dtype)


class FixedWidthPagedCompressor(_PagedBase):
    """No-rANS family: per-token width w in widths, saturating uniform grids,
    per-coord scale alpha_w * code_std. Exact rates; greedy refinement fills
    the page. Width 0 reconstructs r=0 (k_hat = mu_k)."""

    def __init__(self, forward_map, inverse_map, mu, mu_q, code_std,
                 widths, alphas, b_page, ptok=64,
                 omega_tau=0.0, omega_clamp_bits=4.0):
        self.forward_map = forward_map.float()
        self.inverse_map = inverse_map.float()
        self.mu = mu.float()
        self.mu_q = mu_q.float()
        self.code_std = code_std.float().clamp_min(1e-12)
        self.widths = [int(w) for w in widths]
        self.n_rungs = len(self.widths)
        self.alphas = alphas.float()                  # (W,)
        self.b_page = float(b_page)
        self.ptok = int(ptok)
        self.omega_tau = float(omega_tau)
        self.omega_clamp_bits = float(omega_clamp_bits)
        self.id_bits = max(1, math.ceil(math.log2(max(2, self.n_rungs))))
        self.device = mu.device
        self.reset_stats()

    def _side_bits(self, ntok):
        return HEADER_BITS + self.id_bits * ntok

    def _cents_for(self, wi):
        if not hasattr(self, "_cents"):
            from kvq.compression.per_coord import unit_gaussian_centroids
            self._cents = [unit_gaussian_centroids(w).sort().values
                           if 0 < w < self.widths[-1] else torch.zeros(1)
                           for w in self.widths]
        c = self._cents[wi]
        if c.device != self.code_std.device:
            self._cents[wi] = c = c.to(self.code_std.device)
        return c

    def _quant_w(self, r, wi):
        # Bulk widths use the project's Lloyd-Max Gaussian codebooks (zero
        # level + std-scaled — crude uniform grids at <=2 bits clipped bulk
        # coords and cost double-digit F1; see fixes_to_apply.md). The TOP
        # width keeps a covering uniform grid: Lloyd-Max Gaussian levels stop
        # at ~3.7 sigma, but sink coords run 3-5x past q999, and the top
        # width is the tail escape (sequence positions 0-3 are forced here).
        w = self.widths[wi]
        if w == 0:
            return torch.zeros_like(r)
        if w == self.widths[-1]:
            s = (self.alphas[wi] * self.code_std).unsqueeze(0)
            lim = (1 << (w - 1)) - 1
            idx = torch.round(r / s).clamp(-lim, lim)
            return idx * s
        cents = self._cents_for(wi)
        sig = (self.code_std / 3.29).unsqueeze(0)     # q999 -> sigma approx
        y = r / sig
        pos = torch.bucketize(y, cents).clamp_max(cents.numel() - 1)
        left = (pos - 1).clamp_min(0)
        pick_left = (y - cents[left]) < (cents[pos] - y)
        pos = torch.where(pick_left & (pos > 0), left, pos)
        return cents[pos] * sig

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        self._maybe_migrate(states)
        shape = states.shape
        d = shape[-1]
        k = states.reshape(-1, d).float()
        T = k.shape[0]
        dev = k.device
        r = (k - self.mu) @ self.forward_map

        W = self.n_rungs
        Dmat = torch.empty(T, W, device=dev)
        rhat_by_w = []
        for wi in range(W):
            dq = self._quant_w(r, wi)
            Dmat[:, wi] = (r - dq).square().sum(1)
            rhat_by_w.append(dq)
        rates = torch.as_tensor([w * d for w in self.widths],
                                dtype=torch.float64, device=dev)
        Rmat = rates.unsqueeze(0).expand(T, W)

        ptok = self.ptok
        P = (T + ptok - 1) // ptok
        pad = P * ptok - T
        ntok = torch.full((P,), float(ptok), dtype=torch.float64, device=dev)
        if pad:
            ntok[-1] = ptok - pad
        budgets = (self.b_page * d * ntok
                   - torch.as_tensor([self._side_bits(int(n))
                                      for n in ntok.tolist()],
                                     dtype=torch.float64, device=dev))
        if self.omega_tau > 0.0:
            m_sig = (k @ self.mu_q) / math.sqrt(d)
            c = self.omega_clamp_bits * math.log(2.0)
            om = torch.exp((self.omega_tau * (m_sig - m_sig.median()))
                           .clamp(-c, c))
            Dw = Dmat * om.unsqueeze(1)
        else:
            Dw = Dmat
        Dw64 = Dw.double()
        R64 = Rmat.double().clone()
        if pad:
            z = torch.zeros(pad, W, dtype=torch.float64, device=dev)
            Dw64 = torch.cat([Dw64, z])
            R64 = torch.cat([R64, z.clone()])
        Dp, Rp = Dw64.reshape(P, ptok, W), R64.reshape(P, ptok, W)
        # Attention sinks (first 4 sequence positions) are 3-5x beyond q999:
        # every narrow width clips them and the Lagrangian won't buy w_max at
        # tight budgets, yet corrupting them poisons the softmax denominator
        # globally (F1 collapse; see fixes_to_apply.md). Standard practice
        # (KIVI keeps sinks high-precision): force w_max for positions 0-3.
        # Position-based => decoder-derivable, zero sideband; page 0 pays.
        nsink = min(4, T)
        wmax = W - 1
        sink_bits = float(Rp[0, :nsink, wmax].sum())
        budgets = budgets.clone()
        Dp = Dp.clone()
        Dp[0, :nsink, :] = 0.0          # allocator ignores them below
        Rp = Rp.clone()
        Rp[0, :nsink, :] = 0.0
        budgets[0] = budgets[0] - sink_bits
        assign_p = _paged_lambda_assign(Dp, Rp, budgets, ptok)
        assign_p[0, :nsink] = wmax

        # greedy refinement: spend leftover bits on the best D-drop per bit
        # (one token upgraded per page per round; 8 rounds converge in
        # practice since the bisection already lands near the budget)
        used = Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2).sum(1)
        left = budgets - used
        for _ in range(8):
            cur_d = Dp.gather(2, assign_p.unsqueeze(2)).squeeze(2)
            cur_r = Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2)
            gain = cur_d.unsqueeze(2) - Dp
            cost = Rp - cur_r.unsqueeze(2)
            ok = (cost > 0) & (cost <= left.view(P, 1, 1)) & (gain > 0)
            score = torch.where(ok, gain / cost, torch.zeros_like(gain))
            best = score.view(P, -1).argmax(1)
            best_score = score.view(P, -1).gather(
                1, best.unsqueeze(1)).squeeze(1)
            pi = torch.nonzero(best_score > 0).squeeze(1)
            if pi.numel() == 0:
                break
            bt = (best[pi] // W).long()
            bw = (best[pi] % W).long()
            cur_w = assign_p[pi, bt]
            ar = torch.arange(pi.numel(), device=dev)
            old_r = Rp[pi, bt][ar, cur_w]
            new_r = Rp[pi, bt][ar, bw]
            assign_p[pi, bt] = bw
            left[pi] -= (new_r - old_r)

        assign = assign_p.reshape(-1)[:T]
        used = Rp.gather(2, assign_p.unsqueeze(2)).squeeze(2).sum(1)
        self.pages_total += P
        self.pages_overflow += int((used > budgets).sum())
        self.bits_payload += float(used.sum()) + sink_bits
        self.bits_side += float(sum(self._side_bits(int(n))
                                    for n in ntok.tolist()))
        self.tokens_total += T
        binc = torch.bincount(assign, minlength=W)
        for wi in range(W):
            self.rung_hist[wi] += int(binc[wi])

        r_hat = torch.empty_like(r)
        for wi in range(W):
            msk = assign == wi
            if msk.any():
                r_hat[msk] = rhat_by_w[wi][msk]
        out = r_hat @ self.inverse_map + self.mu
        return out.reshape(shape).to(states.dtype)


def load_pgq_compressors_from_bundle(path, k_method, b_page):
    """Build per-(layer, head) compressors for layers 1..L-1 (layer-0 fp16).

    v2 methods: pgq_ea | pgq_plain | pgq_fixed | pgq_fixed_ea | pgq_pagerung.
    v3 (pgq2) methods: pgq_nd_{uni,rdo,ea,eag} | pgq_rvq_{uni,rdo,ea,eag} —
    routed to the norm-direction / residual-VQ families. v4 (pgq3) methods:
    pgq_tcq_{uni,rdo,ea} | pgq_e8_{uni,rdo,ea} | pgq_oscar_uni — routed to
    the trellis / E8-lattice / OSCAR-emulation families. For *_uni the
    press's k_bits carries the RUNG INDEX (w or stage count, rate is fixed
    by construction); otherwise k_bits is the page budget in bits/coord.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if k_method.startswith(("pgq_fold", "pgq_prof")):
        return _load_pgq4(blob, path, k_method, b_page)
    if k_method.startswith(("pgq_tcq_", "pgq_e8_", "pgq_oscar_")):
        return _load_pgq3(blob, path, k_method, b_page)
    if k_method.startswith(("pgq_nd_", "pgq_rvq_")):
        return _load_pgq2(blob, path, k_method, b_page)
    if blob.get("pgq_version") != PGQ_BUNDLE_VERSION:
        raise ValueError(f"pgq bundle version mismatch at {path}")
    L, H = blob["n_layers"], blob["n_kv_heads"]
    tau = float(blob["omega_tau"])
    cb = float(blob["omega_clamp_bits"])
    ptok = int(blob["ptok"])
    comps = {}
    for l in range(1, L):
        for h in range(H):
            args = dict(
                forward_map=blob["forward"][l, h],
                inverse_map=blob["inverse"][l, h],
                mu=blob["mu"][l, h], mu_q=blob["mu_q"][l, h])
            if k_method in ("pgq_fixed", "pgq_fixed_ea"):
                comps[(l, h)] = FixedWidthPagedCompressor(
                    code_std=blob["code_std"][l, h],
                    widths=blob["widths"],
                    alphas=blob["alphas"][l, h],
                    b_page=b_page, ptok=ptok,
                    omega_tau=tau if k_method == "pgq_fixed_ea" else 0.0,
                    omega_clamp_bits=cb, **args)
            else:
                sv = [blob["support_vals"][ri][l, h]
                      for ri in range(len(blob["m_grid"]))]
                sl = [blob["support_lens"][ri][l, h]
                      for ri in range(len(blob["m_grid"]))]
                # -log2 p from stored (normalized, 0-padded) probs. Padded
                # positions are unreachable (snap clamps to lens-1); give
                # them a huge finite cost so any bug shows up as an exploded
                # rate in telemetry rather than NaN in the bisection.
                sn = [(-torch.log2(blob["support_probs"][ri][l, h]
                                   .clamp_min(1e-12))).masked_fill(
                          blob["support_probs"][ri][l, h] <= 0, 1e6)
                      for ri in range(len(blob["m_grid"]))]
                comps[(l, h)] = PagedRDOCompressor(
                    delta_base=float(blob["delta_base"][l, h]),
                    m_grid=blob["m_grid"], support_vals=sv,
                    support_lens=sl, support_nlp=sn,
                    dz=float(blob["dz"]), b_page=b_page, ptok=ptok,
                    mode="pagerung" if k_method == "pgq_pagerung" else "rdo",
                    omega_tau=tau if k_method == "pgq_ea" else 0.0,
                    omega_clamp_bits=cb, **args)
    meta = {k: v for k, v in blob.items() if not torch.is_tensor(v)
            and not isinstance(v, list)}
    return comps, meta


def _load_pgq2(blob, path, k_method, k_bits):
    """pgq2 loader: pgq_{nd,rvq}_{uni,rdo,ea,eag}. omega tau comes from the
    bundle's frozen omega_tau_by_rate (patched in by select_pgq2_hparams
    after the selection-row freeze); eag additionally applies gate_theta."""
    from kvq.compression.norm_direction import NormDirectionPagedCompressor
    from kvq.compression.rvq import ResidualVQPagedCompressor

    if blob.get("pgq_version") != PGQ2_BUNDLE_VERSION:
        raise ValueError(f"pgq2 bundle version mismatch at {path}")
    fam, variant = k_method.split("_")[1], k_method.split("_")[2]
    L, H = blob["n_layers"], blob["n_kv_heads"]
    if variant == "uni":
        b_page, uniform_sel = 8.0, int(k_bits)
        mode = "uniform"
    else:
        b_page, uniform_sel = float(k_bits), None
        mode = "rdo"
    if variant in ("ea", "eag"):
        taus = blob["omega_tau_by_rate"]
        key = f"{b_page:g}"
        if key not in taus:
            raise ValueError(f"no frozen omega tau for rate {key} in {path}")
        tau = float(taus[key])
    else:
        tau = 0.0
    theta = float(blob["gate_theta"]) if variant == "eag" else 0.0

    comps = {}
    for l in range(1, L):
        for h in range(H):
            common = dict(
                forward_map=blob["forward"][l, h],
                inverse_map=blob["inverse"][l, h],
                mu=blob["mu"][l, h], mu_q=blob["mu_q"][l, h],
                b_page=b_page, ptok=int(blob["ptok"]), mode=mode,
                omega_tau=tau,
                omega_clamp_bits=float(blob["omega_clamp_bits"]),
                gate_theta=theta,
                head_m_spread=float(blob["head_m_spread"][l, h]))
            if fam == "nd":
                comps[(l, h)] = NormDirectionPagedCompressor(
                    coord_std_dir=blob["coord_std_dir"][l, h],
                    profiles=blob["nd_profiles"],
                    uniform_rung=uniform_sel, **common)
            else:
                comps[(l, h)] = ResidualVQPagedCompressor(
                    codebooks=[cb[l][h] for cb in blob["rvq_codebooks"]],
                    perm=blob["rvq_perm"][l, h],
                    uniform_stages=uniform_sel, **common)
    meta = {k: v for k, v in blob.items()
            if not torch.is_tensor(v) and not isinstance(v, list)}
    return comps, meta


def _load_pgq4(blob, path, k_method, k_bits):
    """pgq4 loader: pgq_{fold,prof}[mods]_{uni,rdo,ea,pgr}.

    fam modifiers (concatenated after fold/prof): g = fp16 gain/token,
    lm = Lloyd-Max LUT grid (qpca basis only), ob = OSCAR per-layer
    U_Q.H.Pbr basis, rs = r_sym basis, px = prefix-truncation profiles,
    rw = force last 4 prompt pages to the top rung (P7 ablation). For *_uni
    k_bits is the RUNG INDEX; otherwise the page budget in bits/coord.
    Frozen omega taus come from the bundle (carried, no re-selection)."""
    from kvq.compression.pgq4_folded import (
        PGQ4_BUNDLE_VERSION, WIDTH_LADDER, FoldedScalarPagedCompressor,
    )

    if blob.get("pgq_version") != PGQ4_BUNDLE_VERSION:
        raise ValueError(f"pgq4 bundle version mismatch at {path}")
    fam, variant = k_method.split("_")[1], k_method.split("_")[2]
    family = "fold" if fam.startswith("fold") else "prof"
    mods, rest = set(), fam[len(family):]
    while rest:
        for tok in ("lm", "ob", "rs", "px", "rw", "g"):
            if rest.startswith(tok):
                mods.add(tok)
                rest = rest[len(tok):]
                break
        else:
            raise ValueError(f"unknown pgq4 fam modifier in {k_method!r}")

    basis = "oscar" if "ob" in mods else ("r_sym" if "rs" in mods
                                          else "qpca_unc")
    L, H = blob["n_layers"], blob["n_kv_heads"]
    d = blob["head_dim"]
    if variant == "uni":
        b_page, uniform_sel, mode = 8.0, int(k_bits), "uniform"
    elif variant == "pgr":
        b_page, uniform_sel, mode = float(k_bits), None, "pagerung"
    else:
        b_page, uniform_sel, mode = float(k_bits), None, "rdo"
    if variant in ("ea", "pgr"):
        taus = blob["omega_tau_by_rate"]
        key = f"{b_page:g}"
        if key not in taus:
            raise ValueError(f"no frozen omega tau for rate {key} in {path}")
        tau = float(taus[key])
    else:
        tau = 0.0

    bset = blob["bases"][basis]
    stats = blob["stats"][basis]
    blk = d // blob["prof_head"].shape[-1] if "prof_head" in blob else d
    # family A keeps the registered {0,2,3,4}; family B may carry the
    # amendment-A1 extended ladder (bundle field, default legacy)
    if family == "prof":
        ladder = tuple(blob.get("prof_width_ladder", WIDTH_LADDER))
    else:
        ladder = WIDTH_LADDER
    n_pos = len(ladder) - 1

    comps = {}
    for l in range(1, L):
        for h in range(H):
            if bset["forward"].dim() == 3:                   # per-layer basis
                fmap, imap = bset["forward"][l], bset["inverse"][l]
            else:
                fmap, imap = bset["forward"][l, h], bset["inverse"][l, h]
            if family == "fold":
                profiles = torch.tensor(
                    [[w] * d for w in ladder], dtype=torch.long)
            elif "px" in mods:
                profiles = blob["px_profiles"].repeat_interleave(blk, dim=1)
            elif blob["prof_share"] == "layer":
                profiles = blob["prof_layer"][l].repeat_interleave(blk, dim=1)
            else:
                profiles = blob["prof_head"][l, h].repeat_interleave(
                    blk, dim=1)
            lm_cents = blob.get("lm_cents")
            comps[(l, h)] = FoldedScalarPagedCompressor(
                forward_map=fmap, inverse_map=imap,
                mu=blob["mu"][l, h], mu_q=blob["mu_q"][l, h],
                code_std=stats["code_std"][l, h],
                profiles=profiles,
                alphas=stats["alphas"][l, h][:n_pos],
                sink_scale=stats["sink_scale"][l, h],
                b_page=b_page,
                grid="lm" if "lm" in mods else "uniform",
                lm_cents=lm_cents[:n_pos - 1] if lm_cents else None,
                gain="g" in mods,
                ptok=int(blob["ptok"]), mode=mode,
                uniform_rung=uniform_sel, omega_tau=tau,
                omega_clamp_bits=float(blob["omega_clamp_bits"]),
                force_recent_pages=4 if "rw" in mods else 0,
                width_ladder=ladder)
    meta = {k: v for k, v in blob.items()
            if not torch.is_tensor(v) and not isinstance(v, (list, dict))}
    return comps, meta


def _load_pgq3(blob, path, k_method, k_bits):
    """pgq3 loader: pgq_{tcq,e8}_{uni,rdo,ea} | pgq_oscar_uni. Frozen omega
    tau comes from the bundle (fit carries the pgq2 frozen values forward;
    no re-selection). oscar is single-rung: k_bits indexes oscar_widths."""
    from kvq.compression.tcq import TCQPagedCompressor
    from kvq.compression.e8 import E8PagedCompressor
    from kvq.compression.oscar_arm import OscarArmCompressor

    if blob.get("pgq_version") != PGQ3_BUNDLE_VERSION:
        raise ValueError(f"pgq3 bundle version mismatch at {path}")
    fam, variant = k_method.split("_")[1], k_method.split("_")[2]
    L, H = blob["n_layers"], blob["n_kv_heads"]
    if variant == "uni":
        b_page, uniform_sel = 8.0, int(k_bits)
        mode = "uniform"
    else:
        b_page, uniform_sel = float(k_bits), None
        mode = "rdo"
    if variant == "ea":
        taus = blob["omega_tau_by_rate"]
        key = f"{b_page:g}"
        if key not in taus:
            raise ValueError(f"no frozen omega tau for rate {key} in {path}")
        tau = float(taus[key])
    else:
        tau = 0.0

    comps = {}
    for l in range(1, L):
        for h in range(H):
            if fam == "oscar":
                comps[(l, h)] = OscarArmCompressor(
                    forward_map=blob["oscar_mixer"],
                    inverse_map=blob["oscar_mixer"].t().contiguous(),
                    mu=torch.zeros(blob["head_dim"]),
                    mu_q=blob["mu_q"][l, h],
                    width=blob["oscar_widths"][uniform_sel],
                    group_size=blob["oscar_group"],
                    clip_q=blob["oscar_clip_q"],
                    b_page=8.0, ptok=int(blob["ptok"]))
                continue
            common = dict(
                forward_map=blob["forward"][l, h],
                inverse_map=blob["inverse"][l, h],
                mu=blob["mu"][l, h], mu_q=blob["mu_q"][l, h],
                b_page=b_page, ptok=int(blob["ptok"]), mode=mode,
                uniform_rung=uniform_sel, omega_tau=tau,
                omega_clamp_bits=float(blob["omega_clamp_bits"]))
            if fam == "tcq":
                comps[(l, h)] = TCQPagedCompressor(
                    tables=[t[l, h] for t in blob["tcq_tables"]],
                    tcq_widths=blob["tcq_widths"],
                    n_states=int(blob["tcq_states"]),
                    sparse_ks=blob["sparse_ks"],
                    mag_bits=int(blob["mag_bits"]),
                    mixer=blob.get("tcq_mixer"), **common)
            else:
                comps[(l, h)] = E8PagedCompressor(
                    profiles=blob["e8_profiles"][l, h],
                    beta=float(blob["e8_beta"][l, h]),
                    mixer=blob.get("e8_mixer"), **common)
    meta = {k: v for k, v in blob.items()
            if not torch.is_tensor(v) and not isinstance(v, list)}
    return comps, meta
