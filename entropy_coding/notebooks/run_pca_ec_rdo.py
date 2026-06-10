#!/usr/bin/env python3
"""Paged EC-Q-PCA: fixed-size pages (vLLM-style, GLOBAL page size). Compares
TurboQuant, JointQK, EC-Q-PCA (unpaged), Paged (one rung/page), and Paged-RDO
(per-token rung via intra-page Lagrangian RDO).

Page size is the natural fixed buffer: page_bits = b * d * P_tok -- exactly b
bits/coord, IDENTICAL memory to TurboQuant at b bits. Each b is a matched-memory
test. A page is self-contained (vLLM): no cross-page budget sharing.

Two-sided m-ladder (delta multipliers): m<1 buys precision, m>1 quantizes harder,
a coarse terminal rung makes every page provably fit. Every rung is calib-frozen
(m*delta changes the index alphabet -> its own frozen model). Coded length for the
fit/rate proxy is the cross-entropy -sum log2 p under the frozen model (eval
symbols snapped to nearest calib bin), what a range coder achieves.

Paged (per-page): pick the FINEST rung whose page coded length + overhead fits.
  overhead/page = FLUSH + 1*ceil(log2 R)  (one rung index per page)

Paged-RDO (per-token): within a page, pick a rung PER TOKEN by Lagrangian RDO.
  rate R_{t,r}  = cross-entropy tok_bits at rung r  (the proxy we already have)
  dist D_{t,r}  = ||dequant_r(r_t) - r_t||^2 in QPCA coords. The QPCA forward
                  whitens w.r.t. calib Sigma_Q (inv @ Sigma_Q @ inv^T = I), so
                  coord-space MSE EQUALS the held-out logit-error proxy in
                  expectation -- no extra matrices. Pick r(t)=argmin_r(D+lam*R),
                  bisect lam s.t. sum_t R <= payload_budget.
  overhead/page = FLUSH + P_tok*ceil(log2 R)  (one rung index PER TOKEN)

Overhead is folded INSIDE the page for both, so the reported rate is EXACTLY b.
Paged-RDO trades a fatter header for better allocation -- net effect is the test.
The flat log2(R) per index is a conservative upper bound (rung indices are skewed;
an entropy code would cost less). dz is global+fixed, so it costs nothing to signal.

    python run_paged_split.py --calib-idx 0 1 2 3 --eval-idx 4 5 6 7 \
        --bits 2 3 4 --ptok 16 --dz 0.375
"""

import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import matplotlib.pyplot as plt

import run_pca_ec_deadzone as base  # triggers env bootstrap + repo imports (run from repo dir)

FLUSH_BITS = 32.0  # per-page range-coder flush overhead (constriction packs u32 words)


# ---------------------------------------------------------------------------
# Coded length under a frozen rung model (cross-entropy, eval->nearest-calib-bin).
# ---------------------------------------------------------------------------
def _dz_round_np(r, delta, dz):
    """Numpy mirror of base._dz_round. r:(N,d), delta:(d,) broadcasts, dz scalar."""
    return np.sign(r) * np.floor(np.abs(r) / delta + dz)


def _neglogp_codes(idx_np, model_lh, d):
    """idx_np:(N,d) int64. model_lh: per-coord [(vals,p)] frozen on calib.
    Returns (N,d) float64 of -log2 p(symbol); eval symbols snapped to nearest calib
    bin (matches coded_bits_eval). Constant coords (|alphabet|<=1) cost 0 bits."""
    N = idx_np.shape[0]
    out = np.zeros((N, d), dtype=np.float64)
    for j in range(d):
        vals, p = model_lh[j]
        if vals.size <= 1:
            continue
        col = idx_np[:, j]
        pos = np.clip(np.searchsorted(vals, col), 0, vals.size - 1)
        left = np.clip(pos - 1, 0, vals.size - 1)
        choose_left = np.abs(vals[left] - col) <= np.abs(vals[pos] - col)
        sym = np.where(choose_left, left, pos)
        out[:, j] = -np.log2(p[sym])
    return out


def _tok_bits_np(r_np, delta_np, model_lh, dz, d):
    """r_np:(N,d) float64. Returns (N,) per-token summed -log2 p at step delta_np."""
    idx = _dz_round_np(r_np, np.clip(delta_np, 1e-12, None), dz).astype(np.int64)
    return _neglogp_codes(idx, model_lh, d).sum(1)


# ---------------------------------------------------------------------------
# Two-sided ladder (each rung calib-frozen, cached on disk).
# Cache key: (dataset, calib_idx, b, dz, m), m = rung delta multiplier, b = target bitrate.
# ---------------------------------------------------------------------------
def _freeze_rung(fetch_concat, delta_m, n_layers, n_kv, d, dz, root, calib_idx, b, m):
    base.MOMENTS_CACHE.mkdir(parents=True, exist_ok=True)
    key = (f"{root.name}__rung__idx_" + "_".join(str(i) for i in sorted(calib_idx))
           + f"__b{b}__dz{dz:g}__m{m:g}")
    cp = base.MOMENTS_CACHE / f"{key}.pt"
    if cp.exists():
        return torch.load(cp, map_location="cpu", weights_only=False)["model"]
    print(f"    [rung] freezing model at m={m:g} (b={b}, dz={dz:g})...")
    model = base.freeze_coder_model(fetch_concat, delta_m, n_layers, n_kv, d, dz)
    torch.save({"model": model, "m": m, "b": b, "dz": dz}, cp)
    return model


def build_ladder(qpca_cen, qpca_unc, k_mean, b, n_layers, n_kv,
                 fetch_concat, root, calib_idx, m_grid, dz):
    """Returns (rungs, delta0, model0). rungs = [(m, delta_m(L,Hkv,d), model_m)]
    sorted finest-first (ascending m). delta0/model0 are the m=1 base (== unpaged
    EC). m=1 rung reuses build_qpca_ec's closed-form step + model."""
    _, delta0, model0 = base.build_qpca_ec(
        qpca_cen, qpca_unc, k_mean, b, n_layers, n_kv,
        fetch_concat, root, calib_idx, dz=dz, match_rate=False)
    d = delta0.shape[-1]
    rungs = []
    for m in sorted(float(x) for x in m_grid):
        if abs(m - 1.0) < 1e-9:
            rungs.append((1.0, delta0, model0))
        else:
            delta_m = (delta0 * m).float()
            model_m = _freeze_rung(fetch_concat, delta_m, n_layers, n_kv, d, dz,
                                   root, calib_idx, b, m)
            rungs.append((m, delta_m, model_m))
    return rungs, delta0, model0


# ---------------------------------------------------------------------------
# Paged reconstruction (one rung per page). Overhead folded inside the page.
# ---------------------------------------------------------------------------
class PagedUniformECRoundtrip:
    def __init__(self, fwd, inv, mu, rungs_lh, page_bits, overhead, dz, P_tok):
        # rungs_lh: list of (m, delta_lh(d,) tensor, model_lh per-coord), finest-first
        self.fwd = fwd.float()
        self.inv = inv.float()
        self.mu = mu.reshape(1, -1).float()
        self.dz = float(dz)
        self.P = int(P_tok)
        self.page_bits = float(page_bits)
        self.overhead = float(overhead)   # FLUSH + 1 rung index, charged inside the page
        self.d = fwd.shape[-1]
        self.ms = np.array([r[0] for r in rungs_lh], dtype=np.float64)
        self.delta_dev = [r[1].float().reshape(1, -1).clamp_min(1e-12) for r in rungs_lh]
        self.delta_np = [r[1].double().numpy().clip(1e-12) for r in rungs_lh]
        self.models = [r[2] for r in rungs_lh]
        self.R = len(rungs_lh)
        self.forward_map = self.fwd
        self.rung_hist = np.zeros(self.R, dtype=np.int64)
        self.overflow = 0
        self.nblocks = 0

    def to(self, dev):
        self.fwd = self.fwd.to(dev)
        self.inv = self.inv.to(dev)
        self.mu = self.mu.to(dev)
        self.delta_dev = [dl.to(dev) for dl in self.delta_dev]
        self.forward_map = self.fwd
        return self

    def roundtrip(self, k):                          # k:(T,d)
        r = (k - self.mu) @ self.fwd                 # device (T,d)
        T = r.shape[0]
        P = self.P
        nb = (T + P - 1) // P
        r_np = r.detach().to("cpu", dtype=torch.float64).numpy()

        rung_q = []
        block_bits = []
        for dl_dev, dl_np, model in zip(self.delta_dev, self.delta_np, self.models):
            idx = base._dz_round(r, dl_dev, self.dz)
            rung_q.append(base._dz_dequant(idx, dl_dev, self.dz))   # (T,d) device
            tb = _tok_bits_np(r_np, dl_np, model, self.dz, self.d)  # (T,) numpy
            pad = nb * P - T
            if pad:
                tb = np.concatenate([tb, np.zeros(pad)])
            block_bits.append(tb.reshape(nb, P).sum(1))

        blk = np.stack(block_bits, 0)                          # (R, nb) payload only
        fits = (blk + self.overhead) <= self.page_bits
        anyfit = fits.any(0)
        chosen = np.where(anyfit, fits.argmax(0), self.R - 1)  # finest that fits

        self.rung_hist += np.bincount(chosen, minlength=self.R)
        self.overflow += int((~anyfit).sum())
        self.nblocks += nb

        out = torch.empty_like(r)
        for blk_i in range(nb):
            sl = slice(blk_i * P, min((blk_i + 1) * P, T))
            out[sl] = rung_q[int(chosen[blk_i])][sl]
        return out @ self.inv + self.mu


# ---------------------------------------------------------------------------
# Paged-RDO reconstruction (per-token rung via intra-page Lagrangian RDO).
# ---------------------------------------------------------------------------
class PagedRDORoundtrip:
    """Per-token rung selection within a page. rate=cross-entropy tok_bits;
    distortion=coord-space MSE (== held-out logit-error proxy under QPCA whitening).
    Per page: r(t)=argmin_r(D_{t,r}+lam*R_{t,r}), bisect lam s.t. sum_t R <= budget.
    Per-token rung index -> P_tok*ceil(log2 R) bits/page, folded into the budget."""
    def __init__(self, fwd, inv, mu, rungs_lh, page_bits, flush_bits, hdr_bits,
                 dz, P_tok, iters=40):
        self.fwd = fwd.float()
        self.inv = inv.float()
        self.mu = mu.reshape(1, -1).float()
        self.dz = float(dz)
        self.P = int(P_tok)
        self.d = fwd.shape[-1]
        self.ms = np.array([r[0] for r in rungs_lh], dtype=np.float64)
        self.delta_dev = [r[1].float().reshape(1, -1).clamp_min(1e-12) for r in rungs_lh]
        self.delta_np = [r[1].double().numpy().clip(1e-12) for r in rungs_lh]
        self.models = [r[2] for r in rungs_lh]
        self.R = len(rungs_lh)
        self.flush = float(flush_bits)
        self.hdr = float(hdr_bits)
        self.payload_budget = float(page_bits) - self.flush - self.P * self.hdr
        self.iters = int(iters)
        self.forward_map = self.fwd
        self.rung_hist = np.zeros(self.R, dtype=np.int64)
        self.overflow = 0
        self.nblocks = 0

    def to(self, dev):
        self.fwd = self.fwd.to(dev)
        self.inv = self.inv.to(dev)
        self.mu = self.mu.to(dev)
        self.delta_dev = [dl.to(dev) for dl in self.delta_dev]
        self.forward_map = self.fwd
        return self

    def roundtrip(self, k):                          # k:(T,d)
        r = (k - self.mu) @ self.fwd                 # device (T,d)
        T = r.shape[0]
        P = self.P
        R = self.R
        nb = (T + P - 1) // P
        Tp = nb * P
        r_np = r.detach().to("cpu", dtype=torch.float64).numpy()

        rung_q = []
        D = np.zeros((R, Tp), dtype=np.float64)      # coord-space MSE per token
        Rt = np.zeros((R, Tp), dtype=np.float64)     # cross-entropy bits per token
        for ri, (dl_dev, dl_np, model) in enumerate(zip(self.delta_dev, self.delta_np, self.models)):
            idx = base._dz_round(r, dl_dev, self.dz)
            q = base._dz_dequant(idx, dl_dev, self.dz)             # (T,d) device
            rung_q.append(q)
            D[ri, :T] = ((q - r) ** 2).sum(1).detach().to("cpu", dtype=torch.float64).numpy()
            Rt[ri, :T] = _tok_bits_np(r_np, dl_np, model, self.dz, self.d)

        ar = np.arange(Tp)
        budget = self.payload_budget
        lam_hi = max(1.0, float(D.max())) * 1e3       # large enough to force coarsest

        def choose(lam_blk):                          # lam_blk:(nb,)
            lam_tok = np.repeat(lam_blk, P)           # (Tp,)
            ch = (D + lam_tok[None, :] * Rt).argmin(0)            # (Tp,) finest=lowest D at lam->0
            br = Rt[ch, ar].reshape(nb, P).sum(1)                 # (nb,) block payload rate
            return ch, br

        lo = np.zeros(nb)
        hi = np.full(nb, lam_hi)
        for _ in range(self.iters):
            mid = 0.5 * (lo + hi)
            _, br = choose(mid)
            over = br > budget                        # rate too high -> raise lam
            lo = np.where(over, mid, lo)
            hi = np.where(over, hi, mid)
        chosen, br = choose(hi)                       # hi side guarantees rate <= budget where feasible
        ovf_blk = br > budget + 1e-6
        if ovf_blk.any():                             # fall back to guaranteed-fit coarsest
            ch2 = chosen.reshape(nb, P)
            ch2[ovf_blk, :] = R - 1
            chosen = ch2.reshape(-1)

        chosen_real = chosen[:T]
        self.rung_hist += np.bincount(chosen_real, minlength=R)
        self.overflow += int(ovf_blk.sum())
        self.nblocks += nb

        ch_t = torch.from_numpy(chosen_real).to(r.device)
        out = torch.empty_like(r)
        for ri in range(R):
            mask = ch_t == ri
            if bool(mask.any()):
                out[mask] = rung_q[ri][mask]
        return out @ self.inv + self.mu


# ---------------------------------------------------------------------------
# Scoring (self-contained; mirrors base.score_eval but takes n_layers/n_kv and
# does not hardcode a "QPCA" method key). One eval example at a time.
# ---------------------------------------------------------------------------
def _zero_layer():
    return dict(mse_num=0.0, mse_den=0, logit_num=0.0, logit_den=0,
                top1_num=0, top1_den=0, top5_num=0, top5_den=0)


def score_eval_paged(root, manifest, eval_idx, comps_by_method, k_bits, device,
                     n_layers, n_kv, max_tokens=0):
    accums = {m: {b: {l: _zero_layer() for l in range(n_layers)} for b in k_bits}
              for m in comps_by_method}
    for ii, i in enumerate(eval_idx):
        print(f"  [eval {ii+1}/{len(eval_idx)}] ex {i}...", end=" ", flush=True)
        art = torch.load(root / manifest["examples"][i]["file"],
                         map_location="cpu", weights_only=False)
        q_all, k_all = art["q_post"], art["k_post"]
        T = int(art["prompt_length"])
        d = k_all.shape[-1]
        if max_tokens and T > max_tokens:
            T = max_tokens
        gs = q_all.shape[1] // n_kv
        for l in range(n_layers):
            for h in range(n_kv):
                k = k_all[l, h, :T, :].to(device).float()
                q = q_all[l, h * gs:(h + 1) * gs, :T, :].to(device).float().reshape(-1, d)
                qq = q.transpose(0, 1) @ q
                real_top = (q @ k.transpose(0, 1)).argmax(-1)
                n_q = int(real_top.numel())
                k5 = min(5, T)
                for m, cb in comps_by_method.items():
                    for bits in k_bits:
                        kh = cb[bits][(l, h)].roundtrip(k).float()
                        err = k - kh
                        acc = accums[m][bits][l]
                        acc["mse_num"] += float(err.square().sum())
                        acc["mse_den"] += int(err.numel())
                        ee = err.transpose(0, 1) @ err
                        acc["logit_num"] += float((qq * ee).sum())
                        acc["logit_den"] += int(q.shape[0] * T * T)
                        ap = q @ kh.transpose(0, 1)
                        acc["top1_num"] += int((real_top == ap.argmax(-1)).sum())
                        acc["top1_den"] += n_q
                        a5 = ap.topk(k5, dim=-1).indices
                        acc["top5_num"] += int((a5 == real_top.unsqueeze(-1)).any(-1).sum())
                        acc["top5_den"] += n_q
        print("done")
    return accums


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", required=True)
    ap.add_argument("--eval-idx", type=int, nargs="+", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4],
                    help="matched-memory points: page = b*d*P_tok bits (== TurboQuant at b)")
    ap.add_argument("--ptok", type=int, default=16, help="tokens per page (vLLM block size B)")
    ap.add_argument("--dz", type=float, default=0.375, help="deadzone offset (0.5=round-to-nearest)")
    ap.add_argument("--m-grid", type=float, nargs="+",
                    default=[0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0, 3.0, 8.0],
                    help="delta multipliers; <1 buys precision, >1 quantizes harder; "
                         "the coarsest is the guaranteed-fit terminal rung")
    ap.add_argument("--rdo-iters", type=int, default=40, help="Lagrangian bisection iters/page")
    ap.add_argument("--max-eval-tokens", type=int, default=0,
                    help="cap eval T for speed (0=all); fit/RDO is the runtime bottleneck")
    args = ap.parse_args()

    k_bits = sorted(args.bits)
    m_grid = sorted(float(x) for x in args.m_grid)
    if not any(abs(m - 1.0) < 1e-9 for m in m_grid):
        m_grid = sorted(m_grid + [1.0])
    R = len(m_grid)
    header_bits = math.ceil(math.log2(max(2, R)))   # bits per rung index
    overhead_page = FLUSH_BITS + header_bits        # Paged: one index/page

    overlap = set(args.calib_idx) & set(args.eval_idx)
    if overlap:
        sys.exit(f"calib and eval overlap: {sorted(overlap)} -- must be disjoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = base.data_root()
    manifest = base.load_manifest(root)
    nex = len(manifest["examples"])
    for i in args.calib_idx + args.eval_idx:
        if i >= nex:
            sys.exit(f"index {i} out of range (full has {nex})")
    print(f"full={nex} | calib={args.calib_idx} eval={args.eval_idx} | device {device}")
    print(f"P_tok={args.ptok} dz={args.dz:g} | ladder m={m_grid} | header {header_bits} b/index | "
          f"Paged hdr {header_bits} b/page, Paged-RDO hdr {args.ptok*header_bits} b/page (inside)")

    sigma_q, sigma_k, k_mean, k_cov, meta = base.calib_moments(root, manifest, args.calib_idx)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]

    qpca_unc = base.build_qpca_basis(sigma_q, sigma_k)
    qpca_unc["sigma_k"], qpca_unc["sigma_q"] = sigma_k, sigma_q
    qpca_cen = base.build_qpca_basis(sigma_q, k_cov)
    qpca_cen["sigma_k"] = sigma_k
    F = qpca_cen["forward"]

    jq = base.build_jointqk_basis(sigma_q, sigma_k)

    turbo = base.build_turboquant(d, k_bits)
    fetch_concat = base._codes_for_idx(root, manifest, args.calib_idx, F, k_mean, L, Hkv, d)
    fetch_eval_concat = base._codes_for_idx(root, manifest, args.eval_idx, F, k_mean, L, Hkv, d)

    comps = {"TurboQuant": {}, "JointQK": {}, "EC-Q-PCA": {}, "Paged": {}, "Paged-RDO": {}}
    ec_rate = {}
    page_bits = {}
    for b in k_bits:
        comps["TurboQuant"][b] = {(l, h): turbo[b] for l in range(L) for h in range(Hkv)}
        comps["JointQK"][b] = base.build_compressors(jq, b, L, Hkv)

        rungs, delta0, model0 = build_ladder(
            qpca_cen, qpca_unc, k_mean, b, L, Hkv, fetch_concat, root,
            args.calib_idx, m_grid, args.dz)

        comps["EC-Q-PCA"][b] = {
            (l, h): base.UniformECRoundtrip(F[l, h], qpca_cen["inverse"][l, h],
                                            k_mean[l, h], delta0[l, h], dz=args.dz)
            for l in range(L) for h in range(Hkv)}
        ec_rate[b] = base.coded_bits_eval(fetch_eval_concat, delta0, model0, L, Hkv, d, dz=args.dz)

        pb = b * d * args.ptok
        page_bits[b] = pb
        print(f"  [b={b}] page_bits={pb} -> rate {pb/(args.ptok*d):.3f} b/coord | "
              f"unpaged EC held-out = {ec_rate[b]:.3f} | "
              f"payload: Paged {pb-overhead_page:.0f}b, RDO {pb-FLUSH_BITS-args.ptok*header_bits:.0f}b")
        rungs_by_lh = lambda l, h: [(m, dm[l, h], mod[(l, h)]) for (m, dm, mod) in rungs]
        comps["Paged"][b] = {
            (l, h): PagedUniformECRoundtrip(
                F[l, h], qpca_cen["inverse"][l, h], k_mean[l, h],
                rungs_by_lh(l, h), pb, overhead_page, args.dz, args.ptok)
            for l in range(L) for h in range(Hkv)}
        comps["Paged-RDO"][b] = {
            (l, h): PagedRDORoundtrip(
                F[l, h], qpca_cen["inverse"][l, h], k_mean[l, h],
                rungs_by_lh(l, h), pb, FLUSH_BITS, header_bits, args.dz, args.ptok,
                iters=args.rdo_iters)
            for l in range(L) for h in range(Hkv)}

    base.move_to_device(comps, device)
    accums = score_eval_paged(root, manifest, args.eval_idx, comps, k_bits, device,
                              L, Hkv, max_tokens=args.max_eval_tokens)
    final = base.finalize(accums, L, exclude_layer_0=True)

    def paged_stats(method, b):
        hist = np.zeros(R, dtype=np.int64)
        ovf = nb = 0
        for (l, h), c in comps[method][b].items():
            if l == 0:
                continue
            hist += c.rung_hist
            ovf += c.overflow
            nb += c.nblocks
        ms = np.array(sorted(m_grid))
        fit_le1 = hist[ms <= 1.0 + 1e-9].sum() / max(1, hist.sum())
        return fit_le1, ovf / max(1, nb), hist, nb

    def rate_of(m, b):
        if m == "EC-Q-PCA":
            return ec_rate[b]
        if m in ("Paged", "Paged-RDO"):
            return page_bits[b] / (args.ptok * d)   # == b
        return float(b)                              # TurboQuant, JointQK

    paged_methods = ("Paged", "Paged-RDO")
    methods = ["TurboQuant", "JointQK", "EC-Q-PCA", "Paged", "Paged-RDO"]
    print(f"\n{'method':<12} | {'b':<2} | {'rate(b/c)':>9} | {'top-1':>7} | "
          f"{'top-5':>7} | {'k_mse':>11} | {'logit_err':>11} | {'m<=1':>5} | {'ovf':>6}")
    print("-" * 96)
    for m in methods:
        for b in k_bits:
            dd = final[m][b]
            if m in paged_methods:
                fit_le1, ovf, _, _ = paged_stats(m, b)
                f1, ov = f"{fit_le1:.2f}", f"{ovf:.4f}"
            else:
                f1, ov = "", ""
            print(f"{m:<12} | {b:<2} | {rate_of(m, b):>9.3f} | {dd['top1']:>7.4f} | "
                  f"{dd['top5']:>7.4f} | {dd['k_mse']:>11.3e} | {dd['logit_err']:>11.3e} | "
                  f"{f1:>5} | {ov:>6}")
        print()
    print("rate(b/c): TurboQuant/JointQK = grid-bits; EC-Q-PCA = HELD-OUT constriction "
          "bits (unpaged); Paged/Paged-RDO = b EXACTLY (overhead folded in). "
          "m<=1: frac of finest-or-equal selections (pages for Paged, tokens for Paged-RDO); "
          "ovf: frac of pages exceeding budget at coarsest (~0). Calib/eval disjoint; "
          "all rung models calib-frozen.")
    for m in paged_methods:
        for b in k_bits:
            _, _, hist, nb = paged_stats(m, b)
            frac = ", ".join(f"m{mm:g}:{h/max(1,hist.sum()):.2f}"
                             for mm, h in zip(sorted(m_grid), hist) if h)
            print(f"  rung usage [{m} b={b}]: {frac}")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    titles = {"top1": ("top-1", "higher"), "top5": ("top-5", "higher"),
              "k_mse": ("k_mse", "lower"), "logit_err": ("logit_err", "lower")}
    series = {"TurboQuant": ("#888", "s"), "JointQK": ("#328ac1", "o"),
              "EC-Q-PCA": ("#9b1c9b", "P"), "Paged": ("#1c5d2c", "^"),
              "Paged-RDO": ("#c1700a", "D")}
    for ax, met in zip(axes, ["top1", "top5", "k_mse", "logit_err"]):
        for m in methods:
            xs = [rate_of(m, b) for b in k_bits]
            ys = [final[m][b][met] for b in k_bits]
            order = np.argsort(xs)
            color, mk = series[m]
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker=mk, color=color,
                    label=m, linewidth=1.8, markersize=7)
        t, dirn = titles[met]
        ax.set_title(f"{t}\n({dirn})", fontsize=11)
        ax.set_xlabel("bits / coord (stored)")
        if met in ("k_mse", "logit_err"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    base.save_fig(fig, "04_paged_comparison.png")
    print("\nDone.")


if __name__ == "__main__":
    main()