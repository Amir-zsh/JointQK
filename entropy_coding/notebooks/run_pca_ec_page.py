#!/usr/bin/env python3
"""Paged EC-Q-PCA: fixed-size pages (vLLM-style, GLOBAL page size) with a
two-sided m-ladder. Compares TurboQuant, JointQK, EC-Q-PCA (unpaged), and Paged EC-Q-PCA.

Page size is the natural fixed buffer: page_bits = b * d * P_tok -- exactly b
bits/coord, i.e. IDENTICAL memory to TurboQuant at b bits. So each b is a
matched-memory test: at the same bytes, does EC shaping + the ladder beat
TurboQuant's flat allocation? No quantile/q-sweep needed.

A page is self-contained (vLLM): its location never depends on a neighbor, so no
cross-page budget sharing. Each page picks the FINEST rung (smallest delta
multiplier m) whose coded length + per-page overhead fits page_bits:
  * m<1 rungs convert slack into precision (replaces dead padding);
  * m>1 rungs quantize an overflowing page harder until it fits;
  * a guaranteed coarse terminal rung makes every page provably fit.
"Finest that fits" is the RD-optimal discrete choice (distortion decreasing, bits
increasing in step size). The page COSTS page_bits regardless of fill.

Overhead is folded INSIDE the page: fit test is payload + FLUSH + header <= page_bits,
so the reported rate is EXACTLY b (header = ceil(log2|ladder|) bits/page names the
chosen rung; the deadzone dz is global+fixed, so it costs nothing to signal). The
flat log2(R) header is a conservative upper bound -- an entropy code over the
(highly non-uniform) rung index would cost less.

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
# Paged reconstruction object. Same interface as base.UniformECRoundtrip.
# page_bits = b * d * P_tok (fixed buffer). Per-page overhead (FLUSH + rung header)
# is folded INSIDE the page, so fit test is payload + overhead <= page_bits and the
# stored rate is exactly b.
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
        self.overhead = float(overhead)   # FLUSH + header, charged inside the page
        self.d = fwd.shape[-1]
        self.ms = np.array([r[0] for r in rungs_lh], dtype=np.float64)
        self.delta_dev = [r[1].float().reshape(1, -1).clamp_min(1e-12) for r in rungs_lh]
        self.delta_np = [r[1].double().numpy().clip(1e-12) for r in rungs_lh]
        self.models = [r[2] for r in rungs_lh]
        self.R = len(rungs_lh)
        self.forward_map = self.fwd
        # stats
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

        rung_q = []          # device dequant per rung
        block_bits = []      # (nb,) payload bits per rung (no overhead)
        for dl_dev, dl_np, model in zip(self.delta_dev, self.delta_np, self.models):
            idx = base._dz_round(r, dl_dev, self.dz)
            rung_q.append(base._dz_dequant(idx, dl_dev, self.dz))   # (T,d) device
            tb = _tok_bits_np(r_np, dl_np, model, self.dz, self.d)  # (T,) numpy
            pad = nb * P - T
            if pad:
                tb = np.concatenate([tb, np.zeros(pad)])
            block_bits.append(tb.reshape(nb, P).sum(1))

        blk = np.stack(block_bits, 0)                          # (R, nb) payload only
        fits = (blk + self.overhead) <= self.page_bits         # overhead inside page
        anyfit = fits.any(0)
        chosen = np.where(anyfit, fits.argmax(0), self.R - 1)  # argmax -> first True (finest)

        self.rung_hist += np.bincount(chosen, minlength=self.R)
        self.overflow += int((~anyfit).sum())
        self.nblocks += nb

        out = torch.empty_like(r)
        for blk_i in range(nb):
            sl = slice(blk_i * P, min((blk_i + 1) * P, T))
            out[sl] = rung_q[int(chosen[blk_i])][sl]
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
    ap.add_argument("--max-eval-tokens", type=int, default=0,
                    help="cap eval T for speed (0=all); fit test is the runtime bottleneck")
    args = ap.parse_args()

    k_bits = sorted(args.bits)
    m_grid = sorted(float(x) for x in args.m_grid)
    if not any(abs(m - 1.0) < 1e-9 for m in m_grid):
        m_grid = sorted(m_grid + [1.0])
    header_bits = math.ceil(math.log2(max(2, len(m_grid))))
    overhead = FLUSH_BITS + header_bits  # charged inside each page

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
    print(f"P_tok={args.ptok} dz={args.dz:g} | ladder m={m_grid} "
          f"(header {header_bits} b/page, +{FLUSH_BITS:g} flush, folded inside page)")

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

    comps = {"TurboQuant": {}, "JointQK": {}, "EC-Q-PCA": {}, "Paged": {}}
    ec_rate = {}             # held-out unpaged EC rate per b
    page_bits = {}           # page_bits[b] = b * d * P_tok
    for b in k_bits:
        comps["TurboQuant"][b] = {(l, h): turbo[b] for l in range(L) for h in range(Hkv)}
        comps["JointQK"][b] = base.build_compressors(jq, b, L, Hkv)

        rungs, delta0, model0 = build_ladder(
            qpca_cen, qpca_unc, k_mean, b, L, Hkv, fetch_concat, root,
            args.calib_idx, m_grid, args.dz)

        # EC-Q-PCA (unpaged) == m=1 rung applied per token; rate = held-out coded bits
        comps["EC-Q-PCA"][b] = {
            (l, h): base.UniformECRoundtrip(F[l, h], qpca_cen["inverse"][l, h],
                                            k_mean[l, h], delta0[l, h], dz=args.dz)
            for l in range(L) for h in range(Hkv)}
        ec_rate[b] = base.coded_bits_eval(fetch_eval_concat, delta0, model0, L, Hkv, d, dz=args.dz)

        # paged: fixed buffer = b * d * P_tok bits (exactly b bits/coord == TurboQuant @ b)
        pb = b * d * args.ptok
        page_bits[b] = pb
        print(f"  [b={b}] page_bits={pb} = {b}*{d}*{args.ptok} -> rate {pb/(args.ptok*d):.3f} b/coord "
              f"(overhead {overhead:g} b/page inside) | unpaged EC held-out = {ec_rate[b]:.3f} b/coord")
        rungs_by_lh = lambda l, h: [(m, dm[l, h], mod[(l, h)]) for (m, dm, mod) in rungs]
        comps["Paged"][b] = {
            (l, h): PagedUniformECRoundtrip(
                F[l, h], qpca_cen["inverse"][l, h], k_mean[l, h],
                rungs_by_lh(l, h), pb, overhead, args.dz, args.ptok)
            for l in range(L) for h in range(Hkv)}

    base.move_to_device(comps, device)
    accums = score_eval_paged(root, manifest, args.eval_idx, comps, k_bits, device,
                              L, Hkv, max_tokens=args.max_eval_tokens)
    final = base.finalize(accums, L, exclude_layer_0=True)

    # rung-usage stats per b
    def paged_stats(b):
        hist = np.zeros(len(m_grid), dtype=np.int64)
        ovf = nb = 0
        for (l, h), c in comps["Paged"][b].items():
            if l == 0:
                continue
            hist += c.rung_hist
            ovf += c.overflow
            nb += c.nblocks
        ms = np.array(sorted(m_grid))
        fit_le1 = hist[ms <= 1.0 + 1e-9].sum() / max(1, nb)
        return fit_le1, ovf / max(1, nb), hist, nb

    def rate_of(m, b):
        if m == "EC-Q-PCA":
            return ec_rate[b]
        if m == "Paged":
            return page_bits[b] / (args.ptok * d)   # == b
        return float(b)                              # TurboQuant, JointQK

    methods = ["TurboQuant", "JointQK", "EC-Q-PCA", "Paged"]
    print(f"\n{'method':<12} | {'b':<2} | {'rate(b/c)':>9} | {'top-1':>7} | "
          f"{'top-5':>7} | {'k_mse':>11} | {'logit_err':>11} | {'m<=1':>5} | {'ovf':>6}")
    print("-" * 96)
    for m in methods:
        for b in k_bits:
            dd = final[m][b]
            if m == "Paged":
                fit_le1, ovf, _, _ = paged_stats(b)
                f1, ov = f"{fit_le1:.2f}", f"{ovf:.4f}"
            else:
                f1, ov = "", ""
            print(f"{m:<12} | {b:<2} | {rate_of(m, b):>9.3f} | {dd['top1']:>7.4f} | "
                  f"{dd['top5']:>7.4f} | {dd['k_mse']:>11.3e} | {dd['logit_err']:>11.3e} | "
                  f"{f1:>5} | {ov:>6}")
        print()
    print("rate(b/c): TurboQuant/JointQK = grid-bits; EC-Q-PCA = HELD-OUT constriction "
          "bits (unpaged); Paged = page_bits/(P_tok*d) = b EXACTLY (overhead folded in). "
          "m<=1: frac of eval pages at a finest-or-equal rung; ovf: frac exceeding the "
          "coarsest rung (should be ~0). Calib/eval disjoint; all rung models calib-frozen.")
    for b in k_bits:
        _, _, hist, nb = paged_stats(b)
        frac = ", ".join(f"m{m:g}:{h/max(1,nb):.2f}"
                         for m, h in zip(sorted(m_grid), hist) if h)
        print(f"  rung usage [b={b}]: {frac}")

    # RD plot: x=rate, y=metric, all methods over the b grid.
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    titles = {"top1": ("top-1", "higher"), "top5": ("top-5", "higher"),
              "k_mse": ("k_mse", "lower"), "logit_err": ("logit_err", "lower")}
    series = {"TurboQuant": ("#888", "s"), "JointQK": ("#328ac1", "o"),
              "EC-Q-PCA": ("#9b1c9b", "P"), "Paged": ("#1c5d2c", "^")}
    for ax, met in zip(axes, ["top1", "top5", "k_mse", "logit_err"]):
        for m in methods:
            xs = [rate_of(m, b) for b in k_bits]
            ys = [final[m][b][met] for b in k_bits]
            order = np.argsort(xs)
            color, mk = series[m]
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker=mk, color=color,
                    label=m, linewidth=1.8, markersize=8)
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