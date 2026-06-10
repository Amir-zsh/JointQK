#!/usr/bin/env python3
"""Companding on top of EC-UQ (entropy-coded uniform quantization), NO deadzone.

Motivation: the dequantized K is fed to softmax(q . k^T), which amplifies large
logits, so error on large-magnitude code coordinates matters more than error near
zero. A deadzone reallocates resolution toward the active coeffs in a hard, binned
way; a *compander* does the same thing smoothly with one knob.

Compander (per coord, in sigma units; sigma = per-coord std of the QPCA code r):
    forward (compress before uniform quant):  w = sigma * sinh(a * r/sigma) / a
    inverse (expand after dequant):           r_hat = sigma * asinh(a * w/sigma) / a
w'(r) = cosh(a r/sigma) >= 1 and grows with |r|, so the effective step in the code
domain, Delta*sigma/cosh(a r/sigma), is coarse near 0 and exponentially finer in the
tails -- resolution is spent on the values softmax cares about. a=0 -> identity ->
plain EC-UQ. Sweep a; moderate values (~0.25..1.0) are the usable range, larger a
over-compands and the rate solve degrades.

Honesty: sigma, the per-coord delta, and the range-coder frequency model are ALL fit
on the CALIB indices; eval indices are only encoded with the frozen model (held-out
rate) and scored with the frozen compressors. Calib/eval must be disjoint.

    python run_compand_split.py --calib-idx 0 1 2 3 --eval-idx 4 5 6 7 --a 0.25 0.5 1.0
    python run_compand_split.py --calib-idx 0 1 2 3 --eval-idx 4 5 6 7 --bits 2 3 --a 0.5
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pca_ec_deadzone as R          # reuse moments/basis/EC machinery (and its env setup)
import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# Compander (r-units; a=0 is exactly the identity -> plain EC-UQ).
# ---------------------------------------------------------------------------
def _compand(r, sigma, a):
    if abs(a) < 1e-8:
        return r
    return sigma * torch.sinh(a * (r / sigma)) / a


def _expand(w, sigma, a):
    if abs(a) < 1e-8:
        return w
    return sigma * torch.asinh(a * (w / sigma)) / a


class CompandECRoundtrip:
    """QPCA forward -> sinh compand -> uniform round -> dequant -> sinh expand -> inverse."""
    def __init__(self, fwd, inv, mu, sigma, delta, a):
        self.fwd = fwd.float(); self.inv = inv.float()
        self.mu = mu.reshape(1, -1).float()
        self.sigma = sigma.reshape(1, -1).float().clamp_min(1e-12)
        self.delta = delta.reshape(1, -1).float().clamp_min(1e-12)
        self.a = float(a)
        self.forward_map = self.fwd

    def to(self, dev):
        self.fwd = self.fwd.to(dev); self.inv = self.inv.to(dev)
        self.mu = self.mu.to(dev); self.sigma = self.sigma.to(dev)
        self.delta = self.delta.to(dev)
        self.forward_map = self.fwd
        return self

    def roundtrip(self, k):
        r = (k - self.mu) @ self.fwd
        w = _compand(r, self.sigma, self.a)
        idx = torch.round(w / self.delta)
        wh = idx * self.delta
        rh = _expand(wh, self.sigma, self.a)
        return rh @ self.inv + self.mu


def _companded_fetch(base_fetch, sigma, a):
    """Wrap a _codes_for_idx fetch so the EC machinery sees companded codes w (the
    domain delta/model live in). a=0 -> returns the raw codes unchanged."""
    def fetch(l, h):
        r = base_fetch(l, h)                       # (N,d) double, centered QPCA codes
        return _compand(r, sigma[l, h].double(), a)
    return fetch


# ---------------------------------------------------------------------------
# Fit delta + frozen coder model on CALIB companded codes (closed-form step, plain
# round). Cached on (dataset, calib-idx, bits, a).
# ---------------------------------------------------------------------------
def build_compand_ec(F, inv, sigma, k_mean, sigma_q, a, b, n_layers, n_kv, d,
                     cfetch_calib, root, calib_idx):
    key = (f"{root.name}__compand__idx_" + "_".join(str(i) for i in sorted(calib_idx))
           + f"__b{b}__a{a:g}")
    cache_path = R.MOMENTS_CACHE / f"{key}.pt"
    if cache_path.exists():
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        delta, model = c["delta"], c["model"]
        print(f"  [compand] b={b} a={a:g}: model CACHE HIT {cache_path.name}")
    else:
        print(f"  [compand] b={b} a={a:g}: model cache miss, fitting on calib...")
        # q-sensitivity per code coord (same inter-coord weighting as QPCA-EC)
        ell = (F.transpose(-1, -2) @ R.regularize_batch(sigma_q, R.EPS).to(F.dtype) @ F
               ).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        # differential entropy of the COMPANDED calib codes -> closed-form step
        h = R._diff_entropy_from_fetch(cfetch_calib, n_layers, n_kv, d)
        target = h + 0.5 * ell.double().log2()
        Hj = (target - target.mean(-1, keepdim=True) + float(b)).clamp_min(0.0)
        Hj = Hj * (float(b) * d / Hj.sum(-1, keepdim=True).clamp_min(1e-9))
        delta = torch.pow(2.0, (h - Hj)).float()
        model = R.freeze_coder_model(cfetch_calib, delta, n_layers, n_kv, d)  # dz=0.5 -> plain round
        torch.save({"delta": delta, "model": model, "calib_idx": sorted(calib_idx),
                    "b": b, "a": a}, cache_path)
        print(f"    cached -> {cache_path.name}")
    comps = {(l, h): CompandECRoundtrip(F[l, h], inv[l, h], k_mean[l, h],
                                        sigma[l, h], delta[l, h], a)
             for l in range(n_layers) for h in range(n_kv)}
    return comps, delta, model


# ---------------------------------------------------------------------------
# Scoring (one eval example at a time). Methods = a values.
# ---------------------------------------------------------------------------
def score_eval(root, manifest, eval_idx, comps_by_a, k_bits, n_layers, n_kv, device):
    accums = {a: {b: {l: dict(mse_num=0.0, mse_den=0, logit_num=0.0, logit_den=0,
                              top1_num=0, top1_den=0, top5_num=0, top5_den=0)
                      for l in range(n_layers)} for b in k_bits} for a in comps_by_a}
    for ii, i in enumerate(eval_idx):
        print(f"  [eval {ii+1}/{len(eval_idx)}] ex {i}...", end=" ", flush=True)
        art = torch.load(root / manifest["examples"][i]["file"], map_location="cpu", weights_only=False)
        q_all, k_all = art["q_post"], art["k_post"]
        T = int(art["prompt_length"]); d = k_all.shape[-1]
        gs = q_all.shape[1] // n_kv
        for l in range(n_layers):
            for h in range(n_kv):
                k = k_all[l, h, :T, :].to(device).float()
                q = q_all[l, h*gs:(h+1)*gs, :T, :].to(device).float().reshape(-1, d)
                qq = q.transpose(0, 1) @ q
                real_top = (q @ k.transpose(0, 1)).argmax(-1)
                n_q = int(real_top.numel()); k5 = min(5, T)
                for a, cb in comps_by_a.items():
                    for bits in k_bits:
                        kh = cb[bits][(l, h)].roundtrip(k).float()
                        err = k - kh; acc = accums[a][bits][l]
                        acc["mse_num"] += float(err.square().sum()); acc["mse_den"] += int(err.numel())
                        ee = err.transpose(0, 1) @ err
                        acc["logit_num"] += float((qq * ee).sum()); acc["logit_den"] += int(q.shape[0]*T*T)
                        ap = q @ kh.transpose(0, 1)
                        acc["top1_num"] += int((real_top == ap.argmax(-1)).sum()); acc["top1_den"] += n_q
                        a5 = ap.topk(k5, dim=-1).indices
                        acc["top5_num"] += int((a5 == real_top.unsqueeze(-1)).any(-1).sum()); acc["top5_den"] += n_q
        print("done")
    return accums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", required=True)
    ap.add_argument("--eval-idx", type=int, nargs="+", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--a", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                    help="sinh compander strengths (in 1/sigma units); a=0 is added "
                         "automatically as the EC-UQ baseline")
    args = ap.parse_args()
    k_bits = sorted(args.bits)
    a_list = [0.0] + [a for a in sorted(set(args.a)) if abs(a) > 1e-8]
    overlap = set(args.calib_idx) & set(args.eval_idx)
    if overlap:
        sys.exit(f"calib and eval indices overlap: {sorted(overlap)} -- must be disjoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = R.data_root(); manifest = R.load_manifest(root)
    nex = len(manifest["examples"])
    for i in args.calib_idx + args.eval_idx:
        if i >= nex:
            sys.exit(f"index {i} out of range (full has {nex} examples)")
    print(f"full has {nex} examples | calib={args.calib_idx} eval={args.eval_idx} | device {device}")

    sigma_q, sigma_k, k_mean, k_cov, meta = R.calib_moments(root, manifest, args.calib_idx)
    n_layers, n_kv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]

    qpca_cen = R.build_qpca_basis(sigma_q, k_cov)         # centered QPCA basis (matches QPCA-EC path)
    F, inv, sigma = qpca_cen["forward"], qpca_cen["inverse"], qpca_cen["std"]

    print(f"\nEC-UQ + sinh compand (NO deadzone); a sweep = {a_list}; "
          f"sigma, delta, coder model frozen on calib")

    base_calib = R._codes_for_idx(root, manifest, args.calib_idx, F, k_mean, n_layers, n_kv, d)
    base_eval = R._codes_for_idx(root, manifest, args.eval_idx, F, k_mean, n_layers, n_kv, d)

    comps = {a: {} for a in a_list}
    rate = {a: {} for a in a_list}
    for a in a_list:
        cfetch_calib = _companded_fetch(base_calib, sigma, a)
        cfetch_eval = _companded_fetch(base_eval, sigma, a)
        for b in k_bits:
            cb, delta, model = build_compand_ec(F, inv, sigma, k_mean, sigma_q, a, b,
                                                n_layers, n_kv, d, cfetch_calib,
                                                root, args.calib_idx)
            comps[a][b] = cb
            rate[a][b] = R.coded_bits_eval(cfetch_eval, delta, model, n_layers, n_kv, d)
            print(f"  [compand] b={b} a={a:g}: held-out rate = {rate[a][b]:.3f} bits/coord")

    # move compressors to device
    seen = set()
    for a in comps:
        for b in comps[a]:
            for kk in comps[a][b]:
                c = comps[a][b][kk]
                if id(c) not in seen:
                    c.to(device); seen.add(id(c))

    accums = score_eval(root, manifest, args.eval_idx, comps, k_bits, n_layers, n_kv, device)
    final = R.finalize(accums, n_layers, exclude_layer_0=True)

    def label(a):
        return "EC-UQ" if abs(a) < 1e-8 else f"compand a={a:g}"

    print(f"\n{'method':<14} | {'b':<2} | {'rate(b/c)':>9} | {'top-1':>7} | {'top-5':>7} | "
          f"{'k_mse':>11} | {'logit_err':>11}")
    print("-" * 82)
    for a in a_list:
        for b in k_bits:
            dd = final[a][b]
            print(f"{label(a):<14} | {b:<2} | {rate[a][b]:>9.3f} | {dd['top1']:>7.4f} | "
                  f"{dd['top5']:>7.4f} | {dd['k_mse']:>11.3e} | {dd['logit_err']:>11.3e}")
        print()
    print("rate(b/c) is the HELD-OUT constriction rate (companded model frozen on calib, "
          "applied to eval). a=0 == plain EC-UQ. Calib/eval disjoint.")

    # RD plot: x = held-out rate, one curve per a
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    titles = {"top1": ("top-1", "higher"), "top5": ("top-5", "higher"),
              "k_mse": ("k_mse", "lower"), "logit_err": ("logit_err", "lower")}
    cmap = plt.get_cmap("viridis")
    colors = {a: cmap(i / max(1, len(a_list) - 1)) for i, a in enumerate(a_list)}
    for ax, met in zip(axes, ["top1", "top5", "k_mse", "logit_err"]):
        for a in a_list:
            xs = [rate[a][b] for b in k_bits]
            ys = [final[a][b][met] for b in k_bits]
            ax.plot(xs, ys, marker="o", color=colors[a], label=label(a),
                    linewidth=1.8, markersize=7)
        t, dirn = titles[met]
        ax.set_title(f"{t}\n({dirn})", fontsize=11); ax.set_xlabel("bits / coord")
        if met in ("k_mse", "logit_err"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    R.save_fig(fig, "04_compand_comparison.png")
    print("\nDone.")


if __name__ == "__main__":
    main()