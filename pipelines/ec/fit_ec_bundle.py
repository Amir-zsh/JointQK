#!/usr/bin/env python3
"""Fit an entropy-coded (EC) K-side bundle on compact8 TRAIN rows.

Generalizes entropy_coding/run_pca_ec_deadzone.py's build_qpca_ec to three
bases, with rate matching so the held-out coded rate lands at --b-target:

  r_sym     JointQK production basis (R_sym from the calibration bundle).
  hadamard  TurboQuant's own per-layer random rotation (seed 42 + 1000*L),
            i.e. "TQ's basis, strictly better coding".
  qpca      closed-form Q-weighted MSE optimum (control).

Per (layer, head): per-coord uniform deadzone step `delta` solved by bisection
so the deadzoned discrete entropy of the calibration codes water-fills to
b_target bits/coord (weights ell = diag(F^T Sigma_Q F)), then a frequency
model (alphabet + probs) frozen on the same codes. Held-out coded rate is
measured on the disjoint selection rows with constriction (escape symbols
snap to the nearest alphabet entry — the correct OOD penalty).

    python pipelines/ec/fit_ec_bundle.py --basis r_sym --dz 0.375 --device cuda:0
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(REPO / "entropy_coding"))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import run_pca_ec_deadzone as ec  # noqa: E402  (entropy_coding/, library use only)
from kvq.compression.ec_roundtrip import BUNDLE_VERSION, dz_round  # noqa: E402
from pipelines.calibration.analyze_bases import regularize_batch  # noqa: E402

EPS = 1e-4
RUN_ID = "ec_calib_compact8_train_llama31_8b"
RAW_ROOT = REPO / "artifacts/calibration" / RUN_ID / "01_raw"
ROLES = REPO / "artifacts/calibration_splits/ec_compact8_train_26/roles.json"
CCA_STATS = REPO / "artifacts/bases/jointqk_llama31_8b_longbench_compact8_n400.pt"
OUT_DIR = REPO / "artifacts/ec/llama31_8b"


def raw_path(config: str, row_index: int) -> Path:
    name = f"longbench__{config}__row{int(row_index):05d}__train.pt"
    hits = sorted(RAW_ROOT.glob(f"shard_*/{name}"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: found {len(hits)} matches under {RAW_ROOT}")
    return hits[0]


class RawPool:
    """mmap-backed k_post access for a list of (config, row_index) rows."""

    def __init__(self, rows: list[list]) -> None:
        self.rows = [(c, int(i)) for c, i in rows]
        self._arts = {}

    def art(self, key):
        if key not in self._arts:
            self._arts[key] = torch.load(raw_path(*key), map_location="cpu",
                                         mmap=True, weights_only=False)
        return self._arts[key]

    def k_slices(self, l: int, h: int):
        for key in self.rows:
            a = self.art(key)
            T = int(a["prompt_length"])
            yield a["k_post"][l, h, :T].float()

    def fetch_codes(self, l, h, F, mu, device="cpu"):
        """Centered rotated codes, concatenated over the pool: (N, d) float64."""
        Fd = F[l, h].double().to(device)
        mud = mu[l, h].double().to(device)
        return torch.cat(
            [(k.to(device).double() - mud) @ Fd for k in self.k_slices(l, h)], 0)

    def k_mean_and_cov(self, L, Hkv, d, device="cpu"):
        sum1 = torch.zeros(L, Hkv, d, dtype=torch.float64, device=device)
        sum2 = torch.zeros(L, Hkv, d, d, dtype=torch.float64, device=device)
        n = 0
        for key in self.rows:
            a = self.art(key)
            T = int(a["prompt_length"])
            k = a["k_post"][:, :, :T].to(device).double()  # (L,Hkv,T,d)
            sum1 += k.sum(2)
            sum2 += torch.einsum("lhtd,lhte->lhde", k, k)
            n += T
            del k
        mu = sum1 / n
        cov = sum2 / n - torch.einsum("lhd,lhe->lhde", mu, mu)
        return mu.float().cpu(), cov.float().cpu(), n


def entropy_256bins(r: torch.Tensor) -> torch.Tensor:
    """Differential entropy (bits) per column of r (N, d) via 256-bin histograms.
    GPU-vectorized equivalent of ec._diff_entropy_from_fetch's inner loop."""
    n, d = r.shape
    bins = 256
    lo = r.min(0).values
    hi = r.max(0).values
    width = ((hi - lo) / bins).clamp_min(1e-30)
    idx = ((r - lo.unsqueeze(0)) / width.unsqueeze(0)).clamp(0, bins - 1).long()
    flat = idx + torch.arange(d, device=r.device).unsqueeze(0) * bins
    counts = torch.bincount(flat.reshape(-1), minlength=d * bins).double().reshape(d, bins)
    p = counts / counts.sum(1, keepdim=True).clamp_min(1)
    h_disc = -(p * p.clamp_min(1e-30).log2()).sum(1)
    h = h_disc + width.double().log2()
    # Degenerate (constant) coords: empirical entropy -> very low, not -inf.
    return torch.where(hi > lo, h, torch.full_like(h, -30.0))


def entropy_per_coord_dev(idx: torch.Tensor) -> torch.Tensor:
    """Device-safe copy of ec._entropy_per_coord (the original allocates CPU
    helpers and crashes on CUDA inputs)."""
    n, d = idx.shape
    dev = idx.device
    mins = idx.min(dim=0).values
    shifted = idx - mins.unsqueeze(0)
    widths = (shifted.max(dim=0).values + 1).clamp_min(1)
    offsets = torch.zeros(d, dtype=torch.long, device=dev)
    if d > 1:
        offsets[1:] = torch.cumsum(widths, 0)[:-1]
    total = int(offsets[-1].item() + widths[-1].item())
    flat = (shifted + offsets.unsqueeze(0)).reshape(-1)
    counts = torch.bincount(flat, minlength=total).double()
    seg = torch.repeat_interleave(torch.arange(d, device=dev), widths)
    col_sum = torch.zeros(d, dtype=torch.double, device=dev).scatter_add_(0, seg, counts)
    p = counts / col_sum[seg].clamp_min(1.0)
    term = torch.where(p > 0, -p * p.clamp_min(1e-30).log2(), torch.zeros_like(p))
    return torch.zeros(d, dtype=torch.double, device=dev).scatter_add_(0, seg, term)


def solve_delta_matched_dev(r, Hj_head, dz, delta0_head, n_iter=24, bracket_bits=6.0):
    """ec._solve_delta_matched with the device-safe entropy (same bisection)."""
    r = r.double()
    log2d0 = delta0_head.double().clamp_min(1e-30).log2()
    lo = log2d0 - bracket_bits
    hi = log2d0 + bracket_bits
    target = Hj_head.double()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        delta = torch.pow(2.0, mid)
        idx = (torch.sign(r) * torch.floor(r.abs() / delta.unsqueeze(0) + dz)).long()
        too_high = entropy_per_coord_dev(idx) > target
        lo = torch.where(too_high, mid, lo)
        hi = torch.where(too_high, hi, mid)
    return torch.pow(2.0, 0.5 * (lo + hi))


def build_basis(basis: str, cca: dict, k_mean: torch.Tensor, k_cov: torch.Tensor):
    """Return (forward (L,Hkv,d,d), inverse (L,Hkv,d,d)) float32."""
    L, Hkv, d, _ = cca["sigma_q"].shape
    if basis == "r_sym":
        F = cca["R_sym"].float()
        return F, F.transpose(-1, -2).contiguous()
    if basis == "hadamard":
        # TurboQuantPress K-side rotation: MSECompressor(d, bits, seed=42+1000*L).Pi,
        # rotated = k @ Pi.T -> forward = Pi.T, shared across heads of a layer.
        sys.path.insert(0, str(REPO / "vendor"))
        from turboquant_pytorch.turboquant import generate_rotation_matrix
        F = torch.empty(L, Hkv, d, d)
        for layer in range(L):
            Pi = generate_rotation_matrix(d, seed=42 + 1000 * layer, device="cpu").float()
            F[layer] = Pi.T.unsqueeze(0).expand(Hkv, d, d)
        return F, F.transpose(-1, -2).contiguous()
    if basis == "qpca":
        # Centered QPCA (sigma_q x centered K covariance), mirroring the harness.
        # build_qpca_basis whitens via sqrt/rsqrt of Sigma_Q's eigenvalues. Sigma_Q
        # is PSD by construction (a second moment E[qq^T]); over deployed layers
        # (l>=1) its min eigenvalue is +8.4e-6, so QPCA is built on the RAW moments
        # there (faithful, no regularization). The ONLY ill-conditioned block is
        # layer 0 (the attention-sink layer): cond ~7e6, so eigh returns a tiny
        # NEGATIVE eigenvalue (~-2e-5) from float roundoff -> sqrt(neg)=NaN -> the
        # earlier crash. Layer 0 is excluded from every metric AND skipped by the
        # bundle loader (range(1, L)), so its basis is never used; we overwrite it
        # with identity purely so the entropy/rate fit loop (which iterates all L)
        # stays finite. Deployed layers are bit-identical to unregularized QPCA.
        #
        # Layer 0 is sliced off BEFORE build_qpca_basis, not just overwritten
        # after: feeding its indefinite matrix in produces sqrt(neg)=NaN, and the
        # downstream eigh(A) on the NaN-laden matrix spins in LAPACK (minutes-long
        # hang). Building only l>=1 keeps every intermediate finite and fast.
        d = cca["sigma_q"].shape[-1]
        q = ec.build_qpca_basis(cca["sigma_q"][1:], k_cov[1:])
        fwd1, inv1 = q["forward"].float(), q["inverse"].float()
        eye = torch.eye(d).expand(1, fwd1.shape[1], d, d)
        fwd = torch.cat([eye, fwd1], dim=0).contiguous()
        inv = torch.cat([eye, inv1], dim=0).contiguous()
        return fwd, inv
    raise ValueError(f"unknown basis {basis!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", required=True, choices=["r_sym", "hadamard", "qpca"])
    ap.add_argument("--dz", type=float, required=True)
    ap.add_argument("--b-target", type=float, default=1.95)
    ap.add_argument("--no-match-rate", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    match_rate = not args.no_match_rate
    dev = torch.device(args.device)
    t0 = time.time()

    roles = json.loads(ROLES.read_text())
    fit_pool = RawPool(roles["fit"])
    sel_pool = RawPool(roles["selection"])

    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    L, Hkv, d, _ = cca["sigma_q"].shape
    print(f"[fit] basis={args.basis} dz={args.dz:g} b={args.b_target} mr={match_rate} "
          f"| L={L} Hkv={Hkv} d={d} | fit_rows={len(fit_pool.rows)} sel_rows={len(sel_pool.rows)}")

    k_mean, k_cov, ntok = fit_pool.k_mean_and_cov(L, Hkv, d, device=dev)
    print(f"[fit] k_mean/k_cov from {ntok:,} calib tokens ({time.time()-t0:.0f}s)", flush=True)

    F, Finv = build_basis(args.basis, cca, k_mean, k_cov)
    ell = (F.transpose(-1, -2).double()
           @ regularize_batch(cca["sigma_q"], EPS).double()
           @ F.double()).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)  # (L,Hkv,d)

    delta = torch.empty(L, Hkv, d)
    model: dict = {}
    for l in range(L):
        for h in range(Hkv):
            r_gpu = fit_pool.fetch_codes(l, h, F, k_mean, device=dev)  # (N, d) f64
            hcoord = entropy_256bins(r_gpu).cpu()                      # (d,)
            target = hcoord + 0.5 * ell[l, h].log2()
            Hj = (target - target.mean() + args.b_target).clamp_min(0.0)
            Hj = Hj * (args.b_target * d / Hj.sum().clamp_min(1e-9))
            d0 = torch.pow(2.0, (hcoord - Hj))
            if match_rate:
                dlh = solve_delta_matched_dev(r_gpu, Hj.to(dev), args.dz, d0.to(dev)).cpu()
            else:
                dlh = d0
            delta[l, h] = dlh.float()
            idx = dz_round(r_gpu, dlh.double().to(dev).clamp_min(1e-12).unsqueeze(0),
                           args.dz).long()
            per = []
            for j in range(d):
                vals, counts = torch.unique(idx[:, j], return_counts=True)
                p = (counts.double() / counts.sum()).clamp_min(1e-12)
                per.append((vals.cpu().numpy(), (p / p.sum()).cpu().numpy()))
            model[(l, h)] = per
            del r_gpu, idx
        print(f"[fit] layer {l+1}/{L} done ({time.time()-t0:.0f}s)", flush=True)

    # Held-out coded rate (frozen model) on selection rows. Single pass: encode
    # per (l, h, row); pooled = sum over rows, per-task from row membership.
    # Symbol mapping (escape -> nearest alphabet entry) matches coded_bits_eval.
    import constriction
    import numpy as np
    bits_by_task: dict[str, float] = {}
    tok_by_task: dict[str, int] = {}
    for l in range(1, L):                      # layer 0 excluded (convention)
        for h in range(Hkv):
            per = model[(l, h)]
            dlh = delta[l, h].double().clamp_min(1e-12)
            for (task, ridx) in sel_pool.rows:
                a = sel_pool.art((task, ridx))
                T = int(a["prompt_length"])
                k = a["k_post"][l, h, :T].double()
                r = (k - k_mean[l, h].double()) @ F[l, h].double()
                idx = dz_round(r, dlh.unsqueeze(0), args.dz).long().numpy()
                tok_by_task[task] = tok_by_task.get(task, 0) + (T if h == 0 and l == 1 else 0)
                for j in range(d):
                    vals, p = per[j]
                    if vals.size <= 1:
                        continue
                    col = idx[:, j]
                    pos = np.clip(np.searchsorted(vals, col), 0, vals.size - 1)
                    left = np.clip(pos - 1, 0, vals.size - 1)
                    choose_left = np.abs(vals[left] - col) <= np.abs(vals[pos] - col)
                    sym = np.where(choose_left, left, pos).astype(np.int32)
                    enc = constriction.stream.queue.RangeEncoder()
                    enc.encode(sym, constriction.stream.model.Categorical(p, perfect=False))
                    bits_by_task[task] = bits_by_task.get(task, 0.0) + len(enc.get_compressed()) * 32
        print(f"[rate] layer {l+1}/{L} done ({time.time()-t0:.0f}s)", flush=True)
    # bits accumulated over (L-1)*Hkv (l,h) pairs; tokens counted once per row.
    denom = {t: tok_by_task[t] * (L - 1) * Hkv * d for t in tok_by_task}
    per_task = {t: bits_by_task[t] / denom[t] for t in sorted(bits_by_task)}
    pooled_rate = sum(bits_by_task.values()) / sum(denom.values())
    print(f"[rate] pooled held-out = {pooled_rate:.4f} b/c | per-task: "
          + " ".join(f"{t}={r:.3f}" for t, r in per_task.items()))

    # Pad the frozen model into bundle tensors.
    vmax = max(len(v) for per in model.values() for v, _ in per)
    support_vals = torch.full((L, Hkv, d, vmax), float("inf"))
    support_probs = torch.zeros(L, Hkv, d, vmax)
    support_lens = torch.zeros(L, Hkv, d, dtype=torch.int32)
    for (l, h), per in model.items():
        for j, (vals, p) in enumerate(per):
            n = len(vals)
            support_vals[l, h, j, :n] = torch.from_numpy(vals).float()
            support_probs[l, h, j, :n] = torch.from_numpy(p).float()
            support_lens[l, h, j] = n
    print(f"[fit] alphabet vmax={vmax}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (f"ec_bundle__{args.basis}__b{args.b_target:g}"
                     f"__dz{args.dz:g}__compact8train{len(fit_pool.rows)}.pt")
    torch.save({
        "version": BUNDLE_VERSION,
        "model_tag": "llama31_8b",
        "basis": args.basis,
        "dz": args.dz,
        "b_target": args.b_target,
        "match_rate": match_rate,
        "n_layers": L, "n_kv_heads": Hkv, "head_dim": d,
        "forward": F, "inverse": Finv, "mu": k_mean, "delta": delta,
        "support_vals": support_vals, "support_probs": support_probs,
        "support_lens": support_lens,
        "achieved_rate_heldout_pooled": pooled_rate,
        "achieved_rate_per_task": per_task,
        "fit_rows": roles["fit"], "selection_rows": roles["selection"],
        "fit_tokens": ntok,
        "cca_stats_path": str(CCA_STATS),
        "cca_stats_mtime": int(CCA_STATS.stat().st_mtime),
        "hadamard_seed_rule": "42+1000*L",
        "ell_source": "cca_bundle_sigma_q_n400",
    }, out)
    print(f"[fit] wrote {out} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
