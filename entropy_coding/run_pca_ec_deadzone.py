#!/usr/bin/env python3
"""K-cache quantization with a CALIBRATE/EVAL split over the `full` dataset.

Calibrate (basis, std, bit allocation, and QPCA-EC's uniform-step delta + per-coord
entropy model) on a chosen subset of `full` example indices; evaluate top-1/top-5/
MSE/logit-err and the QPCA-EC coded rate on the DISJOINT remaining examples.

This makes the comparison honest: nothing fit on eval. QPCA-EC's range-coder
frequency model is FROZEN on calibration and applied to eval indices, so its
reported bits/coord is the true held-out coded rate (it rises if calibration
generalizes poorly -- the correct penalty).

Calibration second moments are recomputed from the CALIB raw tensors (NOT from
pooled_stats.pt, which was pooled over all of full and would leak eval into the basis).

    python run_pca_split.py --calib-idx 0 1 2 3 --eval-idx 4 5 6 7
    python run_pca_split.py --calib-idx 0 1 2 3 4 5 --eval-idx 6 7 --bits 2 3 4
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _ensure_environment():
    needed = ("matplotlib", "numpy", "torch")
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    if not missing:
        return
    if os.environ.get("_QNB_REEXECED") != "1" and os.environ.get("QNB_NO_REEXEC") != "1":
        cands = []
        if os.environ.get("QNB_PYTHON"):
            cands.append(Path(os.environ["QNB_PYTHON"]))
        for base in (SCRIPT_DIR, SCRIPT_DIR.parent):
            for name in (".venv-qnb", ".venv", "venv", "env"):
                cands.append(base / name / "bin" / "python")
        cur = Path(sys.executable).resolve()
        for py in cands:
            try:
                rp = py.resolve()
            except OSError:
                continue
            if rp.exists() and rp != cur:
                os.environ["_QNB_REEXECED"] = "1"
                print(f"[env] {', '.join(missing)} missing; re-launching under {rp}", flush=True)
                os.execv(str(rp), [str(rp), *sys.argv])
    sys.exit("\nMissing deps. Activate venv: source ../.venv-qnb/bin/activate\n")


_ensure_environment()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FIG_DUMP = SCRIPT_DIR / "fig_dump"
FIG_DUMP.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    path = FIG_DUMP / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] wrote {path}")


for cand in (Path.cwd().resolve(), Path.cwd().resolve().parent, SCRIPT_DIR, SCRIPT_DIR.parent):
    s = str(cand)
    if s not in sys.path:
        sys.path.insert(0, s)

import _bootstrap  # noqa: E402,F401
REPO = Path(_bootstrap.__file__).resolve().parent

from kvq.compression.lloyd_max import Stage1MSECompressor  # noqa: E402
from kvq.compression.per_coord import PerCoordCompressor  # noqa: E402
from pipelines.calibration.analyze_bases import (  # noqa: E402
    allocate_bits, jointqk_basis, regularize_batch,
)

EPS = 1e-4
FULL_DIR = "/mnt/c/JointQK_data/query_stats_longbench_under4k"

def data_root():
    r = REPO / "notebooks" / "data" / FULL_DIR
    if not (r / "manifest.json").exists():
        sys.exit(f"full dataset not found at {r} (expected manifest.json). "
                 f"Download it there first.")
    return r


def load_manifest(root):
    return json.loads((root / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# Calibration moments from a SUBSET of raw examples (no pooled_stats.pt -> no leak).
# ---------------------------------------------------------------------------
def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))


MOMENTS_CACHE = SCRIPT_DIR / "moments_cache"


def _calib_cache_path(root, calib_idx):
    """Cache file keyed by the sorted calib index set + the dataset dir name.
    Re-running with the same --calib-idx reuses the saved (small) moments instead
    of re-reading the multi-GB raw examples."""
    MOMENTS_CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{root.name}__idx_" + "_".join(str(i) for i in sorted(calib_idx))
    return MOMENTS_CACHE / f"{key}.pt"


def calib_moments(root, manifest, calib_idx):
    """Recompute per-(L,H) second moments + mean + covariance of K, and Sigma_Q,
    from ONLY the calib examples. Cached on disk keyed by the calib index set so
    repeat runs skip re-reading the raw examples."""
    cache_path = _calib_cache_path(root, calib_idx)
    if cache_path.exists():
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        meta = c["meta"]
        print(f"calib moments: CACHE HIT {cache_path.name} "
              f"({len(calib_idx)} examples, {c['ntok']} tokens) | "
              f"L={meta['n_layers']} Hkv={meta['n_kv_heads']} d={meta['d_head']}")
        return c["sigma_q"], c["sigma_k"], c["k_mean"], c["k_cov"], meta
    print(f"calib moments: cache miss, computing from raw ({len(calib_idx)} examples)...")
    exs = manifest["examples"]
    first = torch.load(root / exs[calib_idx[0]]["file"], map_location="cpu", weights_only=False)
    q0, k0 = first["q_post"], first["k_post"]
    L, Hq, _, d = q0.shape
    _, Hkv, _, _ = k0.shape
    gs = Hq // Hkv

    sumq = torch.zeros(L, Hq, d, dtype=torch.float64)
    sumk = torch.zeros(L, Hkv, d, dtype=torch.float64)
    sq2 = torch.zeros(L, Hq, d, d, dtype=torch.float64)   # E[q q^T] per q-head
    sk2 = torch.zeros(L, Hkv, d, d, dtype=torch.float64)  # E[k k^T] per kv-head
    ntok = 0
    for i in calib_idx:
        art = torch.load(root / exs[i]["file"], map_location="cpu", weights_only=False)
        T = int(art["prompt_length"])
        q = art["q_post"][:, :, :T, :].double()    # (L,Hq,T,d)
        k = art["k_post"][:, :, :T, :].double()    # (L,Hkv,T,d)
        sumq += q.sum(2); sumk += k.sum(2)
        sq2 += torch.einsum("lhtd,lhte->lhde", q, q)
        sk2 += torch.einsum("lhtd,lhte->lhde", k, k)
        ntok += T
    mq = sumq / ntok                                # (L,Hq,d)
    mk = sumk / ntok                                # (L,Hkv,d)
    Eqq = sq2 / ntok                                # 2nd moment (uncentered) per q-head
    Ekk = sk2 / ntok
    # GQA-pool Sigma_Q over the group (sum, matching build_second_moments)
    sigma_q = Eqq.reshape(L, Hkv, gs, d, d).sum(2)  # (L,Hkv,d,d) uncentered 2nd moment, pooled
    sigma_k = Ekk                                   # (L,Hkv,d,d) uncentered 2nd moment
    k_mean = mk                                     # (L,Hkv,d)
    # centered cov of K per kv-head
    k_cov = Ekk - torch.einsum("lhd,lhe->lhde", mk, mk)
    meta = dict(n_layers=L, n_q_heads=Hq, n_kv_heads=Hkv, d_head=d, group_size=gs)
    print(f"calib moments from {len(calib_idx)} examples, {ntok} tokens | "
          f"L={L} Hkv={Hkv} d={d} (GQA {gs})")
    sigma_q, sigma_k, k_mean, k_cov = sigma_q.float(), sigma_k.float(), k_mean.float(), k_cov.float()
    torch.save({"sigma_q": sigma_q, "sigma_k": sigma_k, "k_mean": k_mean, "k_cov": k_cov,
                "meta": meta, "ntok": ntok, "calib_idx": sorted(calib_idx)}, cache_path)
    print(f"  cached moments -> {cache_path.name}")
    return sigma_q, sigma_k, k_mean, k_cov, meta


# ---------------------------------------------------------------------------
# Bases.
# ---------------------------------------------------------------------------
class TurboQuantWrapper:
    def __init__(self, head_dim, bits, seed=20260505, device="cpu"):
        self.tq = Stage1MSECompressor(head_dim=head_dim, bits=bits, seed=seed, device=device)
        self.forward_map = self.tq.Pi.T
        self.inverse_map = self.tq.Pi

    def to(self, device):
        self.tq.Pi = self.tq.Pi.to(device)
        self.tq.centroids = self.tq.centroids.to(device)
        self.tq.device = str(device)
        self.forward_map = self.tq.Pi.T
        self.inverse_map = self.tq.Pi
        return self

    def roundtrip(self, k):
        return self.tq.roundtrip(k.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def build_turboquant(d_head, bits_list):
    return {b: TurboQuantWrapper(head_dim=d_head, bits=b, seed=20260505) for b in bits_list}


def build_jointqk_basis(sigma_q, sigma_k, eps=EPS):
    R = jointqk_basis(sigma_q, sigma_k, eps=eps)
    forward, inverse = R, R.transpose(-1, -2)
    q_diag = (inverse @ regularize_batch(sigma_q, eps) @ forward).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    k_diag = (inverse @ regularize_batch(sigma_k, eps) @ forward).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    return {"forward": forward, "inverse": inverse, "score": q_diag * k_diag, "std": k_diag.sqrt()}


def build_qpca_basis(sigma_q, sigma_k):
    sq = _sym(sigma_q.to(torch.float64))
    sk = _sym(sigma_k.to(torch.float64))
    ev, U = torch.linalg.eigh(sq)
    sqrt_mq = U @ torch.diag_embed(ev.sqrt()) @ U.transpose(-1, -2)
    isqrt_mq = U @ torch.diag_embed(ev.rsqrt()) @ U.transpose(-1, -2)
    A = _sym(sqrt_mq @ sk @ sqrt_mq)
    lam, V = torch.linalg.eigh(A)
    order = torch.argsort(lam, dim=-1, descending=True)
    lam = torch.gather(lam, -1, order).clamp_min(1e-30)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand(*V.shape[:-1], -1))
    forward = sqrt_mq @ V
    inverse = V.transpose(-1, -2) @ isqrt_mq
    fwdt = forward.transpose(-1, -2)
    q_diag = (fwdt @ sq @ forward).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    k_diag = (fwdt @ sk @ forward).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    score = k_diag * (q_diag) ** 0.55
    return {"forward": forward, "inverse": inverse, "score": score, "std": k_diag.sqrt(),
            "q_diag": q_diag, "k_diag": k_diag, "sigma_q": sigma_q, "sigma_k": sigma_k}


def build_pca_basis(sigma_k, sigma_q, eps=EPS):
    sk = regularize_batch(sigma_k, eps).to(torch.float64)
    sq = regularize_batch(sigma_q, eps).to(torch.float64)
    lam, V = torch.linalg.eigh(sk)
    order = torch.argsort(lam, dim=-1, descending=True)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand(*V.shape[:-1], -1))
    forward, inverse = V, V.transpose(-1, -2)
    fwdt = forward.transpose(-1, -2)
    q_diag = (fwdt @ sq @ forward).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    k_diag = (fwdt @ sk @ forward).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    return {"forward": forward, "inverse": inverse, "score": q_diag * k_diag, "std": k_diag.sqrt()}


# ---------------------------------------------------------------------------
# Fixed-width compressors.
# ---------------------------------------------------------------------------
def build_compressors(basis, b, n_layers, n_kv_heads, std_override=None, max_coord_bits=8):
    forward, inverse = basis["forward"], basis["inverse"]
    std = basis["std"] if std_override is None else std_override
    allocs = allocate_bits(basis["score"], b, max_coord_bits=max_coord_bits).cpu()
    comps = {}
    for l in range(n_layers):
        for h in range(n_kv_heads):
            comps[(l, h)] = PerCoordCompressor(
                bits_per_coord=allocs[l, h], std_per_coord=std[l, h],
                forward_map=forward[l, h], inverse_map=inverse[l, h])
    return comps


class CenteredRoundtrip:
    def __init__(self, inner, mu):
        self.inner = inner
        self.mu = mu.reshape(1, -1).float()
        self.forward_map = inner.forward_map

    def to(self, device):
        self.inner.to(device)
        self.mu = self.mu.to(device)
        self.forward_map = self.inner.forward_map
        return self

    def roundtrip(self, k):
        return self.inner.roundtrip(k - self.mu) + self.mu


def laplacian_lloyd_max(bits, iters=200):
    n = max(1, 2 ** int(bits))
    if n == 1:
        return torch.zeros(1)
    scale = 1.0 / (2 ** 0.5)
    x = torch.linspace(-12, 12, 200001).double()
    p = torch.exp(-x.abs() / scale); p = p / p.sum()
    cent = torch.distributions.Laplace(0.0, scale).icdf(
        torch.linspace(0.5 / n, 1 - 0.5 / n, n).double())
    for _ in range(iters):
        edges = (cent[1:] + cent[:-1]) / 2
        idx = torch.bucketize(x, edges)
        wsum = torch.zeros(n, dtype=torch.double).scatter_add_(0, idx, p)
        xsum = torch.zeros(n, dtype=torch.double).scatter_add_(0, idx, p * x)
        new = torch.where(wsum > 0, xsum / wsum.clamp_min(1e-30), cent)
        if (new - cent).abs().max() < 1e-9:
            cent = new; break
        cent = new
    edges = (cent[1:] + cent[:-1]) / 2
    idx = torch.bucketize(x, edges)
    var = (p * (cent[idx]) ** 2).sum()
    return (cent / var.clamp_min(1e-30).sqrt()).float()


def build_qpca_centered_compressors(qpca_cen, qpca_unc, k_mean, b, n_layers, n_kv_heads,
                                    widen_coords=(0,), widen_mult=2.5):
    fwd_c = qpca_cen["forward"]
    fwdt_c = fwd_c.transpose(-1, -2)
    sk = qpca_unc["sigma_k"].to(fwd_c.dtype)
    su = (fwdt_c @ regularize_batch(sk, EPS).to(fwd_c.dtype) @ fwd_c
          ).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30).sqrt()
    std = su.clone()
    for j in widen_coords:
        std[:, :, j] = std[:, :, j] * widen_mult
    allocs = allocate_bits(qpca_cen["score"], b, max_coord_bits=8)
    comps = {}
    for l in range(n_layers):
        for h in range(n_kv_heads):
            comps[(l, h)] = PerCoordCompressor(
                bits_per_coord=allocs[l, h], std_per_coord=std[l, h],
                forward_map=fwd_c[l, h], inverse_map=qpca_cen["inverse"][l, h])
    lap_cache = {}
    for l in range(n_layers):
        for h in range(n_kv_heads):
            comp = comps[(l, h)]
            for j in [0]:
                bj = int(comp.bits_int[j])
                if bj not in lap_cache:
                    lap_cache[bj] = laplacian_lloyd_max(bj)
                kk = lap_cache[bj].numel()
                comp.codebooks_padded[j, :kk] = lap_cache[bj].to(comp.codebooks_padded) * std[l, h, j]
    return {(l, h): CenteredRoundtrip(comps[(l, h)], k_mean[l, h])
            for l in range(n_layers) for h in range(n_kv_heads)}


# ---------------------------------------------------------------------------
# QPCA-EC: uniform quantizer + ECSQ rate allocation; coder model frozen on calib.
# ---------------------------------------------------------------------------
def _dz_round(r, delta, dz):
    """Deadzone uniform quantizer index. dz=0.5 -> round-to-nearest (== torch.round).
    Smaller dz widens the central (zero) bin: deadzone half-width = (1-dz)*delta,
    every other bin keeps width delta. Sign-magnitude so it stays symmetric."""
    return torch.sign(r) * torch.floor(r.abs() / delta + dz)


def _dz_dequant(idx, delta, dz):
    """Bin-centroid reconstruction for the deadzone quantizer (assumes uniform
    density within a bin). Reduces to idx*delta when dz=0.5."""
    return (idx + torch.sign(idx) * (0.5 - dz)) * delta


class UniformECRoundtrip:
    def __init__(self, fwd, inv, mu, delta, dz=0.5):
        self.fwd = fwd.float(); self.inv = inv.float()
        self.mu = mu.reshape(1, -1).float()
        self.delta = delta.float().reshape(1, -1).clamp_min(1e-12)
        self.dz = float(dz)
        self.forward_map = self.fwd

    def to(self, dev):
        self.fwd = self.fwd.to(dev); self.inv = self.inv.to(dev)
        self.mu = self.mu.to(dev); self.delta = self.delta.to(dev)
        self.forward_map = self.fwd
        return self

    def roundtrip(self, k):
        r = (k - self.mu) @ self.fwd
        idx = _dz_round(r, self.delta, self.dz)
        q = _dz_dequant(idx, self.delta, self.dz)
        return q @ self.inv + self.mu


def _codes_for_idx(root, manifest, idx_list, F, mu, n_layers, n_kv_heads, d):
    """Return a fetch(l,h) -> centered codes r concatenated over idx_list. Loads
    raw examples lazily on first fetch call, so if the caller never fetches (e.g.
    EC model cache hit) nothing is read from disk."""
    cache = {"arts": None}

    def _load():
        if cache["arts"] is None:
            cache["arts"] = [torch.load(root / manifest["examples"][i]["file"],
                                        map_location="cpu", weights_only=False)
                             for i in idx_list]
        return cache["arts"]

    def fetch(l, h):
        cols = []
        for art in _load():
            T = int(art["prompt_length"])
            k = art["k_post"][l, h, :T].double()
            cols.append((k - mu[l, h].double()) @ F[l, h].double())
        return torch.cat(cols, 0)
    return fetch


def _diff_entropy_from_fetch(fetch, n_layers, n_kv_heads, d, bins=256):
    h = torch.full((n_layers, n_kv_heads, d), -30.0, dtype=torch.float64)
    for l in range(n_layers):
        for hh in range(n_kv_heads):
            r = fetch(l, hh)
            for j in range(d):
                x = r[:, j]; lo, hi = float(x.min()), float(x.max())
                if hi <= lo:
                    continue
                width = (hi - lo) / bins
                idx = ((x - lo) / width).clamp(0, bins - 1).long()
                c = torch.bincount(idx, minlength=bins).double()
                p = c / c.sum().clamp_min(1)
                Hbins = -(p * p.clamp_min(1e-30).log2()).sum()
                h[l, hh, j] = Hbins + math.log2(width)
    return h


def freeze_coder_model(fetch, delta, n_layers, n_kv_heads, d, dz=0.5):
    """Build, on CALIB codes, the per-(l,h,coord) frequency model: the sorted unique
    integer bin values and their probabilities. Returned as nested dict so eval indices
    can be encoded with this FIXED model (held-out rate)."""
    model = {}
    for l in range(n_layers):
        for hh in range(n_kv_heads):
            r = fetch(l, hh)
            idx = _dz_round(r, delta[l, hh].double().clamp_min(1e-12), dz).long()
            per = []
            for j in range(d):
                vals, counts = torch.unique(idx[:, j], return_counts=True)
                p = (counts.double() / counts.sum()).clamp_min(1e-12)
                p = (p / p.sum()).cpu().numpy().astype(np.float64)
                per.append((vals.cpu().numpy().astype(np.int64), p))
            model[(l, hh)] = per
    return model


def coded_bits_eval(fetch_eval, delta, model, n_layers, n_kv_heads, d, dz=0.5):
    """Encode EVAL indices with the FROZEN calib model -> true held-out coded bits.
    Symbols unseen in calib are assigned the model's min prob (escape) so the coder
    stays valid; this penalizes poor generalization, as it should."""
    import constriction
    tot_bits, tot_tok = 0, 0
    for l in range(1, n_layers):              # layer 0 excluded (matches metrics)
        for hh in range(n_kv_heads):
            r = fetch_eval(l, hh)
            idx = _dz_round(r, delta[l, hh].double().clamp_min(1e-12), dz).long().cpu().numpy()
            tot_tok += idx.shape[0]
            per = model[(l, hh)]
            for j in range(d):
                vals, p = per[j]
                if vals.size <= 1:
                    continue
                col = idx[:, j]
                # map eval symbols onto calib alphabet; out-of-alphabet -> nearest calib bin
                pos = np.searchsorted(vals, col)
                pos = np.clip(pos, 0, vals.size - 1)
                # snap to nearest (searchsorted gives insertion point)
                left = np.clip(pos - 1, 0, vals.size - 1)
                choose_left = np.abs(vals[left] - col) <= np.abs(vals[pos] - col)
                sym = np.where(choose_left, left, pos).astype(np.int32)
                enc = constriction.stream.queue.RangeEncoder()
                m = constriction.stream.model.Categorical(p, perfect=False)
                enc.encode(sym, m)
                tot_bits += len(enc.get_compressed()) * 32
    return tot_bits / max(1, tot_tok * d)


def _entropy_per_coord(idx):
    """Discrete entropy (bits) of each column of an integer tensor idx (N, d).
    Fully vectorized via a single offset bincount + segment scatter."""
    N, d = idx.shape
    mins = idx.min(dim=0).values
    shifted = idx - mins.unsqueeze(0)                       # >= 0
    widths = (shifted.max(dim=0).values + 1).clamp_min(1)   # (d,) long
    offsets = torch.zeros(d, dtype=torch.long)
    if d > 1:
        offsets[1:] = torch.cumsum(widths, 0)[:-1]
    total = int(offsets[-1].item() + widths[-1].item())
    flat = (shifted + offsets.unsqueeze(0)).reshape(-1)
    counts = torch.bincount(flat, minlength=total).double()
    seg = torch.repeat_interleave(torch.arange(d), widths)  # bin -> column id
    col_sum = torch.zeros(d, dtype=torch.double).scatter_add_(0, seg, counts)
    p = counts / col_sum[seg].clamp_min(1.0)
    term = torch.where(p > 0, -p * p.clamp_min(1e-30).log2(), torch.zeros_like(p))
    return torch.zeros(d, dtype=torch.double).scatter_add_(0, seg, term)


def _solve_delta_matched(r, Hj_head, dz, delta0_head, n_iter=24, bracket_bits=6.0):
    """Per-coord bisection on log2(delta) so the DEADZONED discrete entropy of the
    calib codes r (N,d) matches the per-coord target Hj_head (d,). Entropy is
    monotone decreasing in delta, so this brackets the root around the closed-form
    delta0_head. This is what makes the deadzone an RD knob rather than a free
    rate drop: bits freed at the zero bin are reinvested in a finer step."""
    r = r.double()
    log2d0 = delta0_head.double().clamp_min(1e-30).log2()
    lo = log2d0 - bracket_bits      # small delta -> high entropy
    hi = log2d0 + bracket_bits      # large delta -> low entropy
    target = Hj_head.double()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        delta = torch.pow(2.0, mid)
        idx = (torch.sign(r) * torch.floor(r.abs() / delta.unsqueeze(0) + dz)).long()
        too_high = _entropy_per_coord(idx) > target   # entropy too high -> raise delta
        lo = torch.where(too_high, mid, lo)
        hi = torch.where(too_high, hi, mid)
    return torch.pow(2.0, 0.5 * (lo + hi))


def build_qpca_ec(qpca_cen, qpca_unc, k_mean, b, n_layers, n_kv_heads,
                  fetch_calib, root, calib_idx, dz=0.5, match_rate=False, verbose=False):
    """delta + entropy model from CALIB codes. Cached on disk keyed by
    (dataset, calib-idx, bits, dz, match_rate) -- independent of eval-idx -- so
    sweeping eval splits skips the calib raw-example walk. With match_rate=False,
    delta is the closed-form round-to-nearest step and dz only reshapes the index
    histogram (rate drops). With match_rate=True, delta is re-solved per coord so
    the deadzoned entropy hits Hj and rate stays ~b. Returns (comps, delta, model)."""
    F = qpca_cen["forward"]
    d = F.shape[-1]
    MOMENTS_CACHE.mkdir(parents=True, exist_ok=True)
    key = (f"{root.name}__ecmodel__idx_" + "_".join(str(i) for i in sorted(calib_idx))
           + f"__b{b}__dz{dz:g}__mr{int(match_rate)}")
    cache_path = MOMENTS_CACHE / f"{key}.pt"
    if cache_path.exists():
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        delta, model = c["delta"], c["model"]
        print(f"  [EC] b={b} dz={dz:g} mr={int(match_rate)}: model CACHE HIT {cache_path.name}")
    else:
        print(f"  [EC] b={b} dz={dz:g} mr={int(match_rate)}: model cache miss, fitting on calib...")
        fwdt = F.transpose(-1, -2)
        ell = (fwdt @ regularize_batch(qpca_unc["sigma_q"], EPS).to(F.dtype) @ F
               ).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        h = _diff_entropy_from_fetch(fetch_calib, n_layers, n_kv_heads, d)
        target = h + 0.5 * ell.double().log2()
        Hj = (target - target.mean(-1, keepdim=True) + float(b)).clamp_min(0.0)
        Hj = Hj * (float(b) * d / Hj.sum(-1, keepdim=True).clamp_min(1e-9))
        delta = torch.pow(2.0, (h - Hj))             # closed-form (round-to-nearest) step
        if match_rate:
            print(f"    [EC] match-rate: re-solving deadzone-aware delta (dz={dz:g})...")
            for l in range(n_layers):
                for hh in range(n_kv_heads):
                    delta[l, hh] = _solve_delta_matched(
                        fetch_calib(l, hh), Hj[l, hh], dz, delta[l, hh])
        delta = delta.float()
        model = freeze_coder_model(fetch_calib, delta, n_layers, n_kv_heads, d, dz)
        torch.save({"delta": delta, "model": model, "calib_idx": sorted(calib_idx),
                    "b": b, "dz": dz, "match_rate": match_rate}, cache_path)
        print(f"    cached EC model -> {cache_path.name}")
    comps = {}
    for l in range(n_layers):
        for hh in range(n_kv_heads):
            comps[(l, hh)] = UniformECRoundtrip(F[l, hh], qpca_cen["inverse"][l, hh],
                                                k_mean[l, hh], delta[l, hh], dz=dz)
    return comps, delta, model


# ---------------------------------------------------------------------------
# Scoring (loads one eval example at a time -> bounded memory).
# ---------------------------------------------------------------------------
def score_eval(root, manifest, eval_idx, comps_by_method, k_bits, device):
    L = comps_by_method["QPCA"][k_bits[0]]
    n_layers = max(l for (l, h) in L.keys()) + 1
    n_kv = max(h for (l, h) in L.keys()) + 1
    accums = {m: {b: {l: dict(mse_num=0.0, mse_den=0, logit_num=0.0, logit_den=0,
                              top1_num=0, top1_den=0, top5_num=0, top5_den=0)
                      for l in range(n_layers)} for b in k_bits} for m in comps_by_method}
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
                for m, cb in comps_by_method.items():
                    for bits in k_bits:
                        kh = cb[bits][(l, h)].roundtrip(k).float()
                        err = k - kh; acc = accums[m][bits][l]
                        acc["mse_num"] += float(err.square().sum()); acc["mse_den"] += int(err.numel())
                        ee = err.transpose(0, 1) @ err
                        acc["logit_num"] += float((qq * ee).sum()); acc["logit_den"] += int(q.shape[0]*T*T)
                        ap = q @ kh.transpose(0, 1)
                        acc["top1_num"] += int((real_top == ap.argmax(-1)).sum()); acc["top1_den"] += n_q
                        a5 = ap.topk(k5, dim=-1).indices
                        acc["top5_num"] += int((a5 == real_top.unsqueeze(-1)).any(-1).sum()); acc["top5_den"] += n_q
        print("done")
    return accums, n_layers


def finalize(accums, n_layers, exclude_layer_0=True):
    out = {}
    for m, by_bits in accums.items():
        out[m] = {}
        for bits, by_layer in by_bits.items():
            layers = [l for l in by_layer if not (exclude_layer_0 and l == 0)]
            s = {k: sum(by_layer[l][k] for l in layers) for k in
                 ["mse_num", "mse_den", "logit_num", "logit_den",
                  "top1_num", "top1_den", "top5_num", "top5_den"]}
            out[m][bits] = dict(k_mse=s["mse_num"]/max(1,s["mse_den"]),
                                logit_err=s["logit_num"]/max(1,s["logit_den"]),
                                top1=s["top1_num"]/max(1,s["top1_den"]),
                                top5=s["top5_num"]/max(1,s["top5_den"]))
    return out


def move_to_device(comps, device):
    seen = set()
    for m in comps:
        for b in comps[m]:
            for k in comps[m][b]:
                c = comps[m][b][k]
                if id(c) not in seen:
                    c.to(device); seen.add(id(c))


METHODS = ["TurboQuant", "PCA", "JointQK", "QPCA", "QPCA-EC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", required=True,
                    help="example indices (into full manifest) used for calibration")
    ap.add_argument("--eval-idx", type=int, nargs="+", required=True,
                    help="DISJOINT example indices used for evaluation")
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--widen", type=int, nargs="+", default=[0])
    ap.add_argument("--widen-mult", type=float, default=2.5)
    ap.add_argument("--dz", type=float, default=0.5,
                    help="deadzone rounding offset for QPCA-EC; 0.5=round-to-nearest "
                         "(current behavior), smaller widens the zero bin "
                         "(H.264 intra ~0.375, inter ~0.167)")
    ap.add_argument("--match-rate", action="store_true",
                    help="re-solve QPCA-EC delta per coord so the deadzoned entropy "
                         "hits the target -> rate stays ~b across dz (turns dz into an "
                         "RD knob). Off = closed-form delta, dz just lowers the rate.")
    args = ap.parse_args()
    k_bits = sorted(args.bits)
    overlap = set(args.calib_idx) & set(args.eval_idx)
    if overlap:
        sys.exit(f"calib and eval indices overlap: {sorted(overlap)} -- must be disjoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = data_root(); manifest = load_manifest(root)
    nex = len(manifest["examples"])
    for i in args.calib_idx + args.eval_idx:
        if i >= nex:
            sys.exit(f"index {i} out of range (full has {nex} examples)")
    print(f"full has {nex} examples | calib={args.calib_idx} eval={args.eval_idx} | device {device}")

    sigma_q, sigma_k, k_mean, k_cov, meta = calib_moments(root, manifest, args.calib_idx)
    n_layers, n_kv_heads, d_head = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]

    turbo = build_turboquant(d_head, k_bits)
    jq = build_jointqk_basis(sigma_q, sigma_k)
    pca = build_pca_basis(sigma_k, sigma_q)
    qpca_unc = build_qpca_basis(sigma_q, sigma_k); qpca_unc["sigma_k"] = sigma_k; qpca_unc["sigma_q"] = sigma_q
    qpca_cen = build_qpca_basis(sigma_q, k_cov);   qpca_cen["sigma_k"] = sigma_k

    print(f"\nQPCA widen {tuple(args.widen)} x{args.widen_mult}; QPCA-EC uniform+ECSQ, "
          f"dz={args.dz:g} ({'round-to-nearest' if args.dz == 0.5 else 'deadzone'}), "
          f"match_rate={args.match_rate}, coder model frozen on calib")

    comps = {}
    comps["TurboQuant"] = {b: {(l, h): turbo[b] for l in range(n_layers) for h in range(n_kv_heads)} for b in k_bits}
    comps["PCA"] = {b: build_compressors(pca, b, n_layers, n_kv_heads) for b in k_bits}
    comps["JointQK"] = {b: build_compressors(jq, b, n_layers, n_kv_heads) for b in k_bits}
    comps["QPCA"] = {b: build_qpca_centered_compressors(
        qpca_cen, qpca_unc, k_mean, b, n_layers, n_kv_heads,
        widen_coords=tuple(args.widen), widen_mult=args.widen_mult) for b in k_bits}

    # QPCA-EC: fit delta+model on CALIB codes, measure rate on EVAL codes (frozen model)
    F = qpca_cen["forward"]
    fetch_calib = _codes_for_idx(root, manifest, args.calib_idx, F, k_mean, n_layers, n_kv_heads, d_head)
    fetch_eval = _codes_for_idx(root, manifest, args.eval_idx, F, k_mean, n_layers, n_kv_heads, d_head)
    comps["QPCA-EC"] = {}
    ec_rate = {}
    for b in k_bits:
        ec_comps, delta, model = build_qpca_ec(qpca_cen, qpca_unc, k_mean, b,
                                               n_layers, n_kv_heads, fetch_calib,
                                               root, args.calib_idx, dz=args.dz,
                                               match_rate=args.match_rate)
        comps["QPCA-EC"][b] = ec_comps
        ec_rate[b] = coded_bits_eval(fetch_eval, delta, model, n_layers, n_kv_heads, d_head, dz=args.dz)
        print(f"  [EC] b={b}: held-out constriction rate = {ec_rate[b]:.3f} bits/coord")

    move_to_device(comps, device)

    accums, n_layers = score_eval(root, manifest, args.eval_idx, comps, k_bits, device)
    final = finalize(accums, n_layers, exclude_layer_0=True)

    print(f"\n{'method':<11} | {'b':<2} | {'rate(b/c)':>9} | {'top-1':>7} | {'top-5':>7} | "
          f"{'k_mse':>11} | {'logit_err':>11}")
    print("-" * 78)
    for m in METHODS:
        for b in k_bits:
            dd = final[m][b]
            rate = ec_rate[b] if m == "QPCA-EC" else float(b)
            print(f"{m:<11} | {b:<2} | {rate:>9.3f} | {dd['top1']:>7.4f} | {dd['top5']:>7.4f} | "
                  f"{dd['k_mse']:>11.3e} | {dd['logit_err']:>11.3e}")
        print()
    print("rate(b/c): baselines = grid-bits; QPCA-EC = HELD-OUT constriction bits "
          "(model frozen on calib, applied to eval). Calib/eval disjoint.")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    titles = {"top1": ("top-1","higher"), "top5": ("top-5","higher"),
              "k_mse": ("k_mse","lower"), "logit_err": ("logit_err","lower")}
    colors = {"TurboQuant":"#888","PCA":"#c1700a","JointQK":"#328ac1","QPCA":"#1c5d2c","QPCA-EC":"#9b1c9b"}
    markers = {"TurboQuant":"s","PCA":"D","JointQK":"o","QPCA":"^","QPCA-EC":"P"}
    def xfor(m): return [ec_rate[b] for b in k_bits] if m == "QPCA-EC" else list(k_bits)
    for ax, met in zip(axes, ["top1","top5","k_mse","logit_err"]):
        for m in METHODS:
            ax.plot(xfor(m), [final[m][b][met] for b in k_bits], marker=markers[m],
                    color=colors[m], label=m, linewidth=1.8, markersize=8)
        t, dirn = titles[met]
        ax.set_title(f"{t}\n({dirn})", fontsize=11); ax.set_xlabel("bits / coord")
        if met in ("k_mse","logit_err"): ax.set_yscale("log")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "03_split_comparison.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
