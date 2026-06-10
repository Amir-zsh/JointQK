#!/usr/bin/env python3
"""K-cache quantization: TurboQuant vs JointQK vs QPCA vs PCA.

QPCA here uses the CENTERED basis and CENTERED input (mean subtracted, added back),
but the codebook std comes from the UNCENTERED second moment (Σ_K). On top of that,
coords 0 and 1 (the mean-direction coords) get their std multiplied by WIDEN_MULT,
a knob to over-widen those codebooks.

Figures go to ./fig_dump next to this script. Run from repo root or notebooks/.
    python run_pca.py
    python run_pca.py --bits 3 4
    python run_pca.py --widen 0 1 --widen-mult 2.0
"""

import argparse
import importlib.util
import json
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
    sys.exit(
        f"\nMissing {', '.join(missing)}. Activate the venv:\n"
        "    source ../.venv-qnb/bin/activate    # from notebooks/\n"
    )


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

SPECS = {
    "small": {"dirname": "query_stats_longbench_under4k_small",
              "repo_id": "azaad/longbench-qkv-qwen3-small", "approx_size": "~7 GB"},
    "full": {"dirname": "query_stats_longbench_under4k",
             "repo_id": "azaad/longbench-qkv-qwen3-full", "approx_size": "~57 GB"},
}


def load_data(dataset):
    from huggingface_hub import snapshot_download
    print(f"Repo root: {REPO}")
    data_dir = REPO / "notebooks" / "data"
    spec = SPECS[dataset]
    data_dir.mkdir(parents=True, exist_ok=True)
    data_root = data_dir / spec["dirname"]
    if not (data_root / "manifest.json").exists():
        print(f"Fetching {spec['repo_id']} ({spec['approx_size']})...")
        snapshot_download(repo_id=spec["repo_id"], repo_type="dataset", local_dir=str(data_root))
    manifest = json.loads((data_root / "manifest.json").read_text())
    print(f"Using '{dataset}' at {data_root}  |  {manifest['config']['model']}  |  "
          f"{len(manifest['examples'])} examples")
    return data_root, manifest


# ---------------------------------------------------------------------------
# Second moments.
# ---------------------------------------------------------------------------
def build_second_moments(data_root):
    pooled = torch.load(data_root / "pooled_stats.pt", map_location="cpu", weights_only=False)
    q_second = pooled["q_post"][2]
    k_second = pooled["k_post"][2]
    n_layers, n_q_heads, d_head, _ = q_second.shape
    _, n_kv_heads, _, _ = k_second.shape
    group_size = n_q_heads // n_kv_heads
    sigma_q = q_second.reshape(n_layers, n_kv_heads, group_size, d_head, d_head).sum(dim=2)
    sigma_k = k_second
    print(f"L={n_layers} H_q={n_q_heads} H_kv={n_kv_heads} d={d_head} (GQA group {group_size})")
    meta = dict(n_layers=n_layers, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads,
                d_head=d_head, group_size=group_size)
    return pooled, sigma_q, sigma_k, meta


# ---------------------------------------------------------------------------
# Bases.
# ---------------------------------------------------------------------------
def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))


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
    """QPCA: eigh of A = M_q^{1/2} Σ_K M_q^{1/2}. Pass sigma_k = Σ_K or Cov[K]."""
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
    score = k_diag * (q_diag) ** 0.55          # = λ, the QPCA code variance; M_q-weighting already in the basis
    return {"forward": forward, "inverse": inverse, "score": score, "std": k_diag.sqrt(), "sigma_q": sigma_q, "sigma_k": sigma_k}


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
# Compressors.
# ---------------------------------------------------------------------------
def build_compressors(basis, b, n_layers, n_kv_heads, std_override=None, max_coord_bits=8):
    """std_override: (L,H,d) tensor to use instead of basis['std'] (e.g. uncentered std)."""
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
    """Subtract mu before encoding, add back after."""
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
    """Lloyd-Max centroids for a UNIT-variance Laplacian (mean 0, scale b=1/√2).
    Closed-form iteration on the density, not data. Returns (2^bits,) centroids
    with unit variance, to be scaled by the coord std."""
    n = max(1, 2 ** int(bits))
    if n == 1:
        return torch.zeros(1)
    scale = 1.0 / (2 ** 0.5)                      # Laplacian scale for unit variance
    # sample the density densely and run weighted Lloyd-Max (deterministic, no leak)
    x = torch.linspace(-12, 12, 200001).double()
    p = torch.exp(-x.abs() / scale)
    p = p / p.sum()
    cent = torch.quantile(  # init at density quantiles
        torch.distributions.Laplace(0.0, scale).icdf(torch.linspace(0.001, 0.999, n).double()),
        torch.linspace(0, 1, n).double())
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
    cent = cent / cent.mul(cent).mul(  # renormalize to exactly unit variance
        torch.tensor(1.0)).double().pow(0).mean().sqrt()  # placeholder, see note
    # exact unit-variance renorm against the density:
    edges = (cent[1:] + cent[:-1]) / 2
    idx = torch.bucketize(x, edges)
    var = (p * (cent[idx]) ** 2).sum()
    cent = cent / var.clamp_min(1e-30).sqrt()
    return cent.float()

def build_qpca_centered_compressors(qpca_cen, qpca_unc, k_mean, b, n_layers, n_kv_heads,
                                    widen_coords=(0, 1), widen_mult=1.0):
    """Centered QPCA basis + centered input + uncentered (Σ_K) std.
    widen_coords: std × widen_mult. shrink_coords: score × shrink_factor (<1 makes
    the allocator give those coords fewer bits; freed bits water-fill elsewhere,
    so total budget is held fixed)."""
    fwd_c = qpca_cen["forward"]
    fwdt_c = fwd_c.transpose(-1, -2)
    sk = qpca_unc["sigma_k"].to(fwd_c.dtype)
    su = (fwdt_c @ regularize_batch(sk, EPS).to(fwd_c.dtype) @ fwd_c
          ).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30).sqrt()
    std = su.clone()
    for j in widen_coords:
        std[:, :, j] = std[:, :, j] * widen_mult

    score = qpca_cen["score"].clone()            # q_diag * k_diag
    allocs = allocate_bits(score, b, max_coord_bits=8)
    
    comps = {}
    for l in range(n_layers):
        for h in range(n_kv_heads):
            comps[(l, h)] = PerCoordCompressor(
                bits_per_coord=allocs[l, h], std_per_coord=std[l, h],
                forward_map=fwd_c[l, h], inverse_map=fwd_c[l, h].new_tensor(
                    qpca_cen["inverse"][l, h]) if False else qpca_cen["inverse"][l, h])
    LAPLACE_COORDS = [0]
    lap_cache = {}                          
    for l in range(n_layers):
        for h in range(n_kv_heads):
            comp = comps[(l, h)]
            for j in LAPLACE_COORDS:
                bj = int(comp.bits_int[j])
                if bj not in lap_cache:
                    lap_cache[bj] = laplacian_lloyd_max(bj)
                k = lap_cache[bj].numel()
                comp.codebooks_padded[j, :k] = lap_cache[bj].to(comp.codebooks_padded) * std[l, h, j]
    return {(l, h): CenteredRoundtrip(comps[(l, h)], k_mean[l, h])
        for l in range(n_layers) for h in range(n_kv_heads)}
    
# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------

def score_example(art, comps_by_method, k_bits_list):
    q_all, k_all = art["q_post"], art["k_post"]
    T = int(art["prompt_length"])
    L, H_kv = k_all.shape[0], k_all.shape[1]
    group_size = q_all.shape[1] // H_kv
    d = k_all.shape[-1]
    accums = {m: {b: {l: dict(mse_num=0.0, mse_den=0, logit_num=0.0, logit_den=0,
                              top1_num=0, top1_den=0, top5_num=0, top5_den=0)
                      for l in range(L)} for b in k_bits_list} for m in comps_by_method}
    device = next(iter(next(iter(next(iter(comps_by_method.values())).values())).values())).forward_map.device
    for l in range(L):
        for h in range(H_kv):
            k = k_all[l, h, :T, :].to(device).float()
            q = q_all[l, h * group_size:(h + 1) * group_size, :T, :].to(device).float().reshape(-1, d)
            qq = q.transpose(0, 1) @ q
            real_logits = q @ k.transpose(0, 1)
            real_top = real_logits.argmax(dim=-1)
            n_q = int(real_top.numel())
            k_top5 = min(5, T)
            for method, comps_bits in comps_by_method.items():
                for bits in k_bits_list:
                    comp = comps_bits[bits][(l, h)]
                    k_hat = comp.roundtrip(k).float()
                    err = k - k_hat
                    acc = accums[method][bits][l]
                    acc["mse_num"] += float(err.square().sum())
                    acc["mse_den"] += int(err.numel())
                    ee = err.transpose(0, 1) @ err
                    acc["logit_num"] += float((qq * ee).sum())
                    acc["logit_den"] += int(q.shape[0] * T * T)
                    approx = q @ k_hat.transpose(0, 1)
                    acc["top1_num"] += int((real_top == approx.argmax(dim=-1)).sum())
                    acc["top1_den"] += n_q
                    a5 = approx.topk(k_top5, dim=-1).indices
                    acc["top5_num"] += int((a5 == real_top.unsqueeze(-1)).any(dim=-1).sum())
                    acc["top5_den"] += n_q
    return accums


def merge_accums(a, b):
    for m in b:
        for bits in b[m]:
            for l in b[m][bits]:
                for k in b[m][bits][l]:
                    a[m][bits][l][k] = a[m][bits][l][k] + b[m][bits][l][k]
    return a


def finalize(accums, exclude_layer_0=True):
    out = {}
    for m, by_bits in accums.items():
        out[m] = {}
        for bits, by_layer in by_bits.items():
            layers = [l for l in by_layer if not (exclude_layer_0 and l == 0)]
            s = {k: sum(by_layer[l][k] for l in layers) for k in
                 ["mse_num", "mse_den", "logit_num", "logit_den",
                  "top1_num", "top1_den", "top5_num", "top5_den"]}
            out[m][bits] = dict(
                k_mse=s["mse_num"] / max(1, s["mse_den"]),
                logit_err=s["logit_num"] / max(1, s["logit_den"]),
                top1=s["top1_num"] / max(1, s["top1_den"]),
                top5=s["top5_num"] / max(1, s["top5_den"]))
    return out


def move_comps_to_device(comps_by_method, device):
    moved = set()
    for m in comps_by_method:
        for b in comps_by_method[m]:
            for k in comps_by_method[m][b]:
                comp = comps_by_method[m][b][k]
                if id(comp) not in moved:
                    comp.to(device)
                    moved.add(id(comp))


# ---------------------------------------------------------------------------
# Experiment.
# ---------------------------------------------------------------------------
def run(data_root, manifest, pooled, sigma_q, sigma_k, meta, k_bits, device,
        widen_coords, widen_mult):
    n_layers, n_kv_heads, d_head = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    k_mean = pooled["k_post"][0]
    k_cov = pooled["k_post"][1]

    turbo = build_turboquant(d_head, k_bits)
    jq = build_jointqk_basis(sigma_q, sigma_k)
    pca = build_pca_basis(sigma_k, sigma_q)
    qpca_unc = build_qpca_basis(sigma_q, sigma_k)
    qpca_unc["sigma_k"] = sigma_k          # stash Σ_K for the std projection
    qpca_cen = build_qpca_basis(sigma_q, k_cov)

    print(f"\nQPCA: centered basis+input, uncentered (Σ_K) std, "
          f"widen coords {widen_coords} x{widen_mult}")

    comps = {}
    comps["TurboQuant"] = {b: {(l, h): turbo[b] for l in range(n_layers)
                               for h in range(n_kv_heads)} for b in k_bits}
    comps["PCA"] = {b: build_compressors(pca, b, n_layers, n_kv_heads) for b in k_bits}
    comps["JointQK"] = {b: build_compressors(jq, b, n_layers, n_kv_heads) for b in k_bits}
    comps["QPCA"] = {b: build_qpca_centered_compressors(
        qpca_cen, qpca_unc, k_mean, b, n_layers, n_kv_heads,
        widen_coords=widen_coords, widen_mult=widen_mult) for b in k_bits}

    move_comps_to_device(comps, device)

    pooled_acc = None
    for i, entry in enumerate(manifest["examples"]):
        print(f"  [{i+1}/{len(manifest['examples'])}] {entry['file']}...", end=" ", flush=True)
        art = torch.load(data_root / entry["file"], map_location="cpu", weights_only=False)
        a = score_example(art, comps, k_bits)
        pooled_acc = a if pooled_acc is None else merge_accums(pooled_acc, a)
        print("done")
    final = finalize(pooled_acc, exclude_layer_0=True)

    print(f"\n{'method':<11} | {'b':<2} | {'top-1':>7} | {'top-5':>7} | {'k_mse':>11} | {'logit_err':>11}")
    print("-" * 64)
    for m in ["TurboQuant", "PCA", "JointQK", "QPCA"]:
        for b in k_bits:
            d = final[m][b]
            print(f"{m:<11} | {b:<2} | {d['top1']:>7.4f} | {d['top5']:>7.4f} | "
                  f"{d['k_mse']:>11.3e} | {d['logit_err']:>11.3e}")
        print()

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    titles = {"top1": ("top-1", "higher better"), "top5": ("top-5", "higher better"),
              "k_mse": ("k_mse", "lower better"), "logit_err": ("logit_err", "lower better")}
    colors = {"TurboQuant": "#888", "PCA": "#c1700a", "JointQK": "#328ac1", "QPCA": "#1c5d2c"}
    markers = {"TurboQuant": "s", "PCA": "D", "JointQK": "o", "QPCA": "^"}
    for ax, met in zip(axes, ["top1", "top5", "k_mse", "logit_err"]):
        for m in ["TurboQuant", "PCA", "JointQK", "QPCA"]:
            ax.plot(k_bits, [final[m][b][met] for b in k_bits], marker=markers[m],
                    color=colors[m], label=m, linewidth=1.8, markersize=8)
        t, dirn = titles[met]
        ax.set_title(f"{t}\n({dirn})", fontsize=11)
        ax.set_xlabel("bits"); ax.set_xticks(k_bits)
        if met in ("k_mse", "logit_err"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "02_k_metrics_comparison.png")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["small", "full"], default="small")
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--widen", type=int, nargs="+", default=[0, 1],
                    help="coords whose std is multiplied by --widen-mult")
    ap.add_argument("--widen-mult", type=float, default=1.0,
                    help="std multiplier on --widen coords (1.0 = plain Σ_K std)")
    args = ap.parse_args()
    k_bits = sorted(args.bits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Figures -> {FIG_DUMP}  |  device {device}")
    data_root, manifest = load_data(args.dataset)
    pooled, sigma_q, sigma_k, meta = build_second_moments(data_root)
    run(data_root, manifest, pooled, sigma_q, sigma_k, meta, k_bits, device,
        tuple(args.widen), args.widen_mult)
    print(f"\nDone. Figures in {FIG_DUMP}")


if __name__ == "__main__":
    main()