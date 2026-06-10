#!/usr/bin/env python3
"""
Plot cumulative eigenvalue spectra for:
  1) Keys
  2) Queries
  3) Q-PCA, i.e. A = _sym(sqrt_mq @ sk @ sqrt_mq)

By default, curves are computed for all (layer, head) blocks and the thick line
is the average curve. The individual block curves are drawn faintly.

Optionally restrict to a single block with:
  --layer L --head H
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
    print(f"[fig] wrote {path}")


for cand in (Path.cwd().resolve(), Path.cwd().resolve().parent, SCRIPT_DIR, SCRIPT_DIR.parent):
    s = str(cand)
    if s not in sys.path:
        sys.path.insert(0, s)

import _bootstrap  # noqa: E402,F401
REPO = Path(_bootstrap.__file__).resolve().parent


EPS = 1e-4
FULL_DIR = "/mnt/c/JointQK_data/query_stats_longbench_under4k"


def data_root():
    r = REPO / "notebooks" / "data" / FULL_DIR
    if not (r / "manifest.json").exists():
        sys.exit(f"full dataset not found at {r} (expected manifest.json).")
    return r


def load_manifest(root):
    return json.loads((root / "manifest.json").read_text())


def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))


MOMENTS_CACHE = SCRIPT_DIR / "moments_cache"


def _calib_cache_path(root, calib_idx):
    MOMENTS_CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{root.name}__idx_" + "_".join(str(i) for i in sorted(calib_idx))
    return MOMENTS_CACHE / f"{key}.pt"


def calib_moments(root, manifest, calib_idx):
    cache_path = _calib_cache_path(root, calib_idx)
    if cache_path.exists():
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        meta = c["meta"]
        print(
            f"calib moments: CACHE HIT {cache_path.name} "
            f"({len(calib_idx)} examples, {c['ntok']} tokens) | "
            f"L={meta['n_layers']} Hkv={meta['n_kv_heads']} d={meta['d_head']}"
        )
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
    sq2 = torch.zeros(L, Hq, d, d, dtype=torch.float64)
    sk2 = torch.zeros(L, Hkv, d, d, dtype=torch.float64)
    ntok = 0

    for i in calib_idx:
        art = torch.load(root / exs[i]["file"], map_location="cpu", weights_only=False)
        T = int(art["prompt_length"])
        q = art["q_post"][:, :, :T, :].double()
        k = art["k_post"][:, :, :T, :].double()
        sumq += q.sum(2)
        sumk += k.sum(2)
        sq2 += torch.einsum("lhtd,lhte->lhde", q, q)
        sk2 += torch.einsum("lhtd,lhte->lhde", k, k)
        ntok += T

    mq = sumq / ntok
    mk = sumk / ntok
    Eqq = sq2 / ntok
    Ekk = sk2 / ntok

    sigma_q = Eqq.reshape(L, Hkv, gs, d, d).sum(2)
    sigma_k = Ekk
    k_mean = mk
    k_cov = Ekk - torch.einsum("lhd,lhe->lhde", mk, mk)
    meta = dict(n_layers=L, n_q_heads=Hq, n_kv_heads=Hkv, d_head=d, group_size=gs)

    sigma_q, sigma_k, k_mean, k_cov = (
        sigma_q.float(),
        sigma_k.float(),
        k_mean.float(),
        k_cov.float(),
    )
    torch.save(
        {
            "sigma_q": sigma_q,
            "sigma_k": sigma_k,
            "k_mean": k_mean,
            "k_cov": k_cov,
            "meta": meta,
            "ntok": ntok,
            "calib_idx": sorted(calib_idx),
        },
        cache_path,
    )
    print(f"  cached moments -> {cache_path.name}")
    return sigma_q, sigma_k, k_mean, k_cov, meta


def cumulative_spectrum(mat):
    eigvals = torch.linalg.eigvalsh(_sym(mat).double())
    eigvals = torch.clamp(eigvals, min=0.0)
    eigvals = torch.sort(eigvals, descending=True).values
    denom = eigvals.sum().clamp_min(1e-30)
    return (eigvals.cumsum(0) / denom).cpu().numpy()


def spectra_for_tensor(tensor_4d, layer=None, head=None):
    L, H, _, _ = tensor_4d.shape
    blocks = [(layer, head)] if (layer is not None and head is not None) else [(l, h) for l in range(L) for h in range(H)]
    specs = [cumulative_spectrum(tensor_4d[l, h]) for l, h in blocks]
    maxlen = max(x.shape[0] for x in specs)
    return np.stack([np.pad(x, (0, maxlen - len(x)), mode="edge") for x in specs], axis=0)


def spectra_for_qpca(sigma_q, sigma_k, layer=None, head=None):
    L, H, _, _ = sigma_k.shape
    blocks = [(layer, head)] if (layer is not None and head is not None) else [(l, h) for l in range(L) for h in range(H)]

    specs = []
    for l, h in blocks:
        sq = _sym(sigma_q[l, h].double())
        sk = _sym(sigma_k[l, h].double())
        ev, U = torch.linalg.eigh(sq)
        ev = ev.clamp_min(1e-30)
        sqrt_mq = U @ torch.diag_embed(ev.sqrt()) @ U.transpose(-1, -2)
        A = _sym(sqrt_mq @ sk @ sqrt_mq)
        specs.append(cumulative_spectrum(A))

    maxlen = max(x.shape[0] for x in specs)
    return np.stack([np.pad(x, (0, maxlen - len(x)), mode="edge") for x in specs], axis=0)


def plot_panel(ax, curves, title):
    mean_curve = curves.mean(axis=0)
    lo = np.percentile(curves, 10, axis=0)
    hi = np.percentile(curves, 90, axis=0)

    x = np.arange(1, mean_curve.size + 1)

    # Background spectra
    for c in curves:
        ax.plot(
            x,
            c,
            color="0.75",      # light gray
            alpha=0.12,
            linewidth=0.8,
            zorder=1,
        )

    # Optional uncertainty envelope
    ax.fill_between(
        x,
        lo,
        hi,
        color="0.6",
        alpha=0.08,
        zorder=2,
    )

    # Mean spectrum
    ax.plot(
        x,
        mean_curve,
        color="black",
        linewidth=3.0,
        zorder=3,
        label="Mean",
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Cumulative mass")
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=None, help="optional single layer")
    ap.add_argument("--head", type=int, default=None, help="optional single head")
    ap.add_argument("--out", type=str, default="cumulative_spectra_3panel.png")
    args = ap.parse_args()

    if (args.layer is None) ^ (args.head is None):
        sys.exit("Provide both --layer and --head, or neither.")

    root = data_root()
    manifest = load_manifest(root)
    sigma_q, sigma_k, k_mean, k_cov, meta = calib_moments(root, manifest, args.calib_idx)

    curves_keys = spectra_for_tensor(sigma_k, layer=args.layer, head=args.head)
    curves_queries = spectra_for_tensor(sigma_q, layer=args.layer, head=args.head)
    curves_qpca = spectra_for_qpca(sigma_q, sigma_k, layer=args.layer, head=args.head)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    plot_panel(axes[0], curves_keys, "Keys")
    plot_panel(axes[1], curves_queries, "Queries")
    plot_panel(axes[2], curves_qpca, "Q-PCA")

    if args.layer is None:
        supt = f"Average over all layer/head blocks | calib={args.calib_idx}"
    else:
        supt = f"Layer {args.layer}, head {args.head} | calib={args.calib_idx}"

    fig.suptitle(supt, fontsize=12)
    fig.tight_layout()
    save_fig(fig, args.out)


if __name__ == "__main__":
    main()