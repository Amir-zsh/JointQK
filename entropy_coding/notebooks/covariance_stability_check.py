#!/usr/bin/env python3
"""
Check basis alignment of query/key second-order structure across examples.

For each example and each metric:
  - top-1 subspace overlap
  - top-8 subspace overlap
  - top-32 subspace overlap
  - full-basis diagonal alignment

The score is computed against a pooled reference basis.

Baseline:
  - random orthogonal transform of the pooled basis, sampled via QR from a
    Gaussian matrix, repeated several times to estimate mean/std.

Outputs:
  - terminal table
  - JSON summary
  - PT dump with raw tensors
  - PNG plot with per-example curves and random-orthogonal baselines

Usage:
  ./.venv-qnb/bin/python check_qk_basis_alignment.py \
      --idx 0 1 2 ... 26 \
      --baseline-samples 64
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _ensure_environment():
    needed = ("numpy", "torch", "matplotlib")
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"Missing deps: {missing}. Activate the right environment first.")


_ensure_environment()

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for cand in (Path.cwd().resolve(), Path.cwd().resolve().parent, SCRIPT_DIR, SCRIPT_DIR.parent):
    s = str(cand)
    if s not in sys.path:
        sys.path.insert(0, s)

import _bootstrap  # noqa: E402,F401
REPO = Path(_bootstrap.__file__).resolve().parent

FULL_DIR = "/mnt/c/JointQK_data/query_stats_longbench_under4k"
TOP_RS = (1, 8, 32)


def data_root():
    r = REPO / "notebooks" / "data" / FULL_DIR
    if not (r / "manifest.json").exists():
        sys.exit(f"Dataset not found at {r} (expected manifest.json).")
    return r


def load_manifest(root):
    return json.loads((root / "manifest.json").read_text())


def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))


def _sorted_eigh(M, eps=1e-12):
    """
    Symmetrize, eigendecompose, sort descending by eigenvalue.
    Returns:
      evals: [d]
      evecs: [d, d] with columns sorted by descending eigenvalue
    """
    S = _sym(M).double()
    evals, evecs = torch.linalg.eigh(S)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(eps)
    evecs = evecs[:, order]
    return evals, evecs


def compute_per_example_moments(art, centered=False):
    """
    Returns:
      q_mom: [L, Hq, d, d]
      k_mom: [L, Hkv, d, d]
      T: number of tokens
    """
    q_all = art["q_post"]  # [L, Hq, T, d]
    k_all = art["k_post"]  # [L, Hkv, T, d]
    T = int(art["prompt_length"])

    q = q_all[:, :, :T, :].double()
    k = k_all[:, :, :T, :].double()

    L, Hq, _, d = q.shape
    _, Hkv, _, _ = k.shape

    q_mom = torch.zeros(L, Hq, d, d, dtype=torch.float64)
    k_mom = torch.zeros(L, Hkv, d, d, dtype=torch.float64)

    if centered:
        for l in range(L):
            for h in range(Hq):
                x = q[l, h]
                mu = x.mean(dim=0, keepdim=True)
                xc = x - mu
                q_mom[l, h] = (xc.transpose(0, 1) @ xc) / max(1, T)
            for h in range(Hkv):
                x = k[l, h]
                mu = x.mean(dim=0, keepdim=True)
                xc = x - mu
                k_mom[l, h] = (xc.transpose(0, 1) @ xc) / max(1, T)
    else:
        q_mom = torch.einsum("lhtd,lhte->lhde", q, q) / T
        k_mom = torch.einsum("lhtd,lhte->lhde", k, k) / T

    return q_mom, k_mom, T


def pooled_reference(moms, token_counts):
    """
    Token-weighted pooled moment across examples.
    moms: list of tensors [L, H, d, d]
    token_counts: list[int]
    """
    total = float(sum(token_counts))
    ref = torch.zeros_like(moms[0], dtype=torch.float64)
    for M, T in zip(moms, token_counts):
        ref += M * (float(T) / total)
    return ref


def _projector_alignment(U_ref, U_ex, r):
        """
        Rank-r projector alignment in [0, 1]:
            trace(P_ref P_ex) / r, where P = U_r U_r^T
        """
        d = U_ref.shape[-1]
        r = int(min(max(1, r), d))
        U_ref_r = U_ref[:, :r]
        U_ex_r = U_ex[:, :r]
        G = U_ref_r.transpose(-1, -2) @ U_ex_r
        return float((G.square().sum() / r).item())


def _full_basis_diag_align(U_ref, U_ex):
    """
    Sign-invariant diagonal alignment of the full sorted eigenbases:
      mean_i |u_ref,i^T u_ex,i|
    """
    return float(torch.abs(torch.sum(U_ref * U_ex, dim=0)).mean().item())


def _weighted_projector_alignment(U_ref, evals_ref, U_ex, evals_ex, eps=1e-12):
        """
        Full-basis weighted projector alignment in [0, 1].
        Uses normalized eigenvalue weights from both matrices:
            score = sum_{i,j} w_ref[i] w_ex[j] (u_ref_i^T u_ex_j)^2
                            / sqrt(sum_i w_ref[i]^2 * sum_j w_ex[j]^2)
        """
        w_ref = evals_ref.clamp_min(eps)
        w_ex = evals_ex.clamp_min(eps)
        w_ref = w_ref / w_ref.sum()
        w_ex = w_ex / w_ex.sum()
        G = U_ref.transpose(-1, -2) @ U_ex
        num = (w_ref.unsqueeze(-1) * w_ex.unsqueeze(0) * G.square()).sum()
        den = (w_ref.square().sum() * w_ex.square().sum()).sqrt().clamp_min(eps)
        return float((num / den).item())


def metric_against_reference(M, R, r=None):
    """
    Compute alignment between moment M and pooled reference R.
    If r is None, compute full-basis diagonal alignment.
    Otherwise compute rank-r projector alignment.
    """
    evals_M, UM = _sorted_eigh(M)
    evals_R, UR = _sorted_eigh(R)
    if r is None:
        return _weighted_projector_alignment(UR, evals_R, UM, evals_M)
    return _projector_alignment(UR, UM, r)


def random_orthogonal(d, device=None, dtype=torch.float64):
    """
    Haar-like random orthogonal matrix via QR from Gaussian.
    """
    A = torch.randn(d, d, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # Fix sign ambiguity so Q is closer to a proper Haar sample.
    s = torch.sign(torch.diagonal(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q


def random_baseline_from_reference(R, r, n_samples=64, seed=0):
    """
    Empirical null for a batched reference tensor R of shape [L, H, d, d].
    For each [l, h] matrix:
      1) extract the pooled reference basis
      2) sample random orthogonal matrices via QR
      3) measure the same alignment metric
      4) average over samples, then over layers/heads
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    # R is batched: [L, H, d, d]
    if R.ndim == 2:
        mats = [R]
    elif R.ndim == 4:
        mats = [R[l, h] for l in range(R.shape[0]) for h in range(R.shape[1])]
    else:
        raise ValueError(f"Expected R to be [d,d] or [L,H,d,d], got shape {tuple(R.shape)}")

    per_mat_scores = []
    for Mref in mats:
        evals_ref, Uref = _sorted_eigh(Mref)
        d = Uref.shape[-1]

        vals = []
        for _ in range(n_samples):
            A = torch.randn(d, d, generator=gen, dtype=torch.float64)
            Q, Rq = torch.linalg.qr(A)

            s = torch.sign(torch.diagonal(Rq))
            s[s == 0] = 1.0
            Q = Q * s

            Urand = Q @ Uref
            if r is None:
                vals.append(_weighted_projector_alignment(Uref, evals_ref, Urand, evals_ref))
            else:
                vals.append(_projector_alignment(Uref, Urand, r))

        per_mat_scores.append(float(np.mean(vals)))

    arr = np.asarray(per_mat_scores, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
        "samples": per_mat_scores,
    }


def summarize_metric(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", required=True,
                    help="Example indices to evaluate, e.g. 0 1 2 ...")
    ap.add_argument("--centered", action="store_true",
                    help="Use centered covariance instead of raw second moment.")
    ap.add_argument("--baseline-samples", type=int, default=64,
                    help="Monte Carlo samples for the random orthogonal baseline.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for baseline sampling.")
    ap.add_argument("--out", type=str, default="qk_basis_alignment",
                    help="Output prefix for saved files.")
    args = ap.parse_args()

    out_dir = Path(os.environ.get("MOMENT_CACHE_DIR", "")) if os.environ.get("MOMENT_CACHE_DIR") else None
    if out_dir is None:
        out_dir = SCRIPT_DIR / "moment_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{args.out}_cache.pt"

    root = data_root()
    manifest = load_manifest(root)
    exs = manifest["examples"]
    nex = len(exs)

    for i in args.idx:
        if i < 0 or i >= nex:
            sys.exit(f"Index {i} out of range for dataset of size {nex}.")

    print(f"Dataset size: {nex}")
    print(f"Selected examples: {args.idx}")
    print(f"Mode: {'centered covariance' if args.centered else 'raw second moment'}")

    q_moms = []
    k_moms = []
    token_counts = []

    n_layers = n_q = n_kv = d = None

    if cache_path.exists():
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cache.get("indices") == args.idx and cache.get("centered") == args.centered:
                q_moms = cache["q_moms"]
                k_moms = cache["k_moms"]
                token_counts = cache["token_counts"]
                n_layers = cache["n_layers"]
                n_q = cache["n_q"]
                n_kv = cache["n_kv"]
                d = cache["d"]
                print(f"Cache hit: {cache_path.name} (reuse per-example moments)")
            else:
                print(f"Cache mismatch for {cache_path.name}; recomputing moments")
        except Exception as exc:
            print(f"Cache load failed for {cache_path.name}: {exc}. Recomputing moments.")
            try:
                cache_path.unlink()
            except Exception:
                pass
            
    if not q_moms:
        for rank, i in enumerate(args.idx, 1):
            art = torch.load(root / exs[i]["file"], map_location="cpu", weights_only=False)
            qM, kM, T = compute_per_example_moments(art, centered=args.centered)

            if n_layers is None:
                n_layers, n_q, d = qM.shape[0], qM.shape[1], qM.shape[2]
                n_kv = kM.shape[1]
                print(f"Shape summary: L={n_layers}, Hq={n_q}, Hkv={n_kv}, d={d}")

            q_moms.append(qM)
            k_moms.append(kM)
            token_counts.append(T)
            print(f"  [{rank:02d}/{len(args.idx):02d}] example {i}: T={T}")

        try:
            torch.save(
                {
                    "indices": args.idx,
                    "centered": args.centered,
                    "q_moms": q_moms,
                    "k_moms": k_moms,
                    "token_counts": token_counts,
                    "n_layers": n_layers,
                    "n_q": n_q,
                    "n_kv": n_kv,
                    "d": d,
                },
                cache_path,
            )
            print(f"Cached moments -> {cache_path.name}")
        except Exception as exc:
            print(f"Warning: failed to save cache at {cache_path} ({exc}); continuing without cache")

    q_ref = pooled_reference(q_moms, token_counts)
    k_ref = pooled_reference(k_moms, token_counts)

    # Baselines are constant across examples for a fixed metric, but we estimate
    # them separately for Q and K because their pooled bases differ.
    q_baseline = {}
    k_baseline = {}
    for r in TOP_RS:
        q_baseline[r] = random_baseline_from_reference(
            q_ref, r, n_samples=args.baseline_samples, seed=args.seed + 1000 + r
        )
        k_baseline[r] = random_baseline_from_reference(
            k_ref, r, n_samples=args.baseline_samples, seed=args.seed + 2000 + r
        )
    q_baseline["full"] = random_baseline_from_reference(
        q_ref, None, n_samples=args.baseline_samples, seed=args.seed + 3000
    )
    k_baseline["full"] = random_baseline_from_reference(
        k_ref, None, n_samples=args.baseline_samples, seed=args.seed + 4000
    )

    rows = []
    q_scores = {r: [] for r in TOP_RS}
    k_scores = {r: [] for r in TOP_RS}
    q_full = []
    k_full = []

    for ex_i, (qM, kM, T) in enumerate(zip(q_moms, k_moms, token_counts)):
        row = {
            "example_index": args.idx[ex_i],
            "tokens": int(T),
        }

        for r in TOP_RS:
            q_vals = []
            k_vals = []
            for l in range(n_layers):
                for h in range(n_q):
                    q_vals.append(metric_against_reference(qM[l, h], q_ref[l, h], r=r))
                for h in range(n_kv):
                    k_vals.append(metric_against_reference(kM[l, h], k_ref[l, h], r=r))
            row[f"q_top{r}"] = float(np.mean(q_vals))
            row[f"k_top{r}"] = float(np.mean(k_vals))
            q_scores[r].append(row[f"q_top{r}"])
            k_scores[r].append(row[f"k_top{r}"])

        q_diag_vals = []
        k_diag_vals = []
        for l in range(n_layers):
            for h in range(n_q):
                q_diag_vals.append(metric_against_reference(qM[l, h], q_ref[l, h], r=None))
            for h in range(n_kv):
                k_diag_vals.append(metric_against_reference(kM[l, h], k_ref[l, h], r=None))
        row["q_full"] = float(np.mean(q_diag_vals))
        row["k_full"] = float(np.mean(k_diag_vals))
        q_full.append(row["q_full"])
        k_full.append(row["k_full"])

        rows.append(row)

    header = (
        f"{'idx':>4} | {'T':>6} | {'q@1':>8} | {'k@1':>8} | "
        f"{'q@8':>8} | {'k@8':>8} | {'q@32':>8} | {'k@32':>8} | "
        f"{'q@all':>8} | {'k@all':>8}"
    )
    print("\nPer-example summary against pooled reference:")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['example_index']:>4} | {r['tokens']:>6} | "
            f"{r['q_top1']:>8.5f} | {r['k_top1']:>8.5f} | "
            f"{r['q_top8']:>8.5f} | {r['k_top8']:>8.5f} | "
            f"{r['q_top32']:>8.5f} | {r['k_top32']:>8.5f} | "
            f"{r['q_full']:>8.5f} | {r['k_full']:>8.5f}"
        )

    print("\nAggregate over examples:")
    print(f"  q_top1   : {summarize_metric(q_scores[1])}")
    print(f"  k_top1   : {summarize_metric(k_scores[1])}")
    print(f"  q_top8   : {summarize_metric(q_scores[8])}")
    print(f"  k_top8   : {summarize_metric(k_scores[8])}")
    print(f"  q_top32  : {summarize_metric(q_scores[32])}")
    print(f"  k_top32  : {summarize_metric(k_scores[32])}")
    print(f"  q_full   : {summarize_metric(q_full)}")
    print(f"  k_full   : {summarize_metric(k_full)}")

    # Plot
    x = np.asarray(args.idx, dtype=np.int32)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.ravel()

    plot_specs = [
        ("top-1 projector alignment", 1, "q_top1", "k_top1"),
        ("top-8 projector alignment", 8, "q_top8", "k_top8"),
        ("top-32 projector alignment", 32, "q_top32", "k_top32"),
        ("full-basis weighted projector alignment", None, "q_full", "k_full"),
    ]

    for ax, (title, r, q_key, k_key) in zip(axes, plot_specs):
        ax.plot(x, [row[q_key] for row in rows], marker="o", linewidth=1.8, label="Q")
        ax.plot(x, [row[k_key] for row in rows], marker="s", linewidth=1.8, label="K")

        if r is None:
            q_b = q_baseline["full"]
            k_b = k_baseline["full"]
            ax.axhline(q_b["mean"], linestyle="--", linewidth=1.5, label=f"Q random baseline ({q_b['mean']:.3f})")
            ax.axhline(k_b["mean"], linestyle=":", linewidth=1.5, label=f"K random baseline ({k_b['mean']:.3f})")
            ax.fill_between(
                [x.min(), x.max()],
                [q_b["mean"] - q_b["std"], q_b["mean"] - q_b["std"]],
                [q_b["mean"] + q_b["std"], q_b["mean"] + q_b["std"]],
                alpha=0.12,
            )
            ax.fill_between(
                [x.min(), x.max()],
                [k_b["mean"] - k_b["std"], k_b["mean"] - k_b["std"]],
                [k_b["mean"] + k_b["std"], k_b["mean"] + k_b["std"]],
                alpha=0.12,
            )
        else:
            q_b = q_baseline[r]
            k_b = k_baseline[r]
            ax.axhline(q_b["mean"], linestyle="--", linewidth=1.5, label=f"Q random baseline ({q_b['mean']:.3f})")
            ax.axhline(k_b["mean"], linestyle=":", linewidth=1.5, label=f"K random baseline ({k_b['mean']:.3f})")
            ax.fill_between(
                [x.min(), x.max()],
                [q_b["mean"] - q_b["std"], q_b["mean"] - q_b["std"]],
                [q_b["mean"] + q_b["std"], q_b["mean"] + q_b["std"]],
                alpha=0.12,
            )
            ax.fill_between(
                [x.min(), x.max()],
                [k_b["mean"] - k_b["std"], k_b["mean"] - k_b["std"]],
                [k_b["mean"] + k_b["std"], k_b["mean"] + k_b["std"]],
                alpha=0.12,
            )

        ax.set_title(title)
        ax.set_xlabel("example index")
        ax.set_ylabel("alignment score")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.0, top=1.02)
        ax.legend(fontsize=8)

    fig.tight_layout()

    png_path = out_dir / f"{args.out}.png"
    json_path = out_dir / f"{args.out}.json"
    pt_path = out_dir / f"{args.out}.pt"

    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot:")
    print(f"  {png_path}")

    with open(json_path, "w") as f:
        json.dump(
            {
                "mode": "centered_covariance" if args.centered else "raw_second_moment",
                "indices": args.idx,
                "rows": rows,
                "aggregate": {
                    "q_top1": summarize_metric(q_scores[1]),
                    "k_top1": summarize_metric(k_scores[1]),
                    "q_top8": summarize_metric(q_scores[8]),
                    "k_top8": summarize_metric(k_scores[8]),
                    "q_top32": summarize_metric(q_scores[32]),
                    "k_top32": summarize_metric(k_scores[32]),
                    "q_full": summarize_metric(q_full),
                    "k_full": summarize_metric(k_full),
                },
                "random_baseline": {
                    "q_top1": q_baseline[1],
                    "k_top1": k_baseline[1],
                    "q_top8": q_baseline[8],
                    "k_top8": k_baseline[8],
                    "q_top32": q_baseline[32],
                    "k_top32": k_baseline[32],
                    "q_full": q_baseline["full"],
                    "k_full": k_baseline["full"],
                },
            },
            f,
            indent=2,
        )

    torch.save(
        {
            "q_ref": q_ref,
            "k_ref": k_ref,
            "q_moms": q_moms,
            "k_moms": k_moms,
            "token_counts": token_counts,
            "indices": args.idx,
            "mode": "centered_covariance" if args.centered else "raw_second_moment",
            "rows": rows,
        },
        pt_path,
    )

    print("Saved:")
    print(f"  {json_path}")
    print(f"  {pt_path}")
    print("Done.")


if __name__ == "__main__":
    main()