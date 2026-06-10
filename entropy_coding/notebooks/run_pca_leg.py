#!/usr/bin/env python3
"""Compressing the K-cache: TurboQuant vs. JointQK vs. QPCA vs. PCA (+ V-side study).

Plain-script port of ``quantization_methods_pca.ipynb``. Same math, same data
(LongBench-E Q/K/V for Qwen3-8B), run top-to-bottom. Every figure that the
notebook used to ``plt.show()`` is written to a ``fig_dump/`` folder created
right next to this script.

Run from the repo root (or the notebooks/ dir) so that ``kvq/``, ``pipelines/``,
``vendor/`` and ``_bootstrap.py`` are importable:

    python quantization_methods_pca.py                  # 'small' bundle, b in {2,3,4}
    python quantization_methods_pca.py --dataset full   # finer-grained stats (~57 GB)
    python quantization_methods_pca.py --bits 3 4       # restrict bit widths

Sections mirror the notebook:
  3.    per-(layer, kv-head) second moments + eigenvalue spectra  -> fig 01
  4-6.5 build the four bases (TurboQuant / JointQK / QPCA / PCA)
  7.    bit allocation + Lloyd-Max codebooks
  8-9.  metrics + run the K experiment
  10.   K results table + plot                                    -> fig 02
  10.5  centered vs uncentered keys (PCA + QPCA)                  -> fig 03
  11.   V-vector analysis                                         -> fig 04
  12.   interpretation (printed)
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _ensure_environment():
    """A script can't `source activate` its own shell, so if we're started under
    the wrong interpreter (e.g. conda ``(base)``) and the notebook venv exists,
    re-exec under it. If no usable interpreter is found, exit with instructions
    instead of a bare ModuleNotFoundError.

    Override with QNB_PYTHON=/path/to/python; disable with QNB_NO_REEXEC=1.
    """
    needed = ("matplotlib", "numpy", "torch")
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    if not missing:
        return  # current interpreter already has the deps — nothing to do.

    already_reexeced = os.environ.get("_QNB_REEXECED") == "1"
    opted_out = os.environ.get("QNB_NO_REEXEC") == "1"
    if not already_reexeced and not opted_out:
        candidates = []
        if os.environ.get("QNB_PYTHON"):
            candidates.append(Path(os.environ["QNB_PYTHON"]))
        # Common venv spots, relative to the script and the repo root above it.
        for base in (SCRIPT_DIR, SCRIPT_DIR.parent):
            for name in (".venv-qnb", ".venv", "venv", "env"):
                candidates.append(base / name / "bin" / "python")
        current = Path(sys.executable).resolve()
        for py in candidates:
            try:
                resolved = py.resolve()
            except OSError:
                continue
            if resolved.exists() and resolved != current:
                os.environ["_QNB_REEXECED"] = "1"  # guard against re-exec loops
                print(
                    f"[env] {', '.join(missing)} missing in {current};\n"
                    f"      re-launching under {resolved}",
                    flush=True,
                )
                os.execv(str(resolved), [str(resolved), *sys.argv])

    sys.exit(
        f"\nMissing required packages: {', '.join(missing)}\n"
        f"Active interpreter: {sys.executable}\n\n"
        "Looks like the notebook venv isn't active (you're probably in conda "
        "'(base)'). Activate it first:\n"
        "    source .venv-qnb/bin/activate       # from the repo root\n"
        "    source ../.venv-qnb/bin/activate    # from notebooks/\n\n"
        "or recreate it (notebook §0):\n"
        "    uv venv --python 3.12 .venv-qnb && source .venv-qnb/bin/activate\n"
        "    uv pip install -r notebooks/requirements_quantization_notebook.txt\n\n"
        "Point this script at a specific interpreter with "
        "QNB_PYTHON=/path/to/python,\nor disable auto-switching with QNB_NO_REEXEC=1."
    )


_ensure_environment()

import matplotlib

matplotlib.use("Agg")  # headless: we only ever savefig(), never show()
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Figure dump: a folder sitting just below this script's location.
# (SCRIPT_DIR is defined at the top, alongside the environment guard.)
# ---------------------------------------------------------------------------
FIG_DUMP = SCRIPT_DIR / "fig_dump"
FIG_DUMP.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str) -> None:
    """Write a figure to fig_dump/ and free it."""
    path = FIG_DUMP / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] wrote {path}")


# ---------------------------------------------------------------------------
# §2. Make kvq/ and pipelines/ importable, then load the LongBench Q/K/V data.
# ---------------------------------------------------------------------------
# The notebook lives at <repo>/notebooks/...; the repo root is CWD (jupyter from
# repo root), CWD's parent (jupyter from notebooks/), this file's dir, or its
# parent. Add all candidates; whichever holds _bootstrap.py wins.
for cand in (
    Path.cwd().resolve(),
    Path.cwd().resolve().parent,
    SCRIPT_DIR,
    SCRIPT_DIR.parent,
):
    s = str(cand)
    if s not in sys.path:
        sys.path.insert(0, s)

import _bootstrap  # noqa: E402,F401  — also adds vendor/kvpress to sys.path

REPO = Path(_bootstrap.__file__).resolve().parent

from kvq.compression.lloyd_max import Stage1MSECompressor  # noqa: E402
from kvq.compression.per_coord import (  # noqa: E402
    PerCoordCompressor,
    unit_gaussian_centroids,
)
from kvq.compression.v_compressor_adapter import build_v_compressor  # noqa: E402
from pipelines.calibration.analyze_bases import (  # noqa: E402
    allocate_bits,
    jointqk_basis,
    regularize_batch,
)

EPS = 1e-4

SPECS = {
    "small": {
        "dirname": "query_stats_longbench_under4k_small",
        "repo_id": "azaad/longbench-qkv-qwen3-small",
        "approx_size": "~7 GB",
    },
    "full": {
        "dirname": "query_stats_longbench_under4k",
        "repo_id": "azaad/longbench-qkv-qwen3-full",
        "approx_size": "~57 GB",
    },
}


def load_data(dataset: str):
    """Download (if needed) and load the manifest for the chosen bundle."""
    from huggingface_hub import snapshot_download

    print(f"Repo root: {REPO}")
    data_dir = REPO / "notebooks" / "data"
    spec = SPECS[dataset]
    data_dir.mkdir(parents=True, exist_ok=True)
    data_root = data_dir / spec["dirname"]

    if not (data_root / "manifest.json").exists():
        print(f"Fetching {spec['repo_id']} ({spec['approx_size']})...")
        snapshot_download(
            repo_id=spec["repo_id"], repo_type="dataset", local_dir=str(data_root)
        )

    manifest = json.loads((data_root / "manifest.json").read_text())
    print(f"Using dataset '{dataset}' at {data_root}")
    print(f"Model:   {manifest['config']['model']}")
    print(f"Configs: {manifest['configs']}")
    print(f"Examples: {len(manifest['examples'])}")
    return data_root, manifest


# ---------------------------------------------------------------------------
# §3. Build per-(layer, kv-head) second moments + eigenvalue spectra (fig 01).
# ---------------------------------------------------------------------------
def build_second_moments(data_root: Path):
    pooled = torch.load(
        data_root / "pooled_stats.pt", map_location="cpu", weights_only=False
    )

    q_second = pooled["q_post"][2]  # (L, H_q, d, d), uncentered
    k_second = pooled["k_post"][2]  # (L, H_kv, d, d), uncentered

    n_layers, n_q_heads, d_head, _ = q_second.shape
    _, n_kv_heads, _, _ = k_second.shape
    group_size = n_q_heads // n_kv_heads

    # Pool Σ_Q across the GQA group (4 q-heads per kv-head for Qwen3-8B).
    sigma_q = q_second.reshape(
        n_layers, n_kv_heads, group_size, d_head, d_head
    ).sum(dim=2)  # (L, H_kv, d, d)
    sigma_k = k_second  # (L, H_kv, d, d)

    print(
        f"n_layers = {n_layers}, n_q_heads = {n_q_heads}, "
        f"n_kv_heads = {n_kv_heads}, head_dim = {d_head}"
    )
    print(f"GQA group size = {group_size}")
    print(f"Σ_Q shape = {tuple(sigma_q.shape)}    Σ_K shape = {tuple(sigma_k.shape)}")

    # Sanity: PSD per (L, h). Check the smallest eigenvalue on a sample
    # (skip layer 0 — attention-sink layer, different geometry).
    L_check, H_check = 1, 0
    e_q = torch.linalg.eigvalsh(sigma_q[L_check, H_check])
    e_k = torch.linalg.eigvalsh(sigma_k[L_check, H_check])
    print(f"\nAt (L={L_check}, h={H_check}):")
    print(
        f"  Σ_Q eigvals  min={e_q.min().item():.3e}  max={e_q.max().item():.3e}  "
        f"cond={(e_q.max() / e_q.clamp_min(1e-30).min()).item():.1e}"
    )
    print(
        f"  Σ_K eigvals  min={e_k.min().item():.3e}  max={e_k.max().item():.3e}  "
        f"cond={(e_k.max() / e_k.clamp_min(1e-30).min()).item():.1e}"
    )

    meta = dict(
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        group_size=group_size,
    )
    return pooled, sigma_q, sigma_k, meta


def plot_eigenvalue_spectra(sigma_q, sigma_k, d_head):
    """Standard + generalized eigenvalue spectra -> fig 01."""
    print("\nComputing all standard eigenvalues...")
    all_e_q, _ = torch.sort(torch.linalg.eigvalsh(sigma_q), dim=-1, descending=True)
    all_e_k, _ = torch.sort(torch.linalg.eigvalsh(sigma_k), dim=-1, descending=True)

    flat_e_q = all_e_q.reshape(-1, d_head).numpy()
    flat_e_k = all_e_k.reshape(-1, d_head).numpy()
    mean_e_q, std_e_q = flat_e_q.mean(axis=0), flat_e_q.std(axis=0)
    mean_e_k, std_e_k = flat_e_k.mean(axis=0), flat_e_k.std(axis=0)

    print("Computing generalized eigenvalues...")
    # Generalized: Σ_K x = λ Σ_Q x, via Σ_Q^{-1} Σ_K (ridge for stability).
    eps = 1e-6
    identity = torch.eye(d_head, dtype=sigma_q.dtype, device=sigma_q.device)
    sigma_q_reg = sigma_q + eps * identity
    gen_matrix = torch.linalg.solve(sigma_q_reg, sigma_k)
    e_gen = torch.linalg.eigvals(gen_matrix).real
    e_gen_sorted, _ = torch.sort(e_gen, dim=-1, descending=True)
    flat_e_gen = e_gen_sorted.reshape(-1, d_head).numpy()
    mean_e_gen, std_e_gen = flat_e_gen.mean(axis=0), flat_e_gen.std(axis=0)

    x_axis = np.arange(1, d_head + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(x_axis, mean_e_q, label=r"$\Sigma_Q$", color="royalblue", linewidth=2)
    ax1.fill_between(
        x_axis, mean_e_q - std_e_q, mean_e_q + std_e_q, color="royalblue", alpha=0.2
    )
    ax1.plot(x_axis, mean_e_k, label=r"$\Sigma_K$", color="crimson", linewidth=2)
    ax1.fill_between(
        x_axis, mean_e_k - std_e_k, mean_e_k + std_e_k, color="crimson", alpha=0.2
    )
    ax1.set_yscale("log")
    ax1.set_title("Standard Eigenvalue Spectrum")
    ax1.set_xlabel("Eigenvalue Index (Sorted Descending)")
    ax1.set_ylabel("Eigenvalue Magnitude (Log Scale)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.plot(
        x_axis,
        mean_e_gen,
        label=r"Gen Spectrum ($S_K x = \lambda S_Q x$)",
        color="darkviolet",
        linewidth=2,
    )
    ax2.fill_between(
        x_axis,
        mean_e_gen - std_e_gen,
        mean_e_gen + std_e_gen,
        color="darkviolet",
        alpha=0.2,
    )
    ax2.set_yscale("log")
    ax2.set_title("Generalized Eigenvalue Spectrum")
    ax2.set_xlabel("Eigenvalue Index (Sorted Descending)")
    ax2.set_ylabel(r"Eigenvalue Magnitude $\lambda$ (Log Scale)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    save_fig(fig, "01_eigenvalue_spectrum.png")


# ---------------------------------------------------------------------------
# §4. Method 1 — TurboQuant (data-oblivious random-Hadamard baseline).
# ---------------------------------------------------------------------------
class TurboQuantWrapper:
    """Adapt Stage1MSECompressor to a (T, d) -> (T, d) roundtrip interface.

    Per production v3, the SAME random rotation is shared across all
    (layer, kv-head): Algorithm 1 generates one Pi globally, not per head.
    """

    def __init__(self, head_dim, bits, seed=20260505, device="cpu"):
        self.tq = Stage1MSECompressor(
            head_dim=head_dim, bits=bits, seed=seed, device=device
        )
        self.head_dim = head_dim
        self.bits = bits
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
        out = self.tq.roundtrip(k.unsqueeze(0).unsqueeze(0))  # (1, 1, T, d)
        return out.squeeze(0).squeeze(0)


def build_turboquant(d_head, bits_list):
    turbo = {
        bits: TurboQuantWrapper(head_dim=d_head, bits=bits, seed=20260505)
        for bits in bits_list
    }
    for bits, comp in turbo.items():
        cents = comp.tq.centroids.cpu()
        print(
            f"TurboQuant b={bits}: {cents.numel()} centroids in "
            f"[{cents.min():.4f}, {cents.max():.4f}]  (Beta/Lloyd-Max for "
            f"d={d_head}, data-oblivious)"
        )
    Pi = turbo[bits_list[0]].tq.Pi
    err = (Pi @ Pi.T - torch.eye(d_head)).norm().item()
    print(f"||Pi @ Pi^T - I|| = {err:.3e}  (should be ~0; Pi is orthogonal)")
    print(f"Pi shape: {tuple(Pi.shape)}  (single rotation, NOT per (L, H))")
    return turbo


# ---------------------------------------------------------------------------
# §5. Method 2 — JointQK (production K-side basis).
# ---------------------------------------------------------------------------
def build_jointqk_basis(sigma_q, sigma_k, eps=EPS):
    """Orthogonal eigvecs of (Σ_Q Σ_K + Σ_K Σ_Q)/2.

    Bit-allocation score: diag(Rᵀ Σ_Q R) · diag(Rᵀ Σ_K R)  (logit-variance proxy).
    """
    R = jointqk_basis(sigma_q, sigma_k, eps=eps)  # (L, H, d, d)
    forward = R
    inverse = R.transpose(-1, -2)
    q_diag = (
        (inverse @ regularize_batch(sigma_q, eps) @ forward)
        .diagonal(dim1=-2, dim2=-1)
        .clamp_min(1e-30)
    )
    k_diag = (
        (inverse @ regularize_batch(sigma_k, eps) @ forward)
        .diagonal(dim1=-2, dim2=-1)
        .clamp_min(1e-30)
    )
    score = q_diag * k_diag
    std = k_diag.sqrt()
    return {"forward": forward, "inverse": inverse, "score": score, "std": std}


# ---------------------------------------------------------------------------
# §6. Method 3 — QPCA (closed-form optimum of the all-pairs inner-product loss).
# ---------------------------------------------------------------------------
def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))

def build_qpca_basis(sigma_q, sigma_k, eps=EPS):
    """Closed-form optimum of the all-pairs M_q-weighted key MSE.

    KLT of the WHITENED key tilde_k = M_q^{1/2} k:
        A = M_q^{1/2} Σ_K M_q^{1/2}  (symmetric PSD); eigh(A) = V Λ Vᵀ.
        forward = M_q^{1/2} V,  inverse = Vᵀ M_q^{-1/2}   (row convention: r = k @ forward).
    Then E[r rᵀ] = Λ exactly, so code-coord variance = Λ and the decoder is
    M_q-orthonormal (Gᵀ M_q G = I) — the two properties QPCA's optimality needs.
    Water-fill score = Λ alone (each code coord contributes equally to M_q-MSE).
    """
    #sq = regularize_batch(sigma_q, 0).to(torch.float64)
    #sk = regularize_batch(sigma_k, 0).to(torch.float64)
    sq = _sym(sigma_q.to(torch.float64))
    sk = _sym(sigma_k.to(torch.float64))
    # M_q^{±1/2} from a symmetric eigendecomposition of the SPD Σ_Q.
    ev, U = torch.linalg.eigh(sq)
    sqrt_mq = U @ torch.diag_embed(ev.sqrt()) @ U.transpose(-1, -2)
    isqrt_mq = U @ torch.diag_embed(ev.rsqrt()) @ U.transpose(-1, -2)

    # A = M_q^{1/2} Σ_K M_q^{1/2}, symmetrized; eigh then sort descending.
    A = sqrt_mq @ sk @ sqrt_mq
    A = _sym(A)
    lam, V = torch.linalg.eigh(A)  # ascending
    order = torch.argsort(lam, dim=-1, descending=True)
    lam = torch.gather(lam, -1, order).clamp_min(1e-30)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand(*V.shape[:-1], -1))

    forward = sqrt_mq @ V  # r = k @ forward
    inverse = V.transpose(-1, -2) @ isqrt_mq  # k_hat = r @ inverse
    
    # create q_diag and k_diag for bit allocation scores and std proxy
    fwdt = forward.transpose(-1, -2)
    q_diag = (fwdt @ sq @ forward).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    k_diag = (fwdt @ sk @ forward).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    score = q_diag * k_diag
    std = k_diag.sqrt()
    return {"forward": forward, "inverse": inverse, "score": score, "std": std}


# ---------------------------------------------------------------------------
# §6.5. Method 4 — PCA (plain key PCA == QPCA with M_q = I).
# ---------------------------------------------------------------------------
def build_pca_basis(sigma_k, sigma_q, eps=EPS):
    """Orthonormal eigvecs of the key 2nd-moment (or covariance) matrix.

    Pass the uncentered Σ_K for plain 'PCA', or Cov[K] for the centered variant.
    """
    sk = regularize_batch(sigma_k, eps).to(torch.float64)
    sq = regularize_batch(sigma_q, eps).to(torch.float64)
    lam, V = torch.linalg.eigh(sk)  # ascending
    order = torch.argsort(lam, dim=-1, descending=True)
    lam = torch.gather(lam, -1, order)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand(*V.shape[:-1], -1))
    forward = V
    inverse = V.transpose(-1, -2)
    lam = lam.float().clamp_min(1e-30)
    fwd = forward
    fwdt = forward.transpose(-1, -2)
    q_diag = (fwdt @ sq @ fwd).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    k_diag = (fwdt @ sk @ fwd).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    score = q_diag * k_diag
    return {"forward": forward, "inverse": inverse, "score": score, "std": k_diag.sqrt()}


def report_bases(turbo, jq, qpca, pca, d_head):
    """The orthogonality / inverse sanity checks the notebook prints per method."""
    print("\nJointQK basis:")
    print(f"  forward {tuple(jq['forward'].shape)}, inverse {tuple(jq['inverse'].shape)}")
    print(
        f"  score (q_diag · k_diag) at (L=1, H=0): top-5 = "
        f"{jq['score'][1, 0].topk(5).values.tolist()}"
    )
    err = (jq["forward"] @ jq["inverse"] - torch.eye(d_head)).norm(dim=(-2, -1)).max().item()
    print(f"  ||F @ G - I||_F max over (L, H) = {err:.3e}")

    print("\nQPCA basis:")
    print(f"  forward {tuple(qpca['forward'].shape)}, inverse {tuple(qpca['inverse'].shape)}")
    print(
        f"  eigenvalues Λ at (L=1, H=0): top-5 = "
        f"{qpca['score'][1, 0].topk(5).values.tolist()}"
    )
    err = (qpca["forward"] @ qpca["inverse"] - torch.eye(d_head)).norm(dim=(-2, -1)).max().item()
    print(f"  ||F @ G - I||_F max over (L, H) = {err:.3e}  (F and G are inverses by construction)")
    err_orth = (
        (qpca["forward"].transpose(-1, -2) @ qpca["forward"] - torch.eye(d_head))
        .norm(dim=(-2, -1))
        .max()
        .item()
    )
    print(f"  ||F^T F - I||_F max  = {err_orth:.3e}  (non-zero — F is M_q-orthonormal)")

    print("\nPCA basis:")
    print(f"  forward {tuple(pca['forward'].shape)}, inverse {tuple(pca['inverse'].shape)}")
    print(
        f"  eigenvalues of Σ_K at (L=1, H=0): top-5 = "
        f"{[round(x, 4) for x in pca['score'][1, 0].topk(5).values.tolist()]}"
    )
    err = (pca["forward"] @ pca["inverse"] - torch.eye(d_head)).norm(dim=(-2, -1)).max().item()
    err_orth = (
        (pca["forward"].transpose(-1, -2) @ pca["forward"] - torch.eye(d_head))
        .norm(dim=(-2, -1))
        .max()
        .item()
    )
    print(f"  ||F @ G - I||_F max = {err:.3e}")
    print(f"  ||F^T F - I||_F max = {err_orth:.3e}  (≈0 — PCA is Euclidean-orthonormal)")


# ---------------------------------------------------------------------------
# §7. Bit allocation + per-coord compressor construction.
# ---------------------------------------------------------------------------
def build_compressors(basis, k_bits, n_layers, n_kv_heads, max_coord_bits=8):
    """Per-(layer, kv-head) PerCoordCompressor (JointQK / QPCA / PCA)."""
    forward = basis["forward"]
    inverse = basis["inverse"]
    std = basis["std"]
    bit_allocs = allocate_bits(basis["score"], k_bits, max_coord_bits=max_coord_bits).cpu()

    comps = {}
    for l in range(n_layers):
        for h in range(n_kv_heads):
            comps[(l, h)] = PerCoordCompressor(
                bits_per_coord=bit_allocs[l, h],
                std_per_coord=std[l, h],
                forward_map=forward[l, h],
                inverse_map=inverse[l, h],
            )
    return comps


def print_bit_allocations(turbo_bits_demo, pca, jq, qpca, d_head):
    b_demo = 3
    print(f"\n{'TurboQuant':11s} bit allocation at b_avg={b_demo}:  uniform")
    print(f"   {b_demo} bits: {d_head:3d} coords  " + "#" * d_head)
    print(
        f"   (no water-fill — the shared Beta-optimal codebook has "
        f"{2 ** b_demo} centroids in [-1, 1])\n"
    )
    for name, basis in [("PCA", pca), ("JointQK", jq), ("QPCA", qpca)]:
        alloc = allocate_bits(basis["score"], b_demo)
        sample = alloc[1, 0]
        counts = torch.bincount(sample, minlength=9)
        print(f"{name:11s} bit allocation at (L=1, H=0), b_avg={b_demo}:")
        for b in range(9):
            c = int(counts[b].item())
            print(f"   {b} bits: {c:3d} coords  {'#' * c}")
        print()


# ---------------------------------------------------------------------------
# §8. Metrics.
# ---------------------------------------------------------------------------
def score_example(art, comps_by_method, k_bits_list):
    """k_mse / logit_err / top-1 / top-5 on one example, all methods × bit widths."""
    q_all = art["q_post"]  # (L, H_q, T, d) fp16
    k_all = art["k_post"]  # (L, H_kv, T, d) fp16
    T = int(art["prompt_length"])
    L, H_kv = k_all.shape[0], k_all.shape[1]
    group_size = q_all.shape[1] // H_kv
    d = k_all.shape[-1]

    accums = {
        m: {
            b: {
                l: {
                    "mse_num": 0.0,
                    "mse_den": 0,
                    "logit_num": 0.0,
                    "logit_den": 0,
                    "top1_num": 0,
                    "top1_den": 0,
                    "top5_num": 0,
                    "top5_den": 0,
                }
                for l in range(L)
            }
            for b in k_bits_list
        }
        for m in comps_by_method
    }

    device = next(
        iter(next(iter(next(iter(comps_by_method.values())).values())).values())
    ).forward_map.device
    for l in range(L):
        for h in range(H_kv):
            k = k_all[l, h, :T, :].to(device).float()  # (T, d)
            q = (
                q_all[l, h * group_size : (h + 1) * group_size, :T, :]
                .to(device)
                .float()
                .reshape(-1, d)
            )
            qq = q.transpose(0, 1) @ q  # (d, d)
            k_t = k.transpose(0, 1)  # (d, T)
            k_top5 = min(5, T)
            real_logits = q @ k_t  # (group*T, T)
            real_top = real_logits.argmax(dim=-1)
            n_q = int(real_top.numel())
            for method, comps_bits in comps_by_method.items():
                for bits in k_bits_list:
                    comp = comps_bits[bits][(l, h)]
                    k_hat = comp.roundtrip(k).float()
                    err = k - k_hat
                    acc = accums[method][bits][l]
                    acc["mse_num"] += float(err.square().sum().item())
                    acc["mse_den"] += int(err.numel())
                    ee = err.transpose(0, 1) @ err
                    acc["logit_num"] += float((qq * ee).sum().item())
                    acc["logit_den"] += int(q.shape[0] * T * T)
                    approx_logits = q @ k_hat.transpose(0, 1)
                    approx_top = approx_logits.argmax(dim=-1)
                    top1_match = int((real_top == approx_top).sum().item())
                    approx_top5 = approx_logits.topk(k_top5, dim=-1).indices
                    top5_match = int(
                        (approx_top5 == real_top.unsqueeze(-1)).any(dim=-1).sum().item()
                    )
                    acc["top1_num"] += top1_match
                    acc["top1_den"] += n_q
                    acc["top5_num"] += top5_match
                    acc["top5_den"] += n_q
    return accums


def merge_accums(a, b):
    for m in b:
        for bits in b[m]:
            for l in b[m][bits]:
                for k in b[m][bits][l]:
                    a[m][bits][l][k] = (
                        a.get(m, {}).get(bits, {}).get(l, {}).get(k, 0)
                        + b[m][bits][l][k]
                    )
    return a


def finalize(accums, exclude_layer_0=True):
    out = {}
    for m, by_bits in accums.items():
        out[m] = {}
        for bits, by_layer in by_bits.items():
            layers = [l for l in by_layer if not (exclude_layer_0 and l == 0)]
            sums = {
                k: sum(by_layer[l][k] for l in layers)
                for k in [
                    "mse_num",
                    "mse_den",
                    "logit_num",
                    "logit_den",
                    "top1_num",
                    "top1_den",
                    "top5_num",
                    "top5_den",
                ]
            }
            out[m][bits] = {
                "k_mse": sums["mse_num"] / max(1, sums["mse_den"]),
                "logit_err": sums["logit_num"] / max(1, sums["logit_den"]),
                "top1": sums["top1_num"] / max(1, sums["top1_den"]),
                "top5": sums["top5_num"] / max(1, sums["top5_den"]),
            }
    return out


# ---------------------------------------------------------------------------
# §9. Run the K experiment.
# ---------------------------------------------------------------------------
def move_comps_to_device(comps_by_method, device):
    """Move every (deduped) compressor instance onto `device`."""
    moved = set()
    for m in comps_by_method:
        for b in comps_by_method[m]:
            for k in comps_by_method[m][b]:
                comp = comps_by_method[m][b][k]
                if id(comp) not in moved:
                    comp.to(device)
                    moved.add(id(comp))


def run_k_experiment(data_root, manifest, turbo, pca, jq, qpca, meta, k_bits, device):
    n_layers, n_kv_heads = meta["n_layers"], meta["n_kv_heads"]

    print("\nBuilding compressors...")
    comps_by_method = {}
    comps_by_method["TurboQuant"] = {
        bits: {
            (l, h): turbo[bits]
            for l in range(n_layers)
            for h in range(n_kv_heads)
        }
        for bits in k_bits
    }
    for name, basis in [("PCA", pca), ("JointQK", jq), ("QPCA", qpca)]:
        comps_by_method[name] = {
            b: build_compressors(basis, b, n_layers, n_kv_heads) for b in k_bits
        }
    print(f"  TurboQuant: {len(k_bits)} shared wrappers")
    print(
        f"  PCA + JointQK + QPCA: 3 × {len(k_bits)} × "
        f"{n_layers * n_kv_heads} per-(L, H) compressors"
    )

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        move_comps_to_device(comps_by_method, device)

    k_pooled = None
    for i, entry in enumerate(manifest["examples"]):
        print(
            f"  [{i + 1}/{len(manifest['examples'])}] scoring {entry['file']}...",
            end=" ",
            flush=True,
        )
        art = torch.load(
            data_root / entry["file"], map_location="cpu", weights_only=False
        )
        a = score_example(art, comps_by_method, k_bits)
        k_pooled = a if k_pooled is None else merge_accums(k_pooled, a)
        print("done")

    print("\nFinalizing metrics (layer-0 excluded)...")
    final = finalize(k_pooled, exclude_layer_0=True)
    print("Done.")
    return final


# ---------------------------------------------------------------------------
# §10. K results table + plot (fig 02).
# ---------------------------------------------------------------------------
def report_k_results(final, k_bits):
    print(
        f"\n{'method':<11} | {'b':<2} | {'top-1':>7} | {'top-5':>7} | "
        f"{'k_mse':>11} | {'logit_err':>11}"
    )
    print("-" * 64)
    for m in ["TurboQuant", "PCA", "JointQK", "QPCA"]:
        for b in k_bits:
            d = final[m][b]
            print(
                f"{m:<11} | {b:<2} | {d['top1']:>7.4f} | {d['top5']:>7.4f} | "
                f"{d['k_mse']:>11.3e} | {d['logit_err']:>11.3e}"
            )
        print()

    print("\nWinner per (metric, b):")
    metrics_higher_better = {"top1", "top5"}
    for met in ["top1", "top5", "k_mse", "logit_err"]:
        cmp = max if met in metrics_higher_better else min
        print(f"  {met:<10}:", end="  ")
        for b in k_bits:
            winner = cmp(final, key=lambda m: final[m][b][met])
            v = final[winner][b][met]
            marker = f"{v:.4f}" if met in metrics_higher_better else f"{v:.3e}"
            print(f"b={b}: {winner:<11} ({marker})", end="   ")
        print()


def plot_k_results(final, k_bits):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    metric_titles = {
        "top1": ("top-1 retention", "higher is better"),
        "top5": ("top-5 retention", "higher is better"),
        "k_mse": ("k_mse  (unweighted key MSE)", "lower is better"),
        "logit_err": ("logit_err  (Q-weighted key MSE)", "lower is better"),
    }
    colors = {"TurboQuant": "#888", "PCA": "#c1700a", "JointQK": "#328ac1", "QPCA": "#1c5d2c"}
    markers = {"TurboQuant": "s", "PCA": "D", "JointQK": "o", "QPCA": "^"}

    for ax, met in zip(axes, ["top1", "top5", "k_mse", "logit_err"]):
        for m in ["TurboQuant", "PCA", "JointQK", "QPCA"]:
            y = [final[m][b][met] for b in k_bits]
            ax.plot(k_bits, y, marker=markers[m], color=colors[m], label=m, linewidth=1.8, markersize=8)
        title, direction = metric_titles[met]
        ax.set_title(f"{title}\n({direction})", fontsize=11)
        ax.set_xlabel("bits per coord")
        ax.set_xticks(k_bits)
        if met in ("k_mse", "logit_err"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    save_fig(fig, "02_k_metrics_comparison.png")

# ---------------------------------------------------------------------------
# §10.5. Centered vs uncentered keys (PCA + QPCA) (fig 03).
# ---------------------------------------------------------------------------
class CenteredRoundtrip:
    """Subtract a stored per-head mean before encoding, add it back after decoding."""

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
    
def build_centered_compressors(basis, mu, k_bits, n_layers, n_kv_heads, max_coord_bits=8):
    plain = build_compressors(basis, k_bits, n_layers, n_kv_heads, max_coord_bits=max_coord_bits)
    return {(l, h): CenteredRoundtrip(comp, mu[l, h]) for (l, h), comp in plain.items()}


def compute_mean_unit(k_mean, eps=1e-12):
    """Unit vectors along mean direction per (L, H)."""
    norms = k_mean.norm(dim=-1, keepdim=True).clamp_min(eps)
    return k_mean / norms


def build_subonly_compressors(basis, mu, k_bits, n_layers, n_kv_heads):
    """Pure mean subtraction, NO refit: uncentered basis + uncentered allocation
    + uncentered codebooks (built from Sigma_K via `basis`), just fed k - mu and
    mu added back. Contrast:
      QPCA-centered  : refit basis AND allocation on Cov   (lost top-1)
      QPCA-meanaware : uncentered basis, allocation refit on Cov   (lost top-1)
      QPCA-subonly   : refit NOTHING; only the input is zero-meaned   <-- this
    If subonly ~ uncentered, the damage was always the refit, never the subtraction."""
    inner = {b: build_compressors(basis, b, n_layers, n_kv_heads) for b in k_bits}
    return {
        b: {(l, h): CenteredRoundtrip(inner[b][(l, h)], mu[l, h])
            for l in range(n_layers) for h in range(n_kv_heads)}
        for b in k_bits
    }
    
def build_meanaware_basis(basis, sigma_q, k_cov, eps=EPS):
    """Uncentered forward/inverse maps UNCHANGED; only the per-coord variance the
    allocator and codebook see is mean-removed: diag(Fᵀ Cov[K] F) instead of
    diag(Fᵀ Σ_K F). So the basis the quantizer rotates into is identical to the
    uncentered method — we just stop spending bits/scale on the μ-contaminated
    variance. (q_diag stays uncentered: only K's mean cancels in softmax.)"""
    fwd = basis["forward"]
    fwdt = fwd.transpose(-1, -2)
    sq = regularize_batch(sigma_q, eps).to(fwd.dtype)
    cov = regularize_batch(k_cov, eps).to(fwd.dtype)
    q_diag = (fwdt @ sq @ fwd).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    cen_kdiag = (fwdt @ cov @ fwd).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    return {
        "forward": basis["forward"],
        "inverse": basis["inverse"],
        "score": q_diag * cen_kdiag,   # allocation now blind to μμᵀ
        "std": cen_kdiag.sqrt(),       # codebook scaled to the residual spread
    }


def build_meanaware_compressors(basis, sigma_q, k_cov, k_mean, k_bits, n_layers, n_kv_heads):
    """Uncentered basis + mean-aware allocation, wrapped to subtract/add-back μ
    (offset is argmax-free). build_centered_compressors does the μ wrapping; the
    only difference from the 'centered' variant is `ma` keeps the uncentered maps."""
    ma = build_meanaware_basis(basis, sigma_q, k_cov)
    return {
        b: build_centered_compressors(ma, k_mean, b, n_layers, n_kv_heads)
        for b in k_bits
    }


def run_centered_study(
    data_root, manifest, pooled, sigma_q, sigma_k, pca, qpca, meta, k_bits, device
):
    n_layers, n_kv_heads, d_head = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]

    k_mean = pooled["k_post"][0]  # (L, H_kv, d)
    k_cov = pooled["k_post"][1]   # (L, H_kv, d, d) — centered Cov[K]

    mu_k_s = k_mean[1, 0]
    mu_outer = float((mu_k_s * mu_k_s).sum())
    cov_tr = float(k_cov[1, 0].diagonal().sum())
    print(
        f"\nAt (L=1, H=0):  ‖μ_K‖² = {mu_outer:.4e}  tr(Cov[K]) = {cov_tr:.4e}  "
        f"-> {mu_outer / (mu_outer + cov_tr) * 100:.1f}% of uncentered energy on μ"
    )

    # Centered = refit basis on Cov[K] (the variant that lost top-1).
    qpca_cen = build_qpca_basis(sigma_q, k_cov)
    pca_cen = build_pca_basis(k_cov, sigma_q)

    # How many bits does going mean-aware actually relocate? (at the top width)
    print("Allocation shift, uncentered -> mean-aware (per head, b=%d):" % k_bits[-1])
    for name, basis in [("PCA", pca), ("QPCA", qpca)]:
        ma = build_meanaware_basis(basis, sigma_q, k_cov)
        a_u = allocate_bits(basis["score"], k_bits[-1]).cpu()
        a_m = allocate_bits(ma["score"], k_bits[-1]).cpu()
        moved = float((a_u - a_m).abs().sum()) / 2 / (n_layers * n_kv_heads)
        ndiff = int((a_u != a_m).sum()) / (n_layers * n_kv_heads)
        print(f"  {name:<4}: {ndiff:.1f} coords/head change, ~{moved:.1f} bits/head relocated")

    centered_comps = {
        "PCA-uncentered": {b: build_compressors(pca, b, n_layers, n_kv_heads) for b in k_bits},
        "PCA-centered": {b: build_centered_compressors(pca_cen, k_mean, b, n_layers, n_kv_heads) for b in k_bits},
        "PCA-meanaware": build_meanaware_compressors(pca, sigma_q, k_cov, k_mean, k_bits, n_layers, n_kv_heads),
        "QPCA-uncentered": {b: build_compressors(qpca, b, n_layers, n_kv_heads) for b in k_bits},
        "QPCA-centered": {b: build_centered_compressors(qpca_cen, k_mean, b, n_layers, n_kv_heads) for b in k_bits},
        "QPCA-subonly":    build_subonly_compressors(qpca, k_mean, k_bits, n_layers, n_kv_heads),
    }
    print(f"\nBuilt {len(centered_comps)} variants × {len(k_bits)} bits × {n_layers * n_kv_heads} (L,H).")

    move_comps_to_device(centered_comps, device)

    centered_pooled = None
    for i, entry in enumerate(manifest["examples"]):
        print(f"  [{i + 1}/{len(manifest['examples'])}] scoring {entry['file']}...", end=" ", flush=True)
        art = torch.load(data_root / entry["file"], map_location="cpu", weights_only=False)
        a = score_example(art, centered_comps, k_bits)
        centered_pooled = a if centered_pooled is None else merge_accums(centered_pooled, a)
        print("done")

    final = finalize(centered_pooled, exclude_layer_0=True)

    variant_order = [
        "PCA-uncentered", "PCA-centered", "PCA-meanaware",
        "QPCA-uncentered", "QPCA-centered", "QPCA-subonly",
    ]
    print(f"\n{'variant':<18} | {'b':<2} | {'top-1':>7} | {'top-5':>7} | {'k_mse':>11} | {'logit_err':>11}")
    print("-" * 72)
    for m in variant_order:
        for b in k_bits:
            d = final[m][b]
            print(f"{m:<18} | {b:<2} | {d['top1']:>7.4f} | {d['top5']:>7.4f} | {d['k_mse']:>11.3e} | {d['logit_err']:>11.3e}")
        print()

    print("Δ vs uncentered (per method family):")
    for name in variant_order:
        if name.endswith("-uncentered"):
            continue
        method = name.split("-")[0]              # "PCA" or "QPCA"
        base = f"{method}-uncentered"
        if base not in final:
            continue
        for b in k_bits:
            u = final[base][b]
            m = final[name][b]
            print(
                f"  {name:<16} b={b}:  top1 Δ={m['top1'] - u['top1']:+.4f}   "
                f"top5 Δ={m['top5'] - u['top5']:+.4f}   "
                f"k_mse {m['k_mse'] / max(u['k_mse'], 1e-30):.3f}x   "
                f"logit_err {m['logit_err'] / max(u['logit_err'], 1e-30):.3f}x"
            )    
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    metric_titles = {
        "top1": ("top-1 retention", "higher is better"),
        "top5": ("top-5 retention", "higher is better"),
        "k_mse": ("k_mse", "lower is better"),
        "logit_err": ("logit_err", "lower is better"),
    }
    cstyle = {
        "PCA-uncentered": dict(color="#c1700a", marker="D", linestyle="--", markerfacecolor="none", label="PCA · uncentered"),
        "PCA-centered": dict(color="#c1700a", marker="D", linestyle="-", label="PCA · centered"),
        "PCA-meanaware": dict(color="#c1700a", marker="D", linestyle="-.", label="PCA · meanaware"),
        "QPCA-uncentered": dict(color="#1c5d2c", marker="^", linestyle="--", markerfacecolor="none", label="QPCA · uncentered"),
        "QPCA-centered": dict(color="#1c5d2c", marker="^", linestyle="-", label="QPCA · centered"),
        "QPCA-subonly": dict(color="#1c5d2c", marker="^", linestyle="-.", label="QPCA · meanaware"),
    }
    for ax, met in zip(axes, ["top1", "top5", "k_mse", "logit_err"]):
        for m in variant_order:
            ax.plot(k_bits, [final[m][b][met] for b in k_bits], linewidth=1.8, markersize=8, **cstyle[m])
        title, direction = metric_titles[met]
        ax.set_title(f"{title}\n({direction})", fontsize=11)
        ax.set_xlabel("bits per coord")
        ax.set_xticks(k_bits)
        if met in ("k_mse", "logit_err"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_fig(fig, "03_k_meanaware.png")

from itertools import product


def build_param_compressors(fwd, inv, std, score, mu, b, n_layers, n_kv_heads, max_coord_bits=8):
    """One bit-width. mu=None -> uncentered input; else CenteredRoundtrip(k-mu)+mu."""
    allocs = allocate_bits(score, b, max_coord_bits=max_coord_bits).cpu()
    comps = {}
    for l in range(n_layers):
        for h in range(n_kv_heads):
            c = PerCoordCompressor(
                bits_per_coord=allocs[l, h],
                std_per_coord=std[l, h],
                forward_map=fwd[l, h],
                inverse_map=inv[l, h],
            )
            comps[(l, h)] = CenteredRoundtrip(c, mu[l, h]) if mu is not None else c
    return comps


def run_factorial_study(data_root, manifest, pooled, sigma_q, sigma_k, meta, k_bits, device):
    """Full 2^4 factorial over (Basis, Std, Score, Input) in {uncentered, centered}.
    Isolates which lever drives the QPCA top-1 collapse via main-effects analysis."""
    n_layers, n_kv_heads, d_head = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    k_mean = pooled["k_post"][0]
    k_cov = pooled["k_post"][1]
    b = k_bits[-1]  # factorial at the top width only (cleanest, tractable)
    print(f"\nFactorial at b={b}. Levels: u=uncentered(Σ_K), c=centered(Cov). "
          f"Label = B(asis) S(td) C(=score) I(nput).")

    # The two basis-map choices (forward/inverse only).
    qpca = build_qpca_basis(sigma_q, sigma_k)
    qpca_cen = build_qpca_basis(sigma_q, k_cov)
    maps = {"u": (qpca["forward"], qpca["inverse"]),
            "c": (qpca_cen["forward"], qpca_cen["inverse"])}

    def proj(fwd, S):
        fwdt = fwd.transpose(-1, -2)
        Sr = regularize_batch(S, EPS).to(fwd.dtype)
        return (fwdt @ Sr @ fwd).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)

    # Per-basis projected diagonals (q always uses Σ_Q — K's mean is the only one that cancels).
    diag = {}
    for bk in ("u", "c"):
        fwd, _ = maps[bk]
        diag[bk] = {"q": proj(fwd, sigma_q), "ksig": proj(fwd, sigma_k), "kcov": proj(fwd, k_cov)}

    comps_by_method, combos = {}, {}
    for (bk, sk, ck, ik) in product("uc", repeat=4):
        fwd, inv = maps[bk]
        std = (diag[bk]["ksig"] if sk == "u" else diag[bk]["kcov"]).sqrt()
        kdg = diag[bk]["ksig"] if ck == "u" else diag[bk]["kcov"]
        score = diag[bk]["q"] * kdg
        mu = None if ik == "u" else k_mean
        name = f"B{bk} S{sk} C{ck} I{ik}"
        combos[name] = (bk, sk, ck, ik)
        comps_by_method[name] = {b: build_param_compressors(fwd, inv, std, score, mu, b, n_layers, n_kv_heads)}

    comps_by_method["cLM-CEN widen01"] = build_compand_lm_compressors(
        qpca_cen, k_mean, b, n_layers, n_kv_heads, data_root, manifest,
        p=1.0, widen_basis=qpca, widen_coords=(0, 1))
    combos["cLM-CEN widen01"] = ("c","c","c","c")
    
    move_comps_to_device(comps_by_method, device)

    l,h=1,0
    a = comps_by_method["Bu Su Cu Iu"][b][(l,h)]; a=a.inner if isinstance(a,CenteredRoundtrip) else a
    c = comps_by_method["Bu Sc Cu Iu"][b][(l,h)]; c=c.inner if isinstance(c,CenteredRoundtrip) else c
    print("bits identical:", torch.equal(a.bits_int, c.bits_int))
    print("fwd identical:", torch.allclose(a.forward_map, c.forward_map))
    print("std diff coords:", (a.stds != c.stds).sum().item(), "of", a.stds.numel())
    print("std ratio range:", (c.stds/a.stds).min().item(), (c.stds/a.stds).max().item())
    l,h=1,0
    r = (c.stds.double()/a.stds.double() - 1).abs()
    for thr in [0.5, 0.1, 0.01, 0.001]:
        print(f"coords with |sc/su-1| > {thr}: {(r>thr).sum().item()}")    
        
    pooled_acc = None
    for i, entry in enumerate(manifest["examples"]):
        print(f"  [{i + 1}/{len(manifest['examples'])}] scoring {entry['file']}...", end=" ", flush=True)
        art = torch.load(data_root / entry["file"], map_location="cpu", weights_only=False)
        a = score_example(art, comps_by_method, [b])
        pooled_acc = a if pooled_acc is None else merge_accums(pooled_acc, a)
        print("done")
    final = finalize(pooled_acc, exclude_layer_0=True)

    anchors = {"B u S u C u I u": "uncentered", "B u S u C u I c": "subonly",
               "B u S c C c I c": "meanaware", "B c S c C c I c": "centered"}
    rows = sorted(final.keys(), key=lambda m: final[m][b]["top1"], reverse=True)
    print(f"\n{'label':<14} | {'top-1':>7} | {'top-5':>7} | {'k_mse':>10} | {'logit_err':>10} | note")
    print("-" * 78)
    for m in rows:
        d = final[m][b]
        print(f"{m:<14} | {d['top1']:>7.4f} | {d['top5']:>7.4f} | {d['k_mse']:>10.3e} | "
              f"{d['logit_err']:>10.3e} | {anchors.get(m, '')}")

    # Main effects: mean top-1 over the 8 cells where each lever=u vs =c.
    print("\nMain effect of each lever (mean top-1 over its 8 cells):")
    levers = ["Basis", "Std", "Score", "Input"]
    for li, lname in enumerate(levers):
        u = [final[m][b]["top1"] for m in final if combos[m][li] == "u"]
        c = [final[m][b]["top1"] for m in final if combos[m][li] == "c"]
        um, cm = sum(u) / len(u), sum(c) / len(c)
        print(f"  {lname:<6}: u={um:.4f}  c={cm:.4f}  Δ(u-c)={um - cm:+.4f}")
    print("  (large +Δ => that lever being CENTERED is what hurts top-1)")


def plot_coeff_grid(data_root, manifest, pooled, sigma_q, qpca, meta, k_bits, device,
                    l=1, h=0, coords=(0, 1, 5, 20, 60), b=None):
    import numpy as np
    n_layers, n_kv_heads = meta["n_layers"], meta["n_kv_heads"]
    b = b or k_bits[-1]
    k_mean, k_cov, sigma_k = pooled["k_post"][0], pooled["k_post"][1], pooled["k_post"][2]
    mu = k_mean[l, h].double()
    qpca_cen = build_qpca_basis(sigma_q, k_cov)

    def std_basis(basis, S):
        fwd = basis["forward"]; fwdt = fwd.transpose(-1, -2)
        Sr = regularize_batch(S, EPS).to(fwd.dtype)
        sq = regularize_batch(sigma_q, EPS).to(fwd.dtype)
        kd = (fwdt @ Sr @ fwd).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        qd = (fwdt @ sq @ fwd).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        return {"forward": basis["forward"], "inverse": basis["inverse"],
                "std": kd.sqrt(), "score": qd * kd}

    # (basis_label, basis, center_input?)  x  (std_label, S_for_std)
    base_cfgs = [
        ("UB·UI", qpca,     False),
        ("UB·CI", qpca,     True),
        ("CB·UI", qpca_cen, False),
        ("CB·CI", qpca_cen, True),
    ]
    std_cfgs = [("Su", sigma_k), ("Sc", k_cov)]

    rows = []
    for blabel, basis, cen in base_cfgs:
        F = basis["forward"][l, h].double()
        # coeffs (depend only on basis + input, not std)
        R = []
        for entry in manifest["examples"]:
            art = torch.load(data_root / entry["file"], map_location="cpu", weights_only=False)
            T = int(art["prompt_length"]); k = art["k_post"][l, h, :T].double()
            x = (k - mu) if cen else k
            R.append((x @ F).numpy())
        R = np.concatenate(R)
        for slabel, S in std_cfgs:
            comp = build_compressors(std_basis(basis, S), b, n_layers, n_kv_heads)[(l, h)]
            rows.append((f"{blabel} / std={slabel}", R, comp))

    # emp-LM row: centered basis, centered input, fitted books
    Fc = qpca_cen["forward"][l, h].double()
    al = allocate_bits(qpca_cen["score"], b).cpu()
    Rc = []
    for entry in manifest["examples"]:
        art = torch.load(data_root/entry["file"], map_location="cpu", weights_only=False)
        T = int(art["prompt_length"]); k = art["k_post"][l,h,:T].double()
        Rc.append(((k - mu) @ Fc).numpy())
    Rc = np.concatenate(Rc)
    fitted = {j: empirical_lloyd_max(torch.tensor(Rc[:, j]), int(al[l,h,j])).tolist() for j in coords}
    rows.append(("emp-LM CEN", Rc, None))   # comp=None -> use fitted dict below
    # in the plotting loop, when comp is None pull centroids from `fitted[j]`

    for p in (1.0, 2.0, 3.0):
        fa = {j: _expand(empirical_lloyd_max(_compand(torch.tensor(Rc[:,j]), p),
                          int(al[l,h,j])), p).tolist() for j in coords}
        rows.append((f"cLM-CEN p={p}", Rc, ("fitted", fa)))        
    
    fig, axes = plt.subplots(len(rows), len(coords), figsize=(3.2 * len(coords), 2.4 * len(rows)))
    for ri, (label, R, comp) in enumerate(rows):
        for ci, j in enumerate(coords):
            ax = axes[ri, ci]
            ax.hist(R[:, j], bins=80, density=True, color="#1c5d2c", alpha=0.5)
            if isinstance(comp, tuple) and comp[0] == "fitted":
                cents = comp[1][j]
            elif comp is None:
                cents = fitted[j]
            else:
                cb = comp.codebooks_padded[j]; cents = cb[cb < float('inf')].tolist()
            for cv in cents:
                ax.axvline(cv, color="k", lw=0.3, alpha=0.5)
            if isinstance(comp, tuple) and comp[0] == "fitted":
                ax.set_title(f"c{j} wLM b={int(al[l, h, j])}", fontsize=7)
            elif comp is None:
                ax.set_title(f"c{j} emp-LM b={int(al[l, h, j])}", fontsize=7)
            else:
                ax.set_title(f"c{j} std={comp.stds[j]:.1f} b={int(comp.bits_int[j])}", fontsize=7)           
            ax.tick_params(labelsize=6)
        axes[ri, 0].set_ylabel(label, fontsize=8)
    fig.suptitle(f"coeffs (rows=basis·input × std) + codebook  (L={l},H={h},b={b})", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "07_coeff_grid.png")

def empirical_lloyd_max(samples, bits, iters=15):
    s = samples.flatten().double().sort().values
    n = max(1, 2**int(bits))
    if n == 1: return torch.zeros(1)
    cent = torch.quantile(s, torch.linspace(0,1,n+2)[1:-1].double())
    for _ in range(iters):
        idx = torch.bucketize(s, (cent[1:]+cent[:-1])/2)
        sums = torch.zeros(n, dtype=s.dtype).scatter_add_(0, idx, s)
        cnts = torch.zeros(n, dtype=s.dtype).scatter_add_(0, idx, torch.ones_like(s))
        new = torch.where(cnts>0, sums/cnts.clamp_min(1), cent)
        if (new-cent).abs().max() < 1e-6*(s.std()+1e-12): cent=new; break
        cent = new
    return cent.float()

def weighted_lloyd_max(samples, bits, alpha=0.0, iters=15):
    """Lloyd-Max minimizing E[|r|^{2α}(r-r̂)²]. α=0 = plain MSE (emp-LM).
    α>0 puts more levels in the tails (magnitude-weighted)."""
    s = samples.flatten().double().sort().values
    n = max(1, 2 ** int(bits))
    if n == 1:
        return torch.zeros(1)
    w = s.abs().clamp_min(1e-12).pow(2 * alpha)        # per-sample weight
    cent = torch.quantile(s, torch.linspace(0, 1, n + 2)[1:-1].double())
    for _ in range(iters):
        idx = torch.bucketize(s, (cent[1:] + cent[:-1]) / 2)
        wsum = torch.zeros(n, dtype=s.dtype).scatter_add_(0, idx, w)
        wxsum = torch.zeros(n, dtype=s.dtype).scatter_add_(0, idx, w * s)
        new = torch.where(wsum > 0, wxsum / wsum.clamp_min(1e-30), cent)
        if (new - cent).abs().max() < 1e-6 * (s.std() + 1e-12):
            cent = new; break
        cent = new
    return cent.float()

def build_wlm_compressors(basis, mu, b, n_layers, n_kv_heads,
                          data_root, manifest, alpha=0.0):
    F = basis["forward"]
    al = allocate_bits(basis["score"], b).cpu()
    samples = {(l, h): [] for l in range(n_layers) for h in range(n_kv_heads)}
    for entry in manifest["examples"]:          # LEAKED: fit == score set
        art = torch.load(data_root/entry["file"], map_location="cpu", weights_only=False)
        T = int(art["prompt_length"])
        for l in range(n_layers):
            for h in range(n_kv_heads):
                k = art["k_post"][l,h,:T].double()
                samples[(l,h)].append((k - mu[l,h].double()) @ F[l,h].double())
    res = {b: {}}
    for l in range(n_layers):
        for h in range(n_kv_heads):
            R = torch.cat(samples[(l,h)], 0)
            books = [weighted_lloyd_max(R[:,j], int(al[l,h,j]), alpha) for j in range(F.shape[-1])]
            res[b][(l,h)] = EmpiricalLMRoundtrip(F[l,h], basis["inverse"][l,h], books, mu[l,h])
    return res

class EmpiricalLMRoundtrip:
    """Per-coord codebook fit to the empirical centered-code distribution.
    Transform (centered input) -> per-coord nearest-centroid -> inverse -> +mu.
    codebooks: list of d tensors (variable length 2^bits[j])."""
    def __init__(self, fwd, inv, codebooks, mu):
        self.fwd = fwd.float(); self.inv = inv.float()
        self.mu = mu.reshape(1, -1).float()
        self.d = fwd.shape[0]
        K = max(cb.numel() for cb in codebooks)
        self.cb = torch.full((self.d, K), float("inf"))
        for j, cb in enumerate(codebooks):
            self.cb[j, :cb.numel()] = cb
        self.forward_map = self.fwd

    def to(self, dev):
        self.fwd = self.fwd.to(dev); self.inv = self.inv.to(dev)
        self.mu = self.mu.to(dev); self.cb = self.cb.to(dev)
        self.forward_map = self.fwd
        return self

    def roundtrip(self, k):
        r = (k - self.mu) @ self.fwd                      # (T, d) centered codes
        idx = (r.unsqueeze(-1) - self.cb.unsqueeze(0)).abs().argmin(-1)  # (T, d)
        r_hat = torch.gather(self.cb.unsqueeze(0).expand(r.shape[0], -1, -1), 2,
                             idx.unsqueeze(-1)).squeeze(-1)
        return r_hat @ self.inv + self.mu


def empirical_lloyd_max(samples, bits, iters=15):
    """Plain MSE Lloyd-Max fit to empirical 1D samples (vectorized)."""
    s = samples.flatten().double().sort().values
    n = max(1, 2 ** int(bits))
    if n == 1:
        return torch.zeros(1)
    cent = torch.quantile(s, torch.linspace(0, 1, n + 2)[1:-1].double())
    for _ in range(iters):
        idx = torch.bucketize(s, (cent[1:] + cent[:-1]) / 2)
        wsum = torch.zeros(n, dtype=s.dtype).scatter_add_(0, idx, torch.ones_like(s))
        xsum = torch.zeros(n, dtype=s.dtype).scatter_add_(0, idx, s)
        new = torch.where(wsum > 0, xsum / wsum.clamp_min(1e-30), cent)
        if (new - cent).abs().max() < 1e-6 * (s.std() + 1e-12):
            cent = new; break
        cent = new
    return cent.float()


def _compand(x, p):    # finer tails for p>1
    return x.sign() * x.abs().clamp_min(1e-12).pow(p)
def _expand(y, p):
    return y.sign() * y.abs().clamp_min(1e-12).pow(1.0 / p)


class CompandLMRoundtrip:
    """Warp centered codes by g(r)=sign(r)|r|^p, quantize warped coeffs with an
    empirical-LM codebook fit IN WARPED SPACE, unwarp on decode. p>1 => more
    resolution at large |r| (the warp packs uniform-ish LM bins into stretched tails)."""
    def __init__(self, fwd, inv, books_warped, mu, p):
        self.fwd = fwd.float(); self.inv = inv.float()
        self.mu = mu.reshape(1, -1).float(); self.p = float(p)
        self.d = fwd.shape[0]
        K = max(cb.numel() for cb in books_warped)
        self.cb = torch.full((self.d, K), float("inf"))
        for j, cb in enumerate(books_warped):
            self.cb[j, :cb.numel()] = cb
        self.forward_map = self.fwd

    def to(self, dev):
        self.fwd = self.fwd.to(dev); self.inv = self.inv.to(dev)
        self.mu = self.mu.to(dev); self.cb = self.cb.to(dev)
        self.forward_map = self.fwd
        return self

    def roundtrip(self, k):
        r = (k - self.mu) @ self.fwd            # centered codes
        c = _compand(r, self.p)                 # warp (finer tails)
        idx = (c.unsqueeze(-1) - self.cb.unsqueeze(0)).abs().argmin(-1)
        c_hat = torch.gather(self.cb.unsqueeze(0).expand(c.shape[0], -1, -1), 2,
                             idx.unsqueeze(-1)).squeeze(-1)
        r_hat = _expand(c_hat, self.p)          # unwarp
        return r_hat @ self.inv + self.mu


def build_compand_lm_compressors(basis, mu, b, n_layers, n_kv_heads,
                                 data_root, manifest, p=1.0,
                                 widen_basis=None, widen_coords=(0, 1)):  # <- new
    F = basis["forward"]
    al = allocate_bits(basis["score"], b).cpu()
    samples = {(l, h): [] for l in range(n_layers) for h in range(n_kv_heads)}
    for entry in manifest["examples"]:
        art = torch.load(data_root/entry["file"], map_location="cpu", weights_only=False)
        T = int(art["prompt_length"])
        for l in range(n_layers):
            for h in range(n_kv_heads):
                k = art["k_post"][l,h,:T].double()
                samples[(l,h)].append((k - mu[l,h].double()) @ F[l,h].double())
    res = {b: {}}
    for l in range(n_layers):
        for h in range(n_kv_heads):
            R = torch.cat(samples[(l,h)], 0)
            books = [empirical_lloyd_max(_compand(R[:,j], p), int(al[l,h,j]))
                     for j in range(F.shape[-1])]
            # widen ONLY the mean-direction coords to Σ_K (uncentered) scale
            if widen_basis is not None:
                su = widen_basis["std"][l, h]; sc = basis["std"][l, h]
                for j in widen_coords:
                    books[j] = books[j] * float(su[j] / sc[j].clamp_min(1e-12)) * 3
            res[b][(l,h)] = CompandLMRoundtrip(F[l,h], basis["inverse"][l,h], books, mu[l,h], p)
    return res
        
# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["small", "full"], default="small",
        help="LongBench Q/K/V bundle (default: small, ~7 GB).",
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=[2, 3, 4],
        help="Bits per coordinate to evaluate (default: 2 3 4).",
    )
    parser.add_argument(
        "--minimax", dest="minimax", action="store_true", default=True,
        help="Run the §10.6 minimax / peak-fill study (default: on).",
    )
    parser.add_argument(
        "--no-minimax", dest="minimax", action="store_false",
        help="Skip the §10.6 minimax / peak-fill study.",
    )
    parser.add_argument(
        "--centered", action="store_true",
        help="Also run the §10.5 centered-vs-uncentered study (default: off).",
    )
    parser.add_argument(
        "--v", dest="run_v", action="store_true",
        help="Also run the §11 V-vector analysis (default: off).",
    )
    args = parser.parse_args()
    k_bits = sorted(args.bits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Figures will be written to: {FIG_DUMP}")
    print(f"CUDA available? {torch.cuda.is_available()}  ->  using {device}\n")

    # §2 load.
    data_root, manifest = load_data(args.dataset)

    # §3 second moments + spectra.
    print("\n=== §3  Second moments ===")
    pooled, sigma_q, sigma_k, meta = build_second_moments(data_root)
    plot_eigenvalue_spectra(sigma_q, sigma_k, meta["d_head"])

    # §4-6.5 bases.
    print("\n=== §4-6.5  Building bases ===")
    turbo = build_turboquant(meta["d_head"], k_bits)
    jq = build_jointqk_basis(sigma_q, sigma_k)
    qpca = build_qpca_basis(sigma_q, sigma_k)
    pca = build_pca_basis(sigma_k, sigma_q)
    report_bases(turbo, jq, qpca, pca, meta["d_head"])

    k_cov     = pooled["k_post"][1]   # centered covariance
    sigma_k_2 = pooled["k_post"][2]   # uncentered second moment
    k_mean    = pooled["k_post"][0]

    err = (sigma_k_2 - (k_cov + k_mean.unsqueeze(-1) @ k_mean.unsqueeze(-2))).abs().sum()
    print(f"Σ_K - (Cov + μμᵀ) error: {err:.3e}  (should be ~0)")

    l, h = 1, 0
    F  = qpca["forward"][l, h].double()
    su = (F.T @ regularize_batch(sigma_k_2, EPS)[l,h].double() @ F).diag().clamp_min(1e-30).sqrt()  # Σ_K
    sc = (F.T @ regularize_batch(k_cov,     EPS)[l,h].double() @ F).diag().clamp_min(1e-30).sqrt()  # Cov

    print("median |sc/su - 1| =", ((sc/su - 1).abs().median()).item())
    print("max    |sc/su - 1| =", ((sc/su - 1).abs().max()).item())
    print("tr(Cov)=", k_cov[l,h].diagonal().sum().item(),
        " tr(Σ_K)=", sigma_k_2[l,h].diagonal().sum().item(),
        " ‖μ‖²=", (k_mean[l,h]@k_mean[l,h]).item())
    print("worst-5 coords |sc/su-1|:", (sc/su-1).abs().topk(5).values.tolist())
    print("  their su:", su[(sc/su-1).abs().topk(5).indices].tolist())
    print("  their sc:", sc[(sc/su-1).abs().topk(5).indices].tolist())
    l,h=1,0
    art=torch.load(data_root/manifest["examples"][0]["file"],map_location="cpu",weights_only=False)
    T=int(art["prompt_length"]); k=art["k_post"][l,h,:T].double()
    r0=(k@F)[:,0]                      # coord-0 code, uncentered input
    print("r0 range:", r0.min().item(), r0.max().item())
    print("Su cb0 reach ±", 3*su[0].item(), " Sc cb0 reach ±", 3*sc[0].item())    
    # §7 bit allocation preview.
    print("\n=== §7  Bit allocation ===")
    print_bit_allocations(k_bits, pca, jq, qpca, meta["d_head"])

    # §10.5 centered vs uncentered (opt-in).
    if args.centered:
        print("\n=== §10.5  Factorial: which centering lever hurts? ===")
        plot_coeff_grid(data_root, manifest, pooled, sigma_q, qpca, meta, k_bits, device)        
        run_factorial_study(data_root, manifest, pooled, sigma_q, sigma_k, meta, k_bits, device)        
        print(f"\nAll done. Figures are in {FIG_DUMP}")

    # §9 run K experiment.
    print("\n=== §9  Run K experiment ===")
    final = run_k_experiment(
        data_root, manifest, turbo, pca, jq, qpca, meta, k_bits, device
    )

    # §10 results.
    print("\n=== §10  K results ===")
    report_k_results(final, k_bits)
    plot_k_results(final, k_bits)
    


if __name__ == "__main__":
    main()