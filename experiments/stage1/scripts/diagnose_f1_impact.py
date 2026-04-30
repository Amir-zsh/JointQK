"""Quick diagnostic to estimate F1's impact.

Compares Σ_Q computed two ways on the same prefill data:
    A = E[(mean_g q_g)(mean_g q_g)^T]    (current/buggy in E4)
    B = mean_g E[q_g q_g^T]              (correct, used in E1/E3-global)

Reports per (layer, kv_head):
    - trace ratio A/B
    - frobenius distance ||A - B||_F / ||B||_F
    - eigenvector subspace alignment at top-r (r = 64) — measures how different the V basis is
    - Top-3 ρ values from CCA computed using A vs B — measures how different CCA ρ values are

If subspace alignment is ≥ 0.99 and CCA ρ matches within ~1%, F1 has minimal impact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.stage1.toolkit import compute_cca_basis  # noqa: E402


def compute_AB(payload: dict, n_layers: int, n_kv_heads: int, head_dim: int):
    """Returns (A, B, C_QK, Σ_K) all of shape (n_layers, n_kv_heads, d, d) on CPU."""
    L = int(payload["prompt_length"])
    q = payload["q_post"][:, :, :L, :].float()  # (n_layers, n_q_heads, L, d)
    k = payload["k_post"][:, :, :L, :].float()  # (n_layers, n_kv_heads, L, d)
    n_q_heads = q.shape[1]
    group = n_q_heads // n_kv_heads

    # B: per-query-head outer, then average across the group
    q_per_head_outer = torch.einsum("lhsd,lhse->lhde", q, q) / L  # (n_layers, n_q_heads, d, d)
    B = q_per_head_outer.view(n_layers, n_kv_heads, group, head_dim, head_dim).mean(dim=2)

    # A: average q across the group, then outer
    q_grouped = q.view(n_layers, n_kv_heads, group, L, head_dim).mean(dim=2)  # (n_layers, n_kv_heads, L, d)
    A = torch.einsum("lhsd,lhse->lhde", q_grouped, q_grouped) / L

    # Σ_K and C_QK (we'll need these to compare CCA bases)
    Sigma_K = torch.einsum("lhsd,lhse->lhde", k, k) / L
    C_QK = torch.einsum("lhsd,lhse->lhde", q_grouped, k) / L

    return A, B, C_QK, Sigma_K


def subspace_alignment(M1: torch.Tensor, M2: torch.Tensor, r: int) -> float:
    """Top-r principal subspace alignment, ∈ [0, 1].

    Defined as (1/r) · sum of squared singular values of U1.T @ U2, where U_i is the
    top-r eigvec matrix of M_i. 1.0 = identical subspace, 0.0 = orthogonal subspaces.
    """
    M1_sym = 0.5 * (M1 + M1.transpose(-1, -2))
    M2_sym = 0.5 * (M2 + M2.transpose(-1, -2))
    e1, U1 = torch.linalg.eigh(M1_sym)
    e2, U2 = torch.linalg.eigh(M2_sym)
    # sort descending
    U1 = U1[:, torch.argsort(e1, descending=True)]
    U2 = U2[:, torch.argsort(e2, descending=True)]
    U1r = U1[:, :r]
    U2r = U2[:, :r]
    overlap = U1r.transpose(-1, -2) @ U2r
    s = torch.linalg.svdvals(overlap)
    return (s.pow(2).sum() / r).item()


def main() -> int:
    bundle = REPO / "artifacts/stage1/query_stats_longbench_under4k"
    examples_dir = bundle / "examples"
    paths = sorted(examples_dir.glob("ex_*.pt"))
    print(f"Loaded {len(paths)} example payloads from {bundle.name}")
    if not paths:
        print("ERROR: no examples found", file=sys.stderr)
        return 1

    # Use the first qasper example
    payload = torch.load(paths[0], map_location="cpu", weights_only=False)
    L = int(payload["prompt_length"])
    q_shape = payload["q_post"].shape
    k_shape = payload["k_post"].shape
    n_layers = q_shape[0]
    n_q_heads = q_shape[1]
    n_kv_heads = k_shape[1]
    head_dim = q_shape[-1]
    print(f"Example: {paths[0].name}, prompt_length={L}, n_layers={n_layers}, "
          f"n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print()

    A, B, C_QK, Sigma_K = compute_AB(payload, n_layers, n_kv_heads, head_dim)

    # Per (layer, kv_head) statistics
    rows = []
    for layer in range(n_layers):
        for h in range(n_kv_heads):
            A_h = A[layer, h]
            B_h = B[layer, h]
            trace_ratio = (A_h.diag().sum() / B_h.diag().sum().clamp_min(1e-30)).item()
            frob_rel = ((A_h - B_h).norm() / B_h.norm().clamp_min(1e-30)).item()
            align_r64 = subspace_alignment(A_h, B_h, r=64)
            align_r16 = subspace_alignment(A_h, B_h, r=16)
            rows.append((layer, h, trace_ratio, frob_rel, align_r64, align_r16))

    # Summary across all (layer, head) pairs
    import numpy as np
    arr = np.array(rows)
    print(f"Across {len(rows)} (layer, kv_head) pairs:")
    print(f"  trace(A) / trace(B):     median={np.median(arr[:, 2]):.3f}  "
          f"min={arr[:, 2].min():.3f}  max={arr[:, 2].max():.3f}")
    print(f"  ||A - B||_F / ||B||_F:   median={np.median(arr[:, 3]):.3f}  "
          f"min={arr[:, 3].min():.3f}  max={arr[:, 3].max():.3f}")
    print(f"  top-64 V subspace align: median={np.median(arr[:, 4]):.4f}  "
          f"min={arr[:, 4].min():.4f}  max={arr[:, 4].max():.4f}")
    print(f"  top-16 V subspace align: median={np.median(arr[:, 5]):.4f}  "
          f"min={arr[:, 5].min():.4f}  max={arr[:, 5].max():.4f}")
    print()

    # Pick a few representative (layer, head) pairs and compute CCA both ways
    print("CCA ρ comparison (A-based vs B-based) for sample (layer, kv_head):")
    sample_pairs = [(0, 0), (1, 0), (5, 3), (15, 5), (25, 7), (35, 0)]
    print(f"{'layer':>5} {'head':>5}  {'rho_1 A':>9} {'rho_1 B':>9}   {'rho_5 A':>9} {'rho_5 B':>9}   {'rho_30 A':>9} {'rho_30 B':>9}   {'r95 A':>5} {'r95 B':>5}")
    for layer, h in sample_pairs:
        cca_A = compute_cca_basis(A[layer, h], Sigma_K[layer, h], C_QK[layer, h], eps=1e-4)
        cca_B = compute_cca_basis(B[layer, h], Sigma_K[layer, h], C_QK[layer, h], eps=1e-4)
        rho_A = cca_A["rho"]
        rho_B = cca_B["rho"]
        # r95 with each
        cumA = (rho_A ** 2).cumsum(0); cumA = cumA / cumA[-1].clamp_min(1e-30)
        cumB = (rho_B ** 2).cumsum(0); cumB = cumB / cumB[-1].clamp_min(1e-30)
        r95A = int((cumA >= 0.95).long().argmax().item()) + 1
        r95B = int((cumB >= 0.95).long().argmax().item()) + 1
        print(f"{layer:>5} {h:>5}  {rho_A[0].item():>9.4f} {rho_B[0].item():>9.4f}   "
              f"{rho_A[4].item():>9.4f} {rho_B[4].item():>9.4f}   "
              f"{rho_A[29].item():>9.4f} {rho_B[29].item():>9.4f}   "
              f"{r95A:>5} {r95B:>5}")

    print()
    print("Verdict guidance:")
    print("  - top-64 V subspace align ≥ 0.99 across all (layer, head) → F1 has minimal impact")
    print("  - subspace align < 0.95 → V basis differs noticeably; eval numbers will shift")
    print("  - CCA r95 differs by > 5 ranks consistently → CCA basis is sensitive to F1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
