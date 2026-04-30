"""Test joint-Q-K-aware orthogonal bases vs V_waterfill.

Bennett-optimal-orthogonal objective:
  D_min(R) ∝ (prod_j (R^T Σ_Q R)_jj · (R^T Σ_K R)_jj)^(1/d)
            = exp((1/d) · (sum_j log a_j + sum_j log b_j))

Hadamard inequality: sum_j log a_j(R) ≥ log det(Σ_Q), equality iff R diagonalizes Σ_Q.
                    sum_j log b_j(R) ≥ log det(Σ_K), equality iff R diagonalizes Σ_K.
Lower bound for orthogonal R: log det(Σ_Q) + log det(Σ_K), achieved iff Σ_Q, Σ_K commute.

Candidate orthogonal bases:
  R_V       = eigvec(Σ_Q)                            [V_waterfill — minimizes a, suboptimal on b]
  R_K       = eigvec(Σ_K)                            [minimizes b, suboptimal on a]
  R_QK      = eigvec(Σ_K^(1/2) Σ_Q Σ_K^(1/2))         [Q viewed through K's lens]
  R_sym     = eigvec((Σ_Q Σ_K + Σ_K Σ_Q) / 2)         [symmetric product]
  R_logsum  = eigvec(log Σ_Q + log Σ_K)              [theoretical Bennett optimum]
  R_riemann = SO(d) gradient descent from R_V         [numerical optimum]

For each basis: compute Bennett prediction + run real quantization + measure top-1.
"""
from __future__ import annotations

import sys, math, torch
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "/vault/amir/efficient-llm/teamily-project")

from experiments.stage1.toolkit.per_coord_quantization import PerCoordCompressor, round_bits_to_integer
from experiments.stage1.toolkit.metric_transform import water_fill


REPO = Path("/vault/amir/efficient-llm/teamily-project")
CCA_STATS = REPO / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
BUNDLE = REPO / "artifacts/stage1/query_stats_longbench_under4k"


def matrix_log(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric matrix log via eigendecomposition."""
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = torch.linalg.eigh(M_sym)
    log_eigvals = torch.log(eigvals.clamp_min(eps))
    return eigvecs @ torch.diag(log_eigvals) @ eigvecs.T


def matrix_sqrt(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric PSD matrix square root."""
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = torch.linalg.eigh(M_sym)
    sqrt_eigvals = torch.sqrt(eigvals.clamp_min(eps))
    return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T


def regularize(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    d = M.shape[-1]
    M_sym = 0.5 * (M + M.T)
    return M_sym + eps * (M_sym.diagonal().sum() / d).clamp_min(1e-12) * torch.eye(d)


def eigvec_descending(M: torch.Tensor) -> torch.Tensor:
    """Return eigvecs (cols) of symmetric M sorted by descending eigenvalue."""
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (M + M.T))
    idx = torch.argsort(eigvals, descending=True)
    return eigvecs[:, idx]


def riemannian_optimize(sigma_q: torch.Tensor, sigma_k: torch.Tensor,
                        R_init: torch.Tensor, n_iter: int = 200, lr: float = 0.05) -> torch.Tensor:
    """Riemannian gradient descent on SO(d) to minimize Bennett objective."""
    d = sigma_q.shape[0]
    R = R_init.clone()
    for _ in range(n_iter):
        # Compute objective and gradient
        a = (R.T @ sigma_q @ R).diagonal().clamp_min(1e-12)
        b = (R.T @ sigma_k @ R).diagonal().clamp_min(1e-12)
        # f = sum_j log(a_j) + sum_j log(b_j)
        # df/dR = 2 (Σ_Q R diag(1/a) + Σ_K R diag(1/b))  (gradient in Euclidean sense)
        grad_eucl = 2 * (sigma_q @ R @ torch.diag(1.0 / a) + sigma_k @ R @ torch.diag(1.0 / b))
        # Project to tangent space of SO(d): skew-symmetric direction
        skew = grad_eucl @ R.T - R @ grad_eucl.T  # (d, d) skew-symmetric
        skew = 0.5 * skew  # symmetrize
        # Retraction via QR or matrix exponential of -lr * skew
        # For small step: R_new ≈ R - lr * skew @ R, then re-orthogonalize
        R_new = R - lr * (skew @ R)
        Q, _ = torch.linalg.qr(R_new)
        R = Q
    return R


def bennett_log_obj(R: torch.Tensor, sigma_q: torch.Tensor, sigma_k: torch.Tensor) -> float:
    a = (R.T @ sigma_q @ R).diagonal().clamp_min(1e-12)
    b = (R.T @ sigma_k @ R).diagonal().clamp_min(1e-12)
    return float((a.log().sum() + b.log().sum()).item())


def build_compressor(R: torch.Tensor, sigma_q: torch.Tensor, sigma_k: torch.Tensor,
                     b_avg: float, head_dim: int) -> PerCoordCompressor:
    """Build PerCoordCompressor with orthogonal basis R, water-fill on (a_j × b_j)."""
    a = (R.T @ sigma_q @ R).diagonal().clamp_min(1e-30)  # weights
    b = (R.T @ sigma_k @ R).diagonal().clamp_min(1e-30)  # per-coord variance
    wf_input = (a * b).unsqueeze(0)
    bits_cont = water_fill(wf_input, total_bits=b_avg * head_dim).squeeze(0)
    bits_int = round_bits_to_integer(bits_cont.unsqueeze(0), total_bits=int(b_avg * head_dim)).squeeze(0)
    coord_stds = b.sqrt().float()
    # row-form: c = k @ R (col-form: c_col = R^T k_col), inverse: k = c @ R^T
    return PerCoordCompressor(bits_per_coord=bits_int,
                              std_per_coord=coord_stds,
                              forward_map=R.float(),
                              inverse_map=R.transpose(-1, -2).float())


@torch.no_grad()
def evaluate(comp: PerCoordCompressor, k_pre: torch.Tensor, q_test: torch.Tensor,
             sigma_q: torch.Tensor) -> dict:
    d = k_pre.shape[-1]
    k_recon = comp.roundtrip(k_pre.unsqueeze(0).unsqueeze(0)).squeeze().float()
    err = k_recon - k_pre
    # Q-weighted geometry distortion (matches eval.py convention up to /d)
    geo = torch.einsum("ld,de,le->l", err, sigma_q, err).mean().item() / d

    real_logits = torch.einsum("qd,ld->ql", q_test, k_pre) / math.sqrt(d)
    approx_logits = torch.einsum("qd,ld->ql", q_test, k_recon) / math.sqrt(d)
    real_top1 = real_logits.argmax(dim=-1)
    approx_top1 = approx_logits.argmax(dim=-1)
    top1 = (real_top1 == approx_top1).float().mean().item()

    delta = approx_logits - real_logits
    rng = torch.Generator().manual_seed(42)
    rand_idx = torch.randint(0, k_pre.shape[0], (q_test.shape[0],), generator=rng)
    d_top1 = torch.gather(delta, -1, real_top1.unsqueeze(-1)).squeeze(-1).abs().mean().item()
    d_rand = torch.gather(delta, -1, rand_idx.unsqueeze(-1)).squeeze(-1).abs().mean().item()
    return {"top1": top1, "geo": geo, "rel_err": (err.norm(dim=-1) / k_pre.norm(dim=-1)).mean().item(),
            "amp": d_top1 / max(d_rand, 1e-12), "d_top1": d_top1, "d_rand": d_rand}


def main() -> None:
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    n_kv_heads = int(cca["n_kv_heads"])
    head_dim = int(cca["head_dim"])
    payload = torch.load(BUNDLE / "examples/ex_005.pt", map_location="cpu", weights_only=False)
    L = int(payload["prompt_length"])
    b_avg = 3.0

    test_pairs = [(12, 5), (24, 7), (5, 3), (8, 5), (29, 6)]
    aggregate = defaultdict(list)
    bennett_aggr = defaultdict(list)

    for layer, kv_head in test_pairs:
        sigma_q = cca["sigma_q"][layer, kv_head].float()
        sigma_k = cca["sigma_k"][layer, kv_head].float()
        sigma_q_reg = regularize(sigma_q)
        sigma_k_reg = regularize(sigma_k)

        k_pre = payload["k_post"][layer, kv_head, :L, :].float()
        q_test = payload["q_post"][layer, kv_head*4:(kv_head+1)*4, :L, :].float().reshape(-1, head_dim)[:200]

        # Build candidate bases
        bases = {}
        bases["V (eigvec Σ_Q)"]                 = eigvec_descending(sigma_q_reg)
        bases["K (eigvec Σ_K)"]                 = eigvec_descending(sigma_k_reg)
        Sk_half = matrix_sqrt(sigma_k_reg)
        bases["R_QK (eigvec K^½ Σ_Q K^½)"]      = eigvec_descending(Sk_half @ sigma_q_reg @ Sk_half)
        bases["R_sym (eigvec (Σ_QΣ_K+Σ_KΣ_Q)/2)"] = eigvec_descending(0.5 * (sigma_q_reg @ sigma_k_reg + sigma_k_reg @ sigma_q_reg))
        bases["R_logsum (eigvec logΣ_Q+logΣ_K)"] = eigvec_descending(matrix_log(sigma_q_reg) + matrix_log(sigma_k_reg))
        bases["R_riemann (SO(d) opt)"]          = riemannian_optimize(sigma_q_reg, sigma_k_reg, bases["V (eigvec Σ_Q)"])

        # Bennett lower bound (joint det)
        log_det_q = float(torch.linalg.slogdet(sigma_q_reg)[1])
        log_det_k = float(torch.linalg.slogdet(sigma_k_reg)[1])
        bennett_floor = log_det_q + log_det_k

        print(f"\n{'='*100}")
        print(f"  layer={layer}, kv_head={kv_head}")
        print(f"  log det Σ_Q = {log_det_q:.3f}, log det Σ_K = {log_det_k:.3f}, joint floor = {bennett_floor:.3f}")
        print(f"{'='*100}")

        results = {}
        for name, R in bases.items():
            obj = bennett_log_obj(R, sigma_q_reg, sigma_k_reg)
            comp = build_compressor(R, sigma_q_reg, sigma_k_reg, b_avg, head_dim)
            metrics = evaluate(comp, k_pre, q_test, sigma_q_reg)
            results[name] = (obj, metrics)
            bennett_aggr[name].append(obj - bennett_floor)
            aggregate[name].append(metrics)

        print(f"\n{'basis':<40} {'log obj':>10} {'gap to floor':>14} {'top-1':>8} {'geo':>10} {'rel_err':>9} {'amp@top1':>10}")
        print("-" * 110)
        for name, (obj, m) in results.items():
            gap = obj - bennett_floor
            print(f"{name:<40} {obj:>10.3f} {gap:>14.3f} {m['top1']:>8.4f} {m['geo']:>10.4f} {m['rel_err']:>9.4f} {m['amp']:>10.2f}x")

    # Aggregate
    print(f"\n{'='*110}")
    print(f"  AGGREGATE across {len(test_pairs)} heads (means)")
    print(f"{'='*110}")
    print(f"\n{'basis':<40} {'gap to floor':>14} {'top-1':>8} {'geo':>10} {'rel_err':>9} {'amp@top1':>10}")
    print("-" * 110)
    for name in aggregate.keys():
        gap_mean = sum(bennett_aggr[name]) / len(bennett_aggr[name])
        avg = {k: sum(m[k] for m in aggregate[name]) / len(aggregate[name]) for k in aggregate[name][0]}
        print(f"{name:<40} {gap_mean:>14.3f} {avg['top1']:>8.4f} {avg['geo']:>10.4f} {avg['rel_err']:>9.4f} {avg['amp']:>10.2f}x")


if __name__ == "__main__":
    main()
