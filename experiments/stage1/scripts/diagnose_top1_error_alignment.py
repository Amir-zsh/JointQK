"""Hypothesis: each method's reconstruction-error covariance has a *specific shape*
that determines top-1 retention.

Theory (from per-coord water-fill structure):
  - V_waterfill:  Cov(err) ≈ θ_V * M_q^(-1)  (error perpendicular to Q-mass directions)
  - V3:           Cov(err) ≈ θ_3 * I         (isotropic)
  - CCA_waterfill: Cov(err) ≈ θ_C * Σ_K       (error shaped like keys themselves)

Why this matters for top-1:
  Per-query perturbation variance: Var(q^T err) = q^T Cov(err) q
  - V_waterfill: q^T M_q^(-1) q  ≈ small if q is in M_q's high-eigval directions (which it is)
  - V3:          ‖q‖²              moderate, direction-independent
  - CCA:         q^T Σ_K q         ≈ same scale as true logit variance Var(q^T k)
                 → perturbation noise on the same scale as the signal we want to preserve!

Measurement:
  1. Empirical Cov(err) for each method on real data.
  2. Match to predicted shape (M_q^(-1), I, Σ_K) via Frobenius cosine.
  3. Compare q^T Cov(err) q to q^T Σ_K q across queries (the SNR ratio).
"""
from __future__ import annotations

import json
import os
import sys
import math
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/vault/amir/efficient-llm/teamily-project")

from experiments.stage1.toolkit.per_coord_quantization import build_method_compressor
from experiments.stage1.toolkit.quantization import Stage1MSECompressor


REPO = Path("/vault/amir/efficient-llm/teamily-project")
CCA_STATS = REPO / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
BUNDLE = REPO / "artifacts/stage1/query_stats_longbench_under4k"
OUT = REPO / "artifacts/stage1/cca_vs_waterfill_study/diagnostics_v3_vs_cca"
OUT.mkdir(parents=True, exist_ok=True)


def fro_cosine(A: torch.Tensor, B: torch.Tensor) -> float:
    """Cosine similarity between matrices in Frobenius inner product."""
    A = A.flatten().double()
    B = B.flatten().double()
    return float((A @ B) / (A.norm() * B.norm()).clamp_min(1e-30))


def main() -> None:
    torch.manual_seed(0)
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    n_kv_heads = int(cca["n_kv_heads"])
    head_dim = int(cca["head_dim"])
    layer, kv_head = 12, 5
    sigma_q = cca["sigma_q"][layer, kv_head].float()
    sigma_k = cca["sigma_k"][layer, kv_head].float()
    P_K = cca["P_K"][layer, kv_head].float()
    P_K_inv = cca["P_K_inv"][layer, kv_head].float()
    rho = cca["rho"][layer, kv_head].float()
    mq_eigvals = cca["mq_eigvals"][layer, kv_head].float()
    mq_eigvecs = cca["mq_eigvecs"][layer, kv_head].float()
    payload = torch.load(BUNDLE / "examples/ex_005.pt", map_location="cpu", weights_only=False)
    prompt_length = int(payload["prompt_length"])
    k_pre = payload["k_post"][layer, kv_head, :prompt_length, :].float()
    q_group = payload["q_post"][layer, kv_head*4:(kv_head+1)*4, :prompt_length, :].float()

    L, d = k_pre.shape
    b_avg = 3.0

    # Build compressors (same as before)
    v3 = Stage1MSECompressor(d, int(b_avg), seed=0, device="cpu")
    v_wf = build_method_compressor("v_waterfill", sigma_q, sigma_k, None, mq_eigvals, mq_eigvecs, b_avg, d, 0, d)
    cca_wf = build_method_compressor("cca_waterfill", sigma_q, sigma_k,
                                     {"P_K": P_K, "P_K_inv": P_K_inv, "rho": rho},
                                     mq_eigvals, mq_eigvecs, b_avg, d, 0, d)

    # Reconstruct
    keys_3d = k_pre.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        k_v3 = v3.roundtrip(keys_3d).squeeze().float()
        k_v_wf = v_wf.roundtrip(keys_3d).squeeze().float()
        k_cca_wf = cca_wf.roundtrip(keys_3d).squeeze().float()
    err = {"v3": k_v3 - k_pre, "v_waterfill": k_v_wf - k_pre, "cca_waterfill": k_cca_wf - k_pre}

    # Empirical Cov(err)
    print("=" * 72)
    print("Test 1: empirical Cov(err) shape vs theoretical predictions")
    print("=" * 72)
    M_q = sigma_q  # (uncentered second moment = M_q)
    Sigma_K = sigma_k
    M_q_inv = torch.linalg.pinv(M_q + 1e-4 * (M_q.diagonal().sum() / d) * torch.eye(d))
    eye = torch.eye(d)

    print(f"\n{'method':<15} {'cos w/ M_q^-1':>14} {'cos w/ I':>10} {'cos w/ Sigma_K':>16}")
    for name, e in err.items():
        cov_err = (e.T @ e) / e.shape[0]
        cos_mqinv = fro_cosine(cov_err, M_q_inv)
        cos_eye = fro_cosine(cov_err, eye)
        cos_sk = fro_cosine(cov_err, Sigma_K)
        print(f"{name:<15} {cos_mqinv:>14.4f} {cos_eye:>10.4f} {cos_sk:>16.4f}")

    # Test 2: per-query variance ratio
    print()
    print("=" * 72)
    print("Test 2: per-query perturbation variance vs true-logit variance")
    print("=" * 72)
    # Var(q^T err) = q^T Cov(err) q
    # Var(q^T k) ≈ q^T Sigma_K q (treating keys as a sample from Sigma_K distribution)
    # Ratio = noise variance / signal variance per query.
    # Under our theory:
    #   V3: ratio ≈ θ_3 * ||q||² / (q^T Sigma_K q) — depends on q direction
    #   V_wf: ratio ≈ θ_V * (q^T M_q^-1 q) / (q^T Sigma_K q) — small (q is M_q-typical)
    #   CCA: ratio ≈ θ_C — flat across q!
    q_flat = q_group.reshape(-1, d)[:200]
    print(f"\nUsing {q_flat.shape[0]} prefill queries")

    print(f"\n{'method':<15} {'q^T Cov(err) q':>16} {'q^T Sigma_K q':>14} {'noise/signal':>13}")
    for name, e in err.items():
        cov_err = (e.T @ e) / e.shape[0]
        # Per-query noise variance
        noise_var = torch.einsum("qd,de,qe->q", q_flat, cov_err, q_flat)
        signal_var = torch.einsum("qd,de,qe->q", q_flat, Sigma_K, q_flat)
        ratio = (noise_var / signal_var.clamp_min(1e-30))
        print(f"{name:<15} {noise_var.mean().item():>16.4e} {signal_var.mean().item():>14.4e} {ratio.mean().item():>13.4f}")
        print(f"    quantiles 10/50/90 of ratio: {torch.quantile(ratio, 0.1):.4f} / {torch.quantile(ratio, 0.5):.4f} / {torch.quantile(ratio, 0.9):.4f}")

    # Test 3: predicted vs empirical |delta at top-1|
    # |delta| = sqrt(q^T Cov(err) q), so its mean over q is sqrt(noise_var).mean()
    print()
    print("=" * 72)
    print("Test 3: predicted vs measured |delta at the true-top-1 logit|")
    print("=" * 72)
    print(f"\n{'method':<15} {'sqrt(noise_var) pred':>22} {'measured |delta|':>18}")
    real_logits = torch.einsum("qd,ld->ql", q_flat, k_pre) / math.sqrt(d)
    real_top1 = real_logits.argmax(dim=-1)
    for name, e in err.items():
        cov_err = (e.T @ e) / e.shape[0]
        noise_var = torch.einsum("qd,de,qe->q", q_flat, cov_err, q_flat)
        pred_delta = (noise_var.sqrt() / math.sqrt(d)).mean()  # sqrt(d) for the /sqrt(d) in attention scaling
        # measured: q^T (k_recon - k_pre)_at_top1 / sqrt(d)
        recon = {"v3": k_v3, "v_waterfill": k_v_wf, "cca_waterfill": k_cca_wf}[name]
        approx_logits = torch.einsum("qd,ld->ql", q_flat, recon) / math.sqrt(d)
        delta = approx_logits - real_logits
        delta_at_top1 = torch.gather(delta, -1, real_top1.unsqueeze(-1)).squeeze(-1)
        print(f"{name:<15} {pred_delta.item():>22.4f} {delta_at_top1.abs().mean().item():>18.4f}")

    # ---- Charts ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Cov(err) heatmap-style: show eigval spectra
    ax = axes[0]
    for name, e in err.items():
        cov_err = (e.T @ e) / e.shape[0]
        eigvals = torch.linalg.eigvalsh(cov_err).clamp_min(1e-30)
        eigvals_sorted = torch.sort(eigvals, descending=True).values
        ax.semilogy(eigvals_sorted, label=name,
                    color={"v3":"#777","v_waterfill":"#228833","cca_waterfill":"#4477AA"}[name])
    ax.set_xlabel("eigval index (sorted desc)")
    ax.set_ylabel("eigval(Cov(err))")
    ax.set_title(f"Reconstruction-error covariance spectrum\n(layer={layer}, head={kv_head}, b_avg={b_avg})")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Cosine similarity bar chart
    ax = axes[1]
    methods = ["v3", "v_waterfill", "cca_waterfill"]
    targets = [("M_q⁻¹", M_q_inv, "expected: V_wf"),
               ("I", eye, "expected: V3"),
               ("Σ_K", Sigma_K, "expected: CCA")]
    width = 0.25
    x = np.arange(len(methods))
    for ti, (tname, tmat, ttag) in enumerate(targets):
        cos = []
        for m in methods:
            ce = (err[m].T @ err[m]) / err[m].shape[0]
            cos.append(fro_cosine(ce, tmat))
        ax.bar(x + (ti - 1) * width, cos, width, label=f"{tname}  ({ttag})")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=10)
    ax.set_ylabel("Frobenius cosine sim")
    ax.set_title("Cov(err) shape match to theoretical predictions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # 3. Per-query noise/signal ratio histogram
    ax = axes[2]
    for name in methods:
        ce = (err[name].T @ err[name]) / err[name].shape[0]
        noise_var = torch.einsum("qd,de,qe->q", q_flat, ce, q_flat)
        signal_var = torch.einsum("qd,de,qe->q", q_flat, Sigma_K, q_flat)
        ratio = (noise_var / signal_var.clamp_min(1e-30)).numpy()
        ax.hist(ratio, bins=40, alpha=0.5, label=name,
                color={"v3":"#777","v_waterfill":"#228833","cca_waterfill":"#4477AA"}[name])
    ax.set_xlabel("q^T Cov(err) q  /  q^T Σ_K q  (noise-to-signal var ratio)")
    ax.set_ylabel("queries")
    ax.set_title("Per-query noise/signal variance ratio")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "error_covariance_shape.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {OUT}/error_covariance_shape.png")


if __name__ == "__main__":
    main()
