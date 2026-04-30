"""Why does V3 beat CCA_waterfill on top-1 despite higher geometry distortion?
And why does V_waterfill beat CCA_waterfill on both metrics?

Approach: take a representative (layer, kv_head), build all three compressors,
quantize a real example's prefill keys, and decompose where the error comes from.

Specifically we'll measure:
  1. Per-coord bit allocation under each method.
  2. Per-coord per-key variance and "discarded mass" (b_j = 0 coords).
  3. Reconstruction-error covariance: isotropic vs anisotropic.
  4. Per-query logit error decomposition: error from kept-coord quantization noise
     vs error from discarded-subspace mass.
  5. Top-1-vs-runner-up gap distribution and flip rate per method.

Outputs JSON metrics + a few charts.
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


def main() -> None:
    torch.manual_seed(0)

    # --- Load calibration ---
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    n_kv_heads = int(cca["n_kv_heads"])
    head_dim = int(cca["head_dim"])

    # Pick a representative non-layer-0 head with average behavior.
    layer, kv_head = 12, 5
    h = layer * n_kv_heads + kv_head
    sigma_q = cca["sigma_q"][layer, kv_head].float()
    sigma_k = cca["sigma_k"][layer, kv_head].float()
    P_K = cca["P_K"][layer, kv_head].float()
    P_K_inv = cca["P_K_inv"][layer, kv_head].float()
    rho = cca["rho"][layer, kv_head].float()
    mq_eigvals = cca["mq_eigvals"][layer, kv_head].float()
    mq_eigvecs = cca["mq_eigvecs"][layer, kv_head].float()
    print(f"Using layer={layer}, kv_head={kv_head}; head_dim={head_dim}")

    # --- Load one prefill cache to test on ---
    payload = torch.load(BUNDLE / "examples/ex_005.pt", map_location="cpu", weights_only=False)  # qasper
    prompt_length = int(payload["prompt_length"])
    print(f"Example: ex_005.pt, prompt_length={prompt_length}")

    # k_post: (n_layers, n_kv_heads, T, d). Take prefill keys for our (layer, head).
    k_pre = payload["k_post"][layer, kv_head, :prompt_length, :].float()  # (L, d)
    # Take all prefill queries for this head's GQA group, take first 200 of them.
    n_q_heads = payload["q_post"].shape[1]
    group = n_q_heads // n_kv_heads
    q_group = payload["q_post"][layer, kv_head * group : (kv_head + 1) * group, :prompt_length, :].float()  # (group, L, d)
    n_q_per_pos = group
    print(f"Keys: {tuple(k_pre.shape)}; Queries per position: {n_q_per_pos}")

    L = k_pre.shape[0]
    d = head_dim
    b_avg = 3.0

    # --- Build compressors ---
    print()
    print("Building compressors...")
    v3 = Stage1MSECompressor(head_dim, int(b_avg), seed=0, device="cpu")
    v_wf = build_method_compressor("v_waterfill", sigma_q, sigma_k, None, mq_eigvals, mq_eigvecs, b_avg, head_dim, 0, head_dim)
    cca_wf = build_method_compressor("cca_waterfill", sigma_q, sigma_k,
                                     {"P_K": P_K, "P_K_inv": P_K_inv, "rho": rho},
                                     mq_eigvals, mq_eigvecs, b_avg, head_dim, 0, head_dim)

    bits_v_wf = v_wf.bits_int.tolist()
    bits_cca_wf = cca_wf.bits_int.tolist()
    print(f"V_waterfill bits sorted desc: {sorted(bits_v_wf, reverse=True)[:30]}...")
    print(f"  zero-bit coords: {sum(1 for b in bits_v_wf if b == 0)} / {d}")
    print(f"CCA_waterfill bits sorted desc: {sorted(bits_cca_wf, reverse=True)[:30]}...")
    print(f"  zero-bit coords: {sum(1 for b in bits_cca_wf if b == 0)} / {d}")
    print(f"V3: uniform {b_avg} bits per coord (after unit-normalize)")

    # --- Reconstruct keys ---
    keys_3d = k_pre.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        k_v3 = v3.roundtrip(keys_3d).squeeze().float()
        k_v_wf = v_wf.roundtrip(keys_3d).squeeze().float()
        k_cca_wf = cca_wf.roundtrip(keys_3d).squeeze().float()

    err_v3 = (k_v3 - k_pre)  # (L, d)
    err_v_wf = (k_v_wf - k_pre)
    err_cca_wf = (k_cca_wf - k_pre)

    print()
    print("Reconstruction-error magnitude per key:")
    for name, err in [("v3", err_v3), ("v_waterfill", err_v_wf), ("cca_waterfill", err_cca_wf)]:
        norm = err.norm(dim=-1)
        print(f"  {name}: mean ||err||={norm.mean():.3f}  rel={norm.mean()/k_pre.norm(dim=-1).mean():.3f}")

    # --- Error covariance structure ---
    print()
    print("Error covariance anisotropy (max/median eigval of Cov(err)):")
    for name, err in [("v3", err_v3), ("v_waterfill", err_v_wf), ("cca_waterfill", err_cca_wf)]:
        cov_err = (err.T @ err) / err.shape[0]
        eigvals = torch.linalg.eigvalsh(cov_err).clamp_min(0)
        eigvals_sorted = torch.sort(eigvals, descending=True).values
        anisotropy = eigvals_sorted[0] / eigvals_sorted[d // 2].clamp_min(1e-12)
        rank95 = (torch.cumsum(eigvals_sorted, dim=0) / eigvals_sorted.sum() >= 0.95).int().argmax().item() + 1
        print(f"  {name}: top-eig/median-eig = {anisotropy:.1f}x;  rank-95% energy = {rank95}/{d}")

    # --- Top-1 flip analysis on real queries ---
    # Take first 200 prefill queries (any head in the group, average over group).
    print()
    print("Top-1 retention analysis (first 200 prefill queries, GQA-pooled):")
    n_q = min(200, q_group.shape[1])
    q_test = q_group[:, :n_q, :]  # (group, n_q, d)

    # True logits and reconstructed logits, per Q-head. We use group=4.
    real_logits = torch.einsum("gqd,ld->gql", q_test, k_pre) / math.sqrt(d)
    real_top1 = real_logits.argmax(dim=-1)  # (group, n_q)
    real_top2_score, _ = torch.topk(real_logits, 2, dim=-1)
    gap = (real_top2_score[..., 0] - real_top2_score[..., 1])  # (group, n_q) — top-1 vs runner-up gap

    print(f"  Mean top-1 vs runner-up gap: {gap.mean():.4f}")
    print(f"  Median gap: {gap.median():.4f}")
    print(f"  Min/max gap: {gap.min():.4f} / {gap.max():.4f}")

    method_results = {}
    for name, k_recon, err in [("v3", k_v3, err_v3), ("v_waterfill", k_v_wf, err_v_wf), ("cca_waterfill", k_cca_wf, err_cca_wf)]:
        approx_logits = torch.einsum("gqd,ld->gql", q_test, k_recon) / math.sqrt(d)
        approx_top1 = approx_logits.argmax(dim=-1)
        flip = (approx_top1 != real_top1)
        top1 = (~flip).float().mean().item()
        # For flipped queries, what was their gap?
        flipped_gaps = gap[flip]
        unflipped_gaps = gap[~flip]
        # Distribution of logit perturbation at the true top-1 key
        # delta_at_top1 = q^T (k_top1 - k̂_top1)
        idx = real_top1.unsqueeze(-1)
        delta = torch.gather(approx_logits - real_logits, -1, idx).squeeze(-1)  # (group, n_q)
        # And distribution at the runner-up
        idx2 = real_logits.topk(2, dim=-1).indices[..., 1:2]
        delta_runup = torch.gather(approx_logits - real_logits, -1, idx2).squeeze(-1)
        method_results[name] = {
            "top1": top1,
            "flipped_n": int(flip.sum()),
            "total_n": int(flip.numel()),
            "median_flipped_gap": flipped_gaps.median().item() if flip.any() else None,
            "median_unflipped_gap": unflipped_gaps.median().item(),
            "mean_delta_at_top1_abs": delta.abs().mean().item(),
            "mean_delta_at_runup_abs": delta_runup.abs().mean().item(),
            "delta_at_top1_std": delta.std().item(),
        }
        print(f"  {name}: top-1 = {top1:.4f}, flipped {int(flip.sum())}/{flip.numel()}")
        if flip.any():
            print(f"    flipped queries' median gap: {flipped_gaps.median():.5f}  (vs unflipped {unflipped_gaps.median():.5f})")
        print(f"    mean |delta| at true-top-1: {delta.abs().mean():.5f}  at runner-up: {delta_runup.abs().mean():.5f}")
        print(f"    std of delta at top-1: {delta.std():.5f}  (this is what flips top-1 when > gap)")

    # --- Discarded subspace energy in CCA ---
    print()
    print("Energy decomposition for CCA_waterfill:")
    discarded_coords = [j for j, b in enumerate(bits_cca_wf) if b == 0]
    kept_coords = [j for j, b in enumerate(bits_cca_wf) if b > 0]
    print(f"  Discarded coords: {len(discarded_coords)}, kept: {len(kept_coords)}")

    # Project keys into CCA-coord space
    k_in_cca = k_pre @ P_K.T  # (L, d) — canonical K coords
    discarded_energy = (k_in_cca[:, discarded_coords] ** 2).sum(dim=-1)
    kept_energy = (k_in_cca[:, kept_coords] ** 2).sum(dim=-1)
    total_energy = (k_in_cca ** 2).sum(dim=-1)
    print(f"  Per-key fraction in discarded subspace:")
    print(f"    median: {(discarded_energy / total_energy).median():.4f}")
    print(f"    mean:   {(discarded_energy / total_energy).mean():.4f}")
    print(f"    max:    {(discarded_energy / total_energy).max():.4f}")

    # Same for queries (project Q via the canonical-Q basis is harder; let's use k coords for now to estimate Q's discarded energy)
    # Actually use P_K to project q since that gives us the same coords system
    q_flat = q_test.reshape(-1, d)
    q_in_cca = q_flat @ P_K.T
    q_disc_e = (q_in_cca[:, discarded_coords] ** 2).sum(-1)
    q_total_e = (q_in_cca ** 2).sum(-1)
    print(f"  Per-query fraction in same discarded subspace (CCA-K coords): median {(q_disc_e/q_total_e).median():.4f}")

    # --- Bit allocation chart ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    ax.plot(sorted(bits_v_wf, reverse=True), color="#228833", linewidth=2, label="V + water-fill")
    ax.plot(sorted(bits_cca_wf, reverse=True), color="#4477AA", linewidth=2, label="CCA + water-fill")
    ax.axhline(b_avg, color="#777", linestyle="--", linewidth=1, label="V3 uniform")
    ax.set_xlabel("coord (sorted by allocation, desc)")
    ax.set_ylabel("bits")
    ax.set_title(f"Per-coord bit allocation at b_avg={b_avg} (layer={layer}, head={kv_head})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    fracs = (discarded_energy / total_energy).numpy()
    ax.hist(fracs, bins=40, color="#4477AA", alpha=0.7, edgecolor="black")
    ax.set_xlabel("frac of ||k||² in CCA-discarded subspace")
    ax.set_ylabel("number of keys")
    ax.set_title(f"CCA: per-key energy in discarded subspace ({len(discarded_coords)}/{d} coords have b=0)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "bit_allocation_and_discarded_energy.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {OUT}/bit_allocation_and_discarded_energy.png")

    # --- Logit error distribution chart ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ai, (name, k_recon) in enumerate([("v3", k_v3), ("v_waterfill", k_v_wf), ("cca_waterfill", k_cca_wf)]):
        approx_logits = torch.einsum("gqd,ld->gql", q_test, k_recon) / math.sqrt(d)
        diff = (approx_logits - real_logits).flatten().numpy()
        axes[ai].hist(diff, bins=80, color={"v3":"#777","v_waterfill":"#228833","cca_waterfill":"#4477AA"}[name],
                       alpha=0.85, edgecolor="black", linewidth=0.3)
        axes[ai].set_title(f"{name}\nstd={diff.std():.4f}, kurt={float((diff**4).mean()/(diff**2).mean()**2 - 3):.2f}")
        axes[ai].set_xlabel("logit error  q^T(k - k_hat) / sqrt(d)")
        axes[ai].grid(alpha=0.3)
        axes[ai].set_xlim(-0.05, 0.05)
    fig.suptitle(f"Per-(query, key) logit error distribution at b_avg={b_avg} (layer={layer}, head={kv_head})")
    fig.tight_layout()
    fig.savefig(OUT / "logit_error_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/logit_error_distributions.png")

    # --- Save metrics ---
    metrics = {
        "layer": layer, "kv_head": kv_head, "head_dim": d, "L_keys": L, "n_q": n_q,
        "bit_allocations": {
            "v3_uniform": int(b_avg),
            "v_waterfill_sorted_desc": sorted(bits_v_wf, reverse=True),
            "cca_waterfill_sorted_desc": sorted(bits_cca_wf, reverse=True),
            "v_waterfill_zero_bits": sum(1 for b in bits_v_wf if b == 0),
            "cca_waterfill_zero_bits": sum(1 for b in bits_cca_wf if b == 0),
        },
        "geometry_check_per_key": {
            "v3_mean_relerr": float(err_v3.norm(dim=-1).mean() / k_pre.norm(dim=-1).mean()),
            "v_waterfill_mean_relerr": float(err_v_wf.norm(dim=-1).mean() / k_pre.norm(dim=-1).mean()),
            "cca_waterfill_mean_relerr": float(err_cca_wf.norm(dim=-1).mean() / k_pre.norm(dim=-1).mean()),
        },
        "top1_results": method_results,
        "cca_discarded_subspace_energy_frac": {
            "median": float((discarded_energy/total_energy).median()),
            "mean": float((discarded_energy/total_energy).mean()),
            "max": float((discarded_energy/total_energy).max()),
        },
        "logit_gap": {
            "mean": float(gap.mean()),
            "median": float(gap.median()),
            "p10": float(torch.quantile(gap.flatten(), 0.1)),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  wrote {OUT}/metrics.json")


if __name__ == "__main__":
    main()
