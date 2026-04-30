"""Final test: are top-1 keys MORE EXTREME in CCA-canonical coordinates than random keys?

Theory:
  - The CCA basis aligns Q-K correlation per coordinate.
  - The top-1 key for a query q has extreme canonical-K coordinate values aligned with q's
    canonical direction (because that's exactly what q^T k = sum_j ρ_j q_canon_j k_canon_j
    maximizes when picking top-1).
  - Lloyd-Max scalar quantization on a finite codebook has bigger error at the tail of the
    distribution than in the bulk → top-1 keys' coord values get bigger errors
    → q^T (k - k̂) is bigger at top-1 than at random keys.
  - V3 and V_waterfill don't have this property because their bases aren't joint Q-K-aligned.

Measurement: per query, compute the canonical-coord values of (top-1 key, random key) and
compare their distributions. Specifically, look at how many standard deviations from the
per-coord mean the top-1 key's coord value sits.
"""
from __future__ import annotations

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

REPO = Path("/vault/amir/efficient-llm/teamily-project")
CCA_STATS = REPO / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
BUNDLE = REPO / "artifacts/stage1/query_stats_longbench_under4k"
OUT = REPO / "artifacts/stage1/cca_vs_waterfill_study/diagnostics_v3_vs_cca"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
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
    print(f"L={L} keys, d={d}")

    # Project keys into both bases
    k_in_cca = k_pre @ P_K.T  # (L, d) — canonical K coords
    k_in_v = k_pre @ mq_eigvecs  # (L, d) — V eigenbasis

    # In V3's "basis": just use the original coord values as a stand-in (V3's random rotation
    # produces statistically similar coord-value distributions to the original in expectation).
    k_in_orig = k_pre  # for comparison

    # Compute per-coord std (centered) for each basis
    def per_coord_z_score(x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=0).clamp_min(1e-6)
        return (x - x.mean(dim=0)) / std  # (L, d) z-scored values

    z_cca = per_coord_z_score(k_in_cca)
    z_v = per_coord_z_score(k_in_v)
    z_orig = per_coord_z_score(k_in_orig)

    # For each query, find top-1 key and compute the *maximum* |z| across coords (the tail extremity)
    q_flat = q_group.reshape(-1, d)[:200]
    real_logits = torch.einsum("qd,ld->ql", q_flat, k_pre) / math.sqrt(d)
    sorted_idx = real_logits.argsort(dim=-1, descending=True)
    top1_idx = sorted_idx[:, 0]
    rng = torch.Generator().manual_seed(42)
    rand_idx = torch.randint(0, L, (q_flat.shape[0],), generator=rng)

    print()
    print("Per-coord z-score (centered, scaled by per-coord std) for top-1 vs random keys:")
    print(f"{'basis':<15} {'top1 mean |z|':>14} {'top1 max |z|':>14} {'rand mean |z|':>14} "
          f"{'rand max |z|':>14}  {'top1 max/rand max':>18}")
    for name, z in [("CCA-canonical", z_cca), ("V (eigvec)", z_v), ("orig (≈V3)", z_orig)]:
        z_top1 = z[top1_idx].abs()  # (Q, d)
        z_rand = z[rand_idx].abs()
        m_t1, M_t1 = z_top1.mean(), z_top1.max(dim=-1).values.mean()
        m_r, M_r = z_rand.mean(), z_rand.max(dim=-1).values.mean()
        print(f"{name:<15} {m_t1.item():>14.3f} {M_t1.item():>14.3f} {m_r.item():>14.3f} {M_r.item():>14.3f}  {(M_t1/M_r).item():>16.2f}")

    # Distribution of "where is the largest coord value of the top-1 key"?
    # Specifically, for the top-1 key in each query, find the coord with max |z|. Is it the
    # high-rho coord (where CCA spends bits) or low-rho?
    print()
    print("Distribution of which coord is most extreme for top-1 keys (CCA-canonical, sorted by rho desc):")
    z_cca_top1 = z_cca[top1_idx].abs()
    argmax_coord = z_cca_top1.argmax(dim=-1)  # which coord is the max for each top-1 key
    # Histogram by coord index (CCA basis is sorted by rho desc by construction in compute_cca_basis)
    bins = [0, 8, 16, 32, 64, 96, 128]
    counts = []
    for i in range(len(bins) - 1):
        c = ((argmax_coord >= bins[i]) & (argmax_coord < bins[i+1])).sum().item()
        counts.append(c)
        print(f"  coord [{bins[i]:>3}, {bins[i+1]:>3}): {c}/{len(argmax_coord)}")

    # Same for V basis (sorted by lambda desc) and random
    print()
    print("Same for V basis (sorted by lambda desc):")
    argmax_coord_v = z_v[top1_idx].abs().argmax(dim=-1)
    for i in range(len(bins) - 1):
        c = ((argmax_coord_v >= bins[i]) & (argmax_coord_v < bins[i+1])).sum().item()
        print(f"  coord [{bins[i]:>3}, {bins[i+1]:>3}): {c}/{len(argmax_coord_v)}")
    print()
    print("Same for orig (random) basis:")
    argmax_coord_o = z_orig[top1_idx].abs().argmax(dim=-1)
    for i in range(len(bins) - 1):
        c = ((argmax_coord_o >= bins[i]) & (argmax_coord_o < bins[i+1])).sum().item()
        print(f"  coord [{bins[i]:>3}, {bins[i+1]:>3}): {c}/{len(argmax_coord_o)}")

    # CHART: histograms of |z| for top-1 vs random across each basis
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ai, (name, z) in enumerate([("CCA-canonical (sorted by ρ)", z_cca),
                                     ("V eigenbasis (sorted by λ)", z_v),
                                     ("original (≈V3 input)", z_orig)]):
        z_top1 = z[top1_idx].abs().flatten().numpy()
        z_rand = z[rand_idx].abs().flatten().numpy()
        axes[ai].hist(z_rand, bins=80, density=True, alpha=0.5, color="#999", label="random key")
        axes[ai].hist(z_top1, bins=80, density=True, alpha=0.5, color="#cc3322", label="top-1 key")
        axes[ai].set_xlabel("|z-score| of coord value")
        axes[ai].set_ylabel("density")
        axes[ai].set_title(f"{name}\nmean |z|: top1={z_top1.mean():.3f}, rand={z_rand.mean():.3f}")
        axes[ai].legend()
        axes[ai].grid(alpha=0.3)
        axes[ai].set_xlim(0, 5)
    fig.suptitle(f"Top-1 keys are more extreme than random keys in their basis-aligned coordinates  "
                 f"(layer={layer}, head={kv_head})")
    fig.tight_layout()
    fig.savefig(OUT / "top1_extremeness_per_basis.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {OUT}/top1_extremeness_per_basis.png")


if __name__ == "__main__":
    main()
