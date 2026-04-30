"""Hypothesis refinement: CCA's quantization noise is preferentially large on the
top-1 key, not random keys. Because the CCA basis aligns each key's
high-ρ coordinates with directions that distinguish it from other keys.

Measurement: for each query q,
   - delta_top1 = (q^T k - q^T k_hat) at the true top-1 key
   - delta_rand = (q^T k - q^T k_hat) at a random key
   - delta_runup = same at the true runner-up (top-2)
   - And the |q^T k| value at each of these (the "signal").

If CCA's noise is tied to the signal (bigger noise on bigger logit), then
|delta_top1| / |delta_rand| should be large for CCA but ~1 for V3.

Bonus: also see whether the noise correlates positively with the signal (which
would explain why CCA pulls top-1 keys *away* from their position).
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

from experiments.stage1.toolkit.per_coord_quantization import build_method_compressor
from experiments.stage1.toolkit.quantization import Stage1MSECompressor


REPO = Path("/vault/amir/efficient-llm/teamily-project")
CCA_STATS = REPO / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
BUNDLE = REPO / "artifacts/stage1/query_stats_longbench_under4k"
OUT = REPO / "artifacts/stage1/cca_vs_waterfill_study/diagnostics_v3_vs_cca"
OUT.mkdir(parents=True, exist_ok=True)


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

    v3 = Stage1MSECompressor(d, int(b_avg), seed=0, device="cpu")
    v_wf = build_method_compressor("v_waterfill", sigma_q, sigma_k, None, mq_eigvals, mq_eigvecs, b_avg, d, 0, d)
    cca_wf = build_method_compressor("cca_waterfill", sigma_q, sigma_k,
                                     {"P_K": P_K, "P_K_inv": P_K_inv, "rho": rho},
                                     mq_eigvals, mq_eigvecs, b_avg, d, 0, d)

    keys_3d = k_pre.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        k_v3 = v3.roundtrip(keys_3d).squeeze().float()
        k_v_wf = v_wf.roundtrip(keys_3d).squeeze().float()
        k_cca_wf = cca_wf.roundtrip(keys_3d).squeeze().float()

    q_flat = q_group.reshape(-1, d)[:200]
    Q = q_flat.shape[0]
    print(f"Using {Q} queries, {L} keys, head_dim={d}")

    real_logits = torch.einsum("qd,ld->ql", q_flat, k_pre) / math.sqrt(d)
    sorted_idx = real_logits.argsort(dim=-1, descending=True)
    top1_idx = sorted_idx[:, 0]
    runup_idx = sorted_idx[:, 1]
    rng = torch.Generator().manual_seed(42)
    rand_idx = torch.randint(0, L, (Q,), generator=rng)
    median_idx = sorted_idx[:, L // 2]  # the median-logit key

    print()
    print("|delta| = |q^T(k - k̂)/sqrt(d)| for keys at different logit ranks:")
    print(f"{'method':<15} {'top1':>8} {'runner-up':>10} {'median':>8} {'random':>8} | "
          f"{'top1/rand':>10} {'top1/median':>12}")
    print("-" * 90)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ai, (name, k_recon) in enumerate([
        ("v3", k_v3), ("v_waterfill", k_v_wf), ("cca_waterfill", k_cca_wf)
    ]):
        approx_logits = torch.einsum("qd,ld->ql", q_flat, k_recon) / math.sqrt(d)
        delta = approx_logits - real_logits  # (Q, L)
        d_top1 = torch.gather(delta, -1, top1_idx.unsqueeze(-1)).squeeze(-1).abs()
        d_runup = torch.gather(delta, -1, runup_idx.unsqueeze(-1)).squeeze(-1).abs()
        d_median = torch.gather(delta, -1, median_idx.unsqueeze(-1)).squeeze(-1).abs()
        d_rand = torch.gather(delta, -1, rand_idx.unsqueeze(-1)).squeeze(-1).abs()
        m_top1, m_runup, m_med, m_rand = d_top1.mean(), d_runup.mean(), d_median.mean(), d_rand.mean()
        print(f"{name:<15} {m_top1.item():>8.4f} {m_runup.item():>10.4f} {m_med.item():>8.4f} {m_rand.item():>8.4f} | "
              f"{(m_top1/m_rand).item():>10.2f}x {(m_top1/m_med).item():>12.2f}x")

        # Scatter |delta| vs |signal| per key (averaged over queries) — does noise correlate with logit magnitude?
        ax = axes[ai]
        # Sample 5000 (q, k) pairs randomly
        sub_q = torch.randint(0, Q, (5000,), generator=rng)
        sub_k = torch.randint(0, L, (5000,), generator=rng)
        x = (real_logits[sub_q, sub_k]).abs().numpy()
        y = delta[sub_q, sub_k].abs().numpy()
        ax.scatter(x, y, s=2, alpha=0.3, color={"v3":"#777","v_waterfill":"#228833","cca_waterfill":"#4477AA"}[name])
        # Fit linear regression for trend
        if x.std() > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(0, x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color="red", linewidth=1.5, label=f"slope={slope:.3f}")
            corr = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"{name}\ncorr(|signal|, |error|) = {corr:.3f}")
        else:
            ax.set_title(name)
        ax.set_xlabel("|q^T k / sqrt(d)|  (true logit magnitude)")
        ax.set_ylabel("|q^T (k − k̂) / sqrt(d)|  (error)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(0, max(2, x.max() * 1.05))

    fig.suptitle(f"Per-(query, key) error magnitude vs true-logit magnitude  (layer={layer}, head={kv_head}, b={b_avg})")
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_signal_correlation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {OUT}/error_vs_signal_correlation.png")

    # Also: does CCA's noise have the SIGN systematically pulling logits down?
    # i.e., E[delta | top1 key] vs E[delta | random key]?
    print()
    print("Mean SIGNED delta (positive = method overestimates the logit):")
    print(f"{'method':<15} {'top1':>10} {'runner-up':>10} {'median':>10} {'random':>10}")
    for name, k_recon in [("v3", k_v3), ("v_waterfill", k_v_wf), ("cca_waterfill", k_cca_wf)]:
        approx_logits = torch.einsum("qd,ld->ql", q_flat, k_recon) / math.sqrt(d)
        delta = approx_logits - real_logits
        s_top1 = torch.gather(delta, -1, top1_idx.unsqueeze(-1)).squeeze(-1).mean()
        s_runup = torch.gather(delta, -1, runup_idx.unsqueeze(-1)).squeeze(-1).mean()
        s_med = torch.gather(delta, -1, median_idx.unsqueeze(-1)).squeeze(-1).mean()
        s_rand = torch.gather(delta, -1, rand_idx.unsqueeze(-1)).squeeze(-1).mean()
        print(f"{name:<15} {s_top1.item():>10.4f} {s_runup.item():>10.4f} {s_med.item():>10.4f} {s_rand.item():>10.4f}")


if __name__ == "__main__":
    main()
