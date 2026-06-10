#!/usr/bin/env python3
"""Per-(L,H,coord) near-tie-weighted query leverage, in the QPCA centered basis.

The QPCA allocation score uses q_diag_j = f_j^T Σ_Q f_j  (variance of the
forward-projected query along code axis j) — NOT the decoder leverage (which is
≈1 on every coord because the QPCA decoder is M_q-orthonormal).

This script measures the SAME forward-projected query energy, but conditioned on
near-tie queries (those whose top-1/top-2 logits are close — the queries that
actually decide top-1). If the empirical exponent q_diag^α is a stand-in for this
conditional leverage, then q_diag^α should track cond_leverage for some α<1.

Output: tiny (L,H,d) tensors. Fit and score both on `small` (leaked, directional).
"""
import json
import torch
from pathlib import Path

EPS = 1e-4


def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))


def build_qpca_basis(sigma_q, sigma_k):
    """forward map F (L,H,d,d) of centered-basis QPCA; r = (k-mu) @ F."""
    sq = _sym(sigma_q.double())
    sk = _sym(sigma_k.double())
    ev, U = torch.linalg.eigh(sq)
    sqrt_mq = U @ torch.diag_embed(ev.clamp_min(1e-30).sqrt()) @ U.transpose(-1, -2)
    A = _sym(sqrt_mq @ sk @ sqrt_mq)
    lam, V = torch.linalg.eigh(A)
    order = torch.argsort(lam, dim=-1, descending=True)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand(*V.shape[:-1], -1))
    return sqrt_mq @ V


def main(data_dir, dirname, out_path, tau_frac=1.0):
    root = Path(data_dir) / dirname
    manifest = json.loads((root / "manifest.json").read_text())
    pooled = torch.load(root / "pooled_stats.pt", map_location="cpu", weights_only=False)
    q2, k2 = pooled["q_post"][2], pooled["k_post"][2]
    k_mean, k_cov = pooled["k_post"][0], pooled["k_post"][1]
    L, Hq, d, _ = q2.shape
    _, Hkv, _, _ = k2.shape
    gs = Hq // Hkv
    sigma_q = q2.reshape(L, Hkv, gs, d, d).sum(2)
    F = build_qpca_basis(sigma_q, k_cov)            # (L,H,d,d) forward map

    cond = torch.zeros(L, Hkv, d, dtype=torch.float64)   # near-tie-weighted query energy
    wsum = torch.zeros(L, Hkv, dtype=torch.float64)
    uncond = torch.zeros(L, Hkv, d, dtype=torch.float64) # plain mean query energy (≈ q_diag)
    ncnt = torch.zeros(L, Hkv, dtype=torch.float64)

    for e in manifest["examples"]:
        art = torch.load(root / e["file"], map_location="cpu", weights_only=False)
        T = int(art["prompt_length"])
        for l in range(L):
            for h in range(Hkv):
                Fh = F[l, h]                                     # (d,d)
                k = art["k_post"][l, h, :T].double()
                q = art["q_post"][l, h * gs:(h + 1) * gs, :T].double().reshape(-1, d)
                logits = q @ k.T                                 # (Nq, T)
                top2 = logits.topk(2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1]).clamp_min(0)  # (Nq,)
                tau = tau_frac * margin.median().clamp_min(1e-9)
                w = torch.exp(-0.5 * (margin / tau) ** 2)        # near-tie weight
                qc = q @ Fh                                      # forward-projected query (Nq,d)
                lev = qc ** 2                                    # per-coord query energy
                cond[l, h] += (w[:, None] * lev).sum(0)
                wsum[l, h] += w.sum()
                uncond[l, h] += lev.sum(0)
                ncnt[l, h] += lev.shape[0]

    cond = cond / wsum[..., None].clamp_min(1e-30)
    uncond = uncond / ncnt[..., None].clamp_min(1)

    # sanity: uncond should match q_diag = diag(F^T Σ_Q F) up to the mean/2nd-moment gap
    fwdt = F.transpose(-1, -2)
    q_diag = (fwdt @ _sym(sigma_q.double()) @ F).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    sl = slice(1, L)
    print("sanity uncond vs q_diag (layer 1, head 0, first 5 coords):")
    print("  uncond :", [round(x, 2) for x in uncond[1, 0, :5].tolist()])
    print("  q_diag :", [round(x, 2) for x in q_diag[1, 0, :5].tolist()])

    def dr(x):
        x = x.clamp_min(1e-30)
        return float((x.max() / x.min()).log10())
    print(f"\ndynamic range log10(max/min): conditional={dr(cond[sl]):.2f}  "
          f"unconditional={dr(uncond[sl]):.2f}  q_diag={dr(q_diag[sl]):.2f}")

    print("\nlog-corr of q_diag^alpha with conditional leverage (layer 1+):")
    a = cond[sl].reshape(-1).clamp_min(1e-30).log()
    am = a - a.mean()
    for alpha in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        b = q_diag[sl].reshape(-1).clamp_min(1e-30).pow(alpha).log()
        bm = b - b.mean()
        corr = float((am * bm).sum() / (am.norm() * bm.norm() + 1e-30))
        print(f"  q_diag^{alpha}: log-corr = {corr:+.3f}")

    print("\nlog-corr of cond/uncond ratio with coord index (is conditioning systematic?):")
    ratio = (cond[sl] / uncond[sl].clamp_min(1e-30)).reshape(-1)
    print(f"  median cond/uncond ratio = {ratio.median():.3f}  "
          f"(if <1 on loud coords, near-tie queries de-emphasize them)")

    torch.save({"cond_leverage": cond.float(), "uncond_leverage": uncond.float(),
                "q_diag": q_diag.float(), "basis": "qpca_centered",
                "n_examples": len(manifest["examples"])}, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data",
         "query_stats_longbench_under4k_small", "leverage_stats.pt")