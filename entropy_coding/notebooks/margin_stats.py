#!/usr/bin/env python3
"""Dump per-(L,H,coord) winner-vs-runnerup margin separation S_j, in the QPCA
centered basis, from a calibration dataset. Tiny output (L,H,d) — no raw keys.
Use this S_j to build a margin-aware bit allocation; fit on calibration, score
on a disjoint eval set (no leak)."""
import json, torch
from pathlib import Path

EPS = 1e-4

def _sym(x): return 0.5 * (x + x.transpose(-1, -2))

def regularize_batch(cov, eps):
    d = cov.shape[-1]; sym = _sym(cov)
    tr = sym.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(d, dtype=sym.dtype)
    return sym + (eps * (tr/d).clamp_min(1e-12)).unsqueeze(-1).unsqueeze(-1) * eye

def build_qpca_basis(sigma_q, sigma_k):
    sq = _sym(sigma_q.double()); sk = _sym(sigma_k.double())
    ev, U = torch.linalg.eigh(sq)
    sqrt_mq = U @ torch.diag_embed(ev.clamp_min(1e-30).sqrt()) @ U.transpose(-1,-2)
    A = _sym(sqrt_mq @ sk @ sqrt_mq)
    lam, V = torch.linalg.eigh(A)
    order = torch.argsort(lam, dim=-1, descending=True)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand(*V.shape[:-1], -1))
    return (sqrt_mq @ V)            # forward map (L,H,d,d)

def main(data_dir, dirname, out_path):
    root = Path(data_dir) / dirname
    manifest = json.loads((root / "manifest.json").read_text())
    pooled = torch.load(root / "pooled_stats.pt", map_location="cpu", weights_only=False)
    q2 = pooled["q_post"][2]; k2 = pooled["k_post"][2]
    k_mean, k_cov = pooled["k_post"][0], pooled["k_post"][1]
    L, Hq, d, _ = q2.shape; _, Hkv, _, _ = k2.shape; gs = Hq // Hkv
    sigma_q = q2.reshape(L, Hkv, gs, d, d).sum(2)
    F = build_qpca_basis(sigma_q, k_cov)          # CENTERED-basis QPCA (matches your quantizer)

    S = torch.zeros(L, Hkv, d, dtype=torch.float64)   # accumulated (r_win - r_run)^2
    cnt = torch.zeros(L, Hkv, dtype=torch.float64)
    for e in manifest["examples"]:
        art = torch.load(root / e["file"], map_location="cpu", weights_only=False)
        T = int(art["prompt_length"])
        for l in range(L):
            for h in range(Hkv):
                k = art["k_post"][l, h, :T].double()
                q = art["q_post"][l, h*gs:(h+1)*gs, :T].double().reshape(-1, d)
                r = (k - k_mean[l, h].double()) @ F[l, h]       # centered codes (T,d)
                logits = q @ k.T
                top2 = logits.topk(2, dim=-1).indices
                win, run = top2[:, 0], top2[:, 1]
                S[l, h] += ((r[win] - r[run]) ** 2).sum(0)
                cnt[l, h] += win.numel()
    S = S / cnt[..., None].clamp_min(1)
    torch.save({"margin_sep": S.float(), "basis": "qpca_centered",
                "dirname": dirname, "n_examples": len(manifest["examples"])}, out_path)
    print(f"wrote {out_path}: margin_sep {tuple(S.shape)} from {len(manifest['examples'])} examples")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data",
         "query_stats_longbench_under4k_small",     # <- small, not full
         "margin_stats.pt")