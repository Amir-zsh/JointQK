#!/usr/bin/env python3
"""Diagnose the EC underperformance: is it the CENTERED vs UNCENTERED QPCA basis?
Compares, on held-out real (q,k), the top-1/top-5 attention retention + Q-weighted
K-MSE of: scalar (QPCA-Fixed, uncentered), EC-centered (current bundle basis),
EC-uncentered (the study's ec_qpca_unc basis). Layer-0 excluded. CPU-friendly.
"""
import argparse, torch
import run_pca_ec_deadzone as base

ap = argparse.ArgumentParser()
ap.add_argument("--basis-moments", default="/vault/samuel/data/basis_moments_qwen3_8b_compact8train400/basis_moments.pt")
ap.add_argument("--fit-idx", type=int, nargs="+", default=list(range(20)))   # EC delta fit
ap.add_argument("--eval-idx", type=int, nargs="+", default=[20, 21, 22, 23]) # held-out
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

root = base.data_root(); man = base.load_manifest(root)
B = torch.load(args.basis_moments, map_location="cpu", weights_only=False)
sq, sk, km, kc, meta = B["sigma_q"], B["sigma_k"], B["k_mean"], B["k_cov"], B["meta"]
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
gs = meta["group_size"]

# --- three K-compressors, all @ b=2 ---
qpca_unc = base.build_qpca_basis(sq, sk)
scalar = base.build_qpca_fixed_deadzone(qpca_unc, km, b=2, n_layers=L, n_kv_heads=Hkv, dz=0.375, load=3.0)

def build_ec(basis_cov, tag):
    qc = base.build_qpca_basis(sq, basis_cov); qc["sigma_k"] = sk
    qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
    F, inv = qc["forward"], qc["inverse"]
    fc = base._codes_for_idx(root, man, args.fit_idx, F, km, L, Hkv, d)
    _, delta0, _ = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, args.fit_idx,
                                      dz=0.375, match_rate=False, uniform_step=True)
    return {(l, h): base.UniformECRoundtrip(F[l, h], inv[l, h], km[l, h], delta0[l, h], dz=0.375)
            for l in range(L) for h in range(Hkv)}

ec_cen = build_ec(kc, "centered")     # current bundle basis
ec_unc = build_ec(sk, "uncentered")   # study's ec_qpca_unc basis

methods = {"scalar_unc": scalar, "EC_centered": ec_cen, "EC_uncentered": ec_unc}
acc = {m: {"t1": 0, "t5": 0, "n": 0, "mse": 0.0, "msen": 0} for m in methods}

for i in args.eval_idx:
    art = torch.load(root / man["examples"][i]["file"], map_location="cpu", weights_only=False)
    T = int(art["prompt_length"])
    q = art["q_post"][:, :, :T, :].float()   # (L,Hq,T,d)
    k = art["k_post"][:, :, :T, :].float()   # (L,Hkv,T,d)
    for l in range(1, L):                     # layer-0 excluded
        for h in range(Hkv):
            kh = k[l, h].to(dev)              # (T,d)
            qh = q[l, h*gs:(h+1)*gs].to(dev).reshape(-1, d)  # (gs*T? no) -> pooled group queries
            # real attention argmax (per query) vs reconstructed
            logits = qh @ kh.t()             # (Nq, T)
            real = logits.argmax(-1)
            top5 = logits.topk(min(5, T), dim=-1).indices
            for m, comp in methods.items():
                c = comp[(l, h)].to(dev)
                khat = c.roundtrip(kh)
                lg = qh @ khat.t()
                pred = lg.argmax(-1)
                acc[m]["t1"] += (pred == real).sum().item()
                acc[m]["t5"] += (pred.unsqueeze(-1) == top5).any(-1).sum().item()
                acc[m]["n"] += real.numel()
                # Q-weighted recon MSE proxy: mean squared K error
                acc[m]["mse"] += (khat - kh).pow(2).mean().item(); acc[m]["msen"] += 1

print(f"\nEval prompts {args.eval_idx} | fit {len(args.fit_idx)} prompts | layer-0 excluded")
print(f"{'method':<16} {'top1':>8} {'top5':>8} {'K-MSE':>10}")
for m in methods:
    a = acc[m]
    print(f"{m:<16} {a['t1']/a['n']:8.4f} {a['t5']/a['n']:8.4f} {a['mse']/a['msen']:10.4e}")
