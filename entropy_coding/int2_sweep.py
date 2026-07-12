#!/usr/bin/env python3
"""Sweep INT2 / QPCA-Fixed knobs (dz, load, coord-0 widen) and score each on the
attention top-1/top-5 + K-relMSE proxy over held-out (q,k). Basis-only builds + a
reconstruction pass, so no model and (optionally) no GPU. Prints configs sorted by
top-1 so the best can be rebuilt + F1'd."""
import argparse, itertools, torch
import run_pca_ec_deadzone as base

ap = argparse.ArgumentParser()
ap.add_argument("--basis-moments", default="/vault/samuel/data/basis_moments_qwen3_8b_compact8train400/basis_moments.pt")
ap.add_argument("--eval-idx", type=int, nargs="+", default=[20, 21, 22, 23])
ap.add_argument("--dz", type=float, nargs="+", default=[0.25, 0.375, 0.5])
ap.add_argument("--load", type=float, nargs="+", default=[2.5, 3.0, 3.5, 4.0])
ap.add_argument("--widen", type=float, nargs="+", default=[1.0, 2.5])
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

root = base.data_root(); man = base.load_manifest(root)
B = torch.load(args.basis_moments, map_location="cpu", weights_only=False)
sq, sk, km, meta = B["sigma_q"], B["sigma_k"], B["k_mean"], B["meta"]
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
gs = meta["group_size"]
qpca_unc = base.build_qpca_basis(sq, sk)

# preload eval (q,k)
evals = []
for i in args.eval_idx:
    art = torch.load(root / man["examples"][i]["file"], map_location="cpu", weights_only=False)
    T = int(art["prompt_length"])
    evals.append((art["q_post"][:, :, :T, :].float(), art["k_post"][:, :, :T, :].float()))

def score(comps):
    t1 = t5 = n = 0; mse = 0.0; mn = 0
    for q, k in evals:
        for l in range(1, L):
            for h in range(Hkv):
                kh = k[l, h].to(dev); qh = q[l, h*gs:(h+1)*gs].to(dev).reshape(-1, d)
                lg = qh @ kh.t(); real = lg.argmax(-1); top5 = lg.topk(min(5, kh.shape[0]), -1).indices
                khat = comps[(l, h)].to(dev).roundtrip(kh)
                pr = (qh @ khat.t()).argmax(-1)
                t1 += (pr == real).sum().item(); t5 += (pr.unsqueeze(-1) == top5).any(-1).sum().item()
                n += real.numel(); mse += (khat - kh).pow(2).mean().item() / max(kh.var().item(), 1e-9); mn += 1
    return t1 / n, t5 / n, mse / mn

rows = []
for dz, load, widen in itertools.product(args.dz, args.load, args.widen):
    comps = base.build_qpca_fixed_deadzone(qpca_unc, km, b=2, n_layers=L, n_kv_heads=Hkv,
                                           dz=dz, load=load, widen_mult=widen)
    a1, a5, m = score(comps)
    rows.append((a1, a5, m, dz, load, widen))
    print(f"dz={dz:<5} load={load:<4} widen={widen:<4} | top1={a1:.4f} top5={a5:.4f} relMSE={m:.4f}", flush=True)

rows.sort(reverse=True)
print("\n=== best by top-1 ===")
for a1, a5, m, dz, load, widen in rows[:5]:
    print(f"top1={a1:.4f} top5={a5:.4f} relMSE={m:.4f}  <- dz={dz} load={load} widen={widen}")
print(f"\ncurrent default: dz=0.375 load=3.0 widen=2.5")
