#!/usr/bin/env python3
"""Cheap ECVQ lambda sweep on a few (layer,head): confirm the index ENTROPY drops with
lambda (ECVQ is doing something) and measure the held-out fidelity proxy (top-1/relMSE).
Fixed K=256, G=4 stratified flat -- the milestone deploy config. No F1, no full grid."""
import argparse, math, torch
import run_pca_ec_deadzone as base
from group_vq_codec import _kmeans, group_boundaries, GroupVQCompressor
from train_group_vq_alloc import stratified_perm

ap = argparse.ArgumentParser()
ap.add_argument("--basis-moments", required=True)
ap.add_argument("--data-root", required=True, help="training code pool (compact8train80)")
ap.add_argument("--code-idx", type=int, nargs="+", default=list(range(80)))
ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
ap.add_argument("--heads", type=int, nargs="+", default=[1, 8, 16, 24, 30],
                help="layer indices to test (head 0 each)")
ap.add_argument("--eval-idx", type=int, nargs="+", default=[20, 21], help="held-out q/k (proxy pool)")
ap.add_argument("--G", type=int, default=4)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

B = torch.load(args.basis_moments, map_location="cpu", weights_only=False)
sq, sk, km, kc, meta = B["sigma_q"], B["sigma_k"], B["k_mean"], B["k_cov"], B["meta"]
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"].clone(), qc["inverse"].clone()
perm = stratified_perm(d, args.G)
F = F[:, :, :, perm]; inv = inv[:, :, perm, :]
bounds = group_boundaries(d, args.G)
G = args.G; K = 1 << (2 * G)

from pathlib import Path
troot = Path(args.data_root); tman = base.load_manifest(troot)
fetch = base._codes_for_idx(troot, tman, args.code_idx, F, km, L, Hkv, d)

# held-out q/k from the proxy pool (has q_post+k_post)
proot = base.data_root(); pman = base.load_manifest(proot)
evals = []
for i in args.eval_idx:
    a = torch.load(proot / pman["examples"][i]["file"], map_location="cpu", weights_only=False)
    T = int(a["prompt_length"]); evals.append((a["q_post"][:, :, :T, :].float(), a["k_post"][:, :, :T, :].float(), T))
gs = evals[0][0].shape[1] // Hkv


def index_entropy(seg, cb):
    idx = torch.cdist(seg, cb).argmin(1)
    c = torch.bincount(idx, minlength=cb.shape[0]).float(); p = c / c.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum()), int((c > 0).sum())


print(f"K={K} G={G} | heads(layers)={args.heads} | eval={args.eval_idx}")
print(f"{'lambda':>7} {'idxEntropy':>11} {'usedK':>7} {'top1':>8} {'top5':>8} {'relMSE':>9}")
for lam in args.lambdas:
    ent_acc = []; usedk = []; t1 = t5 = n = 0; mse = 0.0; mn = 0
    for l in args.heads:
        h = 0
        r = fetch(l, h).to(dev)
        cbs = []
        for (s, e, _) in bounds:
            cb = _kmeans(r[:, s:e], K, iters=25, seed=l * Hkv + h, ecvq_lambda=lam)
            cbs.append(cb)
            en, uk = index_entropy(r[:, s:e], cb); ent_acc.append(en); usedk.append(uk)
        comp = GroupVQCompressor(F[l, h].to(dev), inv[l, h].to(dev), km[l, h].to(dev), cbs, bounds)
        for q, k, T in evals:
            kh = k[l, h].to(dev); qh = q[l, h*gs:(h+1)*gs].to(dev).reshape(-1, d)
            real = (qh @ kh.t()).argmax(-1); top5 = (qh @ kh.t()).topk(min(5, T), -1).indices
            khat = comp.roundtrip(kh); lg = qh @ khat.t(); pred = lg.argmax(-1)
            t1 += (pred == real).sum().item(); t5 += (pred.unsqueeze(-1) == top5).any(-1).sum().item(); n += real.numel()
            mse += (khat - kh).pow(2).mean().item() / max(kh.var().item(), 1e-9); mn += 1
    print(f"{lam:>7.2f} {sum(ent_acc)/len(ent_acc):>11.3f} {sum(usedk)/len(usedk):>7.1f} "
          f"{t1/n:>8.4f} {t5/n:>8.4f} {mse/mn:>9.4f}", flush=True)
