#!/usr/bin/env python3
"""Cheap proxy sweep over VQ *config* (grouping/alloc/whiten/ecvq) on a handful of
(layer,head): held-out attention top-1/top-5 + relMSE. Decides which codebook design
to full-train + F1, without spending F1 runs. Mirrors train_group_vq_alloc's config
transforms exactly so the proxy winner transfers to the full trainer."""
import argparse, torch
import run_pca_ec_deadzone as base
from group_vq_codec import _kmeans, group_boundaries, GroupVQCompressor
from train_group_vq_alloc import stratified_perm, group_bit_alloc

ap = argparse.ArgumentParser()
ap.add_argument("--basis-moments", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--code-idx", type=int, nargs="+", default=list(range(80)))
ap.add_argument("--layers", type=int, nargs="+", default=[3, 10, 18, 24, 30])
ap.add_argument("--eval-idx", type=int, nargs="+", default=[20, 21, 22])
ap.add_argument("--G", type=int, default=4)
ap.add_argument("--max-k-bits", type=int, default=13)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

B = torch.load(args.basis_moments, map_location="cpu", weights_only=False)
sq, sk, km, kc, meta = B["sigma_q"], B["sigma_k"], B["k_mean"], B["k_cov"], B["meta"]
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qc = base.build_qpca_basis(sq, kc)
F0, inv0, score0, std0 = qc["forward"].clone(), qc["inverse"].clone(), qc["score"].clone(), qc["std"].clone()
bounds0 = group_boundaries(d, args.G)

from pathlib import Path
troot = Path(args.data_root); tman = base.load_manifest(troot)

# held-out q/k
proot = base.data_root(); pman = base.load_manifest(proot)
evals = []
for i in args.eval_idx:
    a = torch.load(proot / pman["examples"][i]["file"], map_location="cpu", weights_only=False)
    T = int(a["prompt_length"]); evals.append((a["q_post"][:, :, :T, :].float(), a["k_post"][:, :, :T, :].float(), T))
gs = evals[0][0].shape[1] // Hkv


def build_cfg(grouping, alloc, whiten):
    F, inv, score, std = F0.clone(), inv0.clone(), score0.clone(), std0.clone()
    if grouping == "stratified":
        perm = stratified_perm(d, args.G)
        F = F[:, :, :, perm]; inv = inv[:, :, perm, :]; score = score[:, :, perm]; std = std[:, :, perm]
    if whiten:
        F = F / std.unsqueeze(-2).clamp_min(1e-8); inv = inv * std.unsqueeze(-1); score = torch.ones_like(score)
    return F, inv, score


def run(name, grouping, alloc, whiten, ecvq_lambda):
    F, inv, score = build_cfg(grouping, alloc, whiten)
    fetch = base._codes_for_idx(troot, tman, args.code_idx, F, km, L, Hkv, d)
    t1 = t5 = n = 0; mse = 0.0; mn = 0
    for l in args.layers:
        h = 0
        r = fetch(l, h).to(dev)
        if alloc == "waterfill":
            gbits = group_bit_alloc(score[l, h], bounds0, avg_bits=2, max_k_bits=args.max_k_bits)
        else:
            gbits = [b for (_, _, b) in bounds0]
        cbs = []
        for (s, e, _), gb in zip(bounds0, gbits):
            K = 1 << gb
            cbs.append(r[:, s:e].mean(0, keepdim=True) if K <= 1 else
                       _kmeans(r[:, s:e], K, iters=25, seed=l * Hkv + h, ecvq_lambda=ecvq_lambda))
        comp = GroupVQCompressor(F[l, h].to(dev), inv[l, h].to(dev), km[l, h].to(dev), cbs, bounds0)
        for q, k, T in evals:
            kh = k[l, h].to(dev); qh = q[l, h * gs:(h + 1) * gs].to(dev).reshape(-1, d)
            logits = qh @ kh.t()
            real1 = logits.argmax(-1); real5 = logits.topk(min(5, T), -1).indices
            khat = comp.roundtrip(kh); pred = (qh @ khat.t()).argmax(-1)
            t1 += (pred == real1).sum().item(); t5 += (pred.unsqueeze(-1) == real5).any(-1).sum().item(); n += real1.numel()
            mse += (khat - kh).pow(2).mean().item() / max(kh.var().item(), 1e-9); mn += 1
    print(f"{name:>34} {t1/n:>8.4f} {t5/n:>8.4f} {mse/mn:>9.4f}", flush=True)


print(f"layers={args.layers} eval={args.eval_idx} G={args.G}")
print(f"{'config':>34} {'top1':>8} {'top5':>8} {'relMSE':>9}")
run("strat/flat/kmeans (BASELINE)", "stratified", "flat", False, 0.0)
run("strat/flat/whiten/kmeans", "stratified", "flat", True, 0.0)
run("strat/waterfill/kmeans", "stratified", "waterfill", False, 0.0)
run("strat/waterfill/whiten/kmeans", "stratified", "waterfill", True, 0.0)
run("strat/flat/whiten/ecvq0.3", "stratified", "flat", True, 0.3)
run("strat/waterfill/whiten/ecvq0.3", "stratified", "waterfill", True, 0.3)
