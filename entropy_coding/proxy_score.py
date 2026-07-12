#!/usr/bin/env python3
"""Fidelity proxy for the K-compressors: attention top-1 / top-5 retention and
Q-weighted K reconstruction MSE on held-out (q,k), layer-0 excluded. Scores the
KBundle-format bundles (scalar/oscar/ec) and a VQ codebook, all reconstruction-only
so no model is loaded. Eval data = the query_stats pool prompts (have q_post+k_post).
"""
import argparse, torch
import run_pca_ec_deadzone as base
import oscar_codec  # noqa: F401 (unpickle)
from group_vq_codec import GroupVQCompressor

ap = argparse.ArgumentParser()
ap.add_argument("--eval-idx", type=int, nargs="+", default=[20, 21, 22, 23])
ap.add_argument("--bundles", nargs="*", default=[], help="label=path.pt for KBundle dicts")
ap.add_argument("--vq", nargs="*", default=[], help="label=path.pt for VQ codebooks")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
root = base.data_root(); man = base.load_manifest(root)

methods = {}
for spec in args.bundles:
    lab, path = spec.split("=", 1)
    methods[lab] = torch.load(path, map_location="cpu", weights_only=False)["comps"]
for spec in args.vq:
    lab, path = spec.split("=", 1)
    p = torch.load(path, map_location="cpu", weights_only=False)
    F, inv, mean, bounds, cbs = p["forward"], p["inverse"], p["mean"], p["bounds"], p["codebooks"]
    L, Hkv = F.shape[0], F.shape[1]
    comps = {}
    for l in range(L):
        for h in range(Hkv):
            comps[(l, h)] = GroupVQCompressor(F[l, h], inv[l, h], mean[l, h], list(cbs[(l, h)]), bounds)
    methods[lab] = comps

acc = {m: {"t1": 0, "t5": 0, "n": 0, "mse": 0.0, "mn": 0} for m in methods}
first = torch.load(root / man["examples"][args.eval_idx[0]]["file"], map_location="cpu", weights_only=False)
L, Hq, _, d = first["q_post"].shape
Hkv = first["k_post"].shape[1]; gs = Hq // Hkv

for i in args.eval_idx:
    art = torch.load(root / man["examples"][i]["file"], map_location="cpu", weights_only=False)
    T = int(art["prompt_length"])
    q = art["q_post"][:, :, :T, :].float(); k = art["k_post"][:, :, :T, :].float()
    for l in range(1, L):
        for h in range(Hkv):
            kh = k[l, h].to(dev)
            qh = q[l, h*gs:(h+1)*gs].to(dev).reshape(-1, d)
            logits = qh @ kh.t()
            real = logits.argmax(-1)
            top5 = logits.topk(min(5, T), dim=-1).indices
            kvar = kh.var().item()
            for m, comp in methods.items():
                c = comp[(l, h)]
                if hasattr(c, "to"):
                    c = c.to(dev)
                khat = c.roundtrip(kh)
                lg = qh @ khat.t()
                pred = lg.argmax(-1)
                acc[m]["t1"] += (pred == real).sum().item()
                acc[m]["t5"] += (pred.unsqueeze(-1) == top5).any(-1).sum().item()
                acc[m]["n"] += real.numel()
                acc[m]["mse"] += (khat - kh).pow(2).mean().item() / max(kvar, 1e-9)
                acc[m]["mn"] += 1

print(f"\nEval {args.eval_idx} | layer-0 excluded")
print(f"{'method':<18} {'top1':>8} {'top5':>8} {'relMSE':>9}")
for m in methods:
    a = acc[m]
    print(f"{m:<18} {a['t1']/a['n']:8.4f} {a['t5']/a['n']:8.4f} {a['mse']/a['mn']:9.4f}")
