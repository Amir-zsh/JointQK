#!/usr/bin/env python3
"""Option #2 (tractable form): fine-tune a K-compressor to a DOWNSTREAM objective
-- per-head attention-KL between the compressed and full-precision attention
distributions softmax(q k^T) -- instead of reconstruction MSE. Captures which
coordinate errors flip the attention argmax (what F1 cares about), which plain
k-means / MSE ignores. Uses only captured q/k (no model forward). Works for VQ
(optimize centroids) and INT2 (optimize per-coord deadzone step).

Starts from the MSE-trained artifact and gradient-descends on attention-KL, so it
is a refinement, not a from-scratch retrain.
"""
import argparse, torch
import run_pca_ec_deadzone as base

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["vq", "int2"], required=True)
ap.add_argument("--in-codebook", help="vq: vqa_*.pt to refine")
ap.add_argument("--basis-moments", default="/vault/samuel/data/basis_moments_qwen3_8b_compact8train400/basis_moments.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--qk-idx", type=int, nargs="+", default=[0, 1, 2])  # under4k prompts (have q+k)
ap.add_argument("--qk-root", default=None, help="override q/k pool dir (query_stats format w/ q_post+k_post)")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--max-q", type=int, default=512)   # subsample queries/head for speed
ap.add_argument("--dz", type=float, default=0.375)  # int2 only
ap.add_argument("--load", type=float, default=3.0)  # int2 only
args = ap.parse_args()
dev = "cuda"
from pathlib import Path as _P
root = _P(args.qk_root) if args.qk_root else base.data_root()
man = base.load_manifest(root)
B = torch.load(args.basis_moments, map_location="cpu", weights_only=False)
sq, sk, km, kc, meta = B["sigma_q"], B["sigma_k"], B["k_mean"], B["k_cov"], B["meta"]
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]; gs = meta["group_size"]
scale = d ** -0.5

# preload q/k per prompt (under4k has q_post + k_post)
QK = []
for i in args.qk_idx:
    art = torch.load(root / man["examples"][i]["file"], map_location="cpu", weights_only=False)
    T = int(art["prompt_length"])
    QK.append((art["q_post"][:, :, :T, :].float(), art["k_post"][:, :, :T, :].float()))

def head_qk(l, h):
    qs, ks = [], []
    for q, k in QK:
        qs.append(q[l, h*gs:(h+1)*gs].reshape(-1, d)); ks.append(k[l, h])
    return torch.cat(qs, 0).to(dev), torch.cat(ks, 0).to(dev)

if args.method == "vq":
    from group_vq_codec import group_boundaries
    p = torch.load(args.in_codebook, map_location="cpu", weights_only=False)
    F, inv, mean, bounds, cbs = p["forward"], p["inverse"], p["mean"], p["bounds"], p["codebooks"]
    new_cbs = {}
    for l in range(L):
        if (l, 0) not in cbs:   # layer 0 not in codebook
            continue
        Fl, invl, meanl = F[l].to(dev).float(), inv[l].to(dev).float(), mean[l].to(dev).float()
        for h in range(Hkv):
            q, k = head_qk(l, h)
            if q.shape[0] > args.max_q:
                sel = torch.randperm(q.shape[0], device=dev)[:args.max_q]; q = q[sel]
            with torch.no_grad():
                A_fp = torch.softmax(q @ k.t() * scale, dim=-1)         # (Nq, Nk) reference
                r = (k - meanl[h]) @ Fl[h]                              # (Nk, d)
            cent = [c.to(dev).float().clone().requires_grad_(True) for c in cbs[(l, h)]]
            opt = torch.optim.Adam(cent, lr=args.lr)
            for _ in range(args.steps):
                r_hat = torch.empty_like(r)
                for (s, e, _b), c in zip(bounds, cent):
                    idx = torch.cdist(r[:, s:e].detach(), c.detach()).argmin(1)   # fixed assign
                    r_hat[:, s:e] = c[idx]                                        # grad -> centroids
                k_hat = r_hat @ invl[h] + meanl[h]
                A = torch.softmax(q @ k_hat.t() * scale, dim=-1)
                loss = (A_fp * (A_fp.clamp_min(1e-9).log() - A.clamp_min(1e-9).log())).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            new_cbs[(l, h)] = [c.detach().cpu() for c in cent]
        print(f"  layer {l}/{L} done", flush=True) if l % 6 == 0 else None
    p["codebooks"] = new_cbs
    torch.save(p, args.out)
    print(f"SAVED {args.out} (attention-KL fine-tuned VQ)", flush=True)

else:  # int2: optimize per-coord deadzone step (delta) on attention-KL
    qpca_unc = base.build_qpca_basis(sq, sk)
    scalar = base.build_qpca_fixed_deadzone(qpca_unc, km, b=2, n_layers=L, n_kv_heads=Hkv,
                                            dz=args.dz, load=args.load)  # start point
    Fq, invq = qpca_unc["forward"], qpca_unc["inverse"]
    dz = args.dz
    comps = {}
    for l in range(L):
        if l == 0:
            continue
        for h in range(Hkv):
            q, k = head_qk(l, h)
            if q.shape[0] > args.max_q:
                sel = torch.randperm(q.shape[0], device=dev)[:args.max_q]; q = q[sel]
            c0 = scalar[(l, h)].to(dev)
            fwd, inv2, mu = c0.fwd, c0.inv, c0.mu
            qmax = c0.qmax                                     # (1,d) max |index| per coord
            with torch.no_grad():
                A_fp = torch.softmax(q @ k.t() * scale, dim=-1)
                r = (k.to(dev) - mu) @ fwd
            log_delta = c0.delta.clone().log().requires_grad_(True)   # (1,d)
            opt = torch.optim.Adam([log_delta], lr=args.lr)
            for _ in range(args.steps):
                delta = log_delta.exp()
                # integer levels (round is non-diff -> detach); reconstruct as n*delta
                # so the gradient flows to delta through the multiply (STE on round only).
                n = torch.round(r / delta).clamp(-qmax, qmax).detach()
                r_q = n * delta
                k_hat = r_q @ inv2 + mu
                A = torch.softmax(q @ k_hat.t() * scale, dim=-1)
                loss = (A_fp * (A_fp.clamp_min(1e-9).log() - A.clamp_min(1e-9).log())).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            c0.delta = log_delta.detach().exp()
            comps[(l, h)] = c0.to("cpu")
        print(f"  layer {l}/{L} done", flush=True) if l % 6 == 0 else None
    torch.save({"comps": comps, "method": "scalar_int2_attnkl"}, args.out)
    print(f"SAVED {args.out} (attention-KL fine-tuned INT2)", flush=True)
