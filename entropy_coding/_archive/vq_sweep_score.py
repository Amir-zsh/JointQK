#!/usr/bin/env python3
"""Lightweight VQ-only accuracy scorer for the group-size (G) sweep.

Each train_group_vq.py checkpoint already stores its own forward/inverse/mean
QPCA basis, so this skips the expensive full-harness basis+ladder rebuild
(~20 min under load in test_codec_on_data.py) and only does what VQ needs:
load codebooks -> roundtrip eval keys -> top-1/top-5/k_mse vs full-precision
Q.K argmax, using the exact metric from test_codec_on_data.py's score_all.
Layer-0 excluded to match the headline convention.
"""
import argparse
import torch

import run_pca_ec_deadzone as base
from group_vq_codec import GroupVQCompressor


def score_codebook(path, eval_arts, L, Hkv, d, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    F, inv, mean = payload["forward"], payload["inverse"], payload["mean"]
    bounds = payload["bounds"]
    cbs = payload["codebooks"]
    G = payload["G"]
    cfg = f"{payload.get('grouping','cons')[:4]}/{payload.get('allocation','flat')[:4]}" + (
        "/wh" if payload.get("whiten") else "")
    real_bpc = payload.get("bits_per_coord", float("nan"))

    top1_num = top1_den = top5_num = top5_den = 0
    mse_num = 0.0
    mse_den = 0
    for art in eval_arts:
        T = int(art["prompt_length"])
        q_all, k_all = art["q_post"], art["k_post"]
        gs = q_all.shape[1] // Hkv
        for l in range(1, L):          # layer-0 excluded (headline convention)
            for h in range(Hkv):
                comp = GroupVQCompressor(
                    F[l, h].to(device), inv[l, h].to(device), mean[l, h].to(device),
                    [c.to(device) for c in cbs[(l, h)]], bounds)
                k_dev = k_all[l, h, :T, :].to(device).float()
                q = q_all[l, h * gs:(h + 1) * gs, :T, :].to(device).float().reshape(-1, d)
                real_top = (q @ k_dev.transpose(0, 1)).argmax(-1)
                n_q = int(real_top.numel())
                k5 = min(5, T)
                kh = comp.roundtrip(k_dev).float()
                err = k_dev - kh
                mse_num += float(err.square().sum()); mse_den += int(err.numel())
                ap = q @ kh.transpose(0, 1)
                top1_num += int((real_top == ap.argmax(-1)).sum()); top1_den += n_q
                a5 = ap.topk(k5, dim=-1).indices
                top5_num += int((a5 == real_top.unsqueeze(-1)).any(-1).sum()); top5_den += n_q
    return dict(G=G, K=(1 << (2 * G)), top1=top1_num / top1_den, top5=top5_num / top5_den,
                k_mse=mse_num / mse_den, n_groups=len(bounds), cfg=cfg, real_bpc=real_bpc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebooks", type=str, nargs="+", required=True)
    ap.add_argument("--eval-idx", type=int, nargs="+", default=[4, 8, 16, 20])
    ap.add_argument("--calib-idx", type=int, nargs="+", default=[0, 5, 6])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    overlap = set(args.eval_idx) & set(args.calib_idx)
    if overlap:
        raise SystemExit(f"eval/calib overlap: {sorted(overlap)}")

    root = base.data_root(); man = base.load_manifest(root)
    _, _, _, _, meta = base.calib_moments(root, man, args.calib_idx)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    eval_arts = [torch.load(root / man["examples"][i]["file"], map_location="cpu",
                            weights_only=False) for i in args.eval_idx]
    tot_tok = sum(int(a["prompt_length"]) for a in eval_arts)
    print(f"eval_idx={args.eval_idx} ({tot_tok} tokens) | calib_idx={args.calib_idx} | "
          f"samples/centroid at calib=8234:", flush=True)
    print(f"\n{'codebook':<30} {'G':>2} {'cfg':>12} {'bpc':>5} {'top-1':>7} {'top-5':>7} {'k_mse':>11}")
    print(f"{'scalar INT2 (QPCA-Fixed)':<30} {'-':>2} {'-':>12} {'2.00':>5} {0.647:>7.4f} {0.910:>7.4f} {'-':>11}")
    for path in args.codebooks:
        r = score_codebook(path, eval_arts, L, Hkv, d, dev)
        name = path.split("/")[-1]
        print(f"{name:<30} {r['G']:>2} {r.get('cfg','-'):>12} {r.get('real_bpc',float('nan')):>5.2f} "
              f"{r['top1']:>7.4f} {r['top5']:>7.4f} {r['k_mse']:>11.3e}", flush=True)
