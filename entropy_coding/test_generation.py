#!/usr/bin/env python3
"""Option A standalone: compressed K -> GPU rANS decode -> FlashAttention.
Checks attention OUTPUT (not just k_hat) against fp16-K attention, and times
decode + attend vs plain fp16 attention. One (layer,head)-grid, one eval row.
No model, no KVPress, no generation loop."""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

import run_pca_ec_deadzone as base
from kvq_codec import (build_codecs_from_ladder_rans_cuda, load_ext,
                       BatchRANSEncoder, BatchRANSDecoder)


def sdpa_attn(q, k, v):
    # No causal mask: this is a decoded-vs-fp16 K comparison under IDENTICAL
    # attention, so the mask cancels. Q is GQA-grouped on the head axis (gs rows),
    # NOT the sequence axis, so is_causal=True would misalign and mask wrongly.
    return F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0),
                                          v.unsqueeze(0)).squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-idx", type=int, default=4)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--ptok", type=int, default=64)
    ap.add_argument("--dz", type=float, default=0.375)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--m-grid", type=float, nargs="+",
                    default=[1.0, 1.05, 1.1, 1.25, 1.5])
    args = ap.parse_args()
    dev = torch.device("cuda")

    root = base.data_root(); manifest = base.load_manifest(root)
    sigma_q, sigma_k, k_mean, k_cov, meta = base.calib_moments(root, manifest, args.calib_idx)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]

    # QPCA paged codec on real calib (same construction as test_codec_on_data.py)
    qpca_unc = base.build_qpca_basis(sigma_q, sigma_k)
    qpca_unc["sigma_k"], qpca_unc["sigma_q"] = sigma_k, sigma_q
    qpca_cen = base.build_qpca_basis(sigma_q, k_cov); qpca_cen["sigma_k"] = sigma_k
    Ffwd, inv = qpca_cen["forward"], qpca_cen["inverse"]
    fetch_calib = base._codes_for_idx(root, manifest, args.calib_idx, Ffwd, k_mean, L, Hkv, d)
    _, delta0, model0 = base.build_qpca_ec(qpca_cen, qpca_unc, k_mean, args.b, L, Hkv,
                                           fetch_calib, root, args.calib_idx,
                                           dz=args.dz, match_rate=False, uniform_step=True)
    ladder = [(1.0, delta0, model0)]
    for m in sorted(set(args.m_grid) | {1.0}):
        if abs(m - 1.0) > 1e-9:
            dm = (delta0 * m).float()
            ladder.append((m, dm, base.freeze_coder_model(fetch_calib, dm, L, Hkv, d, args.dz)))
    page_bits = args.b * d * args.ptok
    ext = load_ext()
    codecs = build_codecs_from_ladder_rans_cuda(Ffwd, inv, k_mean, ladder, L, Hkv,
                                                page_bits, args.ptok, args.dz,
                                                lanes=args.lanes, ext=ext, device="cuda")
    enc = BatchRANSEncoder(codecs); decod = BatchRANSDecoder(codecs)

    # load one eval row's Q/K/V
    art = torch.load(root / manifest["examples"][args.eval_idx]["file"],
                     map_location="cpu", weights_only=False)
    q_all, k_all, v_all = art["q_post"], art["k_post"], art["v"]
    T = int(art["prompt_length"]); gs = q_all.shape[1] // Hkv

    # encode + decode K across the grid
    k_grid = {(l, h): k_all[l, h, :T, :].float() for l in range(L) for h in range(Hkv)}
    bufs = enc.encode_grid(k_grid)
    k_dec = decod.decode_grid(bufs)            # {(l,h): (T,d) decoded K on GPU}

    # sanity: GPU decode vs codec's own CPU decode of the same buffer (kernel check)
    lh = (1, 0)
    k_cpu = decod.codecs[lh].decode(bufs[lh])
    k_gpu = k_dec[lh].cpu().numpy()
    print("GPU vs CPU decode, same buffer:", float(np.abs(k_gpu - k_cpu).max()))
    knp = k_all[1, 0, :T, :].float().cpu().numpy().astype(np.float64)
    print("raw K abs max:", float(np.abs(knp).max()),
          "decoded abs max:", float(np.abs(k_gpu).max()))

    # attention-output check, layer 1+ (skip sink layer 0)
    max_out_err = 0.0; sum_out_err = 0.0; n_cmp = 0
    for l in range(1, L):
        for h in range(Hkv):
            T_enc = k_grid[(l, h)].shape[0]
            kd = k_dec[(l, h)]
            if l == 1 and h == 0:
                print(f"T={T}  T_enc={T_enc}  k_dec.shape={tuple(kd.shape)}")
                kref = k_all[l, h, :T_enc, :].to(dev).float()
                kdec = kd[:T_enc].to(dev).float()
                print(f"k_err (decoded vs raw, expected ~O(1) at b={args.b}) = "
                      f"{float((kref-kdec).abs().max()):.3e}")
            q = q_all[l, h*gs:(h+1)*gs, :T_enc, :].to(dev).float()
            v = v_all[l, h, :T_enc, :].to(dev).float().unsqueeze(0).expand(gs, -1, -1)
            k_fp16 = k_all[l, h, :T_enc, :].to(dev).float().unsqueeze(0).expand(gs, -1, -1)
            k_hat  = kd[:T_enc].to(dev).float().unsqueeze(0).expand(gs, -1, -1)
            out_ref = sdpa_attn(q, k_fp16, v)
            out_dec = sdpa_attn(q, k_hat,  v)
            d_abs = (out_ref - out_dec).abs()
            max_out_err = max(max_out_err, float(d_abs.max()))
            sum_out_err += float(d_abs.mean()); n_cmp += 1
    print(f"attention-output  max|delta|={max_out_err:.3e}  "
          f"mean|delta|={sum_out_err/n_cmp:.3e}")

    # timing: decode+attend vs fp16 attend, layer 1+ grid.
    # NOTE: on sm_7.5 this uses the SDPA MATH backend (flash gated to sm_80+),
    # so this ratio is NOT the real throughput number — that needs A100+.
    heads = [(l, h) for l in range(1, L) for h in range(Hkv)]
    def attend_fp16():
        for (l, h) in heads:
            q = q_all[l, h*gs:(h+1)*gs, :T, :].to(dev).float()
            v = v_all[l, h, :T, :].to(dev).float().unsqueeze(0).expand(gs, -1, -1)
            k = k_all[l, h, :T, :].to(dev).float().unsqueeze(0).expand(gs, -1, -1)
            _ = sdpa_attn(q, k, v)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    dec = decod.decode_grid(bufs)
    for (l, h) in heads:
        q = q_all[l, h*gs:(h+1)*gs, :T, :].to(dev).float()
        v = v_all[l, h, :T, :].to(dev).float().unsqueeze(0).expand(gs, -1, -1)
        k = dec[(l, h)].to(dev).float().unsqueeze(0).expand(gs, -1, -1)
        _ = sdpa_attn(q, k, v)
    torch.cuda.synchronize(); t_dec = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize(); t0 = time.perf_counter(); attend_fp16(); torch.cuda.synchronize()
    t_fp16 = (time.perf_counter() - t0) * 1e3
    print(f"decode+attend {t_dec:.1f} ms | fp16 attend {t_fp16:.1f} ms | "
          f"ratio {t_dec/max(t_fp16,1e-9):.2f}x | {len(heads)} heads  "
          f"[math backend — not real throughput]")


if __name__ == "__main__":
    main()