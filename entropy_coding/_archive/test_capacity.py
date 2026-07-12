#!/usr/bin/env python3
"""V1 batch-scaling: prefill-capacity throughput, fp16-K vs compressed-K.
OOM-walk to find max fittable batch per scheme at fixed context, then aggregate
tok/s (decode cost included). Prefill-phase only — NOT generation throughput."""
import argparse, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

import run_pca_ec_deadzone as base
from kvq_codec import (build_codecs_from_ladder_rans_cuda, load_ext,
                       BatchRANSEncoder, BatchRANSDecoder)


def flash(q, k, v):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q.half(), k.half(), v.half()).float()


def free():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-idx", type=int, default=4)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--ptok", type=int, default=64)
    ap.add_argument("--dz", type=float, default=0.375)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--m-grid", type=float, nargs="+", default=[1.0, 1.05, 1.1, 1.25, 1.5])
    ap.add_argument("--bmax", type=int, default=256, help="cap the OOM-walk")
    args = ap.parse_args()
    dev = torch.device("cuda")

    root = base.data_root(); manifest = base.load_manifest(root)
    sigma_q, sigma_k, k_mean, k_cov, meta = base.calib_moments(root, manifest, args.calib_idx)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]

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

    art = torch.load(root / manifest["examples"][args.eval_idx]["file"],
                     map_location="cpu", weights_only=False)
    q_all, k_all, v_all = art["q_post"], art["k_post"], art["v"]
    T = int(art["prompt_length"]); gs = q_all.shape[1] // Hkv

    # one sequence's K/Q/V on CPU; we replicate to batch B on demand
    # layer 1+ only (skip sink); use a single representative layer-grid for the sweep
    heads = [(l, h) for l in range(1, L) for h in range(Hkv)]
    # pre-encode the compressed K buffers once (per head); replicated across batch
    k_grid = {(l, h): k_all[l, h, :T, :].float() for l in range(L) for h in range(Hkv)}
    bufs = enc.encode_grid(k_grid)

    def qkv_batch(l, h, B):
        q = q_all[l, h*gs:(h+1)*gs, :T, :].to(dev).float()          # (gs,T,d)
        v = v_all[l, h, :T, :].to(dev).float()                      # (T,d)
        # batch = B independent sequences sharing this head's tensors (capacity proxy)
        qB = q.unsqueeze(0).expand(B, gs, T, d).reshape(B*gs, 1, T, d)
        vB = v.unsqueeze(0).expand(B*gs, T, d).unsqueeze(1)         # (B*gs,1,T,d)
        return qB, vB

    # ---- fp16 path: attention over raw fp16 K, batch B ----
    def run_fp16(B):
        free()
        for (l, h) in heads:
            qB, vB = qkv_batch(l, h, B)
            kf = k_all[l, h, :T, :].to(dev).float()
            kB = kf.unsqueeze(0).expand(B*gs, T, d).unsqueeze(1)
            _ = flash(qB, kB, vB)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    # ---- compressed path: decode K then attend, batch B ----
    def run_comp(B):
        free()
        dec = decod.decode_grid(bufs)        # decode once; replicate across batch
        for (l, h) in heads:
            qB, vB = qkv_batch(l, h, B)
            kf = dec[(l, h)][:T].to(dev).float()
            kB = kf.unsqueeze(0).expand(B*gs, T, d).unsqueeze(1)
            _ = flash(qB, kB, vB)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    def oom_walk(fn, label):
        B, last_ok = 1, 0
        while B <= args.bmax:
            try:
                mem = fn(B)
                last_ok = B
                print(f"  {label} B={B:4d} ok  peak={mem/1e9:.1f} GB")
                B *= 2
            except torch.cuda.OutOfMemoryError:
                print(f"  {label} B={B:4d} OOM")
                break
        return last_ok

    print("OOM-walk fp16:");  Bf = oom_walk(run_fp16, "fp16")
    print("OOM-walk comp:");  Bc = oom_walk(run_comp, "comp")

    def timed(fn, B, reps=3):
        fn(B); torch.cuda.synchronize()           # warmup
        t0 = time.perf_counter()
        for _ in range(reps): fn(B)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    tf = timed(run_fp16, Bf); tc = timed(run_comp, Bc)
    # aggregate throughput proxy: sequences * tokens / wall-time
    tokf = Bf * T / (tf/1e3); tokc = Bc * T / (tc/1e3)
    print(f"\n--- prefill-capacity (T={T}, {len(heads)} heads) ---")
    print(f"fp16: maxB={Bf}  {tf:.0f} ms  {tokf:,.0f} tok/s")
    print(f"comp: maxB={Bc}  {tc:.0f} ms  {tokc:,.0f} tok/s")
    print(f"batch gain {Bc/max(Bf,1):.1f}x | throughput {tokc/max(tokf,1e-9):.2f}x")


if __name__ == "__main__":
    main()