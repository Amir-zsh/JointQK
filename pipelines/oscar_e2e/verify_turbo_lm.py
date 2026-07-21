#!/usr/bin/env python3
"""Gate for the TurboQuant-INT2 write path (fused Hadamard + Lloyd-Max).

T1  kernel-vs-torch parity: run the engine's fused kernel with LLOYD_MAX=1
    on random bf16 K/V, unpack the int2 arena, reconstruct via the readers'
    (q - zero) * scale contract, and compare against a torch replica of the
    exact same pipeline (bf16 load -> f32 FWHT -> bf16 roundtrip -> LM
    bucketize -> DELTA* uniform encoding). Bars: >=99.9% index agreement
    (fp-order threshold ties), reconstruction MSE within 2% of the replica.
T2  quantizer quality on the same data: kernel-path distortion must beat
    min-max (the QuaRot write path) and sit within 5% of exact-LM centroid
    reconstruction.
T3  guard: grouped scales + lloyd_max raises.

    PYTHONPATH=<clone>/sglang-research/python .venv-oscar/bin/python \
        pipelines/oscar_e2e/verify_turbo_lm.py
"""
from __future__ import annotations

import math
import sys

import torch

LM_T = 0.9810652732849121
LM_DELTA = 0.9877367723552428
LM_C_OUT = 1.5095585584640503
LM_C_IN = 2 * LM_T - LM_C_OUT


def fwht(x):
    d = x.shape[-1]
    h = 1
    y = x.clone()
    while h < d:
        y = y.view(*y.shape[:-1], d // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], d)
        h *= 2
    return y / math.sqrt(d)


def torch_replica(x_bf16):
    """Mirror the kernel: bf16 in -> f32 FWHT -> bf16 roundtrip -> LM."""
    acc = fwht(x_bf16.float())
    v = acc.to(torch.bfloat16).float()
    m = v.mean(-1, keepdim=True)
    var = v.var(-1, unbiased=False, keepdim=True)
    sd = (var + 1e-8).sqrt()
    d = v - m
    thr = LM_T * sd
    q = (d >= -thr).to(torch.uint8) + (d >= 0).to(torch.uint8) + (d >= thr).to(torch.uint8)
    scale = LM_DELTA * sd
    zero = 1.5 - m / scale
    rec = (q.float() - zero) * scale
    return v, q, rec


def unpack_int2(packed, head_dim):
    # quarter-interleaved pack: byte j holds dims (j, j+Q, j+2Q, j+3Q)
    Q = head_dim // 4
    qs = torch.stack([(packed >> (2 * i)) & 0x3 for i in range(4)], dim=-1)
    out = torch.empty(*packed.shape[:-1], head_dim, dtype=torch.uint8,
                      device=packed.device)
    for i in range(4):
        out[..., i * Q:(i + 1) * Q] = qs[..., i]
    return out


def main():
    torch.manual_seed(11)
    from sglang.QuantKernel.fused_hadamard_int2_kv import (
        quantized_set_kv_int2_hadamard_fused_triton,
    )
    dev = "cuda"
    T, H, D = 512, 8, 128
    k = (torch.randn(T, H, D, device=dev) * 1.7 + 0.3).to(torch.bfloat16)
    v = (torch.randn(T, H, D, device=dev) * 0.9 - 0.1).to(torch.bfloat16)
    loc = torch.arange(T, device=dev, dtype=torch.int64)
    kbuf = torch.zeros(T, H, D // 4, dtype=torch.uint8, device=dev)
    vbuf = torch.zeros_like(kbuf)
    ksz = torch.zeros(T, H, 2, dtype=torch.float32, device=dev)
    vsz = torch.zeros_like(ksz)

    quantized_set_kv_int2_hadamard_fused_triton(
        k, v, loc, kbuf, vbuf, ksz, vsz, hadamard_order=D, lloyd_max=True)

    ok = True
    for name, x, buf, sz in (("K", k, kbuf, ksz), ("V", v, vbuf, vsz)):
        rot, q_ref, rec_ref = torch_replica(x)
        q_kern = unpack_int2(buf, D)
        agree = (q_kern == q_ref).float().mean().item()
        scale = sz[..., 0:1]
        zero = sz[..., 1:2]
        rec_kern = (q_kern.float() - zero) * scale
        mse_kern = ((rec_kern - rot) ** 2).mean().item()
        mse_ref = ((rec_ref - rot) ** 2).mean().item()
        # exact-LM and min-max references on the same rotated rows
        m = rot.mean(-1, keepdim=True)
        sd = (rot.var(-1, unbiased=False, keepdim=True) + 1e-8).sqrt()
        z = (rot - m) / sd
        qi = (z >= -LM_T).long() + (z >= 0).long() + (z >= LM_T).long()
        cent = torch.tensor([-LM_C_OUT, -LM_C_IN, LM_C_IN, LM_C_OUT], device=dev)
        mse_lm = ((cent[qi] * sd + m - rot) ** 2).mean().item()
        lo, hi = rot.amin(-1, keepdim=True), rot.amax(-1, keepdim=True)
        s = (hi - lo).clamp_min(1e-8) / 3.0
        zz = -lo / s
        qmm = (rot / s + zz + 0.5).floor().clamp(0, 3)
        mse_mm = (((qmm - zz) * s - rot) ** 2).mean().item()
        t1 = agree >= 0.999 and mse_kern <= 1.02 * mse_ref
        t2 = mse_kern < mse_mm and mse_kern <= 1.05 * mse_lm
        print(f"T1/{name}: idx-agree={agree:.5f} mse kern={mse_kern:.5e} "
              f"ref={mse_ref:.5e} -> {'PASS' if t1 else 'FAIL'}")
        print(f"T2/{name}: vs minmax={mse_mm:.5e} vs exactLM={mse_lm:.5e} "
              f"-> {'PASS' if t2 else 'FAIL'}")
        ok &= t1 and t2

    # T4  writer-vs-writer: prefill writes PRE-ROTATED K/V through the plain
    # int2 writer while decode writes raw K/V through the fused-Hadamard
    # writer — both must produce the same arena bytes and scales (this is
    # exactly the two-writer split that fired the memory_pool guard on the
    # first boot attempt).
    from sglang.srt.mem_cache.kv_quant_kernels import quantized_set_kv_int2_triton
    k_rot = fwht(k.float()).to(torch.bfloat16)
    v_rot = fwht(v.float()).to(torch.bfloat16)
    kbuf2 = torch.zeros_like(kbuf)
    vbuf2 = torch.zeros_like(vbuf)
    ksz2 = torch.zeros_like(ksz)
    vsz2 = torch.zeros_like(vsz)
    quantized_set_kv_int2_triton(
        k_rot, v_rot, loc, kbuf2, vbuf2, ksz2, vsz2, lloyd_max=True)
    idx_same = ((kbuf2 == kbuf).float().mean().item()
                + (vbuf2 == vbuf).float().mean().item()) / 2
    sz_rel = ((ksz2 - ksz).abs().max() / ksz.abs().max()).item()
    t4 = idx_same >= 0.999 and sz_rel < 1e-2
    print(f"T4: writer-vs-writer arena-agree={idx_same:.5f} "
          f"scale-relerr={sz_rel:.2e} -> {'PASS' if t4 else 'FAIL'}")
    ok &= t4

    try:
        ksz3 = torch.zeros(T, H, 4, dtype=torch.float32, device=dev)
        quantized_set_kv_int2_hadamard_fused_triton(
            k, v, loc, kbuf, vbuf, ksz3, ksz3.clone(),
            hadamard_order=D, lloyd_max=True)
        print("T3: grouped+LM did NOT raise -> FAIL")
        ok = False
    except ValueError:
        print("T3: grouped+LM raises -> PASS")

    print("ALL GATES PASS" if ok else "GATE FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
