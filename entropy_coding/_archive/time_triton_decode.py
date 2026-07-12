#!/usr/bin/env python3
"""Triton FA attention-only vs batched decode cost, full grid, apples-to-apples.
Both batched (no per-head Python overhead). decode_grid is the real decode cost."""
import numpy as np, torch, time
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, BatchRANSDecoder
from step2b_triton import triton_resid_attn

dev = torch.device("cuda")
root = base.data_root(); man = base.load_manifest(root)
sq, sk, km, kc, meta = base.calib_moments(root, man, [0, 1, 2])
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"], qc["inverse"]
fc = base._codes_for_idx(root, man, [0, 1, 2], F, km, L, Hkv, d)
_, d0, m0 = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, [0, 1, 2],
                               dz=0.375, match_rate=False, uniform_step=True)
ladder = [(1.0, d0, m0)]
for m in [1.05, 1.1, 1.25, 1.5]:
    dm = (d0 * m).float()
    ladder.append((m, dm, base.freeze_coder_model(fc, dm, L, Hkv, d, 0.375)))
ext = load_ext()
cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L, Hkv, 2*d*64, 64, 0.375,
                                         lanes=16, ext=ext, device="cuda")
enc = BatchRANSEncoder(cod); decod = BatchRANSDecoder(cod)
art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T = int(art["prompt_length"]); gs = art["q_post"].shape[1] // Hkv
kg = {(l, h): art["k_post"][l, h, :T, :].float() for l in range(L) for h in range(Hkv)}
bufs = enc.encode_grid(kg)
sm = 1.0 / np.sqrt(d)
heads = [(l, h) for l in range(1, L) for h in range(Hkv)]

print("preloading q', v, pre-decoded r̂ per head...")
qproj = {}; vbuf = {}; rhat_pre = {}
for (l, h) in heads:
    c = cod[(l, h)]
    q = art["q_post"][l, h*gs:(h+1)*gs, :T, :].to(dev).float()
    invT = torch.as_tensor(c.inv, dtype=torch.float32, device=dev)
    qproj[(l, h)] = (q @ invT.T).contiguous()
    vbuf[(l, h)] = art["v"][l, h, :T, :].to(dev).float().contiguous()
    rhat_pre[(l, h)] = c.decode_to_rhat(bytes(bufs[(l, h)]))[:T].contiguous()

def run_attn_only():
    for (l, h) in heads:
        triton_resid_attn(qproj[(l, h)], rhat_pre[(l, h)], vbuf[(l, h)], sm)

def run_decode_only():
    decod.decode_grid(bufs)   # batched rANS decode, one launch+sync

# warmup
run_attn_only(); run_decode_only(); torch.cuda.synchronize()

def timeit(f, reps=3):
    t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3

ta = timeit(run_attn_only)
td = timeit(run_decode_only)
# add to the harness, before the prints:
import torch.nn.functional as Fnn
from torch.nn.attention import sdpa_kernel, SDPBackend
kh_all = {(l,h): cod[(l,h)].decode_to_gpu(bytes(bufs[(l,h)]))[:T].float() for (l,h) in heads}
qraw = {(l,h): art["q_post"][l, h*gs:(h+1)*gs, :T, :].to(dev).float() for (l,h) in heads}
def run_fp16():
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        for (l,h) in heads:
            Fnn.scaled_dot_product_attention(
                qraw[(l,h)].half().unsqueeze(0),
                kh_all[(l,h)].half().unsqueeze(0).unsqueeze(0).expand(1,gs,T,d),
                vbuf[(l,h)].half().unsqueeze(0).unsqueeze(0).expand(1,gs,T,d), is_causal=True)
run_fp16(); torch.cuda.synchronize()
tfp = timeit(run_fp16)
print(f"F. fp16 flash (same grid)      : {tfp:8.1f} ms")
print(f"   Triton attn / fp16 flash    : {ta/tfp:.2f}x")
print(f"\n--- full grid ({len(heads)} heads, T={T}) ---")
print(f"A. Triton FA only (attention)  : {ta:8.1f} ms")
print(f"D. batched decode (decode_grid): {td:8.1f} ms")
print(f"   decode as % of attention    : {td/ta*100:.0f}%")
print(f"   decode + attention total    : {ta+td:8.1f} ms  ({(ta+td)/ta:.2f}x attention)")