#!/usr/bin/env python3
"""Fair three-way per-page cost, all paths ending at usable fp16 K tile:
  BF16   : read 16-bit fp16 (no unpack)        -- the baseline
  INT2   : read 2-bit packed + affine dequant  -- OSCAR-style fixed-width
  rANS   : read 2-bit stream + entropy decode  -- your codec
All forced to touch HBM at their true byte size. INT2 is a real Triton unpack kernel."""
import numpy as np, torch, time, triton, triton.language as tl
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, BatchRANSDecoder

dev = torch.device("cuda")
P = 64; d = 128

# ---------------- INT2 affine unpack kernel (OSCAR-style) ----------------
@triton.jit
def int2_unpack_kernel(packed_ptr, scale_ptr, zero_ptr, out_ptr,
                       n_elem, G: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    byte_idx = offs // 4
    shift = (offs % 4) * 2
    b = tl.load(packed_ptr + byte_idx, mask=mask, other=0).to(tl.uint32)
    q = (b >> shift) & 0x3
    grp = offs // G
    s = tl.load(scale_ptr + grp, mask=mask, other=1.0)
    z = tl.load(zero_ptr + grp, mask=mask, other=0.0)
    val = (q.to(tl.float32) - z) * s
    tl.store(out_ptr + offs, val.to(tl.float16), mask=mask)

def int2_unpack(packed, scale, zero, n_elem, G=64):
    out = torch.empty(n_elem, device=dev, dtype=torch.float16)
    grid = (triton.cdiv(n_elem, 1024),)
    int2_unpack_kernel[grid](packed, scale, zero, out, n_elem, G, 1024)
    return out

# ---------------- fair HBM read (forces full read of given bytes) ----------------
@triton.jit
def touch_kernel(ptr, n, BLOCK: tl.constexpr, out_ptr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0)
    acc = tl.sum(x.to(tl.float32))
    tl.atomic_add(out_ptr, acc)

def touch(buf_tensor):
    n = buf_tensor.numel()
    sink = torch.zeros(1, device=dev, dtype=torch.float32)
    grid = (triton.cdiv(n, 4096),)
    touch_kernel[grid](buf_tensor, n, 4096, sink)
    return sink

# ---------------- setup ----------------
root = base.data_root(); man = base.load_manifest(root)
sq, sk, km, kc, meta = base.calib_moments(root, man, [0, 1, 2])
L, Hkv, dd = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"], qc["inverse"]
fc = base._codes_for_idx(root, man, [0, 1, 2], F, km, L, Hkv, dd)
_, d0, m0 = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, [0, 1, 2],
                               dz=0.375, match_rate=False, uniform_step=True)
ladder = [(1.0, d0, m0)]
for m in [1.05, 1.1, 1.25, 1.5]:
    dm = (d0 * m).float()
    ladder.append((m, dm, base.freeze_coder_model(fc, dm, L, Hkv, dd, 0.375)))
ext = load_ext()
cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L, Hkv, 2*dd*64, 64, 0.375,
                                         lanes=16, ext=ext, device="cuda")
enc = BatchRANSEncoder(cod); decod = BatchRANSDecoder(cod)
art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T0 = int(art["prompt_length"])

def tile_to(x, T):
    reps = (T + x.shape[0] - 1) // x.shape[0]
    return x.repeat(reps, 1)[:T].contiguous()

def timeit(f, reps=30, warmup=10):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps

print(f"{'T':>8} {'BF16_read':>11} {'INT2_r+unpack':>14} {'rANS_r+decode':>14} "
      f"{'INT2/BF16':>10} {'rANS/BF16':>10}")

for T in [4096, 8192, 16384, 32768]:
    nb = (T + P - 1) // P
    kT = tile_to(art["k_post"][1, 0, :T0, :].float(), T)

    # --- BF16 path ---
    kfp16 = kT.to(dev).half().contiguous().view(-1)
    def path_bf16(): touch(kfp16)

    # --- INT2 path (all GPU) ---
    G = 64
    k_dev = kT.to(dev)
    kg = k_dev.view(-1, G)
    mn = kg.min(1, keepdim=True).values; mx = kg.max(1, keepdim=True).values
    scale = ((mx - mn) / 3.0).clamp(min=1e-8)
    zero = (-mn / scale)
    q = ((kg / scale) + zero).round().clamp(0, 3).to(torch.uint8)
    q_flat = q.view(-1).contiguous()
    qf = q_flat
    packed = (qf[0::4] | (qf[1::4] << 2) | (qf[2::4] << 4) | (qf[3::4] << 6)).contiguous()
    scale_g = scale.view(-1).contiguous()
    zero_g = zero.view(-1).contiguous()
    n_elem = q_flat.numel()
    def path_int2():
        int2_unpack(packed, scale_g, zero_g, n_elem, G)

    # --- rANS path ---
    c = cod[(1, 0)]
    buf = c.encode_gpu(kT.cpu().numpy())
    def path_rans(): c.decode_to_rhat(bytes(buf))

    tb = timeit(path_bf16)
    ti = timeit(path_int2)
    tr = timeit(path_rans)
    pb, pi, pr = tb/nb*1e6, ti/nb*1e6, tr/nb*1e6
    print(f"{T:>8} {pb:>9.3f}us {pi:>12.3f}us {pr:>12.3f}us "
          f"{pi/pb:>9.2f}x {pr/pb:>9.2f}x")

print("\nAll paths produce usable fp16 K. BF16 reads 16-bit; INT2 & rANS read 2-bit + unpack/decode.")
print("INT2/BF16 < 1 => fixed-width compression beats BF16 at decode (OSCAR mechanism).")
print("rANS/BF16 = your codec. Compare INT2 vs rANS for entropy-vs-fixed-width cost.")