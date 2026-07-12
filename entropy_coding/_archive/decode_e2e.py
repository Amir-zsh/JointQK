#!/usr/bin/env python3
"""End-to-end decode time INCLUDING reconstruction (affine dequant / codebook gather),
consumed in SRAM into a checksum -- NO fp16 write-back (fused-attention framing). The
r@inv rotation is folded into the query side (q' = q@inv^T, once per query, not per
key), so it is legitimately not a per-key decode cost. Full 36x8 model, T=65536, A100,
per-block partial reduction (no atomic serialization). Actual ms."""
import torch, triton, triton.language as tl, time
dev = torch.device("cuda")
from kvq_codec import load_ext as _load_ext
_vqf = _load_ext(source="vq_fused.cu", name="vq_fused")

L, Hkv, d = 36, 8, 128
T = 65536
PEAK = 1555.0
G = 6; NG = d // G; K = 1 << (2 * G)          # VQ: group 6, 4096-entry codebook, 2 b/coord
GS = 64                                        # INT2 quant group size (scale/zero per 64)
n_heads = L * Hkv
n_elem = n_heads * T * d
n_grp = n_heads * T * NG


@triton.jit
def bf16_dec(kptr, n, BLOCK: tl.constexpr, out):
    pid = tl.program_id(0).to(tl.int64); o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    x = tl.load(kptr + o, mask=m, other=0)                       # read fp16 K (already usable)
    tl.store(out + pid, tl.sum(x.to(tl.float32)))


@triton.jit
def int2_dec(packed, scale, zero, n, GS: tl.constexpr, BLOCK: tl.constexpr, out):
    pid = tl.program_id(0).to(tl.int64); o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    b = tl.load(packed + o // 4, mask=m, other=0).to(tl.uint32)  # read 2-bit packed
    q = (b >> ((o % 4) * 2).to(tl.uint32)) & 0x3
    g = o // GS
    s = tl.load(scale + g, mask=m, other=1.0); z = tl.load(zero + g, mask=m, other=0.0)
    k = (q.to(tl.float32) - z) * s                               # affine dequant
    tl.store(out + pid, tl.sum(k))


@triton.jit
def turbo_dec(packed, cents, norms, n, d, BLOCK: tl.constexpr, out):
    pid = tl.program_id(0).to(tl.int64); o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    b = tl.load(packed + o // 4, mask=m, other=0).to(tl.uint32)  # read 2-bit packed
    q = (b >> ((o % 4) * 2).to(tl.uint32)) & 0x3
    c = tl.load(cents + q, mask=m, other=0.0)                    # 4-entry centroid LUT
    nrm = tl.load(norms + o // d, mask=m, other=1.0)             # per-vector fp16 norm
    tl.store(out + pid, tl.sum(c * nrm))


def timeit(f, reps=30, warmup=10):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps


print(f"End-to-end decode (read + reconstruct in SRAM, no fp16 write-back). "
      f"Full {L}x{Hkv} model, T={T}, A100. Rotation folded into query (not per-key).\n")
print(f"{'method':>12} {'reads':>8} {'decode ms':>10} {'GB/s':>7} {'vs BF16':>8}")

BLK = 8192
# BF16
kv = torch.randn(n_elem, device=dev, dtype=torch.float16)
op = torch.zeros(triton.cdiv(n_elem, BLK), device=dev)
tb = timeit(lambda: bf16_dec[(triton.cdiv(n_elem, BLK),)](kv, n_elem, BLK, op))
bb = n_elem * 2
del kv; torch.cuda.empty_cache()
print(f"{'BF16':>12} {bb/1e9:>6.2f}GB {tb*1e3:>8.3f}ms {bb/1e9/tb:>7.0f} {1.0:>7.2f}x")

# INT2
packed = torch.randint(0, 256, ((n_elem + 3)//4,), device=dev, dtype=torch.uint8)
ng2 = (n_elem + GS - 1)//GS
sc = torch.rand(ng2, device=dev) + 0.1; ze = torch.zeros(ng2, device=dev)
op = torch.zeros(triton.cdiv(n_elem, BLK), device=dev)
ti = timeit(lambda: int2_dec[(triton.cdiv(n_elem, BLK),)](packed, sc, ze, n_elem, GS, BLK, op))
bi = (n_elem + 3)//4 + ng2 * 8
print(f"{'INT2':>12} {bi/1e9:>6.2f}GB {ti*1e3:>8.3f}ms {bi/1e9/ti:>7.0f} {tb/ti:>7.2f}x")
del sc, ze; torch.cuda.empty_cache()

# TurboQuant
cents = torch.randn(4, device=dev); norms = torch.randn(n_heads * T, device=dev, dtype=torch.float16)
op = torch.zeros(triton.cdiv(n_elem, BLK), device=dev)
ttq = timeit(lambda: turbo_dec[(triton.cdiv(n_elem, BLK),)](packed, cents, norms, n_elem, d, BLK, op))
btq = (n_elem + 3)//4 + n_heads * T * 2
print(f"{'TurboQuant':>12} {btq/1e9:>6.2f}GB {ttq*1e3:>8.3f}ms {btq/1e9/ttq:>7.0f} {tb/ttq:>7.2f}x")
del packed, cents, norms; torch.cuda.empty_cache()

# VQ (fused CUDA: read int16 index + shared-mem codebook gather + accumulate)
cb = torch.randn(n_heads * NG * K * G, device=dev, dtype=torch.float16)
idx = torch.randint(0, K, (n_heads * NG * T,), device=dev, dtype=torch.int16)
tv = timeit(lambda: _vqf.vq_fused(idx, cb, n_heads, T, NG, K, G))
bv = idx.numel() * 2 + cb.numel() * 2       # int16 index stream + codebook (staged once)
print(f"{'VQ (fused)':>12} {bv/1e9:>6.2f}GB {tv*1e3:>8.3f}ms {bv/1e9/tv:>7.0f} {tb/tv:>7.2f}x")
del cb, idx; torch.cuda.empty_cache()

# rANS (serial decode, from its real kernel: 12.394 us/page, full-280-head-grid canonical rate)
rans_ms = 12.394e-3 * (n_heads * (T // 64))
print(f"{'rANS':>12} {'~0.6GB':>8} {rans_ms:>8.1f}ms {'--':>7} {tb/(rans_ms/1e3):>7.4f}x  (serial, compute-bound)")

print("\nAll fixed-width rows include reconstruction (INT2 affine dequant, TurboQuant "
      "centroid+norm, VQ codebook gather); consumed in SRAM, no fp16 write-back. rANS from "
      "its measured serial decode kernel. Rotation folded into the query, so not per-key.")
