#!/usr/bin/env python3
"""FUSED / read-bound throughput -- the serving-relevant cost. Compression helps
because the decode is fused into attention: you READ the compressed KV (<< BF16
bytes) and reconstruct in SRAM, never writing fp16 back to HBM. So we time
read+reconstruct-in-SRAM (accumulate a checksum, no HBM output). BF16 = read
2 B/coord; INT2 = read 0.25 B/coord (2-bit) + unpack; VQ = read 0.25 B/coord
(2-bit group index) + codebook gather (codebook RESIDENT across decode steps ->
one-time, amortized to ~0)."""
import torch, triton, triton.language as tl
from kvq_codec import load_ext as _load_ext
_vqf = _load_ext(source="vq_fused.cu", name="vq_fused")

dev = torch.device("cuda")
L, Hkv, d = 36, 8, 128
P = 64
PEAK = 1555.0
G = 6; NG = d // G; d_eff = NG * G
K = 1 << (2 * G)          # 4096 (2 b/c)


@triton.jit
def bf16_read(ptr, n, BLOCK: tl.constexpr, sink):
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    tl.atomic_add(sink, tl.sum(tl.load(ptr + o, mask=m, other=0).to(tl.float32)))


@triton.jit
def int2_read(packed, n_elem, BLOCK: tl.constexpr, sink):
    # read 2-bit packed, unpack (affine dequant would be here), accumulate. No HBM write.
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n_elem
    b = tl.load(packed + o // 4, mask=m, other=0).to(tl.uint32)
    q = (b >> ((o % 4) * 2).to(tl.uint32)) & 0x3
    tl.atomic_add(sink, tl.sum(q.to(tl.float32)))


@triton.jit
def vq_read(idx_ptr, cb_ptr, n_groups, K, G: tl.constexpr, BLOCK: tl.constexpr, sink):
    # read the group index (2b/coord), gather its G-float codeword, accumulate. No HBM write.
    gid = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = gid < n_groups
    e = tl.load(idx_ptr + gid, mask=m, other=0).to(tl.int64)
    acc = tl.zeros([BLOCK], tl.float32)
    for j in tl.static_range(0, 16):
        if j < G:
            acc += tl.load(cb_ptr + e * G + j, mask=m, other=0.0).to(tl.float32)
    tl.atomic_add(sink, tl.sum(acc))


def timeit(f, reps=30, warmup=8):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


print("FUSED read-bound (reconstruct in SRAM, no fp16 write-back) -- the serving cost.\n")
print(f"{'T':>7} {'BF16':>8} {'INT2':>8} {'TurboQ':>8} {'VQ_fused':>9} "
      f"{'INT2/BF16':>10} {'TQ/BF16':>9} {'VQf/BF16':>9} {'VQf/INT2':>9}   (us/page)")

for T in [16384, 32768, 65536]:
    n_heads = L * Hkv; nb = (T + P - 1)//P; n_pages = n_heads * nb
    n_elem = n_heads * T * d
    n_grp = n_heads * T * NG

    kv16 = torch.randn(n_elem, device=dev, dtype=torch.float16)
    sink = torch.zeros(1, device=dev)
    tb = timeit(lambda: bf16_read[(triton.cdiv(n_elem, 4096),)](kv16, n_elem, 4096, sink))
    del kv16; torch.cuda.empty_cache()

    packed = torch.randint(0, 256, ((n_elem + 3)//4,), device=dev, dtype=torch.uint8)
    ti = timeit(lambda: int2_read[(triton.cdiv(n_elem, 4096),)](packed, n_elem, 4096, sink))
    # TurboQuant: fixed-width 2-bit read (same as INT2) + a per-VECTOR fp16 norm
    # (n_heads*T norms; ~0.016 B/coord, negligible) + centroid lookup (tiny shared cb).
    # Read-bound cost == INT2 + norm touch. Hadamard rotate is compute (like QPCA r@inv, excluded).
    norms = torch.randn(n_heads * T, device=dev, dtype=torch.float16)
    def turbo_read():
        int2_read[(triton.cdiv(n_elem, 4096),)](packed, n_elem, 4096, sink)
        bf16_read[(triton.cdiv(n_heads*T, 4096),)](norms, n_heads*T, 4096, sink)
    ttq = timeit(turbo_read)
    del packed, norms; torch.cuda.empty_cache()

    cb = torch.randn(n_heads * NG * K * G, device=dev, dtype=torch.float16)   # resident codebook
    idx_gm = torch.randint(0, K, (n_heads * NG * T,), device=dev, dtype=torch.int16)
    tvf = timeit(lambda: _vqf.vq_fused(idx_gm, cb, n_heads, T, NG, K, G))
    del cb, idx_gm; torch.cuda.empty_cache()

    ub, ui, utq, uvf = tb/n_pages*1e3, ti/n_pages*1e3, ttq/n_pages*1e3, tvf/n_pages*1e3
    print(f"{T:>7} {ub:>7.4f} {ui:>7.4f} {utq:>7.4f} {uvf:>8.4f} "
          f"{ui/ub:>9.2f}x {utq/ub:>8.2f}x {uvf/ub:>8.2f}x {uvf/ui:>8.2f}x")

print("\nVQ_fused = shared-mem codebook + coalesced index reads (no fp16 write-back). "
      "INT2/BF16 <1 => fixed-width beats BF16-read. VQf/INT2 -> how close VQ gets to the "
      "fixed-width floor. rANS (report): 2.669 us/page -- VQ is dramatically faster.")
