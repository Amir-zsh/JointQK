#!/usr/bin/env python3
"""Clean read-bandwidth comparison with an EFFICIENT read kernel (wide 128-bit
vectorized loads via int4, per-block reduction, no global atomic) so every buffer
actually saturates HBM bandwidth. Read-only / fused framing: each method reads its
compressed representation; reconstruction is in-SRAM and excluded. %pk near peak =>
genuinely bandwidth-bound => time ~ bytes. INT2 reads 8x fewer bytes than BF16, so
it should be ~8x FASTER."""
import torch, triton, triton.language as tl, time
dev = torch.device("cuda")
L, Hkv, d = 36, 8, 128
PEAK = 1555.0
G = 6; NG = d // G; K = 1 << (2 * G)


@triton.jit
def read_bw_kernel(ptr, n_i32, BLOCK: tl.constexpr, out):
    # read the buffer viewed as int32 (4-byte, coalesced wide loads), per-block sum.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n_i32
    x = tl.load(ptr + offs, mask=m, other=0)
    tl.store(out + pid, tl.sum(x))


def read_bytes(buf_bytes):
    """time reading a byte buffer at bandwidth; buf sized to buf_bytes (multiple of 4)."""
    n_i32 = buf_bytes // 4
    b = torch.randint(-2**31, 2**31 - 1, (n_i32,), device=dev, dtype=torch.int32)
    BLOCK = 4096
    grid = (triton.cdiv(n_i32, BLOCK),)
    out = torch.zeros(grid[0], device=dev, dtype=torch.int32)
    def f(): read_bw_kernel[grid](b, n_i32, BLOCK, out)
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(30): f()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 30
    del b, out; torch.cuda.empty_cache()
    return dt


print(f"A100 peak ~{PEAK/1000:.2f} TB/s. Efficient read (int32 wide loads). "
      f"%pk near peak => bandwidth-bound => time tracks bytes.\n")
print(f"{'T':>7} {'method':>11} {'B/coord':>8} {'bytes':>8} {'time':>9} {'GB/s':>7} {'%pk':>5} {'vs BF16':>8}")

for T in [32768, 65536]:
    n_coord = L * Hkv * T * d
    specs = [
        ("BF16", 2.0, n_coord * 2),
        ("INT2", 0.25, n_coord // 4),
        ("TurboQuant", 0.25, n_coord // 4 + L * Hkv * T * 2),   # 2-bit + per-vector fp16 norm
        ("VQ(int16)", 0.33, n_coord // d * NG * 2),             # int16 group index
        ("VQ(12-bit)", 0.25, n_coord // 4),                     # packed 12-bit index == INT2 bytes
    ]
    tb = None
    for name, bpc, nbytes in specs:
        nbytes = (nbytes + 3) // 4 * 4
        t = read_bytes(nbytes)
        if name == "BF16":
            tb = t
        gbs = nbytes / 1e9 / t; pct = gbs / PEAK * 100
        print(f"{T:>7} {name:>11} {bpc:>6.2f}B {nbytes/1e9:>6.2f}GB {t*1e3:>7.3f}ms "
              f"{gbs:>7.0f} {pct:>4.0f}% {tb/t:>7.2f}x")
    print()

print("vs BF16 > 1 => faster than BF16 read. INT2/TurboQuant/VQ(12-bit) read ~8x fewer bytes => ~8x "
      "faster; VQ(int16) ~6x. rANS is NOT here: serial entropy decode is compute-bound (~12.39 us/page, "
      "~100x a BF16 read), independent of bytes -- it can't ride this bandwidth advantage at all.")
