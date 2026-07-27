"""Is the vq2 codebook gather overlapped with its DRAM traffic, or additive?

Three variants over the same loop shape, with uint8 indices (what vq2 actually
stores -- an earlier probe used int32 and was itself DRAM-bound, which made the
gather look closer to a hardware ceiling than it is):

  A idx_only     load indices, no gather        -> DRAM + issue cost alone
  B gather_only  gather with loop-invariant idx -> L1/LSU cost alone
  C idx_gather   load indices, then gather      -> the real shape

If C ~ max(A, B) the units are already overlapped and there is nothing to win.
If C ~ A + B they are serialized and pipelining is worth real time.

  docker exec oscar-ab bash -lc '<repo> && /opt/venv-oscar/bin/python logs/probe_overlap.py'
"""
import torch
import triton
import triton.language as tl

GRID = 528          # 132 SMs x 4 CTAs -- saturated, not latency-bound


@triton.jit
def a_idx_only(idx_ptr, cb_ptr, out_ptr, S,
               NG: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    idx_ptr += pid * NG * S
    out_ptr += pid * NG * BLOCK_N
    g = tl.arange(0, NG)[:, None]
    acc = tl.zeros([NG, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, S, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)[None, :]
        idx = tl.load(idx_ptr + g * S + n).to(tl.int32)
        acc += idx.to(tl.float32)
    o = tl.arange(0, NG)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr + o, acc)


@triton.jit
def b_gather_only(idx_ptr, cb_ptr, out_ptr, S,
                  NG: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cb_ptr += pid * NG * K
    out_ptr += pid * NG * BLOCK_N
    g = tl.arange(0, NG)[:, None]
    acc = tl.zeros([NG, BLOCK_N], dtype=tl.float32)
    # Index derived on-chip: same scattered access pattern, zero DRAM traffic.
    base = (tl.arange(0, BLOCK_N)[None, :] * 37 + g * 11) % K
    for start_n in range(0, S, BLOCK_N):
        idx = (base + start_n) % K
        acc += tl.load(cb_ptr + g * K + idx)
    o = tl.arange(0, NG)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr + o, acc)


@triton.jit
def c_idx_gather(idx_ptr, cb_ptr, out_ptr, S,
                 NG: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    idx_ptr += pid * NG * S
    cb_ptr += pid * NG * K
    out_ptr += pid * NG * BLOCK_N
    g = tl.arange(0, NG)[:, None]
    acc = tl.zeros([NG, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, S, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)[None, :]
        idx = tl.load(idx_ptr + g * S + n).to(tl.int32)
        acc += tl.load(cb_ptr + g * K + idx)
    o = tl.arange(0, NG)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr + o, acc)


def main() -> int:
    NG, K, BLOCK_N, S = 32, 256, 128, 32768
    dev = "cuda"
    cb = torch.randn(GRID, NG, K, device=dev, dtype=torch.float32)
    idx = torch.randint(0, K, (GRID, NG, S), device=dev, dtype=torch.uint8)
    out = torch.empty(GRID, NG, BLOCK_N, device=dev, dtype=torch.float32)
    lookups = GRID * NG * S
    dram_gb = idx.numel() / 1e9        # uint8: 1 B per index

    print(f"NG={NG} K={K} BLOCK_N={BLOCK_N} S={S} GRID={GRID}  "
          f"uint8 indices, {dram_gb:.2f} GB of index traffic")
    res = {}
    for stages in (2, 3, 4):
        print(f"\n--- num_stages={stages} ---")
        for name, fn in (("A idx_only    ", a_idx_only),
                         ("B gather_only ", b_gather_only),
                         ("C idx_gather  ", c_idx_gather)):
            ms = triton.testing.do_bench(
                lambda: fn[(GRID,)](idx, cb, out, S, NG=NG, K=K,
                                    BLOCK_N=BLOCK_N, num_warps=8,
                                    num_stages=stages),
                warmup=25, rep=100)
            res[(name.strip()[0], stages)] = ms
            bw = dram_gb / (ms / 1e3) if name.startswith("A") or name.startswith("C") else 0
            print(f"  {name} {ms*1e3:8.1f} us  {lookups/ms/1e6:7.1f} G lookups/s"
                  + (f"  {bw:5.2f} TB/s" if bw else ""))
        a, b, c = (res[(x, stages)] for x in "ABC")
        print(f"  serialized would be A+B = {(a+b)*1e3:.1f} us; "
              f"perfectly overlapped = max(A,B) = {max(a,b)*1e3:.1f} us; "
              f"actual C = {c*1e3:.1f} us")
        span = (a + b) - max(a, b)
        frac = 0.0 if span <= 0 else ((a + b) - c) / span
        print(f"  -> overlap achieved: {frac*100:.0f}% "
              f"(0% = fully serialized, 100% = fully hidden)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
