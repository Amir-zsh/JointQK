"""Can TileLang reach the CUDA gather rate, or does it stall near Triton's?

Reference points measured on this box (NG=32, K=256, uint8 indices, sm90):
    Triton, global/L1 (today's vq2)          600 G lookups/s
    CUDA,   global/L1                       1232
    CUDA,   shared memory                   1649
    CUDA,   shared + vectorized index load  2116

The question is whether TileLang's explicit shared memory (T.alloc_shared) plus
data-dependent indexing gets us CUDA-class numbers while staying in Python. If
it lands near 2000 it is the right vehicle for rewriting vq2 decode attention;
if it lands near 600 we write CUDA instead.

  docker exec oscar-ab bash -lc '<repo> && .venv-tilelang/bin/python logs/probe_tilelang_gather.py'
"""
import torch
import tilelang
import tilelang.language as T

GRID, THREADS = 528, 256
NG, KCB, ITER = 32, 256, 256
S = ITER * THREADS


@tilelang.jit(out_idx=[-1])
def gather_global(unroll: bool = False):
    @T.prim_func
    def main(idx: T.Tensor((GRID, NG, S), "uint8"),
             cb: T.Tensor((GRID, NG, KCB), "int32"),
             out: T.Tensor((GRID, THREADS), "int32")):
        with T.Kernel(GRID, threads=THREADS) as bx:
            acc = T.alloc_fragment((THREADS,), "int32")
            T.clear(acc)
            for it in T.serial(ITER):
                for t in T.Parallel(THREADS):
                    for g in (T.unroll(NG) if unroll else T.serial(NG)):
                        acc[t] += cb[bx, g, idx[bx, g, it * THREADS + t]]
            T.copy(acc, out[bx, :])

    return main


@tilelang.jit(out_idx=[-1])
def gather_shared(unroll: bool = False):
    @T.prim_func
    def main(idx: T.Tensor((GRID, NG, S), "uint8"),
             cb: T.Tensor((GRID, NG, KCB), "int32"),
             out: T.Tensor((GRID, THREADS), "int32")):
        with T.Kernel(GRID, threads=THREADS) as bx:
            # The capability Triton lacks: an explicit shared buffer we can
            # index with a data-dependent value. 32 KiB -> 7 CTAs/SM.
            scb = T.alloc_shared((NG, KCB), "int32")
            acc = T.alloc_fragment((THREADS,), "int32")
            T.copy(cb[bx, :, :], scb)
            T.clear(acc)
            for it in T.serial(ITER):
                for t in T.Parallel(THREADS):
                    for g in (T.unroll(NG) if unroll else T.serial(NG)):
                        acc[t] += scb[g, idx[bx, g, it * THREADS + t]]
            T.copy(acc, out[bx, :])

    return main



@tilelang.jit(out_idx=[-1])
def gather_shared_vec(unroll: bool = True):
    """Shared codebook + vectorized index fetch.

    The arena is already [slot, head, ng], so a token's NG indices are
    contiguous. Viewing them as int32 lets one thread pull all 32 with 8 word
    loads instead of 32 byte loads; bytes come back out with shifts. This is
    the portable form of the CUDA uint4 variant (2116 G lookups/s).
    """
    @T.prim_func
    def main(idx32: T.Tensor((GRID, S, NG // 4), "int32"),
             cb: T.Tensor((GRID, NG, KCB), "int32"),
             out: T.Tensor((GRID, THREADS), "int32")):
        with T.Kernel(GRID, threads=THREADS) as bx:
            scb = T.alloc_shared((NG, KCB), "int32")
            acc = T.alloc_fragment((THREADS,), "int32")
            T.copy(cb[bx, :, :], scb)
            T.clear(acc)
            # Two phases: a VECTORIZED load of the 8 index words into a
            # per-thread local buffer (TileLang emits int4 loads for this shape),
            # then an unrolled compute phase over the bytes. Fusing the two, as
            # a single loop, makes TileLang fall back to 8 scalar loads.
            loc = T.alloc_local((NG // 4,), "int32")
            for it in T.serial(ITER):
                for t in T.Parallel(THREADS):
                    for w in T.vectorized(NG // 4):
                        loc[w] = idx32[bx, it * THREADS + t, w]
                    for w in T.unroll(NG // 4):
                        for b in T.unroll(4):
                            acc[t] += scb[w * 4 + b,
                                          T.shift_right(loc[w], 8 * b) & 255]
            T.copy(acc, out[bx, :])

    return main


def bench(fn, idx, cb, reps=20):
    fn(idx, cb)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn(idx, cb)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def dump() -> int:
    """Print the CUDA TileLang actually generated, so a slow number can be
    attributed to the tool rather than to my kernel being wrong."""
    for label, maker in (("SHARED+VECIDX", gather_shared_vec),
                         ("SHARED/unrolled", gather_shared)):
        src = maker(True).get_kernel_source()
        print(f"\n===== {label}: __shared__ present = {'__shared__' in src} =====")
        keep = ("__shared__", "#pragma", "for (", "__syncthreads",
                "__launch_bounds__", "int4", "make_int4", "*(int4*", "= idx32",
                ">>", "& 255")
        for l in src.splitlines():
            if any(t in l for t in keep):
                print("   ", l.strip()[:118])
    return 0


def main() -> int:
    import sys
    if "--dump" in sys.argv:
        return dump()
    dev = "cuda"
    idx = torch.randint(0, KCB, (GRID, NG, S), device=dev, dtype=torch.uint8)
    cb = torch.ones(GRID, NG, KCB, device=dev, dtype=torch.int32)
    # token-major copy, viewed as int32 words: 4 indices per word
    idx_tok = idx.permute(0, 2, 1).contiguous()          # [GRID, S, NG]
    idx32 = idx_tok.view(torch.int32)                     # [GRID, S, NG//4]
    lookups = GRID * S * NG

    print(f"NG={NG} K={KCB} GRID={GRID} threads={THREADS} S={S}  "
          f"({lookups/1e9:.2f} G lookups/launch)")
    expect = NG * ITER          # codebook is all ones
    print(f"{'variant':38s} {'us':>9} {'G lookups/s':>13} {'vs Triton 600':>14} {'ok':>4}")
    for name, maker, unroll in (
            ("global / L1, serial", gather_global, False),
            ("global / L1, unrolled", gather_global, True),
            ("shared memory, serial", gather_shared, False),
            ("shared memory, unrolled", gather_shared, True),
            ("shared + vectorized idx", gather_shared_vec, True)):
        try:
            k = maker(unroll)
            arg = idx32 if maker is gather_shared_vec else idx
            res = k(arg, cb)
            good = bool((res == expect).all().item())
            ms = bench(k, arg, cb)
            rate = lookups / (ms * 1e-3) / 1e9
            print(f"  {name:36s} {ms*1e3:9.1f} {rate:13.1f} {rate/600:13.2f}x "
                  f"{'PASS' if good else 'FAIL(' + str(res.flatten()[0].item()) + ')':>4}")
        except Exception as e:
            msg = str(e).splitlines()[0][:90] if str(e) else type(e).__name__
            print(f"  {name:36s}  FAILED: {type(e).__name__}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
