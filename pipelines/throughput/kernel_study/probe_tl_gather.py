"""Can `tl.gather` give Triton the shared-memory codebook gather after all?

The report concluded Triton cannot express a per-lane, data-dependent gather out
of shared memory, so the CUDA kernel's one advantage was unreachable. That
conclusion was based on (a) an earlier `tl.gather` probe that measured 4x slower
than scattered global loads and (b) Gluon's `shared_memory_descriptor` indexing
only by uniform scalar.

(a) was measured without checking WHICH lowering it got. Triton's
`GatherLoweringHelper` picks between two: warp shuffles when the gather is
warp-local, and a SHARED-MEMORY SCRATCH path otherwise. Both
`isWarpLocal()` and `getScratchSizeInBytes()` are present in our 3.6.0 binary.
A 4x-slower result is the signature of the shuffle path -- which says nothing
about the shared path.

And the lookup does fit `tl.gather`'s signature, which was the thing missed:

    cb_tile = [KC, NG]              codebook, loop-invariant
    isel    = [BLOCK_N, NG]         per-token group indices
    tl.gather(cb_tile, isel, axis=0) -> out[n, g] = cb_tile[isel[n, g], g]

ranks match, and the non-axis dim matches (NG). This measures the two gathers at
the real decode shape and reports which lowering each got, by counting LDS/LDG
in the SASS -- the number that actually decides it, not the wall clock alone.

  docker exec -e CUDA_VISIBLE_DEVICES=6 oscar-ab bash -lc 'cd <repo> && \
    PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
    /opt/venv-oscar/bin/python -u pipelines/throughput/kernel_study/probe_tl_gather.py'
"""
import collections
import os
import re
import subprocess
import tempfile

import torch
import triton
import triton.language as tl

BLOCK_N, NG, KC = 64, 32, 256
REP = 32          # repeat the gather so it dominates launch + store
NTOK = 30000


@triton.jit
def _gather_global(K_IDX, CB, OUT, n_tok,
                   NG: tl.constexpr, KC: tl.constexpr, BLOCK_N: tl.constexpr,
                   REP: tl.constexpr):
    """What ships today: address computed per element, load from GLOBAL."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, NG)
    mask = offs_n < n_tok
    isel = tl.load(K_IDX + offs_n[:, None] * NG + offs_g[None, :],
                   mask=mask[:, None], other=0).to(tl.int32)
    acc = tl.zeros([BLOCK_N], dtype=tl.int32)
    for _ in range(REP):
        cw = tl.load(CB + offs_g[None, :] * KC + isel, mask=mask[:, None], other=0)
        acc += tl.sum(cw, axis=1)
        isel = (isel + 1) % KC
    tl.store(OUT + offs_n, acc, mask=mask)


@triton.jit
def _gather_tl(K_IDX, CB_T, OUT, n_tok,
               NG: tl.constexpr, KC: tl.constexpr, BLOCK_N: tl.constexpr,
               REP: tl.constexpr):
    """Staged: load the whole [KC, NG] table as a tile, then tl.gather it.

    CB_T is the codebook TRANSPOSED to [KC, NG] so the gather axis (the codeword
    index) is axis 0 and the non-gathered axis (group) matches isel's.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, NG)
    offs_k = tl.arange(0, KC)
    mask = offs_n < n_tok
    cb_tile = tl.load(CB_T + offs_k[:, None] * NG + offs_g[None, :])
    isel = tl.load(K_IDX + offs_n[:, None] * NG + offs_g[None, :],
                   mask=mask[:, None], other=0).to(tl.int32)
    acc = tl.zeros([BLOCK_N], dtype=tl.int32)
    for _ in range(REP):
        cw = tl.gather(cb_tile, isel, axis=0)
        acc += tl.sum(cw, axis=1)
        isel = (isel + 1) % KC
    tl.store(OUT + offs_n, acc, mask=mask)


def sass_hist(cubin: bytes) -> collections.Counter:
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        out = subprocess.run(["nvdisasm", "-c", path],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return collections.Counter()
    finally:
        os.unlink(path)
    c = collections.Counter()
    for line in out.splitlines():
        m = re.search(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m:
            op = m.group(1).split(".")[0]
            c[op] += 1
    return c


def compiled(fn) -> dict:
    for _, tup in (getattr(fn, "device_caches", None) or {}).items():
        kc = tup[0] if isinstance(tup, (tuple, list)) else tup
        for _, ck in kc.items():
            return ck.asm
    return {}


def bench(fn, reps=100):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3


def main() -> int:
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}  BLOCK_N={BLOCK_N} NG={NG} KC={KC} "
          f"n_tok={NTOK}\n")

    k_idx = torch.randint(0, KC, (NTOK, NG), generator=g, device=dev,
                          dtype=torch.uint8)
    cb = torch.randint(-2**31, 2**31 - 1, (NG, KC), generator=g, device=dev,
                       dtype=torch.int32)
    cb_t = cb.t().contiguous()                      # [KC, NG]
    out_a = torch.zeros(NTOK, device=dev, dtype=torch.int32)
    out_b = torch.zeros(NTOK, device=dev, dtype=torch.int32)
    grid = (triton.cdiv(NTOK, BLOCK_N),)

    def run_a():
        _gather_global[grid](k_idx, cb, out_a, NTOK, NG=NG, KC=KC,
                             BLOCK_N=BLOCK_N, REP=REP, num_warps=4, num_stages=3)

    def run_b():
        _gather_tl[grid](k_idx, cb_t, out_b, NTOK, NG=NG, KC=KC,
                         BLOCK_N=BLOCK_N, REP=REP, num_warps=4, num_stages=3)

    run_a()
    try:
        run_b()
    except Exception as e:
        print(f"tl.gather FAILED to compile: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:200]}")
        return 1
    torch.cuda.synchronize()

    same = torch.equal(out_a, out_b)
    print(f"correctness: tl.gather matches the global gather: {same}")
    if not same:
        d = (out_a != out_b).sum().item()
        print(f"  MISMATCH in {d} / {out_a.numel()} elements -- "
              f"the two formulations are not the same lookup")

    for tag, fn, jit in (("global gather (shipped)", run_a, _gather_global),
                         ("tl.gather from tile", run_b, _gather_tl)):
        us = bench(fn)
        asm = compiled(jit)
        h = sass_hist(asm.get("cubin", b""))
        smem = asm.get("shared", "?") if isinstance(asm, dict) else "?"
        try:
            md = [v for v in (getattr(jit, "device_caches", None) or {}).values()]
            smem = list(list(md[0][0].values())[0].metadata.shared for _ in [0])[0]
        except Exception:
            smem = "?"
        print(f"\n{tag}: {us:.1f} us   shared={smem} B")
        if h:
            mem = {k: v for k, v in h.items()
                   if k in ("LDG", "LDS", "LDSM", "STS", "SHFL", "PRMT",
                            "LDL", "STL", "BAR", "STG")}
            print("  memory ops:", mem)
            print("  top:", dict(sorted(h.items(), key=lambda x: -x[1])[:8]))
        else:
            print("  (nvdisasm unavailable)")
    print("\nRead: LDS>0 with SHFL==0 means the shared-scratch lowering; "
          "SHFL>0 means the warp-shuffle lowering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
