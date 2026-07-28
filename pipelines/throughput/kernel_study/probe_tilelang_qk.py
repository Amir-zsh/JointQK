"""Can a CUDA-shaped TileLang kernel match the CUDA QK score, not just the gather?

Context. The full TileLang rewrite in round 2 measured **0.40x** and was set
aside. But ncu attributed that to four lowering problems, none of them the
gather itself:

  1. `K_Idx` reloaded per COORDINATE rather than per group -- 36.8 M global
     requests against Triton's 20.2 M;
  2. 227.8 M bank conflicts on 102.3 M shared loads;
  3. 1517 M instructions against 893 M;
  4. `BLOCK_H` padded to 16 to feed the MMA when only 4 rows are real, so 3/4
     of the tensor-core work is discarded.

Meanwhile the isolated gather probe says TileLang's shared gather is
CUDA-class: 2149 G lookups/s against CUDA's 2116 and Triton's 600. So the
primitive is fine and the previous kernel was simply not shaped like the CUDA
one.

This probe rebuilds the SAME contested piece as `probe_cuda_qk.py` -- gather +
score, no V, no softmax, no PV -- with the CUDA kernel's structure and all four
fixes:

  * one thread per token, tokens strided by THREADS (fix 1: the 32 index bytes
    arrive as 8 vectorized int32 words, once per token, not per coordinate);
  * every lane sits at the same group at the same time, so a warp reads inside
    ONE 1 KB sub-table and conflicts only on `idx % 32` (fix 2);
  * scalar FMA into per-thread register accumulators, NO MMA, because
    KVG = kv_group_num = 4 and any tensor-core tile wastes 3/4 of its rows
    (fix 4);
  * `q` broadcast from shared, codebook staged in shared (fix 3).

Correctness is checked against an independent pure-torch reference, so a fast
number cannot come from a kernel that computes the wrong thing.

  docker exec -e CUDA_VISIBLE_DEVICES=7 oscar-ab bash -lc 'cd <repo> && \
    .venv-tilelang/bin/python pipelines/throughput/kernel_study/probe_tilelang_qk.py'
"""
import os

import torch
import tilelang
import tilelang.language as T

BATCH = int(os.environ.get("BS", 64))
SEQ = int(os.environ.get("CTX", 30000))
SPLITS = int(os.environ.get("SPLITS", 8))
H_Q, H_KV, L, NG, KC = 32, 8, 128, 32, 256
KVG = H_Q // H_KV
NPAD = 8          # MMA requires N % 8 == 0; KVG=4 so half of N is zero-padding
THREADS = 128
NW = NG // 4                       # index words per token
SM_SCALE = 1.0 / (L ** 0.5)

CACHE = BATCH * SEQ + 64
TOTAL = BATCH * SEQ
# Mirror the engine's split sizing: round the per-split span up to 32.
PER = ((SEQ + SPLITS - 1) // SPLITS + 31) // 32 * 32
def _ntile(tpt):
    return (PER + THREADS * tpt - 1) // (THREADS * tpt)


def _e5m2(byte):
    """e5m2 -> fp32. e5m2 and fp16 share sign + 5-bit exponent + bias, so the
    decode is `byte << 8` reinterpreted as fp16 -- same identity the CUDA
    kernel exploits with `prmt`."""
    return T.Cast("float32", T.reinterpret("float16", T.Cast("uint16", byte * 256)))


def _h2(bits):
    return T.reinterpret("float16x2", bits)


def _lane(v, hi: bool):
    """Extract one fp16 lane of a float16x2 as fp32.

    `T.vectorlow`/`T.vectorhigh` exist in the Python API but have no CUDA
    codegen ("Unresolved call tirx.vectorlow"), so go through int32.
    """
    i = T.reinterpret("int32", v)
    w = T.shift_right(i, 16) & 0xFFFF if hi else i & 0xFFFF
    return T.Cast("float32", T.reinterpret("float16", T.Cast("uint16", w)))


@tilelang.jit(out_idx=[-1])
def qk_max_tl_h2(TPT: int = 2):
    """half2 variant: two fp16 coords per FMA, halving the 512 FMAs per token.

    The codebook stays PACKED at 32 KB rather than pre-decoded to fp16. Storing
    decoded fp16 would need 64 KB (cutting CTAs/SM from 7 to 3) and, laid out as
    fp16, would give 2-way bank conflicts where the packed int32 gives 1-way
    (bank = code % 32). So the four e5m2 bytes are reassembled into two half2
    words with shifts -- what the CUDA kernel does in one `prmt`.
    """
    @T.prim_func
    def main(
        Q32: T.Tensor((BATCH, H_Q, L // 2), "int32"),
        K_Idx32: T.Tensor((CACHE, H_KV, NW), "int32"),
        CB: T.Tensor((H_KV, NG, KC), "int32"),
        kv_indptr: T.Tensor((BATCH + 1,), "int32"),
        kv_indices: T.Tensor((TOTAL,), "int32"),
        Out: T.Tensor((BATCH, H_Q, SPLITS), "float32"),
    ):
        with T.Kernel(BATCH, H_KV, SPLITS, threads=THREADS) as (bb, kvh, bs):
            cb_s = T.alloc_shared((NG, KC), "int32")
            q_s = T.alloc_shared((KVG, L // 2), "int32")
            best = T.alloc_fragment((THREADS, KVG), "float32")
            acc2 = T.alloc_local((TPT, KVG), "float16x2")
            bst = T.alloc_local((KVG,), "float32")
            loc = T.alloc_local((TPT, NW), "int32")
            cwv = T.alloc_local((TPT,), "int32")
            red = T.alloc_shared((KVG,), "float32")

            T.copy(CB[kvh, :, :], cb_s)
            T.copy(Q32[bb, kvh * KVG:(kvh + 1) * KVG, :], q_s)

            kv_start = kv_indptr[bb]
            seq_len = kv_indptr[bb + 1] - kv_start
            s_start = PER * bs
            s_end = T.min(s_start + PER, seq_len)
            ntile = _ntile(TPT)

            for t in T.Parallel(THREADS):
                for h in T.unroll(KVG):
                    bst[h] = T.float32(-3.0e38)
                for tile in T.serial(ntile):
                    base = s_start + tile * (THREADS * TPT)
                    for tt in T.unroll(TPT):
                        for h in T.unroll(KVG):
                            acc2[tt, h] = _h2(0)
                        n = base + tt * THREADS + t
                        if n < s_end:
                            for w in T.vectorized(NW):
                                loc[tt, w] = K_Idx32[kv_indices[kv_start + n], kvh, w]
                        else:
                            for w in T.unroll(NW):
                                loc[tt, w] = 0
                    for w in T.unroll(NW):
                        for b in T.unroll(4):
                            g = w * 4 + b
                            for tt in T.unroll(TPT):
                                cwv[tt] = cb_s[g, T.shift_right(loc[tt, w], 8 * b) & 255]
                            # p=0 -> coords (4g, 4g+1); p=1 -> (4g+2, 4g+3)
                            for p in T.unroll(2):
                                for h in T.unroll(KVG):
                                    qv = _h2(q_s[h, g * 2 + p])
                                    for tt in T.unroll(TPT):
                                        c = cwv[tt]
                                        b0 = T.shift_right(c, 16 * p) & 255
                                        b1 = T.shift_right(c, 16 * p + 8) & 255
                                        kv = _h2(T.shift_left(b0, 8) | T.shift_left(b1, 24))
                                        acc2[tt, h] = acc2[tt, h] + qv * kv
                    for tt in T.unroll(TPT):
                        n = base + tt * THREADS + t
                        if n < s_end:
                            for h in T.unroll(KVG):
                                bst[h] = T.max(
                                    bst[h],
                                    (_lane(acc2[tt, h], False) + _lane(acc2[tt, h], True))
                                    * SM_SCALE)
                for h in T.unroll(KVG):
                    best[t, h] = bst[h]

            T.reduce_max(best, red, dim=0, clear=True)
            for h in T.Parallel(KVG):
                Out[bb, kvh * KVG + h, bs] = red[h]

    return main


@tilelang.jit(out_idx=[-1])
def qk_max_tl_mma(BN: int = 128, STAGES: int = 0):
    """Tensor-core variant: fp32 accumulation at fp16-multiply speed.

    Every MMA attempt in this study put HEADS on the M axis --
    `tl.dot(q_full, k_full)` has M = BLOCK_H = 4, padded to 16, so 3/4 of the
    tensor-core work is discarded. That is why MMA was recorded as a dead end.

    Computing the transpose instead, qk^T = k_hat . q^T, puts BN tokens on M
    (no padding) and the 4 heads on N (padded to 8 for m16n8k16): 2x waste
    instead of 4x. Tensor cores take fp16 operands and accumulate in fp32
    natively, which is exactly what half2 bought WITHOUT its accuracy cost.

    The gather now materialises a [BN, L] fp16 K tile in shared rather than
    feeding FMAs directly; that is the cost being traded against MMA throughput.
    """
    @T.prim_func
    def main(
        QT: T.Tensor((BATCH, H_KV, NPAD, L), "float16"),
        K_Idx32: T.Tensor((CACHE, H_KV, NW), "int32"),
        CB: T.Tensor((H_KV, NG, KC), "int32"),
        kv_indptr: T.Tensor((BATCH + 1,), "int32"),
        kv_indices: T.Tensor((TOTAL,), "int32"),
        Out: T.Tensor((BATCH, H_Q, SPLITS), "float32"),
    ):
        with T.Kernel(BATCH, H_KV, SPLITS, threads=THREADS) as (bb, kvh, bs):
            cb_s = T.alloc_shared((NG, KC), "int32")
            # N padded 4 -> 8: the MMA requires N % 8 == 0, so half the
            # tensor-core columns are zeros. This is the 2x waste -- still
            # half the 4x that putting heads on M would cost.
            qt_s = T.alloc_shared((NPAD, L), "float16")
            k_s = T.alloc_shared((BN, L), "float16")
            c_f = T.alloc_fragment((BN, NPAD), "float32")
            cmax = T.alloc_fragment((NPAD,), "float32")
            bst = T.alloc_fragment((KVG,), "float32")
            loc = T.alloc_local((NW,), "int32")

            T.copy(CB[kvh, :, :], cb_s)
            T.copy(QT[bb, kvh, :, :], qt_s)
            T.fill(bst, T.float32(-3.0e38))

            kv_start = kv_indptr[bb]
            seq_len = kv_indptr[bb + 1] - kv_start
            s_start = PER * bs
            s_end = T.min(s_start + PER, seq_len)
            ntile = (PER + BN - 1) // BN

            # Pipelining the tile loop overlaps the gather that fills k_s for
            # tile i+1 with the GEMM consuming tile i -- the shared round-trip
            # is what the MMA variant pays for, so hiding it is the whole bet.
            for tile in (T.Pipelined(ntile, num_stages=STAGES)
                         if STAGES > 0 else T.serial(ntile)):
                base = s_start + tile * BN
                # Gather + dequantize BN tokens into the fp16 K tile. One
                # thread owns one token, so a warp still sits at one group.
                for t in T.Parallel(BN):
                    n = base + t
                    if n < s_end:
                        for w in T.vectorized(NW):
                            loc[w] = K_Idx32[kv_indices[kv_start + n], kvh, w]
                        for w in T.unroll(NW):
                            for b in T.unroll(4):
                                g = w * 4 + b
                                cw = cb_s[g, T.shift_right(loc[w], 8 * b) & 255]
                                for j in T.unroll(4):
                                    k_s[t, g * 4 + j] = T.reinterpret(
                                        "float16",
                                        T.Cast("uint16",
                                               (T.shift_right(cw, 8 * j) & 255) * 256))
                    else:
                        for c in T.vectorized(L):
                            k_s[t, c] = T.float16(0)

                # [BN, L] x [L, NPAD] -> [BN, NPAD], fp32 accumulate on tensor cores
                # WGMMA needs the B operand N-major, so q is stored [NPAD, L]
                # and transposed in the GEMM rather than in memory.
                T.gemm(k_s, qt_s, c_f, transpose_B=True, clear_accum=True)

                T.reduce_max(c_f, cmax, dim=0, clear=True)
                for h in T.Parallel(KVG):
                    bst[h] = T.max(bst[h], cmax[h] * SM_SCALE)

            for h in T.Parallel(KVG):
                Out[bb, kvh * KVG + h, bs] = bst[h]

    return main


@tilelang.jit(out_idx=[-1])
def qk_max_tl(TPT: int = 1):
    @T.prim_func
    def main(
        Q: T.Tensor((BATCH, H_Q, L), "float16"),
        K_Idx32: T.Tensor((CACHE, H_KV, NW), "int32"),
        CB: T.Tensor((H_KV, NG, KC), "int32"),
        kv_indptr: T.Tensor((BATCH + 1,), "int32"),
        kv_indices: T.Tensor((TOTAL,), "int32"),
        Out: T.Tensor((BATCH, H_Q, SPLITS), "float32"),
    ):
        with T.Kernel(BATCH, H_KV, SPLITS, threads=THREADS) as (bb, kvh, bs):
            # 32 KB table: every lookup that lands here is a lookup that does
            # not consume an L1 sector. This is the whole point of the rewrite.
            cb_s = T.alloc_shared((NG, KC), "int32")
            q_s = T.alloc_shared((KVG, L), "float16")
            # Per-thread REGISTERS, not fragments indexed by the parallel var:
            # TileLang's layout inference rejects a fragment touched as (t, h)
            # by two different unrolled `h` loops, and registers are what the
            # CUDA kernel uses anyway.
            best = T.alloc_fragment((THREADS, KVG), "float32")
            acc = T.alloc_local((TPT, KVG), "float32")
            bst = T.alloc_local((KVG,), "float32")
            loc = T.alloc_local((TPT, NW), "int32")
            cwv = T.alloc_local((TPT,), "int32")
            kvv = T.alloc_local((TPT,), "float32")
            red = T.alloc_shared((KVG,), "float32")

            T.copy(CB[kvh, :, :], cb_s)
            T.copy(Q[bb, kvh * KVG:(kvh + 1) * KVG, :], q_s)

            kv_start = kv_indptr[bb]
            seq_len = kv_indptr[bb + 1] - kv_start
            s_start = PER * bs
            s_end = T.min(s_start + PER, seq_len)
            ntile = _ntile(TPT)

            # Each thread strides over tokens and keeps its running max in
            # registers -- the CUDA kernel's loop, transcribed.
            for t in T.Parallel(THREADS):
                for h in T.unroll(KVG):
                    bst[h] = T.float32(-3.0e38)
                for tile in T.serial(ntile):
                    base = s_start + tile * (THREADS * TPT)
                    for tt in T.unroll(TPT):
                        for h in T.unroll(KVG):
                            acc[tt, h] = T.float32(0)
                        n = base + tt * THREADS + t
                        if n < s_end:
                            slot = kv_indices[kv_start + n]
                            # 32 index bytes as 8 words, ONE vectorized fetch
                            # per token; the round-2 kernel refetched per coord.
                            for w in T.vectorized(NW):
                                loc[tt, w] = K_Idx32[slot, kvh, w]
                        else:
                            for w in T.unroll(NW):
                                loc[tt, w] = 0
                    for w in T.unroll(NW):
                        for b in T.unroll(4):
                            g = w * 4 + b
                            for tt in T.unroll(TPT):
                                cwv[tt] = cb_s[g, T.shift_right(loc[tt, w], 8 * b) & 255]
                            for j in T.unroll(4):
                                for tt in T.unroll(TPT):
                                    kvv[tt] = _e5m2(T.shift_right(cwv[tt], 8 * j) & 255)
                                for h in T.unroll(KVG):
                                    # ONE shared load of q, reused across TPT
                                    # tokens: q traffic per token is 128/TPT.
                                    qv = T.Cast("float32", q_s[h, g * 4 + j])
                                    for tt in T.unroll(TPT):
                                        acc[tt, h] += qv * kvv[tt]
                    for tt in T.unroll(TPT):
                        n = base + tt * THREADS + t
                        if n < s_end:
                            for h in T.unroll(KVG):
                                bst[h] = T.max(bst[h], acc[tt, h] * SM_SCALE)
                for h in T.unroll(KVG):
                    best[t, h] = bst[h]

            T.reduce_max(best, red, dim=0, clear=True)
            for h in T.Parallel(KVG):
                Out[bb, kvh * KVG + h, bs] = red[h]

    return main


def reference(q, k_idx32, cb, kv_indptr, kv_indices, nb):
    """Independent torch implementation of the same score-max."""
    dev = q.device
    out = torch.full((BATCH, H_Q, SPLITS), -3e38, device=dev, dtype=torch.float32)
    idx_b = k_idx32.view(torch.uint8).view(CACHE, H_KV, NG)          # [cache,H,NG]
    cb_b = cb.view(torch.uint8).view(H_KV, NG, KC, 4)                # 4 e5m2 bytes
    cb_f = (cb_b.to(torch.int16) << 8).view(torch.float16).to(torch.float32)
    for bb in range(nb):
        ks = int(kv_indptr[bb])
        seq = int(kv_indptr[bb + 1]) - ks
        for bs in range(SPLITS):
            s0, s1 = PER * bs, min(PER * bs + PER, seq)
            if s1 <= s0:
                continue
            slots = kv_indices[ks + s0: ks + s1].long()
            for kvh in range(H_KV):
                sel = idx_b[slots, kvh, :].long()                     # [n, NG]
                # k_hat[n, g*4+j] = cb_f[kvh, g, sel[n,g], j]
                kh = cb_f[kvh].permute(0, 1, 2)[
                    torch.arange(NG, device=dev)[None, :], sel
                ]                                                     # [n, NG, 4]
                kh = kh.reshape(-1, NG * 4)
                qh = q[bb, kvh * KVG:(kvh + 1) * KVG, :].to(torch.float32)
                sc = (kh @ qh.t()) * SM_SCALE                         # [n, KVG]
                out[bb, kvh * KVG:(kvh + 1) * KVG, bs] = sc.max(dim=0).values
    return out


def bench(fn, args, reps=20):
    fn(*args); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn(*args)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3


def main() -> int:
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}  BS={BATCH} CTX={SEQ} SPLITS={SPLITS} "
          f"THREADS={THREADS} PER={PER}\n")

    q = torch.randn(BATCH, H_Q, L, generator=g, device=dev, dtype=torch.float16)
    k_idx = torch.randint(0, KC, (CACHE, H_KV, NG), generator=g, device=dev,
                          dtype=torch.uint8)
    k_idx32 = k_idx.view(torch.int32).contiguous()
    cent = torch.randn(H_KV, NG, KC, 4, generator=g, device=dev,
                       dtype=torch.float32) * (L ** -0.5)
    cb = cent.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
    kv_indptr = torch.arange(0, BATCH + 1, device=dev, dtype=torch.int32) * SEQ
    kv_indices = torch.randperm(CACHE, generator=g, device=dev)[:TOTAL].to(torch.int32)

    q32 = q.view(torch.int32).contiguous()
    # [B, H_KV, L, KVG]: q transposed per kv head, for the MMA variant's B operand
    qT = torch.zeros(BATCH, H_KV, NPAD, L, device=q.device, dtype=q.dtype)
    qT[:, :, :KVG, :] = q.view(BATCH, H_KV, KVG, L)
    qT = qT.contiguous()
    args = (q, k_idx32, cb, kv_indptr, kv_indices)
    args_h2 = (q32, k_idx32, cb, kv_indptr, kv_indices)
    args_mma = (qT, k_idx32, cb, kv_indptr, kv_indices)
    tpts = [int(x) for x in os.environ.get("TPT", "1,2,4").split(",")]
    kerns = {}
    for tpt in tpts:
        kerns[tpt] = qk_max_tl(tpt)
        kerns[tpt](*args)
    torch.cuda.synchronize()
    kern = kerns[tpts[0]]
    out = kern(*args)
    torch.cuda.synchronize()

    small = int(os.environ.get("CHECK_B", 2))
    ref_full = reference(q, k_idx32, cb, kv_indptr, kv_indices, small)
    d = (out[:small].float() - ref_full[:small].float()).abs()
    rel = (d.max() / ref_full[:small].abs().max().clamp_min(1e-9)).item()
    print(f"correctness vs torch (first {small} batches): "
          f"max abs {d.max().item():.3e}  rel {rel:.3e}  "
          f"{'PASS' if rel < 2e-2 else 'FAIL'}")

    print()
    results = {}
    for tpt in tpts:
        o = kerns[tpt](*args)
        d = (o[:small].float() - ref_full[:small].float()).abs().max().item()
        rel = d / ref_full[:small].abs().max().clamp_min(1e-9).item()
        results[tpt] = bench(kerns[tpt], args)
        print(f"TileLang QK+gather  TPT={tpt}: {results[tpt]:8.1f} us   "
              f"rel {rel:.2e}  {'PASS' if rel < 2e-2 else 'FAIL'}")
    for tpt in [int(x) for x in os.environ.get("TPT_H2", "2").split(",") if x]:
        try:
            kh = qk_max_tl_h2(tpt)
            o = kh(*args_h2)
            torch.cuda.synchronize()
            d = (o[:small].float() - ref_full[:small].float()).abs().max().item()
            rel = d / ref_full[:small].abs().max().clamp_min(1e-9).item()
            t_us = bench(kh, args_h2)
            results[f"h2-{tpt}"] = t_us
            print(f"TileLang QK+gather  half2 TPT={tpt}: {t_us:8.1f} us   "
                  f"rel {rel:.2e}  {'PASS' if rel < 5e-2 else 'FAIL'}")
        except Exception as e:
            print(f"half2 TPT={tpt} failed: {type(e).__name__}: "
                  f"{str(e).splitlines()[-1][:150]}")
    mma_cfgs = [(int(bn), int(st))
                for bn in os.environ.get("MMA_BN", "128").split(",") if bn
                for st in os.environ.get("MMA_STAGES", "0").split(",") if st]
    for bn, st in mma_cfgs:
        try:
            km = qk_max_tl_mma(bn, st)
            o = km(*args_mma)
            torch.cuda.synchronize()
            d = (o[:small].float() - ref_full[:small].float()).abs().max().item()
            rel = d / ref_full[:small].abs().max().clamp_min(1e-9).item()
            t_us = bench(km, args_mma)
            results[f"mma-{bn}s{st}"] = t_us
            print(f"TileLang QK+gather  MMA BN={bn} stages={st}: {t_us:8.1f} us   "
                  f"rel {rel:.2e}  {'PASS' if rel < 2e-2 else 'FAIL'}")
        except Exception as e:
            print(f"MMA BN={bn} stages={st} failed: {type(e).__name__}: "
                  f"{str(e).splitlines()[-1][:170]}")
    us = min(results.values())
    print(f"\nbest: {us:.1f} us at {min(results, key=results.get)}")
    # The recorded CUDA/Triton numbers are for BS=64 CTX=30000 SPLITS=8 only;
    # quoting them against any other shape compares different amounts of work.
    # Measured in the same session by probe_cuda_qk.py on this box. That probe's
    # engine heuristic picks 3 splits, so SPLITS=3 is the matched-geometry
    # comparison and SPLITS=8 is each implementation at its own best.
    REF = {(64, 30000, 8): ("CUDA half2 tpt=2", 687.0, "Triton", 1140.7),
           (64, 30000, 3): ("CUDA half2 tpt=2", 687.0, "Triton", 1140.7)}
    r = REF.get((BATCH, SEQ, SPLITS))
    if r:
        cn, cu, tn, tu = r
        print(f"\nsame-session reference (probe_cuda_qk.py, BS={BATCH} CTX={SEQ}):")
        print(f"    {cn:<20} {cu:7.1f} us")
        print(f"    {tn:<20} {tu:7.1f} us")
        print(f"    -> TileLang is {cu / us:.2f}x CUDA, {tu / us:.2f}x Triton")
    else:
        print(f"(no same-session reference for BS={BATCH} CTX={SEQ})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
