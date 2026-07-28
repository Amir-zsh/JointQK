"""Full TileLang vq2 stage-1 (fp32 accumulate), validated against the shipped
Triton kernel's outputs and benchmarked at the served shape.

Why this exists: the fp32 projection question could not be settled from the
contested-piece probe. TileLang beats shipped Triton there by 1.29x (883 vs
1141 us), but "full kernel = piece + remainder" is assumption-sensitive:
CUDA's remainder (1827-751 = 1076 us) projects TileLang to ~1960 us, while
Triton's own remainder (1803-1141 = 662 us) projects ~1545 us -- on opposite
sides of both the retuned Triton (1803 us) and OSCAR int2 (1636-1650 us).
So: build it, measure it.

Structure = the CUDA kernel's, in TileLang's SPMD style (`get_thread_binding`
+ explicit `sync_threads`, not `T.Parallel`): per tile of BLOCK_N tokens,
phase A computes gather+scores one thread per token (the 883 us core) and
stages packed V + scales into shared; phase B flips the thread axis to one
thread per output dim (d = tid, packed byte d%32, quarter d//32) and runs PV
with per-thread fp32 accumulators held across tiles. Softmax state (m, l,
rescale) lives in shared, updated by KVG lane-0 threads between phases.

Numerics mirror the Triton kernel exactly where rounding matters:
  * scores fp32; k ptn scale folded into columns after the dot;
  * p is fp32 for the l update but rounded to f16 for PV (Triton does
    `tl.dot(p.to(f16), v16)`);
  * V dequant in f16: (q2 - zero16) * scale16.

Golden data: logs/stage1_ref_{small,big}.pt from dump_stage1_ref.py. Invalid
splits (>= per-batch engine split count) hold 0/0 = NaN by design in BOTH
implementations and are excluded from comparison; stage 2 never reads them.

  docker exec -e CUDA_VISIBLE_DEVICES=7 oscar-ab bash -lc 'cd <repo> && \
    REF=logs/stage1_ref_small.pt .venv-tilelang/bin/python \
    pipelines/throughput/kernel_study/tilelang_stage1_full.py'
"""
import os

import torch
import tilelang
import tilelang.language as T

H_Q, H_KV, L, NG, KC = 32, 8, 128, 32, 256
KVG = H_Q // H_KV
THREADS = 128
NW = NG // 4          # K index int32 words per (token, head)
NWV = L // 16         # packed-V int32 words per (token, head)
MIN_BLOCK_KV = 32


@tilelang.jit
def stage1(BATCH: int, CACHE: int, TOTAL: int, SPLITS: int, SM_SCALE: float,
           TPT: int = 2, ABL: int = 0, MINB: int = 1):
    BLOCK_N = THREADS * TPT

    @T.prim_func
    def main(
        Q: T.Tensor((BATCH, H_Q, L), "float16"),
        KIdx32: T.Tensor((CACHE, H_KV, NW), "int32"),
        CB: T.Tensor((H_KV, NG, KC), "int32"),
        VB32: T.Tensor((CACHE, H_KV, NWV), "int32"),
        KSZ: T.Tensor((CACHE, H_KV, 2), "float32"),
        VSZ: T.Tensor((CACHE, H_KV, 2), "float32"),
        kv_indptr: T.Tensor((BATCH + 1,), "int32"),
        kv_indices: T.Tensor((TOTAL,), "int32"),
        SplitsT: T.Tensor((BATCH,), "int32"),
        Out: T.Tensor((BATCH, H_Q, SPLITS, L), "float32"),
        Lse: T.Tensor((BATCH, H_Q, SPLITS), "float32"),
    ):
        with T.Kernel(BATCH, H_KV, SPLITS, threads=THREADS) as (bb, kvh, bs):
            # TileLang emits __launch_bounds__(threads, 1) by default -- nvcc
            # then optimises for ONE resident block and spends up to 255 regs,
            # capping occupancy at 1-2 CTAs/SM. Raising minBlocks caps regs.
            T.annotate_min_blocks_per_sm(MINB)
            cb_s = T.alloc_shared((NG, KC), "int32")
            q_s = T.alloc_shared((KVG, L), "float16")
            p_s = T.alloc_shared((BLOCK_N, KVG), "float32")
            v_s = T.alloc_shared((BLOCK_N, NWV), "int32")
            vsc_s = T.alloc_shared((BLOCK_N,), "float16")
            vzr_s = T.alloc_shared((BLOCK_N,), "float16")
            tmax_s = T.alloc_shared((THREADS, KVG), "float32")
            psum_s = T.alloc_shared((THREADS, KVG), "float32")
            m_s = T.alloc_shared((KVG,), "float32")
            l_s = T.alloc_shared((KVG,), "float32")
            rs_s = T.alloc_shared((KVG,), "float32")

            accv = T.alloc_local((KVG,), "float32")
            sc = T.alloc_local((KVG,), "float32")
            mloc = T.alloc_local((KVG,), "float32")
            psl = T.alloc_local((KVG,), "float32")
            loc = T.alloc_local((NW,), "int32")
            kv4 = T.alloc_local((4,), "float32")
            q4 = T.alloc_local((4,), "float16")
            pl4 = T.alloc_local((KVG,), "float32")

            T.copy(CB[kvh, :, :], cb_s)
            T.copy(Q[bb, kvh * KVG:(kvh + 1) * KVG, :], q_s)

            tid = T.get_thread_binding()
            if tid < KVG:
                m_s[tid] = T.float32(-3.0e38)
                l_s[tid] = T.float32(0)
            for h in T.unroll(KVG):
                accv[h] = T.float32(0)

            kv_start = kv_indptr[bb]
            seq_len = kv_indptr[bb + 1] - kv_start
            ksp = SplitsT[bb]
            per = T.ceildiv(T.ceildiv(seq_len, ksp), MIN_BLOCK_KV) * MIN_BLOCK_KV
            s_start = per * bs
            s_end = T.min(s_start + per, seq_len)
            span = T.max(s_end - s_start, 0)
            ntile = T.ceildiv(span, BLOCK_N)

            # PV dim ownership: d = tid; packed byte d%32 = word jw, byte jbw;
            # quarter qm selects the 2-bit field (dims {b, b+32, b+64, b+96}).
            jw = (tid % 32) // 4
            jbw = (tid % 32) % 4
            qm = tid // 32
            T.sync_threads()

            for tile in T.serial(ntile):
                base = s_start + tile * BLOCK_N
                # -- Phase A: one thread per token: gather K, score, stage V --
                for h in T.unroll(KVG):
                    mloc[h] = T.float32(-3.0e38)
                for tt in T.unroll(TPT):
                    row = tt * THREADS + tid
                    n = base + row
                    if n < s_end:
                        slot = kv_indices[kv_start + n] if ABL not in (8, 9) else n
                        for w in T.vectorized(NW if ABL not in (8, 9) else 0):
                            loc[w] = KIdx32[slot, kvh, w]
                        for h in T.unroll(KVG):
                            sc[h] = T.float32(0)
                        for w in T.unroll(NW if ABL not in (2, 7, 8, 9) else 0):
                            for b4 in T.unroll(4):
                                g = w * 4 + b4
                                cw = cb_s[g, T.shift_right(loc[w], 8 * b4) & 255]
                                for j in T.unroll(4):
                                    kv4[j] = T.Cast(
                                        "float32",
                                        T.reinterpret(
                                            "float16",
                                            T.Cast("uint16",
                                                   (T.shift_right(cw, 8 * j) & 255)
                                                   * 256)))
                                for j in T.unroll(4):
                                    for h in T.unroll(KVG):
                                        sc[h] += (T.Cast("float32", q_s[h, g * 4 + j])
                                                  * kv4[j])
                        ksc = KSZ[slot, kvh, 0]
                        for h in T.unroll(KVG):
                            val = sc[h] * ksc * SM_SCALE
                            p_s[row, h] = val
                            mloc[h] = T.max(mloc[h], val)
                        for w in T.vectorized(NWV):
                            v_s[row, w] = VB32[slot, kvh, w]
                        vsc_s[row] = T.Cast("float16", VSZ[slot, kvh, 0])
                        vzr_s[row] = T.Cast("float16", VSZ[slot, kvh, 1])
                    else:
                        for h in T.unroll(KVG):
                            p_s[row, h] = T.float32(-3.0e38)
                        for w in T.unroll(NWV):
                            v_s[row, w] = 0
                        vsc_s[row] = T.float16(0)
                        vzr_s[row] = T.float16(0)
                for h in T.unroll(KVG):
                    tmax_s[tid, h] = mloc[h]
                if ABL != 9:
                    T.sync_threads()

                # -- A2: tile max -> running max + rescale (KVG threads) --
                if ABL != 3 and tid < KVG:
                    tmv = T.alloc_var("float32")
                    tmv = T.float32(-3.0e38)
                    for i in T.serial(THREADS):
                        tmv = T.max(tmv, tmax_s[i, tid])
                    mnew = T.max(m_s[tid], tmv)
                    rs_s[tid] = T.exp(m_s[tid] - mnew)
                    m_s[tid] = mnew
                if ABL != 9:
                    T.sync_threads()

                # -- A3: p = exp(s - m); per-thread partial sums for l --
                for h in T.unroll(KVG):
                    psl[h] = T.float32(0)
                for tt in T.unroll(TPT if ABL != 3 else 0):
                    row = tt * THREADS + tid
                    for h in T.unroll(KVG):
                        pv = T.exp(p_s[row, h] - m_s[h])
                        p_s[row, h] = pv
                        psl[h] += pv
                for h in T.unroll(KVG):
                    psum_s[tid, h] = psl[h]
                if ABL != 9:
                    T.sync_threads()

                # -- Phase B: one thread per dim: PV; l update on KVG threads --
                if tid < KVG:
                    ssv = T.alloc_var("float32")
                    ssv = T.float32(0)
                    for i in T.serial(THREADS):
                        ssv += psum_s[i, tid]
                    l_s[tid] = l_s[tid] * rs_s[tid] + ssv
                for h in T.unroll(KVG):
                    accv[h] = accv[h] * rs_s[h]
                for row in T.serial(BLOCK_N if ABL not in (1, 7, 8, 9) else 0):
                    word = v_s[row, jw]
                    q2 = T.shift_right(word, 8 * jbw + 2 * qm) & 3
                    v16 = (T.Cast("float16", q2) - vzr_s[row]) * vsc_s[row]
                    vf = T.Cast("float32", v16)
                    for h in T.vectorized(KVG):
                        pl4[h] = p_s[row, h]
                    for h in T.unroll(KVG):
                        accv[h] += T.Cast("float32", T.Cast("float16", pl4[h])) * vf
                if ABL != 9:
                    T.sync_threads()

            for h in T.unroll(KVG):
                Out[bb, kvh * KVG + h, bs, tid] = accv[h] / l_s[h]
            if tid < KVG:
                Lse[bb, kvh * KVG + tid, bs] = m_s[tid] + T.log(l_s[tid])

    return main


@tilelang.jit
def stage1_gemm_pv(BATCH: int, CACHE: int, TOTAL: int, SPLITS: int,
                   SM_SCALE: float, TPT: int = 1):
    """v2: scalar QK gather core + tensor-core PV.

    v1's all-scalar PV costs ~2300 us of remainder against Triton's 662. The
    asymmetry: PV is a K-dim reduction, which is what MMA does well -- Triton's
    `tl.dot(p16, v16)` wins there even with M=16 padding -- while QK is where
    the gather lives and scalar wins. So: phase A keeps the scalar core but
    dequantizes V into a [BLOCK_N, L] f16 tile; phase B is one T.gemm
    accumulating into a persistent [16, L] fp32 fragment.

    p is staged TRANSPOSED [16, BLOCK_N] f16 (rows 4..15 zero), so the gemm is
    row-major x row-major -- the layout-inference happy path. Rounding p to
    f16 matches Triton's `p.to(v_q0.dtype)` exactly; l still sums fp32 p.
    """
    BLOCK_N = THREADS * TPT
    MPAD = 16

    @T.prim_func
    def main(
        Q: T.Tensor((BATCH, H_Q, L), "float16"),
        KIdx32: T.Tensor((CACHE, H_KV, NW), "int32"),
        CB: T.Tensor((H_KV, NG, KC), "int32"),
        VB32: T.Tensor((CACHE, H_KV, NWV), "int32"),
        KSZ: T.Tensor((CACHE, H_KV, 2), "float32"),
        VSZ: T.Tensor((CACHE, H_KV, 2), "float32"),
        kv_indptr: T.Tensor((BATCH + 1,), "int32"),
        kv_indices: T.Tensor((TOTAL,), "int32"),
        SplitsT: T.Tensor((BATCH,), "int32"),
        Out: T.Tensor((BATCH, H_Q, SPLITS, L), "float32"),
        Lse: T.Tensor((BATCH, H_Q, SPLITS), "float32"),
    ):
        with T.Kernel(BATCH, H_KV, SPLITS, threads=THREADS) as (bb, kvh, bs):
            cb_s = T.alloc_shared((NG, KC), "int32")
            q_s = T.alloc_shared((KVG, L), "float16")
            pT_s = T.alloc_shared((MPAD, BLOCK_N), "float16")
            sc_s = T.alloc_shared((BLOCK_N, KVG), "float32")
            vt_s = T.alloc_shared((BLOCK_N, L), "float16")
            tmax_s = T.alloc_shared((THREADS, KVG), "float32")
            psum_s = T.alloc_shared((THREADS, KVG), "float32")
            m_s = T.alloc_shared((KVG,), "float32")
            l_s = T.alloc_shared((KVG,), "float32")
            rs_s = T.alloc_shared((MPAD,), "float32")

            acc_c = T.alloc_fragment((MPAD, L), "float32")
            sc = T.alloc_local((KVG,), "float32")
            mloc = T.alloc_local((KVG,), "float32")
            psl = T.alloc_local((KVG,), "float32")
            loc = T.alloc_local((NW,), "int32")

            T.copy(CB[kvh, :, :], cb_s)
            T.copy(Q[bb, kvh * KVG:(kvh + 1) * KVG, :], q_s)
            T.clear(acc_c)

            tid = T.get_thread_binding()
            if tid < KVG:
                m_s[tid] = T.float32(-3.0e38)
                l_s[tid] = T.float32(0)
            if tid < MPAD:
                rs_s[tid] = T.float32(1)
            # zero the padded p rows once; they are never written again
            for i in T.serial(MPAD - KVG):
                for c in T.vectorized(TPT):
                    pT_s[KVG + i, tid * TPT + c] = T.float16(0)

            kv_start = kv_indptr[bb]
            seq_len = kv_indptr[bb + 1] - kv_start
            ksp = SplitsT[bb]
            per = T.ceildiv(T.ceildiv(seq_len, ksp), MIN_BLOCK_KV) * MIN_BLOCK_KV
            s_start = per * bs
            s_end = T.min(s_start + per, seq_len)
            span = T.max(s_end - s_start, 0)
            ntile = T.ceildiv(span, BLOCK_N)
            T.sync_threads()

            for tile in T.serial(ntile):
                base = s_start + tile * BLOCK_N
                # -- Phase A: scalar QK + V dequant into the f16 tile --
                for h in T.unroll(KVG):
                    mloc[h] = T.float32(-3.0e38)
                for tt in T.unroll(TPT):
                    row = tt * THREADS + tid
                    n = base + row
                    if n < s_end:
                        slot = kv_indices[kv_start + n] if ABL not in (8, 9) else n
                        for w in T.vectorized(NW if ABL not in (8, 9) else 0):
                            loc[w] = KIdx32[slot, kvh, w]
                        for h in T.unroll(KVG):
                            sc[h] = T.float32(0)
                        for w in T.unroll(NW if ABL not in (2, 7, 8, 9) else 0):
                            for b4 in T.unroll(4):
                                g = w * 4 + b4
                                cw = cb_s[g, T.shift_right(loc[w], 8 * b4) & 255]
                                for j in T.unroll(4):
                                    kvf = T.Cast(
                                        "float32",
                                        T.reinterpret(
                                            "float16",
                                            T.Cast("uint16",
                                                   (T.shift_right(cw, 8 * j) & 255)
                                                   * 256)))
                                    for h in T.unroll(KVG):
                                        sc[h] += (
                                            T.Cast("float32", q_s[h, g * 4 + j])
                                            * kvf)
                        ksc = KSZ[slot, kvh, 0]
                        for h in T.unroll(KVG):
                            val = sc[h] * ksc * SM_SCALE
                            sc_s[row, h] = val
                            mloc[h] = T.max(mloc[h], val)
                        vscl = T.Cast("float16", VSZ[slot, kvh, 0])
                        vzrl = T.Cast("float16", VSZ[slot, kvh, 1])
                        for w in T.unroll(NWV):
                            wd = VB32[slot, kvh, w]
                            for b4 in T.unroll(4):
                                byte = T.shift_right(wd, 8 * b4) & 255
                                jb = w * 4 + b4
                                for qm in T.unroll(4):
                                    vt_s[row, jb + 32 * qm] = (
                                        (T.Cast("float16",
                                                T.shift_right(byte, 2 * qm) & 3)
                                         - vzrl) * vscl)
                    else:
                        for h in T.unroll(KVG):
                            sc_s[row, h] = T.float32(-3.0e38)
                        for c in T.vectorized(L):
                            vt_s[row, c] = T.float16(0)
                for h in T.unroll(KVG):
                    tmax_s[tid, h] = mloc[h]
                T.sync_threads()

                # -- A2: tile max -> running max + rescale --
                if tid < KVG:
                    tmv = T.alloc_var("float32")
                    tmv = T.float32(-3.0e38)
                    for i in T.serial(THREADS):
                        tmv = T.max(tmv, tmax_s[i, tid])
                    mnew = T.max(m_s[tid], tmv)
                    rs_s[tid] = T.exp(m_s[tid] - mnew)
                    m_s[tid] = mnew
                T.sync_threads()

                # -- A3: p = exp(s - m): f16 transposed for the gemm, fp32 sums --
                for h in T.unroll(KVG):
                    psl[h] = T.float32(0)
                for tt in T.unroll(TPT):
                    row = tt * THREADS + tid
                    for h in T.unroll(KVG):
                        pv = T.exp(sc_s[row, h] - m_s[h])
                        pT_s[h, row] = T.Cast("float16", pv)
                        psl[h] += pv
                for h in T.unroll(KVG):
                    psum_s[tid, h] = psl[h]
                if tid < KVG:
                    ssv = T.alloc_var("float32")
                    ssv = T.float32(0)
                    for i in T.serial(THREADS):
                        ssv += psum_s[i, tid]
                    l_s[tid] = l_s[tid] * rs_s[tid] + ssv
                T.sync_threads()

                # -- Phase B: rescale fragment, then tensor-core PV --
                for i, j in T.Parallel(MPAD, L):
                    acc_c[i, j] = acc_c[i, j] * rs_s[i]
                T.gemm(pT_s, vt_s, acc_c)
                T.sync_threads()

            for h in T.unroll(KVG):
                Out[bb, kvh * KVG + h, bs, tid] = T.float32(0)
            T.sync_threads()
            for i, j in T.Parallel(KVG, L):
                Out[bb, kvh * KVG + i, bs, j] = acc_c[i, j] / l_s[i]
            if tid < KVG:
                Lse[bb, kvh * KVG + tid, bs] = m_s[tid] + T.log(l_s[tid])

    return main


@tilelang.jit
def stage1_tile(BATCH: int, CACHE: int, TOTAL: int, SPLITS: int,
                SM_SCALE: float, TPT: int = 1, NTS: int = 0, STAGES: int = 0):
    """v4: Triton's structure, pure tile style, shared-codebook gather.

    The ncu profile of the scalar kernel settles the diagnosis: 84 regs, zero
    local traffic, issue_active 74% -- instruction-THROUGHPUT bound at 2.14 G
    executed instructions vs Triton's 893 M. A scalar FMA is 32 MACs per warp
    instruction; an HMMA is 2048. Both dots must be MMA to compete.

    v2 mixed raw SPMD code with T.gemm and lost; the hypothesis here is that
    the SPMD style itself (get_thread_binding + manual barriers) defeats the
    block-level pipeliner. v4 is written entirely in block ops -- T.Parallel
    producers, T.gemm dots, fragment softmax (the flashattn-decode pattern) --
    with the one thing Triton cannot express: the codeword gather reads a
    SHARED-memory codebook.

    BLOCK_H = 16 rows (4 real heads + 12 zero) exactly as Triton pads.
    """
    BLOCK_N = 128 * TPT
    MPAD = 16

    @T.prim_func
    def main(
        Q: T.Tensor((BATCH, H_Q, L), "float16"),
        KIdx32: T.Tensor((CACHE, H_KV, NW), "int32"),
        CB: T.Tensor((H_KV, NG, KC), "int32"),
        VB32: T.Tensor((CACHE, H_KV, NWV), "int32"),
        KSZ: T.Tensor((CACHE, H_KV, 2), "float32"),
        VSZ: T.Tensor((CACHE, H_KV, 2), "float32"),
        kv_indptr: T.Tensor((BATCH + 1,), "int32"),
        kv_indices: T.Tensor((TOTAL,), "int32"),
        SplitsT: T.Tensor((BATCH,), "int32"),
        Out: T.Tensor((BATCH, H_Q, SPLITS, L), "float32"),
        Lse: T.Tensor((BATCH, H_Q, SPLITS), "float32"),
    ):
        with T.Kernel(BATCH, H_KV, SPLITS, threads=128) as (bb, kvh, bs):
            cb_s = T.alloc_shared((NG, KC), "int32")
            q_s = T.alloc_shared((MPAD, L), "float16")
            kt_s = T.alloc_shared((BLOCK_N, L), "float16")
            vt_s = T.alloc_shared((BLOCK_N, L), "float16")
            ksc_s = T.alloc_shared((BLOCK_N,), "float32")

            qk = T.alloc_fragment((MPAD, BLOCK_N), "float32")
            # p goes through SHARED: the qk accumulator fragment and the gemm-A
            # operand fragment have incompatible thread layouts, so an
            # elementwise fragment->fragment cast fails layout inference.
            p16_s = T.alloc_shared((MPAD, BLOCK_N), "float16")
            acc = T.alloc_fragment((MPAD, L), "float32")
            mst = T.alloc_fragment((MPAD,), "float32")
            lst = T.alloc_fragment((MPAD,), "float32")
            tmax = T.alloc_fragment((MPAD,), "float32")
            rs = T.alloc_fragment((MPAD,), "float32")
            psum = T.alloc_fragment((MPAD,), "float32")

            T.copy(CB[kvh, :, :], cb_s)
            for i, j in T.Parallel(MPAD, L):
                q_s[i, j] = T.if_then_else(
                    i < KVG, Q[bb, kvh * KVG + i, j], T.float16(0))
            T.clear(acc)
            T.fill(mst, T.float32(-3.0e38))
            T.fill(lst, T.float32(0))

            kv_start = kv_indptr[bb]
            seq_len = kv_indptr[bb + 1] - kv_start
            ksp = SplitsT[bb]
            per = T.ceildiv(T.ceildiv(seq_len, ksp), MIN_BLOCK_KV) * MIN_BLOCK_KV
            s_start = per * bs
            s_end = T.min(s_start + per, seq_len)
            span = T.max(s_end - s_start, 0)
            ntile = T.ceildiv(span, BLOCK_N)

            # NTS > 0: compile-time trip count (bench shapes are uniform), the
            # only form T.Pipelined can software-pipeline. Masking makes the
            # padded iterations no-ops.
            for tile in (T.Pipelined(NTS, num_stages=STAGES) if NTS > 0
                         else T.serial(ntile)):
                base = s_start + tile * BLOCK_N
                # gather + decode K and V into f16 tiles; the codebook lookup
                # is the shared-memory gather Triton cannot express
                for i, w in T.Parallel(BLOCK_N, NW):
                    if base + i < s_end:
                        slot = kv_indices[kv_start + base + i]
                        wd = KIdx32[slot, kvh, w]
                        vw = VB32[slot, kvh, w]
                        vsc = T.Cast("float16", VSZ[slot, kvh, 0])
                        vzr = T.Cast("float16", VSZ[slot, kvh, 1])
                        for b4 in T.unroll(4):
                            g = w * 4 + b4
                            cw = cb_s[g, T.shift_right(wd, 8 * b4) & 255]
                            for j in T.unroll(4):
                                kt_s[i, g * 4 + j] = T.reinterpret(
                                    "float16",
                                    T.Cast("uint16",
                                           (T.shift_right(cw, 8 * j) & 255)
                                           * 256))
                            jb = w * 4 + b4
                            for qm in T.unroll(4):
                                vt_s[i, jb + 32 * qm] = (
                                    (T.Cast("float16",
                                            T.shift_right(vw, 8 * b4 + 2 * qm)
                                            & 3) - vzr) * vsc)
                    else:
                        for c in T.unroll(16):
                            kt_s[i, w * 16 + c] = T.float16(0)
                            vt_s[i, w * 16 + c] = T.float16(0)
                for i in T.Parallel(BLOCK_N):
                    ksc_s[i] = T.if_then_else(
                        base + i < s_end,
                        KSZ[kv_indices[kv_start + base + i], kvh, 0],
                        T.float32(0))

                T.gemm(q_s, kt_s, qk, transpose_B=True, clear_accum=True)

                for i, j in T.Parallel(MPAD, BLOCK_N):
                    qk[i, j] = T.if_then_else(
                        base + j < s_end,
                        qk[i, j] * ksc_s[j] * SM_SCALE,
                        T.float32(-3.0e38))
                T.reduce_max(qk, tmax, dim=1, clear=True)
                for i in T.Parallel(MPAD):
                    tmax[i] = T.max(tmax[i], mst[i])
                    rs[i] = T.exp(mst[i] - tmax[i])
                    mst[i] = tmax[i]
                for i, j in T.Parallel(MPAD, BLOCK_N):
                    qk[i, j] = T.exp(qk[i, j] - mst[i])
                T.reduce_sum(qk, psum, dim=1, clear=True)
                for i in T.Parallel(MPAD):
                    lst[i] = lst[i] * rs[i] + psum[i]
                for i, j in T.Parallel(MPAD, BLOCK_N):
                    p16_s[i, j] = T.Cast("float16", qk[i, j])
                for i, j in T.Parallel(MPAD, L):
                    acc[i, j] = acc[i, j] * rs[i]
                T.gemm(p16_s, vt_s, acc)

            for i, j in T.Parallel(MPAD, L):
                if i < KVG:
                    Out[bb, kvh * KVG + i, bs, j] = acc[i, j] / lst[i]
            for i in T.Parallel(MPAD):
                if i < KVG:
                    Lse[bb, kvh * KVG + i, bs] = mst[i] + T.log(lst[i])

    return main


def bench(fn, args, reps=20):
    fn(*args)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn(*args)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3


def main() -> int:
    ref = torch.load(os.environ.get("REF", "logs/stage1_ref_small.pt"),
                     weights_only=False)
    dev = "cuda"
    B, SEQ, SP = ref["BATCH"], ref["SEQ"], ref["SPLITS"]
    q = ref["q"].to(dev)
    k_idx32 = ref["k_idx"].to(dev).view(torch.int32).contiguous()
    cb = ref["cb"].to(dev)
    vb32 = ref["v_buf"].to(dev).view(torch.int32).contiguous()
    ksz = ref["k_sz"].to(dev).float().contiguous()
    vsz = ref["v_sz"].to(dev).float().contiguous()
    kv_indptr = ref["kv_indptr"].to(dev)
    kv_indices = ref["kv_indices"].to(dev)
    splits = ref["splits"].to(dev)
    CACHE, TOTAL = k_idx32.shape[0], kv_indices.shape[0]
    v = ref["splits_used"]
    print(f"{torch.cuda.get_device_name(0)}  bs={B} ctx={SEQ} max_splits={SP} "
          f"valid_splits={v}")

    out = torch.zeros(B, H_Q, SP, L, device=dev, dtype=torch.float32)
    lse = torch.zeros(B, H_Q, SP, device=dev, dtype=torch.float32)
    tpt = int(os.environ.get("TPT", 2))
    style = os.environ.get("STYLE", "scalar")
    maker = stage1_gemm_pv if style == "gemm" else stage1
    abl = int(os.environ.get("ABL", 0))
    minb = int(os.environ.get("MINB", 1))
    if style == "scalar":
        kern = stage1(B, CACHE, TOTAL, SP, ref["SM_SCALE"], tpt, abl, minb)
    elif style == "tile":
        nts = int(os.environ.get("NTS", 0))
        stg = int(os.environ.get("STAGES", 0))
        kern = stage1_tile(B, CACHE, TOTAL, SP, ref["SM_SCALE"], tpt, nts, stg)
    else:
        kern = maker(B, CACHE, TOTAL, SP, ref["SM_SCALE"], tpt)
    args = (q, k_idx32, cb, vb32, ksz, vsz, kv_indptr, kv_indices, splits,
            out, lse)
    kern(*args)
    torch.cuda.synchronize()

    ro, rl = ref["out"].to(dev), ref["lse"].to(dev)
    go, gl = out[:, :, :v, :], ro[:, :, :v, :]
    if int(os.environ.get("ABL", 0)) == 0:
        assert torch.isfinite(go).all(), "NaN/Inf in valid Out region"
    d = (go - gl).abs()
    rel = (d.max() / gl.abs().max().clamp_min(1e-9)).item()
    dl = (lse[:, :, :v] - rl[:, :, :v]).abs().max().item()
    ok = rel < 2e-3 and dl < 1e-3
    print(f"Out  vs Triton golden: max abs {d.max().item():.3e}  rel {rel:.3e}")
    print(f"Lse  vs Triton golden: max abs {dl:.3e}")
    print("PARITY " + ("PASS" if ok else "FAIL"))
    if not ok and int(os.environ.get("ABL", 0)) == 0:
        return 1

    us = bench(kern, args)
    print(f"\nTileLang full stage-1 (fp32, {style}, TPT={tpt}, MINB={os.environ.get('MINB', 1)}): {us:8.1f} us")
    if (B, SEQ) == (64, 30000):
        print("same-box references, full stage-1, fp32, bs=64/ctx=30k:")
        print("    OSCAR int2 (Triton)      1636-1650 us")
        print("    vq2 Triton retuned       1803 us")
        print("    vq2 CUDA fp32            1827 us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
