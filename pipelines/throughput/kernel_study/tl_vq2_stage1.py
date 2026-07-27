"""Fused vq2 decode-attention stage-1 in TileLang.

Drop-in replacement for `_fwd_grouped_kernel_stage1_quant_vq2`
(decode_attention.py:2538). Same inputs, same output layout, same numerics --
the only difference is that the per-(layer, kv-head) codebook is staged in
SHARED memory and the VQ indices are fetched with vectorized int4 loads.

Why: measured on this box (NG=32, K=256, uint8 idx, sm90), the isolated gather
runs at 600 G lookups/s in Triton and 2147 G/s in TileLang -- 3.58x. Triton
cannot express an explicit shared-memory gather at all (`tl.gather` lowers to
warp shuffles and benchmarks 4x SLOWER than its scattered loads).

Config supported here: e5m2 codewords, INT2 V, no logit_cap, no xai temperature.
Those are what serve_oscar.sh sets; the Triton kernel stays the fallback.
"""
# NOTE: no `from __future__ import annotations` -- TileLang evaluates the
# T.Tensor shape annotations via get_type_hints(), and PEP 563 lazy strings
# cannot see this function's closure (NameError: name 'H_Q' is not defined).
import tilelang
import tilelang.language as T

MIN_BLOCK_KV = 32          # must match _MIN_BLOCK_KV in decode_attention.py


def _fp8e5m2_to_f16(byte):
    """e5m2 and f16 share sign and a 5-bit exponent with the same bias, so the
    decode is a pure bit shift. Verified exact over all 256 byte values."""
    return T.reinterpret("float16", T.Cast("uint16", byte) << T.Cast("uint16", 8))


@tilelang.jit
def vq2_stage1(
    BATCH: int, H_Q: int, H_KV: int, L: int, NG: int, KC: int,
    CACHE: int, TOTAL_KV: int, SPLITS: int,
    BLOCK_N: int = 64, BLOCK_H: int = 16,   # T.gemm needs M % 16 == 0
    num_stages: int = 2, threads: int = 128, ablate: str = "",
    cb_shared: bool = True,
):
    KVG = H_Q // H_KV                       # kv_group_num
    VALID_H = min(BLOCK_H, KVG)
    HEAD_BLOCKS = (H_Q + VALID_H - 1) // VALID_H
    LP = L // 4                             # packed INT2 bytes / quarter width
    NEG_INF = -3.0e38

    @T.prim_func
    def main(
        Q: T.Tensor((BATCH, H_Q, L), "float16"),
        K_Idx: T.Tensor((CACHE, H_KV, NG), "uint8"),
        CB: T.Tensor((H_KV, NG, KC), "int32"),
        V_Buf: T.Tensor((CACHE, H_KV, LP // 4), "int32"),  # 4 packed bytes/word
        K_SZ: T.Tensor((CACHE, H_KV, 2), "float32"),
        V_SZ: T.Tensor((CACHE, H_KV, 2), "float32"),
        kv_indptr: T.Tensor((BATCH + 1,), "int32"),
        kv_indices: T.Tensor((TOTAL_KV,), "int32"),
        num_kv_splits: T.Tensor((BATCH,), "int32"),
        sm_scale: T.float32,
        Att_Out: T.Tensor((BATCH, H_Q, SPLITS, L), "float32"),
        Att_Lse: T.Tensor((BATCH, H_Q, SPLITS), "float32"),
    ):
        with T.Kernel(BATCH, HEAD_BLOCKS, SPLITS, threads=threads) as (bb, bh, bs):
            kv_head = bh                                  # BLOCK_H >= KVG
            q_base = bh * VALID_H

            # Staging the codebook costs 32 KiB of shared -- a third of the
            # budget. Triton reads it straight from global/L1 and wins overall,
            # and with K register-resident the gather is ~4 us either way.
            cb_s = T.alloc_shared((NG, KC) if cb_shared else (1, 1), "int32")
            q_s = T.alloc_shared((BLOCK_H, L), "float16")
            k_f = T.alloc_fragment((BLOCK_N, L), "float16")  # registers
            v_s = T.alloc_shared((BLOCK_N, L), "float16")
            p_s = T.alloc_shared((BLOCK_N, BLOCK_H), "float16")
            # Per-token values hoisted ONCE per KV block. Reading kv_indices
            # inside the (n, g) loop fetched every slot id NG=32 times over, and
            # K_SZ inside the (h, n) loop fetched every scale BLOCK_H=16 times.
            loc_s = T.alloc_shared((BLOCK_N,), "int32")
            ksc_s = T.alloc_shared((BLOCK_N,), "float32")
            vsc_s = T.alloc_shared((BLOCK_N,), "float32")
            vze_s = T.alloc_shared((BLOCK_N,), "float32")

            qk = T.alloc_fragment((BLOCK_N, BLOCK_H), "float32")
            acc = T.alloc_fragment((BLOCK_H, L), "float32")
            m_cur = T.alloc_shared((BLOCK_H,), "float32")
            m_prev = T.alloc_shared((BLOCK_H,), "float32")
            m_scale = T.alloc_shared((BLOCK_H,), "float32")
            row_sum = T.alloc_shared((BLOCK_H,), "float32")
            lse = T.alloc_shared((BLOCK_H,), "float32")

            kv_start = kv_indptr[bb]
            seq_len = kv_indptr[bb + 1] - kv_start
            splits = num_kv_splits[bb]
            per = T.ceildiv(T.ceildiv(seq_len, splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
            s_start = per * bs
            s_end = T.min(s_start + per, seq_len)

            if s_end > s_start:
                if cb_shared:
                    T.copy(CB[kv_head, :, :], cb_s)
                # BLOCK_H is padded to 16 for the MMA, but only VALID_H rows
                # are real, so the head index is clamped as well as predicated:
                # q_base + h can otherwise run past H_Q on the last head block.
                for h, d in T.Parallel(BLOCK_H, L):
                    q_s[h, d] = T.if_then_else(
                        h < VALID_H,
                        Q[bb, T.min(q_base + h, H_Q - 1), d],
                        T.Cast("float16", 0))
                T.clear(acc)
                T.fill(lse, 0.0)
                T.fill(m_cur, NEG_INF)

                for blk in T.Pipelined(T.ceildiv(s_end - s_start, BLOCK_N),
                                       num_stages=num_stages):
                    n0 = s_start + blk * BLOCK_N

                    for n in T.Parallel(BLOCK_N):
                        ok = (n0 + n) < s_end
                        loc_s[n] = T.if_then_else(
                            ok, kv_indices[kv_start + n0 + n], 0)
                        ksc_s[n] = T.if_then_else(
                            ok, K_SZ[loc_s[n], kv_head, 0], 0.0)
                        vsc_s[n] = V_SZ[loc_s[n], kv_head, 0]
                        vze_s[n] = V_SZ[loc_s[n], kv_head, 1]

                    # ---- K: paged gather -> registers ----
                    # Iterate the FRAGMENT's own index space (BLOCK_N, L) so the
                    # write index is the loop variable; TileLang cannot lower a
                    # fragment store whose index (g*4+m) is computed instead.
                    # The per-group codeword load is hoisted by CSE across the
                    # four coords a thread owns within one group.
                    for n, d in T.Parallel(BLOCK_N, L):
                        ok = (n0 + n) < s_end
                        c = T.if_then_else(
                            ok,
                            (cb_s[d // 4,
                                  T.Cast("int32", K_Idx[loc_s[n], kv_head, d // 4])]
                             if cb_shared else
                             CB[kv_head, d // 4,
                                T.Cast("int32", K_Idx[loc_s[n], kv_head, d // 4])]),
                            0)
                        k_f[n, d] = T.if_then_else(
                            ok,
                            _fp8e5m2_to_f16(T.shift_right(c, (d % 4) * 8) & 255),
                            T.Cast("float16", 0))

                    # ---- V: INT2 unpack, affine (scale, zero) ----
                    # Read 4 packed bytes per lane as one int32: at one byte per
                    # lane a warp pulled only 32 useful bytes per instruction
                    # (measured 236 GB/s); a word per lane pulls 128.
                    for n, w in T.Parallel(BLOCK_N, LP // 4):
                        ok = (n0 + n) < s_end
                        word = 0x55555555 if "v" in ablate else \
                            V_Buf[loc_s[n], kv_head, w]
                        vsc = vsc_s[n]
                        vze = vze_s[n]
                        for b in T.unroll(4):
                            pk = T.shift_right(word, b * 8) & 255
                            for m in T.unroll(4):
                                q2 = T.Cast("float32", T.shift_right(pk, m * 2) & 3)
                                v_s[n, m * LP + w * 4 + b] = T.if_then_else(
                                    ok, T.Cast("float16", (q2 - vze) * vsc),
                                    T.Cast("float16", 0))

                    # ---- scores ----
                    T.clear(qk)
                    if "g" not in ablate:
                        T.gemm(k_f, q_s, qk, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                    for n, h in T.Parallel(BLOCK_N, BLOCK_H):
                        ok = ((n0 + n) < s_end) and (h < VALID_H)
                        # per-token ptn RMS scale folds into the score column:
                        # q . (cb * s) == (q . cb) * s
                        qk[n, h] = T.if_then_else(
                            ok, qk[n, h] * ksc_s[n] * sm_scale, NEG_INF)

                    # ---- online softmax ----
                    if "s" not in ablate:
                        T.copy(m_cur, m_prev)
                        T.reduce_max(qk, m_cur, dim=0, clear=False)
                        for h in T.Parallel(BLOCK_H):
                            m_scale[h] = T.exp(m_prev[h] - m_cur[h])
                        for n, h in T.Parallel(BLOCK_N, BLOCK_H):
                            qk[n, h] = T.exp(qk[n, h] - m_cur[h])
                        T.reduce_sum(qk, row_sum, dim=0)
                        for h in T.Parallel(BLOCK_H):
                            lse[h] = lse[h] * m_scale[h] + row_sum[h]
                        for h, d in T.Parallel(BLOCK_H, L):
                            acc[h, d] *= m_scale[h]
                    T.copy(qk, p_s)
                    if "g" not in ablate:
                        T.gemm(p_s, v_s, acc, transpose_A=True,
                               policy=T.GemmWarpPolicy.FullCol)

                # ---- epilogue ----
                for h, d in T.Parallel(BLOCK_H, L):
                    if h < VALID_H:
                        Att_Out[bb, q_base + h, bs, d] = acc[h, d] / lse[h]
                for h in T.Parallel(BLOCK_H):
                    if h < VALID_H:
                        Att_Lse[bb, q_base + h, bs] = m_cur[h] + T.log(lse[h])

    return main
