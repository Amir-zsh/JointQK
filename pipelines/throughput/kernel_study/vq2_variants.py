"""Group-major vq2 stage-1 variants, benched against the engine's own kernels.

The shipped kernel loads `isel` as [BLOCK_N, NG]. Triton maps a warp's 32 lanes
along the trailing axis, so the codebook gather `cb[g*KC + isel]` sends every
lane into a *different* group sub-table, 1 KB apart -- 32 guaranteed-distinct
L1 sectors per warp instruction, the worst case for the LSU's address-divergence
processing. That gather is 52% of the kernel (1009 us of 1934 at bs=64/30k).

Group-major flips `isel` to [NG, BLOCK_N]: the 32 lanes then land inside ONE
group's 1 KB sub-table (32 sectors of 8 codewords each), so the expected number
of distinct sectors drops to 32*(1-(31/32)^32) ~= 20.4. That is ~1.57x fewer
wavefronts for the same lookups -- the only lever the earlier sweep left untried.

The previously-recorded blocker was that `tl.join` interleaves the trailing axis,
so group-major planes cannot be woven back into [BLOCK_N, L]. That blocker is
avoidable: DON'T rebuild the [BLOCK_N, L] tile at all. With plane m holding coord
4g+m, the score decomposes exactly as

    qk[h,n] = sum_c q[h,c] k[n,c] = sum_m sum_g q[h,4g+m] * p_m[g,n]
            = sum_m dot(q_m, p_m)      with q_m[h,g] = Q[h, 4g+m]

so four K=NG dots replace one K=L dot, the planes are consumed in their native
[NG, BLOCK_N] layout, and `tl.trans(kg)` disappears too. The q slices are strided
loads hoisted out of the KV loop. No codebook relabeling and no new bundle -- the
strided-coordinate layout the study notes assumed would be required is NOT needed.

Run (GPU 0):
  docker exec -e CUDA_VISIBLE_DEVICES=0 oscar-ab bash -lc 'cd <repo> && \
    PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
    /opt/venv-oscar/bin/python pipelines/throughput/kernel_study/vq2_variants.py'
"""
import itertools
import os

import torch
import triton
import triton.language as tl

BATCH = int(os.environ.get("BS", 64))
SEQ = int(os.environ.get("CTX", 30000))
SPLITS = int(os.environ.get("SPLITS", 8))
H_Q, H_KV, L, NG, KC = 32, 8, 128, 32, 256
SM_SCALE = 1.0 / (L ** 0.5)
_MIN_BLOCK_KV = 32


@triton.jit
def _fp8_to_f16(b, FP8E4: tl.constexpr):
    if FP8E4:
        return b.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float16)
    return b.to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)


@triton.jit
def _stage1_gm(
    Q, K_Idx, CB, V_Buffer, K_Scales_Zeros, V_Scales_Zeros,
    sm_scale, kv_indptr, kv_indices, Att_Out, Att_Lse, num_kv_splits,
    stride_qbs, stride_qh, stride_ki_bs, stride_ki_h,
    stride_buf_vbs, stride_buf_vh,
    stride_sz_kbs, stride_sz_kh, stride_sz_vbs, stride_sz_vh,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr, logit_cap: tl.constexpr,
    L: tl.constexpr, NG: tl.constexpr, KC: tl.constexpr,
    FP8E4: tl.constexpr, COALESCED_IDX: tl.constexpr, MODE: tl.constexpr,
):
    """Four K=NG dots instead of one K=L dot, so the [BLOCK_N, L] K tile is
    never materialised.

    MODE=0 gathers group-major ([NG, BLOCK_N]) -- fewer L1 sectors in theory,
    measurably worse in practice. MODE=1 keeps the shipped row-major gather
    (which ncu shows is already at 0.75 sectors/lane, i.e. lanes DO share
    sectors) and only drops `kg`: the [BLOCK_N, NG*4] fp16 tile costs ~64
    registers per thread, and at 243 regs the kernel is pinned to 4 CTAs/SM =
    8 warps/SM = 12% occupancy. Registers, not addressing, are what starve it.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_q0 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q1 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q2 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q3 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        offs_g = tl.arange(0, NG)
        # q sliced by byte-lane: q_m[h, g] = Q[h, 4g + m]. Hoisted out of the
        # loop, so the strided (uncoalesced) access is paid once per CTA.
        q_base = Q + cur_batch * stride_qbs + cur_head[:, None] * stride_qh
        qm = mask_h[:, None]
        q0 = tl.load(q_base + offs_g[None, :] * 4 + 0, mask=qm, other=0.0).to(tl.float16)
        q1 = tl.load(q_base + offs_g[None, :] * 4 + 1, mask=qm, other=0.0).to(tl.float16)
        q2 = tl.load(q_base + offs_g[None, :] * 4 + 2, mask=qm, other=0.0).to(tl.float16)
        q3 = tl.load(q_base + offs_g[None, :] * 4 + 3, mask=qm, other=0.0).to(tl.float16)

        cb_head_base = CB + cur_kv_head.to(tl.int64) * (NG * KC)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n, mask=mask_n, other=0
            ).to(tl.int64)

            if MODE == 1:
                # Row-major gather, exactly as shipped: [BLOCK_N, NG].
                isel = tl.load(
                    K_Idx + kv_loc[:, None] * stride_ki_bs
                    + cur_kv_head * stride_ki_h + offs_g[None, :],
                    mask=mask_n[:, None], other=0,
                ).to(tl.int32)
                cw = tl.load(
                    cb_head_base + offs_g[None, :] * KC + isel,
                    mask=mask_n[:, None], other=0,
                ).to(tl.int32)
                # Transpose the four fp16 planes rather than weaving them into
                # a [BLOCK_N, L] tile. Triton folds tl.trans of a dot operand
                # into ldmatrix.trans, so this costs nothing the shipped kernel
                # was not already paying for tl.trans(kg).
                qk = tl.dot(q0, tl.trans(_fp8_to_f16(cw & 0xFF, FP8E4)))
                qk += tl.dot(q1, tl.trans(_fp8_to_f16((cw >> 8) & 0xFF, FP8E4)))
                qk += tl.dot(q2, tl.trans(_fp8_to_f16((cw >> 16) & 0xFF, FP8E4)))
                qk += tl.dot(q3, tl.trans(_fp8_to_f16((cw >> 24) & 0xFF, FP8E4)))
            else:
                if COALESCED_IDX:
                    # Load [BLOCK_N, NG] (NG contiguous per token, fully
                    # coalesced) then relayout, keeping the index stream at
                    # 1 sector per token.
                    isel_t = tl.trans(
                        tl.load(
                            K_Idx + kv_loc[:, None] * stride_ki_bs
                            + cur_kv_head * stride_ki_h + offs_g[None, :],
                            mask=mask_n[:, None], other=0,
                        ).to(tl.int32)
                    )
                else:
                    isel_t = tl.load(
                        K_Idx + kv_loc[None, :] * stride_ki_bs
                        + cur_kv_head * stride_ki_h + offs_g[:, None],
                        mask=mask_n[None, :], other=0,
                    ).to(tl.int32)

                # Lanes run along BLOCK_N, so a warp stays inside group g's
                # single 1 KB sub-table.
                cw = tl.load(
                    cb_head_base + offs_g[:, None] * KC + isel_t,
                    mask=mask_n[None, :], other=0,
                ).to(tl.int32)

                qk = tl.dot(q0, _fp8_to_f16(cw & 0xFF, FP8E4))
                qk += tl.dot(q1, _fp8_to_f16((cw >> 8) & 0xFF, FP8E4))
                qk += tl.dot(q2, _fp8_to_f16((cw >> 16) & 0xFF, FP8E4))
                qk += tl.dot(q3, _fp8_to_f16((cw >> 24) & 0xFF, FP8E4))

            k_scale = tl.load(
                K_Scales_Zeros + kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh,
                mask=mask_n, other=1.0,
            ).to(tl.float32)
            qk = qk * k_scale[None, :] * sm_scale

            if logit_cap > 0:
                qk = logit_cap * tl.math.tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            offs_d_packed_v = tl.arange(0, BLOCK_D // 4)
            v_packed = tl.load(
                V_Buffer + kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh + offs_d_packed_v[None, :],
                mask=mask_n[:, None] & (offs_d_packed_v[None, :] < (L // 4)), other=0,
            )
            offs_sz_v_1d = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
            v_scale_1d = tl.load(
                V_Scales_Zeros + offs_sz_v_1d + 0, mask=mask_n, other=1.0
            ).to(tl.float16)
            v_zero_1d = tl.load(
                V_Scales_Zeros + offs_sz_v_1d + 1, mask=mask_n, other=0.0
            ).to(tl.float16)
            v_q0 = ((v_packed & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]
            v_q1 = (((v_packed >> 2) & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]
            v_q2 = (((v_packed >> 4) & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]
            v_q3 = (((v_packed >> 6) & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            acc_q0 *= re_scale[:, None]
            acc_q1 *= re_scale[:, None]
            acc_q2 *= re_scale[:, None]
            acc_q3 *= re_scale[:, None]
            acc_q0 += tl.dot(p.to(v_q0.dtype), v_q0)
            acc_q1 += tl.dot(p.to(v_q1.dtype), v_q1)
            acc_q2 += tl.dot(p.to(v_q2.dtype), v_q2)
            acc_q3 += tl.dot(p.to(v_q3.dtype), v_q3)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_dv = tl.arange(0, BLOCK_D // 4)
        mask_dv = offs_dv < (L // 4)
        base_mid_o = (
            cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        m = mask_h[:, None] & mask_dv[None, :]
        tl.store(Att_Out + base_mid_o + offs_dv[None, :], acc_q0 / e_sum[:, None], mask=m)
        tl.store(Att_Out + base_mid_o + (offs_dv + L // 4)[None, :], acc_q1 / e_sum[:, None], mask=m)
        tl.store(Att_Out + base_mid_o + (offs_dv + 2 * (L // 4))[None, :], acc_q2 / e_sum[:, None], mask=m)
        tl.store(Att_Out + base_mid_o + (offs_dv + 3 * (L // 4))[None, :], acc_q3 / e_sum[:, None], mask=m)

        offs_lse = (
            cur_batch * stride_mid_ob + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // L
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _stage1_ev(
    Q, K_Idx, CB, V_Buffer, K_Scales_Zeros, V_Scales_Zeros,
    sm_scale, kv_indptr, kv_indices, Att_Out, Att_Lse, num_kv_splits,
    stride_qbs, stride_qh, stride_ki_bs, stride_ki_h,
    stride_buf_vbs, stride_buf_vh,
    stride_sz_kbs, stride_sz_kh, stride_sz_vbs, stride_sz_vh,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr, logit_cap: tl.constexpr,
    L: tl.constexpr, NG: tl.constexpr, KC: tl.constexpr,
    FP8E4: tl.constexpr, EV_STREAM: tl.constexpr, EV_CB: tl.constexpr,
    CM_STREAM: tl.constexpr, CM_CB: tl.constexpr,
):
    """Shipped vq2 layout, but with explicit L1 cache-residency hints.

    The codebook is only NG*KC*4 = 32 KB per kv head and is re-read every KV
    block, so it *should* stay L1-resident. The K indices, V bytes and scales
    are pure streams -- each byte read once -- yet they flow through the same
    L1 and evict the table. EV_STREAM marks those evict_first and EV_CB marks
    the codebook evict_last, which costs nothing to try and changes no maths.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < L

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_q0 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q1 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q2 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q3 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q_main = tl.load(
            Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0
        ).to(tl.float16)
        offs_g = tl.arange(0, NG)
        cb_head_base = CB + cur_kv_head.to(tl.int64) * (NG * KC)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n, mask=mask_n, other=0,
                eviction_policy=EV_STREAM, cache_modifier=CM_STREAM,
            ).to(tl.int64)

            isel = tl.load(
                K_Idx + kv_loc[:, None] * stride_ki_bs
                + cur_kv_head * stride_ki_h + offs_g[None, :],
                mask=mask_n[:, None], other=0, eviction_policy=EV_STREAM,
                cache_modifier=CM_STREAM,
            ).to(tl.int32)
            cw = tl.load(
                cb_head_base + offs_g[None, :] * KC + isel,
                mask=mask_n[:, None], other=0, eviction_policy=EV_CB,
                cache_modifier=CM_CB,
            ).to(tl.int32)
            p0 = _fp8_to_f16(cw & 0xFF, FP8E4)
            p1 = _fp8_to_f16((cw >> 8) & 0xFF, FP8E4)
            p2 = _fp8_to_f16((cw >> 16) & 0xFF, FP8E4)
            p3 = _fp8_to_f16((cw >> 24) & 0xFF, FP8E4)
            kg = tl.reshape(
                tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BLOCK_N, NG * 4)
            )
            qk = tl.dot(q_main, tl.trans(kg))

            k_scale = tl.load(
                K_Scales_Zeros + kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh,
                mask=mask_n, other=1.0, eviction_policy=EV_STREAM,
            ).to(tl.float32)
            qk = qk * k_scale[None, :] * sm_scale
            if logit_cap > 0:
                qk = logit_cap * tl.math.tanh(qk / logit_cap)
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            offs_d_packed_v = tl.arange(0, BLOCK_D // 4)
            v_packed = tl.load(
                V_Buffer + kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh + offs_d_packed_v[None, :],
                mask=mask_n[:, None] & (offs_d_packed_v[None, :] < (L // 4)), other=0,
                eviction_policy=EV_STREAM, cache_modifier=CM_STREAM,
            )
            offs_sz_v_1d = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
            v_scale_1d = tl.load(
                V_Scales_Zeros + offs_sz_v_1d + 0, mask=mask_n, other=1.0,
                eviction_policy=EV_STREAM, cache_modifier=CM_STREAM,
            ).to(tl.float16)
            v_zero_1d = tl.load(
                V_Scales_Zeros + offs_sz_v_1d + 1, mask=mask_n, other=0.0,
                eviction_policy=EV_STREAM, cache_modifier=CM_STREAM,
            ).to(tl.float16)
            v_q0 = ((v_packed & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]
            v_q1 = (((v_packed >> 2) & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]
            v_q2 = (((v_packed >> 4) & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]
            v_q3 = (((v_packed >> 6) & 0x03).to(tl.float16) - v_zero_1d[:, None]) * v_scale_1d[:, None]

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc_q0 *= re_scale[:, None]
            acc_q1 *= re_scale[:, None]
            acc_q2 *= re_scale[:, None]
            acc_q3 *= re_scale[:, None]
            acc_q0 += tl.dot(p.to(v_q0.dtype), v_q0)
            acc_q1 += tl.dot(p.to(v_q1.dtype), v_q1)
            acc_q2 += tl.dot(p.to(v_q2.dtype), v_q2)
            acc_q3 += tl.dot(p.to(v_q3.dtype), v_q3)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_dv = tl.arange(0, BLOCK_D // 4)
        mask_dv = offs_dv < (L // 4)
        base_mid_o = (
            cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        m = mask_h[:, None] & mask_dv[None, :]
        tl.store(Att_Out + base_mid_o + offs_dv[None, :], acc_q0 / e_sum[:, None], mask=m)
        tl.store(Att_Out + base_mid_o + (offs_dv + L // 4)[None, :], acc_q1 / e_sum[:, None], mask=m)
        tl.store(Att_Out + base_mid_o + (offs_dv + 2 * (L // 4))[None, :], acc_q2 / e_sum[:, None], mask=m)
        tl.store(Att_Out + base_mid_o + (offs_dv + 3 * (L // 4))[None, :], acc_q3 / e_sum[:, None], mask=m)
        offs_lse = (
            cur_batch * stride_mid_ob + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // L
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_h)


def launch_ev(d, block_n, block_h, warps, stages, ev_stream="", ev_cb="",
              cm_stream="", cm_cb=""):
    q, out, lse = d["q"], d["out"], d["lse"]
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num // d["k_idx"].shape[1]
    grid = (batch, triton.cdiv(head_num, min(block_h, kv_group_num)), SPLITS)
    _stage1_ev[grid](
        q, d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
        SM_SCALE, d["kv_indptr"], d["kv_indices"], out, lse, d["splits"],
        q.stride(0), q.stride(1), d["k_idx"].stride(0), d["k_idx"].stride(1),
        d["v_buf"].stride(0), d["v_buf"].stride(1),
        d["k_sz"].stride(0), d["k_sz"].stride(1),
        d["v_sz"].stride(0), d["v_sz"].stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        kv_group_num=kv_group_num, q_head_num=head_num,
        BLOCK_D=triton.next_power_of_2(L), BLOCK_N=block_n, BLOCK_H=block_h,
        MIN_BLOCK_KV=_MIN_BLOCK_KV, logit_cap=0.0,
        L=L, NG=NG, KC=KC, FP8E4=False,
        EV_STREAM=ev_stream, EV_CB=ev_cb, CM_STREAM=cm_stream, CM_CB=cm_cb,
        num_warps=warps, num_stages=stages,
    )


def launch_gm(d, block_n, block_h, warps, stages, coalesced_idx=True, mode=0):
    q, out, lse = d["q"], d["out"], d["lse"]
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num // d["k_idx"].shape[1]
    grid = (batch, triton.cdiv(head_num, min(block_h, kv_group_num)), SPLITS)
    _stage1_gm[grid](
        q, d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
        SM_SCALE, d["kv_indptr"], d["kv_indices"], out, lse, d["splits"],
        q.stride(0), q.stride(1), d["k_idx"].stride(0), d["k_idx"].stride(1),
        d["v_buf"].stride(0), d["v_buf"].stride(1),
        d["k_sz"].stride(0), d["k_sz"].stride(1),
        d["v_sz"].stride(0), d["v_sz"].stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        kv_group_num=kv_group_num, q_head_num=head_num,
        BLOCK_D=triton.next_power_of_2(L), BLOCK_N=block_n, BLOCK_H=block_h,
        MIN_BLOCK_KV=_MIN_BLOCK_KV, logit_cap=0.0,
        L=L, NG=NG, KC=KC, FP8E4=False, COALESCED_IDX=coalesced_idx, MODE=mode,
        num_warps=warps, num_stages=stages,
    )


def build(dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    total = BATCH * SEQ
    cache = total + 64
    q = torch.randn(BATCH, H_Q, L, generator=g, device=dev, dtype=torch.float16)
    k_idx = torch.randint(0, KC, (cache, H_KV, NG), generator=g, device=dev, dtype=torch.uint8)
    cent = torch.randn(H_KV, NG, KC, 4, generator=g, device=dev, dtype=torch.float32) * (L ** -0.5)
    cb = cent.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
    v_buf = torch.randint(0, 256, (cache, H_KV, L // 4), generator=g, device=dev, dtype=torch.uint8)
    k_int2 = torch.randint(0, 256, (cache, H_KV, L // 4), generator=g, device=dev, dtype=torch.uint8)
    k_sz = torch.rand(cache, H_KV, 2, generator=g, device=dev) * 0.5 + 0.75
    v_sz = torch.rand(cache, H_KV, 2, generator=g, device=dev) * 0.1 + 0.05
    kv_indptr = torch.arange(0, BATCH + 1, device=dev, dtype=torch.int32) * SEQ
    kv_indices = torch.randperm(cache, generator=g, device=dev)[:total].to(torch.int32)
    splits = torch.empty(BATCH, device=dev, dtype=torch.int32)
    seq_lens = torch.full((BATCH,), SEQ, device=dev, dtype=torch.int32)
    from sglang.srt.layers.attention.triton_backend import get_num_kv_splits_triton
    _cores = torch.cuda.get_device_properties(0).multi_processor_count
    get_num_kv_splits_triton[(1,)](
        splits, seq_lens, BATCH, 1, H_Q, H_KV, SPLITS, _cores,
        MAX_NUM_SEQ=256 if BATCH < 256 else triton.next_power_of_2(BATCH))
    out = torch.zeros(BATCH, H_Q, SPLITS, L, device=dev, dtype=torch.float32)
    lse = torch.zeros(BATCH, H_Q, SPLITS, device=dev, dtype=torch.float32)
    return locals()


def bench(fn, reps=20):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def stats(K):
    dc = getattr(K, "device_caches", None) or {}
    for _, tup in dc.items():
        kc = tup[0] if isinstance(tup, (tuple, list)) else tup
        for _, ck in list(kc.items())[:1]:
            thr = ck.metadata.num_warps * 32
            return (ck.n_regs, ck.n_spills, ck.metadata.shared,
                    min(65536 // max(ck.n_regs * thr, 1),
                        (228 * 1024) // max(ck.metadata.shared, 1)))
    return (0, 0, 0, 0)


def main() -> int:
    d = build()
    from sglang.srt.layers.attention.triton_ops import decode_attention as D

    def ref_vq2():
        D._decode_grouped_att_m_fwd_quant_vq2(
            d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
            d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["splits"],
            SPLITS, SM_SCALE, 0.0)

    def ref_int2():
        D._decode_grouped_att_m_fwd_quant_int2(
            d["q"], d["k_int2"], d["v_buf"], d["k_sz"], d["v_sz"],
            d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["splits"],
            SPLITS, SM_SCALE, 0.0)

    sp = d["splits"]
    print(f"shape: bs={BATCH} ctx={SEQ} max_splits={SPLITS} -> engine splits="
          f"{int(sp.min())}..{int(sp.max())}")

    if os.environ.get("PROFILE"):
        # One launch of each kernel, nothing else, so ncu's --launch-count lines
        # up with the three kernels under comparison.
        os.environ.update(SGL_VQ2_BLOCK_N="64", SGL_VQ2_BLOCK_H="8",
                          SGL_VQ2_NUM_WARPS="2", SGL_VQ2_NUM_STAGES="3")
        ref_int2(); torch.cuda.synchronize()
        ref_vq2(); torch.cuda.synchronize()
        launch_gm(d, 64, 8, 2, 3); torch.cuda.synchronize()
        return 0

    # --- numerical validation against the shipped kernel ---
    os.environ["SGL_VQ2_BLOCK_N"] = "64"
    os.environ["SGL_VQ2_BLOCK_H"] = "8"
    os.environ["SGL_VQ2_NUM_WARPS"] = "2"
    ref_vq2(); torch.cuda.synchronize()
    o_ref, l_ref = d["out"].clone(), d["lse"].clone()
    print()
    for name, fn in (("gm  (mode 0)", lambda: launch_gm(d, 64, 8, 2, 3, True, 0)),
                     ("4dot(mode 1)", lambda: launch_gm(d, 64, 8, 2, 3, True, 1)),
                     ("ev  (hints) ", lambda: launch_ev(d, 64, 8, 2, 3,
                                                        "evict_first", "evict_last"))):
        d["out"].zero_(); d["lse"].zero_()
        fn(); torch.cuda.synchronize()
        err_o = (d["out"] - o_ref).abs().max().item() / max(o_ref.abs().max().item(), 1e-9)
        err_l = (d["lse"] - l_ref).abs().max().item() / max(l_ref.abs().max().item(), 1e-9)
        print(f"validation {name} vs shipped vq2:  out rel {err_o:.2e}  "
              f"lse rel {err_l:.2e}  {'OK' if err_o < 5e-3 else 'MISMATCH'}")

    ms_i = bench(ref_int2)
    r, s, sh, c = stats(D._fwd_grouped_kernel_stage1_quant_int2)
    print(f"\nint2 (target):        {ms_i*1e3:8.1f} us  regs={r} sp={s} smem={sh} CTA/SM={c}")
    ms_v = bench(ref_vq2)
    r, s, sh, c = stats(D._fwd_grouped_kernel_stage1_quant_vq2)
    print(f"vq2 shipped (64/8/2): {ms_v*1e3:8.1f} us  regs={r} sp={s} smem={sh} CTA/SM={c}"
          f"  ({ms_v/ms_i:.2f}x int2)")

    only = os.environ.get("ONLY", "")

    if only in ("", "gm"):
        print(f"\n=== four-dot sweep (mode 1 = row-major gather, no kg tile) ===")
        print(f"{'variant':>12} {'BN':>4} {'BH':>3} {'W':>2} {'S':>2} {'us':>9} "
              f"{'vs int2':>8} {'vs vq2':>7} {'regs':>5} {'sp':>4} {'CTA':>4}")
        rows = []
        for mode, coal, bn, bh, nw, ns in itertools.product(
                (1, 0), (True, False), (32, 64, 128, 256), (4, 8, 16),
                (1, 2, 4, 8), (2, 3)):
            if mode == 1 and not coal:
                continue                      # COALESCED_IDX is unread in mode 1
            try:
                ms = bench(lambda: launch_gm(d, bn, bh, nw, ns, coal, mode))
            except Exception:
                continue
            r, s, _, c = stats(_stage1_gm)
            rows.append((ms, mode, coal, bn, bh, nw, ns, r, s, c))
        rows.sort()
        for ms, mode, coal, bn, bh, nw, ns, r, s, c in rows[:10]:
            tag = "4dot" if mode == 1 else ("gm-coal" if coal else "gm")
            print(f"{tag:>12} {bn:>4} {bh:>3} {nw:>2} {ns:>2} {ms*1e3:>9.1f} "
                  f"{ms/ms_i:>7.2f}x {ms/ms_v:>6.2f}x {r:>5} {s:>4} {c:>4}")
        if rows:
            ms, mode, coal, bn, bh, nw, ns = rows[0][:7]
            print(f"\nbest: mode={mode} BLOCK_N={bn} BLOCK_H={bh} warps={nw} "
                  f"stages={ns}  {ms*1e3:.1f} us = {ms/ms_i:.2f}x int2, "
                  f"{ms/ms_v:.2f}x shipped vq2")

    if only in ("", "ev"):
        # L1 residency hints. "" means "no hint" -- Triton's default.
        # ".cg" bypasses L1 for the streams, leaving the whole 32 KB codebook
        # resident; ".cs" is the weaker evict-first form. ncu says L1 hit rate
        # is only ~38%, so the table is being trampled by K/V traffic.
        print(f"\n=== cache-policy sweep (shipped layout + L1 residency hints) ===")
        print(f"{'stream cm':>10} {'cb cm':>7} {'stream ev':>12} {'cb ev':>11} "
              f"{'BN':>4} {'W':>2} {'S':>2} {'us':>9} {'vs int2':>8} {'vs vq2':>7}")
        rows = []
        combos = [(cm, "", ev, "") for cm in ("", ".cg", ".cs") for ev in ("",)]
        combos += [(cm, "", "evict_first", "evict_last") for cm in ("", ".cg", ".cs")]
        combos += [(".cg", ".ca", "", ""), (".cs", ".ca", "evict_first", "evict_last")]
        for cms, cmc, evs, evc in combos:
            for bn, nw, ns in itertools.product((64, 128), (2, 4), (3,)):
                try:
                    ms = bench(lambda: launch_ev(d, bn, 8, nw, ns, evs, evc, cms, cmc))
                except Exception as e:
                    print(f"  {cms}/{cmc}/{evs}/{evc} {bn}/{nw}: {type(e).__name__}: "
                          f"{str(e).splitlines()[0][:80]}")
                    continue
                rows.append((ms, cms, cmc, evs, evc, bn, nw, ns))
        rows.sort()
        for ms, cms, cmc, evs, evc, bn, nw, ns in rows[:12]:
            print(f"{cms or '-':>10} {cmc or '-':>7} {evs or '-':>12} {evc or '-':>11} "
                  f"{bn:>4} {nw:>2} {ns:>2} {ms*1e3:>9.1f} {ms/ms_i:>7.2f}x {ms/ms_v:>6.2f}x")
        if rows:
            ms, cms, cmc, evs, evc, bn, nw, ns = rows[0]
            print(f"\nbest cache-policy: stream_cm={cms or 'default'} cb_cm={cmc or 'default'} "
                  f"stream_ev={evs or 'default'} cb_ev={evc or 'default'} BLOCK_N={bn} "
                  f"warps={nw}\n  {ms*1e3:.1f} us = {ms/ms_i:.2f}x int2, "
                  f"{ms/ms_v:.2f}x shipped vq2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
