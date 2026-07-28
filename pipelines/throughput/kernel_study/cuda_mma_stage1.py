"""CUDA mma.sync vq2 stage-1: shared-codebook gather fused into MMA fragments.

Why. At fp32 accuracy every existing route converges above OSCAR-INT2: scalar
CUDA 1827 us, retuned Triton 1803, TileLang 2995 (OSCAR: 1636-1650). ncu shows
the scalar CUDA kernel is instruction-bound (1.071 G executed, issue 60.6%)
against Triton's 893 M -- a scalar warp FMA retires 32 MACs, an HMMA retires
2048. The one design not yet built anywhere fuses the two winning halves:

  * the SHARED-memory codebook gather (kills the 392 M L1 sectors; Triton
    cannot express it, TileLang cannot pipeline around it), and
  * m16n8k16 mma.sync with f16 operands and f32 accumulators (kills the
    instruction deficit; f16 products are EXACT in f32, so this is the same
    numerics class as Triton's tl.dot -- no accuracy loss).

The fusion is the CUDA-only part: for the QK dot each lane decodes, from the
shared codebook, exactly the 8 halves its A-fragment position owns -- computed
scores never touch a materialized k_hat tile, no ldmatrix, no shared
round-trip. PTX m16n8k16 fragment layout (groupID = lane>>2, tig = lane&3):

  A (row-major [16,16]): a0={A[g][2t],A[g][2t+1]} a1={A[g+8][...]}
                          a2={A[g][2t+8],A[g][2t+9]} a3={A[g+8][...]}
  B (col-major [16,8]):   b0={B[2t][g],B[2t+1][g]} b1={B[2t+8][g],B[2t+9][g]}
  C (f32 [16,8]):         c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]

QK computes S^T = k_hat . q^T ([BLOCK_N,8]; tokens on M so the 4-head padding
is 2x, not the 4x that killed every earlier MMA attempt), PV computes
acc^T = V^T . p ([128 dims, 8 heads]) with V int2-dequantized straight into
A-fragments the same way. Online softmax runs on the C fragments with
butterfly shuffles + tiny shared scans, mirroring the Triton kernel's
semantics exactly (fp32 scores and l; p rounded to f16 for the PV operand).

Validated against the shipped Triton kernel's golden outputs
(logs/stage1_ref_{small,big}.pt); invalid splits are NaN by design in both.

  docker exec -e CUDA_VISIBLE_DEVICES=7 oscar-ab bash -lc 'cd <repo> && \
    mkdir -p /tmp/vq2mma && REF=logs/stage1_ref_small.pt \
    /opt/venv-oscar/bin/python pipelines/throughput/kernel_study/cuda_mma_stage1.py'
"""
import os

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#define NG 32
#define KC 256
#define L 128
#define KVG 4
#define NPAD 8
#define THR 128
#define NWARP 4
#define BLOCK_N 128            // tokens per tile: NWARP warps x CHUNKS x 16
#define CHUNKS 2               // 16-token chunks per warp per tile
#define MIN_BLOCK_KV 32
// padded shared strides: token rows at 32 B land every row on the same banks
// (32 B == bank cycle), measured 54% conflict rate. 48 B keeps uint4 staging
// aligned and spreads rows: bank = (t*12 + w) %% 32.
#define IDXS 12               // idx_s row stride in ints   (8 used)
#define VPKS 48               // vpk_s row stride in bytes  (32 used)
#define PSTR (BLOCK_N + 8)    // p_s row stride in halves
#define VTS (BLOCK_N + 4)     // vpkT row stride in bytes (transposed V)

// two e5m2 bytes of a packed codeword -> half2 (each byte<<8 is its fp16)
__device__ __forceinline__ unsigned int unpack2u(unsigned int cw, unsigned int sel) {
    return __byte_perm(cw, 0u, sel);
}
#define SEL_LO 0x1404u
#define SEL_HI 0x3424u
#define SZF 4                 // per-token float slots: v_scale, v_zero, k_scale

__device__ __forceinline__ void cpa16(void* dst, const void* src) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void cpa8(void* dst, const void* src) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void cpa4(void* dst, const void* src) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void cpa_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cpa_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ void mma16816(
    float& c0, float& c1, float& c2, float& c3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <typename IT>
__global__ __launch_bounds__(THR) void vq2_mma_stage1(
    const __half* __restrict__ Q,          // [B, H_Q, L]
    const uint8_t* __restrict__ KIdx,      // [cache, H_KV, NG]
    const int* __restrict__ CB,            // [H_KV, NG, KC]
    const uint8_t* __restrict__ VB,        // [cache, H_KV, L/4]
    const float* __restrict__ KSZ,         // [cache, H_KV, 2]
    const float* __restrict__ VSZ,         // [cache, H_KV, 2]
    const int* __restrict__ kv_indptr,
    const IT* __restrict__ kv_indices,
    const int* __restrict__ num_kv_splits,
    float* __restrict__ Out,               // [B, H_Q, S, L]
    float* __restrict__ Lse,               // [B, H_Q, S]
    int H_KV, int H_Q, int SPLITS, float sm_scale,
    long s_ki_bs, long s_ki_h, long s_vb_bs, long s_vb_h,
    long s_sz_kbs, long s_sz_kh, long s_sz_vbs, long s_sz_vh)
{
    const int bb   = blockIdx.x;
    const int kvh  = blockIdx.y;
    const int bs   = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g8   = lane >> 2, t4 = lane & 3;

    extern __shared__ __align__(16) char smem_raw[];
    int*     cb_s  = reinterpret_cast<int*>(smem_raw);                    // 32 KB
    // double-buffered tile arenas so cp.async fills tile i+1 while tile i
    // computes; szf packs (v_scale, v_zero, k_scale, pad) per token
    int*     idx_s = cb_s + NG * KC;                                      // [2][BLOCK_N][IDXS]
    uint8_t* vpk_s = reinterpret_cast<uint8_t*>(idx_s + 2 * BLOCK_N * IDXS);
    float*   szf_s = reinterpret_cast<float*>(vpk_s + 2 * BLOCK_N * VPKS); // [2][BLOCK_N][SZF]
    __half*  p_s   = reinterpret_cast<__half*>(szf_s + 2 * BLOCK_N * SZF);
    float*   wred  = reinterpret_cast<float*>(p_s + NPAD * PSTR);         // [NWARP][NPAD]
    float*   m_s   = wred + NWARP * NPAD;                                 // [NPAD]
    float*   l_s   = m_s + NPAD;
    float*   rs_s  = l_s + NPAD;

    const int kv_start = kv_indptr[bb];
    const int seq_len  = kv_indptr[bb + 1] - kv_start;
    const int ksp      = num_kv_splits[bb];
    const int per      = ((seq_len + ksp - 1) / ksp + MIN_BLOCK_KV - 1)
                         / MIN_BLOCK_KV * MIN_BLOCK_KV;
    const int s_start  = per * bs;
    const int s_end    = min(s_start + per, seq_len);

    // codebook for this head, staged once
    const int* cb_head = CB + (long)kvh * (NG * KC);
    for (int i = tid; i < NG * KC; i += THR) cb_s[i] = cb_head[i];
    if (tid < NPAD) { m_s[tid] = -3.0e38f; l_s[tid] = 0.f; }

    // q as B-fragments (S^T = k_hat . q^T), constant for the whole kernel:
    // per kstep kk: qb0 = {q[g8][16kk+2t4], q[g8][16kk+2t4+1]}, qb1 = +8.
    // heads >= KVG are zero.
    unsigned int qb0[8], qb1[8];
    {
        const __half* qh = Q + ((long)bb * H_Q + kvh * KVG + g8) * L;
        const bool real = g8 < KVG;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            if (real) {
                const int c = kk * 16 + 2 * t4;
                qb0[kk] = *reinterpret_cast<const unsigned int*>(qh + c);
                qb1[kk] = *reinterpret_cast<const unsigned int*>(qh + c + 8);
            } else { qb0[kk] = 0u; qb1[kk] = 0u; }
        }
    }

    // PV accumulators: acc^T [16 dims x 8 heads] per chunk; this warp owns
    // dims [warp*32 + ch*16, +16). Rows g8/g8+8 = dims, cols 2t4/2t4+1 = heads.
    float acc[CHUNKS][4], accB[CHUNKS][4];
#pragma unroll
    for (int ch = 0; ch < CHUNKS; ++ch)
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[ch][i] = 0.f; accB[ch][i] = 0.f; }
    __syncthreads();

    // stage(tile) issues this thread's token via cp.async into buffer `buf`
    auto issue_stage = [&](int base2, int buf) {
        const int n = base2 + tid;
        if (tid < BLOCK_N && n < s_end) {
            const long loc = (long)kv_indices[kv_start + n];
            const uint8_t* ip = KIdx + loc * s_ki_bs + (long)kvh * s_ki_h;
            int* idst = idx_s + (buf * BLOCK_N + tid) * IDXS;
            cpa16(idst, ip);
            cpa16(idst + 4, ip + 16);
            const uint8_t* vp = VB + loc * s_vb_bs + (long)kvh * s_vb_h;
            uint8_t* vdst = vpk_s + (buf * BLOCK_N + tid) * VPKS;
            cpa16(vdst, vp);
            cpa16(vdst + 16, vp + 16);
            float* fdst = szf_s + (buf * BLOCK_N + tid) * SZF;
            cpa8(fdst, VSZ + loc * s_sz_vbs + kvh * s_sz_vh);
            cpa4(fdst + 2, KSZ + loc * s_sz_kbs + kvh * s_sz_kh);
        }
        cpa_commit();
    };
    int cur = 0;
    if (s_start < s_end) issue_stage(s_start, 0);

    for (int base = s_start; base < s_end; base += BLOCK_N) {
        cpa_wait_all();
        // zero-fill masked tail tokens (skipped by issue_stage)
        if (tid < BLOCK_N && base + tid >= s_end) {
            int* idst = idx_s + (cur * BLOCK_N + tid) * IDXS;
            uint4* iv = reinterpret_cast<uint4*>(idst);
            iv[0] = make_uint4(0, 0, 0, 0); iv[1] = make_uint4(0, 0, 0, 0);
            uint4* vv = reinterpret_cast<uint4*>(vpk_s + (cur * BLOCK_N + tid) * VPKS);
            vv[0] = make_uint4(0, 0, 0, 0); vv[1] = make_uint4(0, 0, 0, 0);
            float* fdst = szf_s + (cur * BLOCK_N + tid) * SZF;
            fdst[0] = 0.f; fdst[1] = 0.f; fdst[2] = 0.f;
        }
        __syncthreads();
        if (base + BLOCK_N < s_end) issue_stage(base + BLOCK_N, cur ^ 1);
        const int* idxb = idx_s + cur * BLOCK_N * IDXS;
        const uint8_t* vpkb = vpk_s + cur * BLOCK_N * VPKS;
        const float* szfb = szf_s + cur * BLOCK_N * SZF;
        // In-place V repack: byte g+8k of a row moves to word g, byte k. A PV
        // lane then fetches ONE word per token covering both chunks and both
        // dims, and the bank pattern (t*12 + g8) % 32 is conflict-free.
        // Safe without a new barrier: each thread repacks only its own row,
        // and the first cross-thread read is two barriers later.
        if (tid < BLOCK_N) {
            uint4* row = reinterpret_cast<uint4*>(
                const_cast<uint8_t*>(vpkb) + tid * VPKS);
            const uint4 r0 = row[0], r1 = row[1];
            const unsigned int w[8] = {r0.x, r0.y, r0.z, r0.w,
                                       r1.x, r1.y, r1.z, r1.w};
            uint4 n0, n1;
            unsigned int* nw0 = reinterpret_cast<unsigned int*>(&n0);
            unsigned int* nw1 = reinterpret_cast<unsigned int*>(&n1);
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                // new word g = bytes {g, g+8, g+16, g+24} = byte g%4 of words
                // {g/4, g/4+2, g/4+4, g/4+6}
                const unsigned int lo = __byte_perm(w[0], w[2], 0x0040u + g * 0x11u);
                const unsigned int hi = __byte_perm(w[4], w[6], 0x0040u + g * 0x11u);
                nw0[g] = __byte_perm(lo, hi, 0x5410u);
                const unsigned int lo2 = __byte_perm(w[1], w[3], 0x0040u + g * 0x11u);
                const unsigned int hi2 = __byte_perm(w[5], w[7], 0x0040u + g * 0x11u);
                nw1[g] = __byte_perm(lo2, hi2, 0x5410u);
            }
            row[0] = n0; row[1] = n1;
        }

        // ---- QK: decode-into-fragment gather + 16 MMAs per warp ----
        float sc[CHUNKS][4];
        float colmax[2] = {-3.0e38f, -3.0e38f};   // this lane's 2 head-columns
#pragma unroll
        for (int ch = 0; ch < CHUNKS; ++ch) {
            const int trow = warp * (CHUNKS * 16) + ch * 16;   // tile-local
            const int T0 = trow + g8, T1 = T0 + 8;
            // this lane's two tokens' index words, kept in registers for all
            // 8 ksteps (the byte for group g is word g/4, byte g%4)
            const uint4 w0a = *reinterpret_cast<const uint4*>(idxb + T0 * IDXS);
            const uint4 w0b = *reinterpret_cast<const uint4*>(idxb + T0 * IDXS + 4);
            const uint4 w1a = *reinterpret_cast<const uint4*>(idxb + T1 * IDXS);
            const uint4 w1b = *reinterpret_cast<const uint4*>(idxb + T1 * IDXS + 4);
            const unsigned int iw0[8] = {w0a.x, w0a.y, w0a.z, w0a.w,
                                         w0b.x, w0b.y, w0b.z, w0b.w};
            const unsigned int iw1[8] = {w1a.x, w1a.y, w1a.z, w1a.w,
                                         w1b.x, w1b.y, w1b.z, w1b.w};
            // two independent accumulate chains (even/odd kstep): the 8 MMAs
            // otherwise serialize on the same C registers at MMA latency
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            const unsigned int sel = (t4 & 1) ? SEL_HI : SEL_LO;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                // coords (16kk + 2t4, +1) -> group gA = 4kk + (t4>>1),
                // bytes (0,1) or (2,3); coords +8 -> group gA + 2.
                const int gA = 4 * kk + (t4 >> 1), gB = gA + 2;
                const unsigned int iA0 = (iw0[gA >> 2] >> ((gA & 3) * 8)) & 255u;
                const unsigned int iB0 = (iw0[gB >> 2] >> ((gB & 3) * 8)) & 255u;
                const unsigned int iA1 = (iw1[gA >> 2] >> ((gA & 3) * 8)) & 255u;
                const unsigned int iB1 = (iw1[gB >> 2] >> ((gB & 3) * 8)) & 255u;
                const unsigned int a0 = unpack2u(cb_s[gA * KC + iA0], sel);
                const unsigned int a2 = unpack2u(cb_s[gB * KC + iB0], sel);
                const unsigned int a1 = unpack2u(cb_s[gA * KC + iA1], sel);
                const unsigned int a3 = unpack2u(cb_s[gB * KC + iB1], sel);
                if (kk & 1) mma16816(d0, d1, d2, d3, a0, a1, a2, a3, qb0[kk], qb1[kk]);
                else        mma16816(c0, c1, c2, c3, a0, a1, a2, a3, qb0[kk], qb1[kk]);
            }
            c0 += d0; c1 += d1; c2 += d2; c3 += d3;
            // scores: fold per-token ptn scale + sm_scale; mask past s_end
            const int n0 = base + T0, n1 = base + T1;
            const float k0 = szfb[T0 * SZF + 2] * sm_scale, k1 = szfb[T1 * SZF + 2] * sm_scale;
            c0 = n0 < s_end ? c0 * k0 : -3.0e38f;
            c1 = n0 < s_end ? c1 * k0 : -3.0e38f;
            c2 = n1 < s_end ? c2 * k1 : -3.0e38f;
            c3 = n1 < s_end ? c3 * k1 : -3.0e38f;
            sc[ch][0] = c0; sc[ch][1] = c1; sc[ch][2] = c2; sc[ch][3] = c3;
            colmax[0] = fmaxf(colmax[0], fmaxf(c0, c2));
            colmax[1] = fmaxf(colmax[1], fmaxf(c1, c3));
        }
        // per-warp column(head) max: butterfly across g8 (strides 4, 8, 16)
#pragma unroll
        for (int o = 4; o <= 16; o <<= 1) {
            colmax[0] = fmaxf(colmax[0], __shfl_xor_sync(0xffffffffu, colmax[0], o));
            colmax[1] = fmaxf(colmax[1], __shfl_xor_sync(0xffffffffu, colmax[1], o));
        }
        if (g8 == 0) {
            wred[warp * NPAD + 2 * t4]     = colmax[0];
            wred[warp * NPAD + 2 * t4 + 1] = colmax[1];
        }
        __syncthreads();
        if (tid < NPAD) {
            float m = m_s[tid];
            float t = wred[tid];
#pragma unroll
            for (int w = 1; w < NWARP; ++w) t = fmaxf(t, wred[w * NPAD + tid]);
            const float mn = fmaxf(m, t);
            rs_s[tid] = __expf(m - mn);
            m_s[tid] = mn;
        }
        __syncthreads();

        // ---- p = exp(s - m): f16 to p_s for the PV B-operand, fp32 sums ----
        const float m0 = m_s[2 * t4], m1 = m_s[2 * t4 + 1];
        float colsum[2] = {0.f, 0.f};
#pragma unroll
        for (int ch = 0; ch < CHUNKS; ++ch) {
            const int trow = warp * (CHUNKS * 16) + ch * 16;
            const int T0 = trow + g8, T1 = T0 + 8;
            const float p00 = __expf(sc[ch][0] - m0);
            const float p01 = __expf(sc[ch][1] - m1);
            const float p10 = __expf(sc[ch][2] - m0);
            const float p11 = __expf(sc[ch][3] - m1);
            colsum[0] += p00 + p10;
            colsum[1] += p01 + p11;
            p_s[(2 * t4) * PSTR + T0]     = __float2half(p00);
            p_s[(2 * t4 + 1) * PSTR + T0] = __float2half(p01);
            p_s[(2 * t4) * PSTR + T1]     = __float2half(p10);
            p_s[(2 * t4 + 1) * PSTR + T1] = __float2half(p11);
        }
#pragma unroll
        for (int o = 4; o <= 16; o <<= 1) {
            colsum[0] += __shfl_xor_sync(0xffffffffu, colsum[0], o);
            colsum[1] += __shfl_xor_sync(0xffffffffu, colsum[1], o);
        }
        if (g8 == 0) {
            wred[warp * NPAD + 2 * t4]     = colsum[0];
            wred[warp * NPAD + 2 * t4 + 1] = colsum[1];
        }
        __syncthreads();
        if (tid < NPAD) {
            float s = wred[tid];
#pragma unroll
            for (int w = 1; w < NWARP; ++w) s += wred[w * NPAD + tid];
            l_s[tid] = l_s[tid] * rs_s[tid] + s;
        }
        __syncthreads();

        // ---- PV: acc^T[dim, head] += V^T . p, V decoded into A-fragments ----
        const float r0 = rs_s[2 * t4], r1 = rs_s[2 * t4 + 1];
#pragma unroll
        for (int ch = 0; ch < CHUNKS; ++ch) {
            acc[ch][0] *= r0;  acc[ch][1] *= r1;
            acc[ch][2] *= r0;  acc[ch][3] *= r1;
            accB[ch][0] *= r0; accB[ch][1] *= r1;
            accB[ch][2] *= r0; accB[ch][3] *= r1;
        }
        // dim D = warp*(CHUNKS*16) + ch*16 + (g8 | +8). In the repacked word
        // (bytes {g8, g8+8, g8+16, g8+24}) D lives at byte k = (D&31)>>3 and
        // its 2-bit field sits at 2*(D>>5). g8<8 never crosses either boundary,
        // so both are per-(warp, ch) constants.
        const int dbase = warp * (CHUNKS * 16);
#pragma unroll
        for (int tk = 0; tk < 8; ++tk) {              // K-steps over tokens
            const int t0 = tk * 16 + 2 * t4;          // adjacent token pair
            const int t2 = t0 + 8;
            // token loads hoisted across chunks: one repacked word per token
            const unsigned int vw0 = *reinterpret_cast<const unsigned int*>(
                vpkb + t0 * VPKS + g8 * 4);
            const unsigned int vw1 = *reinterpret_cast<const unsigned int*>(
                vpkb + (t0 + 1) * VPKS + g8 * 4);
            const unsigned int vw2 = *reinterpret_cast<const unsigned int*>(
                vpkb + t2 * VPKS + g8 * 4);
            const unsigned int vw3 = *reinterpret_cast<const unsigned int*>(
                vpkb + (t2 + 1) * VPKS + g8 * 4);
            const float4 f0a = *reinterpret_cast<const float4*>(szfb + t0 * SZF);
            const float4 f0b = *reinterpret_cast<const float4*>(szfb + (t0 + 1) * SZF);
            const float4 f2a = *reinterpret_cast<const float4*>(szfb + t2 * SZF);
            const float4 f2b = *reinterpret_cast<const float4*>(szfb + (t2 + 1) * SZF);
            const __half2 sc0 = __floats2half2_rn(f0a.x, f0b.x);
            const __half2 zr0 = __floats2half2_rn(f0a.y, f0b.y);
            const __half2 sc2 = __floats2half2_rn(f2a.x, f2b.x);
            const __half2 zr2 = __floats2half2_rn(f2a.y, f2b.y);
            const unsigned int pb0 =
                *reinterpret_cast<const unsigned int*>(p_s + g8 * PSTR + t0);
            const unsigned int pb1 =
                *reinterpret_cast<const unsigned int*>(p_s + g8 * PSTR + t2);
#pragma unroll
            for (int ch = 0; ch < CHUNKS; ++ch) {
                const int k0  = (((dbase + ch * 16) & 31) >> 3) * 8;
                const int qsh = ((dbase + ch * 16) >> 5) * 2;
                const unsigned int y00 = ((vw0 >> k0) >> qsh) & 3u;
                const unsigned int y01 = ((vw1 >> k0) >> qsh) & 3u;
                const unsigned int y20 = ((vw2 >> k0) >> qsh) & 3u;
                const unsigned int y21 = ((vw3 >> k0) >> qsh) & 3u;
                const unsigned int z00 = ((vw0 >> (k0 + 8)) >> qsh) & 3u;
                const unsigned int z01 = ((vw1 >> (k0 + 8)) >> qsh) & 3u;
                const unsigned int z20 = ((vw2 >> (k0 + 8)) >> qsh) & 3u;
                const unsigned int z21 = ((vw3 >> (k0 + 8)) >> qsh) & 3u;
                union { unsigned int u; __half2 h; } a0, a1, a2, a3;
                a0.h = __hmul2(__hsub2(__halves2half2(__ushort2half_rn(y00),
                                                      __ushort2half_rn(y01)), zr0), sc0);
                a2.h = __hmul2(__hsub2(__halves2half2(__ushort2half_rn(y20),
                                                      __ushort2half_rn(y21)), zr2), sc2);
                a1.h = __hmul2(__hsub2(__halves2half2(__ushort2half_rn(z00),
                                                      __ushort2half_rn(z01)), zr0), sc0);
                a3.h = __hmul2(__hsub2(__halves2half2(__ushort2half_rn(z20),
                                                      __ushort2half_rn(z21)), zr2), sc2);
                if (tk & 1)
                    mma16816(accB[ch][0], accB[ch][1], accB[ch][2], accB[ch][3],
                             a0.u, a1.u, a2.u, a3.u, pb0, pb1);
                else
                    mma16816(acc[ch][0], acc[ch][1], acc[ch][2], acc[ch][3],
                             a0.u, a1.u, a2.u, a3.u, pb0, pb1);
            }
        }
        __syncthreads();
        cur ^= 1;
    }

    // ---- epilogue: Out[b, h, split, dim] = acc^T[dim, h] / l[h] ----
    const float l0 = l_s[2 * t4], l1 = l_s[2 * t4 + 1];
    const long ob = ((long)bb * H_Q + kvh * KVG) * SPLITS + bs;
#pragma unroll
    for (int ch = 0; ch < CHUNKS; ++ch) {
        const int D0 = warp * (CHUNKS * 16) + ch * 16 + g8, D1 = D0 + 8;
        if (2 * t4 < KVG) {
            Out[(ob + (long)(2 * t4) * SPLITS) * L + D0] = (acc[ch][0] + accB[ch][0]) / l0;
            Out[(ob + (long)(2 * t4) * SPLITS) * L + D1] = (acc[ch][2] + accB[ch][2]) / l0;
        }
        if (2 * t4 + 1 < KVG) {
            Out[(ob + (long)(2 * t4 + 1) * SPLITS) * L + D0] = (acc[ch][1] + accB[ch][1]) / l1;
            Out[(ob + (long)(2 * t4 + 1) * SPLITS) * L + D1] = (acc[ch][3] + accB[ch][3]) / l1;
        }
    }
    if (tid < KVG)
        Lse[((long)bb * H_Q + kvh * KVG + tid) * SPLITS + bs] =
            m_s[tid] + logf(l_s[tid]);
}

void vq2_mma(torch::Tensor q, torch::Tensor k_idx, torch::Tensor cb,
             torch::Tensor v_buf, torch::Tensor k_sz, torch::Tensor v_sz,
             torch::Tensor kv_indptr, torch::Tensor kv_indices,
             torch::Tensor splits, torch::Tensor out, torch::Tensor lse,
             int64_t max_splits, double sm_scale)
{
    const int B = q.size(0), H_Q = q.size(1), H_KV = k_idx.size(1);
    dim3 grid(B, H_KV, (int)max_splits);
    const size_t sm = NG * KC * 4 + 2 * BLOCK_N * 12 * 4 + 2 * BLOCK_N * 48
                    + 2 * BLOCK_N * 4 * 4
                    + NPAD * (BLOCK_N + 8) * 2 + (NWARP * NPAD + 3 * NPAD) * 4 + 64;
#define GOI(IT) do { \
        static bool init = false; \
        if (!init) { \
            cudaFuncSetAttribute(vq2_mma_stage1<IT>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); \
            init = true; \
        } \
        vq2_mma_stage1<IT><<<grid, THR, sm, at::cuda::getCurrentCUDAStream()>>>( \
            reinterpret_cast<const __half*>(q.data_ptr()), \
            k_idx.data_ptr<uint8_t>(), cb.data_ptr<int>(), \
            v_buf.data_ptr<uint8_t>(), k_sz.data_ptr<float>(), \
            v_sz.data_ptr<float>(), kv_indptr.data_ptr<int>(), \
            kv_indices.data_ptr<IT>(), splits.data_ptr<int>(), \
            out.data_ptr<float>(), lse.data_ptr<float>(), \
            H_KV, H_Q, (int)max_splits, (float)sm_scale, \
            k_idx.stride(0), k_idx.stride(1), v_buf.stride(0), v_buf.stride(1), \
            k_sz.stride(0), k_sz.stride(1), v_sz.stride(0), v_sz.stride(1)); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    } while (0)
    if (kv_indices.scalar_type() == torch::kInt) GOI(int);
    else GOI(long);
#undef GOI
}
"""

CPP_SRC = """
void vq2_mma(torch::Tensor q, torch::Tensor k_idx, torch::Tensor cb,
             torch::Tensor v_buf, torch::Tensor k_sz, torch::Tensor v_sz,
             torch::Tensor kv_indptr, torch::Tensor kv_indices,
             torch::Tensor splits, torch::Tensor out, torch::Tensor lse,
             int64_t max_splits, double sm_scale);
"""


def build():
    os.makedirs("/tmp/vq2mma", exist_ok=True)
    return load_inline(
        name="vq2mma", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
        functions=["vq2_mma"], build_directory="/tmp/vq2mma", verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90a"])


def bench(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3


def main() -> int:
    ext = build()
    ref = torch.load(os.environ.get("REF", "logs/stage1_ref_small.pt"),
                     weights_only=False)
    dev = "cuda"
    B, SP = ref["BATCH"], ref["SPLITS"]
    q = ref["q"].to(dev)
    k_idx = ref["k_idx"].to(dev)
    cb = ref["cb"].to(dev)
    v_buf = ref["v_buf"].to(dev)
    k_sz = ref["k_sz"].to(dev).float().contiguous()
    v_sz = ref["v_sz"].to(dev).float().contiguous()
    kv_indptr = ref["kv_indptr"].to(dev)
    kv_indices = ref["kv_indices"].to(dev)
    splits = ref["splits"].to(dev)
    out = torch.zeros_like(ref["out"]).to(dev)
    lse = torch.zeros_like(ref["lse"]).to(dev)
    v = ref["splits_used"]
    print(f"{torch.cuda.get_device_name(0)}  bs={B} ctx={ref['SEQ']} "
          f"max_splits={SP} valid={v}")

    def run():
        ext.vq2_mma(q, k_idx, cb, v_buf, k_sz, v_sz, kv_indptr, kv_indices,
                    splits, out, lse, SP, ref["SM_SCALE"])

    run()
    torch.cuda.synchronize()
    ro, rl = ref["out"].to(dev), ref["lse"].to(dev)
    go, gl = out[:, :, :v, :], ro[:, :, :v, :]
    assert torch.isfinite(go).all(), "NaN/Inf in valid Out region"
    d = (go - gl).abs()
    rel = (d.max() / gl.abs().max().clamp_min(1e-9)).item()
    dl = (lse[:, :, :v] - rl[:, :, :v]).abs().max().item()
    print(f"Out vs Triton golden: max abs {d.max().item():.3e}  rel {rel:.3e}")
    print(f"Lse vs Triton golden: max abs {dl:.3e}")
    ok = rel < 2e-3 and dl < 1e-3
    print("PARITY " + ("PASS" if ok else "FAIL"))
    if not ok:
        return 1
    us = bench(run)
    print(f"\nCUDA mma.sync stage-1 (fp32 acc): {us:8.1f} us")
    if (B, ref["SEQ"]) == (64, 30000):
        print("same-box references, full stage-1, fp32, bs=64/ctx=30k:")
        print("    OSCAR int2 (Triton)      1636-1650 us")
        print("    vq2 Triton retuned       1803 us")
        print("    vq2 CUDA scalar fp32     1827 us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
