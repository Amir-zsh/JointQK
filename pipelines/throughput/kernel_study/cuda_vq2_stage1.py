"""Full vq2 decode stage-1 in CUDA, with the codebook in shared memory.

Why this exists
---------------
ncu (see profile_ncu.py) shows the shipped Triton kernel is bound by L1 SECTOR
THROUGHPUT, not by address divergence as the first round of this study assumed:

    metric                int2      vq2
    global ld sectors     94.1 M   392.4 M
    sectors / request     17.77     19.43     <- nearly identical
    L1 throughput          30.4%     73.8%    <- vq2 is at the wall

The codebook gather is ~94% of vq2's L1 traffic. 32 lanes fetching 4 B each out
of a 32 KB table cost ~24 sectors per warp instruction however the table is
addressed, so group-major layout, cache modifiers, eviction policy and launch
config all failed to move it (each measured; see README).

Shared memory has no sectors. Staging the 32 KB table there makes a gather cost
only bank conflicts. probe_cuda_qk.py measures the isolated gather+score at
684 us against Triton's 1134 us -- 1.66x -- so this kernel carries that through
the whole of stage-1.

What it is
----------
Numerically equivalent to `_fwd_grouped_kernel_stage1_quant_vq2`
(decode_attention.py:2538) for the served configuration: e5m2 codewords, INT2 V,
no logit_cap, no xai temperature, GQA.

RESULT: it closes the vq2-vs-int2 decode-kernel gap. Across four shapes it runs
at 0.94-1.05x int2 and 0.75-0.83x the shipped Triton kernel, at rel out 3.2e-04.
Tuned config: PV=3, half2, TPT=1, THR=128, GUNR=32, PUNR=4.

What mattered, in order: the shared codebook; PV register tiling (PV=3); FULLY
unrolling the 32-group score loop (GUNR=32, worth 1843 -> 1630 us alone, because
the score accumulators are a serial dependency chain); loading the q half2 pair
as ONE 8 B access; and unrolling the PV loop (PUNR=4).

This is a study artifact, not an engine change -- nothing in vendor/OSCAR-vq is
touched. GUNR/PUNR are env-tunable; the other knobs are the `pv`/`h2`/`tpt`/`thr`
arguments.

The half2 score path accumulates in fp16 (2.3e-03 relative on qk, but only
3.2e-04 on the stage-1 output since softmax damps it); the fp32 path is kept for
comparison and is ~13% slower.

  docker exec -e CUDA_VISIBLE_DEVICES=0 oscar-ab bash -lc 'cd <repo> && \
    PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
    /opt/venv-oscar/bin/python pipelines/throughput/kernel_study/cuda_vq2_stage1.py'
"""
import os

import torch
import triton
from torch.utils.cpp_extension import load_inline

BATCH = int(os.environ.get("BS", 64))
SEQ = int(os.environ.get("CTX", 30000))
SPLITS = int(os.environ.get("SPLITS", 8))
H_Q, H_KV, L, NG, KC = 32, 8, 128, 32, 256
KVG = H_Q // H_KV
SM_SCALE = 1.0 / (L ** 0.5)
_MIN_BLOCK_KV = 32

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <math.h>

#define NG 32
#define KC 256
#define L  128
#define KVG 4
// V rows are 8 words (32 B), unpadded. Phase C has all 32 lanes read ONE row
// (byte `lane` of row `nl`), so they hit 8 distinct words with 4 lanes sharing
// each -- different bytes of the same word, which broadcasts rather than
// conflicts. Padding would only matter if lanes read different rows, and it
// breaks the 16 B alignment the uint4 stores need.
#define ROWW 8

__device__ __forceinline__ __half2 unpack2(unsigned int cw, unsigned int sel) {
    // e5m2 -> fp16 is a <<8 per byte; prmt does two at once by building
    // [0x00, b_lo, 0x00, b_hi]. One instruction replaces four.
    union { unsigned int u; __half2 h; } c;
    c.u = __byte_perm(cw, 0u, sel);
    return c.h;
}
#define SEL_LO 0x1404u
#define SEL_HI 0x3424u

// BLOCK_N = THR * TPT tokens per KV block. TPT>1 amortises the q broadcast
// loads across tokens; the whole block must fit in shared for the PV pass.
template <int TPT, bool H2, int PV, int THR>
__global__ __launch_bounds__(THR) void vq2_stage1_kernel(
    const __half*  __restrict__ Q,          // [B, H_Q, L]
    const uint8_t* __restrict__ K_Idx,      // [cache, H_KV, NG]
    const int*     __restrict__ CB,         // [H_KV, NG, KC]
    const uint8_t* __restrict__ V_Buf,      // [cache, H_KV, L/4] packed INT2
    const float*   __restrict__ K_SZ,       // [cache, H_KV, 2] slot0 = ptn scale
    const float*   __restrict__ V_SZ,       // [cache, H_KV, 2] scale, zero
    const int*     __restrict__ kv_indptr,
    const int*     __restrict__ kv_indices,
    const int*     __restrict__ num_kv_splits,
    float*         __restrict__ Att_Out,    // [B, H_Q, SPLITS, L]
    float*         __restrict__ Att_Lse,    // [B, H_Q, SPLITS]
    int SPLITS_, float sm_scale,
    long stride_ki_bs, long stride_vb_bs, long stride_ksz_bs, long stride_vsz_bs)
{
    constexpr int BLOCK_N = THR * TPT;
    constexpr int NW = THR / 32;
    const int bb = blockIdx.x, kvh = blockIdx.y, bs = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // Dynamic shared: the 32 KB table alone puts any BLOCK_N >= 256 past the
    // 48 KB static cap, so the carve-up is manual and the opt-in attribute is
    // set on the host side. Order is alignment-driven: float4 first, then the
    // table (a 16 B multiple), then the uint4-written V bytes.
    extern __shared__ __align__(16) char smem_raw[];
    float4*  q4_s  = reinterpret_cast<float4*>(smem_raw);
    int*     cb_s  = reinterpret_cast<int*>(q4_s + KVG * NG);
    unsigned int* vpk32_s = reinterpret_cast<unsigned int*>(cb_s + NG * KC);
    uint8_t* vpk_s = reinterpret_cast<uint8_t*>(vpk32_s);
    __half2* q2_s  = reinterpret_cast<__half2*>(vpk32_s + BLOCK_N * ROWW);
    // p stored head-MAJOR-per-token as a float4: the PV pass needs all four
    // heads for token n, so one 16 B broadcast LDS replaces four scalar ones.
    // That is 2 shared instructions/token/warp instead of 5 (measured 225.6 M
    // shared loads against Triton's 40.5 M -- the dominant PV cost).
    float4*  qk_s  = reinterpret_cast<float4*>(q2_s + KVG * NG * 2);
    float*   vsc_s = reinterpret_cast<float*>(qk_s + BLOCK_N);
    float*   vze_s = vsc_s + BLOCK_N;
    // Only live in the epilogue, so it reuses the V stage. 512 floats:
    // one per (head, dim), summed across warps with atomicAdd.
    float*   acc_s = reinterpret_cast<float*>(vpk32_s);
    __shared__ float   redm[KVG][8], reds[KVG][8];
    __shared__ float   m_s[KVG], l_s[KVG];

    const int kv_start = kv_indptr[bb];
    const int seq_len  = kv_indptr[bb + 1] - kv_start;
    const int splits   = num_kv_splits[bb];
    const int per = ((seq_len + splits - 1) / splits + _MINBK - 1) / _MINBK * _MINBK;
    const int s_start = per * bs;
    const int s_end   = min(s_start + per, seq_len);
    if (s_end <= s_start) return;   // Triton leaves these cells untouched too

    const int* cb_head = CB + (long)kvh * (NG * KC);
    for (int i = tid; i < NG * KC; i += THR) cb_s[i] = cb_head[i];
    for (int i = tid; i < KVG * NG; i += THR) {
        const int h = i / NG, g = i % NG;
        const __half* qp = Q + ((long)bb * (gridDim.y * KVG) + kvh * KVG + h) * L + g * 4;
        q2_s[2 * i + 0] = __halves2half2(qp[0], qp[1]);
        q2_s[2 * i + 1] = __halves2half2(qp[2], qp[3]);
        q4_s[i] = make_float4(__half2float(qp[0]), __half2float(qp[1]),
                              __half2float(qp[2]), __half2float(qp[3]));
    }
    if (tid < KVG) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }

    // PV assignment: warp == head, and each thread owns FOUR CONSECUTIVE output
    // dims d = 4*lane + j. Then
    //   - qk_s[head][n] is one address for the whole warp  -> broadcast,
    //   - the four dims live in bytes 4*(lane&7)..+3 of row n, i.e. ONE int32,
    //     and lanes 0,8,16,24 share that word -> broadcast, not a conflict.
    // Both reads are a single wavefront, so PV costs 8 wavefronts/token instead
    // of the 32 that "thread owns dim tid, all heads" cost (measured 2333 us).
    // Two PV assignments, both covering the 4*128 = 512 (head, dim) pairs with
    // 128 threads, trading shared traffic against dequant ALU:
    //   PV=0  thread owns dim tid for ALL 4 heads. One dequantised V value feeds
    //         4 FMAs, so only 128 dequants/token -- the minimum. Costs a 4-way
    //         bank conflict on the V byte: 32 wavefronts/token.
    //   PV=1  warp owns a head, thread owns 4 consecutive dims. Both shared
    //         reads become broadcasts (8 wavefronts/token) but each of the 4
    //         FMAs needs its own dequant: 512/token, 4x the ALU.
    // Measured: PV=0 2333 us, PV=1 3126 us. The dequant ALU dominates.
    const int my_h = warp, b8 = lane & 7, qtr = lane >> 3, my_d = tid;
    float acco[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acco[j] = 0.f;

    // PV=3: the fix. Tokens are partitioned ACROSS WARPS (warp-strided), and a
    // lane owns the four quarters of ONE packed V byte -- dims {lane, lane+32,
    // lane+64, lane+96}, which are exactly that byte's four 2-bit fields. So one
    // byte + one float4 of p feed 16 FMAs, and each token is touched by a single
    // warp: 2 shared loads/token instead of the 8 that PV=0 pays by having all
    // 128 threads walk every token (2 loads for only 4 FMAs).
    // The (scale, zero) affine is folded out of the inner loop entirely:
    //   sum_n p*((q2-z)*s) = sum_n (p*s)*q2  -  sum_n (p*s)*z
    // so phase B stores pp = p*vsc and accumulates corr[h] = sum pp*vze, leaving
    // phase C as a pure int2-unpack + FMA.
    float acc16[KVG][4], corrl[KVG];
#pragma unroll
    for (int h = 0; h < KVG; ++h) {
        corrl[h] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc16[h][j] = 0.f;
    }
    __syncthreads();

    // ---- KV block loop ----
    // Loads are issued per thread and consumed in the same iteration. Two ways
    // of raising memory-level parallelism were tried and BOTH measured worse:
    //   - bulk-staging the tile through shared: 2609 vs 2032 us (extra shared
    //     write+read and two more barriers cost more than the latency hidden);
    //   - software pipelining block bi+1 into registers: 1966 vs 1930 us (the
    //     doubled register state costs more occupancy than it buys, and with a
    //     runtime slot index nvcc demotes the buffers to local memory -- 43%
    //     slower until the buffers were named and rotated by register copy).
    // Each lane already holds its own token, so a warp has 32 tokens in flight.
    for (int base = s_start; base < s_end; base += BLOCK_N) {
        const int nmax = min(BLOCK_N, s_end - base);

        // ---- phase A: gather + score, and stage V for this block ----
        float qkv[KVG][TPT], vsc_r[TPT], vze_r[TPT];
#pragma unroll
        for (int t = 0; t < TPT; ++t) {
            const int nl = t * THR + tid;
            const int n  = base + nl;
            const bool ok = n < s_end;
            const long loc = ok ? kv_indices[kv_start + n] : 0;

            const uint8_t* idxp = K_Idx + loc * stride_ki_bs + (long)kvh * NG;
            const uint4 i0 = *reinterpret_cast<const uint4*>(idxp);
            const uint4 i1 = *reinterpret_cast<const uint4*>(idxp + 16);
            unsigned int iw[8];
            iw[0] = i0.x; iw[1] = i0.y; iw[2] = i0.z; iw[3] = i0.w;
            iw[4] = i1.x; iw[5] = i1.y; iw[6] = i1.z; iw[7] = i1.w;

            __half2 a2[KVG];
            float    af[KVG];
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                a2[h] = __floats2half2_rn(0.f, 0.f);
                af[h] = 0.f;
            }
#pragma unroll 4
            for (int g = 0; g < NG; ++g) {
                const int idx = (iw[g >> 2] >> ((g & 3) * 8)) & 0xFF;
                const unsigned int cw = cb_s[g * KC + idx];
                const __half2 k01 = unpack2(cw, SEL_LO);
                const __half2 k23 = unpack2(cw, SEL_HI);
                if (H2) {
#pragma unroll
                    for (int h = 0; h < KVG; ++h) {
                        // The two half2s are adjacent, so ONE 8 B load gets
                        // both. Loading them as two __half2 costs 256 shared
                        // loads/token against the fp32 path's 128 float4 loads,
                        // which is why half2 won in the isolated probe (FMA
                        // bound) but LOST in the full kernel (shared bound).
                        if (PV == 9 && h > 0) continue;
                        const uint2 qq = *reinterpret_cast<const uint2*>(
                            &q2_s[(h * NG + g) * 2]);
                        union { unsigned int u; __half2 h2; } c0, c1;
                        c0.u = qq.x; c1.u = qq.y;
                        a2[h] = __hfma2(c0.h2, k01, a2[h]);
                        a2[h] = __hfma2(c1.h2, k23, a2[h]);
                    }
                } else {
                    const float k0 = __low2float(k01), k1 = __high2float(k01);
                    const float k2 = __low2float(k23), k3 = __high2float(k23);
#pragma unroll
                    for (int h = 0; h < KVG; ++h) {
                        const float4 qv = q4_s[h * NG + g];
                        af[h] = fmaf(qv.x, k0, af[h]);
                        af[h] = fmaf(qv.y, k1, af[h]);
                        af[h] = fmaf(qv.z, k2, af[h]);
                        af[h] = fmaf(qv.w, k3, af[h]);
                    }
                }
            }
            const float ksc = ok ? K_SZ[loc * stride_ksz_bs + kvh * 2 + 0] : 0.f;
            // (scale, zero) are adjacent, so one 8 B load replaces two 4 B ones.
            const float2 vsz = ok
                ? *reinterpret_cast<const float2*>(V_SZ + loc * stride_vsz_bs + kvh * 2)
                : make_float2(0.f, 0.f);
            vsc_r[t] = vsz.x;
            vze_r[t] = vsz.y;
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                const float v = H2 ? (__low2float(a2[h]) + __high2float(a2[h])) : af[h];
                qkv[h][t] = ok ? (v * ksc * sm_scale) : -INFINITY;
            }

            if (PV != 7) {
                const uint8_t* vp = V_Buf + loc * stride_vb_bs + (long)kvh * (L / 4);
                *reinterpret_cast<uint4*>(vpk_s + nl * (ROWW * 4))      =
                    *reinterpret_cast<const uint4*>(vp);
                *reinterpret_cast<uint4*>(vpk_s + nl * (ROWW * 4) + 16) =
                    *reinterpret_cast<const uint4*>(vp + 16);
            }
            if (PV < 3) { vsc_s[nl] = vsc_r[t]; vze_s[nl] = vze_r[t]; }
        }

        if (PV == 8) {
#pragma unroll
            for (int t = 0; t < TPT; ++t)
#pragma unroll
                for (int h = 0; h < KVG; ++h) acc16[h][0] += qkv[h][t];
            continue;   // skip softmax + PV entirely
        }
        // ---- block max per head ----
        float mloc[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            mloc[h] = -INFINITY;
#pragma unroll
            for (int t = 0; t < TPT; ++t) mloc[h] = fmaxf(mloc[h], qkv[h][t]);
            for (int o = 16; o; o >>= 1)
                mloc[h] = fmaxf(mloc[h], __shfl_down_sync(0xffffffff, mloc[h], o));
            if (lane == 0) redm[h][warp] = mloc[h];
        }
        __syncthreads();

        float mnew[KVG], rsc[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            float b = redm[h][0];
#pragma unroll
            for (int w = 1; w < NW; ++w) b = fmaxf(b, redm[h][w]);
            mnew[h] = fmaxf(m_s[h], b);
            rsc[h]  = (m_s[h] == -INFINITY) ? 0.f : __expf(m_s[h] - mnew[h]);
        }
        // Rescale the running accumulators BEFORE this block contributes to
        // them. corr is accumulated in phase B (below), so rescaling it after
        // phase B would scale this block's own contribution as well -- which is
        // the bug that made PV=3 fast and wrong (rel 4.2e-01).
        if (PV >= 3) {
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                corrl[h] *= rsc[h];      // corr is linear in p, so it rescales alike
#pragma unroll
                for (int j = 0; j < 4; ++j) acc16[h][j] *= rsc[h];
            }
        }

        // ---- exp, running sum, and publish p into shared for the PV pass ----
        float sloc[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) sloc[h] = 0.f;
#pragma unroll
        for (int t = 0; t < TPT; ++t) {
            const int nl = t * THR + tid;
            float pv4[KVG];
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                pv4[h] = (qkv[h][t] == -INFINITY) ? 0.f : __expf(qkv[h][t] - mnew[h]);
                sloc[h] += pv4[h];       // l uses the plain exp, not p*vsc
            }
            if (PV >= 3) {
                // vsc/vze were loaded by THIS thread for THIS token in phase A,
                // so they are still in registers -- no shared round-trip.
                const float vs = vsc_r[t], vz = vze_r[t];
#pragma unroll
                for (int h = 0; h < KVG; ++h) {
                    pv4[h] *= vs;
                    corrl[h] += pv4[h] * vz;
                }
            }
            qk_s[nl] = make_float4(pv4[0], pv4[1], pv4[2], pv4[3]);
        }
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            for (int o = 16; o; o >>= 1) sloc[h] += __shfl_down_sync(0xffffffff, sloc[h], o);
            if (lane == 0) reds[h][warp] = sloc[h];
        }
        __syncthreads();

#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            float sm = reds[h][0];
#pragma unroll
            for (int w = 1; w < NW; ++w) sm += reds[h][w];
            if (tid == 0) { m_s[h] = mnew[h]; l_s[h] = l_s[h] * rsc[h] + sm; }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) acco[j] *= (PV != 1 ? rsc[j] : rsc[my_h]);

        // ---- phase C: PV ----
        if (PV >= 6) {
            // ablation: no PV at all
        } else if (PV == 3 || PV == 4) {
            // Same ILP problem the score loop had: acc16 is a serial dependency
            // chain and the two shared loads feed it directly, so without
            // unrolling only one load pair is ever in flight. Fully unrolling
            // the score loop (GUNR 4 -> 32) was worth 1843 -> 1630 us.
#pragma unroll 4
            for (int nl = warp; nl < nmax; nl += NW) {
                const unsigned int byte = vpk_s[nl * (ROWW * 4) + lane];
                const float4 pp = qk_s[nl];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float q2 = (float)((byte >> (2 * j)) & 0x3);
                    // PV=4 is an ABLATION: same loads, same unpack, 4 FMAs
                    // instead of 16. Bounds what a tensor-core PV could save
                    // from the CURRENT structure (the earlier PV=2 bound was
                    // measured against the much slower PV=0 one).
                    acc16[0][j] = fmaf(pp.x, q2, acc16[0][j]);
                    if (PV != 4) {
                        acc16[1][j] = fmaf(pp.y, q2, acc16[1][j]);
                        acc16[2][j] = fmaf(pp.z, q2, acc16[2][j]);
                        acc16[3][j] = fmaf(pp.w, q2, acc16[3][j]);
                    }
                }
            }
        } else if (PV != 1) {
#pragma unroll 4
            for (int nl = 0; nl < nmax; ++nl) {
                const int q2b = (vpk_s[nl * (ROWW * 4) + lane] >> (warp * 2)) & 0x3;
                const float vv = ((float)q2b - vze_s[nl]) * vsc_s[nl];
                const float4 p = qk_s[nl];          // one LDS for all four heads
                // PV=2 is an ABLATION, not a variant: identical loads and
                // dequant, but 1 FMA instead of 4. The delta bounds how much a
                // tensor-core PV could ever save, since mma removes exactly the
                // FMAs and nothing else.
                acco[0] = fmaf(p.x, vv, acco[0]);
                if (PV != 2) {
                    acco[1] = fmaf(p.y, vv, acco[1]);
                    acco[2] = fmaf(p.z, vv, acco[2]);
                    acco[3] = fmaf(p.w, vv, acco[3]);
                }
            }
        } else {
#pragma unroll 4
            for (int nl = 0; nl < nmax; ++nl) {
                const unsigned int w =
                    *reinterpret_cast<const unsigned int*>(vpk_s + nl * (ROWW * 4) + b8 * 4);
                const float4 p4 = qk_s[nl];
                const float p = (my_h == 0) ? p4.x : (my_h == 1) ? p4.y
                              : (my_h == 2) ? p4.z : p4.w;
                const float vsc = vsc_s[nl], vze = vze_s[nl];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int q2b = (w >> (8 * j + 2 * qtr)) & 0x3;
                    acco[j] = fmaf(p, ((float)q2b - vze) * vsc, acco[j]);
                }
            }
        }
        __syncthreads();   // shared is reused by the next KV block
    }

    // ---- epilogue ----
    if (PV >= 3) {
        // Each warp holds a partial over its own token subset; sum them.
        // A 2 KB shared-atomicAdd version of this was tried (to shrink the
        // NW*8 KB buffer and raise occupancy) and measured 30% SLOWER, twice,
        // against stable baselines -- occupancy went UP and the kernel got
        // worse, so plain per-warp slots stay.
#pragma unroll
        for (int h = 0; h < KVG; ++h)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                acc_s[((warp * KVG + h) * 4 + j) * 32 + lane] = acc16[h][j];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            float c = corrl[h];
            for (int o = 16; o; o >>= 1) c += __shfl_down_sync(0xffffffff, c, o);
            if (lane == 0) reds[h][warp] = c;
        }
        __syncthreads();
        // Exactly KVG*32 threads produce the KVG*128 outputs (4 dims each).
        if (tid >= KVG * 32) return;
        const int oh = tid >> 5, ol = tid & 31;
        float corr = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) corr += reds[oh][w];
        const long qh = (long)kvh * KVG + oh;
        const long ob = ((long)bb * (gridDim.y * KVG) + qh) * SPLITS_ + bs;
        const float inv = 1.f / l_s[oh];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float t = 0.f;
#pragma unroll
            for (int w = 0; w < NW; ++w)
                t += acc_s[((w * KVG + oh) * 4 + j) * 32 + ol];
            Att_Out[ob * L + j * 32 + ol] = (t - corr) * inv;
        }
        if (ol == 0) Att_Lse[ob] = m_s[oh] + logf(l_s[oh]);
    } else if (PV != 1) {
        for (int h = 0; h < KVG; ++h) {
            const long qh = (long)kvh * KVG + h;
            const long ob = ((long)bb * (gridDim.y * KVG) + qh) * SPLITS_ + bs;
            Att_Out[ob * L + my_d] = acco[h] / l_s[h];
            if (tid == 0) Att_Lse[ob] = m_s[h] + logf(l_s[h]);
        }
    } else {
        const long qh = (long)kvh * KVG + my_h;
        const long ob = ((long)bb * (gridDim.y * KVG) + qh) * SPLITS_ + bs;
        const float inv = 1.f / l_s[my_h];
#pragma unroll
        for (int j = 0; j < 4; ++j) Att_Out[ob * L + 4 * lane + j] = acco[j] * inv;
        if (lane == 0) Att_Lse[ob] = m_s[my_h] + logf(l_s[my_h]);
    }
}


// ---------------------------------------------------------------------------
// q-reuse variant (pv=5).
//
// ncu on the pv=3 kernel: 113 M shared loads = 7.4 per token, of which
//   32   codebook gathers   (irreducible: NG=32 lookups per token)
//   128  q broadcasts       (4 heads x 32 groups)  <-- 80% of the traffic
//    2   PV
// and L1 throughput sits at 75%, the same wall Triton hits. The q broadcasts
// are pure reuse: every token in the CTA wants the same q.
//
// TPT was supposed to amortise them, but in the pv=3 kernel BLOCK_N = THR*TPT,
// so raising TPT grows vpk_s/qk_s and costs occupancy -- which is why TPT>1
// measured worse there. Those two roles are independent:
//
//   * the Q-REUSE group is TPT tokens per thread, held in registers;
//   * the SOFTMAX/STAGING block stays THR tokens, so shared is constant in TPT.
//
// Phase A scores TPT*THR tokens with one q load per (head, group) reused TPT
// times; then TPT sub-passes each run an online softmax + PV over THR tokens
// through the same small buffers. Shared loads/token: 32 + 128/TPT + 2.
template <int TPT, int THR>
__global__ __launch_bounds__(THR) void vq2_qreuse_kernel(
    const __half*  __restrict__ Q,
    const uint8_t* __restrict__ K_Idx,
    const int*     __restrict__ CB,
    const uint8_t* __restrict__ V_Buf,
    const float*   __restrict__ K_SZ,
    const float*   __restrict__ V_SZ,
    const int*     __restrict__ kv_indptr,
    const int*     __restrict__ kv_indices,
    const int*     __restrict__ num_kv_splits,
    float*         __restrict__ Att_Out,
    float*         __restrict__ Att_Lse,
    int SPLITS_, float sm_scale,
    long stride_ki_bs, long stride_vb_bs, long stride_ksz_bs, long stride_vsz_bs)
{
    constexpr int NW = THR / 32;
    const int bb = blockIdx.x, kvh = blockIdx.y, bs = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    extern __shared__ __align__(16) char smem_raw[];
    float4*  q4_s  = reinterpret_cast<float4*>(smem_raw);
    int*     cb_s  = reinterpret_cast<int*>(q4_s + KVG * NG);
    uint8_t* vpk_s = reinterpret_cast<uint8_t*>(cb_s + NG * KC);
    float4*  qk_s  = reinterpret_cast<float4*>(vpk_s + THR * (L / 4));
    float*   acc_s = reinterpret_cast<float*>(vpk_s);   // epilogue only
    __shared__ float redm[KVG][8], reds[KVG][8], m_s[KVG], l_s[KVG];

    const int kv_start = kv_indptr[bb];
    const int seq_len  = kv_indptr[bb + 1] - kv_start;
    const int splits   = num_kv_splits[bb];
    const int per = ((seq_len + splits - 1) / splits + _MINBK - 1) / _MINBK * _MINBK;
    const int s_start = per * bs;
    const int s_end   = min(s_start + per, seq_len);
    if (s_end <= s_start) return;

    const int* cb_head = CB + (long)kvh * (NG * KC);
    for (int i = tid; i < NG * KC; i += THR) cb_s[i] = cb_head[i];
    for (int i = tid; i < KVG * NG; i += THR) {
        const int h = i / NG, g = i % NG;
        const __half* qp = Q + ((long)bb * (gridDim.y * KVG) + kvh * KVG + h) * L + g * 4;
        q4_s[i] = make_float4(__half2float(qp[0]), __half2float(qp[1]),
                              __half2float(qp[2]), __half2float(qp[3]));
    }
    if (tid < KVG) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
    float acc16[KVG][4], corrl[KVG];
#pragma unroll
    for (int h = 0; h < KVG; ++h) {
        corrl[h] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc16[h][j] = 0.f;
    }
    __syncthreads();

    for (int outer = s_start; outer < s_end; outer += THR * TPT) {
        // ---- phase A: score TPT tokens/thread, q loaded once per (h, g) ----
        unsigned int iw[TPT][8];
        float ksc[TPT];
        long  loc_r[TPT];
        bool  ok[TPT];
#pragma unroll
        for (int t = 0; t < TPT; ++t) {
            const int n = outer + t * THR + tid;
            ok[t] = n < s_end;
            loc_r[t] = ok[t] ? kv_indices[kv_start + n] : 0;
            const uint8_t* ip = K_Idx + loc_r[t] * stride_ki_bs + (long)kvh * NG;
            const uint4 a = *reinterpret_cast<const uint4*>(ip);
            const uint4 b = *reinterpret_cast<const uint4*>(ip + 16);
            iw[t][0] = a.x; iw[t][1] = a.y; iw[t][2] = a.z; iw[t][3] = a.w;
            iw[t][4] = b.x; iw[t][5] = b.y; iw[t][6] = b.z; iw[t][7] = b.w;
            ksc[t] = ok[t] ? K_SZ[loc_r[t] * stride_ksz_bs + kvh * 2 + 0] : 0.f;
        }

        float qkv[KVG][TPT];
#pragma unroll
        for (int h = 0; h < KVG; ++h)
#pragma unroll
            for (int t = 0; t < TPT; ++t) qkv[h][t] = 0.f;

#pragma unroll 2
        for (int g = 0; g < NG; ++g) {
            float k0[TPT], k1[TPT], k2[TPT], k3[TPT];
#pragma unroll
            for (int t = 0; t < TPT; ++t) {
                const int idx = (iw[t][g >> 2] >> ((g & 3) * 8)) & 0xFF;
                const unsigned int cw = cb_s[g * KC + idx];
                const __half2 a = unpack2(cw, SEL_LO);
                const __half2 b = unpack2(cw, SEL_HI);
                k0[t] = __low2float(a); k1[t] = __high2float(a);
                k2[t] = __low2float(b); k3[t] = __high2float(b);
            }
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                const float4 qv = q4_s[h * NG + g];   // ONE load, TPT tokens
#pragma unroll
                for (int t = 0; t < TPT; ++t) {
                    qkv[h][t] = fmaf(qv.x, k0[t], qkv[h][t]);
                    qkv[h][t] = fmaf(qv.y, k1[t], qkv[h][t]);
                    qkv[h][t] = fmaf(qv.z, k2[t], qkv[h][t]);
                    qkv[h][t] = fmaf(qv.w, k3[t], qkv[h][t]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TPT; ++t)
#pragma unroll
            for (int h = 0; h < KVG; ++h)
                qkv[h][t] = ok[t] ? qkv[h][t] * ksc[t] * sm_scale : -INFINITY;

        // ---- TPT sub-passes: online softmax + PV over THR tokens each ----
#pragma unroll 1
        for (int t = 0; t < TPT; ++t) {
            const int nsub = min(THR, s_end - (outer + t * THR));
            if (nsub <= 0) break;
            __syncthreads();
            const uint8_t* vp = V_Buf + loc_r[t] * stride_vb_bs + (long)kvh * (L / 4);
            *reinterpret_cast<uint4*>(vpk_s + tid * (L / 4))      =
                *reinterpret_cast<const uint4*>(vp);
            *reinterpret_cast<uint4*>(vpk_s + tid * (L / 4) + 16) =
                *reinterpret_cast<const uint4*>(vp + 16);
            const float vsc = ok[t] ? V_SZ[loc_r[t] * stride_vsz_bs + kvh * 2 + 0] : 0.f;
            const float vze = ok[t] ? V_SZ[loc_r[t] * stride_vsz_bs + kvh * 2 + 1] : 0.f;

            float mloc[KVG];
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                mloc[h] = qkv[h][t];
                for (int o = 16; o; o >>= 1)
                    mloc[h] = fmaxf(mloc[h], __shfl_down_sync(0xffffffff, mloc[h], o));
                if (lane == 0) redm[h][warp] = mloc[h];
            }
            __syncthreads();

            float mnew[KVG], rsc[KVG];
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                float bmax = redm[h][0];
#pragma unroll
                for (int w = 1; w < NW; ++w) bmax = fmaxf(bmax, redm[h][w]);
                mnew[h] = fmaxf(m_s[h], bmax);
                rsc[h] = (m_s[h] == -INFINITY) ? 0.f : __expf(m_s[h] - mnew[h]);
                corrl[h] *= rsc[h];
#pragma unroll
                for (int j = 0; j < 4; ++j) acc16[h][j] *= rsc[h];
            }

            float pv4[KVG], sloc[KVG];
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                const float e = (qkv[h][t] == -INFINITY) ? 0.f
                                                        : __expf(qkv[h][t] - mnew[h]);
                sloc[h] = e;
                pv4[h] = e * vsc;
                corrl[h] += pv4[h] * vze;
            }
            qk_s[tid] = make_float4(pv4[0], pv4[1], pv4[2], pv4[3]);
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                for (int o = 16; o; o >>= 1) sloc[h] += __shfl_down_sync(0xffffffff, sloc[h], o);
                if (lane == 0) reds[h][warp] = sloc[h];
            }
            __syncthreads();
#pragma unroll
            for (int h = 0; h < KVG; ++h) {
                float sm = reds[h][0];
#pragma unroll
                for (int w = 1; w < NW; ++w) sm += reds[h][w];
                if (tid == 0) { m_s[h] = mnew[h]; l_s[h] = l_s[h] * rsc[h] + sm; }
            }

            for (int nl = warp; nl < nsub; nl += NW) {
                const unsigned int byte = vpk_s[nl * (L / 4) + lane];
                const float4 pp = qk_s[nl];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float q2 = (float)((byte >> (2 * j)) & 0x3);
                    acc16[0][j] = fmaf(pp.x, q2, acc16[0][j]);
                    acc16[1][j] = fmaf(pp.y, q2, acc16[1][j]);
                    acc16[2][j] = fmaf(pp.z, q2, acc16[2][j]);
                    acc16[3][j] = fmaf(pp.w, q2, acc16[3][j]);
                }
            }
        }
    }

    __syncthreads();
#pragma unroll
    for (int h = 0; h < KVG; ++h)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc_s[((warp * KVG + h) * 4 + j) * 32 + lane] = acc16[h][j];
#pragma unroll
    for (int h = 0; h < KVG; ++h) {
        float c = corrl[h];
        for (int o = 16; o; o >>= 1) c += __shfl_down_sync(0xffffffff, c, o);
        if (lane == 0) reds[h][warp] = c;
    }
    __syncthreads();
    if (tid >= KVG * 32) return;
    const int oh = tid >> 5, ol = tid & 31;
    float corr = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) corr += reds[oh][w];
    const long qh = (long)kvh * KVG + oh;
    const long ob = ((long)bb * (gridDim.y * KVG) + qh) * SPLITS_ + bs;
    const float inv = 1.f / l_s[oh];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float acc = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) acc += acc_s[((w * KVG + oh) * 4 + j) * 32 + ol];
        Att_Out[ob * L + j * 32 + ol] = (acc - corr) * inv;
    }
    if (ol == 0) Att_Lse[ob] = m_s[oh] + logf(l_s[oh]);
}


// ---------------------------------------------------------------------------
// LUT variant (pv=10): fold q INTO the table.
//
// ncu on the pv=3 kernel: L1 throughput 76.2% -- the wall -- from 111 M global
// sectors plus ~119 M shared wavefronts. The shared side is 32 codebook
// gathers + 128 q broadcasts per token.
//
// But q is fixed for the whole CTA, so the score
//     qk[h] = sum_g q[h, 4g..4g+3] . cb[g, idx_g]
// can be rewritten by precomputing, once per CTA,
//     LUT[g][c][h] = q[h, 4g..4g+3] . cb[g, c]
// after which qk[h] = sum_g LUT[g][idx_g][h].
//
// Laying the 4 heads out CONTIGUOUSLY makes one 8 B load per group return all
// four heads' partial scores, so the token loop costs 32 shared loads instead
// of 160, and 64 half2 adds instead of 256 hfma2. Both codebook gathers and q
// broadcasts disappear.
//
// Cost: the table is NG*KC*KVG halves = 64 KB (vs the codebook's 32 KB), so
// occupancy drops to ~3 CTAs/SM. Building it is NG*KC*KVG = 32768 dot-4s per
// CTA, negligible against ~10k tokens x 512 MACs.
template <int THR>
__global__ __launch_bounds__(THR) void vq2_lut_kernel(
    const __half*  __restrict__ Q,
    const uint8_t* __restrict__ K_Idx,
    const int*     __restrict__ CB,
    const uint8_t* __restrict__ V_Buf,
    const float*   __restrict__ K_SZ,
    const float*   __restrict__ V_SZ,
    const int*     __restrict__ kv_indptr,
    const int*     __restrict__ kv_indices,
    const int*     __restrict__ num_kv_splits,
    float*         __restrict__ Att_Out,
    float*         __restrict__ Att_Lse,
    int SPLITS_, float sm_scale,
    long stride_ki_bs, long stride_vb_bs, long stride_ksz_bs, long stride_vsz_bs)
{
    constexpr int NW = THR / 32;
    constexpr int BLOCK_N = THR;
    const int bb = blockIdx.x, kvh = blockIdx.y, bs = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    extern __shared__ __align__(16) char smem_raw[];
    __half*  lut_s = reinterpret_cast<__half*>(smem_raw);          // NG*KC*KVG
    uint8_t* vpk_s = reinterpret_cast<uint8_t*>(lut_s + NG * KC * KVG);
    // vpk_s and acc_s share a region; qk_s starts past whichever is larger.
    // acc_s holds NW per-warp partials of KVG*4*32 floats, i.e. NW*8192/4 B --
    // sizing this region from the V tile alone overruns it into qk_s.
    constexpr int AREG = NW * KVG * 4 * 32 * 4;
    constexpr int VREG = BLOCK_N * (L / 4) > AREG ? BLOCK_N * (L / 4) : AREG;
    float4*  qk_s  = reinterpret_cast<float4*>(vpk_s + VREG);
    float*   acc_s = reinterpret_cast<float*>(vpk_s);              // epilogue only
    __shared__ float redm[KVG][8], reds[KVG][8], m_s[KVG], l_s[KVG];

    const int kv_start = kv_indptr[bb];
    const int seq_len  = kv_indptr[bb + 1] - kv_start;
    const int splits   = num_kv_splits[bb];
    const int per = ((seq_len + splits - 1) / splits + _MINBK - 1) / _MINBK * _MINBK;
    const int s_start = per * bs;
    const int s_end   = min(s_start + per, seq_len);
    if (s_end <= s_start) return;

    // ---- build the LUT: g outer so q stays in registers ----
    const __half* qbase = Q + ((long)bb * (gridDim.y * KVG) + kvh * KVG) * L;
    const int* cb_head = CB + (long)kvh * (NG * KC);
    for (int g = 0; g < NG; ++g) {
        float q0[KVG], q1[KVG], q2v[KVG], q3[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            const __half* qp = qbase + (long)h * L + g * 4;
            q0[h] = __half2float(qp[0]); q1[h] = __half2float(qp[1]);
            q2v[h] = __half2float(qp[2]); q3[h] = __half2float(qp[3]);
        }
        for (int c = tid; c < KC; c += THR) {
            const unsigned int cw = cb_head[g * KC + c];
            const __half2 a = unpack2(cw, SEL_LO);
            const __half2 b = unpack2(cw, SEL_HI);
            const float k0 = __low2float(a), k1 = __high2float(a);
            const float k2 = __low2float(b), k3 = __high2float(b);
#pragma unroll
            for (int h = 0; h < KVG; ++h)
                lut_s[(g * KC + c) * KVG + h] =
                    __float2half(q0[h] * k0 + q1[h] * k1 + q2v[h] * k2 + q3[h] * k3);
        }
    }
    if (tid < KVG) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
    float acc16[KVG][4], corrl[KVG];
#pragma unroll
    for (int h = 0; h < KVG; ++h) {
        corrl[h] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc16[h][j] = 0.f;
    }
    __syncthreads();

    for (int base = s_start; base < s_end; base += BLOCK_N) {
        const int nmax = min(BLOCK_N, s_end - base);
        const int n = base + tid;
        const bool ok = n < s_end;
        const long loc = ok ? kv_indices[kv_start + n] : 0;

        const uint8_t* ip = K_Idx + loc * stride_ki_bs + (long)kvh * NG;
        const uint4 i0 = *reinterpret_cast<const uint4*>(ip);
        const uint4 i1 = *reinterpret_cast<const uint4*>(ip + 16);
        unsigned int iw[8];
        iw[0] = i0.x; iw[1] = i0.y; iw[2] = i0.z; iw[3] = i0.w;
        iw[4] = i1.x; iw[5] = i1.y; iw[6] = i1.z; iw[7] = i1.w;

        // One 8 B load per group returns all four heads' partial scores.
        __half2 a01 = __floats2half2_rn(0.f, 0.f), a23 = __floats2half2_rn(0.f, 0.f);
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int idx = (iw[g >> 2] >> ((g & 3) * 8)) & 0xFF;
            const uint2 e = *reinterpret_cast<const uint2*>(
                &lut_s[(g * KC + idx) * KVG]);
            union { unsigned int u; __half2 h2; } c0, c1;
            c0.u = e.x; c1.u = e.y;
            a01 = __hadd2(a01, c0.h2);
            a23 = __hadd2(a23, c1.h2);
        }
        const float ksc = ok ? K_SZ[loc * stride_ksz_bs + kvh * 2 + 0] : 0.f;
        float qkv[KVG];
        qkv[0] = __low2float(a01);  qkv[1] = __high2float(a01);
        qkv[2] = __low2float(a23);  qkv[3] = __high2float(a23);
#pragma unroll
        for (int h = 0; h < KVG; ++h)
            qkv[h] = ok ? qkv[h] * ksc * sm_scale : -INFINITY;

        const uint8_t* vp = V_Buf + loc * stride_vb_bs + (long)kvh * (L / 4);
        const uint4 v0 = *reinterpret_cast<const uint4*>(vp);
        const uint4 v1 = *reinterpret_cast<const uint4*>(vp + 16);
        const float vsc = ok ? V_SZ[loc * stride_vsz_bs + kvh * 2 + 0] : 0.f;
        const float vze = ok ? V_SZ[loc * stride_vsz_bs + kvh * 2 + 1] : 0.f;
        __syncthreads();
        *reinterpret_cast<uint4*>(vpk_s + tid * (L / 4))      = v0;
        *reinterpret_cast<uint4*>(vpk_s + tid * (L / 4) + 16) = v1;

        float mloc[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            mloc[h] = qkv[h];
            for (int o = 16; o; o >>= 1)
                mloc[h] = fmaxf(mloc[h], __shfl_down_sync(0xffffffff, mloc[h], o));
            if (lane == 0) redm[h][warp] = mloc[h];
        }
        __syncthreads();

        float mnew[KVG], rsc[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            float bmax = redm[h][0];
#pragma unroll
            for (int w = 1; w < NW; ++w) bmax = fmaxf(bmax, redm[h][w]);
            mnew[h] = fmaxf(m_s[h], bmax);
            rsc[h] = (m_s[h] == -INFINITY) ? 0.f : __expf(m_s[h] - mnew[h]);
            corrl[h] *= rsc[h];
#pragma unroll
            for (int j = 0; j < 4; ++j) acc16[h][j] *= rsc[h];
        }

        float pv4[KVG], sloc[KVG];
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            const float e = (qkv[h] == -INFINITY) ? 0.f : __expf(qkv[h] - mnew[h]);
            sloc[h] = e;
            pv4[h] = e * vsc;
            corrl[h] += pv4[h] * vze;
        }
        qk_s[tid] = make_float4(pv4[0], pv4[1], pv4[2], pv4[3]);
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            for (int o = 16; o; o >>= 1) sloc[h] += __shfl_down_sync(0xffffffff, sloc[h], o);
            if (lane == 0) reds[h][warp] = sloc[h];
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < KVG; ++h) {
            float sm = reds[h][0];
#pragma unroll
            for (int w = 1; w < NW; ++w) sm += reds[h][w];
            if (tid == 0) { m_s[h] = mnew[h]; l_s[h] = l_s[h] * rsc[h] + sm; }
        }

#pragma unroll 4
        for (int nl = warp; nl < nmax; nl += NW) {
            const unsigned int byte = vpk_s[nl * (L / 4) + lane];
            const float4 pp = qk_s[nl];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float q2 = (float)((byte >> (2 * j)) & 0x3);
                acc16[0][j] = fmaf(pp.x, q2, acc16[0][j]);
                acc16[1][j] = fmaf(pp.y, q2, acc16[1][j]);
                acc16[2][j] = fmaf(pp.z, q2, acc16[2][j]);
                acc16[3][j] = fmaf(pp.w, q2, acc16[3][j]);
            }
        }
    }

    __syncthreads();
#pragma unroll
    for (int h = 0; h < KVG; ++h)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc_s[((warp * KVG + h) * 4 + j) * 32 + lane] = acc16[h][j];
#pragma unroll
    for (int h = 0; h < KVG; ++h) {
        float c = corrl[h];
        for (int o = 16; o; o >>= 1) c += __shfl_down_sync(0xffffffff, c, o);
        if (lane == 0) reds[h][warp] = c;
    }
    __syncthreads();
    if (tid >= KVG * 32) return;
    const int oh = tid >> 5, ol = tid & 31;
    float corr = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) corr += reds[oh][w];
    const long qh = (long)kvh * KVG + oh;
    const long ob = ((long)bb * (gridDim.y * KVG) + qh) * SPLITS_ + bs;
    const float inv = 1.f / l_s[oh];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float t = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) t += acc_s[((w * KVG + oh) * 4 + j) * 32 + ol];
        Att_Out[ob * L + j * 32 + ol] = (t - corr) * inv;
    }
    if (ol == 0) Att_Lse[ob] = m_s[oh] + logf(l_s[oh]);
}

void vq2_stage1(torch::Tensor q, torch::Tensor k_idx, torch::Tensor cb,
                torch::Tensor v_buf, torch::Tensor k_sz, torch::Tensor v_sz,
                torch::Tensor kv_indptr, torch::Tensor kv_indices,
                torch::Tensor splits, torch::Tensor att_out, torch::Tensor att_lse,
                int64_t n_splits, double sm_scale, int64_t tpt, int64_t h2, int64_t pv, int64_t thr) {
    const int B = q.size(0), H_KV_ = k_idx.size(1);
    dim3 grid(B, H_KV_, n_splits);
#define ARGS                                                                   \
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),               \
        k_idx.data_ptr<uint8_t>(), cb.data_ptr<int>(),                         \
        v_buf.data_ptr<uint8_t>(), k_sz.data_ptr<float>(), v_sz.data_ptr<float>(), \
        kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(),                 \
        splits.data_ptr<int>(), att_out.data_ptr<float>(),                     \
        att_lse.data_ptr<float>(), (int)n_splits, (float)sm_scale,             \
        k_idx.stride(0), v_buf.stride(0), k_sz.stride(0), v_sz.stride(0)
#define SMEM(T, THR_, PV_) (int)(sizeof(float4) * KVG * NG + sizeof(int) * NG * KC        \
        + ((THR_) * (T) * (L / 4) > ((THR_) / 32) * KVG * 4 * 32 * 4            \
           ? (THR_) * (T) * (L / 4) : ((THR_) / 32) * KVG * 4 * 32 * 4)         \
        + sizeof(__half2) * KVG * NG * 2                                        \
        + sizeof(float) * (KVG * ((THR_) * (T))                                  \
                          + ((PV_) < 3 ? 2 * ((THR_) * (T)) : 0)))
#define LAUNCH2(T, H, P, THR_) do {                                            \
        const int sm2 = SMEM(T, THR_, P);                                          \
        cudaFuncSetAttribute(vq2_stage1_kernel<T, H, P, THR_>,                  \
            cudaFuncAttributeMaxDynamicSharedMemorySize, sm2);                  \
        vq2_stage1_kernel<T, H, P, THR_><<<grid, THR_, sm2>>>(ARGS);            \
    } while (0)
#define LAUNCH1(T, H, P) do {                                                   \
        if (thr == 256) { LAUNCH2(T, H, P, 256); }                              \
        else            { LAUNCH2(T, H, P, 128); }                              \
    } while (0)
#define LAUNCH(T) do {                                                         \
        if (h2 && pv == 1) { LAUNCH1(T, true, 1); }                            \
        else if (h2 && pv == 2) { LAUNCH1(T, true, 2); }                       \
        else if (h2 && pv == 3) { LAUNCH1(T, true, 3); }                       \
        else if (h2 && pv == 4) { LAUNCH1(T, true, 4); }                       \
        else if (h2 && pv == 9) { LAUNCH1(T, true, 9); }                       \
        else if (h2 && pv == 6) { LAUNCH1(T, true, 6); }                       \
        else if (h2 && pv == 7) { LAUNCH1(T, true, 7); }                       \
        else if (h2 && pv == 8) { LAUNCH1(T, true, 8); }                       \
        else if (h2)   { LAUNCH1(T, true, 0); }                                 \
        else if (pv == 1) { LAUNCH1(T, false, 1); }                             \
        else if (pv == 2) { LAUNCH1(T, false, 2); }                             \
        else if (pv == 3) { LAUNCH1(T, false, 3); }                             \
        else if (pv == 4) { LAUNCH1(T, false, 4); }                             \
        else if (pv == 6) { LAUNCH1(T, false, 6); }                             \
        else if (pv == 7) { LAUNCH1(T, false, 7); }                             \
        else if (pv == 8) { LAUNCH1(T, false, 8); }                             \
        else           { LAUNCH1(T, false, 0); }                                \
    } while (0)
    if (pv == 10) {
#define LSMEM(THR_) (int)(sizeof(__half) * NG * KC * KVG                        \
        + ((THR_) * (L / 4) > ((THR_) / 32) * KVG * 4 * 32 * 4                   \
           ? (THR_) * (L / 4) : ((THR_) / 32) * KVG * 4 * 32 * 4)                \
        + sizeof(float4) * (THR_))
        const int sm2 = LSMEM(128);
        cudaFuncSetAttribute(vq2_lut_kernel<128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, sm2);
        vq2_lut_kernel<128><<<grid, 128, sm2>>>(ARGS);
#undef LSMEM
        return;
    }
    if (pv == 5) {
#define QSMEM(THR_) (int)(sizeof(float4) * KVG * NG + sizeof(int) * NG * KC      \
        + ((THR_) * (L / 4) > ((THR_) / 32) * KVG * 4 * 32 * 4                   \
           ? (THR_) * (L / 4) : ((THR_) / 32) * KVG * 4 * 32 * 4)                \
        + sizeof(float4) * (THR_))
#define QLAUNCH(T) do {                                                          \
        const int sm2 = QSMEM(128);                                              \
        cudaFuncSetAttribute(vq2_qreuse_kernel<T, 128>,                          \
            cudaFuncAttributeMaxDynamicSharedMemorySize, sm2);                   \
        vq2_qreuse_kernel<T, 128><<<grid, 128, sm2>>>(ARGS);                     \
    } while (0)
        switch (tpt) {
            case 1: QLAUNCH(1); break;
            case 2: QLAUNCH(2); break;
            case 4: QLAUNCH(4); break;
            case 8: QLAUNCH(8); break;
            default: TORCH_CHECK(false, "unsupported tpt");
        }
#undef QLAUNCH
#undef QSMEM
        return;
    }
    switch (tpt) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 4: LAUNCH(4); break;
        default: TORCH_CHECK(false, "unsupported tpt");
    }
#undef LAUNCH
#undef LAUNCH1
#undef LAUNCH2
#undef SMEM
#undef ARGS
}
""".replace("_MINBK", str(_MIN_BLOCK_KV)).replace(
    # `#pragma unroll GUNR` does not macro-expand -- pragmas do not expand
    # their arguments -- so the literal is substituted textually.
    "#pragma unroll 4\n            for (int g",
    "#pragma unroll " + os.environ.get("GUNR", "32") + "\n            for (int g"
).replace(
    "#pragma unroll 4\n            for (int nl = warp",
    "#pragma unroll " + os.environ.get("PUNR", "4") + "\n            for (int nl = warp")


def build(dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    total, cache = BATCH * SEQ, BATCH * SEQ + 64
    q = torch.randn(BATCH, H_Q, L, generator=g, device=dev, dtype=torch.float16)
    k_idx = torch.randint(0, KC, (cache, H_KV, NG), generator=g, device=dev, dtype=torch.uint8)
    cent = torch.randn(H_KV, NG, KC, 4, generator=g, device=dev, dtype=torch.float32) * (L ** -0.5)
    cb = cent.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
    v_buf = torch.randint(0, 256, (cache, H_KV, L // 4), generator=g, device=dev, dtype=torch.uint8)
    k_sz = torch.rand(cache, H_KV, 2, generator=g, device=dev) * 0.5 + 0.75
    v_sz = torch.empty(cache, H_KV, 2, device=dev, dtype=torch.float32)
    v_sz[..., 0] = torch.rand(cache, H_KV, generator=g, device=dev) * 0.1 + 0.05
    v_sz[..., 1] = 1.5
    kv_indptr = torch.arange(0, BATCH + 1, device=dev, dtype=torch.int32) * SEQ
    kv_indices = torch.randperm(cache, generator=g, device=dev)[:total].to(torch.int32)
    splits = torch.empty(BATCH, device=dev, dtype=torch.int32)
    seq_lens = torch.full((BATCH,), SEQ, device=dev, dtype=torch.int32)
    from sglang.srt.layers.attention.triton_backend import get_num_kv_splits_triton
    cores = torch.cuda.get_device_properties(0).multi_processor_count
    get_num_kv_splits_triton[(1,)](
        splits, seq_lens, BATCH, 1, H_Q, H_KV, SPLITS, cores,
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


def main() -> int:
    d = build()
    from sglang.srt.layers.attention.triton_ops import decode_attention as D
    ext = load_inline(
        name="vq2_stage1_cuda",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "void vq2_stage1(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, int64_t, double, int64_t, int64_t, int64_t, int64_t);"
        ),
        cuda_sources=CUDA_SRC, functions=["vq2_stage1"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        build_directory=os.environ.get("TORCH_EXT_DIR", "/tmp/vq2s1"), verbose=False,
    )

    def triton_fn():
        D._decode_grouped_att_m_fwd_quant_vq2(
            d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
            d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["splits"],
            SPLITS, SM_SCALE, 0.0)

    def int2_fn():
        D._decode_grouped_att_m_fwd_quant_int2(
            d["q"], d["v_buf"], d["v_buf"], d["k_sz"], d["v_sz"],
            d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["splits"],
            SPLITS, SM_SCALE, 0.0)

    def cuda_fn(tpt=2, h2=1, pv=3, thr=128):
        ext.vq2_stage1(d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
                       d["kv_indptr"], d["kv_indices"], d["splits"], d["out"],
                       d["lse"], SPLITS, SM_SCALE, tpt, h2, pv, thr)

    os.environ.update(SGL_VQ2_BLOCK_N="64", SGL_VQ2_BLOCK_H="8",
                      SGL_VQ2_NUM_WARPS="2", SGL_VQ2_NUM_STAGES="3")
    print(f"shape bs={BATCH} ctx={SEQ} splits={int(d['splits'].min())}")

    if os.environ.get("PROFILE"):
        triton_fn(); torch.cuda.synchronize()
        cuda_fn(int(os.environ.get("TPT", 2)), int(os.environ.get("H2", 1)),
                int(os.environ.get("PV", 3)), int(os.environ.get("THR", 128)))
        torch.cuda.synchronize()
        return 0

    triton_fn(); torch.cuda.synchronize()
    o_ref, l_ref = d["out"].clone(), d["lse"].clone()
    scale_o = o_ref.abs().max().item()
    fin = torch.isfinite(l_ref)

    ms_i = bench(int2_fn)
    ms_t = bench(triton_fn)
    print(f"\n  int2   (target)        {ms_i*1e3:8.1f} us")
    print(f"  triton vq2 (shipped)   {ms_t*1e3:8.1f} us   {ms_t/ms_i:.2f}x int2\n")
    print(f"  {'variant':>18} {'us':>9} {'vs int2':>9} {'vs triton':>10} "
          f"{'rel out':>9} {'rel lse':>9}")
    # (pv, threads): 3 = the tuned kernel; 0 = the first working PV
    # assignment; 10 = the LUT variant. 4/9 are FMA ablations and
    # 6/7/8 are phase ablations -- see the PV comment in the kernel.
    for pv, thr in ((3, 128), (0, 128), (10, 128)):
      for h2 in (0, 1):
        for tpt in (1,):
            d["out"].zero_(); d["lse"].zero_()
            try:
                cuda_fn(tpt, h2, pv, thr); torch.cuda.synchronize()
            except Exception as e:
                print(f"  {'h2' if h2 else 'fp32'} tpt={tpt}: {type(e).__name__}: {e}")
                continue
            eo = (d["out"] - o_ref).abs().max().item() / max(scale_o, 1e-9)
            el = ((d["lse"][fin] - l_ref[fin]).abs().max().item()
                  / max(l_ref[fin].abs().max().item(), 1e-9))
            ms = bench(lambda: cuda_fn(tpt, h2, pv, thr))
            tag = f"{'h2' if h2 else 'f32'} tpt={tpt} pv{pv} thr{thr}"
            print(f"  {tag:>18} {ms*1e3:>9.1f} {ms/ms_i:>8.2f}x {ms/ms_t:>9.2f}x "
                  f"{eo:>9.1e} {el:>9.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
