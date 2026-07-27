"""Decisive probe: is a shared-memory codebook worth a full CUDA kernel?

ncu says the shipped Triton vq2 kernel is L1-SECTOR-THROUGHPUT bound -- 392 M
L1 sectors at 73.8% of peak, against int2's 94 M at 30.4%. The codebook gather
is ~94% of that traffic. Cache hints, eviction policy, group-major addressing
and launch-config sweeps all fail to move it, because none of them reduce
sector count: 32 lanes gathering 4 B each from a 32 KB table cost ~24 sectors
per warp instruction no matter how the table is addressed.

Shared memory has no sectors. A warp gathering from ONE group's 1 KB sub-table
pays only bank conflicts: bank = idx % 32, so 32 random indices give ~3.4-way
conflict instead of ~24 sectors. That should be ~7x cheaper. Triton cannot
express it (tl.gather lowers to warp shuffles and measures 4x SLOWER), and the
TileLang attempt loses to unrelated lowering problems (227 M bank conflicts,
36.8 M global requests, 1.7x instruction bloat).

So this probe isolates exactly the contested part -- gather + score, no V, no
softmax, no PV -- and runs the SAME work in CUDA and in Triton at the served
shape and grid. If CUDA does not win decisively here, the full CUDA kernel is
not worth building and the shipped Triton kernel is the end of the road.

Both kernels reduce to a per-(head, split) max so nothing is dead-code
eliminated, and the two results are compared before either timing is believed.

  docker exec -e CUDA_VISIBLE_DEVICES=0 oscar-ab bash -lc 'cd <repo> && \
    PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
    /opt/venv-oscar/bin/python pipelines/throughput/kernel_study/probe_cuda_qk.py'
"""
import os

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

BATCH = int(os.environ.get("BS", 64))
SEQ = int(os.environ.get("CTX", 30000))
SPLITS = int(os.environ.get("SPLITS", 8))
H_Q, H_KV, L, NG, KC = 32, 8, 128, 32, 256
KVG = H_Q // H_KV                      # 4 q heads per kv head
SM_SCALE = 1.0 / (L ** 0.5)
_MIN_BLOCK_KV = 32

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>

#define NG 32
#define KC 256
#define L  128
#define KVG 4
#define THREADS 128

// e5m2 and fp16 share sign + 5-bit exponent + bias, so the decode is a shift.
__device__ __forceinline__ float fp8e5m2_to_f32(unsigned int b) {
    __half h = __ushort_as_half((unsigned short)(b << 8));
    return __half2float(h);
}

// Two e5m2 bytes -> one half2, in ONE instruction. prmt builds
// [0x00, b_lo, 0x00, b_hi], which is exactly the pair of <<8 shifts that
// decode e5m2 to fp16. The scalar path needs 8 instructions for the same 4
// coords (4 shifts + 4 converts).
__device__ __forceinline__ __half2 unpack2(unsigned int cw, unsigned int sel) {
    union { unsigned int u; __half2 h; } c;
    c.u = __byte_perm(cw, 0u, sel);
    return c.h;
}
#define SEL_LO 0x1404u    // bytes 0,1
#define SEL_HI 0x3424u    // bytes 2,3

// TPT = tokens per thread. Each q4_s broadcast is reused across TPT tokens, so
// shared traffic per token falls from 32 codebook + 128 q to 32 + 128/TPT.
// At TPT=1 the q broadcasts are 4/5 of all shared loads and L1 sits at 93%.
template <int TPT, bool H2>
__global__ __launch_bounds__(THREADS) void qk_max_kernel(
    const __half* __restrict__ Q,          // [B, H_Q, L]
    const uint8_t* __restrict__ K_Idx,     // [cache, H_KV, NG]
    const int*     __restrict__ CB,        // [H_KV, NG, KC]
    const int*     __restrict__ kv_indptr, // [B+1]
    const int*     __restrict__ kv_indices,
    const int*     __restrict__ num_kv_splits,
    float*         __restrict__ Out,       // [B, H_Q, SPLITS]
    int H_KV_, int SPLITS_, float sm_scale, long stride_ki_bs)
{
    const int bb  = blockIdx.x;
    const int kvh = blockIdx.y;
    const int bs  = blockIdx.z;
    const int tid = threadIdx.x;

    // 32 KB table + 2 KB of q. The table is the whole point: every lookup that
    // lands here is a lookup that does not consume an L1 sector.
    __shared__ int    cb_s[NG * KC];
    // q as float4 per (head, group): one 16 B broadcast LDS feeds all four
    // coords of a codeword. Holding q in registers instead costs 512 live
    // values per thread and spills catastrophically (measured: 255 regs,
    // 150 M local loads, 9.5 GB DRAM -- 5x slower than Triton).
    __shared__ float4 q4_s[KVG * NG];
    __shared__ __half2 q2_s[KVG * NG * 2];   // H2 path: q as half2 pairs
    __shared__ float  red[KVG][THREADS / 32];

    const int kv_start = kv_indptr[bb];
    const int seq_len  = kv_indptr[bb + 1] - kv_start;
    const int splits   = num_kv_splits[bb];
    const int per = ((seq_len + splits - 1) / splits + 31) / 32 * 32;
    const int s_start = per * bs;
    const int s_end   = min(s_start + per, seq_len);

    const int* cb_head = CB + (long)kvh * (NG * KC);
    for (int i = tid; i < NG * KC; i += THREADS) cb_s[i] = cb_head[i];
    for (int i = tid; i < KVG * NG; i += THREADS) {
        const int h = i / NG, g = i % NG;
        const __half* qp =
            Q + ((long)bb * gridDim.y * KVG + kvh * KVG + h) * L + g * 4;
        q4_s[i] = make_float4(__half2float(qp[0]), __half2float(qp[1]),
                              __half2float(qp[2]), __half2float(qp[3]));
        q2_s[2 * i + 0] = __halves2half2(qp[0], qp[1]);
        q2_s[2 * i + 1] = __halves2half2(qp[2], qp[3]);
    }
    __syncthreads();

    float best[KVG];
#pragma unroll
    for (int h = 0; h < KVG; ++h) best[h] = -INFINITY;

    // Tokens are strided by THREADS so every index/global load stays coalesced.
    // Every lane is at the same g at the same time, so a warp reads inside ONE
    // 1 KB sub-table and conflicts only on idx % 32.
    for (int base = s_start; base < s_end; base += THREADS * TPT) {
        unsigned int iw[TPT][8];
        bool ok[TPT];
#pragma unroll
        for (int t = 0; t < TPT; ++t) {
            const int n = base + t * THREADS + tid;
            ok[t] = n < s_end;
            const int loc = ok[t] ? kv_indices[kv_start + n] : 0;
            const uint8_t* idxp = K_Idx + (long)loc * stride_ki_bs + (long)kvh * NG;
            // NG=32 index bytes as 8 registers. Taking the address of a local
            // ((uint8_t*)&word) to index it byte-wise makes nvcc demote the
            // whole thing to local memory, so the bytes come out by shifting.
            const uint4 i0 = *reinterpret_cast<const uint4*>(idxp);
            const uint4 i1 = *reinterpret_cast<const uint4*>(idxp + 16);
            iw[t][0] = i0.x; iw[t][1] = i0.y; iw[t][2] = i0.z; iw[t][3] = i0.w;
            iw[t][4] = i1.x; iw[t][5] = i1.y; iw[t][6] = i1.z; iw[t][7] = i1.w;
        }

        float acc[KVG][TPT];
        __half2 acc2[KVG][TPT];
#pragma unroll
        for (int h = 0; h < KVG; ++h)
#pragma unroll
            for (int t = 0; t < TPT; ++t) {
                acc[h][t] = 0.f;
                acc2[h][t] = __floats2half2_rn(0.f, 0.f);
            }

        // Unroll by 4, not 32: the fully unrolled body has 512*TPT FMAs whose
        // live ranges overflow the register file and spill to local memory.
#pragma unroll 4
        for (int g = 0; g < NG; ++g) {
            if (H2) {
                __half2 k01[TPT], k23[TPT];
#pragma unroll
                for (int t = 0; t < TPT; ++t) {
                    const int idx = (iw[t][g >> 2] >> ((g & 3) * 8)) & 0xFF;
                    const unsigned int cw = cb_s[g * KC + idx];
                    k01[t] = unpack2(cw, SEL_LO);
                    k23[t] = unpack2(cw, SEL_HI);
                }
#pragma unroll
                for (int h = 0; h < KVG; ++h) {
                    const __half2 q01 = q2_s[(h * NG + g) * 2 + 0];
                    const __half2 q23 = q2_s[(h * NG + g) * 2 + 1];
#pragma unroll
                    for (int t = 0; t < TPT; ++t) {
                        acc2[h][t] = __hfma2(q01, k01[t], acc2[h][t]);
                        acc2[h][t] = __hfma2(q23, k23[t], acc2[h][t]);
                    }
                }
            } else {
                float kk[TPT][4];
#pragma unroll
                for (int t = 0; t < TPT; ++t) {
                    const int idx = (iw[t][g >> 2] >> ((g & 3) * 8)) & 0xFF;
                    const int cw = cb_s[g * KC + idx];
                    kk[t][0] = fp8e5m2_to_f32(cw & 0xFF);
                    kk[t][1] = fp8e5m2_to_f32((cw >> 8) & 0xFF);
                    kk[t][2] = fp8e5m2_to_f32((cw >> 16) & 0xFF);
                    kk[t][3] = fp8e5m2_to_f32((cw >> 24) & 0xFF);
                }
#pragma unroll
                for (int h = 0; h < KVG; ++h) {
                    const float4 qv = q4_s[h * NG + g];   // amortised over TPT
#pragma unroll
                    for (int t = 0; t < TPT; ++t) {
                        acc[h][t] = fmaf(qv.x, kk[t][0], acc[h][t]);
                        acc[h][t] = fmaf(qv.y, kk[t][1], acc[h][t]);
                        acc[h][t] = fmaf(qv.z, kk[t][2], acc[h][t]);
                        acc[h][t] = fmaf(qv.w, kk[t][3], acc[h][t]);
                    }
                }
            }
        }
#pragma unroll
        for (int h = 0; h < KVG; ++h)
#pragma unroll
            for (int t = 0; t < TPT; ++t) {
                const float v = H2 ? (__low2float(acc2[h][t]) + __high2float(acc2[h][t]))
                                   : acc[h][t];
                if (ok[t]) best[h] = fmaxf(best[h], v * sm_scale);
            }
    }

    const int lane = tid & 31, warp = tid >> 5;
#pragma unroll
    for (int h = 0; h < KVG; ++h) {
        float v = best[h];
        for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
        if (lane == 0) red[h][warp] = v;
    }
    __syncthreads();
    if (tid < KVG) {
        float v = red[tid][0];
        for (int w = 1; w < THREADS / 32; ++w) v = fmaxf(v, red[tid][w]);
        Out[((long)bb * (gridDim.y * KVG) + kvh * KVG + tid) * SPLITS_ + bs] =
            (s_end > s_start) ? v : -INFINITY;
    }
}

void qk_max(torch::Tensor q, torch::Tensor k_idx, torch::Tensor cb,
            torch::Tensor kv_indptr, torch::Tensor kv_indices,
            torch::Tensor splits, torch::Tensor out, int64_t n_splits,
            double sm_scale, int64_t tpt, int64_t h2) {
    const int B = q.size(0), H_KV_ = k_idx.size(1);
    dim3 grid(B, H_KV_, n_splits);
#define ARGS                                                                  \
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),              \
        k_idx.data_ptr<uint8_t>(), cb.data_ptr<int>(),                        \
        kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(),                \
        splits.data_ptr<int>(), out.data_ptr<float>(),                        \
        H_KV_, (int)n_splits, (float)sm_scale, k_idx.stride(0)
#define LAUNCH(T) do {                                                        \
        if (h2) qk_max_kernel<T, true><<<grid, THREADS>>>(ARGS);              \
        else    qk_max_kernel<T, false><<<grid, THREADS>>>(ARGS);             \
    } while (0)
    switch (tpt) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 4: LAUNCH(4); break;
        case 8: LAUNCH(8); break;
        default: TORCH_CHECK(false, "unsupported tpt");
    }
#undef LAUNCH
#undef ARGS
}
"""


@triton.jit
def _qk_max_triton(
    Q, K_Idx, CB, kv_indptr, kv_indices, num_kv_splits, Out,
    sm_scale, stride_qbs, stride_qh, stride_ki_bs, stride_ki_h, stride_ob,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr, L: tl.constexpr, NG: tl.constexpr,
    KC: tl.constexpr, SPLITS: tl.constexpr,
):
    """Same work as the CUDA kernel, in the shipped kernel's idiom: row-major
    gather, join-interleave to [BLOCK_N, L], one K=L dot, running max."""
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_D)
    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    splits = tl.load(num_kv_splits + cur_batch)
    per = tl.cdiv(tl.cdiv(seq_len, splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    s_start = per * split_kv_id
    s_end = tl.minimum(s_start + per, seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    if s_end > s_start:
        q_main = tl.load(
            Q + cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :],
            mask=mask_h[:, None] & (offs_d[None, :] < L), other=0.0,
        ).to(tl.float16)
        offs_g = tl.arange(0, NG)
        cb_base = CB + cur_kv_head.to(tl.int64) * (NG * KC)
        for start_n in range(s_start, s_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < s_end
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0).to(tl.int64)
            isel = tl.load(
                K_Idx + kv_loc[:, None] * stride_ki_bs + cur_kv_head * stride_ki_h
                + offs_g[None, :], mask=mask_n[:, None], other=0,
            ).to(tl.int32)
            cw = tl.load(cb_base + offs_g[None, :] * KC + isel,
                         mask=mask_n[:, None], other=0).to(tl.int32)
            p0 = (cw & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
            p1 = ((cw >> 8) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
            p2 = ((cw >> 16) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
            p3 = ((cw >> 24) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
            kg = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BLOCK_N, NG * 4))
            qk = tl.dot(q_main, tl.trans(kg)) * sm_scale
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))
            e_max = tl.maximum(tl.max(qk, 1), e_max)
    tl.store(Out + cur_batch * stride_ob + cur_head * SPLITS + split_kv_id,
             e_max, mask=mask_h)


def build(dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    total, cache = BATCH * SEQ, BATCH * SEQ + 64
    q = torch.randn(BATCH, H_Q, L, generator=g, device=dev, dtype=torch.float16)
    k_idx = torch.randint(0, KC, (cache, H_KV, NG), generator=g, device=dev, dtype=torch.uint8)
    cent = torch.randn(H_KV, NG, KC, 4, generator=g, device=dev, dtype=torch.float32) * (L ** -0.5)
    cb = cent.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
    kv_indptr = torch.arange(0, BATCH + 1, device=dev, dtype=torch.int32) * SEQ
    kv_indices = torch.randperm(cache, generator=g, device=dev)[:total].to(torch.int32)
    splits = torch.empty(BATCH, device=dev, dtype=torch.int32)
    seq_lens = torch.full((BATCH,), SEQ, device=dev, dtype=torch.int32)
    from sglang.srt.layers.attention.triton_backend import get_num_kv_splits_triton
    cores = torch.cuda.get_device_properties(0).multi_processor_count
    get_num_kv_splits_triton[(1,)](
        splits, seq_lens, BATCH, 1, H_Q, H_KV, SPLITS, cores,
        MAX_NUM_SEQ=256 if BATCH < 256 else triton.next_power_of_2(BATCH))
    out = torch.zeros(BATCH, H_Q, SPLITS, device=dev, dtype=torch.float32)
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
    ext = load_inline(
        name="vq2_qk_probe",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "void qk_max(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double,"
            " int64_t, int64_t);"
        ),
        cuda_sources=CUDA_SRC,
        functions=["qk_max"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        build_directory=os.environ.get("TORCH_EXT_DIR", "/tmp/vq2ext"), verbose=False,
    )

    def cuda_fn(tpt=4, h2=0):
        ext.qk_max(d["q"], d["k_idx"], d["cb"], d["kv_indptr"], d["kv_indices"],
                   d["splits"], d["out"], SPLITS, SM_SCALE, tpt, h2)

    def triton_fn():
        grid = (BATCH, triton.cdiv(H_Q, min(8, KVG)), SPLITS)
        _qk_max_triton[grid](
            d["q"], d["k_idx"], d["cb"], d["kv_indptr"], d["kv_indices"],
            d["splits"], d["out"], SM_SCALE,
            d["q"].stride(0), d["q"].stride(1),
            d["k_idx"].stride(0), d["k_idx"].stride(1), d["out"].stride(0),
            kv_group_num=KVG, q_head_num=H_Q, BLOCK_D=L, BLOCK_N=64, BLOCK_H=8,
            MIN_BLOCK_KV=_MIN_BLOCK_KV, L=L, NG=NG, KC=KC, SPLITS=SPLITS,
            num_warps=2, num_stages=3)

    sp = d["splits"]
    print(f"shape bs={BATCH} ctx={SEQ} splits={int(sp.min())}..{int(sp.max())} "
          f"grid=({BATCH}, {H_KV}, {SPLITS})")

    triton_fn(); torch.cuda.synchronize()
    o_t = d["out"].clone()
    d["out"].zero_()
    cuda_fn(4); torch.cuda.synchronize()
    o_c = d["out"].clone()
    finite = torch.isfinite(o_t) & torch.isfinite(o_c)
    err = (o_c[finite] - o_t[finite]).abs().max().item()
    rel = err / max(o_t[finite].abs().max().item(), 1e-9)
    print(f"validation: max|d| {err:.3e}  rel {rel:.2e}  "
          f"{'OK' if rel < 2e-2 else 'MISMATCH'}   ({int(finite.sum())} finite cells)")

    if os.environ.get("PROFILE"):
        triton_fn(); torch.cuda.synchronize()
        cuda_fn(int(os.environ.get("TPT", 8)), int(os.environ.get("H2", 0)))
        torch.cuda.synchronize()
        return 0

    ms_t = bench(triton_fn)
    print(f"\n  triton gather+qk   {ms_t*1e3:8.1f} us")
    for h2 in (0, 1):
        for tpt in (1, 2, 4, 8):
            d["out"].zero_(); cuda_fn(tpt, h2); torch.cuda.synchronize()
            f = torch.isfinite(o_t) & torch.isfinite(d["out"])
            rel = ((d["out"][f] - o_t[f]).abs().max().item()
                   / max(o_t[f].abs().max().item(), 1e-9))
            ms_c = bench(lambda: cuda_fn(tpt, h2))
            print(f"  CUDA {'half2' if h2 else 'fp32'} tpt={tpt}  {ms_c*1e3:8.1f} us"
                  f"   {ms_t/ms_c:.2f}x   rel {rel:.1e}")
    print(f"\nFor scale: the full shipped vq2 stage-1 is ~1940 us and int2 ~1650 us,")
    print(f"so the contested piece is {ms_t*1e3:.0f} us of that 1940.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
