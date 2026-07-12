// fused_attn_do.cu — FIX A (base): DECODE-ONCE schedule.
// Schedule change only; the decode block is identical to the proven naive kernel
// (preserves the 6.4e-7 correctness). One block per KV head, page-outer:
//   for each page: decode ONCE into SRAM, then sweep every (g, query-row) that
//   attends it, carrying online-softmax (m, l, acc) in HBM across the page loop.
// Decodes drop from ~nb^2/2 (per query-tile) to nb (once). K never materialized
// in HBM as fp16; only the partial accumulator (Out) and m/l scratch live in HBM.
//
// do_attn=0 skips the attention sweep -> times pure decode-once, so you can split
// the remaining cost into decode vs attention and decide whether split-K is worth it.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SCALE_BITS 14
#define TOTAL (1u << SCALE_BITS)
#define MASK (TOTAL - 1u)
#define RANS_L (1u << 23)

extern __shared__ unsigned char smem_raw[];

__global__ void fused_attn_do_kernel(
    const uint8_t* __restrict__ blobs,
    const long*    __restrict__ page_byte_off,
    const long*    __restrict__ lane_off,
    const int*     __restrict__ k0a,
    const int*     __restrict__ k1a,
    const int*     __restrict__ ns,
    const int*     __restrict__ rung_of_page,
    const int*     __restrict__ nonconst_cat,
    const int*     __restrict__ nonconst_base,
    const int*     __restrict__ C_all,
    const unsigned int* __restrict__ cdf_flat,
    const int*     __restrict__ cdf_flat_base,
    const int*     __restrict__ cdf_off_all,
    const long*    __restrict__ vals_all,
    const int*     __restrict__ cv_all,
    const unsigned char* __restrict__ nc_mask_all,
    const float*   __restrict__ delta_all,
    const float*   __restrict__ Q,     // (gs, T, d)  pre-projected q'
    const float*   __restrict__ V,     // (T, d)
    float*         __restrict__ Out,   // (gs, T, d)  accumulator -> finalized in place
    float*         __restrict__ Mbuf,  // (gs, T)     online-softmax m
    float*         __restrict__ Lbuf,  // (gs, T)     online-softmax l
    float sm_scale, float dz, int A_max,
    int N, int P, int d, int T, int nb, int gs, int do_attn)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float* s_rhat = (float*)smem_raw;          // P*d floats
    int*   s_pos  = (int*)(s_rhat + P * d);     // P*d ints

    long QT = (long)gs * T;

    // init online-softmax state in HBM
    for (long i = tid; i < QT; i += nthreads) { Mbuf[i] = -INFINITY; Lbuf[i] = 0.0f; }
    for (long i = tid; i < QT * d; i += nthreads) Out[i] = 0.0f;
    __syncthreads();

    for (int pg = 0; pg < nb; ++pg) {
        int ri = rung_of_page[pg];
        int C = C_all[ri];
        const int* nonconst = nonconst_cat + nonconst_base[ri];
        int cflat_base = cdf_flat_base[ri];
        const int* coff = cdf_off_all + ri * (d + 1);
        const long* vals = vals_all + (long)ri * d * A_max;
        const int* cv = cv_all + ri * d;
        const unsigned char* ncm = nc_mask_all + ri * d;
        float delta = delta_all[ri];
        int nrows = ns[pg];

        for (int i = tid; i < P * d; i += nthreads) s_pos[i] = 0;
        __syncthreads();

        // ---- decode page (lanes) : IDENTICAL to naive ----
        if (tid < N) {
            int kk0 = k0a[pg * N + tid];
            int kk1 = k1a[pg * N + tid];
            if (kk0 < kk1) {
                const uint8_t* buf = blobs + page_byte_off[pg] + lane_off[pg * N + tid];
                unsigned int x = (unsigned int)buf[0] | ((unsigned int)buf[1] << 8) |
                                 ((unsigned int)buf[2] << 16) | ((unsigned int)buf[3] << 24);
                int bp = 4;
                for (int k = kk0; k < kk1; ++k) {
                    int t = k / C;
                    int j = nonconst[k % C];
                    unsigned int slot = x & MASK;
                    int lo = cflat_base + coff[j];
                    int hi = cflat_base + coff[j + 1] - 1;
                    int a = lo, b = hi - 1;
                    while (a < b) {
                        int mid = (a + b + 1) >> 1;
                        if (cdf_flat[mid] <= slot) a = mid; else b = mid - 1;
                    }
                    int s = a - lo;
                    unsigned int start = cdf_flat[lo + s];
                    unsigned int freq = cdf_flat[lo + s + 1] - start;
                    x = freq * (x >> SCALE_BITS) + slot - start;
                    while (x < RANS_L) { x = (x << 8) | (unsigned int)buf[bp]; ++bp; }
                    s_pos[t * d + j] = s;
                }
            }
        }
        __syncthreads();

        // ---- vals-map + dequant -> s_rhat : IDENTICAL to naive ----
        for (int i = tid; i < P * d; i += nthreads) {
            int t = i / d, j = i % d;
            float out;
            if (t >= nrows) { out = 0.0f; }
            else {
                long idx;
                if (ncm[j]) idx = vals[(long)j * A_max + s_pos[i]];
                else        idx = cv[j];
                float fi = (float)idx;
                float sgn = (fi > 0.f) ? 1.f : ((fi < 0.f) ? -1.f : 0.f);
                out = sgn * (fabsf(fi) + (0.5f - dz)) * delta;
            }
            s_rhat[i] = out;
        }
        __syncthreads();

        // ---- attention: sweep (g, r) pairs attending this page ----
        if (do_attn) {
            int base_n = pg * P;
            for (long pair = tid; pair < QT; pair += nthreads) {
                int g = (int)(pair / T);
                int r = (int)(pair % T);
                if (r < base_n) continue;                 // page entirely in r's future
                int kmax = (r - base_n < nrows - 1) ? (r - base_n) : (nrows - 1);
                if (kmax < 0) continue;

                long so = (long)g * T + r;
                float m_i = Mbuf[so], l_i = Lbuf[so];
                float* accp = Out + so * d;
                const float* qrow = Q + so * d;
                float acc[128];
                for (int c = 0; c < d; ++c) acc[c] = accp[c];

                for (int kk = 0; kk <= kmax; ++kk) {
                    int n = base_n + kk;
                    float s = 0.0f;
                    for (int c = 0; c < d; ++c) s += qrow[c] * s_rhat[kk * d + c];
                    s *= sm_scale;
                    float m_new = fmaxf(m_i, s);
                    float corr = __expf(m_i - m_new);
                    float pe = __expf(s - m_new);
                    l_i = l_i * corr + pe;
                    for (int c = 0; c < d; ++c)
                        acc[c] = acc[c] * corr + pe * V[(long)n * d + c];
                    m_i = m_new;
                }
                Mbuf[so] = m_i; Lbuf[so] = l_i;
                for (int c = 0; c < d; ++c) accp[c] = acc[c];
            }
        }
        __syncthreads();   // page barrier before next page clobbers s_rhat/s_pos
    }

    // ---- finalize : acc /= l ----
    if (do_attn) {
        for (long pair = tid; pair < QT; pair += nthreads) {
            float l = Lbuf[pair];
            float inv = (l > 0.f) ? (1.0f / l) : 0.0f;
            float* accp = Out + pair * d;
            for (int c = 0; c < d; ++c) accp[c] *= inv;
        }
    }
}

torch::Tensor fused_attn_do(
    torch::Tensor blobs, torch::Tensor page_byte_off, torch::Tensor lane_off,
    torch::Tensor k0a, torch::Tensor k1a, torch::Tensor ns, torch::Tensor rung_of_page,
    torch::Tensor nonconst_cat, torch::Tensor nonconst_base, torch::Tensor C_all,
    torch::Tensor cdf_flat, torch::Tensor cdf_flat_base, torch::Tensor cdf_off_all,
    torch::Tensor vals_all, torch::Tensor cv_all, torch::Tensor nc_mask_all, torch::Tensor delta_all,
    torch::Tensor Q, torch::Tensor V,
    double sm_scale, double dz, int64_t A_max,
    int64_t N, int64_t P, int64_t d, int64_t T, int64_t nb, int64_t gs, int64_t do_attn)
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(blobs.device());
    auto Out  = torch::zeros({gs, T, d}, opts);
    auto Mbuf = torch::empty({gs, T}, opts);
    auto Lbuf = torch::empty({gs, T}, opts);

    int threads = 256;                         // 8 warps; shmem-bound 2 blocks/SM -> ~25% occ
    dim3 grid(1);                              // one block per KV head (launch per head)
    size_t shmem = (size_t)(P * d) * sizeof(float) + (size_t)(P * d) * sizeof(int);
    cudaFuncSetAttribute(fused_attn_do_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    fused_attn_do_kernel<<<grid, threads, shmem>>>(
        blobs.data_ptr<uint8_t>(), page_byte_off.data_ptr<long>(), lane_off.data_ptr<long>(),
        k0a.data_ptr<int>(), k1a.data_ptr<int>(), ns.data_ptr<int>(), rung_of_page.data_ptr<int>(),
        nonconst_cat.data_ptr<int>(), nonconst_base.data_ptr<int>(), C_all.data_ptr<int>(),
        reinterpret_cast<const unsigned int*>(cdf_flat.data_ptr<int32_t>()),
        cdf_flat_base.data_ptr<int>(), cdf_off_all.data_ptr<int>(),
        vals_all.data_ptr<long>(), cv_all.data_ptr<int>(), nc_mask_all.data_ptr<unsigned char>(),
        delta_all.data_ptr<float>(),
        Q.data_ptr<float>(), V.data_ptr<float>(), Out.data_ptr<float>(),
        Mbuf.data_ptr<float>(), Lbuf.data_ptr<float>(),
        (float)sm_scale, (float)dz, (int)A_max, (int)N, (int)P, (int)d, (int)T, (int)nb, (int)gs,
        (int)do_attn);
    return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attn_do", &fused_attn_do,
          "decode-once page-outer fused attention, compressed K in SRAM");
}
