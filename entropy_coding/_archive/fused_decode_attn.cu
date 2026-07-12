// fused_decode_attn.cu — DECODE-PHASE, parallel over keys. One block per (gs-row, head).
// Each thread owns a strip of keys, maintains local online-softmax (m,l,acc[d]) over its
// strip, then ONE tree-merge at the end. No per-key __syncthreads. Decode pages into SRAM
// once per page (shared across the block's threads), then all threads sweep their keys.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SCALE_BITS 14
#define TOTAL (1u << SCALE_BITS)
#define MASK (TOTAL - 1u)
#define RANS_L (1u << 23)
#define MAXD 128

extern __shared__ unsigned char smem_raw[];

__global__ void decode_attn_kernel(
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
    const float*   __restrict__ Q,     // (gs,1,d) projected q'
    const float*   __restrict__ V,     // (T,d)
    float*         __restrict__ Out,   // (gs,1,d)
    float sm_scale, float dz, int A_max,
    int N, int P, int d, int T, int nb, int gs)
{
    int g = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (g >= gs) return;

    float* s_rhat = (float*)smem_raw;               // P*d floats (current page residuals)
    int*   s_pos  = (int*)(s_rhat + P * d);          // P*d ints
    float* s_q    = (float*)(s_pos + P * d);         // d floats (the query)

    // load query
    for (int c = tid; c < d; c += nthreads) s_q[c] = Q[(long)g * d + c];
    __syncthreads();

    // per-thread local online-softmax over the keys THIS thread will handle
    float m_t = -INFINITY, l_t = 0.0f;
    float acc_t[MAXD];
    #pragma unroll
    for (int c = 0; c < MAXD; ++c) acc_t[c] = 0.0f;

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

        // decode page (lanes) into s_pos
        if (tid < N) {
            int kk0 = k0a[pg*N+tid], kk1 = k1a[pg*N+tid];
            if (kk0 < kk1) {
                const uint8_t* buf = blobs + page_byte_off[pg] + lane_off[pg*N+tid];
                unsigned int x = (unsigned int)buf[0]|((unsigned int)buf[1]<<8)|((unsigned int)buf[2]<<16)|((unsigned int)buf[3]<<24);
                int bp = 4;
                for (int k = kk0; k < kk1; ++k) {
                    int t = k/C; int j = nonconst[k%C];
                    unsigned int slot = x & MASK;
                    int lo = cflat_base+coff[j], hi = cflat_base+coff[j+1]-1, a=lo, b=hi-1;
                    while (a<b){int mid=(a+b+1)>>1; if(cdf_flat[mid]<=slot)a=mid; else b=mid-1;}
                    int s=a-lo; unsigned int start=cdf_flat[lo+s], freq=cdf_flat[lo+s+1]-start;
                    x = freq*(x>>SCALE_BITS)+slot-start;
                    while(x<RANS_L){x=(x<<8)|(unsigned int)buf[bp];++bp;}
                    s_pos[t*d+j]=s;
                }
            }
        }
        __syncthreads();

        // dequant -> s_rhat
        for (int i = tid; i < P*d; i += nthreads) {
            int t=i/d, j=i%d; float out;
            if (t>=nrows) out=0.f;
            else { long idx = ncm[j]?vals[(long)j*A_max+s_pos[i]]:(long)cv[j];
                   float fi=(float)idx; float sg=(fi>0.f)?1.f:((fi<0.f)?-1.f:0.f);
                   out=sg*(fabsf(fi)+(0.5f-dz))*delta; }
            s_rhat[i]=out;
        }
        __syncthreads();

        // each thread handles keys kk = tid, tid+nthreads, ... within this page
        int base_n = pg * P;
        for (int kk = tid; kk < nrows; kk += nthreads) {
            int n = base_n + kk;
            if (n >= T) break;
            float score = 0.0f;
            const float* rk = s_rhat + kk*d;
            for (int c = 0; c < d; ++c) score += s_q[c] * rk[c];
            score *= sm_scale;
            float m_new = fmaxf(m_t, score);
            float corr = __expf(m_t - m_new);
            float pe = __expf(score - m_new);
            l_t = l_t * corr + pe;
            const float* vn = V + (long)n*d;
            for (int c = 0; c < d; ++c) acc_t[c] = acc_t[c]*corr + pe*vn[c];
            m_t = m_new;
        }
        __syncthreads();   // page barrier before s_rhat reused
    }

    // ---- merge per-thread (m_t,l_t,acc_t) across the block via shared memory ----
    __shared__ float red_m[256], red_l[256];
    extern __shared__ unsigned char smem_raw2[];   // not used; reuse static below
    // store this thread's partials
    red_m[tid] = m_t; red_l[tid] = l_t;
    // accumulators in shared: we merge pairwise. Use s_rhat region as scratch for acc.
    // Layout scratch: float acc_all[nthreads*d] would be huge; instead tree-merge in place
    // using a shared acc buffer of size nthreads*? — too big. Use sequential merge by thread 0.
    __syncthreads();

    // thread 0 merges all partials (nthreads of them). Each thread first writes its acc to
    // global scratch via Out? No — write acc_t to a shared tile in chunks. Simplit: write
    // each thread's (m,l,acc) to global temp is costly. Use shared accbuf sized nthreads*d
    // only if it fits; for d=128, nthreads=128 -> 64KB, fits in 164KB with s_rhat freed.
    // We reuse s_rhat (P*d floats = 64*128=8192) — too small. So cap nthreads for merge.
    // -> simple correct approach: tree merge with one acc row in shared at a time.
    __shared__ float gm, gl, gacc[MAXD];
    if (tid == 0) {
        gm = -INFINITY; gl = 0.0f;
        #pragma unroll
        for (int c=0;c<MAXD;++c) gacc[c]=0.0f;
    }
    __syncthreads();
    // serialize threads' merges (correct, not fast; nthreads merges)
    for (int p = 0; p < nthreads; ++p) {
        if (tid == p) {
            float m_new = fmaxf(gm, m_t);
            float cg = __expf(gm - m_new);
            float ct = __expf(m_t - m_new);
            gl = gl*cg + l_t*ct;
            for (int c=0;c<d;++c) gacc[c] = gacc[c]*cg + acc_t[c]*ct;
            gm = m_new;
        }
        __syncthreads();
    }

    float inv = (gl > 0.f) ? (1.0f/gl) : 0.0f;
    for (int c = tid; c < d; c += nthreads) Out[(long)g*d + c] = gacc[c]*inv;
}

torch::Tensor decode_attn(
    torch::Tensor blobs, torch::Tensor page_byte_off, torch::Tensor lane_off,
    torch::Tensor k0a, torch::Tensor k1a, torch::Tensor ns, torch::Tensor rung_of_page,
    torch::Tensor nonconst_cat, torch::Tensor nonconst_base, torch::Tensor C_all,
    torch::Tensor cdf_flat, torch::Tensor cdf_flat_base, torch::Tensor cdf_off_all,
    torch::Tensor vals_all, torch::Tensor cv_all, torch::Tensor nc_mask_all, torch::Tensor delta_all,
    torch::Tensor Q, torch::Tensor V,
    double sm_scale, double dz, int64_t A_max,
    int64_t N, int64_t P, int64_t d, int64_t T, int64_t nb, int64_t gs)
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(blobs.device());
    auto Out = torch::zeros({gs,1,d}, opts);
    int threads = 128;
    dim3 grid(gs);
    size_t shmem = (size_t)(P*d)*sizeof(float) + (size_t)(P*d)*sizeof(int) + (size_t)d*sizeof(float);
    cudaFuncSetAttribute(decode_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    decode_attn_kernel<<<grid, threads, shmem>>>(
        blobs.data_ptr<uint8_t>(), page_byte_off.data_ptr<long>(), lane_off.data_ptr<long>(),
        k0a.data_ptr<int>(), k1a.data_ptr<int>(), ns.data_ptr<int>(), rung_of_page.data_ptr<int>(),
        nonconst_cat.data_ptr<int>(), nonconst_base.data_ptr<int>(), C_all.data_ptr<int>(),
        reinterpret_cast<const unsigned int*>(cdf_flat.data_ptr<int32_t>()),
        cdf_flat_base.data_ptr<int>(), cdf_off_all.data_ptr<int>(),
        vals_all.data_ptr<long>(), cv_all.data_ptr<int>(), nc_mask_all.data_ptr<unsigned char>(),
        delta_all.data_ptr<float>(), Q.data_ptr<float>(), V.data_ptr<float>(), Out.data_ptr<float>(),
        (float)sm_scale,(float)dz,(int)A_max,(int)N,(int)P,(int)d,(int)T,(int)nb,(int)gs);
    return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_attn", &decode_attn, "decode-phase parallel-over-keys");
}