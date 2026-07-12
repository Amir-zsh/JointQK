#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// VQ decode with the codebook group staged in SHARED memory. One block per
// (head, group): all P*.. tokens of that (head,group) gather from the same
// K*G codebook, so staging it once in shared turns the per-token scattered
// global gather into fast shared-memory random access (no coalescing/L2-miss
// penalty). Output write is the only remaining global traffic -> the kernel
// becomes write-bound, i.e. the INT2 fixed-width class.
__global__ void vq_decode_kernel(
    const int* __restrict__ idx,     // [n_heads, T, NG]  (row-major)
    const __half* __restrict__ cb,   // [n_heads, NG, K, G]
    __half* __restrict__ out,        // [n_heads, T, d_eff]  d_eff = NG*G
    int n_heads, int T, int NG, int K, int G, int d_eff)
{
    int hg = blockIdx.x;             // head*NG + group
    int head = hg / NG;
    int group = hg % NG;

    extern __shared__ __half s_cb[]; // K*G
    const __half* cb_g = cb + (long)hg * K * G;
    for (int i = threadIdx.x; i < K * G; i += blockDim.x) s_cb[i] = cb_g[i];
    __syncthreads();

    long base_idx = (long)head * T * NG + group;
    long base_out = (long)head * T * d_eff + group * G;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        int e = idx[base_idx + (long)t * NG];        // codebook entry [0,K)
        const __half* cw = s_cb + (long)e * G;
        __half* o = out + base_out + (long)t * d_eff;
        #pragma unroll
        for (int j = 0; j < 16; ++j) if (j < G) o[j] = cw[j];
    }
}

torch::Tensor vq_decode(torch::Tensor idx, torch::Tensor cb,
                        int64_t n_heads, int64_t T, int64_t NG, int64_t K, int64_t G, int64_t d_eff) {
    auto out = torch::empty({n_heads * T * d_eff},
                            torch::TensorOptions().dtype(torch::kFloat16).device(idx.device()));
    int nblocks = (int)(n_heads * NG);
    int threads = 256;
    size_t shmem = (size_t)K * G * sizeof(__half);
    cudaFuncSetAttribute(vq_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    vq_decode_kernel<<<nblocks, threads, shmem>>>(
        idx.data_ptr<int>(), (const __half*)cb.data_ptr<at::Half>(),
        (__half*)out.data_ptr<at::Half>(),
        (int)n_heads, (int)T, (int)NG, (int)K, (int)G, (int)d_eff);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vq_decode", &vq_decode, "VQ decode (codebook group in shared memory)");
}
