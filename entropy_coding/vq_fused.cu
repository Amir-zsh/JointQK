#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fused VQ decode: block per (head, group) stages that group's codebook
// (K*G halves = 48KB) into SHARED memory, then streams the group's T indices
// (group-major layout -> COALESCED index reads), looks up each codeword in
// shared (fast, no scattered global gather), and accumulates (proxy for the
// fused attention consumption -- NO fp16 write-back to HBM). This is the
// serving-relevant decode: HBM traffic = one codebook read (resident/amortized)
// + the ~2b/coord index stream, same as INT2.
__global__ void vq_fused_kernel(
    const short* __restrict__ idx_gm, // [n_heads, NG, T] int16 (K<=65536) -> coalesced
    const __half* __restrict__ cb,    // [n_heads, NG, K, G]
    float* __restrict__ sink,
    int n_heads, int T, int NG, int K, int G)
{
    int hg = blockIdx.x;              // head*NG + group
    extern __shared__ __half s_cb[];  // K*G
    const __half* cb_g = cb + (long)hg * K * G;
    for (int i = threadIdx.x; i < K * G; i += blockDim.x) s_cb[i] = cb_g[i];
    __syncthreads();

    const short* idx_g = idx_gm + (long)hg * T;  // this group's T indices, contiguous
    float acc = 0.f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {   // coalesced index reads
        int e = (int)(unsigned short)idx_g[t];
        const __half* cw = s_cb + (long)e * G;
        #pragma unroll
        for (int j = 0; j < 16; ++j) if (j < G) acc += __half2float(cw[j]);
    }
    atomicAdd(sink, acc);
}

double vq_fused(torch::Tensor idx_gm, torch::Tensor cb,
                int64_t n_heads, int64_t T, int64_t NG, int64_t K, int64_t G) {
    auto sink = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(idx_gm.device()));
    int nblocks = (int)(n_heads * NG);
    int threads = 256;
    size_t shmem = (size_t)K * G * sizeof(__half);
    cudaFuncSetAttribute(vq_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    vq_fused_kernel<<<nblocks, threads, shmem>>>(
        (const short*)idx_gm.data_ptr<int16_t>(), (const __half*)cb.data_ptr<at::Half>(),
        sink.data_ptr<float>(), (int)n_heads, (int)T, (int)NG, (int)K, (int)G);
    return 0.0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vq_fused", &vq_fused, "fused VQ decode (codebook in shared, coalesced indices)");
}
