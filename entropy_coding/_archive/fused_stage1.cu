// fused_stage1.cu — decode one page into SRAM, emit positions to global.
// Validates SRAM-decode produces positions bit-identical to decode_pages_kernel.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SCALE_BITS 14
#define TOTAL (1u << SCALE_BITS)
#define MASK (TOTAL - 1u)
#define RANS_L (1u << 23)

// one block per page; threadIdx.x = lane. Decode into __shared__ positions, copy to global.
__global__ void stage1_kernel(
    const uint8_t* __restrict__ blobs,
    const long*    __restrict__ page_byte_off,
    const long*    __restrict__ lane_off,
    const int*     __restrict__ k0a,
    const int*     __restrict__ k1a,
    const int*     __restrict__ ns,
    const int*     __restrict__ nonconst,
    const unsigned int* __restrict__ cdf_flat,
    const int*     __restrict__ cdf_off,
    int N, int C, int P, int d, int n_pages,
    int* __restrict__ pos_out)
{
    int p = blockIdx.x;
    int lane = threadIdx.x;
    if (p >= n_pages) return;

    extern __shared__ int s_pos[];          // P*d ints
    for (int i = lane; i < P * d; i += blockDim.x) s_pos[i] = 0;
    __syncthreads();

    if (lane < N) {
        int k0 = k0a[p * N + lane];
        int k1 = k1a[p * N + lane];
        if (k0 < k1) {
            const uint8_t* buf = blobs + page_byte_off[p] + lane_off[p * N + lane];
            unsigned int x = (unsigned int)buf[0] | ((unsigned int)buf[1] << 8) |
                             ((unsigned int)buf[2] << 16) | ((unsigned int)buf[3] << 24);
            int bp = 4;
            for (int k = k0; k < k1; ++k) {
                int t = k / C;
                int j = nonconst[k % C];
                unsigned int slot = x & MASK;
                int lo = cdf_off[j];
                int hi = cdf_off[j + 1] - 1;
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

    // copy SRAM positions -> global (positions only; vals-map + dequant is stage 1b)
    for (int i = lane; i < P * d; i += blockDim.x)
        pos_out[(long)p * P * d + i] = s_pos[i];
}

torch::Tensor stage1_decode(
    torch::Tensor blobs, torch::Tensor page_byte_off, torch::Tensor lane_off,
    torch::Tensor k0a, torch::Tensor k1a, torch::Tensor ns, torch::Tensor nonconst,
    torch::Tensor cdf_flat, torch::Tensor cdf_off, int64_t N, int64_t C, int64_t P, int64_t d)
{
    int n_pages = ns.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(blobs.device());
    auto pos_out = torch::zeros({n_pages, P, d}, opts);
    int threads = ((int)N + 31) / 32 * 32;
    size_t shmem = (size_t)P * d * sizeof(int);
    stage1_kernel<<<n_pages, threads, shmem>>>(
        blobs.data_ptr<uint8_t>(), page_byte_off.data_ptr<long>(), lane_off.data_ptr<long>(),
        k0a.data_ptr<int>(), k1a.data_ptr<int>(), ns.data_ptr<int>(), nonconst.data_ptr<int>(),
        reinterpret_cast<const unsigned int*>(cdf_flat.data_ptr<int32_t>()), cdf_off.data_ptr<int>(),
        (int)N, (int)C, (int)P, (int)d, n_pages, pos_out.data_ptr<int>());
    return pos_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stage1_decode", &stage1_decode, "stage1: decode page into SRAM, emit positions");
}