// fused_s1b.cu — decode page into SRAM, map pos->signed idx, dequant to r̂ float, emit.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SCALE_BITS 14
#define TOTAL (1u << SCALE_BITS)
#define MASK (TOTAL - 1u)
#define RANS_L (1u << 23)

__global__ void s1b_kernel(
    const uint8_t* __restrict__ blobs,
    const long*    __restrict__ page_byte_off,
    const long*    __restrict__ lane_off,
    const int*     __restrict__ k0a,
    const int*     __restrict__ k1a,
    const int*     __restrict__ ns,
    const int*     __restrict__ nonconst,
    const unsigned int* __restrict__ cdf_flat,
    const int*     __restrict__ cdf_off,
    const long*    __restrict__ vals,      // (d, A_max) signed quant indices, row-major
    const int*     __restrict__ cv,        // (d,) const value per coord
    const unsigned char* __restrict__ nc_mask, // (d,) 1 if non-constant
    float delta, float dz, int A_max,
    int N, int C, int P, int d, int n_pages,
    float* __restrict__ rhat_out)          // (n_pages, P, d) float
{
    int p = blockIdx.x;
    int lane = threadIdx.x;
    if (p >= n_pages) return;

    extern __shared__ int s_pos[];          // P*d ints (positions)
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

    // map pos -> signed idx (vals[j, pos]) then dequant -> r̂ float. const coords -> cv[j].
    int nrows = ns[p];
    for (int i = lane; i < P * d; i += blockDim.x) {
        int t = i / d, j = i % d;
        float out;
        if (t >= nrows) {
            out = 0.0f;
        } else if (nc_mask[j]) {
            int pos = s_pos[i];
            long idx = vals[(long)j * A_max + pos];     // signed quant index
            float fi = (float)idx;
            float sgn = (fi > 0.f) ? 1.f : ((fi < 0.f) ? -1.f : 0.f);
            out = sgn * (fabsf(fi) + (0.5f - dz)) * delta;
        } else {
            int idx = cv[j];
            float fi = (float)idx;
            float sgn = (fi > 0.f) ? 1.f : ((fi < 0.f) ? -1.f : 0.f);
            out = sgn * (fabsf(fi) + (0.5f - dz)) * delta;
        }
        rhat_out[(long)p * P * d + i] = out;
    }
}

torch::Tensor s1b_decode(
    torch::Tensor blobs, torch::Tensor page_byte_off, torch::Tensor lane_off,
    torch::Tensor k0a, torch::Tensor k1a, torch::Tensor ns, torch::Tensor nonconst,
    torch::Tensor cdf_flat, torch::Tensor cdf_off,
    torch::Tensor vals, torch::Tensor cv, torch::Tensor nc_mask,
    double delta, double dz, int64_t A_max,
    int64_t N, int64_t C, int64_t P, int64_t d)
{
    int n_pages = ns.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(blobs.device());
    auto rhat = torch::zeros({n_pages, P, d}, opts);
    int threads = ((int)N + 31) / 32 * 32;
    size_t shmem = (size_t)P * d * sizeof(int);
    s1b_kernel<<<n_pages, threads, shmem>>>(
        blobs.data_ptr<uint8_t>(), page_byte_off.data_ptr<long>(), lane_off.data_ptr<long>(),
        k0a.data_ptr<int>(), k1a.data_ptr<int>(), ns.data_ptr<int>(), nonconst.data_ptr<int>(),
        reinterpret_cast<const unsigned int*>(cdf_flat.data_ptr<int32_t>()), cdf_off.data_ptr<int>(),
        vals.data_ptr<long>(), cv.data_ptr<int>(), nc_mask.data_ptr<unsigned char>(),
        (float)delta, (float)dz, (int)A_max, (int)N, (int)C, (int)P, (int)d, n_pages,
        rhat.data_ptr<float>());
    return rhat;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("s1b_decode", &s1b_decode, "stage1b: decode + vals-map + dequant -> r̂ in SRAM");
}