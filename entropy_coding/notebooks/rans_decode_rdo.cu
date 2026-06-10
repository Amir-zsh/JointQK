#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SCALE_BITS 14
#define TOTAL (1u << SCALE_BITS)
#define MASK (TOTAL - 1u)
#define RANS_L (1u << 23)

__global__ void decode_pages_rdo_kernel(
    const uint8_t* __restrict__ blobs,
    const long*    __restrict__ page_byte_off,
    const long*    __restrict__ lane_off,
    const int*     __restrict__ k0a,
    const int*     __restrict__ k1a,
    const int*     __restrict__ rung_tok,   // (n_pages * P) per-token rung
    const int*     __restrict__ nc0,        // (C,) coded coords (finest-rung nonconst)
    const unsigned int* __restrict__ cdf,   // stacked across all rungs
    const long*    __restrict__ off2d,      // (R*(d+1),) flat offset of (rung,coord)
    int N, int C, int P, int d, int R, int n_pages,
    int* __restrict__ pos_out)
{
    int p = blockIdx.x;
    int lane = threadIdx.x;
    if (p >= n_pages || lane >= N) return;
    int k0 = k0a[p * N + lane];
    int k1 = k1a[p * N + lane];
    if (k0 >= k1) return;

    const uint8_t* buf = blobs + page_byte_off[p] + lane_off[p * N + lane];
    unsigned int x = (unsigned int)buf[0] | ((unsigned int)buf[1] << 8) |
                     ((unsigned int)buf[2] << 16) | ((unsigned int)buf[3] << 24);
    int bp = 4;
    long page_pos_base = (long)p * P * d;
    const int* rtok = rung_tok + (long)p * P;

    for (int k = k0; k < k1; ++k) {
        int t = k / C;
        int j = nc0[k % C];
        int r = rtok[t];
        long o = (long)r * (d + 1) + j;
        int lo = (int)off2d[o];
        int hi = (int)off2d[o + 1] - 1;     // sentinel index
        unsigned int slot = x & MASK;
        int a = lo, b = hi - 1;
        while (a < b) {
            int mid = (a + b + 1) >> 1;
            if (cdf[mid] <= slot) a = mid; else b = mid - 1;
        }
        int s = a - lo;
        unsigned int start = cdf[lo + s];
        unsigned int freq = cdf[lo + s + 1] - start;
        x = freq * (x >> SCALE_BITS) + slot - start;
        while (x < RANS_L) { x = (x << 8) | (unsigned int)buf[bp]; ++bp; }
        pos_out[page_pos_base + (long)t * d + j] = s;
    }
}

torch::Tensor decode_pages_rdo(
    torch::Tensor blobs, torch::Tensor page_byte_off, torch::Tensor lane_off,
    torch::Tensor k0a, torch::Tensor k1a, torch::Tensor rung_tok, torch::Tensor nc0,
    torch::Tensor cdf, torch::Tensor off2d,
    int64_t N, int64_t C, int64_t P, int64_t d, int64_t R)
{
    int n_pages = k0a.size(0) / (int)N;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(blobs.device());
    auto pos_out = torch::zeros({n_pages, P, d}, opts);
    int threads = ((int)N + 31) / 32 * 32;
    dim3 grid(n_pages), block(threads);
    decode_pages_rdo_kernel<<<grid, block>>>(
        blobs.data_ptr<uint8_t>(), page_byte_off.data_ptr<long>(), lane_off.data_ptr<long>(),
        k0a.data_ptr<int>(), k1a.data_ptr<int>(), rung_tok.data_ptr<int>(), nc0.data_ptr<int>(),
        reinterpret_cast<const unsigned int*>(cdf.data_ptr<int32_t>()), off2d.data_ptr<long>(),
        (int)N, (int)C, (int)P, (int)d, (int)R, n_pages, pos_out.data_ptr<int>());
    return pos_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_pages_rdo", &decode_pages_rdo, "rANS RDO per-token-rung decode");
}