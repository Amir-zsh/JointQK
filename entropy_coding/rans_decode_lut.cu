#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Same rANS decode algorithm as rans_decode.cu (SCALE_BITS=14, byte renorm),
// but slot->symbol resolution is a single global-memory LUT read instead of a
// binary search over the CDF staged in shared memory. Tests whether trading
// shared-memory binary search for a per-coordinate 2^14-entry table (global
// memory, one struct read: packed (start,freq) + symbol id) is a net win --
// rans_interleaved.py's own spec comment already flagged "the 2^14 LUT is too
// big at d=128 distinct tables" as a reason binary search was chosen; this
// measures the actual tradeoff instead of assuming it.
#define SCALE_BITS 14
#define TOTAL (1u << SCALE_BITS)
#define MASK (TOTAL - 1u)
#define RANS_L (1u << 23)

__global__ void decode_pages_lut_kernel(
    const uint8_t* __restrict__ blobs,
    const long*    __restrict__ page_byte_off,
    const long*    __restrict__ lane_off,
    const int*     __restrict__ k0a,
    const int*     __restrict__ k1a,
    const int*     __restrict__ nonconst,
    const unsigned int* __restrict__ lut_meta,   // [j*TOTAL + slot] = (start<<14)|freq
    const short*   __restrict__ lut_sym,         // [j*TOTAL + slot] = symbol index
    int N, int C, int P, int d, int n_pages,
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

    for (int k = k0; k < k1; ++k) {
        int t = k / C;
        int j = nonconst[k % C];
        unsigned int slot = x & MASK;
        long li = (long)j * TOTAL + slot;
        unsigned int meta = lut_meta[li];
        unsigned int start = meta >> SCALE_BITS;
        unsigned int freq = meta & MASK;
        int s = (int)lut_sym[li];
        x = freq * (x >> SCALE_BITS) + slot - start;
        while (x < RANS_L) { x = (x << 8) | (unsigned int)buf[bp]; ++bp; }
        pos_out[page_pos_base + (long)t * d + j] = s;
    }
}

torch::Tensor decode_pages_lut(
    torch::Tensor blobs, torch::Tensor page_byte_off, torch::Tensor lane_off,
    torch::Tensor k0a, torch::Tensor k1a, torch::Tensor nonconst,
    torch::Tensor lut_meta, torch::Tensor lut_sym, int64_t N, int64_t C, int64_t P, int64_t d)
{
    int n_pages = k0a.size(0) / (int)N;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(blobs.device());
    auto pos_out = torch::zeros({n_pages, (int)P, (int)d}, opts);

    int threads = ((int)N + 31) / 32 * 32;
    dim3 grid(n_pages), block(threads);

    decode_pages_lut_kernel<<<grid, block>>>(
        blobs.data_ptr<uint8_t>(), page_byte_off.data_ptr<long>(), lane_off.data_ptr<long>(),
        k0a.data_ptr<int>(), k1a.data_ptr<int>(), nonconst.data_ptr<int>(),
        reinterpret_cast<const unsigned int*>(lut_meta.data_ptr<int32_t>()),
        lut_sym.data_ptr<int16_t>(), (int)N, (int)C, (int)P, (int)d, n_pages,
        pos_out.data_ptr<int>());
    return pos_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_pages_lut", &decode_pages_lut, "rANS decode (warp per page, direct slot LUT)");
}
