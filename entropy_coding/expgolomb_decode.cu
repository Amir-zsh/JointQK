#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Lane-parallel Exp-Golomb-k decode: block = one page, threadIdx.x = lane (0..N-1),
// same page/lane layout convention as rans_decode.cu (one warp per page, N
// independent per-lane byte-streams) so the two kernels are directly comparable.
//
// Per lane: a 64-bit, top-aligned bit window (`win`, `nvalid` valid bits at the
// MSB end) refilled a byte at a time from the lane's private stream. Exp-Golomb-k
// decode of one symbol is three bit-tricks (no per-bit loop): __clzll for the
// unary prefix length Q, a shift+mask for the (Q+1)-bit field m, another for the
// k literal low bits. Divergence is confined to the refill's data-dependent byte
// count, not the unary scan itself.
__device__ __forceinline__ void eg_refill(const uint8_t* buf, long buf_len,
                                          uint64_t& win, int& nvalid, long& bytepos) {
    while (nvalid <= 56 && bytepos < buf_len) {
        win |= ((uint64_t)buf[bytepos]) << (56 - nvalid);
        nvalid += 8;
        ++bytepos;
    }
}

__global__ void eg_decode_pages_kernel(
    const uint8_t* __restrict__ blobs,
    const long*    __restrict__ page_byte_off,
    const long*    __restrict__ lane_off,
    const long*    __restrict__ lane_len,
    const int*     __restrict__ ks,        // per-(page,coord) k: ks[p*d + j], heterogeneous heads OK
    int N, int P, int d, int n_pages,
    int* __restrict__ pos_out)
{
    int p = blockIdx.x;
    int lane = threadIdx.x;
    if (p >= n_pages || lane >= N) return;

    const int* page_ks = ks + (long)p * d;
    int total_syms = P * d;
    int syms_per_lane = total_syms / N;

    const uint8_t* buf = blobs + page_byte_off[p] + lane_off[(long)p * N + lane];
    long buf_len = lane_len[(long)p * N + lane];

    uint64_t win = 0;
    int nvalid = 0;
    long bytepos = 0;
    long page_pos_base = (long)p * P * d;

    for (int i = 0; i < syms_per_lane; ++i) {
        int flat_pos = i * N + lane;          // MUST match the encoder's lane interleaving
        int t = flat_pos / d, j = flat_pos % d;
        int k = page_ks[j];

        eg_refill(buf, buf_len, win, nvalid, bytepos);

        int Q = __clzll(win);                 // leading zero-bit run before the terminator '1'
        win <<= Q; nvalid -= Q;
        int mbits = Q + 1;
        unsigned int m = (unsigned int)(win >> (64 - mbits));
        win <<= mbits; nvalid -= mbits;

        eg_refill(buf, buf_len, win, nvalid, bytepos);
        unsigned int low = k ? (unsigned int)(win >> (64 - k)) : 0u;
        if (k) { win <<= k; nvalid -= k; }

        unsigned int u = ((m - 1u) << k) | low;
        int s = (u & 1u) ? -(int)((u + 1u) >> 1) : (int)(u >> 1);

        pos_out[page_pos_base + (long)t * d + j] = s;
    }
}

torch::Tensor decode_pages(
    torch::Tensor blobs, torch::Tensor page_byte_off, torch::Tensor lane_off,
    torch::Tensor lane_len, torch::Tensor ks, int64_t N, int64_t P, int64_t d)
{
    int n_pages = page_byte_off.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(blobs.device());
    auto pos_out = torch::zeros({n_pages, (int)P, (int)d}, opts);

    int threads = ((int)N + 31) / 32 * 32;
    dim3 grid(n_pages), block(threads);

    eg_decode_pages_kernel<<<grid, block>>>(
        blobs.data_ptr<uint8_t>(), page_byte_off.data_ptr<long>(), lane_off.data_ptr<long>(),
        lane_len.data_ptr<long>(), ks.data_ptr<int>(), (int)N, (int)P, (int)d, n_pages,
        pos_out.data_ptr<int>());
    return pos_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_pages", &decode_pages, "Exp-Golomb-k decode (warp per page, lane-interleaved, per-page k)");
}
