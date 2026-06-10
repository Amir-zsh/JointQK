#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SCALE_BITS 14
#define RANS_L (1u << 23)

// Reverse-rANS encode: block = one page, threadIdx.x = lane.
// Each lane encodes its symbol slice [k0,k1) in REVERSE, mirroring
// _encode_lane_nb byte-for-byte. Output per lane: [4-byte LE state][renorm bytes
// in decode/forward order]. Lengths returned separately; bytes packed at a
// per-lane max stride (host compacts using the returned lengths).
__global__ void encode_pages_kernel(
    const long* __restrict__ freq,       // (total_syms,) per-symbol freq, page-major lane order
    const long* __restrict__ start,      // (total_syms,) per-symbol start (cdf)
    const int*  __restrict__ k0a,        // (n_pages*N,) lane symbol-range start (into this page's syms)
    const int*  __restrict__ k1a,        // (n_pages*N,)
    const long* __restrict__ page_sym_off, // (n_pages,) offset of page's first symbol into freq/start
    int N, int n_pages, int max_lane_bytes,
    uint8_t* __restrict__ out_bytes,     // (n_pages*N*max_lane_bytes,)
    int* __restrict__ out_len)           // (n_pages*N,) bytes written per lane
{
    int p = blockIdx.x;
    int lane = threadIdx.x;
    if (p >= n_pages || lane >= N) return;

    int k0 = k0a[p * N + lane];
    int k1 = k1a[p * N + lane];
    long soff = page_sym_off[p];
    // this lane's output region
    uint8_t* obuf = out_bytes + ((long)(p * N + lane)) * max_lane_bytes;

    if (k0 >= k1) { out_len[p * N + lane] = 0; return; }

    // renorm scratch: write downward from the top of this lane's region.
    // We reserve the first 4 bytes for state, renorm bytes grow down from the end.
    unsigned int x = RANS_L;
    int w = max_lane_bytes;                 // descending write cursor for renorm bytes
    for (int k = k1 - 1; k >= k0; --k) {
        unsigned int f = (unsigned int)freq[soff + k];
        unsigned int s = (unsigned int)start[soff + k];
        unsigned int x_max = ((RANS_L >> SCALE_BITS) << 8) * f;
        while (x >= x_max) {
            --w;
            obuf[w] = (uint8_t)(x & 0xFF);
            x >>= 8;
        }
        x = ((x / f) << SCALE_BITS) + (x % f) + s;
    }
    // final layout: [state:4 LE][renorm bytes from w..max_lane_bytes]
    int nren = max_lane_bytes - w;
    int total = 4 + nren;
    obuf[0] = (uint8_t)(x & 0xFF);
    obuf[1] = (uint8_t)((x >> 8) & 0xFF);
    obuf[2] = (uint8_t)((x >> 16) & 0xFF);
    obuf[3] = (uint8_t)((x >> 24) & 0xFF);
    // move renorm bytes from [w, max) down to [4, 4+nren)
    for (int i = 0; i < nren; ++i)
        obuf[4 + i] = obuf[w + i];
    out_len[p * N + lane] = total;
}

torch::Tensor encode_pages(
    torch::Tensor freq, torch::Tensor start,
    torch::Tensor k0a, torch::Tensor k1a, torch::Tensor page_sym_off,
    int64_t N, int64_t n_pages, int64_t max_lane_bytes,
    torch::Tensor out_bytes, torch::Tensor out_len)
{
    int threads = ((int)N + 31) / 32 * 32;
    dim3 grid(n_pages), block(threads);
    encode_pages_kernel<<<grid, block>>>(
        freq.data_ptr<long>(), start.data_ptr<long>(),
        k0a.data_ptr<int>(), k1a.data_ptr<int>(), page_sym_off.data_ptr<long>(),
        (int)N, (int)n_pages, (int)max_lane_bytes,
        out_bytes.data_ptr<uint8_t>(), out_len.data_ptr<int>());
    return out_len;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_pages", &encode_pages, "reverse rANS encode (lane per thread)");
}