// decode_bench.cu — isolate the decode. Two decoders, identical except how the
// symbol that owns `slot` is found:
//   binsearch : your current divergent `while (a < b)` log2(A) branch-walk
//   lut       : one branchless read of a per-(rung,dim) table
// Both write s_pos to global (no attention, no SRAM staging) so we measure raw
// decode throughput and verify the LUT path is bit-identical to binsearch.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SCALE_BITS 14
#define TOTAL (1u << SCALE_BITS)
#define MASK (TOTAL - 1u)
#define RANS_L (1u << 23)

// lut[(ri*d + j)*TOTAL + slot] = symbol s whose cdf range contains slot
__global__ void build_lut_kernel(
    const unsigned int* __restrict__ cdf_flat,
    const int* __restrict__ cdf_flat_base,
    const int* __restrict__ cdf_off_all,     // (R, d+1)
    unsigned char* __restrict__ lut,         // (R, d, TOTAL)
    int R, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // over R*d
    if (idx >= R * d) return;
    int ri = idx / d, j = idx % d;
    int cflat_base = cdf_flat_base[ri];
    const int* coff = cdf_off_all + ri * (d + 1);
    int lo = cflat_base + coff[j];
    int A  = coff[j + 1] - coff[j] - 1;       // number of symbols
    long base = (long)(ri * d + j) * TOTAL;
    for (int s = 0; s < A; ++s) {
        unsigned int a0 = cdf_flat[lo + s];
        unsigned int a1 = cdf_flat[lo + s + 1];
        for (unsigned int v = a0; v < a1; ++v) lut[base + v] = (unsigned char)s;
    }
}

__global__ void decode_binsearch_kernel(
    const uint8_t* __restrict__ blobs, const long* __restrict__ page_byte_off,
    const long* __restrict__ lane_off, const int* __restrict__ k0a, const int* __restrict__ k1a,
    const int* __restrict__ rung_of_page, const int* __restrict__ nonconst_cat,
    const int* __restrict__ nonconst_base, const int* __restrict__ C_all,
    const unsigned int* __restrict__ cdf_flat, const int* __restrict__ cdf_flat_base,
    const int* __restrict__ cdf_off_all,
    int* __restrict__ s_pos_out,            // (nb, P, d)
    int N, int P, int d)
{
    int pg = blockIdx.x, tid = threadIdx.x;
    if (tid >= N) return;
    int ri = rung_of_page[pg];
    int C = C_all[ri];
    const int* nonconst = nonconst_cat + nonconst_base[ri];
    int cflat_base = cdf_flat_base[ri];
    const int* coff = cdf_off_all + ri * (d + 1);
    int kk0 = k0a[pg * N + tid], kk1 = k1a[pg * N + tid];
    int* spg = s_pos_out + (long)pg * P * d;
    if (kk0 < kk1) {
        const uint8_t* buf = blobs + page_byte_off[pg] + lane_off[pg * N + tid];
        unsigned int x = (unsigned int)buf[0] | ((unsigned int)buf[1] << 8) |
                         ((unsigned int)buf[2] << 16) | ((unsigned int)buf[3] << 24);
        int bp = 4;
        for (int k = kk0; k < kk1; ++k) {
            int t = k / C, j = nonconst[k % C];
            unsigned int slot = x & MASK;
            int lo = cflat_base + coff[j];
            int hi = cflat_base + coff[j + 1] - 1;
            int a = lo, b = hi - 1;
            while (a < b) { int mid = (a + b + 1) >> 1; if (cdf_flat[mid] <= slot) a = mid; else b = mid - 1; }
            int s = a - lo;
            unsigned int start = cdf_flat[lo + s];
            unsigned int freq = cdf_flat[lo + s + 1] - start;
            x = freq * (x >> SCALE_BITS) + slot - start;
            while (x < RANS_L) { x = (x << 8) | (unsigned int)buf[bp]; ++bp; }
            spg[t * d + j] = s;
        }
    }
}

__global__ void decode_lut_kernel(
    const uint8_t* __restrict__ blobs, const long* __restrict__ page_byte_off,
    const long* __restrict__ lane_off, const int* __restrict__ k0a, const int* __restrict__ k1a,
    const int* __restrict__ rung_of_page, const int* __restrict__ nonconst_cat,
    const int* __restrict__ nonconst_base, const int* __restrict__ C_all,
    const unsigned int* __restrict__ cdf_flat, const int* __restrict__ cdf_flat_base,
    const int* __restrict__ cdf_off_all,
    const unsigned char* __restrict__ lut,   // (R, d, TOTAL)
    int* __restrict__ s_pos_out,
    int N, int P, int d)
{
    int pg = blockIdx.x, tid = threadIdx.x;
    if (tid >= N) return;
    int ri = rung_of_page[pg];
    int C = C_all[ri];
    const int* nonconst = nonconst_cat + nonconst_base[ri];
    int cflat_base = cdf_flat_base[ri];
    const int* coff = cdf_off_all + ri * (d + 1);
    int kk0 = k0a[pg * N + tid], kk1 = k1a[pg * N + tid];
    int* spg = s_pos_out + (long)pg * P * d;
    if (kk0 < kk1) {
        const uint8_t* buf = blobs + page_byte_off[pg] + lane_off[pg * N + tid];
        unsigned int x = (unsigned int)buf[0] | ((unsigned int)buf[1] << 8) |
                         ((unsigned int)buf[2] << 16) | ((unsigned int)buf[3] << 24);
        int bp = 4;
        for (int k = kk0; k < kk1; ++k) {
            int t = k / C, j = nonconst[k % C];
            unsigned int slot = x & MASK;
            int s = (int)lut[(long)(ri * d + j) * TOTAL + slot];   // branchless table read
            int lo = cflat_base + coff[j];
            unsigned int start = cdf_flat[lo + s];
            unsigned int freq = cdf_flat[lo + s + 1] - start;
            x = freq * (x >> SCALE_BITS) + slot - start;
            while (x < RANS_L) { x = (x << 8) | (unsigned int)buf[bp]; ++bp; }
            spg[t * d + j] = s;
        }
    }
}

torch::Tensor build_lut(torch::Tensor cdf_flat, torch::Tensor cdf_flat_base,
                        torch::Tensor cdf_off_all, int64_t R, int64_t d) {
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(cdf_flat.device());
    auto lut = torch::zeros({R, d, (long)TOTAL}, opts);
    int total = R * d, thr = 256, blk = (total + thr - 1) / thr;
    build_lut_kernel<<<blk, thr>>>(
        reinterpret_cast<const unsigned int*>(cdf_flat.data_ptr<int32_t>()),
        cdf_flat_base.data_ptr<int>(), cdf_off_all.data_ptr<int>(),
        lut.data_ptr<unsigned char>(), (int)R, (int)d);
    return lut;
}

torch::Tensor decode_run(
    torch::Tensor blobs, torch::Tensor page_byte_off, torch::Tensor lane_off,
    torch::Tensor k0a, torch::Tensor k1a, torch::Tensor rung_of_page,
    torch::Tensor nonconst_cat, torch::Tensor nonconst_base, torch::Tensor C_all,
    torch::Tensor cdf_flat, torch::Tensor cdf_flat_base, torch::Tensor cdf_off_all,
    torch::Tensor lut, int64_t N, int64_t P, int64_t d, int64_t nb, int64_t use_lut)
{
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(blobs.device());
    auto s_pos = torch::zeros({nb, P, d}, opts);
    dim3 grid(nb); int threads = 32;
    if (use_lut) {
        decode_lut_kernel<<<grid, threads>>>(
            blobs.data_ptr<uint8_t>(), page_byte_off.data_ptr<long>(), lane_off.data_ptr<long>(),
            k0a.data_ptr<int>(), k1a.data_ptr<int>(), rung_of_page.data_ptr<int>(),
            nonconst_cat.data_ptr<int>(), nonconst_base.data_ptr<int>(), C_all.data_ptr<int>(),
            reinterpret_cast<const unsigned int*>(cdf_flat.data_ptr<int32_t>()),
            cdf_flat_base.data_ptr<int>(), cdf_off_all.data_ptr<int>(),
            lut.data_ptr<unsigned char>(), s_pos.data_ptr<int>(), (int)N, (int)P, (int)d);
    } else {
        decode_binsearch_kernel<<<grid, threads>>>(
            blobs.data_ptr<uint8_t>(), page_byte_off.data_ptr<long>(), lane_off.data_ptr<long>(),
            k0a.data_ptr<int>(), k1a.data_ptr<int>(), rung_of_page.data_ptr<int>(),
            nonconst_cat.data_ptr<int>(), nonconst_base.data_ptr<int>(), C_all.data_ptr<int>(),
            reinterpret_cast<const unsigned int*>(cdf_flat.data_ptr<int32_t>()),
            cdf_flat_base.data_ptr<int>(), cdf_off_all.data_ptr<int>(),
            s_pos.data_ptr<int>(), (int)N, (int)P, (int)d);
    }
    return s_pos;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_lut", &build_lut, "build per-(rung,dim) rANS decode LUT");
    m.def("decode_run", &decode_run, "decode all pages (use_lut=0 binsearch, 1 table)");
}
