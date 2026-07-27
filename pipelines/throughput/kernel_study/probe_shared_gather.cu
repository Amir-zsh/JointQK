// Is the vq2 codebook lookup limited by address-divergence processing rather
// than by L1 sector throughput?
//
// Evidence for the question: shrinking the codebook 16x (32 sectors -> 2) only
// bought 1.28%. Sector-bound would have bought ~16x. What IS invariant under
// table size is the coalescing unit's per-warp address-divergence work. Shared
// memory has no coalescing unit -- divergent access is resolved by bank
// arbitration instead -- so if that model is right, moving the SAME access
// pattern from global/L1 to shared should be substantially faster.
//
// Triton cannot express an explicit shared-memory gather (tl.gather lowers to
// warp shuffles, measured 1.7-4.3x slower), so this asks the hardware directly.
//
//   nvcc -O3 -arch=sm_90 logs/probe_shared_gather.cu -o /tmp/psg && /tmp/psg
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define NG   32          // groups per (token, head)  = head_dim / G
#define KCB  256         // codewords per group
#define ITER 256         // BLOCK_N-sized iterations per block

__global__ __launch_bounds__(256) void gather_global(
    const uint8_t* __restrict__ idx, const int* __restrict__ cb,
    int* __restrict__ out, int S)
{
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int* mycb = cb + (size_t)blk * NG * KCB;
    const uint8_t* myidx = idx + (size_t)blk * NG * (size_t)S;
    int acc = 0;
    for (int it = 0; it < ITER; ++it) {
        const int tok = it * blockDim.x + tid;
        #pragma unroll
        for (int g = 0; g < NG; ++g) {
            // Divergent: every lane a different index into the same 1 KiB
            // sub-table. This is exactly today's vq2 access pattern.
            int c = myidx[(size_t)g * S + tok];
            acc += mycb[g * KCB + c];
        }
    }
    out[blk * blockDim.x + tid] = acc;
}

__global__ __launch_bounds__(256) void gather_shared(
    const uint8_t* __restrict__ idx, const int* __restrict__ cb,
    int* __restrict__ out, int S)
{
    __shared__ int scb[NG * KCB];            // 32 KiB, staged once per block
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int* mycb = cb + (size_t)blk * NG * KCB;
    for (int i = tid; i < NG * KCB; i += blockDim.x) scb[i] = mycb[i];
    __syncthreads();

    const uint8_t* myidx = idx + (size_t)blk * NG * (size_t)S;
    int acc = 0;
    for (int it = 0; it < ITER; ++it) {
        const int tok = it * blockDim.x + tid;
        #pragma unroll
        for (int g = 0; g < NG; ++g) {
            int c = myidx[(size_t)g * S + tok];
            acc += scb[g * KCB + c];         // bank arbitration, no coalescing
        }
    }
    out[blk * blockDim.x + tid] = acc;
}

// Same as shared, but the sub-table is padded so that consecutive groups start
// on different banks -- reduces systematic conflicts if any exist.
__global__ __launch_bounds__(256) void gather_shared_swizzled(
    const uint8_t* __restrict__ idx, const int* __restrict__ cb,
    int* __restrict__ out, int S)
{
    __shared__ int scb[NG * (KCB + 1)];
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int* mycb = cb + (size_t)blk * NG * KCB;
    for (int i = tid; i < NG * KCB; i += blockDim.x)
        scb[(i / KCB) * (KCB + 1) + (i % KCB)] = mycb[i];
    __syncthreads();

    const uint8_t* myidx = idx + (size_t)blk * NG * (size_t)S;
    int acc = 0;
    for (int it = 0; it < ITER; ++it) {
        const int tok = it * blockDim.x + tid;
        #pragma unroll
        for (int g = 0; g < NG; ++g) {
            int c = myidx[(size_t)g * S + tok];
            acc += scb[g * (KCB + 1) + c];
        }
    }
    out[blk * blockDim.x + tid] = acc;
}


// The real arena is [slot, head, ng]: a token's NG indices are CONTIGUOUS, so
// all 32 can be fetched with one 32 B load instead of 32 byte-loads. Combined
// with shared-memory lookup and full unrolling for memory-level parallelism.
__global__ __launch_bounds__(256) void gather_shared_vecidx(
    const uint8_t* __restrict__ idx, const int* __restrict__ cb,
    int* __restrict__ out, int S)
{
    __shared__ int scb[NG * KCB];
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int* mycb = cb + (size_t)blk * NG * KCB;
    for (int i = tid; i < NG * KCB; i += blockDim.x) scb[i] = mycb[i];
    __syncthreads();

    // token-major indices: [S][NG]
    const uint8_t* myidx = idx + (size_t)blk * (size_t)S * NG;
    int acc = 0;
    for (int it = 0; it < ITER; ++it) {
        const int tok = it * blockDim.x + tid;
        const uint4* p = reinterpret_cast<const uint4*>(myidx + (size_t)tok * NG);
        uint4 a = p[0], b = p[1];            // 32 indices in two 16 B loads
        const uint8_t* c = reinterpret_cast<const uint8_t*>(&a);
        #pragma unroll
        for (int g = 0; g < 16; ++g) acc += scb[g * KCB + c[g]];
        const uint8_t* d = reinterpret_cast<const uint8_t*>(&b);
        #pragma unroll
        for (int g = 0; g < 16; ++g) acc += scb[(g + 16) * KCB + d[g]];
    }
    out[blk * blockDim.x + tid] = acc;
}

template <typename F>
static double bench(F f, int reps = 20) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    f(); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

int main() {
    const int GRID = 528, THREADS = 256;
    const int S = ITER * THREADS;
    size_t nidx = (size_t)GRID * NG * S;
    size_t ncb  = (size_t)GRID * NG * KCB;

    uint8_t *d_idx; int *d_cb, *d_out;
    cudaMalloc(&d_idx, nidx); cudaMalloc(&d_cb, ncb * 4);
    cudaMalloc(&d_out, (size_t)GRID * THREADS * 4);
    cudaMemset(d_idx, 0x5A, nidx);          // replaced below with random
    {   // random indices on host (pattern matters: must be divergent)
        uint8_t* h = (uint8_t*)malloc(nidx);
        for (size_t i = 0; i < nidx; ++i) h[i] = (uint8_t)(rand() & 0xFF);
        cudaMemcpy(d_idx, h, nidx, cudaMemcpyHostToDevice); free(h);
    }
    cudaMemset(d_cb, 1, ncb * 4);

    double lookups = (double)GRID * NG * S;
    printf("NG=%d K=%d GRID=%d threads=%d S=%d  (%.2f G lookups/launch)\n",
           NG, KCB, GRID, THREADS, S, lookups / 1e9);

    struct { const char* name; void (*k)(const uint8_t*, const int*, int*, int); } v[] = {
        {"global / L1 (today's pattern)", gather_global},
        {"shared memory",                 gather_shared},
        {"shared memory + pad swizzle",   gather_shared_swizzled},
        {"shared + vectorized idx load",  gather_shared_vecidx},
    };
    double base = 0;
    for (auto& e : v) {
        double ms = bench([&]{ e.k<<<GRID, THREADS>>>(d_idx, d_cb, d_out, S); });
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { printf("  %-32s FAILED: %s\n", e.name, cudaGetErrorString(err)); continue; }
        double rate = lookups / (ms * 1e-3) / 1e9;
        if (!base) base = rate;
        printf("  %-32s %8.1f us   %7.1f G lookups/s   %5.2fx\n",
               e.name, ms * 1e3, rate, rate / base);
    }
    return 0;
}
