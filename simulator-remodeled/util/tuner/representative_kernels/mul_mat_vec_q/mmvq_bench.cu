// Standalone mul_mat_vec_q (Q8_0 weight x q8_1 activation) GEMV benchmark.
//
// This is a minimal, self-contained reimplementation of llama.cpp's decode workhorse
// (ggml/src/ggml-cuda/mmvq.cu, the ncols_dst=1 path). It is faithful in the parts that
// matter for an architecture/memory study:
//   - exact block layout (block_q8_0 = 34 B, block_q8_1 = 36 B) -> exact DRAM byte counts
//   - int8 dot via __dp4a (same compute primitive as ggml's vec_dot_q8_0_q8_1)
//   - one weight read per element (memory-bandwidth-bound GEMV)
// It is intentionally simpler than ggml's tiled/fused kernel (no SwiGLU fusion, no MMQ,
// one output row per CTA) -- enough to reproduce the dominant decode cost and to drive
// the simulator. See ../DESIGN.md.
//
// Build:  make            (sm_90)   |   make ARCH='-gencode=arch=compute_80,code=sm_80'
// Run:    ./mmvq_bench            (sweeps the Qwen3-14B shapes; prints time + GB/s)

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "shapes_qwen3_14b.h"

#define QK8_0 32
#define QK8_1 32
#define WARP_SIZE 32
#define NWARPS 4            // 128 threads/CTA split the K dimension

// FUNCSIM_SAFE: build a variant with NO __dp4a and NO fp16, for GPGPU-Sim's
// execution-driven (PTX functional) mode, whose parser/executor chokes on dp4a and the
// fp16 pack/cvt ops. Same shapes and access pattern; scales are f32 and the int8 dot is a
// scalar loop. Block sizes differ slightly (36/40 B vs ggml's 34/36 B), so DRAM byte
// counts are ~6% higher -- fine for functional/relative exec-driven studies. For
// bit-exact ggml traffic use the default (dp4a/fp16) build + trace-driven mode.
#ifdef FUNCSIM_SAFE
typedef struct { float d;          int8_t qs[QK8_0]; } block_q8_0;   // 36 bytes
typedef struct { float d; float s; int8_t qs[QK8_1]; } block_q8_1;   // 40 bytes
#else
typedef struct { half  d;  int8_t qs[QK8_0]; } block_q8_0;   // 34 bytes
typedef struct { half2 ds; int8_t qs[QK8_1]; } block_q8_1;   // 36 bytes (ds = {d, sum})
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 B");
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 B");
#endif

#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

// read 4 consecutive int8 as one 32-bit word, exactly like ggml's get_int_from_int8
// (`*((const int *) p)`). This emits a single LDG.E.32 per 4 quants -- matching ggml's
// load count/traffic. q8_0's qs is 2-byte-aligned, so this is an unaligned 32-bit global
// load, which NVIDIA GPUs service correctly (ggml relies on the same behavior). Using
// memcpy here would instead emit 4 byte-loads and mis-model the access pattern.
static __device__ __forceinline__ int load_int(const int8_t *p) {
    return *reinterpret_cast<const int *>(p);
}

// one q8_0 block (32 quants) . one q8_1 block (32 quants); q8_0 is symmetric so only the
// q8_1 scale (ds.x) is used, matching ggml's vec_dot_q8_0_q8_1.
static __device__ __forceinline__ float vec_dot_q8_0_q8_1(const block_q8_0 *x, const block_q8_1 *y) {
#ifdef FUNCSIM_SAFE
    // float accumulate so nvcc can't fold the int8 MAC back into dp4a (which the
    // execution-driven PTX functional sim can't execute).
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) sumf += (float) x->qs[j] * (float) y->qs[j];
    return x->d * y->d * sumf;
#else
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < QK8_0 / 4; ++i) {
        sumi = __dp4a(load_int(x->qs + 4 * i), load_int(y->qs + 4 * i), sumi);
    }
    const float2 yds = __half22float2(y->ds);
    return __half2float(x->d) * yds.x * (float) sumi;
#endif
}

// GEMV: dst[N] = W[N x K] (q8_0) * y[K] (q8_1). One CTA per output row.
__global__ void mmvq_q8_0(const block_q8_0 *__restrict__ vx,
                          const block_q8_1 *__restrict__ vy,
                          float *__restrict__ dst, int K, int N) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int nthreads = NWARPS * WARP_SIZE;
    const int bpr = K / QK8_0;                       // weight blocks per row
    const block_q8_0 *xrow = vx + (size_t) row * bpr;

    float acc = 0.0f;
    for (int kb = tid; kb < bpr; kb += nthreads) {
        acc += vec_dot_q8_0_q8_1(&xrow[kb], &vy[kb]);
    }
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    __shared__ float warp_sums[NWARPS];
    if (threadIdx.x == 0) warp_sums[threadIdx.y] = acc;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) s += warp_sums[w];
        dst[row] = s;
    }
}

// deterministic fill: every quant = 1, every scale d = 1.0 -> dst[row] == K exactly.
__global__ void fill_q8_0(block_q8_0 *b, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#ifdef FUNCSIM_SAFE
    b[i].d = 1.0f;
#else
    b[i].d = __float2half(1.0f);
#endif
    for (int j = 0; j < QK8_0; ++j) b[i].qs[j] = 1;
}
__global__ void fill_q8_1(block_q8_1 *b, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#ifdef FUNCSIM_SAFE
    b[i].d = 1.0f; b[i].s = 32.0f;
#else
    b[i].ds = __floats2half2_rn(1.0f, 32.0f);  // d=1, sum=32 (unused by q8_0 dot)
#endif
    for (int j = 0; j < QK8_1; ++j) b[i].qs[j] = 1;
}

static int run_shape(int K, int N, int fusion, const char *name, bool verify) {
    const int bpr = K / QK8_0;
    const size_t n_wblocks = (size_t) N * bpr;
    const size_t n_yblocks = bpr;
    const size_t w_bytes = n_wblocks * sizeof(block_q8_0);

    block_q8_0 *d_w; block_q8_1 *d_y; float *d_dst;
    CHECK(cudaMalloc(&d_w, n_wblocks * sizeof(block_q8_0)));
    CHECK(cudaMalloc(&d_y, n_yblocks * sizeof(block_q8_1)));
    CHECK(cudaMalloc(&d_dst, (size_t) N * sizeof(float)));

    fill_q8_0<<<(n_wblocks + 255) / 256, 256>>>(d_w, n_wblocks);
    fill_q8_1<<<(n_yblocks + 255) / 256, 256>>>(d_y, n_yblocks);
    CHECK(cudaGetLastError());

    dim3 block(WARP_SIZE, NWARPS);
    dim3 grid(N);

    // warmup + correctness
    mmvq_q8_0<<<grid, block>>>(d_w, d_y, d_dst, K, N);
    CHECK(cudaDeviceSynchronize());
    if (verify) {
        float h0 = 0.0f;
        CHECK(cudaMemcpy(&h0, d_dst, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  [verify] dst[0]=%.1f expected=%d  %s\n", h0, K,
               (h0 == (float) K) ? "OK" : "MISMATCH");
    }

    const int iters = 20;
    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int it = 0; it < iters; ++it) mmvq_q8_0<<<grid, block>>>(d_w, d_y, d_dst, K, N);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double us = ms * 1e3 / iters;
    double gbs = w_bytes / (us * 1e-6) / 1e9;
    printf("  %-8s K=%-6d N=%-6d fusion=%d  W=%6.1f MB  %8.2f us  %7.1f GB/s\n",
           name, K, N, fusion, w_bytes / 1e6, us, gbs);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_w); cudaFree(d_y); cudaFree(d_dst);
    return 0;
}

int main() {
    int dev = 0; cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Device: %s  (mmvq Q8_0 GEMV, ncols_dst=1)\n", p.name);

    // tiny correctness check first
    printf("[correctness]\n");
    if (run_shape(256, 64, 0, "tiny", true)) return 1;

    // MMVQ_TINY=1: stop after the tiny shape. The functional (execution-driven)
    // simulator runs every thread, so the big shapes (lm_head = ~19M threads) are
    // impractical there; use this to validate exec-driven mode quickly.
    if (getenv("MMVQ_TINY")) { printf("[MMVQ_TINY set -> skipping full sweep]\n"); return 0; }

    printf("[Qwen3-14B decode shapes]\n");
    for (int i = 0; i < qwen3_14b_num_shapes; ++i) {
        const mmvq_shape &s = qwen3_14b_shapes[i];
        if (run_shape(s.K, s.N, s.fusion, s.name, false)) return 1;
    }
    return 0;
}
