// Standalone quantize_q8_1 benchmark — llama.cpp activation quant before GEMV.
//
// Faithful to ggml/src/ggml-cuda/quantize.cu quantize_q8_1:
//   grid (ceil(ne0/256), ne1, ne2*ne3), block 256.
// FUNCSIM_SAFE: f32 scale + sum per block (no half2); same grid and block_q8_1 layout.
//
// Build:  make | make exec
// Run:    ./quantize_bench
//         ./quantize_bench_exec 5120 1          # ne0=hidden, ne1=tokens(=batch)
//         BATCH=32 ./quantize_bench_exec 5120 1 # batch multiplies ne1 (the token/row dim)
// batch = number of tokens quantized this step (decode: one per sequence) -> scales ne1.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../common/bench_common.h"
#include "../common/shapes_qwen3_14b_decode.h"

#define QK8_1 32
#define QUANT_BLOCK 256

#ifdef FUNCSIM_SAFE
typedef struct {
    float  d;
    float  s;
    int8_t qs[QK8_1];
} block_q8_1; // 40 B — slightly larger than ggml's 36 B
#else
typedef struct {
    half2  ds;
    int8_t qs[QK8_1];
} block_q8_1;
#endif

#ifdef FUNCSIM_SAFE
__device__ float warp_sum(float x) {
    __shared__ float sm[QUANT_BLOCK];
    sm[threadIdx.x] = x;
    __syncthreads();
    const int lane = threadIdx.x % 32;
    const int base = threadIdx.x - lane;
    if (lane < 16) sm[base + lane] += sm[base + lane + 16];
    __syncthreads();
    if (lane < 8) sm[base + lane] += sm[base + lane + 8];
    __syncthreads();
    if (lane < 4) sm[base + lane] += sm[base + lane + 4];
    __syncthreads();
    if (lane < 2) sm[base + lane] += sm[base + lane + 2];
    __syncthreads();
    if (lane < 1) sm[base + lane] += sm[base + lane + 1];
    __syncthreads();
    return sm[base];
}

__device__ float warp_max(float x) {
    __shared__ float sm[QUANT_BLOCK];
    sm[threadIdx.x] = x;
    __syncthreads();
    const int lane = threadIdx.x % 32;
    const int base = threadIdx.x - lane;
    if (lane < 16) sm[base + lane] = fmaxf(sm[base + lane], sm[base + lane + 16]);
    __syncthreads();
    if (lane < 8) sm[base + lane] = fmaxf(sm[base + lane], sm[base + lane + 8]);
    __syncthreads();
    if (lane < 4) sm[base + lane] = fmaxf(sm[base + lane], sm[base + lane + 4]);
    __syncthreads();
    if (lane < 2) sm[base + lane] = fmaxf(sm[base + lane], sm[base + lane + 2]);
    __syncthreads();
    if (lane < 1) sm[base + lane] = fmaxf(sm[base + lane], sm[base + lane + 1]);
    __syncthreads();
    return sm[base];
}
#else
__device__ float warp_sum(float x) {
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, off);
    }
    return x;
}

__device__ float warp_max(float x) {
    for (int off = 16; off > 0; off >>= 1) {
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, off));
    }
    return x;
}
#endif

// Simplified quantize_q8_1 — contiguous f32 input, no fastdiv strides.
__global__ void quantize_q8_1(const float *__restrict__ x, block_q8_1 *__restrict__ y, int ne0, int ne1) {
    const int i0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }
    const int i1 = blockIdx.y;
    const int64_t i_cont = (int64_t) i1 * ne0 + i0;
    const int64_t ib     = i_cont / QK8_1;
    const int64_t iqs    = i_cont % QK8_1;

    const float xi   = x[i_cont];
    float       amax = fabsf(xi);
    float       sum  = xi;

    amax = warp_max(amax);
    sum  = warp_sum(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) lrintf(xi / d);
    y[ib].qs[iqs]  = q;

    if (iqs != 0) {
        return;
    }
#ifdef FUNCSIM_SAFE
    y[ib].d = d;
    y[ib].s = sum;
#else
    y[ib].ds = make_half2(__float2half(d), __float2half(sum));
#endif
}

static int run_shape(int ne0, int ne1, int ne2, int ne3, const char *name) {
    const int64_t n_elem  = (int64_t) ne0 * ne1 * ne2 * ne3;
    const int64_t n_blocks = (n_elem + QK8_1 - 1) / QK8_1;

    float *d_x = nullptr;
    block_q8_1 *d_y = nullptr;
    float *h_x = new float[n_elem];
    fill_f32(h_x, (size_t) n_elem, 0.25f);

    BENCH_CHECK(cudaMalloc(&d_x, (size_t) n_elem * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_y, (size_t) n_blocks * sizeof(block_q8_1)));
    BENCH_CHECK(cudaMemcpy(d_x, h_x, (size_t) n_elem * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 grid((ne0 + QUANT_BLOCK - 1) / QUANT_BLOCK, ne1, ne2 * ne3);
    const dim3 block(QUANT_BLOCK, 1, 1);

    const double bytes = (double) n_elem * sizeof(float) + (double) n_blocks * sizeof(block_q8_1);
    printf(">>> quantize_q8_1 %s  ne=(%d,%d,%d,%d) grid=(%u,%u,%u)  traffic=%.3f MB\n",
           name, ne0, ne1, ne2, ne3, grid.x, grid.y, grid.z, bytes / 1e6);

    quantize_q8_1<<<grid, block>>>(d_x, d_y, ne0, ne1);
    BENCH_CHECK(cudaGetLastError());
    BENCH_CHECK(cudaDeviceSynchronize());

    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}

int main(int argc, char **argv) {
    cudaDeviceProp prop{};
    BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (quantize_q8_1 decode benchmark)\n", prop.name);

    // batch = parallel decode tokens; multiplies ne1 (the row/token dimension).
    int batch = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
    if (batch < 1) batch = 1;

    if (argc >= 3) {
        const int ne0 = atoi(argv[1]);
        const int ne1 = atoi(argv[2]) * batch;
        const int ne2 = (argc > 3) ? atoi(argv[3]) : 1;
        const int ne3 = (argc > 4) ? atoi(argv[4]) : 1;
        return run_shape(ne0, ne1, ne2, ne3, "custom");
    }

    for (int i = 0; i < qwen3_14b_quant_num_shapes; ++i) {
        const quant_shape &s = qwen3_14b_quant_shapes[i];
        if (run_shape(s.ne0, s.ne1 * batch, s.ne2, s.ne3, s.name)) {
            return 1;
        }
    }
    return 0;
}
