// Standalone rope_neox benchmark — llama.cpp decode RoPE.
//
// Faithful to ggml/src/ggml-cuda/rope.cu rope_neox (NEOX pair rotation):
//   one block per (head-row); thread i rotates the pair (i, i+rot/2) within a
//   head of width rot = head_dim. Q path keeps f32 output; K path stores f16
//   (out_f16=1) feeding set_rows into the f16 KV cache.
// Functional-sim-safe: scalar f32 math, hand-rolled f32->f16 bit pack.
//
// Build:  make            (sm_90)
//         make exec       (sm_70, GPGPU-Sim execution-driven)
// Run:    ./rope_bench
//         ./rope_bench <n_rows> <rot> <out_f16>     # custom one-shot
//         BATCH=32 ./rope_bench                      # batch multiplies n_rows (tokens)
// batch = parallel decode tokens; each token contributes one set of head-rows.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "../common/bench_common.h"

static __device__ __forceinline__ uint16_t f32_to_f16_bits(float f) {
    uint32_t u; memcpy(&u, &f, 4); uint16_t s = (u >> 16) & 0x8000;
    int e = ((u >> 23) & 0xff) - 112; if (e <= 0) return s; if (e >= 31) return s | 0x7c00;
    return s | (uint16_t)((e << 10) | ((u >> 13) & 0x3ff));
}

// NEOX rope: rotate pairs (i, i+rot/2) of each head row. out_f16=1 -> store f16 bits.
__global__ void rope_neox(const float *__restrict__ src, void *__restrict__ dst,
                          int n_rows, int rot, int pos, float theta_base, int out_f16) {
    const int row = blockIdx.x; if (row >= n_rows) return;
    const int i = threadIdx.x;  if (i >= rot / 2) return;
    const float *sr = src + (size_t) row * rot;
    const float theta = (float) pos * powf(theta_base, -2.0f * (float) i / (float) rot);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = sr[i], x1 = sr[i + rot / 2];
    const float r0 = x0 * c - x1 * s, r1 = x0 * s + x1 * c;
    if (out_f16) {
        uint16_t *d = (uint16_t *) dst + (size_t) row * rot;
        d[i] = f32_to_f16_bits(r0); d[i + rot / 2] = f32_to_f16_bits(r1);
    } else {
        float *d = (float *) dst + (size_t) row * rot; d[i] = r0; d[i + rot / 2] = r1;
    }
}

static int run_shape(int n_rows, int rot, int out_f16, const char *name) {
    const size_t n_elem = (size_t) n_rows * rot;
    float *d_src = nullptr; void *d_dst = nullptr;
    float *h_src = new float[n_elem];
    fill_f32(h_src, n_elem, 0.5f);

    BENCH_CHECK(cudaMalloc(&d_src, n_elem * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_dst, n_elem * (out_f16 ? sizeof(uint16_t) : sizeof(float))));
    BENCH_CHECK(cudaMemcpy(d_src, h_src, n_elem * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 grid(n_rows), block(rot / 2);
    const double bytes = (double) n_elem * sizeof(float) +
                         (double) n_elem * (out_f16 ? sizeof(uint16_t) : sizeof(float));
    printf(">>> rope_neox %s  n_rows=%d rot=%d out_f16=%d grid=(%u) block=%u  traffic=%.3f MB\n",
           name, n_rows, rot, out_f16, grid.x, block.x, bytes / 1e6);

    rope_neox<<<grid, block>>>(d_src, d_dst, n_rows, rot, /*pos*/1024, /*theta_base*/1e6f, out_f16);
    BENCH_CHECK(cudaGetLastError());
    BENCH_CHECK(cudaDeviceSynchronize());

    delete[] h_src;
    cudaFree(d_src); cudaFree(d_dst);
    return 0;
}

int main(int argc, char **argv) {
    cudaDeviceProp prop{};
    BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (rope_neox decode benchmark)\n", prop.name);

    // batch = parallel decode tokens; multiplies the head-row count.
    int batch = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
    if (batch < 1) batch = 1;

    if (argc >= 4) {
        const int n_rows  = atoi(argv[1]) * batch;
        const int rot     = atoi(argv[2]);
        const int out_f16 = atoi(argv[3]);
        return run_shape(n_rows, rot, out_f16, "custom");
    }

    // Qwen3-14B decode: head_dim=128, n_q_heads=40, n_kv_heads=8 (one token/seq).
    //   ropeQ: 40 head-rows, f32 out;  ropeK: 8 head-rows, f16 out (-> KV cache).
    if (run_shape(40 * batch, 128, 0, "ropeQ_qwen3_14b")) return 1;
    if (run_shape(8  * batch, 128, 1, "ropeK_qwen3_14b_f16")) return 1;
    return 0;
}
