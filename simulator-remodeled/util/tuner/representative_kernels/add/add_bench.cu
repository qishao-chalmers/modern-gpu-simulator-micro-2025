// Standalone residual-add benchmark — llama.cpp decode residual connections.
//
// Faithful to ggml_cuda add (ggml-cuda/binbcast.cu): x[i] += y[i], elementwise
//   over the hidden width. Two per layer (attn residual, ffn residual).
// Functional-sim-safe: scalar f32 math.
//
// Build:  make            (sm_90)
//         make exec       (sm_70, GPGPU-Sim execution-driven)
// Run:    ./add_bench
//         ./add_bench <n>          # custom one-shot (hidden elements)
//         BATCH=32 ./add_bench      # batch multiplies n (one row per token)
// batch = parallel decode tokens; each contributes one hidden-state row.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "../common/bench_common.h"

__global__ void add_inplace(float *x, const float *y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += y[i];
}

#define NT 256

static int run_shape(int n, const char *name) {
    float *d_x = nullptr, *d_y = nullptr;
    float *h = new float[n];
    fill_f32(h, n, 0.5f);

    BENCH_CHECK(cudaMalloc(&d_x, (size_t) n * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_y, (size_t) n * sizeof(float)));
    BENCH_CHECK(cudaMemcpy(d_x, h, (size_t) n * sizeof(float), cudaMemcpyHostToDevice));
    BENCH_CHECK(cudaMemcpy(d_y, h, (size_t) n * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 grid((n + NT - 1) / NT), block(NT);
    const double bytes = (double) n * sizeof(float) * 3.0;  // read x + read y + write x
    printf(">>> add %s  n=%d grid=(%u) block=%u  traffic=%.3f MB\n", name, n, grid.x, block.x, bytes / 1e6);

    add_inplace<<<grid, block>>>(d_x, d_y, n);
    BENCH_CHECK(cudaGetLastError());
    BENCH_CHECK(cudaDeviceSynchronize());

    // verify: 0.5 + 0.5 = 1.0
    float h0 = 0.0f;
    BENCH_CHECK(cudaMemcpy(&h0, d_x, sizeof(float), cudaMemcpyDeviceToHost));
    printf("    [verify] x[0]=%.4f expected=1.0000  %s\n", h0, (fabsf(h0 - 1.0f) < 1e-4f) ? "OK" : "MISMATCH");

    delete[] h;
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}

int main(int argc, char **argv) {
    cudaDeviceProp prop{};
    BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (residual-add decode benchmark)\n", prop.name);

    int batch = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
    if (batch < 1) batch = 1;

    if (argc >= 2) {
        const int n = atoi(argv[1]) * batch;
        return run_shape(n, "custom");
    }

    // Qwen3-14B decode: hidden width = 5120 per token (attn + ffn residual, 2 per layer).
    if (run_shape(5120 * batch, "add_residual_qwen3_14b")) return 1;
    return 0;
}
