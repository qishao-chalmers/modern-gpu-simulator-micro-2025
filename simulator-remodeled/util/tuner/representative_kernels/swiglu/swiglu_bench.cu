// Standalone SwiGLU benchmark — llama.cpp FFN gate activation.
//
// Faithful to ggml/src/ggml-cuda/unary.cu swiglu: o[i] = silu(g[i]) * u[i],
//   silu(x) = x / (1 + exp(-x)). Elementwise over the FFN intermediate width.
// Functional-sim-safe: scalar f32 math.
//
// Build:  make            (sm_90)
//         make exec       (sm_70, GPGPU-Sim execution-driven)
// Run:    ./swiglu_bench
//         ./swiglu_bench <n>          # custom one-shot (intermediate elements)
//         BATCH=32 ./swiglu_bench      # batch multiplies n (one row per token)
// batch = parallel decode tokens; each contributes one FFN intermediate row.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "../common/bench_common.h"

__global__ void swiglu(const float *g, const float *u, float *o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const float gv = g[i]; o[i] = (gv / (1.0f + expf(-gv))) * u[i];
}

#define NT 256

static int run_shape(int n, const char *name) {
    float *d_g = nullptr, *d_u = nullptr, *d_o = nullptr;
    float *h = new float[n];
    fill_f32(h, n, 0.5f);

    BENCH_CHECK(cudaMalloc(&d_g, (size_t) n * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_u, (size_t) n * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_o, (size_t) n * sizeof(float)));
    BENCH_CHECK(cudaMemcpy(d_g, h, (size_t) n * sizeof(float), cudaMemcpyHostToDevice));
    BENCH_CHECK(cudaMemcpy(d_u, h, (size_t) n * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 grid((n + NT - 1) / NT), block(NT);
    const double bytes = (double) n * sizeof(float) * 3.0;  // read g + read u + write o
    printf(">>> swiglu %s  n=%d grid=(%u) block=%u  traffic=%.3f MB\n", name, n, grid.x, block.x, bytes / 1e6);

    swiglu<<<grid, block>>>(d_g, d_u, d_o, n);
    BENCH_CHECK(cudaGetLastError());
    BENCH_CHECK(cudaDeviceSynchronize());

    // verify: silu(0.5)*0.5 = (0.5/(1+e^-0.5))*0.5 ~= 0.15561
    float h0 = 0.0f;
    BENCH_CHECK(cudaMemcpy(&h0, d_o, sizeof(float), cudaMemcpyDeviceToHost));
    printf("    [verify] o[0]=%.5f expected=0.15561  %s\n", h0, (fabsf(h0 - 0.15561f) < 1e-3f) ? "OK" : "MISMATCH");

    delete[] h;
    cudaFree(d_g); cudaFree(d_u); cudaFree(d_o);
    return 0;
}

int main(int argc, char **argv) {
    cudaDeviceProp prop{};
    BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (swiglu FFN benchmark)\n", prop.name);

    int batch = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
    if (batch < 1) batch = 1;

    if (argc >= 2) {
        const int n = atoi(argv[1]) * batch;
        return run_shape(n, "custom");
    }

    // Qwen3-14B decode: FFN intermediate width = 17408 per token.
    if (run_shape(17408 * batch, "swiglu_qwen3_14b")) return 1;
    return 0;
}
