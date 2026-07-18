// Standalone set_rows benchmark — llama.cpp KV-cache write (ggml_cuda_cpy / set_rows).
//
// After ropeK and V quantization, the new token's K and V head-rows (f16) are
// scattered into the persistent KV cache at sequence position `pos`. Models the
// per-decode-step write of n_kv_heads * head_dim f16 elements into a seq-strided
// cache. Functional-sim-safe: plain f16-bit copy (uint16_t).
//
// Build:  make            (sm_90)
//         make exec       (sm_70, GPGPU-Sim execution-driven)
// Run:    ./set_rows_bench
//         ./set_rows_bench <n_elem> <seq_len>        # custom one-shot
//         BATCH=32 ./set_rows_bench                   # batch multiplies rows written
// batch = parallel decode tokens; each writes one K row and one V row.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../common/bench_common.h"

// Write a contiguous head-row block (f16) into the cache at row `pos`.
__global__ void k_set_rows_f16(const uint16_t *__restrict__ src, uint16_t *__restrict__ cache,
                               int n_elem, int pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n_elem) return;
    cache[(size_t) pos * n_elem + i] = src[i];
}

#define NT 256

static int run_shape(int n_elem, int seq_len, int rows, const char *name) {
    uint16_t *d_src = nullptr, *d_cache = nullptr;
    const size_t cache_elems = (size_t) seq_len * n_elem;
    BENCH_CHECK(cudaMalloc(&d_src, (size_t) n_elem * sizeof(uint16_t)));
    BENCH_CHECK(cudaMalloc(&d_cache, cache_elems * sizeof(uint16_t)));
    BENCH_CHECK(cudaMemset(d_src, 0x3c, (size_t) n_elem * sizeof(uint16_t)));   // ~f16 1.0 pattern

    const dim3 grid((n_elem + NT - 1) / NT), block(NT);
    const double bytes = (double) rows * n_elem * sizeof(uint16_t) * 2.0;       // read src + write cache
    printf(">>> set_rows %s  n_elem=%d seq_len=%d rows=%d grid=(%u) block=%u  traffic=%.3f MB\n",
           name, n_elem, seq_len, rows, grid.x, block.x, bytes / 1e6);

    // one launch per row written (matches per-token KV scatter); pos within seq range.
    for (int r = 0; r < rows; ++r) {
        k_set_rows_f16<<<grid, block>>>(d_src, d_cache, n_elem, /*pos*/ (seq_len - 1));
    }
    BENCH_CHECK(cudaGetLastError());
    BENCH_CHECK(cudaDeviceSynchronize());

    cudaFree(d_src); cudaFree(d_cache);
    return 0;
}

int main(int argc, char **argv) {
    cudaDeviceProp prop{};
    BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (set_rows KV-cache write benchmark)\n", prop.name);

    int batch = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
    if (batch < 1) batch = 1;

    if (argc >= 3) {
        const int n_elem  = atoi(argv[1]);
        const int seq_len = atoi(argv[2]);
        return run_shape(n_elem, seq_len, batch, "custom");
    }

    // Qwen3-14B decode: n_kv_heads=8, head_dim=128 -> 1024 f16 per row; seq_len=1088.
    //   set_rows K and set_rows V each write one such row per token.
    if (run_shape(8 * 128, 1088, 2 * batch, "set_rows_KV_qwen3_14b")) return 1;
    return 0;
}
