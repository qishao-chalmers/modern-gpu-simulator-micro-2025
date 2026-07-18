#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#define BENCH_CHECK(x)                                                                 \
    do {                                                                               \
        cudaError_t e = (x);                                                           \
        if (e) {                                                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(e));                                            \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

static inline void fill_f32(float *p, size_t n, float v = 1.0f) {
    for (size_t i = 0; i < n; ++i) {
        p[i] = v;
    }
}
