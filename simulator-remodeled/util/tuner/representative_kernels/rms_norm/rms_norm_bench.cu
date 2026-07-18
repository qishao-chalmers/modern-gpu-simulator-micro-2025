// Standalone rms_norm_f32 benchmark — llama.cpp decode norms.
//
// Faithful to ggml/src/ggml-cuda/norm.cu rms_norm_f32<block_size,false>:
//   one block per (row, channel, sample); RMS over ncols.
// Functional-sim-safe: shared-memory block reduce (no warp shuffles).
//
// Build:  make            (sm_90)
//         make exec       (sm_70, GPGPU-Sim execution-driven)
// Run:    ./rms_norm_bench
//         ./rms_norm_bench_exec 5120 1 1024     # ncols nrows block_size
//         BATCH=32 ./rms_norm_bench_exec 5120 1 1024  # batch multiplies nrows (token dim)
// batch = parallel decode tokens; each adds a row to normalize -> scales nrows (grid.x).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "../common/bench_common.h"
#include "../common/shapes_qwen3_14b_decode.h"

#define WARP_SIZE 32
static constexpr float kEps = 1e-5f;

template <int block_size>
__device__ float block_sum(float val) {
    __shared__ float sm[block_size];
    sm[threadIdx.x] = val;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sm[threadIdx.x] += sm[threadIdx.x + s];
        }
        __syncthreads();
    }
    return sm[0];
}

// Matches llama.cpp rms_norm_f32<block_size, false> (norm.cu).
template <int block_size>
__global__ void rms_norm_f32(const float *x, float *dst, int ncols, int64_t stride_row,
                             int64_t stride_channel, int64_t stride_sample, float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;
    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    tmp = block_sum<block_size>(tmp);

    const float scale = rsqrtf(tmp / (float) ncols + eps);
    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

template <int block_size>
static void launch_rms(const float *x, float *dst, int ncols, int nrows, int nchannels, int nsamples,
                       cudaStream_t stream) {
    const dim3 grid(nrows, nchannels, nsamples);
    const dim3 block(block_size, 1, 1);
    const size_t shmem = block_size > WARP_SIZE ? 0 : 0;
    rms_norm_f32<block_size><<<grid, block, shmem, stream>>>(x, dst, ncols, ncols, ncols * nrows,
                                                             ncols * nrows * nchannels, kEps);
}

static int run_shape(int ncols, int nrows, int nchannels, int nsamples, int block_size, const char *name) {
    const size_t n_in  = (size_t) ncols * nrows * nchannels * nsamples;
    const size_t n_out = n_in;

    float *d_x = nullptr;
    float *d_dst = nullptr;
    float *h_x = new float[n_in];
    fill_f32(h_x, n_in, 0.5f);

    BENCH_CHECK(cudaMalloc(&d_x, n_in * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_dst, n_out * sizeof(float)));
    BENCH_CHECK(cudaMemcpy(d_x, h_x, n_in * sizeof(float), cudaMemcpyHostToDevice));

    const double bytes = (double) n_in * sizeof(float) * 2.0; // read x + write dst
    printf(">>> rms_norm %s  ncols=%d nrows=%d nch=%d nsamples=%d block=%d  traffic=%.3f MB\n",
           name, ncols, nrows, nchannels, nsamples, block_size, bytes / 1e6);

    if (block_size == 256) {
        launch_rms<256>(d_x, d_dst, ncols, nrows, nchannels, nsamples, 0);
    } else {
        launch_rms<1024>(d_x, d_dst, ncols, nrows, nchannels, nsamples, 0);
    }
    BENCH_CHECK(cudaDeviceSynchronize());

    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_dst);
    return 0;
}

int main(int argc, char **argv) {
    cudaDeviceProp prop{};
    BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (rms_norm decode benchmark)\n", prop.name);

    // batch = parallel decode tokens; multiplies nrows (one normalized row per token).
    int batch = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
    if (batch < 1) batch = 1;

    if (argc >= 4) {
        const int ncols      = atoi(argv[1]);
        const int nrows      = atoi(argv[2]) * batch;
        const int block_size = atoi(argv[3]);
        const int nch        = (argc > 4) ? atoi(argv[4]) : 1;
        return run_shape(ncols, nrows, nch, 1, block_size, "custom");
    }

    for (int i = 0; i < qwen3_14b_rms_num_shapes; ++i) {
        const rms_shape &s = qwen3_14b_rms_shapes[i];
        if (run_shape(s.ncols, s.nrows * batch, s.nchannels, s.nsamples, s.block_size, s.name)) {
            return 1;
        }
    }
    return 0;
}
