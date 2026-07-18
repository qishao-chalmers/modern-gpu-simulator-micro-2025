// Decode GEMV with 4-bit activations: W[N×K](Q8_0/Q4_K/Q2_K) · y[K](q4_1) -> dst[N].
//
// Differs from mmvq_kquant.cu (y = q8_1, int8×int8 / __dp4a):
//   - y stored as packed int4 blocks (block_q4_1) — ~half the activation traffic
//   - hot dot is int4×int4 via dp8a_i4 (8 nibble MACs per 32-bit pair)
//
// Hardware story for Accel-Sim: build with -DSIM_DP8A to emit PTX `dp8a.s32.s32`
// (parsed/executed by GPGPU-Sim; models Blackwell-class int4 dot density).
// Without SIM_DP8A, the same math runs as a scalar unpack (works on real H100).
//
// Build:  make 4bit              (sm_90, software int4)
//         make 4bit_exec         (sm_70 + FUNCSIM_SAFE, software int4)
//         make 4bit_exec_dp8a    (compute_70 PTX + FUNCSIM_SAFE + SIM_DP8A)
// Run:    ./mmvq_kquant_4bit 4096 4096 q8_0
//         ./mmvq_kquant_4bit 8b q_proj q2_k

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../common/shapes_qwen3_gemv.h"

#define QK8_0 32
#define QK4_1 32
#define QK_K  256
#define K_SCALE_SIZE 12
#define WARP_SIZE 32
#define NWARPS 4
#define NTHREADS (NWARPS * WARP_SIZE)

// --- weight layouts (same as mmvq_kquant) + q4_1 activations -----------------
#ifdef FUNCSIM_SAFE
typedef struct { float d;          int8_t qs[QK8_0]; } block_q8_0;
typedef struct { float d; float dmin; uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; } block_q2_K;
typedef struct { float d; float dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { float d; float s; uint8_t qs[QK4_1 / 2]; } block_q4_1; // 32 nibbles
#else
typedef struct { half  d;  int8_t qs[QK8_0]; } block_q8_0;
typedef struct { half2 dm; uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; } block_q2_K;
typedef struct { half2 dm; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { half2 ds; uint8_t qs[QK4_1 / 2]; } block_q4_1;
static_assert(sizeof(block_q8_0) == 34, "block_q8_0");
static_assert(sizeof(block_q2_K) == 84, "block_q2_K");
static_assert(sizeof(block_q4_K) == 144, "block_q4_K");
static_assert(sizeof(block_q4_1) == 20, "block_q4_1");
#endif

enum quant_type { QUANT_Q8_0 = 0, QUANT_Q4_K = 1, QUANT_Q2_K = 2 };

#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

// 8× int4 MAC: d = c + sum_i a.nibble[i] * b.nibble[i]  (unsigned 0..15)
// SIM_DP8A: emit custom PTX for Accel-Sim / GPGPU-Sim. Else: scalar (real GPU).
static __device__ __forceinline__ int dp8a_i4(int a, int b, int c) {
#ifdef SIM_DP8A
    int out;
    asm volatile("dp8a.s32.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
#else
    int acc = c;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ai = (a >> (4 * i)) & 0x0f;
        const int bi = (b >> (4 * i)) & 0x0f;
        acc += ai * bi;
    }
    return acc;
#endif
}

// Pack 8 int8 values' low nibbles into one int (for Q8_0 × q4 path).
static __device__ __forceinline__ int pack_i4x8_from_i8(const int8_t *p) {
    int r = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) r |= ((int)(p[i] & 0x0f)) << (4 * i);
    return r;
}

// Pack 8× 2-bit weights into int4 lanes (zero-extend) for Q2_K × q4.
static __device__ __forceinline__ int pack_i4x8_from_q2(const uint8_t *p) {
    // p has 2 bytes = 8× 2-bit
    int r = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int byte = p[i / 4];
        const int q2 = (byte >> ((i % 4) * 2)) & 0x3;
        r |= q2 << (4 * i);
    }
    return r;
}

static __device__ __forceinline__ float scale_q8_0(const block_q8_0 *x) {
#ifdef FUNCSIM_SAFE
    return x->d;
#else
    return __half2float(x->d);
#endif
}

static __device__ __forceinline__ float scale_q4_1(const block_q4_1 *y) {
#ifdef FUNCSIM_SAFE
    return y->d;
#else
    return __low2float(y->ds);
#endif
}

static __device__ __forceinline__ float scale_q4_k(const block_q4_K *x) {
#ifdef FUNCSIM_SAFE
    return x->d;
#else
    return __half22float2(x->dm).x;
#endif
}

static __device__ __forceinline__ float scale_q2_k(const block_q2_K *x) {
#ifdef FUNCSIM_SAFE
    return x->d;
#else
    return __half22float2(x->dm).x;
#endif
}

static __device__ __forceinline__ void touch_scales(const uint8_t *scales, int nbytes) {
    int t = 0;
#pragma unroll
    for (int i = 0; i < nbytes; ++i) t += scales[i];
    (void)t;
}

// Q8_0 (int8 storage, low-nibble used) × q4_1 — 4× dp8a over 32 elems.
static __device__ __forceinline__ float vec_dot_q8_0_q4_1(const block_q8_0 *x, const block_q4_1 *y) {
    const uint8_t *yq = y->qs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int wa = pack_i4x8_from_i8(x->qs + 8 * i);
        const int yb = *reinterpret_cast<const int *>(yq + 4 * i);
        sumi = dp8a_i4(wa, yb, sumi);
    }
    return scale_q8_0(x) * scale_q4_1(y) * (float)sumi;
}

// One Q4_K sub (32 nibbles) × one q4_1 block.
static __device__ __forceinline__ float vec_dot_q4_k_q4_1_sub(const block_q4_K *x,
                                                              const block_q4_1 *y, int sub) {
    touch_scales(x->scales, K_SCALE_SIZE);
    const uint8_t *wq = x->qs + sub * 16;
    const uint8_t *yq = y->qs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int wa = *reinterpret_cast<const int *>(wq + 4 * i);
        const int yb = *reinterpret_cast<const int *>(yq + 4 * i);
        sumi = dp8a_i4(wa, yb, sumi);
    }
    return scale_q4_k(x) * scale_q4_1(y) * (float)sumi;
}

// One Q2_K sub (32× 2-bit → int4 lanes) × one q4_1 block.
static __device__ __forceinline__ float vec_dot_q2_k_q4_1_sub(const block_q2_K *x,
                                                              const block_q4_1 *y, int sub) {
    (void)x->scales[2 * sub];
    (void)x->scales[2 * sub + 1];
    const uint8_t *wq = x->qs + sub * 8;
    const uint8_t *yq = y->qs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int wa = pack_i4x8_from_q2(wq + 2 * i);
        const int yb = *reinterpret_cast<const int *>(yq + 4 * i);
        sumi = dp8a_i4(wa, yb, sumi);
    }
    return scale_q2_k(x) * scale_q4_1(y) * (float)sumi;
}

template <quant_type Q>
__global__ void mmvq_gemv_q4act(const void *__restrict__ vx,
                                const block_q4_1 *__restrict__ vy,
                                float *__restrict__ dst, int K, int N) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int nthreads = NTHREADS;

    float acc = 0.0f;

    if (Q == QUANT_Q8_0) {
        const int bpr = K / QK8_0;
        const block_q8_0 *xrow = (const block_q8_0 *)vx + (size_t)row * bpr;
        for (int kb = tid; kb < bpr; kb += nthreads) {
            acc += vec_dot_q8_0_q4_1(&xrow[kb], &vy[kb]);
        }
    } else {
        const int nsub = K / QK4_1;
        const int sub_per = QK_K / QK4_1; // 8
        if (Q == QUANT_Q4_K) {
            const block_q4_K *xrow = (const block_q4_K *)vx + (size_t)row * (K / QK_K);
            for (int is = tid; is < nsub; is += nthreads) {
                acc += vec_dot_q4_k_q4_1_sub(&xrow[is / sub_per], &vy[is], is % sub_per);
            }
        } else {
            const block_q2_K *xrow = (const block_q2_K *)vx + (size_t)row * (K / QK_K);
            for (int is = tid; is < nsub; is += nthreads) {
                acc += vec_dot_q2_k_q4_1_sub(&xrow[is / sub_per], &vy[is], is % sub_per);
            }
        }
    }

#ifdef FUNCSIM_SAFE
    __shared__ float sm[NTHREADS];
    sm[tid] = acc;
    __syncthreads();
    for (int s = NTHREADS / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) dst[row] = sm[0];
#else
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
#endif
}

__global__ void fill_q8_0(block_q8_0 *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#ifdef FUNCSIM_SAFE
    b[i].d = 1.0f;
#else
    b[i].d = __float2half(1.0f);
#endif
    for (int j = 0; j < QK8_0; ++j) b[i].qs[j] = 1;
}

__global__ void fill_q4_1(block_q4_1 *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#ifdef FUNCSIM_SAFE
    b[i].d = 1.0f; b[i].s = 32.0f;
#else
    b[i].ds = __floats2half2_rn(1.0f, 32.0f);
#endif
    // 0x11 → both nibbles = 1
    for (int j = 0; j < QK4_1 / 2; ++j) b[i].qs[j] = 0x11;
}

__global__ void fill_q4_k(block_q4_K *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#ifdef FUNCSIM_SAFE
    b[i].d = 1.0f; b[i].dmin = 0.0f;
#else
    b[i].dm = __floats2half2_rn(1.0f, 0.0f);
#endif
    for (int j = 0; j < K_SCALE_SIZE; ++j) b[i].scales[j] = 1;
    for (int j = 0; j < QK_K / 2; ++j) b[i].qs[j] = 0x11;
}

__global__ void fill_q2_k(block_q2_K *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#ifdef FUNCSIM_SAFE
    b[i].d = 1.0f; b[i].dmin = 0.0f;
#else
    b[i].dm = __floats2half2_rn(1.0f, 0.0f);
#endif
    for (int j = 0; j < QK_K / 16; ++j) b[i].scales[j] = 0x11;
    for (int j = 0; j < QK_K / 4; ++j) b[i].qs[j] = 0x55; // all 2-bit fields = 1
}

static size_t weight_bytes(quant_type q, int K, int N) {
    if (q == QUANT_Q8_0) return (size_t)N * (K / QK8_0) * sizeof(block_q8_0);
    return (size_t)N * (K / QK_K) * (q == QUANT_Q4_K ? sizeof(block_q4_K) : sizeof(block_q2_K));
}

static size_t act_bytes(int K) {
    return (size_t)(K / QK4_1) * sizeof(block_q4_1);
}

static const char *quant_name(quant_type q) {
    switch (q) {
        case QUANT_Q8_0: return "q8_0";
        case QUANT_Q4_K: return "q4_k";
        case QUANT_Q2_K: return "q2_k";
    }
    return "?";
}

static int parse_quant(const char *s, quant_type *q) {
    if (!s) return -1;
    if (strcmp(s, "q8_0") == 0 || strcmp(s, "Q8_0") == 0) { *q = QUANT_Q8_0; return 0; }
    if (strcmp(s, "q4_k") == 0 || strcmp(s, "Q4_K") == 0) { *q = QUANT_Q4_K; return 0; }
    if (strcmp(s, "q2_k") == 0 || strcmp(s, "Q2_K") == 0) { *q = QUANT_Q2_K; return 0; }
    return -1;
}

static bool is_all_digits(const char *s) {
    if (!s || !*s) return false;
    for (; *s; ++s) if (*s < '0' || *s > '9') return false;
    return true;
}

static int run_one(int K, int N, quant_type q, const char *label, bool verify, bool timeit) {
    const int k_align = (q == QUANT_Q8_0) ? QK8_0 : QK_K;
    if (K % k_align != 0) {
        printf("K=%d must be divisible by %d for %s\n", K, k_align, quant_name(q));
        return 1;
    }
    if (N <= 0) {
        printf("N=%d must be positive\n", N);
        return 1;
    }

    const size_t w_bytes = weight_bytes(q, K, N);
    const size_t y_bytes = act_bytes(K);
    const size_t n_yblocks = (size_t)(K / QK4_1);
    const bool skip_fill = getenv("MMVQ_SKIP_FILL") != nullptr;

    void *d_w = nullptr;
    block_q4_1 *d_y = nullptr;
    float *d_dst = nullptr;

    if (q == QUANT_Q8_0) {
        const size_t n = (size_t)N * (K / QK8_0);
        CHECK(cudaMalloc(&d_w, n * sizeof(block_q8_0)));
        if (!skip_fill) fill_q8_0<<<(n + 255) / 256, 256>>>((block_q8_0 *)d_w, n);
    } else if (q == QUANT_Q4_K) {
        const size_t n = (size_t)N * (K / QK_K);
        CHECK(cudaMalloc(&d_w, n * sizeof(block_q4_K)));
        if (!skip_fill) fill_q4_k<<<(n + 255) / 256, 256>>>((block_q4_K *)d_w, n);
    } else {
        const size_t n = (size_t)N * (K / QK_K);
        CHECK(cudaMalloc(&d_w, n * sizeof(block_q2_K)));
        if (!skip_fill) fill_q2_k<<<(n + 255) / 256, 256>>>((block_q2_K *)d_w, n);
    }

    CHECK(cudaMalloc(&d_y, n_yblocks * sizeof(block_q4_1)));
    CHECK(cudaMalloc(&d_dst, (size_t)N * sizeof(float)));
    if (!skip_fill) fill_q4_1<<<(n_yblocks + 255) / 256, 256>>>(d_y, n_yblocks);
    CHECK(cudaGetLastError());

    dim3 block(WARP_SIZE, NWARPS);
    dim3 grid(N);

    auto launch = [&]() {
        switch (q) {
            case QUANT_Q8_0: mmvq_gemv_q4act<QUANT_Q8_0><<<grid, block>>>(d_w, d_y, d_dst, K, N); break;
            case QUANT_Q4_K: mmvq_gemv_q4act<QUANT_Q4_K><<<grid, block>>>(d_w, d_y, d_dst, K, N); break;
            case QUANT_Q2_K: mmvq_gemv_q4act<QUANT_Q2_K><<<grid, block>>>(d_w, d_y, d_dst, K, N); break;
        }
    };

    launch();
    CHECK(cudaDeviceSynchronize());

    if (verify && !skip_fill) {
        float h0 = 0.0f;
        CHECK(cudaMemcpy(&h0, d_dst, sizeof(float), cudaMemcpyDeviceToHost));
        const float expect = (float)K;
        const bool ok = fabsf(h0 - expect) < 0.5f;
        printf("  [verify] dst[0]=%.1f expect=%.1f %s  (%s %s×q4_1 K=%d N=%d W=%.2f MB Y=%.3f MB)\n",
               h0, expect, ok ? "OK" : "FAIL",
               label ? label : "-", quant_name(q), K, N, w_bytes / 1e6, y_bytes / 1e6);
        if (!ok) {
            cudaFree(d_w); cudaFree(d_y); cudaFree(d_dst);
            return 1;
        }
    } else if (verify && skip_fill) {
        printf("  [verify] skipped (MMVQ_SKIP_FILL)  (%s %s×q4_1 K=%d N=%d)\n",
               label ? label : "-", quant_name(q), K, N);
    }

    if (timeit) {
        const int iters = 20;
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));
        CHECK(cudaEventRecord(t0));
        for (int it = 0; it < iters; ++it) launch();
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        const double us = ms * 1e3 / iters;
        const double gbs_w = w_bytes / (us * 1e-6) / 1e9;
        printf("  %-10s %-5s×q4_1 K=%-6d N=%-6d  W=%7.2f MB Y=%6.3f MB  %8.2f us  %7.1f GB/s(W)\n",
               label ? label : "-", quant_name(q), K, N, w_bytes / 1e6, y_bytes / 1e6, us, gbs_w);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    } else {
        printf(">>> launch %s %s×q4_1 K=%d N=%d W=%.3f MB Y=%.3f MB\n",
               label ? label : "-", quant_name(q), K, N, w_bytes / 1e6, y_bytes / 1e6);
    }

    cudaFree(d_w);
    cudaFree(d_y);
    cudaFree(d_dst);
    return 0;
}

static void usage(const char *argv0) {
    printf("Usage:\n");
    printf("  %s <model> <op|all> <q8_0|q4_k|q2_k>\n", argv0);
    printf("  %s <K> <N> <q8_0|q4_k|q2_k>\n", argv0);
    printf("  model: 8b | 14b   (activations always q4_1 / int4×int4)\n");
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Device: %s  (GEMV: W∈{Q8_0,Q4_K,Q2_K} × y=q4_1, int4×int4/dp8a)\n", p.name);

#ifdef FUNCSIM_SAFE
    printf("Build: FUNCSIM_SAFE");
#ifdef SIM_DP8A
    printf(" + SIM_DP8A (PTX dp8a → Accel-Sim int4 MAC)\n");
#else
    printf(" (software int4 unpack)\n");
#endif
#elif defined(SIM_DP8A)
    printf("Build: SIM_DP8A\n");
#endif

    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }

    quant_type q;
    if (parse_quant(argv[3], &q) != 0) {
        printf("Unknown quant: %s\n", argv[3]);
        return 1;
    }

    if (is_all_digits(argv[1]) && is_all_digits(argv[2])) {
        return run_one(atoi(argv[1]), atoi(argv[2]), q, "custom", true, !getenv("MMVQ_NO_TIME"));
    }

    if (getenv("MMVQ_TINY")) {
        return run_one(256, 64, q, "tiny", true, !getenv("MMVQ_NO_TIME"));
    }

    const char *model = argv[1];
    const char *op = argv[2];
    int n_shapes = 0;
    const gemv_shape *tbl = gemv_shapes_for_model(model, &n_shapes);

    if (strcmp(op, "all") == 0) {
        printf("[model=%s W=%s y=q4_1 — %d shapes]\n", model, quant_name(q), n_shapes);
        for (int i = 0; i < n_shapes; ++i) {
            if (run_one(tbl[i].K, tbl[i].N, q, tbl[i].name, false, !getenv("MMVQ_NO_TIME")))
                return 1;
        }
        return 0;
    }

    int K = 0, N = 0;
    if (gemv_shape_lookup(tbl, n_shapes, op, &K, &N) != 0) {
        printf("Unknown op '%s' for model %s\n", op, model);
        return 1;
    }
    return run_one(K, N, q, op, true, !getenv("MMVQ_NO_TIME"));
}
