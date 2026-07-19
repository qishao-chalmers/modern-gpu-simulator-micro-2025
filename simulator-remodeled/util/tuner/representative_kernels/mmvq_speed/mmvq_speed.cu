// SPEED Q||R strip GEMV prototype (modes 1 / 2). Sibling of mmvq_kquant — do not merge.
// Layout per row: [ QxG | RxG ] [ QxG | RxG ] ...  G >= 1 (e.g. 1..64+), group = 256 (QK_K).
// Constraint: K must be divisible by G*256 (e.g. G=64 => K multiple of 16384).
//   mode 1: Q-only GEMV (skip R gaps) — compare layout tax vs packed mmvq_kquant
//   mode 2: SW rebuild  acc += dot(Q,y)+dot(R,y)  — dequant-space W_hat = Q+R
//   mode 3: HW FR rebuild — CUDA is plain Q8; only meaningful under the simulator (stub).
//
// Build:  make | make exec
// Run:    ./mmvq_speed 1 4 q4_k q2_k 4096 4096
//         ./mmvq_speed 2 4 q2_k q4_k 8b q_proj
// See DESIGN.md.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../common/shapes_qwen3_gemv.h"

#define QK8_0 32
#define QK8_1 32
#define QK_K  256
#define K_SCALE_SIZE 12
#define WARP_SIZE 32
#define NWARPS 4
#define NTHREADS (NWARPS * WARP_SIZE)

#ifdef FUNCSIM_SAFE
typedef struct { float d;          int8_t qs[QK8_0]; } block_q8_0;
typedef struct { float d; float s; int8_t qs[QK8_1]; } block_q8_1;
typedef struct { float d; float dmin; uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; } block_q2_K;
typedef struct { float d; float dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { float d; uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; } block_q3_K;
#else
typedef struct { half  d;  int8_t qs[QK8_0]; } block_q8_0;
typedef struct { half2 ds; int8_t qs[QK8_1]; } block_q8_1;
typedef struct { half2 dm; uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; } block_q2_K;
typedef struct { half2 dm; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { half d; uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; } block_q3_K;
static_assert(sizeof(block_q8_0) == 34, "block_q8_0");
static_assert(sizeof(block_q8_1) == 36, "block_q8_1");
static_assert(sizeof(block_q2_K) == 84, "block_q2_K");
static_assert(sizeof(block_q4_K) == 144, "block_q4_K");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K");
#endif

enum quant_type { QUANT_Q4_K = 0, QUANT_Q2_K = 1, QUANT_Q3_K = 2, QUANT_Q8_0 = 3 };
enum run_mode   { MODE_Q_ONLY = 1, MODE_SW_REBUILD = 2, MODE_HW_FR = 3 };

#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

// Weight alloc: under GPGPU-Sim, mark the region for Mode3/QWC. Provide the symbol in
// this TU so `make exec` links against stock --cudart shared; the sim lib's
// gpgpuSimMarkWeightRegion is resolved weakly at runtime when LD_LIBRARY_PATH
// points at the patched libcudart.
#ifdef FUNCSIM_SAFE
extern "C" void gpgpuSimMarkWeightRegion(void *ptr, size_t size) __attribute__((weak));
extern "C" cudaError_t cudaMallocWeight(void **devPtr, size_t size) {
    cudaError_t e = cudaMalloc(devPtr, size);
    if (e == cudaSuccess && devPtr && *devPtr && gpgpuSimMarkWeightRegion)
        gpgpuSimMarkWeightRegion(*devPtr, size);
    return e;
}
static inline cudaError_t mmvq_malloc_weight(void **p, size_t n) {
    return cudaMallocWeight(p, n);
}
#else
static inline cudaError_t mmvq_malloc_weight(void **p, size_t n) {
    return cudaMalloc(p, n);
}
#endif

#ifndef FUNCSIM_SAFE
static __device__ __forceinline__ int load_int(const int8_t *p) {
    const uint16_t *p16 = reinterpret_cast<const uint16_t *>(p);
    return (int)p16[0] | ((int)p16[1] << 16);
}
#endif

static __device__ __forceinline__ float block_scale_q8_0(const block_q8_0 *x) {
#ifdef FUNCSIM_SAFE
    return x->d;
#else
    return __half2float(x->d);
#endif
}
static __device__ __forceinline__ float block_scale_q8_1(const block_q8_1 *y) {
#ifdef FUNCSIM_SAFE
    return y->d;
#else
    return __low2float(y->ds);
#endif
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(const block_q8_0 *x, const block_q8_1 *y) {
#ifdef FUNCSIM_SAFE
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) sumf += (float)x->qs[j] * (float)y->qs[j];
    return block_scale_q8_0(x) * block_scale_q8_1(y) * sumf;
#else
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < QK8_0 / 4; ++i)
        sumi = __dp4a(load_int(x->qs + 4 * i), load_int(y->qs + 4 * i), sumi);
    return block_scale_q8_0(x) * block_scale_q8_1(y) * (float)sumi;
#endif
}

static __device__ __forceinline__ void touch_scales(const uint8_t *scales, int nbytes) {
    int t = 0;
#pragma unroll
    for (int i = 0; i < nbytes; ++i) t += scales[i];
    (void)t;
}

static __device__ __forceinline__ float kquant_scale_q4(const block_q4_K *x, const block_q8_1 *y) {
#ifdef FUNCSIM_SAFE
    return x->d * block_scale_q8_1(y);
#else
    return __half22float2(x->dm).x * block_scale_q8_1(y);
#endif
}
static __device__ __forceinline__ float kquant_scale_q2(const block_q2_K *x, const block_q8_1 *y) {
#ifdef FUNCSIM_SAFE
    return x->d * block_scale_q8_1(y);
#else
    return __half22float2(x->dm).x * block_scale_q8_1(y);
#endif
}
static __device__ __forceinline__ float kquant_scale_q3(const block_q3_K *x, const block_q8_1 *y) {
#ifdef FUNCSIM_SAFE
    return x->d * block_scale_q8_1(y);
#else
    return __half2float(x->d) * block_scale_q8_1(y);
#endif
}

static __device__ __forceinline__ float vec_dot_q4_k_sub(const block_q4_K *x, const block_q8_1 *y, int sub) {
    touch_scales(x->scales, K_SCALE_SIZE);
    const uint8_t *qs = x->qs + sub * 16;
    const int8_t *yq = y->qs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int b0 = qs[2 * i], b1 = qs[2 * i + 1];
#ifndef FUNCSIM_SAFE
        const int q8 = (b0 & 0x0f) | ((b0 & 0xf0) << 4) | ((b1 & 0x0f) << 16) | ((b1 & 0xf0) << 20);
        sumi = __dp4a(q8, load_int(yq + 4 * i), sumi);
#else
        sumi += (b0 & 0x0f) * (int)yq[4 * i] + (b0 >> 4) * (int)yq[4 * i + 1]
              + (b1 & 0x0f) * (int)yq[4 * i + 2] + (b1 >> 4) * (int)yq[4 * i + 3];
#endif
    }
    return kquant_scale_q4(x, y) * (float)sumi;
}

static __device__ __forceinline__ float vec_dot_q2_k_sub(const block_q2_K *x, const block_q8_1 *y, int sub) {
    (void)x->scales[2 * sub];
    (void)x->scales[2 * sub + 1];
    const uint8_t *qs = x->qs + sub * 8;
    const int8_t *yq = y->qs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int v = qs[i];
#ifndef FUNCSIM_SAFE
        const int q8 = (v & 3) | (((v >> 2) & 3) << 8) | (((v >> 4) & 3) << 16) | (((v >> 6) & 3) << 24);
        sumi = __dp4a(q8, load_int(yq + 4 * i), sumi);
#else
        sumi += (v & 3) * (int)yq[4 * i] + ((v >> 2) & 3) * (int)yq[4 * i + 1]
              + ((v >> 4) & 3) * (int)yq[4 * i + 2] + ((v >> 6) & 3) * (int)yq[4 * i + 3];
#endif
    }
    return kquant_scale_q2(x, y) * (float)sumi;
}

static __device__ __forceinline__ float vec_dot_q3_k_sub(const block_q3_K *x, const block_q8_1 *y, int sub) {
    (void)x->scales[(2 * sub) % 12];
    const uint8_t *qs = x->qs + sub * 8;
    const uint8_t *hm = x->hmask + sub * 4;
    const int8_t *yq = y->qs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int v = qs[i];
        int q[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int widx = i * 4 + b;
            const int hb = (hm[widx >> 3] >> (widx & 7)) & 1;
            q[b] = ((v >> (2 * b)) & 3) | (hb << 2);
        }
#ifndef FUNCSIM_SAFE
        const int q8 = (q[0] & 0xff) | ((q[1] & 0xff) << 8) | ((q[2] & 0xff) << 16) | ((q[3] & 0xff) << 24);
        sumi = __dp4a(q8, load_int(yq + 4 * i), sumi);
#else
        sumi += q[0] * (int)yq[4 * i] + q[1] * (int)yq[4 * i + 1]
              + q[2] * (int)yq[4 * i + 2] + q[3] * (int)yq[4 * i + 3];
#endif
    }
    return kquant_scale_q3(x, y) * (float)sumi;
}

static __host__ __device__ __forceinline__ size_t quant_block_bytes(quant_type q) {
    switch (q) {
        case QUANT_Q4_K: return sizeof(block_q4_K);
        case QUANT_Q2_K: return sizeof(block_q2_K);
        case QUANT_Q3_K: return sizeof(block_q3_K);
        case QUANT_Q8_0: return sizeof(block_q8_0);
    }
    return 0;
}

// Map superblock sb -> device pointer into striped [QxG|RxG] buffer (want_q=1 -> Q, 0 -> R).
static __device__ __forceinline__ const char *strip_block_ptr(const char *row_base, int sb, int G,
                                                             size_t q_blk, size_t r_blk, int want_q) {
    const int strip = sb / G;
    const int local = sb % G;
    const size_t strip_bytes = (size_t)G * q_blk + (size_t)G * r_blk;
    const char *s = row_base + (size_t)strip * strip_bytes;
    if (want_q) return s + (size_t)local * q_blk;
    return s + (size_t)G * q_blk + (size_t)local * r_blk;
}

static __device__ __forceinline__ float dot_k_sub(quant_type q, const char *blk,
                                                 const block_q8_1 *y, int sub) {
    if (q == QUANT_Q4_K) return vec_dot_q4_k_sub((const block_q4_K *)blk, y, sub);
    if (q == QUANT_Q3_K) return vec_dot_q3_k_sub((const block_q3_K *)blk, y, sub);
    return vec_dot_q2_k_sub((const block_q2_K *)blk, y, sub);
}

// mode: 1 = Q only, 2 = Q+R. Q/R are runtime enums (not template) for one binary.
__global__ void mmvq_speed_gemv(const char *__restrict__ vx,
                                const block_q8_1 *__restrict__ vy,
                                float *__restrict__ dst,
                                int K, int N, int G, int mode,
                                quant_type q_ty, quant_type r_ty) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    const size_t q_blk = quant_block_bytes(q_ty);
    const size_t r_blk = quant_block_bytes(r_ty);
    const size_t strip_bytes = (size_t)G * q_blk + (size_t)G * r_blk;
    const int n_strips = K / (G * QK_K);
    const char *row_base = vx + (size_t)row * (size_t)n_strips * strip_bytes;

    const int nsub = K / QK8_1;
    const int sub_per = QK_K / QK8_1; // 8
    float acc = 0.0f;

    for (int is = tid; is < nsub; is += NTHREADS) {
        const int sb = is / sub_per;
        const int sub = is % sub_per;
        const char *qb = strip_block_ptr(row_base, sb, G, q_blk, r_blk, 1);
        acc += dot_k_sub(q_ty, qb, &vy[is], sub);
        if (mode == MODE_SW_REBUILD) {
            const char *rb = strip_block_ptr(row_base, sb, G, q_blk, r_blk, 0);
            acc += dot_k_sub(r_ty, rb, &vy[is], sub);
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

// Mode 3 CUDA stand-in: packed Q8_0 GEMV (same compute as mmvq_kquant Q8). FR effect is sim-only.
__global__ void mmvq_q8_packed(const block_q8_0 *__restrict__ vx,
                               const block_q8_1 *__restrict__ vy,
                               float *__restrict__ dst, int K, int N) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int bpr = K / QK8_0;
    const block_q8_0 *xrow = vx + (size_t)row * bpr;
    float acc = 0.0f;
    for (int kb = tid; kb < bpr; kb += NTHREADS)
        acc += vec_dot_q8_0_q8_1(&xrow[kb], &vy[kb]);
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

__global__ void fill_q8_1(block_q8_1 *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#ifdef FUNCSIM_SAFE
    b[i].d = 1.0f; b[i].s = 32.0f;
#else
    b[i].ds = __floats2half2_rn(1.0f, 32.0f);
#endif
    for (int j = 0; j < QK8_1; ++j) b[i].qs[j] = 1;
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
    for (int j = 0; j < QK_K / 4; ++j) b[i].qs[j] = 0x55;
}

__global__ void fill_q3_k(block_q3_K *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#ifdef FUNCSIM_SAFE
    b[i].d = 1.0f;
#else
    b[i].d = __float2half(1.0f);
#endif
    for (int j = 0; j < QK_K / 8; ++j) b[i].hmask[j] = 0x00;
    for (int j = 0; j < QK_K / 4; ++j) b[i].qs[j] = 0x55;
    for (int j = 0; j < 12; ++j) b[i].scales[j] = 0x11;
}

// Fill every Q (and optionally R) block inside the striped buffer.
__global__ void fill_striped(char *base, int N, int n_strips, int G,
                             size_t q_blk, size_t r_blk,
                             quant_type q_ty, quant_type r_ty, int fill_r) {
    const size_t strip_bytes = (size_t)G * q_blk + (size_t)G * r_blk;
    const size_t n_q = (size_t)N * (size_t)n_strips * (size_t)G;
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_q) {
        const size_t row = tid / ((size_t)n_strips * G);
        const size_t rem = tid % ((size_t)n_strips * G);
        const size_t strip = rem / G;
        const size_t local = rem % G;
        char *p = base + row * (size_t)n_strips * strip_bytes + strip * strip_bytes + local * q_blk;
        if (q_ty == QUANT_Q4_K) {
            block_q4_K *b = (block_q4_K *)p;
#ifdef FUNCSIM_SAFE
            b->d = 1.0f; b->dmin = 0.0f;
#else
            b->dm = __floats2half2_rn(1.0f, 0.0f);
#endif
            for (int j = 0; j < K_SCALE_SIZE; ++j) b->scales[j] = 1;
            for (int j = 0; j < QK_K / 2; ++j) b->qs[j] = 0x11;
        } else if (q_ty == QUANT_Q3_K) {
            block_q3_K *b = (block_q3_K *)p;
#ifdef FUNCSIM_SAFE
            b->d = 1.0f;
#else
            b->d = __float2half(1.0f);
#endif
            for (int j = 0; j < QK_K / 8; ++j) b->hmask[j] = 0;
            for (int j = 0; j < QK_K / 4; ++j) b->qs[j] = 0x55;
            for (int j = 0; j < 12; ++j) b->scales[j] = 0x11;
        } else {
            block_q2_K *b = (block_q2_K *)p;
#ifdef FUNCSIM_SAFE
            b->d = 1.0f; b->dmin = 0.0f;
#else
            b->dm = __floats2half2_rn(1.0f, 0.0f);
#endif
            for (int j = 0; j < QK_K / 16; ++j) b->scales[j] = 0x11;
            for (int j = 0; j < QK_K / 4; ++j) b->qs[j] = 0x55;
        }
    }
    if (!fill_r) return;
    if (tid < n_q) {
        const size_t row = tid / ((size_t)n_strips * G);
        const size_t rem = tid % ((size_t)n_strips * G);
        const size_t strip = rem / G;
        const size_t local = rem % G;
        char *p = base + row * (size_t)n_strips * strip_bytes + strip * strip_bytes
                + (size_t)G * q_blk + local * r_blk;
        if (r_ty == QUANT_Q4_K) {
            block_q4_K *b = (block_q4_K *)p;
#ifdef FUNCSIM_SAFE
            b->d = 1.0f; b->dmin = 0.0f;
#else
            b->dm = __floats2half2_rn(1.0f, 0.0f);
#endif
            for (int j = 0; j < K_SCALE_SIZE; ++j) b->scales[j] = 1;
            for (int j = 0; j < QK_K / 2; ++j) b->qs[j] = 0x11;
        } else if (r_ty == QUANT_Q3_K) {
            block_q3_K *b = (block_q3_K *)p;
#ifdef FUNCSIM_SAFE
            b->d = 1.0f;
#else
            b->d = __float2half(1.0f);
#endif
            for (int j = 0; j < QK_K / 8; ++j) b->hmask[j] = 0;
            for (int j = 0; j < QK_K / 4; ++j) b->qs[j] = 0x55;
            for (int j = 0; j < 12; ++j) b->scales[j] = 0x11;
        } else {
            block_q2_K *b = (block_q2_K *)p;
#ifdef FUNCSIM_SAFE
            b->d = 1.0f; b->dmin = 0.0f;
#else
            b->dm = __floats2half2_rn(1.0f, 0.0f);
#endif
            for (int j = 0; j < QK_K / 16; ++j) b->scales[j] = 0x11;
            for (int j = 0; j < QK_K / 4; ++j) b->qs[j] = 0x55;
        }
    }
}

static const char *quant_name(quant_type q) {
    switch (q) {
        case QUANT_Q4_K: return "q4_k";
        case QUANT_Q2_K: return "q2_k";
        case QUANT_Q3_K: return "q3_k";
        case QUANT_Q8_0: return "q8_0";
    }
    return "?";
}

static int parse_quant_k(const char *s, quant_type *q) {
    if (!s) return -1;
    if (strcmp(s, "q4_k") == 0 || strcmp(s, "Q4_K") == 0) { *q = QUANT_Q4_K; return 0; }
    if (strcmp(s, "q2_k") == 0 || strcmp(s, "Q2_K") == 0) { *q = QUANT_Q2_K; return 0; }
    if (strcmp(s, "q3_k") == 0 || strcmp(s, "Q3_K") == 0) { *q = QUANT_Q3_K; return 0; }
    return -1;
}

static bool is_all_digits(const char *s) {
    if (!s || !*s) return false;
    for (; *s; ++s) if (*s < '0' || *s > '9') return false;
    return true;
}

static size_t striped_bytes(int K, int N, int G, quant_type q, quant_type r) {
    const int n_strips = K / (G * QK_K);
    const size_t strip = (size_t)G * quant_block_bytes(q) + (size_t)G * quant_block_bytes(r);
    return (size_t)N * (size_t)n_strips * strip;
}

static size_t q_only_touch_bytes(int K, int N, int G, quant_type q) {
    // Bytes of Q payload only (what mode 1 actually reads), excluding R gaps.
    (void)G;
    return (size_t)N * (size_t)(K / QK_K) * quant_block_bytes(q);
}

static int run_one(int mode, int G, quant_type q_ty, quant_type r_ty,
                   int K, int N, const char *label, bool verify, bool timeit) {
    if (G < 1) {
        printf("G=%d must be >= 1\n", G);
        return 1;
    }
    if (N <= 0) {
        printf("N=%d must be positive\n", N);
        return 1;
    }
    if (mode != MODE_HW_FR && K % (G * QK_K) != 0) {
        printf("K=%d must be divisible by G*256=%d\n", K, G * QK_K);
        return 1;
    }
    if (mode == MODE_HW_FR && K % QK8_0 != 0) {
        printf("K=%d must be divisible by 32 for mode-3 Q8\n", K);
        return 1;
    }

    const bool skip_fill = getenv("MMVQ_SKIP_FILL") != nullptr;
    const size_t n_yblocks = (size_t)(K / QK8_0);
    block_q8_1 *d_y = nullptr;
    float *d_dst = nullptr;
    CHECK(cudaMalloc(&d_y, n_yblocks * sizeof(block_q8_1)));
    CHECK(cudaMalloc(&d_dst, (size_t)N * sizeof(float)));
    if (!skip_fill) fill_q8_1<<<(n_yblocks + 255) / 256, 256>>>(d_y, n_yblocks);

    dim3 block(WARP_SIZE, NWARPS);
    dim3 grid(N);

    if (mode == MODE_HW_FR) {
        // CUDA stand-in only: packed Q8. Real SPEED effect needs sim FR + Q||R DRAM.
        printf("  [mode 3] CUDA runs packed Q8_0 GEMV (FR rebuild is simulator-only; "
               "see DESIGN.md). Compare traffic under sim, not this host timer alone.\n");
        const size_t n = (size_t)N * (K / QK8_0);
        block_q8_0 *d_w = nullptr;
        CHECK(mmvq_malloc_weight((void **)&d_w, n * sizeof(block_q8_0)));
        if (!skip_fill) fill_q8_0<<<(n + 255) / 256, 256>>>(d_w, n);
        CHECK(cudaGetLastError());

        auto launch = [&]() { mmvq_q8_packed<<<grid, block>>>(d_w, d_y, d_dst, K, N); };
        launch();
        CHECK(cudaDeviceSynchronize());

        if (verify && !skip_fill) {
            float h0 = 0.0f;
            CHECK(cudaMemcpy(&h0, d_dst, sizeof(float), cudaMemcpyDeviceToHost));
            const float expect = (float)K;
            const bool ok = fabsf(h0 - expect) < 0.5f;
            printf("  [verify] dst[0]=%.1f expect=%.1f %s  (mode3-Q8 %s K=%d N=%d)\n",
                   h0, expect, ok ? "OK" : "FAIL", label ? label : "-", K, N);
            if (!ok) { cudaFree(d_w); cudaFree(d_y); cudaFree(d_dst); return 1; }
        }

        if (timeit) {
            const int iters = 20;
            cudaEvent_t t0, t1;
            CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
            CHECK(cudaEventRecord(t0));
            for (int it = 0; it < iters; ++it) launch();
            CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            const double us = ms * 1e3 / iters;
            const double w_bytes = (double)n * sizeof(block_q8_0);
            printf("  %-10s mode3-Q8  K=%-6d N=%-6d  W=%7.2f MB  %8.2f us  %7.1f GB/s\n",
                   label ? label : "-", K, N, w_bytes / 1e6, us, w_bytes / (us * 1e-6) / 1e9);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }
        cudaFree(d_w); cudaFree(d_y); cudaFree(d_dst);
        return 0;
    }

    // Modes 1 / 2: striped Q||R buffer
    const int n_strips = K / (G * QK_K);
    const size_t q_blk = quant_block_bytes(q_ty);
    const size_t r_blk = quant_block_bytes(r_ty);
    const size_t w_bytes = striped_bytes(K, N, G, q_ty, r_ty);
    const size_t q_touch = q_only_touch_bytes(K, N, G, q_ty);
    const size_t r_touch = (size_t)N * (size_t)(K / QK_K) * quant_block_bytes(r_ty);

    char *d_w = nullptr;
    CHECK(mmvq_malloc_weight((void **)&d_w, w_bytes));
    if (!skip_fill) {
        const size_t n_q = (size_t)N * (size_t)n_strips * (size_t)G;
        fill_striped<<<(unsigned)((n_q + 255) / 256), 256>>>(
            d_w, N, n_strips, G, q_blk, r_blk, q_ty, r_ty, /*fill_r=*/1);
    }
    CHECK(cudaGetLastError());

    auto launch = [&]() {
        mmvq_speed_gemv<<<grid, block>>>(d_w, d_y, d_dst, K, N, G, mode, q_ty, r_ty);
    };
    launch();
    CHECK(cudaDeviceSynchronize());

    if (verify && !skip_fill) {
        float h0 = 0.0f;
        CHECK(cudaMemcpy(&h0, d_dst, sizeof(float), cudaMemcpyDeviceToHost));
        const float expect = (mode == MODE_SW_REBUILD) ? (float)(2 * K) : (float)K;
        const bool ok = fabsf(h0 - expect) < 0.5f;
        printf("  [verify] dst[0]=%.1f expect=%.1f %s  (mode%d G=%d %s+%s %s K=%d N=%d "
               "strip=%.2f MB q_touch=%.2f MB)\n",
               h0, expect, ok ? "OK" : "FAIL",
               mode, G, quant_name(q_ty), quant_name(r_ty), label ? label : "-", K, N,
               w_bytes / 1e6, q_touch / 1e6);
        if (!ok) { cudaFree(d_w); cudaFree(d_y); cudaFree(d_dst); return 1; }
    } else if (verify && skip_fill) {
        printf("  [verify] skipped (MMVQ_SKIP_FILL)  mode%d G=%d %s+%s K=%d N=%d\n",
               mode, G, quant_name(q_ty), quant_name(r_ty), K, N);
    }

    if (timeit) {
        const int iters = 20;
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        CHECK(cudaEventRecord(t0));
        for (int it = 0; it < iters; ++it) launch();
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        const double us = ms * 1e3 / iters;
        const double touch = (mode == MODE_SW_REBUILD)
            ? (double)(q_touch + r_touch) : (double)q_touch;
        printf("  %-10s mode%d G=%-2d %-5s+%-5s K=%-6d N=%-6d  "
               "alloc=%7.2f MB touch=%7.2f MB  %8.2f us  %7.1f GB/s\n",
               label ? label : "-", mode, G, quant_name(q_ty), quant_name(r_ty),
               K, N, w_bytes / 1e6, touch / 1e6, us, touch / (us * 1e-6) / 1e9);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    } else {
        printf(">>> launch mode%d G=%d %s+%s K=%d N=%d alloc=%.3f MB\n",
               mode, G, quant_name(q_ty), quant_name(r_ty), K, N, w_bytes / 1e6);
    }

    cudaFree(d_w); cudaFree(d_y); cudaFree(d_dst);
    return 0;
}

static void usage(const char *argv0) {
    printf("Usage:\n");
    printf("  %s <mode> <G> <q> <r> <K> <N>\n", argv0);
    printf("  %s <mode> <G> <q> <r> <8b|14b> <op|all>\n", argv0);
    printf("  mode: 1=Q-only  2=SW Q+R rebuild  3=Q8 stand-in (FR=sim-only)\n");
    printf("  G: >=1 (e.g. 1,2,4,8,16,32,64)  K %% (G*256)==0;  q,r: q2_k|q3_k|q4_k\n");
    printf("Compare packed baseline: ../mmvq_kquant/mmvq_kquant <K> <N> <q>\n");
    printf("Error+speed harness:     make test_compare && ./test_compare ...\n");
}

#ifndef MMVQ_NO_MAIN
int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Device: %s  (SPEED Q||R strip GEMV — modes 1/2 CUDA, 3=Q8 stand-in)\n", p.name);
#ifdef FUNCSIM_SAFE
    printf("Build: FUNCSIM_SAFE (execution-driven / GPGPU-Sim)\n");
#endif

    if (argc < 7) { usage(argv[0]); return 1; }

    const int mode = atoi(argv[1]);
    const int G = atoi(argv[2]);
    if (mode < 1 || mode > 3) { printf("mode must be 1, 2, or 3\n"); return 1; }

    quant_type q_ty = QUANT_Q4_K, r_ty = QUANT_Q2_K;
    if (mode != MODE_HW_FR) {
        if (parse_quant_k(argv[3], &q_ty) != 0) { printf("Unknown q: %s\n", argv[3]); return 1; }
        if (parse_quant_k(argv[4], &r_ty) != 0) { printf("Unknown r: %s\n", argv[4]); return 1; }
    }

    const bool timeit = !getenv("MMVQ_NO_TIME");

    if (getenv("MMVQ_TINY")) {
        return run_one(mode, G, q_ty, r_ty, 256, 64, "tiny", true, timeit);
    }

    if (is_all_digits(argv[5]) && is_all_digits(argv[6])) {
        return run_one(mode, G, q_ty, r_ty, atoi(argv[5]), atoi(argv[6]), "custom", true, timeit);
    }

    const char *model = argv[5];
    const char *op = argv[6];
    int n_shapes = 0;
    const gemv_shape *tbl = gemv_shapes_for_model(model, &n_shapes);

    if (strcmp(op, "all") == 0) {
        printf("[model=%s mode=%d G=%d q=%s r=%s — %d shapes]\n",
               model, mode, G, quant_name(q_ty), quant_name(r_ty), n_shapes);
        for (int i = 0; i < n_shapes; ++i) {
            if (tbl[i].K % (G * QK_K) != 0) {
                printf("  skip %s K=%d (not divisible by G*256=%d)\n",
                       tbl[i].name, tbl[i].K, G * QK_K);
                continue;
            }
            if (run_one(mode, G, q_ty, r_ty, tbl[i].K, tbl[i].N, tbl[i].name, false, timeit))
                return 1;
        }
        return 0;
    }

    int K = 0, N = 0;
    if (gemv_shape_lookup(tbl, n_shapes, op, &K, &N) != 0) {
        printf("Unknown op '%s' for model %s\n", op, model);
        return 1;
    }
    return run_one(mode, G, q_ty, r_ty, K, N, op, true, timeit);
}
#endif // MMVQ_NO_MAIN
