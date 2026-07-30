// Decode GEMV / batched multi-vector: weight [N x K] x activation [K x B] -> dst[N x B].
// Supports real ggml block layouts for Q8_0, Q4_K, Q3_K/Q3_K_M, Q2_K
// (llama.cpp decode path).
//
// Dispatch mirrors llama.cpp ggml-cuda (mmvq.cuh / mmq.cuh):
//   B=1              -> mul_mat_vec_q GEMV
//   2..MMVQ_MAX_BATCH -> MMVQ (weight reused across ncols in registers)
//   B>MMVQ_MAX_BATCH  -> MMQ tiled smem (Q8); real llama uses tensor-core MMA here
//
// Build:  make            (sm_90, dp4a/fp16)
//         make exec       (sm_70 + PTX, FUNCSIM_SAFE for GPGPU-Sim execution-driven)
// Run:    ./mmvq_kquant 14b q_proj q4_k
//         ./mmvq_kquant 8b all q3_k_m
//         ./mmvq_kquant 4096 5120 q8_0 8   (explicit K N quant batch)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>
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
#define MAX_BATCH 256
#define BATCH_CHUNK 64
// llama.cpp ggml-cuda/mmvq.cuh: MMVQ_MAX_BATCH_SIZE = 8. Above that they switch to
// mul_mat_q (shared-memory tiles + tensor-core MMA). We mirror that split.
#define MMVQ_MAX_BATCH 8
#define MMQ_BM 32
#define MMQ_BN 64
#define MMQ_WPB (QK8_0 / 4)
#define MMQ_THREADS 256
#define MMQ_OUTS ((MMQ_BM * MMQ_BN) / MMQ_THREADS) // 8
// dp4a MMQ is slower than MMVQ for small B; switch once the flat tile curve wins.
#define MMQ_SWITCH_BATCH 32

// --- ggml block layouts (FUNCSIM_SAFE uses f32 instead of fp16) -----------------
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

enum quant_type { QUANT_Q8_0 = 0, QUANT_Q4_K = 1, QUANT_Q2_K = 2, QUANT_Q3_K = 3 };

#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

#ifndef FUNCSIM_SAFE
// qs in block_q8_0 sits at offset 2 (after half d) — 2-byte aligned, not 4.
// A raw int* load traps as "misaligned address" on H100; assemble from two u16s
// (same approach as ggml's get_int_b2). y->qs is 4-byte aligned after half2.
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
    for (int i = 0; i < QK8_0 / 4; ++i) {
        sumi = __dp4a(load_int(x->qs + 4 * i), load_int(y->qs + 4 * i), sumi);
    }
    return block_scale_q8_0(x) * block_scale_q8_1(y) * (float)sumi;
#endif
}

// Touch scale bytes so they count in the working set (ggml block traffic).
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

// Q4_K / Q2_K: one 32-wide sub per call. Parallelize over K/32 subs (same as Q8_0) so all
// threads stay busy; the parent super-block is cache-shared across the 8 subs.
static __device__ __forceinline__ float vec_dot_q4_k_sub(const block_q4_K *x, const block_q8_1 *y, int sub) {
    touch_scales(x->scales, K_SCALE_SIZE);
    const uint8_t *qs = x->qs + sub * 16;
    const int8_t *yq = y->qs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int b0 = qs[2 * i];
        const int b1 = qs[2 * i + 1];
#ifndef FUNCSIM_SAFE
        const int q8 = (b0 & 0x0f) | ((b0 & 0xf0) << 4) | ((b1 & 0x0f) << 16) | ((b1 & 0xf0) << 20);
        sumi = __dp4a(q8, load_int(yq + 4 * i), sumi);
#else
        sumi += (b0 & 0x0f) * (int)yq[4 * i];
        sumi += (b0 >> 4) * (int)yq[4 * i + 1];
        sumi += (b1 & 0x0f) * (int)yq[4 * i + 2];
        sumi += (b1 >> 4) * (int)yq[4 * i + 3];
#endif
    }
    return kquant_scale_q4(x, y) * (float)sumi;
}

static __device__ __forceinline__ float vec_dot_q2_k_sub(const block_q2_K *x, const block_q8_1 *y, int sub) {
    // 16 groups of 16 weights; this sub covers groups [2*sub, 2*sub+2).
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
        sumi += (v & 3) * (int)yq[4 * i];
        sumi += ((v >> 2) & 3) * (int)yq[4 * i + 1];
        sumi += ((v >> 4) & 3) * (int)yq[4 * i + 2];
        sumi += ((v >> 6) & 3) * (int)yq[4 * i + 3];
#endif
    }
    return kquant_scale_q2(x, y) * (float)sumi;
}

// Q3_K: each 3-bit weight = 2 low bits (from qs) | 1 high bit (from hmask). One 32-wide sub per
// call. qs: 8 bytes/sub (4 weights/byte); hmask: 4 bytes/sub (8 weights/byte).
static __device__ __forceinline__ float vec_dot_q3_k_sub(const block_q3_K *x, const block_q8_1 *y, int sub) {
    (void)x->scales[(2 * sub) % 12];
    const uint8_t *qs = x->qs + sub * 8;
    const uint8_t *hm = x->hmask + sub * 4;
    const int8_t *yq = y->qs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {          // 8 bytes of qs = 32 weights
        const int v = qs[i];
        int q[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int widx = i * 4 + b;    // 0..31 within the sub
            const int hb = (hm[widx >> 3] >> (widx & 7)) & 1;
            q[b] = ((v >> (2 * b)) & 3) | (hb << 2);   // 3-bit value 0..7
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

static __device__ __forceinline__ void reduce_store_row(float acc, float *__restrict__ dst,
                                                       int tid) {
#ifdef FUNCSIM_SAFE
    __shared__ float sm[NTHREADS];
    sm[tid] = acc;
    __syncthreads();
    for (int s = NTHREADS / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) *dst = sm[0];
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
        *dst = s;
    }
#endif
}

// ncols-specialized reducer.
template <int ncols>
static __device__ __forceinline__ void reduce_store_ncols(float acc[ncols], int nb,
                                                         float *__restrict__ dst_row,
                                                         int tid) {
#ifdef FUNCSIM_SAFE
    // Column-by-column to keep smem bounded (ncols may be 64).
    for (int bi = 0; bi < nb; ++bi)
        reduce_store_row(acc[bi], &dst_row[bi], tid);
#else
    __shared__ float warp_sums[NWARPS][ncols];
#pragma unroll
    for (int bi = 0; bi < ncols; ++bi) {
        float a = (bi < nb) ? acc[bi] : 0.0f;
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, off);
        if (threadIdx.x == 0) warp_sums[threadIdx.y][bi] = a;
    }
    __syncthreads();
    if (tid < nb) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) s += warp_sums[w][tid];
        dst_row[tid] = s;
    }
    __syncthreads();
#endif
}

template <quant_type Q>
__global__ void mmvq_gemv_single(const void *__restrict__ vx,
                                 const block_q8_1 *__restrict__ vy,
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
            acc += vec_dot_q8_0_q8_1(&xrow[kb], &vy[kb]);
        }
    } else {
        const int nsub = K / QK8_1;
        const int sub_per = QK_K / QK8_1; // 8
        if (Q == QUANT_Q4_K) {
            const block_q4_K *xrow = (const block_q4_K *)vx + (size_t)row * (K / QK_K);
            for (int is = tid; is < nsub; is += nthreads) {
                acc += vec_dot_q4_k_sub(&xrow[is / sub_per], &vy[is], is % sub_per);
            }
        } else if (Q == QUANT_Q3_K) {
            const block_q3_K *xrow = (const block_q3_K *)vx + (size_t)row * (K / QK_K);
            for (int is = tid; is < nsub; is += nthreads) {
                acc += vec_dot_q3_k_sub(&xrow[is / sub_per], &vy[is], is % sub_per);
            }
        } else {
            const block_q2_K *xrow = (const block_q2_K *)vx + (size_t)row * (K / QK_K);
            for (int is = tid; is < nsub; is += nthreads) {
                acc += vec_dot_q2_k_sub(&xrow[is / sub_per], &vy[is], is % sub_per);
            }
        }
    }

    reduce_store_row(acc, &dst[row], tid);
}

// Same CTA mapping as GEMV (one block per output row). Each weight block is loaded
// once from DRAM and reused across ncols tokens in registers.
// ncols is a compile-time constant (2/4/8/16) so the compiler keeps occupancy high.
template <quant_type Q, int ncols>
__global__ void __launch_bounds__(NTHREADS, 4)
mmvq_gemv_ncols(const void *__restrict__ vx,
                const block_q8_1 *__restrict__ vy,
                float *__restrict__ dst, int K, int N, int batch) {
    static_assert(ncols >= 1 && ncols <= BATCH_CHUNK, "ncols");
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int nthreads = NTHREADS;

    for (int b0 = 0; b0 < batch; b0 += ncols) {
        const int nb = batch - b0 < ncols ? batch - b0 : ncols;
        float acc[ncols];
#pragma unroll
        for (int i = 0; i < ncols; ++i) acc[i] = 0.0f;

        if (Q == QUANT_Q8_0) {
            const int bpr = K / QK8_0;
            const block_q8_0 *xrow = (const block_q8_0 *)vx + (size_t)row * bpr;
            for (int kb = tid; kb < bpr; kb += nthreads) {
                int wq[QK8_0 / 4];
#pragma unroll
                for (int i = 0; i < QK8_0 / 4; ++i) {
#ifdef FUNCSIM_SAFE
                    wq[i] = reinterpret_cast<const int *>(xrow[kb].qs)[i];
#else
                    wq[i] = load_int(xrow[kb].qs + 4 * i);
#endif
                }
                const float wd = block_scale_q8_0(&xrow[kb]);
#pragma unroll
                for (int bi = 0; bi < ncols; ++bi) {
                    if (bi >= nb) continue;
                    const block_q8_1 *yb = &vy[(size_t)(b0 + bi) * bpr + kb];
                    int sumi = 0;
#ifdef FUNCSIM_SAFE
#pragma unroll
                    for (int i = 0; i < QK8_0 / 4; ++i) {
                        const int yq = reinterpret_cast<const int *>(yb->qs)[i];
                        const int8_t *wa = reinterpret_cast<const int8_t *>(&wq[i]);
                        const int8_t *ya = reinterpret_cast<const int8_t *>(&yq);
                        sumi += (int)wa[0] * (int)ya[0] + (int)wa[1] * (int)ya[1]
                              + (int)wa[2] * (int)ya[2] + (int)wa[3] * (int)ya[3];
                    }
#else
#pragma unroll
                    for (int i = 0; i < QK8_0 / 4; ++i)
                        sumi = __dp4a(wq[i], reinterpret_cast<const int *>(yb->qs)[i], sumi);
#endif
                    acc[bi] += wd * block_scale_q8_1(yb) * (float)sumi;
                }
            }
        } else {
            const int nsub = K / QK8_1;
            const int sub_per = QK_K / QK8_1;
            if (Q == QUANT_Q4_K) {
                const block_q4_K *xrow = (const block_q4_K *)vx + (size_t)row * (K / QK_K);
                for (int is = tid; is < nsub; is += nthreads) {
                    const block_q4_K *w = &xrow[is / sub_per];
                    const int sub = is % sub_per;
#pragma unroll
                    for (int bi = 0; bi < ncols; ++bi) {
                        if (bi < nb)
                            acc[bi] += vec_dot_q4_k_sub(w, &vy[(size_t)(b0 + bi) * nsub + is], sub);
                    }
                }
            } else if (Q == QUANT_Q3_K) {
                const block_q3_K *xrow = (const block_q3_K *)vx + (size_t)row * (K / QK_K);
                for (int is = tid; is < nsub; is += nthreads) {
                    const block_q3_K *w = &xrow[is / sub_per];
                    const int sub = is % sub_per;
#pragma unroll
                    for (int bi = 0; bi < ncols; ++bi) {
                        if (bi < nb)
                            acc[bi] += vec_dot_q3_k_sub(w, &vy[(size_t)(b0 + bi) * nsub + is], sub);
                    }
                }
            } else {
                const block_q2_K *xrow = (const block_q2_K *)vx + (size_t)row * (K / QK_K);
                for (int is = tid; is < nsub; is += nthreads) {
                    const block_q2_K *w = &xrow[is / sub_per];
                    const int sub = is % sub_per;
#pragma unroll
                    for (int bi = 0; bi < ncols; ++bi) {
                        if (bi < nb)
                            acc[bi] += vec_dot_q2_k_sub(w, &vy[(size_t)(b0 + bi) * nsub + is], sub);
                    }
                }
            }
        }

        reduce_store_ncols<ncols>(acc, nb, &dst[(size_t)row * batch + b0], tid);
    }
}

// llama-style mul_mat_q (dp4a fallback, no MMA): BM x BN output tile, W/Y staged in
// shared memory with +1 pad on W K-dim to reduce bank conflicts (mmq.cuh pattern).
// Used when batch > MMVQ_MAX_BATCH. Real llama prefers tensor-core MMA here.
__global__ void __launch_bounds__(MMQ_THREADS, 4)
mmq_q8_tiled(const block_q8_0 *__restrict__ vx,
             const block_q8_1 *__restrict__ vy,
             float *__restrict__ dst, int K, int N, int batch) {
    const int bpr = K / QK8_0;
    const int row0 = blockIdx.x * MMQ_BM;
    const int tok0 = blockIdx.y * MMQ_BN;
    const int tid = threadIdx.x;

    // sW: BM rows x (WPB+1) ints — trailing pad avoids 32-way bank conflicts on column reads
    __shared__ int   sW[MMQ_BM][MMQ_WPB + 1];
    __shared__ float sWs[MMQ_BM];
    __shared__ int   sX[MMQ_BN][MMQ_WPB];
    __shared__ float sXs[MMQ_BN];

    int oi[MMQ_OUTS], oj[MMQ_OUTS];
    float acc[MMQ_OUTS];
#pragma unroll
    for (int t = 0; t < MMQ_OUTS; ++t) {
        const int idx = tid + t * MMQ_THREADS;
        oi[t] = idx / MMQ_BN;
        oj[t] = idx % MMQ_BN;
        acc[t] = 0.0f;
    }

    for (int kb = 0; kb < bpr; ++kb) {
        for (int t = tid; t < MMQ_BM * (MMQ_WPB + 1); t += MMQ_THREADS) {
            const int r = t / (MMQ_WPB + 1);
            const int w = t % (MMQ_WPB + 1);
            const int gr = row0 + r;
            if (w < MMQ_WPB && gr < N) {
#ifdef FUNCSIM_SAFE
                sW[r][w] = reinterpret_cast<const int *>(vx[(size_t)gr * bpr + kb].qs)[w];
#else
                sW[r][w] = load_int(vx[(size_t)gr * bpr + kb].qs + 4 * w);
#endif
            } else {
                sW[r][w] = 0;
            }
        }
        if (tid < MMQ_BM) {
            const int gr = row0 + tid;
            sWs[tid] = (gr < N) ? block_scale_q8_0(&vx[(size_t)gr * bpr + kb]) : 0.0f;
        }
        for (int t = tid; t < MMQ_BN * MMQ_WPB; t += MMQ_THREADS) {
            const int r = t / MMQ_WPB;
            const int w = t % MMQ_WPB;
            const int gt = tok0 + r;
            if (gt < batch) {
#ifdef FUNCSIM_SAFE
                sX[r][w] = reinterpret_cast<const int *>(vy[(size_t)gt * bpr + kb].qs)[w];
#else
                sX[r][w] = reinterpret_cast<const int *>(vy[(size_t)gt * bpr + kb].qs)[w];
#endif
            } else {
                sX[r][w] = 0;
            }
        }
        if (tid < MMQ_BN) {
            const int gt = tok0 + tid;
            sXs[tid] = (gt < batch) ? block_scale_q8_1(&vy[(size_t)gt * bpr + kb]) : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < MMQ_OUTS; ++t) {
            const int i = oi[t];
            const int j = oj[t];
            int sumi = 0;
#ifdef FUNCSIM_SAFE
#pragma unroll
            for (int w = 0; w < MMQ_WPB; ++w) {
                const int8_t *wa = reinterpret_cast<const int8_t *>(&sW[i][w]);
                const int8_t *ya = reinterpret_cast<const int8_t *>(&sX[j][w]);
                sumi += (int)wa[0] * (int)ya[0] + (int)wa[1] * (int)ya[1]
                      + (int)wa[2] * (int)ya[2] + (int)wa[3] * (int)ya[3];
            }
#else
#pragma unroll
            for (int w = 0; w < MMQ_WPB; ++w)
                sumi = __dp4a(sW[i][w], sX[j][w], sumi);
#endif
            acc[t] += sWs[i] * sXs[j] * (float)sumi;
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < MMQ_OUTS; ++t) {
        const int row = row0 + oi[t];
        const int tok = tok0 + oj[t];
        if (row < N && tok < batch)
            dst[(size_t)row * batch + tok] = acc[t];
    }
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
    // 0x55 = 0b01010101 → every 2-bit field is 1 (all weights = 1 for verify)
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
    // all weights = 1: low2=01 (qs 0x55), high bit = 0 (hmask 0x00)
    for (int j = 0; j < QK_K / 8; ++j) b[i].hmask[j] = 0x00;
    for (int j = 0; j < QK_K / 4; ++j) b[i].qs[j] = 0x55;
    for (int j = 0; j < 12; ++j) b[i].scales[j] = 0x11;
}

static size_t weight_bytes(quant_type q, int K, int N) {
    if (q == QUANT_Q8_0) {
        return (size_t)N * (K / QK8_0) * sizeof(block_q8_0);
    }
    size_t blk = (q == QUANT_Q4_K) ? sizeof(block_q4_K)
               : (q == QUANT_Q3_K) ? sizeof(block_q3_K)
                                    : sizeof(block_q2_K);
    return (size_t)N * (K / QK_K) * blk;
}

static const char *quant_name(quant_type q) {
    switch (q) {
        case QUANT_Q8_0: return "q8_0";
        case QUANT_Q4_K: return "q4_k";
        case QUANT_Q3_K: return "q3_k_m";
        case QUANT_Q2_K: return "q2_k";
    }
    return "?";
}

static int parse_quant(const char *s, quant_type *q) {
    if (!s) return -1;
    if (strcmp(s, "q8_0") == 0 || strcmp(s, "Q8_0") == 0) { *q = QUANT_Q8_0; return 0; }
    if (strcmp(s, "q4_k") == 0 || strcmp(s, "Q4_K") == 0) { *q = QUANT_Q4_K; return 0; }
    if (strcmp(s, "q3_k") == 0 || strcmp(s, "Q3_K") == 0 ||
        strcmp(s, "q3_k_m") == 0 || strcmp(s, "Q3_K_M") == 0) {
        *q = QUANT_Q3_K;
        return 0;
    }
    if (strcmp(s, "q2_k") == 0 || strcmp(s, "Q2_K") == 0) { *q = QUANT_Q2_K; return 0; }
    return -1;
}

static bool is_all_digits(const char *s) {
    if (!s || !*s) return false;
    for (; *s; ++s) {
        if (*s < '0' || *s > '9') return false;
    }
    return true;
}

static int parse_batch_arg(const char *s) {
    if (!s || !*s) return 1;
    const int b = atoi(s);
    return b > 0 ? b : 1;
}

static int run_one(int K, int N, int batch, quant_type q, const char *label, bool verify, bool timeit) {
    const int k_align = (q == QUANT_Q8_0) ? QK8_0 : QK_K;
    if (K % k_align != 0) {
        printf("K=%d must be divisible by %d for %s\n", K, k_align, quant_name(q));
        return 1;
    }
    if (N <= 0) {
        printf("N=%d must be positive\n", N);
        return 1;
    }
    if (batch <= 0 || batch > MAX_BATCH) {
        printf("batch=%d must be in [1, %d]\n", batch, MAX_BATCH);
        return 1;
    }

    const size_t w_bytes = weight_bytes(q, K, N);
    const size_t n_yblocks = (size_t)(K / QK8_0);
    const bool skip_fill = getenv("MMVQ_SKIP_FILL") != nullptr;

    void *d_w = nullptr;
    block_q8_1 *d_y = nullptr;
    float *d_dst = nullptr;

    if (q == QUANT_Q8_0) {
        const size_t n = (size_t)N * (K / QK8_0);
        CHECK(cudaMalloc(&d_w, n * sizeof(block_q8_0)));
        if (!skip_fill) fill_q8_0<<<(n + 255) / 256, 256>>>((block_q8_0 *)d_w, n);
    } else if (q == QUANT_Q4_K) {
        const size_t n = (size_t)N * (K / QK_K);
        CHECK(cudaMalloc(&d_w, n * sizeof(block_q4_K)));
        if (!skip_fill) fill_q4_k<<<(n + 255) / 256, 256>>>((block_q4_K *)d_w, n);
    } else if (q == QUANT_Q3_K) {
        const size_t n = (size_t)N * (K / QK_K);
        CHECK(cudaMalloc(&d_w, n * sizeof(block_q3_K)));
        if (!skip_fill) fill_q3_k<<<(n + 255) / 256, 256>>>((block_q3_K *)d_w, n);
    } else {
        const size_t n = (size_t)N * (K / QK_K);
        CHECK(cudaMalloc(&d_w, n * sizeof(block_q2_K)));
        if (!skip_fill) fill_q2_k<<<(n + 255) / 256, 256>>>((block_q2_K *)d_w, n);
    }

    const size_t y_blocks_total = (size_t)batch * n_yblocks;
    CHECK(cudaMalloc(&d_y, y_blocks_total * sizeof(block_q8_1)));
    CHECK(cudaMalloc(&d_dst, (size_t)N * batch * sizeof(float)));
    if (!skip_fill) fill_q8_1<<<(y_blocks_total + 255) / 256, 256>>>(d_y, y_blocks_total);
    CHECK(cudaGetLastError());

    // Path selection (llama.cpp ggml-cuda.cu):
    //   B == 1              -> GEMV
    //   2 <= B <= MMVQ_MAX  -> MMVQ (register ncols, same CTA map as GEMV)
    //   B >  MMVQ_MAX       -> MMQ  (smem tiles) for Q8_0; MMVQ-chunked for K-quants
    // Override: MMVQ_FORCE_MULTI=1 keeps register-multi for all B>1.
    const bool force_multi = getenv("MMVQ_FORCE_MULTI") != nullptr;
    const bool use_mmq = (!force_multi && q == QUANT_Q8_0 && batch > MMQ_SWITCH_BATCH);

    auto launch_multi = [&](quant_type qq) {
        dim3 block(WARP_SIZE, NWARPS);
        dim3 grid(N);
        auto go = [&](auto qtag) {
            using QT = decltype(qtag);
            constexpr quant_type QV = QT::value;
            if (batch <= 2) {
                mmvq_gemv_ncols<QV, 2><<<grid, block>>>(d_w, d_y, d_dst, K, N, batch);
            } else if (batch <= 4) {
                mmvq_gemv_ncols<QV, 4><<<grid, block>>>(d_w, d_y, d_dst, K, N, batch);
            } else if (batch <= 8) {
                mmvq_gemv_ncols<QV, 8><<<grid, block>>>(d_w, d_y, d_dst, K, N, batch);
            } else if (batch <= 16) {
                mmvq_gemv_ncols<QV, 16><<<grid, block>>>(d_w, d_y, d_dst, K, N, batch);
            } else if (batch <= 32) {
                mmvq_gemv_ncols<QV, 32><<<grid, block>>>(d_w, d_y, d_dst, K, N, batch);
            } else {
                mmvq_gemv_ncols<QV, 64><<<grid, block>>>(d_w, d_y, d_dst, K, N, batch);
            }
        };
        switch (qq) {
            case QUANT_Q8_0: go(std::integral_constant<quant_type, QUANT_Q8_0>{}); break;
            case QUANT_Q4_K: go(std::integral_constant<quant_type, QUANT_Q4_K>{}); break;
            case QUANT_Q3_K: go(std::integral_constant<quant_type, QUANT_Q3_K>{}); break;
            case QUANT_Q2_K: go(std::integral_constant<quant_type, QUANT_Q2_K>{}); break;
        }
    };

    auto launch = [&](void *stream_placeholder) {
        (void)stream_placeholder;
        dim3 block(WARP_SIZE, NWARPS);
        dim3 grid(N);
        if (batch == 1) {
            switch (q) {
                case QUANT_Q8_0: mmvq_gemv_single<QUANT_Q8_0><<<grid, block>>>(d_w, d_y, d_dst, K, N); break;
                case QUANT_Q4_K: mmvq_gemv_single<QUANT_Q4_K><<<grid, block>>>(d_w, d_y, d_dst, K, N); break;
                case QUANT_Q3_K: mmvq_gemv_single<QUANT_Q3_K><<<grid, block>>>(d_w, d_y, d_dst, K, N); break;
                case QUANT_Q2_K: mmvq_gemv_single<QUANT_Q2_K><<<grid, block>>>(d_w, d_y, d_dst, K, N); break;
            }
        } else if (use_mmq) {
            dim3 mblock(MMQ_THREADS);
            dim3 mgrid((N + MMQ_BM - 1) / MMQ_BM, (batch + MMQ_BN - 1) / MMQ_BN);
            mmq_q8_tiled<<<mgrid, mblock>>>((const block_q8_0 *)d_w, d_y, d_dst, K, N, batch);
        } else {
            launch_multi(q);
        }
    };

    launch(nullptr);
    CHECK(cudaDeviceSynchronize());

    if (verify && !skip_fill) {
        // With unit fill (weights=1, y=1, scales=1) every quant yields dst[row] == K.
        float h0 = 0.0f;
        CHECK(cudaMemcpy(&h0, d_dst, sizeof(float), cudaMemcpyDeviceToHost));
        const float expect = (float)K;
        const bool ok = fabsf(h0 - expect) < 0.5f;
        printf("  [verify] dst[0]=%.1f expect=%.1f %s  (%s %s K=%d N=%d B=%d W=%.2f MB)\n",
               h0, expect, ok ? "OK" : "FAIL",
               label ? label : "-", quant_name(q), K, N, batch, w_bytes / 1e6);
        if (!ok) {
            cudaFree(d_w); cudaFree(d_y); cudaFree(d_dst);
            return 1;
        }
    } else if (verify && skip_fill) {
        printf("  [verify] skipped (MMVQ_SKIP_FILL)  (%s %s K=%d N=%d B=%d W=%.2f MB)\n",
               label ? label : "-", quant_name(q), K, N, batch, w_bytes / 1e6);
    }

    if (timeit) {
        const int iters = 20;
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));
        CHECK(cudaEventRecord(t0));
        for (int it = 0; it < iters; ++it) launch(nullptr);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        const double us = ms * 1e3 / iters;
        const double batch_tiles = use_mmq
            ? (double)((batch + MMQ_BN - 1) / MMQ_BN)
            : 1.0;
        const double bytes = batch_tiles * (double)w_bytes;
        const double gbs = bytes / (us * 1e-6) / 1e9;
        const char *tag = batch == 1 ? " [gemv]" : (use_mmq ? " [mmq]" : " [mmvq]");
        printf("  %-10s %-5s K=%-6d N=%-6d B=%-3d  W=%7.2f MB  %8.2f us  %7.1f GB/s(reused-W)%s\n",
               label ? label : "-", quant_name(q), K, N, batch, w_bytes / 1e6, us, gbs, tag);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    } else {
        printf(">>> launch %s %s K=%d N=%d B=%d W=%.3f MB (sim will print gpu_tot_sim_cycle)\n",
               label ? label : "-", quant_name(q), K, N, batch, w_bytes / 1e6);
    }

    cudaFree(d_w);
    cudaFree(d_y);
    cudaFree(d_dst);
    return 0;
}

static void usage(const char *argv0) {
    printf("Usage:\n");
    printf("  %s <model> <op|all> <q8_0|q4_k|q3_k_m|q2_k> [batch]\n", argv0);
    printf("  %s <K> <N> <q8_0|q4_k|q3_k_m|q2_k> [batch]\n", argv0);
    printf("  model: 8b | 14b\n");
    printf("  op: q_proj k_proj v_proj o_proj gate up down lm_head | all\n");
    printf("  batch: 1..%d (CLI; or BATCH env)\n", MAX_BATCH);
    printf("  paths: B=1 gemv | B=2..%d mmvq | B>%d mmq(q8 smem); MMVQ_FORCE_MULTI=1\n",
           MMQ_SWITCH_BATCH, MMQ_SWITCH_BATCH);
}

#ifndef MMVQ_NO_MAIN
int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Device: %s  (decode GEMV only: Q8_0 / Q4_K / Q3_K_M / Q2_K)\n", p.name);

#ifdef FUNCSIM_SAFE
    printf("Build: FUNCSIM_SAFE (execution-driven / GPGPU-Sim)\n");
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

    const int batch = (argc >= 5) ? parse_batch_arg(argv[4])
                                   : (getenv("BATCH") ? atoi(getenv("BATCH")) : 1);

    // explicit K N path (both args must be pure integers — "8b"/"14b" are models)
    if (is_all_digits(argv[1]) && is_all_digits(argv[2])) {
        const int K = atoi(argv[1]);
        const int N = atoi(argv[2]);
        return run_one(K, N, batch, q, "custom", true, !getenv("MMVQ_NO_TIME"));
    }

    if (getenv("MMVQ_TINY")) {
        return run_one(256, 64, batch, q, "tiny", true, !getenv("MMVQ_NO_TIME"));
    }

    const char *model = argv[1];
    const char *op = argv[2];
    int n_shapes = 0;
    const gemv_shape *tbl = gemv_shapes_for_model(model, &n_shapes);

    if (strcmp(op, "all") == 0) {
        printf("[model=%s quant=%s batch=%d — %d decode GEMV shapes]\n", model, quant_name(q), batch, n_shapes);
        for (int i = 0; i < n_shapes; ++i) {
            if (run_one(tbl[i].K, tbl[i].N, batch, q, tbl[i].name, false, !getenv("MMVQ_NO_TIME"))) {
                return 1;
            }
        }
        return 0;
    }

    int K = 0, N = 0;
    if (gemv_shape_lookup(tbl, n_shapes, op, &K, &N) != 0) {
        printf("Unknown op '%s' for model %s\n", op, model);
        return 1;
    }
    return run_one(K, N, batch, q, op, true, !getenv("MMVQ_NO_TIME"));
}
#endif // MMVQ_NO_MAIN
