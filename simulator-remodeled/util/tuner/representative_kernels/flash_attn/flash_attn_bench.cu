// Standalone flash_attn_ext_vec decode benchmark.
//
// Split-K (parallel-K) decode flash attention matching llama.cpp fattn-vec.cuh:
//   - flash_attn_ext_vec_decode: grid (1, PK, batch*n_q_heads) — each block owns a KV slice
//   - flash_attn_combine: grid (1, 1, batch*n_q_heads) — merges PK partial softmax states
//   - PK via PARALLEL_K env (default 8) or 5th positional arg after n_kv_heads
//
// Build:  make | make exec
// Run:    ./flash_attn_bench
//         ./flash_attn_bench_exec 1024        # seq_len, batch=1, PK=8
//         PARALLEL_K=4 ./flash_attn_bench_exec 1024 1
//         ./flash_attn_bench_exec 1024 1 40 8 16   # seq batch n_q n_kv PK
//         VERIFY_RAMP=1 ./flash_attn_bench_exec 64 1 40 8 4
// batch = parallel sequences, each with its OWN KV cache -> KV traffic scales with batch.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "../common/bench_common.h"
#include "../common/shapes_qwen3_14b_decode.h"

#define NTHREADS 128
#define NWARPS   (NTHREADS / 32)
#define GGML_UNUSED(x) (void)(x)

// FP16 bits -> FP32, exec-driven safe (no __half intrinsics). Pure integer bitfield
// remap for normal numbers (rebias exponent 15->127, shift mantissa) — NO transcendental,
// unlike the old exp2f-per-element decode that dominated the KV-read cost.
static __device__ __forceinline__ float load_f16(const uint16_t *p) {
#ifdef NO_DEQUANT
    // FR-offload model: the memory controller already reconstructed the value, so the SM
    // does the same 2-byte load (DRAM traffic unchanged) but NO decode — one trivial cast.
    // Timing-only: the numeric value is not the true f16, so verify will report MISMATCH.
    return (float)(int)(*p);
#endif
    const uint32_t h = *p;
    const uint32_t x = h & 0x7fffu;                 // exponent + mantissa
    uint32_t f;
    if (x == 0u) {                                   // +/-0
        f = (h & 0x8000u) << 16;
    } else if ((x >> 10) == 0x1fu) {                 // inf / nan
        f = ((h & 0x8000u) << 16) | 0x7f800000u | ((x & 0x3ffu) << 13);
    } else if ((x >> 10) == 0u) {                    // subnormal (rare) — normalize
        uint32_t m = x & 0x3ffu; int e = 0;
        do { m <<= 1; ++e; } while ((m & 0x400u) == 0u);
        f = ((h & 0x8000u) << 16) | ((uint32_t)(127 - 15 - e + 1) << 23) | ((m & 0x3ffu) << 13);
    } else {                                          // normal: rebias exp by (127-15), shift
        f = ((h & 0x8000u) << 16) | ((x + (uint32_t)((127 - 15) << 10)) << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o, 32);
    return v;
}

// Merge PK split partials into final attention output (standard cross-split online-softmax).
template <int D>
__global__ void flash_attn_combine(const float *__restrict__ p_m, const float *__restrict__ p_l,
                                   const float *__restrict__ p_acc, float *__restrict__ dst, int PK) {
    constexpr int WARP = 32;
    constexpr int NW   = D / WARP;

    const int qh   = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP;
    const int warp = tid / WARP;

    float gm = -INFINITY;
    for (int s = 0; s < PK; ++s) {
        gm = fmaxf(gm, p_m[qh * PK + s]);
    }

    float den = 0.0f;
    for (int s = 0; s < PK; ++s) {
        den += p_l[qh * PK + s] * expf(p_m[qh * PK + s] - gm);
    }

    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            const int e = lane + k * WARP;
            float num = 0.0f;
            for (int s = 0; s < PK; ++s) {
                num += p_acc[((int64_t) qh * PK + s) * D + e] * expf(p_m[qh * PK + s] - gm);
            }
            dst[(int64_t) qh * D + e] = num / fmaxf(den, 1e-20f);
        }
    }
}

// Decode flash-attention vec kernel (F16 K/V/Q). Split-K: blockIdx.y selects the KV slice.
// Matches ggml flash_attn_ext_vec parallelism: threads map to KV *positions*, not a
// single head-dim each. The block's NW warps stride over the KV cache (warp w owns
// positions t = w, w+NW, w+2·NW, …); the Q·K[t] dot is a WARP-shuffle reduction (no
// per-position block barrier). Each warp keeps its own online-softmax partial (m,l,acc);
// the NW partials are combined once at the end. Critical path ≈ seq/NW warp-steps with
// a single __syncthreads — vs. the old seq sequential block-reduction (seq×log2(D) barriers).
//
// Thread↦output-dim mapping for the V accumulator: lane owns dims e = lane + k·32,
// k = 0..NW-1, so each thread holds NW accumulator elements (D = NW·32).
// Batched decode: grid.z = batch * n_q_heads; each sequence has its OWN KV cache
// K/V[batch][n_kv_heads][seq_len][D] -> KV read volume scales with batch.
template <int D>
__global__ void flash_attn_ext_vec_decode(const uint16_t *__restrict__ Q, const uint16_t *__restrict__ K,
                                          const uint16_t *__restrict__ V, float *__restrict__ p_m,
                                          float *__restrict__ p_l, float *__restrict__ p_acc,
                                          int n_q_heads, int n_kv_heads, int seq_len, int PK,
                                          float scale) {
    constexpr int WARP = 32;
    constexpr int NW   = D / WARP;                         // warps per block (blockDim.x == D)

    const int split   = blockIdx.y;
    const int qh      = blockIdx.z;                        // batch*n_q_heads index
    const int seq_b   = qh / n_q_heads;
    const int head    = qh % n_q_heads;
    const int kv_head = head * n_kv_heads / n_q_heads;
    const int tid     = threadIdx.x;
    const int lane    = tid % WARP;
    const int warp    = tid / WARP;

    const int per_split = (seq_len + PK - 1) / PK;
    const int t_begin   = split * per_split;
    const int t_end     = t_begin + per_split < seq_len ? t_begin + per_split : seq_len;
    const int pidx      = qh * PK + split;

    const int64_t q_base  = (int64_t) qh * D;
    const int64_t kv_base = (int64_t)(seq_b * n_kv_heads + kv_head) * seq_len * D;

    if (t_begin >= t_end) {
        if (tid == 0) {
            p_m[pidx] = -INFINITY;
            p_l[pidx] = 0.0f;
        }
        if (warp == 0) {
#pragma unroll
            for (int k = 0; k < NW; ++k) {
                p_acc[(int64_t) pidx * D + lane + k * WARP] = 0.0f;
            }
        }
        return;
    }

    __shared__ float q_sh[D];
    q_sh[tid] = load_f16(Q + q_base + tid);
    __syncthreads();

    float acc[NW];
#pragma unroll
    for (int k = 0; k < NW; ++k) acc[k] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int t = t_begin + warp; t < t_end; t += NW) {
        float partial = 0.0f;
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            const int d = lane + k * WARP;
            partial += q_sh[d] * load_f16(K + (kv_base + (int64_t) t * D + d));
        }
        const float dot   = warp_reduce_sum(partial) * scale;
        const float m_new = fmaxf(m, dot);
        const float alpha = expf(m - m_new);
        const float p     = expf(dot - m_new);
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            const int d = lane + k * WARP;
            acc[k] = acc[k] * alpha + p * load_f16(V + (kv_base + (int64_t) t * D + d));
        }
        l = l * alpha + p;
        m = m_new;
    }

    __shared__ float sm_m[NW], sm_l[NW], sm_acc[NW][D];
    if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
#pragma unroll
    for (int k = 0; k < NW; ++k) sm_acc[warp][lane + k * WARP] = acc[k];
    __syncthreads();

    if (warp == 0) {
        float gm = -INFINITY;
#pragma unroll
        for (int w = 0; w < NW; ++w) gm = fmaxf(gm, sm_m[w]);
        float den = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) den += expf(sm_m[w] - gm) * sm_l[w];
        if (lane == 0) {
            p_m[pidx] = gm;
            p_l[pidx] = den;
        }
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            const int e = lane + k * WARP;
            float num = 0.0f;
#pragma unroll
            for (int w = 0; w < NW; ++w) num += expf(sm_m[w] - gm) * sm_acc[w][e];
            p_acc[(int64_t) pidx * D + e] = num;
        }
    }
}

// ----------------------------- INT8 (q8_0) KV/Q path -----------------------------
// q8_0 block: 32 int8 quants + one f16 scale (34 B / 32 elems = 1.0625 B/elem, vs FP16's
// 2 B/elem) -> ~halves KV-cache traffic. Matches ggml q8_0 KV cache. Dequantized to f32
// in-kernel (funcsim-safe: no dp4a); the reduced *memory volume* is what the study needs.
#define QK8 32
struct blk_q8 { uint16_t d; int8_t qs[QK8]; };   // 34 bytes

// Same warp-parallel-over-KV structure as the FP16 kernel, but K/V/Q are q8_0 blocks.
// Row of D elems = NW blocks (NW == D/32 == warps/block). Dim e = lane + k*32 lives in
// block k at qs[lane]; a warp reads one whole block (coalesced qs + broadcast scale).
template <int D>
__global__ void flash_attn_ext_vec_q8_decode(const blk_q8 *__restrict__ Q, const blk_q8 *__restrict__ K,
                                             const blk_q8 *__restrict__ V, float *__restrict__ p_m,
                                             float *__restrict__ p_l, float *__restrict__ p_acc,
                                             int n_q_heads, int n_kv_heads, int seq_len, int PK,
                                             float scale) {
    constexpr int WARP = 32;
    constexpr int NW   = D / WARP;

    const int split   = blockIdx.y;
    const int qh      = blockIdx.z;
    const int seq_b   = qh / n_q_heads;
    const int head    = qh % n_q_heads;
    const int kv_head = head * n_kv_heads / n_q_heads;
    const int tid  = threadIdx.x, lane = tid % WARP, warp = tid / WARP;

    const int per_split = (seq_len + PK - 1) / PK;
    const int t_begin   = split * per_split;
    const int t_end     = t_begin + per_split < seq_len ? t_begin + per_split : seq_len;
    const int pidx      = qh * PK + split;

    const int64_t q_row  = (int64_t) qh * NW;
    const int64_t kv_row = (int64_t)(seq_b * n_kv_heads + kv_head) * seq_len * NW;
    const int64_t q_out  = (int64_t) qh * D;

    if (t_begin >= t_end) {
        if (tid == 0) {
            p_m[pidx] = -INFINITY;
            p_l[pidx] = 0.0f;
        }
        if (warp == 0) {
#pragma unroll
            for (int k = 0; k < NW; ++k) {
                p_acc[(int64_t) pidx * D + lane + k * WARP] = 0.0f;
            }
        }
        return;
    }

    __shared__ float q_sh[D];
    { const blk_q8 *b = &Q[q_row + warp]; q_sh[tid] = load_f16(&b->d) * (float) b->qs[lane]; }
    __syncthreads();

    float acc[NW];
#pragma unroll
    for (int k = 0; k < NW; ++k) acc[k] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int t = t_begin + warp; t < t_end; t += NW) {
        float partial = 0.0f;
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            const blk_q8 *b = &K[kv_row + (int64_t) t * NW + k];
            partial += q_sh[lane + k * WARP] * (load_f16(&b->d) * (float) b->qs[lane]);
        }
        const float dot = warp_reduce_sum(partial) * scale;
        const float mn = fmaxf(m, dot), al = expf(m - mn), pp = expf(dot - mn);
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            const blk_q8 *b = &V[kv_row + (int64_t) t * NW + k];
            acc[k] = acc[k] * al + pp * (load_f16(&b->d) * (float) b->qs[lane]);
        }
        l = l * al + pp; m = mn;
    }

    __shared__ float sm_m[NW], sm_l[NW], sm_acc[NW][D];
    if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
#pragma unroll
    for (int k = 0; k < NW; ++k) sm_acc[warp][lane + k * WARP] = acc[k];
    __syncthreads();

    if (warp == 0) {
        float gm = -INFINITY;
#pragma unroll
        for (int w = 0; w < NW; ++w) gm = fmaxf(gm, sm_m[w]);
        float den = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) den += expf(sm_m[w] - gm) * sm_l[w];
        if (lane == 0) {
            p_m[pidx] = gm;
            p_l[pidx] = den;
        }
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            const int e = lane + k * WARP;
            float num = 0.0f;
#pragma unroll
            for (int w = 0; w < NW; ++w) num += expf(sm_m[w] - gm) * sm_acc[w][e];
            p_acc[(int64_t) pidx * D + e] = num;
        }
    }
    GGML_UNUSED(q_out);
}

static int clamp_parallel_k(int seq_len, int pk) {
    if (pk < 1) pk = 1;
    if (pk > seq_len) pk = seq_len;
    return pk;
}

static int get_parallel_k(int seq_len, int argc, char **argv) {
    int pk = 8;
    if (const char *e = getenv("PARALLEL_K")) pk = atoi(e);
    if (argv != nullptr && argc >= 6) pk = atoi(argv[5]);
    return clamp_parallel_k(seq_len, pk);
}

static void fill_kv_uniform(uint16_t *h_K, uint16_t *h_V, size_t n_kv_elems) {
    for (size_t i = 0; i < n_kv_elems; ++i) {
        h_K[i] = 0x3c00;
        h_V[i] = 0x3c00;
    }
}

// Ramp K along sequence dimension to exercise cross-split rescaling (V stays 1.0).
static void fill_kv_ramp(uint16_t *h_K, uint16_t *h_V, int D, int seq_len, int n_kv, int batch) {
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < n_kv; ++h) {
            for (int t = 0; t < seq_len; ++t) {
                const float kv = 0.5f + 0.001f * (float) t;
                const uint16_t bits = (uint16_t) (int) (kv * 256.0f); // rough f16-ish ramp
                for (int d = 0; d < D; ++d) {
                    const size_t idx = (size_t) (((b * n_kv + h) * seq_len + t) * D + d);
                    h_K[idx] = bits;
                    h_V[idx] = 0x3c00;
                }
            }
        }
    }
}

static int run_shape_q8(const fattn_shape &s, int batch, int PK) {
    const int D = s.head_dim, n_q = s.n_q_heads, n_kv = s.n_kv_heads, seq = s.seq_len;
    const int NW = D / 32;
    const size_t q_blocks  = (size_t) batch * n_q  * NW;          // [batch][n_q][NW]
    const size_t kv_blocks = (size_t) batch * n_kv * seq * NW;    // [batch][n_kv][seq][NW]

    blk_q8 *h_Q = new blk_q8[q_blocks];
    blk_q8 *h_K = new blk_q8[kv_blocks];
    blk_q8 *h_V = new blk_q8[kv_blocks];
    for (size_t i = 0; i < q_blocks;  ++i) { h_Q[i].d = 0x3c00; for (int j = 0; j < 32; ++j) h_Q[i].qs[j] = 1; } // 1.0
    for (size_t i = 0; i < kv_blocks; ++i) { h_K[i].d = 0x3c00; h_V[i].d = 0x3c00;
        for (int j = 0; j < 32; ++j) { h_K[i].qs[j] = 1; h_V[i].qs[j] = 1; } }

    blk_q8 *d_Q, *d_K, *d_V; float *d_dst, *d_pm, *d_pl, *d_pacc;
    const int n_qh = batch * n_q;
    BENCH_CHECK(cudaMalloc(&d_Q, q_blocks  * sizeof(blk_q8)));
    BENCH_CHECK(cudaMalloc(&d_K, kv_blocks * sizeof(blk_q8)));
    BENCH_CHECK(cudaMalloc(&d_V, kv_blocks * sizeof(blk_q8)));
    BENCH_CHECK(cudaMalloc(&d_dst, (size_t) n_qh * D * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_pm, (size_t) n_qh * PK * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_pl, (size_t) n_qh * PK * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_pacc, (size_t) n_qh * PK * D * sizeof(float)));
    BENCH_CHECK(cudaMemcpy(d_Q, h_Q, q_blocks  * sizeof(blk_q8), cudaMemcpyHostToDevice));
    BENCH_CHECK(cudaMemcpy(d_K, h_K, kv_blocks * sizeof(blk_q8), cudaMemcpyHostToDevice));
    BENCH_CHECK(cudaMemcpy(d_V, h_V, kv_blocks * sizeof(blk_q8), cudaMemcpyHostToDevice));

    const dim3 grid_attn(1, PK, n_qh), grid_combine(1, 1, n_qh), block(D, 1, 1);
    const float scale = 1.0f / sqrtf((float) D);
    const double kv_bytes = 2.0 * (double) kv_blocks * sizeof(blk_q8);
    printf(">>> flash_attn_ext_vec_q8 %s  D=%d heads=%d/%d seq=%d batch=%d PK=%d "
           "grid_attn=(%u,%u,%u) grid_combine=(%u,%u,%u) block=%d  KV=%.3f MB (int8)\n",
           s.name, D, n_q, n_kv, seq, batch, PK,
           grid_attn.x, grid_attn.y, grid_attn.z,
           grid_combine.x, grid_combine.y, grid_combine.z, D, kv_bytes / 1e6);

    if (D == 128) {
        flash_attn_ext_vec_q8_decode<128><<<grid_attn, block>>>(
            d_Q, d_K, d_V, d_pm, d_pl, d_pacc, n_q, n_kv, seq, PK, scale);
        flash_attn_combine<128><<<grid_combine, block>>>(d_pm, d_pl, d_pacc, d_dst, PK);
    } else { fprintf(stderr, "unsupported head_dim %d (need 128)\n", D); return 1; }
    BENCH_CHECK(cudaGetLastError());
    BENCH_CHECK(cudaDeviceSynchronize());

    float h0 = 0.0f;
    BENCH_CHECK(cudaMemcpy(&h0, d_dst, sizeof(float), cudaMemcpyDeviceToHost));
    printf("    [verify] dst[0]=%.4f expected=1.0000  %s\n", h0, (fabsf(h0 - 1.0f) < 1e-2f) ? "OK" : "MISMATCH");

    delete[] h_Q; delete[] h_K; delete[] h_V;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_dst);
    cudaFree(d_pm); cudaFree(d_pl); cudaFree(d_pacc);
    return 0;
}

static int run_shape(const fattn_shape &s, int batch, int PK) {
    const int D          = s.head_dim;
    const int n_q        = s.n_q_heads;
    const int n_kv       = s.n_kv_heads;
    const int seq        = s.seq_len;

    const size_t n_q_elems  = (size_t) batch * n_q * D;
    const size_t n_kv_elems = (size_t) batch * n_kv * seq * D;
    const int n_qh = batch * n_q;
    const bool verify_ramp = getenv("VERIFY_RAMP") != nullptr;

    uint16_t *d_Q = nullptr;
    uint16_t *d_K = nullptr;
    uint16_t *d_V = nullptr;
    float *d_dst = nullptr;
    float *d_pm = nullptr;
    float *d_pl = nullptr;
    float *d_pacc = nullptr;

    uint16_t *h_Q = new uint16_t[n_q_elems];
    uint16_t *h_K = new uint16_t[n_kv_elems];
    uint16_t *h_V = new uint16_t[n_kv_elems];
    for (size_t i = 0; i < n_q_elems; ++i) h_Q[i] = 0x2e66; // f16 ~0.1
    if (verify_ramp) {
        fill_kv_ramp(h_K, h_V, D, seq, n_kv, batch);
    } else {
        fill_kv_uniform(h_K, h_V, n_kv_elems);
    }

    BENCH_CHECK(cudaMalloc(&d_Q, n_q_elems * sizeof(uint16_t)));
    BENCH_CHECK(cudaMalloc(&d_K, n_kv_elems * sizeof(uint16_t)));
    BENCH_CHECK(cudaMalloc(&d_V, n_kv_elems * sizeof(uint16_t)));
    BENCH_CHECK(cudaMalloc(&d_dst, n_q_elems * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_pm, (size_t) n_qh * PK * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_pl, (size_t) n_qh * PK * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_pacc, (size_t) n_qh * PK * D * sizeof(float)));
    BENCH_CHECK(cudaMemcpy(d_Q, h_Q, n_q_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    BENCH_CHECK(cudaMemcpy(d_K, h_K, n_kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    BENCH_CHECK(cudaMemcpy(d_V, h_V, n_kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));

    const dim3 grid_attn(1, PK, n_qh), grid_combine(1, 1, n_qh), block(D, 1, 1);
    const float scale = 1.0f / sqrtf((float) D);

    const double kv_bytes = 2.0 * (double) n_kv_elems * sizeof(uint16_t);
    printf(">>> flash_attn_ext_vec %s  D=%d heads=%d/%d seq=%d batch=%d PK=%d "
           "grid_attn=(%u,%u,%u) grid_combine=(%u,%u,%u) block=%d  KV=%.3f MB%s\n",
           s.name, D, n_q, n_kv, seq, batch, PK,
           grid_attn.x, grid_attn.y, grid_attn.z,
           grid_combine.x, grid_combine.y, grid_combine.z, D, kv_bytes / 1e6,
           verify_ramp ? "  [VERIFY_RAMP]" : "");

    if (D == 128) {
        flash_attn_ext_vec_decode<128><<<grid_attn, block>>>(
            d_Q, d_K, d_V, d_pm, d_pl, d_pacc, n_q, n_kv, seq, PK, scale);
        flash_attn_combine<128><<<grid_combine, block>>>(d_pm, d_pl, d_pacc, d_dst, PK);
    } else {
        fprintf(stderr, "unsupported head_dim %d (need 128)\n", D);
        return 1;
    }
    BENCH_CHECK(cudaGetLastError());
    BENCH_CHECK(cudaDeviceSynchronize());

    // verify: uniform K=V=1.0, Q~0.1 -> softmax uniform -> out = mean(V) = 1.0
    float h0 = 0.0f;
    BENCH_CHECK(cudaMemcpy(&h0, d_dst, sizeof(float), cudaMemcpyDeviceToHost));
    if (verify_ramp) {
        printf("    [verify_ramp] dst[0]=%.4f (non-uniform K; no fixed expected)\n", h0);
    } else {
        printf("    [verify] dst[0]=%.4f expected=1.0000  %s\n",
               h0, (fabsf(h0 - 1.0f) < 1e-2f) ? "OK" : "MISMATCH");
    }

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_dst);
    cudaFree(d_pm);
    cudaFree(d_pl);
    cudaFree(d_pacc);
    return 0;
}

int main(int argc, char **argv) {
    cudaDeviceProp prop{};
    BENCH_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (flash_attn_ext_vec decode benchmark)\n", prop.name);

    // batch = number of sequences decoding in parallel. From 2nd positional arg or BATCH env.
    int batch = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
    if (argc >= 3) batch = atoi(argv[2]);
    if (batch < 1) batch = 1;

    // KV/Q precision: 16 = FP16 cache (default), 8 = q8_0 int8 cache (~half KV traffic).
    const int kvbits = getenv("KVBITS") ? atoi(getenv("KVBITS")) : 16;
    if (kvbits != 8 && kvbits != 16) { fprintf(stderr, "KVBITS must be 8 or 16\n"); return 1; }
    printf("KV/Q precision: %d-bit %s\n", kvbits, kvbits == 8 ? "(q8_0 int8 cache)" : "(FP16 cache)");

    if (argc >= 2) {
        fattn_shape s = qwen3_14b_fattn_shapes[0];
        s.seq_len = atoi(argv[1]);
        if (argc >= 4) s.n_q_heads  = atoi(argv[3]);
        if (argc >= 5) s.n_kv_heads = atoi(argv[4]);
        const int PK = get_parallel_k(s.seq_len, argc, argv);
        printf("PARALLEL_K=%d (splits KV sequence across grid.y)\n", PK);
        return kvbits == 8 ? run_shape_q8(s, batch, PK) : run_shape(s, batch, PK);
        // usage: [KVBITS=8|16] [PARALLEL_K=N] ./flash_attn_bench <seq_len> [batch] [n_q] [n_kv] [PK]
    }

    for (int i = 0; i < qwen3_14b_fattn_num_shapes; ++i) {
        const int PK = get_parallel_k(qwen3_14b_fattn_shapes[i].seq_len, 0, nullptr);
        const int rc = kvbits == 8 ? run_shape_q8(qwen3_14b_fattn_shapes[i], batch, PK)
                                   : run_shape(qwen3_14b_fattn_shapes[i], batch, PK);
        if (rc) return 1;
    }
    return 0;
}
