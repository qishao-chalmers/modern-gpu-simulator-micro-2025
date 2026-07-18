// Shared kernels + helpers for the Qwen3 decoder-layer assembly.
//
// Three drivers include this so the kernels (and thus their timing) are identical:
//   layer_all_bench.cu      — every kernel in the layer, in decode launch order
//   layer_gemm_bench.cu     — ONLY the weight GEMV/GEMM (Q/K/V/O/gate/up/down).
//                             These are the kernels affected by 2/3-bit weight
//                             compression (QWC); re-run this per QWC_BITS.
//   layer_nongemm_bench.cu  — everything else (rms_norm, quantize, rope, set_rows,
//                             flash_attn, swiglu, add). Independent of weight bits,
//                             so run ONCE and reuse.
//
//   layer_time(bits) = nongemm_time (once) + gemm_time(bits)
//   model_time       = n_layers * layer_time(bits) + lm_head(bits)
//
// FUNCSIM_SAFE (set by `make exec`): f32 scales, no dp4a/fp16 intrinsics, so the
// execution-driven PTX functional sim can run it.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#ifndef FUNCSIM_SAFE
#include <cuda_fp16.h>
#endif

#define QK 32
#define WARP_SIZE 32
#define EPS 1e-5f
#define NWARPS 4
#define QUANT_BLOCK 256

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifdef FUNCSIM_SAFE
typedef struct { float d;          int8_t qs[QK]; } block_q8_0;   // 36 B
typedef struct { float d; float s; int8_t qs[QK]; } block_q8_1;   // 40 B
#else
typedef struct { half  d;  int8_t qs[QK]; } block_q8_0;           // 34 B
typedef struct { half2 ds; int8_t qs[QK]; } block_q8_1;           // 36 B
#endif

// ----------------------------- weight GEMV/GEMM (QWC-affected) -----------------------------
static __device__ __forceinline__ float dot_q8(const block_q8_0 *x, const block_q8_1 *y) {
#ifdef FUNCSIM_SAFE
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < QK; ++j) s += (float)x->qs[j]*(float)y->qs[j];
    return x->d * y->d * s;
#else
    int s = 0;
#pragma unroll
    for (int i = 0; i < QK/4; ++i)
        s = __dp4a(*(const int*)(x->qs+4*i), *(const int*)(y->qs+4*i), s);
    return __half2float(x->d) * __half22float2(y->ds).x * (float)s;
#endif
}
// batch-1 GEMV: dst[N] = W[N x K](q8_0) . y[K](q8_1). One CTA per output row.
__global__ void mmvq(const block_q8_0 *__restrict__ W, const block_q8_1 *__restrict__ y,
                     float *__restrict__ dst, int K, int N) {
    const int row = blockIdx.x; if (row >= N) return;
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x, nt = NWARPS*WARP_SIZE;
    const int bpr = K/QK;
    const block_q8_0 *wr = W + (size_t)row*bpr;
    float acc = 0.0f;
    for (int kb = tid; kb < bpr; kb += nt) acc += dot_q8(&wr[kb], &y[kb]);
    __shared__ float sm[NWARPS*WARP_SIZE];
    sm[tid] = acc; __syncthreads();
    for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) sm[tid]+=sm[tid+o]; __syncthreads(); }
    if (tid == 0) dst[row] = sm[0];
}

// ----------------------------- non-GEMM kernels (compression-independent) -----------------------------
template <int BS>
__device__ float block_sum(float v) {
    __shared__ float sm[BS];
    sm[threadIdx.x] = v; __syncthreads();
    for (int s = BS/2; s > 0; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x+s]; __syncthreads(); }
    return sm[0];
}
template <int BS>
__global__ void rms_norm_f32(const float *x, float *dst, int ncols) {
    const int row = blockIdx.x, tid = threadIdx.x;
    x += (size_t)row*ncols; dst += (size_t)row*ncols;
    float t = 0.0f;
    for (int c = tid; c < ncols; c += BS) { float xi = x[c]; t += xi*xi; }
    t = block_sum<BS>(t);
    const float scale = rsqrtf(t/(float)ncols + EPS);
    for (int c = tid; c < ncols; c += BS) dst[c] = scale * x[c];
}

__global__ void quantize_q8_1(const float *__restrict__ x, block_q8_1 *__restrict__ y, int ne0) {
    const int i0 = blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= ne0) return;
    const int i1 = blockIdx.y;
    const int64_t ic = (int64_t)i1*ne0 + i0;
    const int64_t ib = ic / QK, iqs = ic % QK;
    const float xi = x[ic];
    __shared__ float sa[QUANT_BLOCK], ss[QUANT_BLOCK];
    sa[threadIdx.x] = fabsf(xi); ss[threadIdx.x] = xi; __syncthreads();
    const int base = threadIdx.x - (threadIdx.x % QK);
    for (int o = QK/2; o > 0; o >>= 1) {
        if ((threadIdx.x % QK) < o) { sa[threadIdx.x]=fmaxf(sa[threadIdx.x],sa[threadIdx.x+o]); ss[threadIdx.x]+=ss[threadIdx.x+o]; }
        __syncthreads();
    }
    const float amax = sa[base], sum = ss[base];
    const float dd = amax/127.0f;
    y[ib].qs[iqs] = (amax==0.0f)?0:(int8_t)lrintf(xi/dd);
    if (iqs != 0) return;
#ifdef FUNCSIM_SAFE
    y[ib].d = dd; y[ib].s = sum;
#else
    y[ib].ds = __floats2half2_rn(dd, sum);
#endif
}

static __device__ __forceinline__ uint16_t f32_to_f16_bits(float f) {
#ifdef FUNCSIM_SAFE
    uint32_t u; memcpy(&u,&f,4); uint16_t s=(u>>16)&0x8000;
    int e=((u>>23)&0xff)-112; if(e<=0) return s; if(e>=31) return s|0x7c00;
    return s|(uint16_t)((e<<10)|((u>>13)&0x3ff));
#else
    return __half_as_ushort(__float2half(f));
#endif
}
static __device__ __forceinline__ float ld_f16(const uint16_t *p) {
#ifdef FUNCSIM_SAFE
    // Integer bitfield remap (rebias exp 15->127, shift mantissa) — NO exp2f transcendental
    // per element, which otherwise dominates KV-read cost in funcsim.
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
    } else {                                          // normal
        f = ((h & 0x8000u) << 16) | ((x + (uint32_t)((127 - 15) << 10)) << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
#else
    return __half2float(__ushort_as_half(*p));
#endif
}
// rope_neox: rotate pairs (i, i+rot/2) of each head row. out_f16=1 -> store f16 bits.
__global__ void rope_neox(const float *__restrict__ src, void *__restrict__ dst,
                          int n_rows, int rot, int pos, float theta_base, int out_f16) {
    const int row = blockIdx.x; if (row >= n_rows) return;
    const int i = threadIdx.x; if (i >= rot/2) return;
    const float *sr = src + (size_t)row*rot;
    const float theta = (float)pos * powf(theta_base, -2.0f*(float)i/(float)rot);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = sr[i], x1 = sr[i + rot/2];
    const float r0 = x0*c - x1*s, r1 = x0*s + x1*c;
    if (out_f16) {
        uint16_t *d = (uint16_t*)dst + (size_t)row*rot;
        d[i]=f32_to_f16_bits(r0); d[i+rot/2]=f32_to_f16_bits(r1);
    } else {
        float *d = (float*)dst + (size_t)row*rot; d[i]=r0; d[i+rot/2]=r1;
    }
}
// k_set_rows: write a head-row of K/V (f16) into the cache at row `pos`.
__global__ void k_set_rows_f16(const uint16_t *__restrict__ src, uint16_t *__restrict__ cache,
                               int n_elem, int pos) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n_elem) return;
    cache[(size_t)pos*n_elem + i] = src[i];
}
static __device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o, 32);
    return v;
}
// flash_attn over seq KV positions (f16 cache), one block (D threads) per q_head.
// Matches ggml flash_attn_ext_vec: threads map to KV *positions* (warp w strides over
// t = w, w+NW, …), the Q·K[t] dot is a WARP-shuffle reduction (no per-position block
// barrier), each warp keeps its own online-softmax partial (m,l,acc), and the NW partials
// are combined with a single __syncthreads. Thread↦output-dim: lane owns e = lane + k·32.
template <int D>
__global__ void flash_attn_ext_vec(const float *__restrict__ Q, const uint16_t *__restrict__ Kc,
                                   const uint16_t *__restrict__ Vc, float *__restrict__ dst,
                                   int n_q, int n_kv, int seq, float scale) {
    constexpr int WARP = 32, NW = D / WARP;               // blockDim.x == D
    const int head = blockIdx.x; if (head >= n_q) return;
    const int kvh  = head * n_kv / n_q;
    const int tid = threadIdx.x, lane = tid % WARP, warp = tid / WARP;
    const size_t q_base  = (size_t)head * D;
    const size_t kv_base = (size_t)kvh * seq * D;

    __shared__ float q[D];
    q[tid] = Q[q_base + tid];
    __syncthreads();

    float acc[NW];
#pragma unroll
    for (int k = 0; k < NW; ++k) acc[k] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int t = warp; t < seq; t += NW) {
        float partial = 0.0f;
#pragma unroll
        for (int k = 0; k < NW; ++k) { const int d = lane + k*WARP; partial += q[d] * ld_f16(Kc + kv_base + (size_t)t*D + d); }
        const float dot = warp_reduce_sum(partial) * scale;
        const float mn = fmaxf(m, dot), al = expf(m - mn), pp = expf(dot - mn);
#pragma unroll
        for (int k = 0; k < NW; ++k) { const int d = lane + k*WARP; acc[k] = acc[k]*al + pp * ld_f16(Vc + kv_base + (size_t)t*D + d); }
        l = l*al + pp; m = mn;
    }

    __shared__ float sm_m[NW], sm_l[NW], sm_acc[NW][D];
    if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
#pragma unroll
    for (int k = 0; k < NW; ++k) sm_acc[warp][lane + k*WARP] = acc[k];
    __syncthreads();

    if (warp == 0) {
        float gm = -INFINITY;
#pragma unroll
        for (int w = 0; w < NW; ++w) gm = fmaxf(gm, sm_m[w]);
        float den = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) den += expf(sm_m[w] - gm) * sm_l[w];
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            const int e = lane + k*WARP;
            float num = 0.0f;
#pragma unroll
            for (int w = 0; w < NW; ++w) num += expf(sm_m[w] - gm) * sm_acc[w][e];
            dst[q_base + e] = num / fmaxf(den, 1e-20f);
        }
    }
}
__global__ void swiglu(const float *g, const float *u, float *o, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    const float gv = g[i]; o[i] = (gv/(1.0f+expf(-gv))) * u[i];
}
__global__ void add_inplace(float *x, const float *y, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) x[i] += y[i];
}

// ----------------------------- fills + host helpers -----------------------------
__global__ void fW(block_q8_0 *b, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
#ifdef FUNCSIM_SAFE
    b[i].d=1.0f;
#else
    b[i].d=__float2half(1.0f);
#endif
    for(int j=0;j<QK;++j) b[i].qs[j]=1; }
__global__ void fQ(block_q8_1 *b, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
#ifdef FUNCSIM_SAFE
    b[i].d=1.0f; b[i].s=32.0f;
#else
    b[i].ds=__floats2half2_rn(1.0f,32.0f);
#endif
    for(int j=0;j<QK;++j) b[i].qs[j]=1; }
__global__ void fF(float *p, size_t n, float v){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)p[i]=v; }
__global__ void fH(uint16_t *p, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)p[i]=0x3c00; }

static inline block_q8_0* wmalloc(int K, int N){ block_q8_0 *p; size_t nb=(size_t)N*(K/QK);
    CK(cudaMalloc(&p, nb*sizeof(block_q8_0))); fW<<<(nb+255)/256,256>>>(p,nb); return p; }
static inline block_q8_1* qmalloc(int K){ block_q8_1 *p; size_t nb=K/QK;
    CK(cudaMalloc(&p, nb*sizeof(block_q8_1))); fQ<<<(nb+255)/256,256>>>(p,nb); return p; }
static inline float* fmalloc(size_t n, float v){ float *p; CK(cudaMalloc(&p,n*sizeof(float))); fF<<<(n+255)/256,256>>>(p,n,v); return p; }
static inline uint16_t* hmalloc(size_t n){ uint16_t *p; CK(cudaMalloc(&p,n*sizeof(uint16_t))); fH<<<(n+255)/256,256>>>(p,n); return p; }

static inline void quant(const float *x, block_q8_1 *y, int n){
    quantize_q8_1<<<dim3((n+QUANT_BLOCK-1)/QUANT_BLOCK,1,1), QUANT_BLOCK>>>(x,y,n); }
static inline void gemv(block_q8_0 *W, block_q8_1 *xq, float *dst, int K, int N){
    mmvq<<<dim3(N), dim3(WARP_SIZE,NWARPS)>>>(W,xq,dst,K,N); }

// ----------------------------- model dims -----------------------------
struct Dims { int hidden, qheads, kvheads, headdim, ffn, seq, vocab, layers, lmhead; };
static inline Dims parse_dims(int argc, char **argv) {
    Dims d{5120,40,8,128,17408,1088,151936,1,0};   // Qwen3-14B, batch-1 decode
    for (int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--tiny")){ d.hidden=512; d.qheads=8; d.kvheads=2; d.headdim=64; d.ffn=1024; d.seq=128; d.vocab=4096; }
        else if(!strcmp(argv[i],"--8b")){ d.hidden=4096; d.qheads=32; d.kvheads=8; d.headdim=128; d.ffn=12288; d.vocab=151936; d.layers=1; }
        else if(!strcmp(argv[i],"--layers")&&i+1<argc) d.layers=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--lmhead")) d.lmhead=1;
        else if(!strcmp(argv[i],"--hidden")&&i+1<argc) d.hidden=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--qheads")&&i+1<argc) d.qheads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--kvheads")&&i+1<argc) d.kvheads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--headdim")&&i+1<argc) d.headdim=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ffn")&&i+1<argc) d.ffn=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seq")&&i+1<argc) d.seq=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--vocab")&&i+1<argc) d.vocab=atoi(argv[++i]);
    }
    return d;
}
