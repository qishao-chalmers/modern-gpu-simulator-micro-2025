// mul_mat_vec_q weight-bit-width sweep.
//
// Same memory-bound GEMV as mul_mat_vec_q, but the weight quant width is a RUNTIME
// parameter (2/3/4/8 bit) so you can read off "duration vs weight bits" directly in the
// simulator. Decode mmvq is bandwidth-bound, so cycles should scale ~with weight bytes:
//   weight quants per 32-element block = qwords int32 words, where qwords = bits
//   (32 weights * bits / 8 / 4 = bits).  8-bit -> 8 words (32 B), 4-bit -> 4 (16 B),
//   3-bit -> 3 (12 B), 2-bit -> 2 (8 B), plus one f32 scale per block.
//
// Functional-sim-safe: float accumulation (no __dp4a), f32 scales (no fp16), and a
// shared-memory reduction (no __shfl, which GPGPU-Sim's PTX functional model mishandles).
// So it runs AND verifies under execution-driven mode.
//
// Build:  make            (sm_90)   |   make ARCH='-gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70'
// Run:    ./mmvq_bitwidth         (sweeps bits = 8,4,3,2 at one shape; prints bytes; sim prints cycles per launch)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define QK 32          // weights per block
#define NTHREADS 128   // threads per CTA (one output row per CTA)

#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

// W: packed weight quants, qwords int32 per block, laid out [ (row*bpr + kb) * qwords ].
// Ws: one f32 scale per weight block.  Y: activation int8 (q8-style), 32 per block.
// Ys: one f32 scale per activation block.  qwords (= bits) is runtime.
__global__ void mmvq_bw(const int *__restrict__ W, const float *__restrict__ Ws,
                        const signed char *__restrict__ Y, const float *__restrict__ Ys,
                        float *__restrict__ dst, int K, int N, int qwords) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int bpr = K / QK;  // blocks per row

    float acc = 0.0f;
    for (int kb = tid; kb < bpr; kb += NTHREADS) {
        const int *wq = W + (size_t)(row * bpr + kb) * qwords;
        const signed char *yq = Y + (size_t)kb * QK;
#ifndef NO_DEQUANT
        const float ws = Ws[(size_t)row * bpr + kb];   // per-block scales — consumed by the FR, not the SM
        const float ys = Ys[kb];
#endif
        int sumi = 0;
        // dot over the qwords*4 int8 lanes the (compressed) weight provides
#ifdef USE_DP4A
        // Real llama.cpp compute: one __dp4a = 4 int8 MACs per instruction (needs sm_61+;
        // the exec-driven functional model now implements dp4a). Realistic, memory-bound.
        #pragma unroll
        for (int w = 0; w < qwords; ++w) {
            const int yv = *(const int *)(yq + w * 4);   // 4 packed int8 activations (4B-aligned)
            sumi = __dp4a(wq[w], yv, sumi);
        }
#else
        for (int w = 0; w < qwords; ++w) {
            const int wv = wq[w];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const signed char wb = (signed char)((wv >> (b * 8)) & 0xff);
                sumi += (int)wb * (int)yq[w * 4 + b];
            }
        }
#endif
#ifdef NO_DEQUANT
        acc += (float)sumi;              // FR delivered reconstructed weights; SM skips the scale-apply
#else
        acc += ws * ys * (float)sumi;    // dequant: apply weight+activation block scales
#endif
    }

    __shared__ float sm[NTHREADS];
    sm[tid] = acc;
    __syncthreads();
    for (int s = NTHREADS / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) dst[row] = sm[0];
}

__global__ void fill_i(int *p, size_t n)         { size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=0x01010101; } // 4x int8 = 1
__global__ void fill_i8(signed char *p, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=1; }
__global__ void fill_f(float *p, size_t n)       { size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=1.0f; }

static int run_bits(int K, int N, int bits, bool verify) {
    const int bpr = K / QK;
    const int qwords = bits;                 // 32*bits/8/4 = bits
    const size_t nW  = (size_t)N * bpr * qwords;
    const size_t nWs = (size_t)N * bpr;
    const size_t nY  = (size_t)bpr * QK;
    const size_t nYs = (size_t)bpr;
    const double wbytes = (double)nW * 4 + (double)nWs * 4;  // quants + scales

    int *dW; float *dWs; signed char *dY; float *dYs; float *dDst;
    CHECK(cudaMalloc(&dW,  nW  * sizeof(int)));
    CHECK(cudaMalloc(&dWs, nWs * sizeof(float)));
    CHECK(cudaMalloc(&dY,  nY  * sizeof(signed char)));
    CHECK(cudaMalloc(&dYs, nYs * sizeof(float)));
    CHECK(cudaMalloc(&dDst, (size_t)N * sizeof(float)));
    fill_i <<<(nW +255)/256,256>>>(dW,  nW);
    fill_f <<<(nWs+255)/256,256>>>(dWs, nWs);
    fill_i8<<<(nY +255)/256,256>>>(dY,  nY);
    fill_f <<<(nYs+255)/256,256>>>(dYs, nYs);
    CHECK(cudaGetLastError());

    printf(">>> bits=%d qwords=%d  K=%d N=%d  weight=%.3f MB\n", bits, qwords, K, N, wbytes/1e6);
    mmvq_bw<<<N, NTHREADS>>>(dW, dWs, dY, dYs, dDst, K, N, qwords);
    CHECK(cudaDeviceSynchronize());
    if (verify) {
        float h0 = 0.0f; CHECK(cudaMemcpy(&h0, dDst, sizeof(float), cudaMemcpyDeviceToHost));
        const float exp = (float)bpr * qwords * 4;  // all-ones dot
        printf("    [verify] dst[0]=%.0f expected=%.0f %s\n", h0, exp, h0==exp?"OK":"MISMATCH");
    }
    cudaFree(dW); cudaFree(dWs); cudaFree(dY); cudaFree(dYs); cudaFree(dDst);
    return 0;
}

int main(int argc, char **argv) {
    // usage: mmvq_bitwidth [K] [N] [bits]
    //   K, N : GEMV dims ([1xK]*[KxN]); default 2048 64 (small, exec-driven friendly)
    //   bits : if given, run ONLY that weight bit-width (2/3/4/8); else sweep 8,4,3,2.
    //          Use a single bit-width for big/real shapes (functional sim runs every thread).
    int dev = 0; cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Device: %s  (mmvq weight-bit-width sweep)\n", p.name);
    int K = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 64;
    if (argc > 3) {
        return run_bits(K, N, atoi(argv[3]), true);
    }
    const int bitset[] = {8, 4, 3, 2};
    for (int i = 0; i < 4; ++i)
        if (run_bits(K, N, bitset[i], true)) return 1;
    return 0;
}
