// mul_mat_q (Q8_0 batched GEMM) — representative kernel.
//
// This is the batched / prefill analogue of mul_mat_vec_q. When ncols_dst > 1
// (batch decode, or prompt prefill) llama.cpp stops using the GEMV kernel
// mul_mat_vec_q and switches to the tiled int8 GEMM
//   mul_mat_q<(ggml_type)8 = Q8_0, (int)32 = mmq_x, (bool)0 = need_check>
// (seen in the Qwen3-14B prefill nsys trace, ~35-92 µs per launch, paired with a
//  mul_mat_q_stream_k_fixup reduction).
//
// vs the GEMV: there is now an M (= ncols_dst = batch tokens) dimension. Each
// weight block is read once from DRAM but REUSED across all M tokens in shared
// memory, so arithmetic intensity rises with M and the kernel moves off the pure
// memory-bandwidth wall toward compute. That is exactly the regime where a
// memory-controller weight-decompression unit (the QWC lever) pays off less than
// it does for the M=1 GEMV — which is the point of having this companion bench.
//
// The real mmq.cuh uses tensor-core MMA + stream-K decomposition + warp shuffles,
// none of which are GPGPU-Sim functional-sim-safe. This is a faithful-TRAFFIC,
// simplified-COMPUTE reimplementation: a classic shared-memory-tiled int8 GEMM
// with __dp4a dot and float accumulation. No __shfl, no MMA, no stream-K fixup.
//
// Weights stay full Q8_0 (8-bit). To study weight bit-width / DRAM compression run
// the UNCHANGED 8-bit kernel under the QWC-patched simulator (QWC_BITS=2/3); the
// big weight buffer is the dominant cudaMalloc, so it is auto-detected as a weight
// region. (Same model as the mmvq QWC work; do NOT cut the kernel's compute.)
//
// Build:  make            (sm_90 native)
//         make exec       (sm_70 SASS+PTX, --cudart shared, for GPGPU-Sim)
// Run:    ./mmq_bench [K] [N] [M]
//   K = contraction dim (input features), N = output rows, M = batch tokens.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define QK 32          // weights per Q8_0 block (== contraction tile BK)
#define BM 32          // output rows (N dim)   per CTA tile  (matches mmq_x=32)
#define BN 8           // tokens     (M dim)    per CTA tile
#define NT (BM * BN)   // 256 threads, one output element each
#define WPB (QK / 4)   // int32 words per Q8_0 block = 8 (4 int8 lanes each)

#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

// W : Q8_0 weights, [N rows][bpr blocks][WPB int32], packed 4 int8 quants/word.
// Ws: one f32 scale per weight block,  [N][bpr].
// X : q8_1 activations, [M tokens][bpr blocks][WPB int32].
// Xs: one f32 scale per activation block, [M][bpr].
// dst: f32 [N][M].   Each CTA computes a BM x BN output tile.
// qwords = int32 words per WEIGHT block (software bit-width: 8-bit=8, 3-bit=3, 2-bit=2).
// The weight buffer + its DRAM traffic shrink with qwords, and the dp4a dot runs over
// qwords words (fewer MACs) — matching the SPEED draft path (weights AND compute shrink).
// The activation stays full q8_1 (WPB words).
__global__ void mmq(const int *__restrict__ W, const float *__restrict__ Ws,
                    const int *__restrict__ X, const float *__restrict__ Xs,
                    float *__restrict__ dst, int K, int N, int M, int qwords) {
    const int bpr  = K / QK;
    const int row0 = blockIdx.x * BM;          // first output row  (N)
    const int tok0 = blockIdx.y * BN;          // first token       (M)
    const int tid  = threadIdx.x;
    const int i    = tid / BN;                 // local row   in [0,BM)
    const int j    = tid % BN;                 // local token in [0,BN)
    const int row  = row0 + i;
    const int tok  = tok0 + j;

    __shared__ int   sW[BM][WPB];              // weight tile  : BM rows x one K-block
    __shared__ float sWs[BM];
    __shared__ int   sX[BN][WPB];              // activation tile
    __shared__ float sXs[BN];

    float acc = 0.0f;
    for (int kb = 0; kb < bpr; ++kb) {
        // cooperative load of this K-block's tiles into shared memory
        // sW: BM*WPB = 256 ints -> one per thread
        // weight tile: BM rows x qwords words (compressed)
        if (tid < BM * qwords) {
            int r = tid / qwords, w = tid % qwords;
            int gr = row0 + r;
            sW[r][w] = (gr < N) ? W[((size_t)gr * bpr + kb) * qwords + w] : 0;
        }
        if (tid < BM) { int gr = row0 + tid; sWs[tid] = (gr < N) ? Ws[(size_t)gr * bpr + kb] : 0.0f; }
        // sX: BN*WPB = 64 ints
        if (tid < BN * WPB) {
            int r = tid / WPB, w = tid % WPB;
            int gt = tok0 + r;
            sX[r][w] = (gt < M) ? X[((size_t)gt * bpr + kb) * WPB + w] : 0;
        }
        if (tid < BN) { int gt = tok0 + tid; sXs[tid] = (gt < M) ? Xs[(size_t)gt * bpr + kb] : 0.0f; }
        __syncthreads();

        int sumi = 0;
        for (int w = 0; w < qwords; ++w)                 // dot over the compressed weight's words
            sumi = __dp4a(sW[i][w], sX[j][w], sumi);     // 4 int8 MACs per word
        acc += sWs[i] * sXs[j] * (float)sumi;
        __syncthreads();
    }

    if (row < N && tok < M) dst[(size_t)row * M + tok] = acc;
}

__global__ void fill_i(int *p, size_t n)   { size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=0x01010101; } // 4x int8 = 1
__global__ void fill_f(float *p, size_t n) { size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=1.0f; }

static int run(int K, int N, int M, int bits) {
    const int qwords = bits;                                  // words/block: 8-bit=8, 3-bit=3, 2-bit=2
    const int bpr = K / QK;
    const size_t nW  = (size_t)N * bpr * qwords;              // compressed weight
    const size_t nWs = (size_t)N * bpr;
    const size_t nX  = (size_t)M * bpr * WPB;                 // activation stays full q8_1
    const size_t nXs = (size_t)M * bpr;
    const double wbytes = (double)nW * 4 + (double)nWs * 4;   // weight traffic (shrinks with bits)
    const double xbytes = (double)nX * 4 + (double)nXs * 4;

    int *dW, *dX; float *dWs, *dXs, *dDst;
    CHECK(cudaMalloc(&dW,  nW  * sizeof(int)));      // dominant alloc -> QWC weight region
    CHECK(cudaMalloc(&dWs, nWs * sizeof(float)));
    CHECK(cudaMalloc(&dX,  nX  * sizeof(int)));
    CHECK(cudaMalloc(&dXs, nXs * sizeof(float)));
    CHECK(cudaMalloc(&dDst, (size_t)N * M * sizeof(float)));
    fill_i<<<(nW +255)/256,256>>>(dW,  nW);
    fill_f<<<(nWs+255)/256,256>>>(dWs, nWs);
    fill_i<<<(nX +255)/256,256>>>(dX,  nX);
    fill_f<<<(nXs+255)/256,256>>>(dXs, nXs);
    CHECK(cudaGetLastError());

    dim3 grid((N + BM - 1) / BM, (M + BN - 1) / BN);
    printf(">>> mul_mat_q  bits=%d qwords=%d  K=%d N=%d M=%d  weight=%.3f MB act=%.3f MB  grid=(%u,%u) block=%d\n",
           bits, qwords, K, N, M, wbytes/1e6, xbytes/1e6, grid.x, grid.y, NT);
    mmq<<<grid, NT>>>(dW, dWs, dX, dXs, dDst, K, N, M, qwords);
    CHECK(cudaDeviceSynchronize());

    float h0 = 0.0f; CHECK(cudaMemcpy(&h0, dDst, sizeof(float), cudaMemcpyDeviceToHost));
    const float exp = (float)bpr * qwords * 4;   // all-ones: bpr blocks * qwords words * 4 lanes
    printf("    [verify] dst[0]=%.0f expected=%.0f %s\n", h0, exp, h0==exp?"OK":"MISMATCH");

    cudaFree(dW); cudaFree(dWs); cudaFree(dX); cudaFree(dXs); cudaFree(dDst);
    return 0;
}

int main(int argc, char **argv) {
    // usage: mmq_bench [K] [N] [M] [bits]   (M = batch tokens; bits = weight bit-width, default 8)
    int dev = 0; cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Device: %s  (mul_mat_q batched GEMM, software bit-width)\n", p.name);
    int K    = (argc > 1) ? atoi(argv[1]) : 2048;
    int N    = (argc > 2) ? atoi(argv[2]) : 64;
    int M    = (argc > 3) ? atoi(argv[3]) : 16;
    int bits = (argc > 4) ? atoi(argv[4]) : 8;
    return run(K, N, M, bits);
}
