// layer_gemm_bench — ONLY the weight GEMV/GEMM kernels of one Qwen3 decoder layer,
// in decode order: Q, K, V, O, gate, up, down (batch-1, mul_mat_vec_q).
//
// These are the only kernels whose DRAM traffic changes under 2/3-bit weight
// compression (QWC), so this is the file you re-run per QWC_BITS. The activation
// inputs are pre-quantized dummy q8_1 buffers (the quantize kernels live in the
// non-gemm file and don't depend on weight bits).
//
//   gemm_time(bits)  <- run this under QWC_BITS=bits
//   layer_time(bits) = nongemm_time (run once) + gemm_time(bits)
//
// Build: make gemm | make gemm_exec      Run: ./layer_gemm_bench [--8b|--tiny] [dim flags]
#include "layer_kernels.cuh"

int main(int argc, char **argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    Dims d = parse_dims(argc, argv);
    const int H=d.hidden, QD=d.qheads*d.headdim, KD=d.kvheads*d.headdim, F=d.ffn;
    printf("Device: %s\nlayer_gemm: hidden=%d Qdim=%d KVdim=%d ffn=%d  (QWC_BITS=%s)\n",
           p.name, H, QD, KD, F, getenv("QWC_BITS")?getenv("QWC_BITS"):"off");

    // weights (q8_0) — the QWC weight regions
    block_q8_0 *Wq=wmalloc(H,QD), *Wk=wmalloc(H,KD), *Wv=wmalloc(H,KD), *Wo=wmalloc(QD,H);
    block_q8_0 *Wg=wmalloc(H,F),  *Wu=wmalloc(H,F),  *Wd=wmalloc(F,H);
    // pre-quantized activations (q8_1) — produced by quantize in the non-gemm file
    block_q8_1 *aH=qmalloc(H), *aQ=qmalloc(QD), *aF=qmalloc(F);
    // outputs
    float *q=fmalloc(QD,0.0f), *kk=fmalloc(KD,0.0f), *vv=fmalloc(KD,0.0f), *o=fmalloc(H,0.0f);
    float *g=fmalloc(F,0.0f), *u=fmalloc(F,0.0f), *down=fmalloc(H,0.0f);

    for (int L=0; L<d.layers; ++L) {
        gemv(Wq, aH, q,  H,  QD);   // Q proj
        gemv(Wk, aH, kk, H,  KD);   // K proj
        gemv(Wv, aH, vv, H,  KD);   // V proj
        gemv(Wo, aQ, o,  QD, H);    // O proj
        gemv(Wg, aH, g,  H,  F);    // gate
        gemv(Wu, aH, u,  H,  F);    // up
        gemv(Wd, aF, down, F, H);   // down
    }
    CK(cudaDeviceSynchronize());
    float h0=0.0f; CK(cudaMemcpy(&h0, down, sizeof(float), cudaMemcpyDeviceToHost));
    printf("[gemm done] %d layer(s)  down[0]=%g expected=%d  %s\n",
           d.layers, h0, F, (h0==(float)F)?"OK":"(value-insensitive)");
    return 0;
}
