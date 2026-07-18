// layer_all_bench — the FULL Qwen3 decoder layer: every kernel, in batch-1 decode
// launch order (the reference / cross-check for nongemm + gemm split).
//   model_time ~= n_layers * layer_time + lm_head_time
// Build: make all | make all_exec   Run: ./layer_all_bench [--8b|--tiny|--layers N|--lmhead] [dim flags]
#include "layer_kernels.cuh"

int main(int argc, char **argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    Dims d = parse_dims(argc, argv);
    const int H=d.hidden, QD=d.qheads*d.headdim, KD=d.kvheads*d.headdim, F=d.ffn, S=d.seq, hd=d.headdim;
    printf("Device: %s\nlayer_all: hidden=%d q=%d kv=%d hd=%d ffn=%d seq=%d layers=%d lmhead=%d\n",
           p.name, H, d.qheads, d.kvheads, hd, F, S, d.layers, d.lmhead);

    block_q8_0 *Wq=wmalloc(H,QD), *Wk=wmalloc(H,KD), *Wv=wmalloc(H,KD), *Wo=wmalloc(QD,H);
    block_q8_0 *Wg=wmalloc(H,F),  *Wu=wmalloc(H,F),  *Wd=wmalloc(F,H);
    block_q8_0 *Wlm = d.lmhead ? wmalloc(H,d.vocab) : nullptr;

    float *x=fmalloc(H,0.3f), *xn=fmalloc(H,0.0f);
    float *q=fmalloc(QD,0.0f), *qn=fmalloc(QD,0.0f), *qr=fmalloc(QD,0.0f);
    float *kk=fmalloc(KD,0.0f), *kn=fmalloc(KD,0.0f), *vv=fmalloc(KD,0.0f);
    uint16_t *kr=hmalloc(KD), *vr=hmalloc(KD);
    float *attn=fmalloc(QD,0.0f), *o=fmalloc(H,0.0f);
    float *g=fmalloc(F,0.0f), *u=fmalloc(F,0.0f), *hglu=fmalloc(F,0.0f), *down=fmalloc(H,0.0f);
    block_q8_1 *aH=qmalloc(H), *aQ=qmalloc(QD), *aF=qmalloc(F);
    uint16_t *Kc=hmalloc((size_t)d.kvheads*S*hd), *Vc=hmalloc((size_t)d.kvheads*S*hd);
    float *logits = d.lmhead ? fmalloc(d.vocab,0.0f) : nullptr;
    const float scale = 1.0f/sqrtf((float)hd);
    const int pos = S-1;

    for (int L=0; L<d.layers; ++L) {
        rms_norm_f32<1024><<<1,1024>>>(x, xn, H);
        quant(xn, aH, H);  gemv(Wq, aH, q, H, QD);
        rms_norm_f32<256><<<dim3(d.qheads),256>>>(q, qn, hd);
        rope_neox<<<dim3(d.qheads),dim3(hd/2)>>>(qn, qr, d.qheads, hd, pos, 1e6f, 0);
        quant(xn, aH, H);  gemv(Wk, aH, kk, H, KD);
        quant(xn, aH, H);  gemv(Wv, aH, vv, H, KD);
        rms_norm_f32<256><<<dim3(d.kvheads),256>>>(kk, kn, hd);
        rope_neox<<<dim3(d.kvheads),dim3(hd/2)>>>(kn, kr, d.kvheads, hd, pos, 1e6f, 1);
        k_set_rows_f16<<<(KD+255)/256,256>>>(kr, Kc, KD, pos);
        rope_neox<<<dim3(d.kvheads),dim3(hd/2)>>>(vv, vr, d.kvheads, hd, 0, 1e6f, 1);
        k_set_rows_f16<<<(KD+255)/256,256>>>(vr, Vc, KD, pos);
        if (hd==128) flash_attn_ext_vec<128><<<dim3(d.qheads),128>>>(qr, Kc, Vc, attn, d.qheads, d.kvheads, S, scale);
        else if (hd==64) flash_attn_ext_vec<64><<<dim3(d.qheads),64>>>(qr, Kc, Vc, attn, d.qheads, d.kvheads, S, scale);
        quant(attn, aQ, QD);  gemv(Wo, aQ, o, QD, H);
        add_inplace<<<(H+255)/256,256>>>(x, o, H);
        rms_norm_f32<1024><<<1,1024>>>(x, xn, H);
        quant(xn, aH, H);  gemv(Wg, aH, g, H, F);
        quant(xn, aH, H);  gemv(Wu, aH, u, H, F);
        swiglu<<<(F+255)/256,256>>>(g, u, hglu, F);
        quant(hglu, aF, F);  gemv(Wd, aF, down, F, H);
        add_inplace<<<(H+255)/256,256>>>(x, down, H);
    }
    if (d.lmhead) { rms_norm_f32<1024><<<1,1024>>>(x, xn, H); quant(xn, aH, H); gemv(Wlm, aH, logits, H, d.vocab); }
    CK(cudaDeviceSynchronize());
    float h0=0.0f; CK(cudaMemcpy(&h0, d.lmhead?logits:x, sizeof(float), cudaMemcpyDeviceToHost));
    printf("[all done] %d layer(s)%s  out[0]=%g  %s\n", d.layers, d.lmhead?" + lm_head":"",
           h0, isfinite(h0)?"OK":"NONFINITE");
    return 0;
}
