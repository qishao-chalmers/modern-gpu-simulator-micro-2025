// layer_nongemm_bench — every NON-weight-matmul kernel of one Qwen3 decoder layer,
// in decode order: rms_norm x4, quantize_q8_1 x(QKV+O+gate+up+down), rope x(Q,K,V-pack),
// k_set_rows x2, flash_attn, swiglu, residual-add x2.
//
// None of these depend on the weight bit-width, so run this ONCE and reuse its time
// for every compression setting:
//   layer_time(bits) = nongemm_time (this, once) + gemm_time(bits)
//
// Build: make nongemm | make nongemm_exec   Run: ./layer_nongemm_bench [--8b|--tiny] [dim flags]
#include "layer_kernels.cuh"

int main(int argc, char **argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    Dims d = parse_dims(argc, argv);
    const int H=d.hidden, QD=d.qheads*d.headdim, KD=d.kvheads*d.headdim, F=d.ffn, S=d.seq, hd=d.headdim;
    printf("Device: %s\nlayer_nongemm: hidden=%d q=%d kv=%d hd=%d ffn=%d seq=%d\n",
           p.name, H, d.qheads, d.kvheads, hd, F, S);

    float *x=fmalloc(H,0.3f), *xn=fmalloc(H,0.0f);
    float *q=fmalloc(QD,0.2f), *qn=fmalloc(QD,0.0f), *qr=fmalloc(QD,0.0f);
    float *kk=fmalloc(KD,0.2f), *kn=fmalloc(KD,0.0f), *vv=fmalloc(KD,0.2f);
    uint16_t *kr=hmalloc(KD), *vr=hmalloc(KD);
    float *attn=fmalloc(QD,0.0f), *o=fmalloc(H,0.1f);
    float *g=fmalloc(F,0.2f), *u=fmalloc(F,0.2f), *hglu=fmalloc(F,0.0f), *down=fmalloc(H,0.1f);
    block_q8_1 *aH=qmalloc(H), *aQ=qmalloc(QD), *aF=qmalloc(F);
    uint16_t *Kc=hmalloc((size_t)d.kvheads*S*hd), *Vc=hmalloc((size_t)d.kvheads*S*hd);
    const float scale = 1.0f/sqrtf((float)hd);
    const int pos = S-1;

    for (int L=0; L<d.layers; ++L) {
        // attention
        rms_norm_f32<1024><<<1,1024>>>(x, xn, H);                              // attn_norm
        quant(xn, aH, H);                                                     // quant (for Q)
        rms_norm_f32<256><<<dim3(d.qheads),256>>>(q, qn, hd);                  // q_norm
        rope_neox<<<dim3(d.qheads),dim3(hd/2)>>>(qn, qr, d.qheads, hd, pos, 1e6f, 0); // Q rope
        quant(xn, aH, H);                                                     // quant (for K)
        quant(xn, aH, H);                                                     // quant (for V)
        rms_norm_f32<256><<<dim3(d.kvheads),256>>>(kk, kn, hd);               // k_norm
        rope_neox<<<dim3(d.kvheads),dim3(hd/2)>>>(kn, kr, d.kvheads, hd, pos, 1e6f, 1); // K rope ->f16
        k_set_rows_f16<<<(KD+255)/256,256>>>(kr, Kc, KD, pos);                // K -> cache
        rope_neox<<<dim3(d.kvheads),dim3(hd/2)>>>(vv, vr, d.kvheads, hd, 0, 1e6f, 1);   // V pack ->f16
        k_set_rows_f16<<<(KD+255)/256,256>>>(vr, Vc, KD, pos);                // V -> cache
        if (hd==128) flash_attn_ext_vec<128><<<dim3(d.qheads),128>>>(qr, Kc, Vc, attn, d.qheads, d.kvheads, S, scale);
        else if (hd==64) flash_attn_ext_vec<64><<<dim3(d.qheads),64>>>(qr, Kc, Vc, attn, d.qheads, d.kvheads, S, scale);
        quant(attn, aQ, QD);                                                  // quant (for O)
        add_inplace<<<(H+255)/256,256>>>(x, o, H);                            // residual
        // ffn
        rms_norm_f32<1024><<<1,1024>>>(x, xn, H);                             // ffn_norm
        quant(xn, aH, H);                                                     // quant (for gate)
        quant(xn, aH, H);                                                     // quant (for up)
        swiglu<<<(F+255)/256,256>>>(g, u, hglu, F);                           // silu(gate)*up
        quant(hglu, aF, F);                                                   // quant (for down)
        add_inplace<<<(H+255)/256,256>>>(x, down, H);                         // residual
    }
    if (d.lmhead) { rms_norm_f32<1024><<<1,1024>>>(x, xn, H); quant(xn, aH, H); } // final norm + lm_head quant
    CK(cudaDeviceSynchronize());
    float h0=0.0f; CK(cudaMemcpy(&h0, x, sizeof(float), cudaMemcpyDeviceToHost));
    printf("[nongemm done] %d layer(s)  x[0]=%g  %s\n", d.layers, h0, isfinite(h0)?"OK":"NONFINITE");
    return 0;
}
