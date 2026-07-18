// Qwen3-14B decode kernel shape tables (from trace grid sizes + model dims).
// hidden=5120, n_q_heads=40, n_kv_heads=8, head_dim=128, ffn=17408, vocab=151936.
#pragma once

struct rms_shape {
    int         ncols;
    int         nrows;
    int         nchannels;
    int         nsamples;
    int         block_size; // 256 or 1024 — matches rms_norm_f32<BS> in llama.cpp norm.cu
    const char *name;
};

static const rms_shape qwen3_14b_rms_shapes[] = {
    { 5120, 1, 1, 1, 1024, "attn_norm" },
    {  128, 40, 1, 1,  256, "q_norm"    },
    {  128,  8, 1, 1,  256, "k_norm"    },
    { 5120,  1, 1, 1, 1024, "ffn_norm"  },
};
static const int qwen3_14b_rms_num_shapes =
    sizeof(qwen3_14b_rms_shapes) / sizeof(qwen3_14b_rms_shapes[0]);

struct quant_shape {
    int         ne0; // must be multiple of 32 (QK8_1)
    int         ne1;
    int         ne2;
    int         ne3;
    const char *name;
};

// grid.x = ceil(ne0 / 256) for CUDA_QUANTIZE_BLOCK_SIZE=256 (llama.cpp quantize.cu).
static const quant_shape qwen3_14b_quant_shapes[] = {
    { 5120,  1, 1, 1, "hidden_5120"  }, // grid 20 — before Q/K/V/O/FFN-input projections
    { 17408, 1, 1, 1, "ffn_mid_17408" }, // grid 68 — after gate+up (trace grid)
};
static const int qwen3_14b_quant_num_shapes =
    sizeof(qwen3_14b_quant_shapes) / sizeof(qwen3_14b_quant_shapes[0]);

struct fattn_shape {
    int         head_dim;
    int         n_q_heads;
    int         n_kv_heads;
    int         q_cols;      // Q->ne[1]; 2 => split-K cols_per_block=2
    int         parallel_k;  // launch_fattn blocks_num.y
    int         seq_len;     // KV cache length (tokens)
    const char *name;
};

// flash_attn_ext_vec decode: grid (1, parallel_k, ntiles_z_gqa*n_kv_heads) = (1, 2, 40) for 14B.
static const fattn_shape qwen3_14b_fattn_shapes[] = {
    { 128, 40, 8, 2, 2,  1088, "decode_p1024" }, // -p 1024 prefill + ~64 decode tokens
    { 128, 40, 8, 2, 2,  1024, "prefill_only" },
    { 128, 40, 8, 2, 2,   512, "short_ctx"    },
};
static const int qwen3_14b_fattn_num_shapes =
    sizeof(qwen3_14b_fattn_shapes) / sizeof(qwen3_14b_fattn_shapes[0]);
