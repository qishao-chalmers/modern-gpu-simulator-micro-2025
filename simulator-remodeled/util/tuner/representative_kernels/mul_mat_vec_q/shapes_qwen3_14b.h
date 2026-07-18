// Qwen3-14B mul_mat_vec_q (decode, GEMV) shape table.
//   K = ncols_x = contraction dim (input features)
//   N = nrows_x = output rows (one CTA group per row)
//   fusion = SwiGLU gate+up fusion in the real kernel (gate/up only); the minimal
//            harness models the matmul traffic only (fusion is a TODO).
// Model: hidden=5120, n_q_heads=40, n_kv_heads=8, head_dim=128, FFN=17408, vocab=151936.
#pragma once

struct mmvq_shape {
    int         K;
    int         N;
    int         fusion;
    const char *name;
};

static const mmvq_shape qwen3_14b_shapes[] = {
    {  5120,   1024, 0, "k_proj"  },
    {  5120,   1024, 0, "v_proj"  },
    {  5120,   5120, 0, "q_proj"  },
    {  5120,   5120, 0, "o_proj"  },
    {  5120,  17408, 1, "gate"    },
    {  5120,  17408, 1, "up"      },
    { 17408,   5120, 0, "down"    },
    {  5120, 151936, 0, "lm_head" },
};
static const int qwen3_14b_num_shapes =
    sizeof(qwen3_14b_shapes) / sizeof(qwen3_14b_shapes[0]);
