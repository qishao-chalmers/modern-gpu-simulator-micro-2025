// Qwen3-14B mul_mat_q (batched / prefill GEMM) shape table.
//   K = contraction dim (input features)
//   N = output rows
//   M = ncols_dst = batch tokens (the new dim vs the GEMV; the prefill batch size).
// Same projection (K,N) as the GEMV table; mul_mat_q is selected over
// mul_mat_vec_q once M (ncols_dst) > 1. Sweep M = 8/16/32 to watch the kernel move
// off the memory-bandwidth wall (weights are reused across all M tokens).
// Model: hidden=5120, n_q_heads=40, n_kv_heads=8, head_dim=128, FFN=17408, vocab=151936.
#pragma once

struct mmq_shape {
    int         K;
    int         N;
    const char *name;
};

static const mmq_shape qwen3_14b_mmq_shapes[] = {
    {  5120,   1024, "k_proj"  },
    {  5120,   1024, "v_proj"  },
    {  5120,   5120, "q_proj"  },
    {  5120,   5120, "o_proj"  },
    {  5120,  17408, "gate"    },
    {  5120,  17408, "up"      },
    { 17408,   5120, "down"    },
    {  5120, 151936, "lm_head" },
};
static const int qwen3_14b_num_mmq_shapes =
    sizeof(qwen3_14b_mmq_shapes) / sizeof(qwen3_14b_mmq_shapes[0]);

// Typical prefill batch sizes captured in the nsys traces.
static const int qwen3_14b_mmq_batches[] = { 8, 16, 32 };
static const int qwen3_14b_num_mmq_batches =
    sizeof(qwen3_14b_mmq_batches) / sizeof(qwen3_14b_mmq_batches[0]);
