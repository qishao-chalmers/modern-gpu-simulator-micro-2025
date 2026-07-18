// Qwen3 decode GEMV (mul_mat_vec_q) shape tables — matrix-vector only, no layer fusion.
//   dst[N] = W[N x K] * y[K]   with W in {Q8_0, Q4_K, Q2_K}, y in q8_1.
// Model dims:
//   8B:  H=4096, n_q=32, n_kv=8, hd=128, F=12288, vocab=151936
//   14B: H=5120, n_q=40, n_kv=8, hd=128, F=17408, vocab=151936
#pragma once

struct gemv_shape {
    int         K;
    int         N;
    const char *name;
};

static const gemv_shape qwen3_8b_gemv[] = {
    { 4096,   1024, "k_proj"  },
    { 4096,   1024, "v_proj"  },
    { 4096,   4096, "q_proj"  },
    { 4096,   4096, "o_proj"  },
    { 4096,  12288, "gate"    },
    { 4096,  12288, "up"      },
    { 12288,  4096, "down"    },
    { 4096, 151936, "lm_head" },
};
static const int qwen3_8b_gemv_num =
    sizeof(qwen3_8b_gemv) / sizeof(qwen3_8b_gemv[0]);

static const gemv_shape qwen3_14b_gemv[] = {
    { 5120,   1024, "k_proj"  },
    { 5120,   1024, "v_proj"  },
    { 5120,   5120, "q_proj"  },
    { 5120,   5120, "o_proj"  },
    { 5120,  17408, "gate"    },
    { 5120,  17408, "up"      },
    { 17408,  5120, "down"    },
    { 5120, 151936, "lm_head" },
};
static const int qwen3_14b_gemv_num =
    sizeof(qwen3_14b_gemv) / sizeof(qwen3_14b_gemv[0]);

static inline const gemv_shape * gemv_shapes_for_model(const char *model, int *n_out) {
    if (model[0] == '8') {
        *n_out = qwen3_8b_gemv_num;
        return qwen3_8b_gemv;
    }
    *n_out = qwen3_14b_gemv_num;
    return qwen3_14b_gemv;
}

#include <cstring>

static inline int gemv_shape_lookup(const gemv_shape *tbl, int n, const char *name,
                                    int *K, int *N) {
    for (int i = 0; i < n; ++i) {
        if (name && tbl[i].name && std::strcmp(tbl[i].name, name) == 0) {
            *K = tbl[i].K;
            *N = tbl[i].N;
            return 0;
        }
    }
    return -1;
}
