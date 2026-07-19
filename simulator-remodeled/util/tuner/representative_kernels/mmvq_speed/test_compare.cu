// test_compare — shared-init error + speed harness.
//
// Pulls in mmvq_speed.cu (MMVQ_NO_MAIN). Compares:
//   • SPEED mode1 / mode2  (striped Q||R)
//   • packed baseline      (configurable: q8_0 | q2_k | q3_k | q4_k)
//                           ≡ ../mmvq_kquant/mmvq_kquant K N <baseline>
//   • packed same-Q        (packed layout, same quant as SPEED Q)
//                           — mode1 should match this exactly
//
// Build:  make test_compare
// Run:    ./test_compare <G> <q> <r> <baseline> <K> <N>
//   e.g.  ./test_compare 8 q4_k q4_k q8_0 8192 8192
//         ./test_compare 8 q2_k q2_k q4_k 8192 8192
// Env:    MMVQ_SEED=42

#define MMVQ_NO_MAIN
#include "mmvq_speed.cu"

#include <vector>
#include <cstring>

static uint32_t g_rng;

static uint32_t rng_u32() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
static float rng_u01() { return (rng_u32() >> 8) * (1.0f / 16777216.0f); }

static void set_block_d_q8_0(block_q8_0 *b, float d) {
#ifdef FUNCSIM_SAFE
    b->d = d;
#else
    b->d = __float2half(d);
#endif
}
static void set_block_ds_q8_1(block_q8_1 *b, float d, float s) {
#ifdef FUNCSIM_SAFE
    b->d = d; b->s = s;
#else
    b->ds = __floats2half2_rn(d, s);
#endif
}
static void set_block_dm_q4(block_q4_K *b, float d, float dmin) {
#ifdef FUNCSIM_SAFE
    b->d = d; b->dmin = dmin;
#else
    b->dm = __floats2half2_rn(d, dmin);
#endif
}
static void set_block_dm_q2(block_q2_K *b, float d, float dmin) {
#ifdef FUNCSIM_SAFE
    b->d = d; b->dmin = dmin;
#else
    b->dm = __floats2half2_rn(d, dmin);
#endif
}
static void set_block_d_q3(block_q3_K *b, float d) {
#ifdef FUNCSIM_SAFE
    b->d = d;
#else
    b->d = __float2half(d);
#endif
}

static void fill_wy(float *W, float *y, int K, int N) {
    for (int i = 0; i < K; ++i) y[i] = 0.05f + 0.95f * rng_u01();
    for (size_t i = 0; i < (size_t)N * (size_t)K; ++i) W[i] = 0.05f + 0.95f * rng_u01();
}

static void pack_q8_0_row(block_q8_0 *dst, const float *row, int K) {
    const int bpr = K / QK8_0;
    for (int b = 0; b < bpr; ++b) {
        const float *x = row + b * QK8_0;
        float amax = 0.0f;
        for (int i = 0; i < QK8_0; ++i) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax / 127.0f;
        if (d < 1e-8f) d = 1e-8f;
        set_block_d_q8_0(&dst[b], d);
        for (int i = 0; i < QK8_0; ++i) {
            int q = (int)lrintf(x[i] / d);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            dst[b].qs[i] = (int8_t)q;
        }
    }
}

static void pack_q8_1_vec(block_q8_1 *dst, const float *y, int K) {
    const int nb = K / QK8_1;
    for (int b = 0; b < nb; ++b) {
        const float *x = y + b * QK8_1;
        float amax = 0.0f;
        for (int i = 0; i < QK8_1; ++i) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax / 127.0f;
        if (d < 1e-8f) d = 1e-8f;
        float sum = 0.0f;
        for (int i = 0; i < QK8_1; ++i) {
            int q = (int)lrintf(x[i] / d);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            dst[b].qs[i] = (int8_t)q;
            sum += (float)q;
        }
        set_block_ds_q8_1(&dst[b], d, d * sum);
    }
}

static int kquant_levels(quant_type q) {
    if (q == QUANT_Q2_K) return 3;
    if (q == QUANT_Q3_K) return 7;
    return 15;
}

static void pack_kquant_block(quant_type ty, char *blk, const float *x256, float *dequant_out) {
    const int levels = kquant_levels(ty);
    float amax = 0.0f;
    for (int i = 0; i < QK_K; ++i) amax = fmaxf(amax, fabsf(x256[i]));
    float d = amax / (float)levels;
    if (d < 1e-8f) d = 1e-8f;

    if (ty == QUANT_Q4_K) {
        block_q4_K *b = (block_q4_K *)blk;
        memset(b, 0, sizeof(*b));
        set_block_dm_q4(b, d, 0.0f);
        for (int j = 0; j < K_SCALE_SIZE; ++j) b->scales[j] = 1;
        for (int i = 0; i < QK_K; i += 2) {
            int q0 = (int)floorf(x256[i] / d + 1e-6f);
            int q1 = (int)floorf(x256[i + 1] / d + 1e-6f);
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            b->qs[i / 2] = (uint8_t)(q0 | (q1 << 4));
            if (dequant_out) {
                dequant_out[i] = (float)q0 * d;
                dequant_out[i + 1] = (float)q1 * d;
            }
        }
    } else if (ty == QUANT_Q2_K) {
        block_q2_K *b = (block_q2_K *)blk;
        memset(b, 0, sizeof(*b));
        set_block_dm_q2(b, d, 0.0f);
        for (int j = 0; j < QK_K / 16; ++j) b->scales[j] = 0x11;
        for (int i = 0; i < QK_K; i += 4) {
            int q[4];
            for (int t = 0; t < 4; ++t) {
                q[t] = (int)floorf(x256[i + t] / d + 1e-6f);
                if (q[t] < 0) q[t] = 0; if (q[t] > 3) q[t] = 3;
                if (dequant_out) dequant_out[i + t] = (float)q[t] * d;
            }
            b->qs[i / 4] = (uint8_t)(q[0] | (q[1] << 2) | (q[2] << 4) | (q[3] << 6));
        }
    } else {
        block_q3_K *b = (block_q3_K *)blk;
        memset(b, 0, sizeof(*b));
        set_block_d_q3(b, d);
        for (int j = 0; j < 12; ++j) b->scales[j] = 0x11;
        memset(b->hmask, 0, sizeof(b->hmask));
        for (int i = 0; i < QK_K; ++i) {
            int q = (int)floorf(x256[i] / d + 1e-6f);
            if (q < 0) q = 0; if (q > 7) q = 7;
            const int low = q & 3;
            const int hi = (q >> 2) & 1;
            b->qs[i / 4] |= (uint8_t)(low << (2 * (i % 4)));
            if (hi) b->hmask[i / 8] |= (uint8_t)(1u << (i % 8));
            if (dequant_out) dequant_out[i] = (float)q * d;
        }
    }
}

// Pack SPEED striped buffer; optionally mirror identical Q blocks into packed_q (kquant layout).
static void pack_striped_from_W(char *striped, char *packed_q_out,
                                const float *W, int K, int N, int G,
                                quant_type q_ty, quant_type r_ty, int with_residual) {
    const size_t q_blk = quant_block_bytes(q_ty);
    const size_t r_blk = quant_block_bytes(r_ty);
    const size_t strip_bytes = (size_t)G * q_blk + (size_t)G * r_blk;
    const int n_strips = K / (G * QK_K);
    const int n_sb = K / QK_K;
    std::vector<float> dq(QK_K), resid(QK_K);

    for (int row = 0; row < N; ++row) {
        const float *wrow = W + (size_t)row * K;
        char *row_base = striped + (size_t)row * (size_t)n_strips * strip_bytes;
        char *pack_row = packed_q_out ? packed_q_out + (size_t)row * (size_t)n_sb * q_blk : nullptr;
        for (int sb = 0; sb < n_sb; ++sb) {
            const int strip = sb / G;
            const int local = sb % G;
            char *qb = row_base + (size_t)strip * strip_bytes + (size_t)local * q_blk;
            pack_kquant_block(q_ty, qb, wrow + sb * QK_K, dq.data());
            if (pack_row) memcpy(pack_row + (size_t)sb * q_blk, qb, q_blk);
            if (with_residual) {
                for (int i = 0; i < QK_K; ++i) {
                    float r = wrow[sb * QK_K + i] - dq[i];
                    if (r < 0.0f) r = 0.0f;
                    resid[i] = r;
                }
                char *rb = row_base + (size_t)strip * strip_bytes + (size_t)G * q_blk
                         + (size_t)local * r_blk;
                pack_kquant_block(r_ty, rb, resid.data(), nullptr);
            } else if (r_blk) {
                char *rb = row_base + (size_t)strip * strip_bytes + (size_t)G * q_blk
                         + (size_t)local * r_blk;
                memset(rb, 0, r_blk);
            }
        }
    }
}

// Independent packed baseline from W (may differ from SPEED Q type).
static void pack_baseline_from_W(char *packed, const float *W, int K, int N, quant_type base_ty) {
    if (base_ty == QUANT_Q8_0) {
        block_q8_0 *p = (block_q8_0 *)packed;
        for (int r = 0; r < N; ++r)
            pack_q8_0_row(p + (size_t)r * (K / QK8_0), W + (size_t)r * K, K);
        return;
    }
    const size_t blk = quant_block_bytes(base_ty);
    const int n_sb = K / QK_K;
    for (int row = 0; row < N; ++row) {
        const float *wrow = W + (size_t)row * K;
        char *rowp = packed + (size_t)row * (size_t)n_sb * blk;
        for (int sb = 0; sb < n_sb; ++sb)
            pack_kquant_block(base_ty, rowp + (size_t)sb * blk, wrow + sb * QK_K, nullptr);
    }
}

static size_t packed_bytes(quant_type ty, int K, int N) {
    if (ty == QUANT_Q8_0)
        return (size_t)N * (size_t)(K / QK8_0) * sizeof(block_q8_0);
    return (size_t)N * (size_t)(K / QK_K) * quant_block_bytes(ty);
}

static int parse_baseline(const char *s, quant_type *q) {
    if (!s) return -1;
    if (strcmp(s, "q8_0") == 0 || strcmp(s, "Q8_0") == 0) { *q = QUANT_Q8_0; return 0; }
    return parse_quant_k(s, q);
}

__global__ void gemv_fp32(const float *__restrict__ W, const float *__restrict__ y,
                          float *__restrict__ dst, int K, int N) {
    const int row = blockIdx.x;
    if (row >= N) return;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) acc += W[(size_t)row * K + k] * y[k];
    __shared__ float sm[256];
    const int tid = threadIdx.x;
    sm[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) dst[row] = sm[0];
}

// Packed K-quant GEMV — same layout/compute as mmvq_kquant for q2/q3/q4.
__global__ void mmvq_packed_kquant(const char *__restrict__ vx,
                                   const block_q8_1 *__restrict__ vy,
                                   float *__restrict__ dst,
                                   int K, int N, quant_type q_ty) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const size_t blk = quant_block_bytes(q_ty);
    const int n_sb = K / QK_K;
    const char *xrow = vx + (size_t)row * (size_t)n_sb * blk;
    const int nsub = K / QK8_1;
    const int sub_per = QK_K / QK8_1;
    float acc = 0.0f;
    for (int is = tid; is < nsub; is += NTHREADS) {
        const int sb = is / sub_per;
        const int sub = is % sub_per;
        acc += dot_k_sub(q_ty, xrow + (size_t)sb * blk, &vy[is], sub);
    }
#ifdef FUNCSIM_SAFE
    __shared__ float sm[NTHREADS];
    sm[tid] = acc;
    __syncthreads();
    for (int s = NTHREADS / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) dst[row] = sm[0];
#else
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float warp_sums[NWARPS];
    if (threadIdx.x == 0) warp_sums[threadIdx.y] = acc;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) s += warp_sums[w];
        dst[row] = s;
    }
#endif
}

struct err_stats { double max_abs, mean_abs, rmse; };

static err_stats diff_stats(const float *a, const float *b, int n) {
    err_stats e{0, 0, 0};
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n; ++i) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > e.max_abs) e.max_abs = d;
        sum += d;
        sum2 += d * d;
    }
    e.mean_abs = sum / n;
    e.rmse = sqrt(sum2 / n);
    return e;
}

static float time_kernel_ms(void (*fn)(void *), void *ctx, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn(ctx);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) fn(ctx);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / iters;
}

struct launch_ctx {
    dim3 grid, block;
    const char *striped;
    const char *packed;       // baseline or same-Q packed
    block_q8_0 *q8;
    block_q8_1 *y;
    float *dst, *W, *y_f;
    int K, N, G, mode;
    quant_type q_ty, r_ty, packed_ty;
    int use_q8;               // 1 => mmvq_q8_packed
};

static void launch_fp32(void *p) {
    auto *c = (launch_ctx *)p;
    gemv_fp32<<<c->grid, 256>>>(c->W, c->y_f, c->dst, c->K, c->N);
}
static void launch_baseline(void *p) {
    auto *c = (launch_ctx *)p;
    if (c->use_q8)
        mmvq_q8_packed<<<c->grid, c->block>>>(c->q8, c->y, c->dst, c->K, c->N);
    else
        mmvq_packed_kquant<<<c->grid, c->block>>>(c->packed, c->y, c->dst, c->K, c->N, c->packed_ty);
}
static void launch_speed(void *p) {
    auto *c = (launch_ctx *)p;
    mmvq_speed_gemv<<<c->grid, c->block>>>(c->striped, c->y, c->dst, c->K, c->N,
                                           c->G, c->mode, c->q_ty, c->r_ty);
}

static void print_err(const char *name, const err_stats &e) {
    printf("  %-36s  max_abs=%10.4g  mean_abs=%10.4g  rmse=%10.4g\n",
           name, e.max_abs, e.mean_abs, e.rmse);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        printf("Usage: %s <G> <q> <r> <baseline> <K> <N>\n", argv[0]);
        printf("  q,r       : SPEED quant / residual   (q2_k|q3_k|q4_k)\n");
        printf("  baseline  : packed kquant reference  (q8_0|q2_k|q3_k|q4_k)\n");
        printf("Examples:\n");
        printf("  %s 8 q4_k q4_k q8_0 8192 8192   # mode2 vs Q8 \"full\" model\n", argv[0]);
        printf("  %s 8 q2_k q2_k q4_k 8192 8192   # Q2+R2 vs Q4 full model\n", argv[0]);
        printf("  %s 8 q4_k q4_k q4_k 8192 8192   # mode1 vs same-prec kquant (+ baseline=q4)\n", argv[0]);
        return 1;
    }

    const int G = atoi(argv[1]);
    quant_type q_ty, r_ty, base_ty;
    if (parse_quant_k(argv[2], &q_ty) || parse_quant_k(argv[3], &r_ty)) {
        printf("q/r must be q2_k|q3_k|q4_k\n");
        return 1;
    }
    if (parse_baseline(argv[4], &base_ty)) {
        printf("baseline must be q8_0|q2_k|q3_k|q4_k\n");
        return 1;
    }
    const int K = atoi(argv[5]);
    const int N = atoi(argv[6]);
    if (G < 1 || N <= 0 || K % (G * QK_K) != 0) {
        printf("Need G>=1, N>0, K divisible by G*256\n");
        return 1;
    }
    if (base_ty == QUANT_Q8_0 && K % QK8_0 != 0) {
        printf("K must be divisible by 32 for q8_0 baseline\n");
        return 1;
    }

    const char *seed_env = getenv("MMVQ_SEED");
    g_rng = seed_env ? (uint32_t)strtoul(seed_env, nullptr, 10) : 42u;
    if (g_rng == 0) g_rng = 42u;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("test_compare  G=%d SPEED q=%s r=%s | baseline=%s | K=%d N=%d seed=%u\n",
           G, quant_name(q_ty), quant_name(r_ty), quant_name(base_ty), K, N, g_rng);
    printf("  packed baseline ≡ ../mmvq_kquant/mmvq_kquant %d %d %s\n", K, N, quant_name(base_ty));
    printf("  packed same-Q   ≡ ../mmvq_kquant/mmvq_kquant %d %d %s  (mode1 exact-match check)\n",
           K, N, quant_name(q_ty));

    std::vector<float> hW((size_t)N * K), hy(K);
    fill_wy(hW.data(), hy.data(), K, N);

    std::vector<block_q8_1> h_y(K / QK8_1);
    pack_q8_1_vec(h_y.data(), hy.data(), K);

    const size_t strip_bytes = striped_bytes(K, N, G, q_ty, r_ty);
    const size_t sameq_bytes = packed_bytes(q_ty, K, N);
    const size_t base_bytes = packed_bytes(base_ty, K, N);

    std::vector<char> h_m1(strip_bytes), h_m2(strip_bytes);
    std::vector<char> h_sameq(sameq_bytes), h_base(base_bytes);

    // mode1 striped + identical Q bytes into packed same-Q
    pack_striped_from_W(h_m1.data(), h_sameq.data(), hW.data(), K, N, G, q_ty, r_ty, 0);
    pack_striped_from_W(h_m2.data(), nullptr, hW.data(), K, N, G, q_ty, r_ty, 1);
    pack_baseline_from_W(h_base.data(), hW.data(), K, N, base_ty);

    float *d_W = nullptr, *d_yf = nullptr;
    block_q8_1 *d_y = nullptr;
    block_q8_0 *d_q8 = nullptr;
    char *d_m1 = nullptr, *d_m2 = nullptr, *d_sameq = nullptr, *d_base = nullptr;
    float *d_fp = nullptr, *d_base_o = nullptr, *d_sameq_o = nullptr;
    float *d_m1o = nullptr, *d_m2o = nullptr;

    CHECK(cudaMalloc(&d_W, hW.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_yf, hy.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_y, h_y.size() * sizeof(block_q8_1)));
    CHECK(cudaMalloc(&d_m1, strip_bytes));
    CHECK(cudaMalloc(&d_m2, strip_bytes));
    CHECK(cudaMalloc(&d_sameq, sameq_bytes));
    CHECK(cudaMalloc(&d_base, base_bytes));
    CHECK(cudaMalloc(&d_fp, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_base_o, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_sameq_o, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_m1o, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_m2o, (size_t)N * sizeof(float)));

    if (base_ty == QUANT_Q8_0) {
        CHECK(cudaMalloc(&d_q8, base_bytes));
        CHECK(cudaMemcpy(d_q8, h_base.data(), base_bytes, cudaMemcpyHostToDevice));
    }

    CHECK(cudaMemcpy(d_W, hW.data(), hW.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_yf, hy.data(), hy.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y.data(), h_y.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_m1, h_m1.data(), strip_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_m2, h_m2.data(), strip_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sameq, h_sameq.data(), sameq_bytes, cudaMemcpyHostToDevice));
    if (base_ty != QUANT_Q8_0)
        CHECK(cudaMemcpy(d_base, h_base.data(), base_bytes, cudaMemcpyHostToDevice));

    dim3 block(WARP_SIZE, NWARPS);
    dim3 grid(N);

    launch_ctx c{};
    c.grid = grid; c.block = block;
    c.y = d_y; c.W = d_W; c.y_f = d_yf;
    c.K = K; c.N = N; c.G = G; c.q_ty = q_ty; c.r_ty = r_ty;
    c.q8 = d_q8;

    // correctness launches
    c.dst = d_fp; time_kernel_ms(launch_fp32, &c, 1, 1);

    c.use_q8 = (base_ty == QUANT_Q8_0);
    c.packed = d_base; c.packed_ty = base_ty; c.dst = d_base_o;
    time_kernel_ms(launch_baseline, &c, 1, 1);

    c.use_q8 = 0; c.packed = d_sameq; c.packed_ty = q_ty; c.dst = d_sameq_o;
    time_kernel_ms(launch_baseline, &c, 1, 1);

    c.striped = d_m1; c.mode = 1; c.dst = d_m1o;
    time_kernel_ms(launch_speed, &c, 1, 1);
    c.striped = d_m2; c.mode = 2; c.dst = d_m2o;
    time_kernel_ms(launch_speed, &c, 1, 1);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> h_fp(N), h_base_o(N), h_sameq_o(N), h_m1o(N), h_m2o(N);
    CHECK(cudaMemcpy(h_fp.data(), d_fp, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_base_o.data(), d_base_o, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_sameq_o.data(), d_sameq_o, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_m1o.data(), d_m1o, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_m2o.data(), d_m2o, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

    char label_base[64], label_sameq[64];
    snprintf(label_base, sizeof(label_base), "baseline packed %s", quant_name(base_ty));
    snprintf(label_sameq, sizeof(label_sameq), "kquant packed %s (same Q)", quant_name(q_ty));

    printf("\n[1] vs fp32 GEMV (same W,y):\n");
    print_err(label_base, diff_stats(h_base_o.data(), h_fp.data(), N));
    print_err(label_sameq, diff_stats(h_sameq_o.data(), h_fp.data(), N));
    print_err("SPEED mode1 Q-only", diff_stats(h_m1o.data(), h_fp.data(), N));
    print_err("SPEED mode2 Q+R", diff_stats(h_m2o.data(), h_fp.data(), N));

    printf("\n[2] vs configurable baseline (packed %s ≡ mmvq_kquant):\n", quant_name(base_ty));
    print_err("SPEED mode1 vs baseline", diff_stats(h_m1o.data(), h_base_o.data(), N));
    print_err("SPEED mode2 vs baseline", diff_stats(h_m2o.data(), h_base_o.data(), N));

    printf("\n[3] mode1 vs same-precision packed kquant (expect ~0):\n");
    print_err("SPEED mode1 vs packed same-Q", diff_stats(h_m1o.data(), h_sameq_o.data(), N));

    printf("\n[4] other:\n");
    print_err("mode2 vs mode1", diff_stats(h_m2o.data(), h_m1o.data(), N));
    if (base_ty != q_ty)
        print_err("packed same-Q vs baseline", diff_stats(h_sameq_o.data(), h_base_o.data(), N));

    const int warm = 5, iters = 20;
    c.dst = d_fp; const float ms_fp = time_kernel_ms(launch_fp32, &c, warm, iters);

    c.use_q8 = (base_ty == QUANT_Q8_0);
    c.packed = d_base; c.packed_ty = base_ty; c.dst = d_base_o;
    const float ms_base = time_kernel_ms(launch_baseline, &c, warm, iters);

    c.use_q8 = 0; c.packed = d_sameq; c.packed_ty = q_ty; c.dst = d_sameq_o;
    const float ms_sameq = time_kernel_ms(launch_baseline, &c, warm, iters);

    c.striped = d_m1; c.mode = 1; c.dst = d_m1o;
    const float ms_m1 = time_kernel_ms(launch_speed, &c, warm, iters);
    c.striped = d_m2; c.mode = 2; c.dst = d_m2o;
    const float ms_m2 = time_kernel_ms(launch_speed, &c, warm, iters);

    const double fp_bytes = (double)N * K * sizeof(float);
    const double base_touch = (double)base_bytes;
    const double sameq_touch = (double)sameq_bytes;
    const double m1_touch = (double)N * (K / QK_K) * quant_block_bytes(q_ty);
    const double m2_touch = m1_touch + (double)N * (K / QK_K) * quant_block_bytes(r_ty);

    printf("\nSpeed (avg of %d iters, %d warmup):\n", iters, warm);
    printf("  %-36s  %8.2f us  %7.1f GB/s\n", "fp32 GEMV", ms_fp * 1e3, fp_bytes / (ms_fp * 1e-3) / 1e9);
    printf("  %-36s  %8.2f us  %7.1f GB/s\n", label_base, ms_base * 1e3, base_touch / (ms_base * 1e-3) / 1e9);
    printf("  %-36s  %8.2f us  %7.1f GB/s\n", label_sameq, ms_sameq * 1e3, sameq_touch / (ms_sameq * 1e-3) / 1e9);
    printf("  %-36s  %8.2f us  %7.1f GB/s\n", "SPEED mode1", ms_m1 * 1e3, m1_touch / (ms_m1 * 1e-3) / 1e9);
    printf("  %-36s  %8.2f us  %7.1f GB/s\n", "SPEED mode2", ms_m2 * 1e3, m2_touch / (ms_m2 * 1e-3) / 1e9);

    printf("\nSample dst[0]: fp32=%.4f  base=%.4f  sameQ=%.4f  m1=%.4f  m2=%.4f\n",
           h_fp[0], h_base_o[0], h_sameq_o[0], h_m1o[0], h_m2o[0]);

    const err_stats eq = diff_stats(h_m1o.data(), h_sameq_o.data(), N);
    if (eq.max_abs > 1e-3)
        printf("WARNING: mode1 vs packed same-Q max_abs=%.4g (expected ~0 — layout/math bug?)\n",
               eq.max_abs);
    else
        printf("OK: mode1 matches packed same-Q (max_abs=%.4g)\n", eq.max_abs);

    cudaFree(d_W); cudaFree(d_yf); cudaFree(d_y);
    if (d_q8) cudaFree(d_q8);
    cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_sameq); cudaFree(d_base);
    cudaFree(d_fp); cudaFree(d_base_o); cudaFree(d_sameq_o); cudaFree(d_m1o); cudaFree(d_m2o);
    return 0;
}
