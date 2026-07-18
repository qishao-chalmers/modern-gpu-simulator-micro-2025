# Representative LLM decode kernels — catalog

Self-contained, launchable microbenchmarks for the handful of CUDA kernels that make up a
llama.cpp **decode** step. One kernel per folder, shaped to **Qwen3-14B**. See
[`DESIGN.md`](DESIGN.md) for *why* this harness exists; this file describes *what each
kernel is, its Qwen3-14B dimensions, and how many times it runs per decoder layer*.

The `layer_decode/` folder assembles these into a full layer; see
[`layer_decode/README.md`](layer_decode/README.md). The composition identity:

```
model_time  ≈  n_layers · layer_time(bits)  +  lm_head(bits)
layer_time(bits)  =  nongemm_time (run once)  +  gemm_time(bits)   (re-run per weight bit-width)
```

## Qwen3-14B dimensions

| symbol | meaning | value |
|---|---|---|
| `H`   | hidden size            | 5120 |
| `n_q` | query heads            | 40 |
| `n_kv`| key/value heads (GQA)  | 8 |
| `hd`  | head dim               | 128 |
| `QD`  | `n_q·hd` (Q/O width)   | 5120 |
| `KD`  | `n_kv·hd` (K/V width)  | 1024 |
| `F`   | FFN intermediate       | 17408 |
| `S`   | KV sequence length     | 1088 |
| `vocab` | LM-head output       | 151936 |
| layers | decoder blocks        | 40 |

## Kernel catalog

| folder | kernel | what it does | source analogue |
|---|---|---|---|
| `mul_mat_vec_q/` | `mmvq` | batch-1 weight GEMV `dst[N]=W[N×K]·y[K]` (q8_0·q8_1, `__dp4a`) | `ggml-cuda/mmvq.cu` |
| `mul_mat_q/`     | `mmq`  | batched weight GEMM (tiled q8·q8), used when batch > 1 | `ggml-cuda/mmq.cu` |
| `mmvq_bitwidth/` | `mmvq` w/ bit arg | GEMV whose weight bytes/compute shrink with a bit-width arg (**software** compression) | — |
| `quantize_q8_1/` | `quantize_q8_1` | quantize an activation row to q8_1 before each matmul | `ggml-cuda/quantize.cu` |
| `rms_norm/`      | `rms_norm_f32` | RMSNorm over a row | `ggml-cuda/norm.cu` |
| `rope/`          | `rope_neox` | NEOX RoPE: rotate pairs `(i,i+rot/2)` per head | `ggml-cuda/rope.cu` |
| `set_rows/`      | `k_set_rows_f16` | scatter the new token's K/V head-row into the f16 KV cache | `ggml-cuda` set_rows/cpy |
| `flash_attn/`    | `flash_attn_ext_vec` | online-softmax attention over `S` KV positions | `ggml-cuda/fattn-vec.cuh` |
| `swiglu/`        | `swiglu` | `silu(g)·u` FFN gate activation | `ggml-cuda/unary.cu` |
| `add/`           | `add_inplace` | residual add `x += y` | `ggml-cuda/binbcast.cu` |

All benches build two ways — `make` (native sm_90, trace target) and `make exec` (sm_70 +
`FUNCSIM_SAFE`, GPGPU-Sim execution-driven) — and take a `BATCH` env var that multiplies the
token/row dimension.

## Layer structure — kernel order

The batch-1 decode launch sequence of one Qwen3 decoder layer (from
`layer_decode/layer_all_bench.cu`): an **attention block** then an **FFN block**, each
ending in a residual add. `[w]` marks the weight matmuls that compression touches
(`N×K` weight shape shown); every other kernel is weight-bit-independent.

```
x ─────────────────────────────────────────────── residual ─┐  (attention block)
 │                                                           │
 ├─ rms_norm  (attn_norm, H=5120)                            │
 ├─ quantize ─▶ mmvq [w] Q   5120×5120  ─▶ rms_norm (q_norm) ─▶ rope_neox (ropeQ, f32) ─┐
 ├─ quantize ─▶ mmvq [w] K   5120×1024  ─▶ rms_norm (k_norm) ─▶ rope_neox (ropeK, f16) ─▶ set_rows K ─┐
 ├─ quantize ─▶ mmvq [w] V   5120×1024  ─────────────────────▶ rope_neox (V→f16 cast) ─▶ set_rows V ─┤
 │                                                                                       ▼           ▼
 │                                          flash_attn_ext_vec (n_q=40, n_kv=8, hd=128, seq=1088) ◀──┘
 │                                                           │
 ├─ quantize ─▶ mmvq [w] O   5120×5120  ─▶ attn ────────────┘
 └─ add  (x += attn_out) ────────────────────────────────────▶ x
                                                              │
x ─────────────────────────────────────────────── residual ─┤  (FFN block)
 ├─ rms_norm  (ffn_norm, H=5120)                              │
 ├─ quantize ─▶ mmvq [w] gate 5120×17408 ─┐                   │
 ├─ quantize ─▶ mmvq [w] up   5120×17408 ─▶ swiglu ─▶ quantize ─▶ mmvq [w] down 17408×5120 ─┐
 └─ add  (x += down_out) ◀───────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                        (× 40 layers)  then LM head:
                                        rms_norm ─▶ quantize ─▶ mmvq [w] 5120×151936 ─▶ logits
```

Linear order (26 kernel launches/layer, `[w]` = weight matmul):

```
attn:  rms_norm → quant → [w]Q → rms_norm(q) → ropeQ
             → quant → [w]K → rms_norm(k) → ropeK → set_rows(K)
             → quant → [w]V → ropeV → set_rows(V)
             → flash_attn → quant → [w]O → add
ffn:   rms_norm → quant → [w]gate → quant → [w]up → swiglu → quant → [w]down → add
```

## Per-layer usage (batch-1 decode)

Launch order and count of one Qwen3-14B decoder layer (from `layer_decode/layer_all_bench.cu`):

| # | kernel | per layer | dimensions (each instance) |
|---|---|:--:|---|
| 1 | `rms_norm_f32` | **4** | attn_norm & ffn_norm: `ncols=H=5120`, 1 row (block 1024); q_norm: `ncols=hd=128`, `n_q=40` rows; k_norm: `ncols=128`, `n_kv=8` rows (block 256) |
| 2 | `quantize_q8_1` | **7** | 6× width `5120` (before Q,K,V,O,gate,up); 1× width `F=17408` (before down) |
| 3 | `mmvq` (GEMV) | **7** | Q `5120×5120`, K `5120×1024`, V `5120×1024`, O `5120×5120`, gate `5120×17408`, up `5120×17408`, down `17408×5120` (`N×K`) |
| 4 | `rope_neox` | **3** | ropeQ: `n_q=40` rows, `rot=128`, f32 out; ropeK: `n_kv=8` rows, `rot=128`, f16 out; V→f16 cast reuses the kernel (`pos=0`), 8 rows |
| 5 | `k_set_rows_f16` | **2** | K and V: `n_elem=KD=1024` f16 → cache of `S=1088` rows |
| 6 | `flash_attn_ext_vec` | **1** | `n_q=40`, `n_kv=8`, `hd=128`, `seq=S=1088` |
| 7 | `swiglu` | **1** | `n=F=17408` |
| 8 | `add_inplace` | **2** | attn residual & ffn residual: `n=H=5120` |

**Once per model** (LM head, after the 40 layers): `rms_norm_f32` (`H=5120`, 1 row) + `quantize_q8_1`
(width 5120) + `mmvq` `5120×151936`.

### GEMV weight shapes `N×K` (the QWC-affected matmuls)

| projection | N (out) | K (in) | note |
|---|---:|---:|---|
| Q      | 5120   | 5120  | `n_q·hd × H` |
| K, V   | 1024   | 5120  | `n_kv·hd × H` (GQA → small) |
| O      | 5120   | 5120  | `H × QD` |
| gate, up | 17408 | 5120 | `F × H` |
| down   | 5120   | 17408 | `H × F` |
| lm_head | 151936 | 5120 | `vocab × H` (once/model, dominant kernel) |

### Which kernels weight-compression touches

Only the **GEMV/GEMM** rows (`mmvq`/`mmq`, and `mmvq_bitwidth` for the software model) read
the compressed weights — re-simulate those per bit-width. The other seven kernels
(`quantize_q8_1`, `rms_norm`, `rope`, `set_rows`, `flash_attn`, `swiglu`, `add`) are
weight-bit-independent: run **once** and reuse. That is exactly the `layer_gemm_bench` /
`layer_nongemm_bench` split.

## Batch > 1

`BATCH=b` scales the token dimension of each kernel:
- `quantize`, `rms_norm`, `rope`, `set_rows`, `swiglu`, `add` → one extra row/token (linear).
- attention: `flash_attn` grid `z = batch·n_q`, each sequence its own KV cache.
- projections: switch from `mmvq` (GEMV) to `mmq` (GEMM), `M=batch` tokens.
