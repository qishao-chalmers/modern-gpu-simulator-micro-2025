# Measured kernel durations — Qwen3-14B decode

Execution-driven GPGPU-Sim (`SM7_QV100`, PTX functional model), one decode token.
Metric: `gpu_tot_sim_cycle`. Logs live under
`accel-sim-framework-official/test/soft_compression_test/<kernel>/`.

> **Read the caveats before quoting these.** These are *relative* funcsim cycles on a
> V100 model, not H100 wall-time. The counter is **cumulative within a process**, so the
> per-kernel numbers below are *deltas* (kernel N total − kernel N−1 total). `flash_attn`
> uses the faithful ggml parallelization (threads↦KV positions, warp-shuffle dot reduction)
> with FP16 Q/K/V and an integer-bitfield FP16→FP32 decode (no `exp2f`): its seq=1024 cost
> fell 1,576,261 → 991,430 (parallel restructure) → **781,631** (cheap decode + FP16 Q) and
> is a legitimate model estimate (still V100/funcsim fidelity), no longer a naive-kernel
> artifact. The weight-matmul comparison across bit-widths remains the trustworthy signal.

## Per-kernel duration, mapped to the layer (batch 1)

`[w]` = weight matmul (compression-affected, see next table). All others are
compression-independent — measured once, reused for every bit-width.

### Attention block

| # | kernel | dims | cycles | ×/layer |
|--:|---|---|--:|:--:|
| 1 | `rms_norm` (attn_norm) | ncols=5120, 1 row | 9,097 | 1 |
| 2 | `quantize_q8_1` | width 5120 | 6,360 | (see below) |
| 3 | `[w]` Q proj | 5120×5120 | *table* | 1 |
| 4 | `rms_norm` (q_norm) | ncols=128, 40 rows | 6,309 | 1 |
| 5 | `rope_neox` (ropeQ, f32) | 40 rows × 128 | 6,948 | 1 |
| 6 | `[w]` K proj | 5120×1024 | *table* | 1 |
| 7 | `rms_norm` (k_norm) | ncols=128, 8 rows | 6,296 | 1 |
| 8 | `rope_neox` (ropeK, f16) | 8 rows × 128 | 7,096 | 1 |
| 9 | `set_rows` K | 1024 f16 → seq 1088 | 5,629 | 1 |
| 10 | `[w]` V proj | 5120×1024 | *table* | 1 |
| 11 | `rope_neox` (V→f16 cast) | 8 rows × 128 | ~7,096 | 1 |
| 12 | `set_rows` V | 1024 f16 → seq 1088 | 5,479 | 1 |
| 13 | `flash_attn_ext_vec` | n_q=40, n_kv=8, hd=128, **seq=1024**, FP16 Q/K/V | **781,631** | 1 |
| 14 | `[w]` O proj | 5120×5120 | *table* | 1 |
| 15 | `add` (attn residual) | n=5120 | 5,544 | 1 |

### FFN block

| # | kernel | dims | cycles | ×/layer |
|--:|---|---|--:|:--:|
| 16 | `rms_norm` (ffn_norm) | ncols=5120, 1 row | 9,097 | 1 |
| 17 | `quantize_q8_1` | width 5120 | 6,360 | (see below) |
| 18 | `[w]` gate proj | 5120×17408 | *table* | 1 |
| 19 | `[w]` up proj | 5120×17408 | *table* | 1 |
| 20 | `swiglu` | n=17408 | 5,973 | 1 |
| 21 | `quantize_q8_1` | width 17408 | 6,417 | 1 |
| 22 | `[w]` down proj | 17408×5120 | *table* | 1 |
| 23 | `add` (ffn residual) | n=5120 | 5,544 | 1 |

**`quantize_q8_1` per layer:** 6× width-5120 (before Q,K,V,O,gate,up) @ 6,360 + 1× width-17408
(before down) @ 6,417 = **44,577**.

## Weight matmuls `[w]` — cycles by bit-width

`mmvq_bitwidth` (whole-process cycles incl. fill; consistent with prior collection).
This is the compression study: fewer weight bits → fewer DRAM bytes → fewer cycles.

| projection | shape (K×N) | ×/layer | 8-bit | 3-bit | 2-bit |
|---|---|:--:|--:|--:|--:|
| Q, O    | 5120×5120   | 2 | 200,718 | 94,555 | 76,793 |
| K, V    | 5120×1024   | 2 | 58,875 | 41,558 | 38,167 |
| gate, up| 5120×17408  | 2 | 575,062 | 240,336 | 182,054 |
| down    | 17408×5120  | 1 | 566,860 | 242,364 | 178,180 |
| **lm_head** (once/model) | 5120×151936 | — | 4,618,575 | 1,807,343 | 1,295,762 |

**Weight-matmul cycles per layer** (Q+K+V+O+gate+up+down):

| | 8-bit | 3-bit | 2-bit |
|---|--:|--:|--:|
| gemm/layer | 2,236,170 | 995,262 | 772,208 |
| vs 8-bit | 1.00× | 0.445× | 0.345× |

## Layer & model roll-up

```
layer_time(bits) = nongemm_once + gemm(bits)
model_time       = 40 · layer_time(bits) + lm_head(bits)
```

Non-gemm+attention bucket, **once per layer** (batch 1, seq 1024):
rms_norm×4 (30,799) + quant×7 (44,577) + rope×3 (21,140) + set_rows×2 (11,108)
+ flash_attn (781,631) + swiglu (5,973) + add×2 (11,088) = **906,316**.

| | 8-bit | 3-bit | 2-bit |
|---|--:|--:|--:|
| gemm / layer | 2,236,170 | 995,262 | 772,208 |
| + nongemm / layer | 906,316 | 906,316 | 906,316 |
| **layer total** | **3,142,486** | **1,901,578** | **1,678,524** |
| × 40 layers | 125,699,440 | 76,063,120 | 67,140,960 |
| + lm_head | 4,618,575 | 1,807,343 | 1,295,762 |
| **model total** | **130,318,015** | **77,870,463** | **68,436,722** |
| vs 8-bit | 1.00× | 0.60× | 0.53× |

> `flash_attn` (faithful ggml kernel, FP16 + cheap decode) is still the largest single
> nongemm term (781,631/layer → 31.3M across 40 layers, ~86% of the nongemm bucket). It's a
> legitimate model estimate now, but attention is O(seq) and memory-bound, so on real HW its
> share is smaller. The gemm-only ratios (0.445×/0.345×) remain the clean compression signal.

## Batch scaling (measured points)

Non-gemm kernels, batch 1 → 8 (`BATCH` multiplies the token/row dim):

| kernel | batch 1 | batch 8 |
|---|--:|--:|
| ropeQ | 6,948 | 7,111 |
| ropeK | 7,096 | 7,097 |
| set_rows (per write) | 5,479 | 5,479 |
| swiglu | 5,973 | 7,020 |
| add | 5,544 | 5,802 |

Barely move — even at batch 8 these grids fit ~one wave on the 80-SM V100 model.

`flash_attn` (seq 1024) does scale, but sub-linearly (batch 1 only fills 40 of 80 SMs).
Batch-1 is the new ggml kernel (781,631, FP16+cheap decode); **batch 2–32 rows below are the OLD kernel and
need re-running** with the rewritten kernel:

| batch | grid.z | KV (MB) | cycles | kernel |
|--:|--:|--:|--:|---|
| 1 | 40 | 4.2 | 781,631 | new (ggml, FP16+cheap decode) |
| 2 | 80 | 8.4 | 1,849,658 | old — refresh |
| 4 | 160 | 16.8 | 1,955,157 | old — refresh |
| 8 | 320 | 33.6 | 1,986,973 | old — refresh |
| 16 | 640 | 67.1 | 2,077,071 | old — refresh |
| 32 | 1280 | 134.2 | 2,503,222 | old — refresh |

## FR dequant-offload — no-dequant differential

Models the SPEED Full-Precision Rebuilder taking the **dequantization** off the SMs (the FR
delivers ready values). Method: a `-DNO_DEQUANT` build keeps the **memory loads identical**
(same DRAM traffic) but removes the decode / scale-apply compute; the cycle delta is the
offload saving — this automatically accounts for overlap (a hidden dequant shows ~0 delta).
**Quantization is NOT offloaded** (needs a per-group max reduction the streaming FR lacks —
future task); only dequant.

| kernel (batch 1) | with dequant | NO_DEQUANT | saving |
|---|--:|--:|--:|
| `flash_attn_ext_vec` seq=1024 (latency-bound) | 781,631 | 659,655 | **−15.6%** |
| `mmvq_bw` 5120×5120 8-bit (BW-bound) | 145,813¹ | 138,438² | **−5.1%** |

¹ mmvq_bw kernel only = 200,718 total − 54,905 fill kernels.
² mmvq_bw kernel only = 193,309 total − 54,871 fill kernels. Confirms the ≈0 prediction:
the matmul is bandwidth-bound, so removing the dequant scale-apply barely moves cycles.

> **Read this as an upper bound.** flash_attn's 15.6% is in the low-MLP, batch-1,
> latency-bound regime where the software decode sits on the critical path. On real HW (and
> at larger batch/MLP) attention is bandwidth-bound and the dequant is hidden, so the saving
> shrinks toward zero. For the bandwidth-bound matmuls (~90% of decode) it's confirmed
> small — measured **−5.1%**. Takeaway: **dequant-offload buys little decode *latency*; its real payoff is energy and
> freeing SM compute** — the FR's cycle value stays the weight-BW reduction on draft steps.
> (`NO_DEQUANT` is timing-only: values aren't the true dequantized result, so verify mismatches.)

## Reproduce

```bash
# non-gemm bucket, one kernel:
cd soft_compression_test/<kernel> && bash run.sh
# batch / model sweep of the whole non-gemm+attn bucket:
representative_kernels/run_models.sh          # MODELS="14b 8b" BATCHES="1 8 16 32"
# weight matmuls at each bit-width:
cd soft_compression_test/gemv/<2|3|8>bit && bash run.sh
# FR dequant-offload differential (build with -DNO_DEQUANT, same loads, no decode):
nvcc <exec flags> -DNO_DEQUANT flash_attn/flash_attn_bench.cu   -o flash_attn_nodequant_exec
nvcc <exec flags> -DNO_DEQUANT mmvq_bitwidth/mmvq_bitwidth.cu    -o mmvq_bitwidth_nodequant_exec
```

Qwen3-8B (`H=4096 n_q=32 n_kv=8 hd=128 F=12288`) numbers: run `MODELS="8b" run_models.sh`
(not yet collected here).
