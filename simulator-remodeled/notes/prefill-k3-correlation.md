# Prefill k3 correlation — real H100 vs `SM90_H100` simulator

Correlation document for the **k3 `mul_mat_q` calibration gap** on llama.cpp prefill (H100 vs `SM90_H100` sim). Ground truth: nsys **kernel_id 35–37**.

| Item | Value |
|------|-------|
| Model | Qwen3-8B-Q8_0 (`embedding_length = 4096`, 36 layers, 32 heads) |
| Trace | `/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/` |
| Sim config | `SM90_H100/gpgpusim.config` @ **1.620 GHz** |
| Sim log (baseline) | `log/prefill/sim.log` |

---

## 1. What we ran

### Hardware trace capture (cluster)

```bash
LD_PRELOAD=./tracer_tool.so \
  /home/bsc/bsc747505/project/llama.cpp/build_release/bin/llama-bench \
  -m .../Qwen3-8B-Q8_0.gguf \
  -p 1024 -n 64 -b 2048 -ngl 99 --flash-attn 1 --no-warmup -r 1
```

Only **3 kernels** were retained in the protobuf trace (likely via kernel-limit env vars during tracing):

| uid | Trace file | Kernel | Real H100 (profiled) | nsys `kernel_id` |
|-----|------------|--------|----------------------|------------------|
| k1 | `kernel-1.trace` | `rms_norm_f32<1024>` | **16,078 ns** | **35** |
| k2 | `kernel-2.trace` | `quantize_mmq_q8_1<ds_layout0>` | **12,270 ns** | **36** |
| k3 | `kernel-3.trace` | `mul_mat_q<Q8_0, mmq_x=128>` | **102,621 ns** | **37** |

**Do not use** `kernel_id` 23/24/25 (17,348 / 12,714 / **297,103** ns) — same kernel names but a **later layer pass** (likely FFN; `mul_mat_q` ~3× longer). **Do not use** profile `kernel_id` 1/2/3 — early launches include graph/warmup artifacts (e.g. `rms_norm` at 516 µs).

### Simulator runs

Baseline and knob sweeps under `simulator-remodeled/`:

| Experiment | Config change | k3 cycles | k3 sim time | vs real (id 37) |
|------------|---------------|-----------|-------------|-----------------|
| **baseline** | `SM90_H100` default | **258,275** | 159,429 ns | **+55%** (too slow) |
| `prefill_best` | `SM90_H100_best` | 243,698 | 150,431 ns | +47% |
| `trate512` | `tensor_rate_per_cycle 512` | 284,048 | 175,338 ns | +71% |
| `sim_another_bk` | older L1D / memory config | **757,765** | 467,757 ns | **+356%** |
| k1/k2 (baseline) | — | 28,039 / 21,444 | ~17.3 / ~13.2 µs | +8% / +8% |

**Target to match hardware:** `102,621 ns × 1.620 GHz ≈ 166,246 cycles` (baseline sim has 258,275 → **+55% too many cycles**).

*Previously we wrongly used kernel_id 23–25 (297 µs k3); that made sim look 46% too fast.*

Scripts: `summarize_prefill_results.sh`, `run_prefill_experiments_all.sh`, per-variant `run_prefill_*.sh`.

---

## 2. Pipeline context (what k3 does in llama.cpp)

For each transformer layer, activations flow through RMS norm → quantize → quantized GEMM. The traced trio is one slice of that path:

```
input [4096 × T] f32
    │
    ▼  k1: rms_norm_f32
    │
    ▼  k2: quantize_mmq_q8_1     (f32 activations → Q8_1 MMQ tile layout)
    │
    ▼  k3: mul_mat_q<Q8_0>       (quantized GEMM, stream-K on H100)
    │      W_q [4096×4096] × act_q8_1 [4096×T] → Q [4096×T] f32
    │
    ▼  (optional, not in this 3-kernel trace)
       mul_mat_q_stream_k_fixup   (partial-K reduction across SMs)
```

**Function:** `ggml_cuda_mul_mat_q()` in `llama.cpp/ggml/src/ggml-cuda/mmq.cu` launches the tiled int8 tensor-core matmul used for all Q8_0 weights (attention projections, FFN `ffn_up` / `ffn_gate`, etc.). The trace does not label which layer/op this is, but the shape **4096×4096** weight with **512-token** activations is consistent with a standard hidden-size projection in Qwen3-8B.

**Algorithm:** [Stream-K](https://arxiv.org/abs/2301.03598) partitioning — each H100 SM gets one CTA (`grid.x = 132`), and blocks cooperatively sweep the `(M, N, K)` tile space along K instead of launching one CTA per output tile.

---

## 3. Launch geometry (trace ↔ source)

From `dynamic_trace.pb` and `stats.csv`:

| | k1 rms_norm | k2 quantize | k3 mul_mat_q |
|--|-------------|-------------|--------------|
| Grid | 512×1×1 | 512×8×1 | **132×1×1** |
| Block | 1024×1×1 | 128×1×1 | **32×8×1** (256 threads) |
| Shared mem | 128 B | — | **57,856 B** |
| Registers | 31 | 20 | **224** |

**k3 ↔ `launch_mul_mat_q()` (`mmq.cuh`):**

- `block_nums_stream_k(nsm, 1, 1)` → **132 blocks** = H100 SM count
- `block_dims(warp_size, nwarps, 1)` → 32×8 = 256 threads
- Template `Li128` → **`mmq_x = 128`** tile width on Hopper (not 16; 16 is for smaller `mmq_x` specializations)

---

## 4. Dimensions — why M=4096, N=512, K=4096 for Qwen

### ggml tensor mapping (`mmq_args` in `mmq.cu`)

| Symbol | ggml field | Meaning | Value |
|--------|------------|---------|-------|
| **K** | `ne00` / `ncols_x` | Inner (hidden) dimension | **4096** |
| **M** | `ne01` / `nrows_x` | Rows of weight matrix | **4096** |
| **N** | `ne1` / `ncols_dst` | Output columns (tokens) | **512** |

**GEMM:** `C[M,N] = W[M,K] × A[K,N]` → **4096 × 512 × 4096**

### Why these numbers for Qwen3-8B

| Factor | Explanation |
|--------|-------------|
| **4096** | Qwen3-8B `embedding_length` from GGUF metadata |
| **4096 × 4096 weight** | Square projection (Q/K/V, `ffn_up`, etc.) at full hidden size |
| **512 tokens (not 1024)** | llama-bench default **`n_ubatch = 512`**. Prefill processes the prompt in 512-token chunks even with `-p 1024` |
| **`-b 2048`** | Max batch **capacity** for the context, not the active column count in this kernel |
| **512 blocks on k1/k2** | `rms_norm` and `quantize_mmq_q8_1` grid-x = **512** → confirms **T = 512** for this ubatch |

Full 1024-token prefill would run **two** `mul_mat_q` launches per layer (2×512). This trace captures **one** ubatch.

### Tile breakdown (k3)

With `mmq_x = mmq_y = 128`:

```
ntx = ⌈N / 128⌉ = ⌈512/128⌉ = 4
nty = ⌈M / 128⌉ = ⌈4096/128⌉ = 32
K-blocks per tile = K / qk = 4096 / 32 = 128     (Q8_0 block size qk=32)

Output tiles = ntx × nty = 128
Stream-K work units = 128 tiles × 128 K-blocks = 16,384
```

**Stream-K fixup:** `ntx×nty = 128` is not divisible by 132 SMs → `mul_mat_q_stream_k_fixup` is required (not in the 3-kernel trace window).

---

## 5. Operations inside k3 (static SASS template)

One inner-loop body from `enhanced_execution_info.json` (6352 static instructions, `Li128`):

| Opcode (per loop iter) | Count | Role |
|------------------------|------:|------|
| `IMMA.16832.S8.S8` | **256** | int8×int8 tensor-core dot (16×8×32 per op) |
| `I2FP.F32.S32` | 1024 | int32 accum → fp32 |
| `FMUL.FTZ` | 1024 | apply block scales |
| `FFMA.FTZ` | 1024 | fused scale + accumulate to fp32 output |
| `LDS` / `STS` | 320 / 147 | shared-memory tile staging |
| `LDG.E` (const/global) | 136+ | load weight / activation tiles |
| `STG.E` | 128 | store fp32 results |

**Per IMMA:** 16×8×32 = **4096 int8 MACs** per warp.

**Dynamic trace:** 107,077,088 reported insts across 132×8 warps → ~16 static-loop iterations per warp, matching stream-K work partition (`16384 / (132×8) ≈ 15.5`).

**Important:** This is **not** a pure IMMA GEMM. For every 256 IMMA there are **3072 FP32 ops** (dequant / scale). The kernel is a **fused quantized matmul + dequant**, not dense int8 GEMM.

---

## 6. MACs and TOPS

### MAC count

```
MACs = M × N × K = 4096 × 512 × 4096 = 8,589,934,592  (~8.59×10⁹)
```

NVIDIA INT8 **TOPS** convention: 2 ops per MAC (multiply + add).

### Achieved throughput

| | Duration | INT8 TOPS | % of H100 peak (1979) |
|--|----------|-----------|------------------------|
| **Real H100** | 297.1 µs | **~58** | **~2.9%** |
| **Sim baseline** | 159.4 µs | **~108** | **~5.4%** |
| Sim `trate512` | 175.3 µs | ~98 | ~5.0% |
| Sim `sim_another_bk` | 467.8 µs | ~37 | ~1.9% |

Reference ceilings:

| Reference | TOPS | Notes |
|-----------|------|-------|
| H100 SXM5 datasheet (dense INT8) | 1979 | Theoretical peak |
| CUTLASS large INT8 GEMM | ~500–600 | ~25–30% peak, large square problems |
| IMMA ubench (1 warp, dep chain) | ~553 / warp | Not whole-GPU sustained |
| This kernel (real) | **~58** | Small N, mixed IMMA+FP, stream-K |

### Gap summary

| Metric | Real / Sim ratio |
|--------|------------------|
| Duration | sim **1.87× faster** (159 vs 297 µs) |
| Effective TOPS | sim **1.87× higher** (108 vs 58) |
| Cycles | 258,275 vs target 481,307 (**−46%**) |

---

## 7. Simulator behavior (baseline `log/prefill/sim.log`)

### k3 summary stats

| Stat | Value | Interpretation |
|------|------:|----------------|
| `gpu_sim_cycle` | 258,275 | Kernel-critical-path cycles |
| `gpu_sim_insn` | 1,170,378,752 | Dynamic insn count (remodeled expansion) |
| `gpu_ipc` | ~4531 | High — many parallel warp insts / cycle |
| `gpu_occupancy` | ~12.5% | Low per-kernel warp occupancy |
| `gpu_tot_sms_occupancy` | ~93.7% | SMs busy across the 3-kernel run |
| `gpu_stall_dramfull` | **2,186,115** | DRAM / memory back-pressure stalls |
| L1D miss rate | **~52%** per core | Half of L1D accesses miss |
| `MISS_QUEUE_FULL` (reads) | **~2.1M** | MSHR / miss-queue saturation |
| Issue stage issuing | **~29%** of cycles | |
| Stall: next stage N/A | **~47%** of cycles | Pipeline blocked waiting on memory / prior stages |
| L2 BW (k3 region) | ~1020 GB/s | High DRAM/L2 traffic |

k1/k2 for comparison: k1 `MISS_QUEUE_FULL` ~68k; k2 ~310k; k3 **~2.1M** — memory contention is **an order of magnitude worse** on k3.

### Remodel knobs (current baseline)

From `SM90_H100/gpgpusim.config`:

```
-tensor_rate_per_cycle 2048      # 32768/2048 = 16 cycles/IMMA in model
-tensor_latency 32
-gpgpu_cache:dl1 ... A:512:64,16:0,32   # MSHR 16, miss-queue 32
```

IMMA microbenchmark on real H100 (`config_tensor_imma16832`):

- Measured **~24 clk/IMMA** (implies rate ≈ 1365; config uses 2048 after `round_up_2n`)
- Throughput: **~0.66 IMMA/clk/SM** under synthetic full-SM load

---

## 8. Experiments and what they tell us

| Hypothesis | Knob | k3 effect | Conclusion |
|------------|------|-----------|------------|
| TC too fast | `tensor_rate` 2048→512 | +10% cycles | TC is **not** the main bottleneck |
| TC init / extra latency | `tinit*`, `textra*` | minimal on k3 | HMMA-only extras don't apply to IMMA |
| MSHR / miss queue | `mq32`, `mq64`, `mshr1024` | TBD / moderate | Right **class** of knob |
| Memory config (old) | `sim_another_bk` | **+193%** cycles | Memory model dominates k3 error |
| No post-compute | `nopc` variant | small | Not the main issue |

**Central finding:** Swinging **memory subsystem** parameters moves k3 by **tens of percent to 2×**; swinging **tensor_rate** moves it by **≤10%**. The −46% gap is primarily **under-modeled memory latency / contention**, not tensor-core throughput.

---

## 9. Why memory-bound, not compute-bound

A GEMM can still be memory-/latency-bound if it fails to saturate tensor cores.

### (a) Very low peak utilization (~3%)

Real hardware achieves **~58 TOPS vs 1979 peak**. Large dense INT8 GEMMs reach 25–30% peak; this kernel is an order of magnitude lower.

### (b) Tall-skinny shape (N ≪ M, K)

```
M = 4096,  N = 512,  K = 4096   →   N / M = 12.5%
```

Only **4 column tiles** (`ntx=4`) across **32 row tiles**. With stream-K and 132 SMs, SMs spend much time **streaming K** and reloading tiles rather than sustaining IMMA pipelines.

### (c) Mixed IMMA + heavy FP32 dequant

Per 256 IMMA: **3072** FP32 ops (I2FP/FMUL/FFMA) plus LDG/LDS/STS. Cycles go to **scalar FP and memory**, not just tensor cores.

### (d) Memory latency dominates in sim and hardware

| Evidence | Value |
|----------|-------|
| L1D miss rate | ~52% |
| `gpu_stall_dramfull` | 2.2M |
| `MISS_QUEUE_FULL` | 2.1M |
| Issue stalled (next stage) | ~47% of cycles |
| L2 bandwidth | ~1 TB/s (high traffic, not necessarily bandwidth-saturated) |

This profile is **latency / MSHR / queue contention**, not "compute units idle because DRAM BW is maxed."

### (e) Roofline placement

Rough traffic for one ubatch:

| Data | ~Size |
|------|------:|
| Q8_0 weights | ~17 MB (with scales) |
| Q8_1 activations | ~9 MB |
| f32 output | ~8 MB |
| **Total (naive)** | **~35–40 MB** |

```
Arithmetic intensity ≈ 2 × MACs / bytes ≈ 440 ops/byte
H100 ridge point     ≈ 500–600 ops/byte  (at ~3.3 TB/s vs 1979 TOPS)
```

The kernel sits **near the ridge** but **below** the compute roof because:

- reuse is imperfect (quantized tiles, stream-K striping),
- IMMA duty cycle is low (FP + load/store between TC ops),
- memory queues stall the issue stage.

**Analogy:** It is a GEMM arithmetically, but behaves like a **memory-latency-limited fused kernel**, not a square CUTLASS GEMM.

---

## 10. Correlation diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  llama-bench prefill (Qwen3-8B-Q8_0, ubatch=512)                        │
└─────────────────────────────────────────────────────────────────────────┘
         │                          │                          │
         ▼                          ▼                          ▼
    k1 rms_norm              k2 quantize_mmq            k3 mul_mat_q
    512×4096 f32             f32→Q8_1                  W[4096×4096]×A[4096×512]
    17.3 µs real ✓           12.7 µs real ✓             297 µs real
    17.3 µs sim ✓            13.2 µs sim ✓              159 µs sim ✗ (−46%)

                              MACs = 8.59×10⁹
                              Real:  ~58 TOPS  (3% peak)
                              Sim:  ~108 TOPS  (5% peak)  ← 1.87× too optimistic

┌─────────────────────── SIMULATOR GAP MECHANISM ───────────────────────┐
│  tensor_rate / IMMA latency     →  ≤10% cycle change   (ruled out)      │
│  L1D misses + MISS_QUEUE_FULL   →  2.1M events on k3                  │
│  gpu_stall_dramfull             →  2.2M stalls                          │
│  issue stage blocked            →  47% cycles (next stage N/A)          │
│  memory config (sim_another_bk) →  +193% cycles (bracket real time)     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Open calibration actions

1. **Bracket k3 between baseline and `sim_another_bk`** — tune L1D MSHR (`16`), miss queue (`32`), associativity, and DRAM scheduling until sim ≈ 481k cycles.
2. **Stop `tensor_rate` sweeps** for k3 — confirmed low sensitivity.
3. **Account for ubatch** when comparing to full-prompt benchmarks (this trace = 512 of 1024 tokens).
4. **Optional:** include `mul_mat_q_stream_k_fixup` in trace for end-to-end layer time.
5. **Optional:** fix IMMA ubench `round_up_2n` (print rate ≈ 1365 vs configured 2048).

---

## 12. Key file references

| Topic | Path |
|-------|------|
| Trace index | `gpu_traces/modern/prefill_traces/dynamic_trace.pb` |
| Per-kernel stats | `gpu_traces/modern/prefill_traces/stats.csv` |
| Static SASS template | `.../extra_info/enhanced_execution_info.json` |
| MMQ kernel / launch | `llama.cpp/ggml/src/ggml-cuda/mmq.cuh`, `mmq.cu` |
| Quantize launch (N=512) | `llama.cpp/ggml/src/ggml-cuda/quantize.cu` |
| Sim baseline log | `simulator-remodeled/log/prefill/sim.log` |
| H100 config | `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config` |
| IMMA microbench | `util/tuner/GPU_Microbenchmark/ubench/core/config_tensor_imma16832/` |
| Experiment summary | `summarize_prefill_results.sh` |

---

## 13. `mul_mat_q` memory pipeline on H100 (no TMA / no ping-pong)

Source: `llama.cpp/ggml/src/ggml-cuda/mmq.cuh` (`mul_mat_q_process_tile`, `vec_dot_q8_0_q8_1_mma`), static SASS in `enhanced_execution_info.json`, dynamic trace uid 3.

### What the kernel uses

`mul_mat_q<Q8_0, Li128>` on H100 uses the **classic MMQ path** — **IMMA + shared-memory tiles** — not Hopper **TMA** (`UTMALDG` / `UTMASTG`) or **WGMMA**. The trace confirms: `LDG`/`STS` → `LDS` → `IMMA`, with `BSSY`/`BSYNC` barriers. No `LDGSTS`, no `UTMA*`, no `cp.async`.

### Main K-loop (`mul_mat_q_process_tile`)

```cpp
for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
    {
        // copy activation tile (first K-half) into shared tile_y
        ...
    }
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, 0);
    __syncthreads();
    {
        // reload tile_y with second K-half
        ...
    }
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();
}
```

`ITER_K = 256` (see `MMQ_ITER_K`); each outer step is **synchronous load → barrier → compute**, repeated for both K-halves of the activation tile.

### Per K-tile flow (current tile only)

1. **`load_tiles`** — global → shared `tile_x` (quantized weights). For Q8_0 MMA, both K-halves of the weight tile are written into shared at once.
2. **Copy activations** — global → shared `tile_y` (first K-half).
3. **`__syncthreads()`**
4. **`vec_dot`** — `load_ldmatrix` from shared → registers → **IMMA** (+ FP32 dequant: `I2FP`/`FMUL`/`FFMA`).
5. **Reload `tile_y`** (second K-half), sync, second **`vec_dot`**.

When IMMA executes, operands are already in **shared memory** (loaded to registers via LDSM). IMMA does **not** read global memory directly.

### No ping-pong across K-tiles

Shared memory has a **single** buffer set:

```cpp
int * tile_y = data_mul_mat_q + mmq_x;
int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);
```

There is no `tile_x[0]` / `tile_x[1]` double buffer. The next K-chunk is loaded only after the current `vec_dot` completes and barriers finish.

Additional facts:

- No `cp.async` / `LDGSTS` anywhere in `mmq.cuh`.
- Weights: both K-halves staged into shared in one `load_tiles` call.
- Activations: still loaded in **two serial phases** around the first `vec_dot`.

**Pattern:** stage-then-compute for the current tile, **not** overlap next-tile prefetch with current-tile IMMA.

### Comparison to CUTLASS / Hopper TMA GEMMs

| Pattern | CUTLASS / Hopper TMA GEMM | llama `mul_mat_q` |
|---------|---------------------------|-------------------|
| Double-buffered shared tiles | Often yes (2+ stages) | **No** (one `tile_x`, one `tile_y`) |
| Async global→shared | TMA / `cp.async` | **Sync** `LDG` → `STS` |
| IMMA while loading next tile | Yes (software pipeline) | **No** — barrier between load and compute |
| Data in shared before IMMA | Yes | **Yes, for current tile only** |

### Implication for the k3 sim gap (+47–55% vs real)

This is **not** mainly a “missing TMA ping-pong” issue. The NVBit trace already captures the real instruction pattern: synchronous tile loads + barriers + IMMA from shared.

Where the simulator can still be pessimistic:

1. **Barriers** — `__syncthreads` / `BSYNC` are modeled; real hardware may keep more useful work in flight across warps.
2. **Limited cross-warp overlap** — even without software ping-pong, real GPUs hide some load latency via other warps; sim shows ~47% issue-stage stalls (“next stage not available”) and heavy `MISS_QUEUE_FULL` / `gpu_stall_dramfull` on k3.
3. **Within-tile path is structurally correct** — `LDG`→`STS`→`LDS`→`IMMA` matches source and trace; the remaining gap points at **memory latency / MSHR / issue-path modeling**, not a hidden TMA prefetch buffer the tracer failed to record.

**Bottom line:** Data **is** in shared memory before IMMA starts for each tile, but there is **no ping-pong prefetch of the next tile** while IMMA runs on the current one. That limits latency hiding and helps explain why real H100 only reaches **~8.5% of peak** (~167 INT8 TOPS) on this kernel — but sim being **+47–55% slower** (baseline ~108 TOPS effective) still implicates **memory-queue / remodel issue modeling** more than missing TMA.

---

## 14. Ping-pong vs what `mul_mat_q` actually does

Ping-pong / double-buffering is standard in CUTLASS and Hopper TMA GEMMs. It is reasonable to expect llama.cpp to use it — and **it does elsewhere** — but **`mul_mat_q` does not**.

### llama.cpp uses `cp.async` pipelining in other kernels

Flash attention (`fattn-mma-f16.cuh` + `cp-async.cuh`) implements multi-stage prefetch:

- `nstages` parameter and `cp_async_cg_16` loads
- `cp_async_wait_all()` between stages
- preload of the next K/V tile while computing on the current tile

`mmq.cuh` has **no** references to `cp.async`, `cp_async`, `nstages`, or `wait_all`. The MMQ path and flash-attn path are intentionally different.

### What looks like “double buffering” but is not

| Pattern in MMQ | What it actually is |
|----------------|---------------------|
| `2*MMQ_TILE_NE_K` in `tile_x` layout | **Two K-halves in one shared buffer** (offsets `+0` and `+MMQ_TILE_NE_K`), not alternating buffers |
| Two `vec_dot` calls per `kb0` step | Process K-half 0 then K-half 1 on the **same** `tile_x`; `tile_y` is **overwritten** between calls |
| `tile_A[ntx][MMQ_TILE_NE_K/QI8_0]` in `vec_dot` | **Register staging** (LDSM from shared → regs before IMMA), after `__syncthreads` |
| Stream-K (`grid.x = 132` on H100) | **Inter-SM** work partitioning, not intra-block load/compute overlap |

### Actual pipeline (one `kb0` iteration)

```
global ──LDG/STS──► tile_x, tile_y  (sync)
                         │
                    __syncthreads__
                         │
              LDSM ──► regs ──► IMMA  (vec_dot, k00=0)
                         │
                    __syncthreads__
                         │
              reload tile_y only    (sync)
                         │
              LDSM ──► regs ──► IMMA  (vec_dot, k00=32)
                         │
                    __syncthreads__
                         │
              next kb0: reload tile_x + tile_y  (no overlap with prior IMMA)
```

Trace/SASS for k3 (`Li128`): `LDG` → `STS` → `BSYNC` → `LDS`/`LDSM` → `IMMA`. Barriers separate load and compute phases; no `LDGSTS`, no `cp.async`, no TMA.

### Why MMQ may omit software ping-pong

| Factor | Notes |
|--------|-------|
| Shared memory | k3 uses **~57 KiB** / block (`mmq_x=128`); true double-buffering of `tile_x`+`tile_y` ≈ **2×** that |
| Kernel shape | Tall-skinny `4096×512×4096` + stream-K; low sustained TC duty cycle (~8.5% peak on real H100) |
| Fused dequant | ~3072 FP32 ops per 256 IMMA — not a clean dense GEMM inner loop |
| Template explosion | MMQ supports many quant types; flash-attn got the `cp.async` investment |

### Verdict table

| Claim | Verdict |
|-------|---------|
| Ping-pong is standard for high-performance GEMM | **Yes** |
| llama.cpp never uses async prefetch | **No** — flash-attn does (`cp.async`, multi-stage) |
| `mul_mat_q` uses shared-memory ping-pong | **No** — serial load → sync → compute per tile |
| Operands in shared before IMMA (current tile) | **Yes** |
| Next tile prefetched during IMMA | **No** — `__syncthreads` / `BSYNC` block overlap |
| Sim gap mainly from “missing ping-pong in trace” | **Unlikely** — trace matches the synchronous pattern in source |

### Simulator note (LDG cache path)

k3 weight loads use `LDG.E.CONSTANT` (no `STRONG.GPU`). In trace-driven sim these are treated as **global `LDG` → L1D** (`CACHE_ALL`), not constant cache. See §13 and `trace_driven.cc` — only `LDC` routes to L1C; `LDG.E.CONSTANT` is not special-cased. A separate calibration knob is `-gpgpu_gmem_skip_L1D` (force L1 bypass for all global loads).
