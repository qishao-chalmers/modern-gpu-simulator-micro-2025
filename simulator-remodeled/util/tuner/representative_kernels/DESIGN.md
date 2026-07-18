# Representative LLM Kernels — design notes

## Goal

Extract the handful of CUDA kernels that dominate llama.cpp **decode** into small,
self-contained, launchable microbenchmarks — one kernel per file, driven by a shape
table that matches a real model (Qwen3-14B to start). This lets us study the kernels in
the simulator **without** the full llama.cpp app (no CUDA graphs, no cuBLAS, no dlopen,
no 14B model load), and to evaluate **hardware ideas that don't exist yet** (e.g. a
quantization/decompression unit in the memory controller, KV-cache compression).

It is the LLM-kernel analogue of the existing `util/tuner/GPU_Microbenchmark` suite,
which already does this for primitive ops (FMA, mem bandwidth/latency, IMMA).

## Why this approach (vs. tracing full llama.cpp)

| approach | pros | cons |
|---|---|---|
| Trace full llama.cpp (current flow) | faithful SASS, real shapes | needs the target GPU to trace; whole-app capture; CUDA-graph handling |
| **Standalone kernel harness (this)** | tiny, controllable, no app infra; can run **execution-driven** for hypothetical HW | PTX-fidelity in exec-driven mode; shapes must be supplied |

Two complementary ways to use the harness:

1. **Real, accessible GPUs** (A100, L40S, B200-on-cloud): compile the harness for that
   arch and **trace it** (protobuf tracer) → trace-driven sim with the calibrated model.
   The harness makes tracing trivial (a few kernel launches, no graph/cuBLAS mess), so a
   short rented session captures all the shapes.
2. **Hypothetical / unavailable hardware** (future arch, or a *new MC design* such as the
   quantization unit): run **execution-driven** (the sim *is* the GPU; `libcudart.so`
   shim + PTX functional model). This is the only option when there is no silicon to
   trace — which is exactly the case for a novel-HW paper. Caveat: exec-driven is
   PTX-level, lower fidelity than the SASS trace-driven path; validate the baseline
   shapes against trace-driven numbers where they overlap.

## Why `mul_mat_vec_q` first

It is *the* decode workhorse — every projection (Q, K, V, O, gate, up, down) **and** the
LM head are this one kernel. In the Qwen3-14B decode trace it is also the single largest
kernel (the lm_head instance, ~528 µs).

Key facts established from the trace + source (`llama.cpp ggml/src/ggml-cuda/mmvq.cu`):

- It is **one template instantiation**, launched many times with **different runtime
  dimensions**. The duration differences in the trace are purely shape-driven, not code.
- Decode case is `ncols_dst = 1` → a **GEMV** `[1×K] · [K×N]`, reading **every weight
  once**. It is **memory-bandwidth-bound**, so:
  `duration ≈ N·K·bytes_per_weight / DRAM_BW`.
- Weights are **Q8_0** (`ggml_type 8`); the activation is quantized to **q8_1** by a
  separate `quantize_q8_1` kernel; the dot product is **int8×int8** (`vec_dot_q8_0_q8_1`).
- Two variants seen in the trace:
  - `mul_mat_vec_q<Q8_0, 1, has_fusion=false, multi_token=false>` — plain projections
    (Q/K/V/O, down, lm_head).
  - `mul_mat_vec_q<Q8_0, 1, has_fusion=true,  false>` — **SwiGLU-fused gate/up** (the
    `ggml_cuda_mm_fusion_args_device` arg carries gate/bias/glu_op).

## Qwen3-14B shape table (the experiment is the shapes)

Model: hidden=5120, n_q_heads=40, n_kv_heads=8 (GQA), head_dim=128, FFN intermediate=17408,
vocab=151936, n_layers=40. KV cache = f16, ≈160 KiB/token.

| op | K (ncols_x) | N (nrows_x) | fusion | Q8_0 weight | ~time @1.56 TB/s |
|---|---|---|---|---|---|
| K proj | 5120 | 1024 | no | 5.6 MB | ~4 µs |
| V proj | 5120 | 1024 | no | 5.6 MB | ~4 µs |
| Q proj | 5120 | 5120 | no | 28 MB | ~18 µs |
| O proj | 5120 | 5120 | no | 28 MB | ~18 µs |
| gate | 5120 | 17408 | yes (SwiGLU) | 94 MB | ~60 µs |
| up | 5120 | 17408 | yes | 94 MB | ~60 µs |
| down | 17408 | 5120 | no | 94 MB | ~60 µs |
| **lm_head** | 5120 | 151936 | no | **826 MB** | **~530 µs** |

(Sizes/durations corroborated by the weight-region detector and the nsys trace; the
lm_head 826 MB / 528 µs ⇒ ~1.56 TB/s, matching this H100 part's ~1.63 TB/s peak.)

## Harness design (per kernel)

```
representative_kernels/
  DESIGN.md
  Makefile                <- build all benches
  common/
    shapes_qwen3_14b_decode.h   <- rms / quant / fattn decode shape tables
    bench_common.h
    Makefile.inc
  mmvq_bitwidth/            <- weight bit-width sweep (exec-driven, done)
  mul_mat_vec_q/            <- Q8_0 GEMV (native / trace target)
  rms_norm/                 <- rms_norm_f32 decode norms
  quantize_q8_1/            <- activation quant before GEMV
  flash_attn/               <- flash_attn_ext_vec decode (simplified)
  <future: rope/, set_rows/, flash_attn_combine/>
```

Each `*_bench.cu`:
1. include the real ggml kernel + its `vec_dot` device fn + block structs (`block_q8_0`,
   `block_q8_1`) — copied/`#include`d from llama.cpp, kept minimal.
2. for each shape: `cudaMalloc` Q8_0 weight `[K×N]`, q8_1 activation `[K]`, f32 output
   `[N]`; fill with dummy data (values don't matter — memory-bound, shape-driven).
3. launch with `ncols_dst=1`, the model strides, and the matching grid
   (`grid.x = N / rows_per_cuda_block`).
4. time it (events) for a HW sanity check; under the simulator the timing comes from the
   model.

Keep data values arbitrary: duration depends on shape + bit-width, not contents.

## How this serves the MC-quantization / compression study

- Sweep the **same shapes at different weight bit-widths** (Q8_0 baseline → 3-bit / 2-bit
  compressed) and read off the DRAM-BW / latency delta directly — reusing the existing
  `-is_quantized_weight_dram_compression_enabled` lever and `gen_weight_regions.sh`.
- It generalizes to other models by swapping the `(K, N)` table (Qwen-7B/32B, Llama, …).
- For a **new MC design** with no silicon, run execution-driven so the simulator models
  the proposed unit on these representative kernels.

## Open questions / decisions

- **Exact vs analytic shapes:** use the analytic Qwen3-14B `(K,N)` above (clean, within
  rounding of the trace), or extract the literal per-launch `(K, N, strides)` from the
  trace kernel args for bit-exactness? Start analytic; add an extractor if needed.
- **Exec-driven fidelity:** confirm the remodeled-SM timing path is wired for exec-driven
  mode, or restrict exec-driven use to baseline/relative studies and keep absolute
  numbers on the trace-driven path.
- **CUDA version:** the exec-driven `libcudart.so` shim is an old ABI — keep the harness
  to bare kernel launches + `cudaMalloc`/`cudaMemcpy` (no cuBLAS/CUDA-12-only APIs).
- **Kernel coverage (Qwen3-14B decode):**
  | kernel | dir | llama.cpp source | status |
  |---|---|---|---|
  | `mul_mat_vec_q` / bit-width sweep | `mul_mat_vec_q/`, `mmvq_bitwidth/` | `mmvq.cu` | mmvq_bitwidth done |
  | `mul_mat_q` (batched/prefill GEMM) | `mul_mat_q/` | `mmq.cuh` | bench added (Q8_0 tiled int8 GEMM; M=ncols_dst batch; funcsim-safe, no MMA/stream-K; QWC-aware) |
  | **assembled decoder layer** | `layer_decode/` | full layer | all reps chained in batch-1 decode order → per-layer time. Split into `gemm`(QWC-affected, re-run per bits) + `nongemm`(run once) + `all`. `model ≈ n_layers·layer + lm_head`. See `layer_decode/README.md` |
  | `rms_norm_f32` | `rms_norm/` | `norm.cu` | bench added |
  | `quantize_q8_1` | `quantize_q8_1/` | `quantize.cu` | bench added |
  | `flash_attn_ext_vec` | `flash_attn/` | `fattn-vec.cuh` | simplified bench |
  | `rope_neox`, `k_set_rows`, `flash_attn_combine` | — | `rope.cu`, etc. | TODO |

  Each bench: `make` (sm_90 native) or `make exec` (sm_70 + `FUNCSIM_SAFE` for
  GPGPU-Sim). Or build everything at once:

  ```bash
  cd util/tuner/representative_kernels
  ./compile_all.sh           # native + exec
  ./compile_all.sh --exec    # GPGPU-Sim binaries only
  ```

  Run under official Accel-Sim with `sweep_shapes.sh` + QV100 config
  (same flow as `accel-sim-framework-official/local_test/`).
