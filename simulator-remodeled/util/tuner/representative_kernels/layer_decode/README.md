# layer_decode — assembled Qwen3 decoder layer for Accel-Sim

A Qwen3 decoder layer (batch-1 decode) assembled from the representative kernels, so
Accel-Sim estimates **per-layer** time. The whole model is a stack of identical layers:

```
model_time  ≈  n_layers · layer_time(bits)  +  lm_head(bits)
layer_time(bits)  =  nongemm_time (run once)  +  gemm_time(bits)   (re-run per QWC_BITS)
```

## Why three files

The 2/3-bit weight-compression (QWC) study only changes the **weight-loading matmuls**.
Everything else is identical regardless of weight bits, so we don't re-simulate it:

| file | kernels | role |
|---|---|---|
| `layer_gemm_bench.cu` | Q,K,V,O,gate,up,down (`mmvq`) | **QWC-affected** — re-run per `QWC_BITS` |
| `layer_nongemm_bench.cu` | rms_norm, quantize_q8_1, rope_neox, k_set_rows, flash_attn, swiglu, add | compression-independent — **run once, reuse** |
| `layer_all_bench.cu` | the full layer in order | reference / cross-check (`all ≈ gemm + nongemm`) |

All three `#include "layer_kernels.cuh"` so the kernels (and timings) are identical.

## Dimensions (the experiment)

Defaults = Qwen3-14B (`hidden=5120 qheads=40 kvheads=8 headdim=128 ffn=17408 seq=1088 vocab=151936`).
Flags: `--8b`, `--tiny`, `--layers N`, `--lmhead`, and per-dim `--hidden/--qheads/--kvheads/--headdim/--ffn/--seq/--vocab`.
Each kernel's grid derives from these, so changing a dim reshapes the matching kernel —
this is the per-kernel dimension sweep.

## Build

```bash
make           # native sm_90  -> layer_{all,gemm,nongemm}_bench       (trace target)
make exec      # sm_70 + FUNCSIM_SAFE -> *_exec                        (GPGPU-Sim exec-driven)
```

## Path A — execution-driven (this box, no GPU)  ← compression study

Functional sim runs every thread, so use small dims (`--tiny`, or modest `--ffn/--hidden`);
real FFN/lm_head dims are impractical here. Good for *relative* comparisons (baseline vs QWC).

```bash
LIB=<official>/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11050/debug
# from a dir with SM7_QV100 gpgpusim.config + config_volta_islip.icnt + *.xml:

# 1) non-gemm part — once
CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH=$LIB ./layer_nongemm_bench_exec --tiny | tee nongemm.log

# 2) gemm part — once per compression setting
CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH=$LIB                       ./layer_gemm_bench_exec --tiny | tee gemm_8bit.log
QWC_BITS=3 QWC_MIN_BYTES=100000 CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH=$LIB ./layer_gemm_bench_exec --tiny | tee gemm_3bit.log
QWC_BITS=2 QWC_MIN_BYTES=100000 CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH=$LIB ./layer_gemm_bench_exec --tiny | tee gemm_2bit.log
```

Sum `gpu_sim_cycle` over the layer's compute kernels (ignore the `fW/fF/fH` fill kernels),
then `layer_cycles(bits) = nongemm + gemm(bits)`.
(To see the QWC DRAM-read drop, use a NORMAL, small L2 and a weight that exceeds it —
sector L2 keeps small weights resident; see `execution-driven-qwc-port` memory note.)

## Path B — trace-driven on H100 (absolute timing)

`make` native (sm_90), trace `layer_all_bench` (or the two parts) on an H100 with the
protobuf tracer (tiny to trace — no CUDA graph / cuBLAS), then simulate the `dynamic_trace.pb`
with `SM90_H100`. This gives the calibrated absolute per-layer time.
