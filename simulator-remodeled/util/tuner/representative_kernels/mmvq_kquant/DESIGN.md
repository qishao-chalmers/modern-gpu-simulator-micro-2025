# mmvq_kquant — decode GEMV / small-batch multi-vector (Q8_0 / Q4_K / Q3_K_M / Q2_K)

## Goal

Simulate llama.cpp decode matvec (`mul_mat_vec_q`) under Accel-Sim / GPGPU-Sim and compare **weight quant** speedup:

```
dst[N×B] = W[N×K] · Y[K×B]
```

- `W`: Q8_0, Q4_K, Q3_K_M, or Q2_K (ggml block layouts)
- `Y`: q8_1 activations (`B=1` is the original decode GEMV; `B>1` is a small batched multi-vector extension)
- **No** layer assembly, attention, RMSNorm, etc.

Primary study shape: **K=4096, N=4096** (Qwen3-8B `q_proj` / `o_proj`).

## Layout / traffic (faithful)

| type | block size | weights/block | bytes/weight (incl. scales) |
|------|------------|---------------|-----------------------------|
| Q8_0 | 34 B (native) / 36 B (FUNCSIM_SAFE f32) | 32 | ~1.06–1.125 B |
| Q4_K | 144 B / ~148 B FUNCSIM | 256 | ~0.56 B |
| Q3_K_M (`block_q3_K`) | 110 B / ~112 B FUNCSIM | 256 | ~0.43–0.44 B |
| Q2_K | 84 B / ~88 B FUNCSIM | 256 | ~0.33 B |

For 4096×4096:

| quant | weight footprint (approx) |
|-------|---------------------------|
| Q8_0  | ~17 MB |
| Q2_K  | ~5.3–5.8 MB |

Expected bandwidth-bound speedup ≈ **weight-byte ratio** (~3× Q8→Q2) if compute is not inflated.

## Implementation notes

**Path:** `representative_kernels/mmvq_kquant/`

| file | role |
|------|------|
| `mmvq_kquant.cu` | host + kernels |
| `../common/shapes_qwen3_gemv.h` | 8B / 14B projection shapes |
| `Makefile` | `make` (native), `make exec` (sm_70 + `FUNCSIM_SAFE`) |
| `sweep_quant.sh` | optional model×op×quant sweep |

**CLI:**

```bash
./mmvq_kquant_exec <K> <N> <q8_0|q4_k|q3_k_m|q2_k> [batch]
./mmvq_kquant_exec 8b q_proj q2_k
./mmvq_kquant_exec 14b all q3_k_m 8
```

**Env:**

| var | effect |
|-----|--------|
| `MMVQ_NO_TIME=1` | skip CUDA events (sim) |
| `MMVQ_SKIP_FILL=1` | skip fill kernels — **required for large K×N under GPGPU-Sim** (fill OOM/kills) |
| `MMVQ_TINY=1` | tiny shape only |
| `BATCH=8` | optional fallback if CLI batch arg is omitted |
| `MMVQ_FORCE_MULTI=1` | keep register-MMVQ for all `B>1` (skip smem MMQ) |

**Kernel dispatch** (mirrors llama.cpp `mmvq.cuh` / `mmq.cuh`, dp4a-only here):

| batch | path | tag |
|-------|------|-----|
| `B=1` | `mmvq_gemv_single` — one CTA/row, stream K | `[gemv]` |
| `2 ≤ B ≤ 32` | `mmvq_gemv_ncols` — same CTA map; W packed in regs, reused across ncols | `[mmvq]` |
| `B > 32` (Q8_0) | `mmq_q8_tiled` — BM=32×BN=64 smem tiles, `WPB+1` bank pad | `[mmq]` |

llama.cpp switches MMVQ→MMQ at **B>8** and uses **tensor-core MMA** in MMQ. Our dp4a MMQ only wins on absolute time once `B>32`, so `MMQ_SWITCH_BATCH=32`.

## H100 native: batch scaling (2026-07-30)

Shape: **K=8192, N=16384, q8_0**, W ≈ 142.6 MB. Device: NVIDIA H100.

| B | path | latency | notes |
|---|------|---------|-------|
| 1 | gemv | **96.8 µs** | ~1.47 TB/s on W — HBM-bound |
| 8 | mmvq | 156 µs | ~1.6× B=1 |
| 16 | mmvq | 267 µs | |
| 32 | mmvq | 543 µs | roughly linear in B from 16→32 |
| 64 | mmq | **1149 µs** | smem tile; ~flat vs forcing mmq at B=16–64 (~1.0–1.15 ms) |
| 128 | mmq | 2271 µs | 2× BN=64 batch tiles → ~2× B=64 |

Forced comparisons (same shape):

| B | `[mmvq]` (FORCE_MULTI) | `[mmq]` |
|---|------------------------|---------|
| 16 | **267 µs** | 1028 µs |
| 64 | 1184 µs | **1149 µs** |

So MMQ flattens the **large-B** curve but is slower than register-MMVQ for mid batch — hence the B>32 switch.

### Why B=16…64 still grew on the GEMV/MMVQ path

1. **Not missing W reuse.** One-pass `ncols=64` (single W DRAM pass) still tracks ~linear time → past ~B=8 the kernel is **dp4a compute-bound** (and y traffic), not W re-fetch.
2. **Shared-memory GEMM alone does not restore B=1 latency.** Early BM×BN tile kernels made B=1 ~6× slower (~572 µs) by changing the CTA map; we kept GEMV for B=1.
3. **llama.cpp’s flat large-batch curve needs MMA (or cuBLAS).** Their MMQ uses tensor-core int8 MMA after `MMVQ_MAX_BATCH_SIZE=8`; on H100 they may hand `ne11≥64` to cuBLAS. Our MMQ is a **dp4a traffic-faithful** stand-in — flat across B when used, but ~1 ms floor on this shape, not ~100 µs.

Bank-conflict swizzling / vector loads: reduction smem is tiny on MMVQ; MMQ uses `WPB+1` pad (llama-style). Neither closes the B=1→B=16 gap on the GEMV map.

### Gap vs goal

| goal | status |
|------|--------|
| Keep B=1 ≈ historical GEMV (~97 µs on this shape) | **done** |
| B=2…8 ≈ B=1 via W reuse | **mostly** (~1.6× at B=8) |
| B=16…64 ≈ B=1 (classic “tile A/B, reuse W”) | **not with dp4a GEMV**; MMQ flattens B but at ~1.1 ms, not ~0.1 ms |
| Match llama large-B absolute speed | **gap** — need MMA MMQ (or cuBLAS reference) |

### Plan

1. **Native H100:** add an optional **tensor-core MMA** MMQ path (or thin cuBLAS/cublasLt reference) for `B>8`, matching llama’s dispatch so B=16…64 stay near the MMA floor instead of the dp4a ~1 ms floor.
2. **Keep dp4a MMQ** as the Accel-Sim / `FUNCSIM_SAFE` representative (no MMA): traffic + bank-pad layout; document that absolute µs will not match MMA.
3. **K-quants (`q4_k` / `q3_k_m` / `q2_k`):** still MMVQ-style for all B; add smem MMQ variants only if batch study needs them.
4. **Update this table** after MMA (or cuBLAS) numbers land on the same K/N/q8_0 shape.
5. **Sim:** continue `MMVQ_SKIP_FILL=1` + protobuf/execution-driven runs; compare `gpu_tot_sim_cycle` across quants at fixed B.

## Caveats (why 4096×4096 failed / was cancelled)

1. **Fill kernels** at 4096×4096 allocate/sim ~524k Q8_0 blocks → process **Killed** (OOM). Fix: `MMVQ_SKIP_FILL=1` (rebuilt into exec).
2. **`FUNCSIM_SAFE` / naive K-quant loops** can look compute-bound if work is
   striped over `K/256` superblocks (only ~16 of 128 threads busy at K=4096) or
   if each weight is scalar-unpacked. The kernel parallelizes over `K/32` subs and
   uses byte-wise int MACs (`__dp4a` on native Q2). Note: activation `q8_1` traffic
   is identical across quants, so end-to-end speedup is **less than** the weight-byte
   ratio (~1.4–1.6× Q8→Q2 at 4096×4096, not ~3×).
3. Full 4096 GEMV under execution-driven mode is **slow** (tens of minutes per quant). Smoke-test at 512×512 first.

## Status when cancelled

- Smoke 512×512: both quants ran under GPGPU-Sim.
- First 4096 Q8_0 (with fill): **Killed**.
- Second attempt with `MMVQ_SKIP_FILL`: started; **user cancelled** before clean Q8_0 + Q2_K cycle pair.

## Resume checklist

```bash
# rebuild
make -C .../mmvq_kquant exec

# optional smaller shape first
K=1024 N=1024 ./run.sh   # under result/mmvq_kquant_4096 if you extend run.sh

# full pair
cd .../result/mmvq_kquant_4096
./run.sh                 # already sets MMVQ_SKIP_FILL=1
./run.sh summarize
```

Compare `results.csv` cycles; speedup = `cycles_q8_0 / cycles_q2_k`

---

## Sibling: `mmvq_kquant_4bit.cu` (y = q4_1, int4×int4)

Same weight quants, but activations are **q4_1** (packed nibbles) and the hot op is
**`dp8a`** = 8× int4 MACs per 32-bit pair (Blackwell-class 4-bit density for Accel-Sim).

| build | binary | notes |
|-------|--------|-------|
| `make 4bit` | `mmvq_kquant_4bit` | software int4 (runs on H100) |
| `make 4bit_exec` | `mmvq_kquant_4bit_exec` | FUNCSIM_SAFE, software int4 |
| `make 4bit_exec_dp8a` | `mmvq_kquant_4bit_exec_dp8a` | PTX `dp8a.s32.s32` for GPGPU-Sim |

Accel-Sim (official tree) gains PTX opcode `dp8a` + SASS alias `IDP8A` (same INT pipe as `IDP4A` by default; retune latency/throughput in config for “stronger int4 HW”).

### Accel-Sim: `ptxas` ERROR 65280 with `_exec_dp8a`

NVIDIA `ptxas` does **not** know `dp8a`. GPGPU-Sim still calls it for register/occupancy
info → `ERROR ** while loading PTX (b) 65280` (misleading “Ensure ptxas is in your path”).

**Fix without waiting for a sim rebuild** — use the shim (already under `cuda_dp8a_shim/`):

```bash
REPK=.../mmvq_kquant
FAKE=$REPK/cuda_dp8a_shim
export REAL_PTXAS=$(dirname $(dirname $(which ptxas)))/bin/ptxas
export CUDA_INSTALL_PATH=$FAKE
export PATH=$FAKE/bin:$PATH
# then run as usual with LD_LIBRARY_PATH=.../gpgpu-sim/lib/.../release
```

Or use `result/mmvq_kquant_4bit_quant/run.sh`.

Longer-term: rebuild Accel-Sim after the `ptx_loader.cc` patch (rewrites dp8a→dp4a for
ptxas only). Functional sim still executes real `dp8a`.

**Alternative:** `make 4bit_exec` (software int4, no `dp8a`) runs under stock Accel-Sim.
