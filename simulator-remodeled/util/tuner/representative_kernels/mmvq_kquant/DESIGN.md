# mmvq_kquant — decode GEMV only (Q8_0 / Q4_K / Q2_K)

## Goal

Simulate **one** llama.cpp decode matvec (`mul_mat_vec_q`) under Accel-Sim / GPGPU-Sim and compare **weight quant** speedup:

```
dst[N] = W[N×K] · y[K]
```

- `W`: Q8_0, Q4_K, or Q2_K (ggml block layouts)
- `y`: q8_1 activations
- **No** layer assembly, attention, RMSNorm, etc.

Primary study shape: **K=4096, N=4096** (Qwen3-8B `q_proj` / `o_proj`).

## Layout / traffic (faithful)

| type | block size | weights/block | bytes/weight (incl. scales) |
|------|------------|---------------|-----------------------------|
| Q8_0 | 34 B (native) / 36 B (FUNCSIM_SAFE f32) | 32 | ~1.06–1.125 B |
| Q4_K | 144 B / ~148 B FUNCSIM | 256 | ~0.56 B |
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
./mmvq_kquant_exec <K> <N> <q8_0|q4_k|q2_k>
./mmvq_kquant_exec 8b q_proj q2_k
./mmvq_kquant_exec 14b all q4_k
```

**Env:**

| var | effect |
|-----|--------|
| `MMVQ_NO_TIME=1` | skip CUDA events (sim) |
| `MMVQ_SKIP_FILL=1` | skip fill kernels — **required for large K×N under GPGPU-Sim** (fill OOM/kills) |
| `MMVQ_TINY=1` | tiny shape only |

**Accel-Sim runner:**  
`accel-sim-framework-official/result/mmvq_kquant_4096/run.sh`  
Uses mem-bound QV100 config from `result/qwc_dp4a_gemv/`.

Metric: `gpu_tot_sim_cycle` for kernel `_Z9mmvq_gemvIL10quant_type…`  
(`0`=Q8_0, `1`=Q4_K, `2`=Q2_K), not fill kernels.

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
