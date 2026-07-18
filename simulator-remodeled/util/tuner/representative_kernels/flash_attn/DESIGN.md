# Flash-attention decode benchmark — design document

**File:** `representative_kernels/flash_attn/flash_attn_bench.cu`  
**llama.cpp reference:** `ggml/src/ggml-cuda/fattn-vec.cuh` (`flash_attn_ext_vec`), `fattn-common.cuh` (`launch_fattn`)  
**Model target:** Qwen3-14B decode (`hidden=5120`, `n_q_heads=40`, `n_kv_heads=8`, `head_dim=128`)

---

## 1. Why this exists (parent project context)

The **representative_kernels** suite (`util/tuner/representative_kernels/`) extracts the CUDA kernels that dominate llama.cpp **decode** into tiny standalone microbenchmarks. Goals:

1. Run under **GPGPU-Sim execution-driven** mode (official Accel-Sim + QV100 config in `accel-sim-framework-official/local_test/`) without tracing full llama.cpp (no CUDA graphs, no 14B model load).
2. Sweep shapes / bit-widths / batch size and read `gpu_sim_cycle` from simulator logs.
3. Eventually compare against **trace-driven H100** sim (`modern-gpu-simulator-micro-2025`, protobuf traces).

Sibling benches: `mmvq_bitwidth/`, `rms_norm/`, `quantize_q8_1/`, `mul_mat_vec_q/`, etc. See `../DESIGN.md`.

---

## 2. What this bench models

In Qwen3-14B decode (one token, context ~1024+), the attention kernel is:

| Trace field | Value |
|-------------|-------|
| Kernel name | `flash_attn_ext_vec` (F16 K/V cache) |
| Head dim D | 128 |
| Q heads | 40 |
| KV heads | 8 (GQA, ratio 5:1) |
| Typical seq | 1024 (prefill) … 1088 (+ decode tokens) |
| Real H100 time | ~4.35 µs (14B, post-tuning) |

Real ggml also uses **split-K** over the KV sequence and a **combine** kernel (`flash_attn_combine_results`, grid 40). This bench implements that pattern in simplified form.

---

## 3. High-level architecture (two-kernel split-K)

```
                    ┌─────────────────────────────────────────┐
  Q [batch][40][128]│  flash_attn_ext_vec_decode              │
  K [batch][8][seq][128]  grid = (1, PK, batch*40)            │
  V [batch][8][seq][128]  block = 128 threads                 │
                    │  Each block: one (head, KV-split)       │
                    │  → partial softmax state                │
                    └──────────────┬──────────────────────────┘
                                   │
                    p_m[qh*PK + split]     max logit (rescaled)
                    p_l[qh*PK + split]     sum of exp (rescaled)
                    p_acc[(qh*PK+split)*D+e]  weighted V sum (rescaled)
                                   │
                    ┌──────────────▼──────────────────────────┐
                    │  flash_attn_combine                     │
                    │  grid = (1, 1, batch*40)                │
                    │  Merges PK partials → final dst         │
                    └──────────────┬──────────────────────────┘
                                   │
                    dst [batch][40][128]  (f32 output)
```

**Rationale:** At batch=1, a naive single kernel launches only **40 blocks** (one per Q head) on **80 SMs** → half idle. Each block also runs a long sequential online-softmax loop over all `seq_len` KV positions. Split-K launches **40×PK blocks**, shortens each block's loop by ~PK×, and uses a cheap combine pass — matching what real llama.cpp does via `parallel_blocks` in `launch_fattn`.

**Important:** Total KV **read volume is unchanged** (each KV element still read exactly once across splits). This is a **latency / occupancy** optimization, not a memory-traffic reduction.

---

## 4. Kernel 1: `flash_attn_ext_vec_decode<D>`

### 4.1 Launch geometry

| Parameter | Value (Qwen3-14B, batch=1) |
|-----------|----------------------------|
| `grid` | `(1, PK, batch * n_q_heads)` → e.g. `(1, 8, 40)` |
| `block` | `(D, 1, 1)` → `(128, 1, 1)` |
| `blockIdx.z` | `qh` = query-head index `0 .. batch*n_q_heads-1` |
| `blockIdx.y` | `split` = KV slice `0 .. PK-1` |

Decode from `qh`:
```c
seq_b   = qh / n_q_heads;
head    = qh % n_q_heads;
kv_head = head * n_kv_heads / n_q_heads;   // GQA mapping
```

### 4.2 KV slice assignment

```c
per_split = ceil(seq_len / PK);
t_begin   = split * per_split;
t_end     = min(t_begin + per_split, seq_len);
```

Empty splits (`t_begin >= t_end`) write neutral partials: `p_m=-∞`, `p_l=0`, `p_acc=0`.

### 4.3 Intra-block parallelism (warp-over-KV)

This matches the parallelism model we adopted from ggml's vec attention:

- **D = 128 threads** = **4 warps** (NW = D/32).
- Warps **stride over KV positions** in their slice: warp `w` owns `t = t_begin+w, t_begin+w+NW, …`.
- For each position `t`:
  1. Partial dot `Q·K[t]` via lane-local multiply + `warp_reduce_sum` (`__shfl_xor`).
  2. Online softmax update per warp: running `(m, l, acc[NW])`.
- After the KV loop: **one `__syncthreads`**, then warp 0 merges the NW warp partials (same math as cross-split combine, but inside the block).

Thread ↔ head-dim mapping: lane `lane` in warp `w` owns output dims `e = lane + k*32` for `k=0..NW-1` (each thread holds NW accumulator elements).

### 4.4 Partial output (not final dst)

After intra-block merge, warp 0 writes to scratch (indexed `pidx = qh*PK + split`):

| Buffer | Meaning |
|--------|---------|
| `p_m[pidx]` | `gm` — max logit in this split (after warp merge) |
| `p_l[pidx]` | `den` — rescaled sum of exp weights at `gm` |
| `p_acc[pidx*D + e]` | `num[e]` — rescaled weighted sum of V at `gm` (not yet divided) |

These are the standard online-softmax state variables, already rescaled to a common `gm` within the split, so they compose cleanly in the combine kernel.

### 4.5 Memory layout

```
Q:  [batch][n_q_heads][D]           uint16_t FP16
K:  [batch][n_kv_heads][seq][D]     uint16_t FP16
V:  [batch][n_kv_heads][seq][D]     uint16_t FP16
dst:[batch][n_q_heads][D]           float (written by combine)
```

KV bytes (FP16): `2 * batch * n_kv * seq * D * 2`  
Example: batch=1, n_kv=8, seq=1024, D=128 → **4.194 MB** (matches log line).

---

## 5. Kernel 2: `flash_attn_combine<D>`

### 5.1 Launch geometry

| Parameter | Value |
|-----------|-------|
| `grid` | `(1, 1, batch * n_q_heads)` |
| `block` | `(D, 1, 1)` |
| `blockIdx.z` | `qh` |

### 5.2 Math (cross-split online-softmax merge)

For each head `qh` and output dim `e`:

```c
gm  = max_{s in 0..PK-1} p_m[qh*PK + s]
den = sum_s  p_l[qh*PK + s] * exp(p_m[qh*PK + s] - gm)
num = sum_s  p_acc[(qh*PK + s)*D + e] * exp(p_m[qh*PK + s] - gm)
dst[qh*D + e] = num / max(den, 1e-20)
```

Empty splits contribute zero via `exp(-∞ - gm) = 0`.

**PK=1 regression:** one partial → combine is algebraically identical to the old single-kernel path that wrote `dst = num/den` directly.

---

## 6. Q8_0 KV path (`KVBITS=8`)

Optional second precision path for studying **compressed KV cache**:

- `flash_attn_ext_vec_q8_decode<D>` — same split-K + partial layout, but Q/K/V stored as `blk_q8` (32 int8 + f16 scale, 34 B/block).
- Reuses `flash_attn_combine` (partials are always float).
- KV traffic ≈ **half** of FP16 path (1.06 B/elem vs 2 B/elem).

---

## 7. What is faithful vs simplified (vs real llama.cpp)

| Aspect | This bench | Real `fattn-vec.cuh` |
|--------|------------|----------------------|
| Split-K over KV | Yes (`PK` on `grid.y`) | Yes (`parallel_blocks` on `grid.y`) |
| Combine kernel | Yes (`flash_attn_combine`) | Yes (`flash_attn_combine_results` / fixup) |
| Split over Q columns (`ncols=2`) | **No** (decode has 1 Q column) | Yes for prefill / wide Q |
| GQA head mapping | Yes | Yes |
| Warp-parallel KV streaming | Simplified but same idea | Full template (`D`, `ncols`, type_K/V) |
| Mask / ALiBi / softcap | No | Yes |
| Tensor cores / MMA | No | Optional paths |
| `FUNCSIM_SAFE` f16 decode | Bitfield `load_f16()` | Native half / tensor loads |

We intentionally keep the kernel **self-contained** (~500 lines) rather than `#include` all of `fattn-vec.cuh` + 69 template instances — sufficient for **relative** exec-driven studies (PK sweep, batch sweep, q8 vs f16 KV).

---

## 8. Build and run

### 8.1 Build

```bash
cd simulator-remodeled/util/tuner/representative_kernels/flash_attn

make              # native GPU (sm_90 default; override ARCH=sm_80)
make exec         # GPGPU-Sim: sm_70 PTX + --cudart shared

# Or from parent:
../compile_all.sh --exec
```

### 8.2 CLI arguments

```
./flash_attn_bench_exec  <seq_len>  [batch]  [n_q_heads]  [n_kv_heads]  [PK]
```

| Arg / env | Default | Meaning |
|-----------|---------|---------|
| `seq_len` | (required) | KV cache length |
| `batch` / `BATCH` | 1 | Parallel decode sequences (each has own KV) |
| `n_q_heads` | 40 | Qwen3-14B |
| `n_kv_heads` | 8 | Qwen3-14B GQA |
| `PK` / `PARALLEL_K` | 8 | KV sequence splits |

**Common mistake:** `./flash_attn_bench_exec 1024 1 40 8` — the trailing `8` is **n_kv_heads**, not PK. For PK=16 use **six** tokens after the binary:

```bash
./flash_attn_bench_exec 1024 1 40 8 16
# → PARALLEL_K=16, grid_attn=(1,16,40)
```

### 8.3 GPGPU-Sim sweep

```bash
# Copy gpgpusim.config from SM7_QV100; then:
./sweep_shapes.sh          # seq_len sweep at default PK
./sweep_pk.sh              # PK ∈ {1,2,4,8,16} at seq=1024
```

Read `gpu_sim_cycle` from logs — **sum attn kernel + combine kernel** for total attention cost.

### 8.4 Verification

| Mode | Command | Expected |
|------|---------|----------|
| Uniform | default fill Q≈0.1, K=V=1.0 | `dst[0] ≈ 1.0` |
| PK regression | `PARALLEL_K=1` | Same result as pre-split-K single kernel |
| Ramp | `VERIFY_RAMP=1` | Exercises cross-split rescaling (no fixed expected) |

---

## 9. Expected simulator behavior (PK sweep)

At `seq=1024`, `batch=1`, Qwen3-14B:

| PK | grid_attn | Blocks launched | KV positions / split |
|----|-----------|-----------------|----------------------|
| 1 | (1,1,40) | 40 | 1024 |
| 8 | (1,8,40) | 320 | 128 |
| 16 | (1,16,40) | 640 | 64 |

**Hypothesis:** `gpu_sim_cycle` for (decode + combine) should **drop** as PK increases from 1, then flatten once `40*PK` exceeds SM count (80 on V100 config used for exec-driven). Baseline before split-K was ~781k cycles (single long kernel); expect improvement at PK=2–8.

Combine kernel should stay a small fraction of total (40 blocks, PK-wide reduction only).

---

## 10. File map

```
flash_attn/
  DESIGN.md              ← this document
  flash_attn_bench.cu    ← kernels + host launcher
  Makefile               ← make / make exec
  sweep_shapes.sh        ← seq_len sweep under GPGPU-Sim
  sweep_pk.sh            ← PK sweep for cycle comparison

common/
  shapes_qwen3_14b_decode.h   ← fattn_shape table
  bench_common.h                ← CHECK macro, fill helpers
```

---

## 11. Future work

- [ ] Mirror exact ggml launch for Q-column split (`ncols=2`, grid.x) when studying prefill.
- [ ] Trace this harness on H100 → compare exec-driven QV100 vs trace-driven SM90.
- [ ] Add `rope_neox`, `k_set_rows` to complete the decode layer chain (`layer_decode/`).
- [ ] Optional `NO_DEQUANT=1` build path for MC offload studies (skip f16 decode on SM).

---

## 12. Quick reference — correct log line

Command:
```bash
./flash_attn_bench_exec 1024 1 40 8 8
```

Expected:
```
KV/Q precision: 16-bit (FP16 cache)
PARALLEL_K=8 (splits KV sequence across grid.y)
>>> flash_attn_ext_vec ... D=128 heads=40/8 seq=1024 batch=1 PK=8
    grid_attn=(1,8,40) grid_combine=(1,1,40) block=128  KV=4.194 MB
    [verify] dst[0]=1.0000 expected=1.0000  OK
```

If you see `PK=8` but intended `PK=16`, you did not pass the 6th argument — check §8.2.
