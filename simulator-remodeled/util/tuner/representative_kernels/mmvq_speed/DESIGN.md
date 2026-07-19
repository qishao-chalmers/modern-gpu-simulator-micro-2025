# mmvq_speed — SPEED Q∥R strip layout (prototype)

Sibling of [`../mmvq_kquant/`](../mmvq_kquant/). **Does not modify** `mmvq_kquant.cu`.

## Goal

Prototype the SPEED memory layout: **contiguous quantized groups (Q)**, then **contiguous residual groups (R)**, so a DRAM row open can serve a long sequential burst. Evaluate two **CUDA** modes against packed `mmvq_kquant`, and leave mode 3 for **simulator-only** FR rebuild.

```
dst[N] = W[N×K] · y[K]     y = q8_1 (same as mmvq_kquant)
```

## Memory layout

One **group** = **256** weights (`QK_K`), stored as one ggml K-quant block (`Q2_K` / `Q3_K` / `Q4_K`).

Pack **G ∈ {1, 2, 4, 8}** groups per strip:

```text
row:  [ Q×G | R×G ] [ Q×G | R×G ] …     # n_strips = K / (G·256)
```

Example `G=4`, `K=4096`: each strip is **1024** Q weights then **1024** R weights; 4 strips per row.

| Why G > 1 | Longer sequential Q (then R) run → better row-buffer / burst utilization |
|-----------|--------------------------------------------------------------------------|

`K` must be divisible by `G * 256`. Q and R types are chosen independently (`q2_k` / `q3_k` / `q4_k`).

Address of Q-block `sb` (superblock index `0 .. K/256 - 1`) on a row:

```text
strip = sb / G
local = sb % G
q_ptr = row_base + strip * (G·|Q| + G·|R|) + local · |Q|
r_ptr = row_base + strip * (G·|Q| + G·|R|) + G·|Q|      + local · |R|
```

## Modes

| Mode | Name | Who rebuilds? | CUDA GEMV | Compare to |
|------|------|---------------|-----------|------------|
| **1** | Q-only | — | K-quant GEMV on **Q regions only** (skip R gaps) | `mmvq_kquant` same quant — **layout tax** |
| **2** | SW SPEED | CUDA | `acc += dot(Q,y) + dot(R,y)` ≡ `Ŵ = dequant(Q)+dequant(R)` | mode 1 / `mmvq_kquant` — **2× weight traffic + rebuild math** |
| **3** | HW SPEED | MC/FR in **sim** | CUDA binary = **plain Q8_0** matvec (unchanged) | sim-only; not a different CUDA kernel |

### Mode 1 detail

Same compute as `mmvq_kquant` for the Q type, but after each QG the next Q pack is **not** contiguous (R sits in between). Runnable on real CUDA → measure gapped vs packed bandwidth.

### Mode 2 detail

Software reconstruction in **dequantized space** (linear FR):

```text
⟨Ŵ, y⟩ = ⟨dequant(Q), y⟩ + ⟨dequant(R), y⟩
```

implemented as two K-quant dots per 32-wide sub (reuses the same `vec_dot_*_k_sub` as `mmvq_kquant`). Numerics match `Ŵ = Q + R` after dequant; not the INT8 saturate path yet (that is mode 3 / FR RTL).

### Mode 3 detail (future, sim-only)

- DRAM stores Q∥R strips (e.g. Q2 + R4).
- FR rebuilds online → INT8; SM runs **ordinary Q8 GEMV**.
- CUDA code ≡ `mmvq_kquant` / Q8 path; simulator adds dual-region fetch + reconstruction latency.
- **Not implemented in this CUDA binary** (see stub / docs only).

This differs from earlier QWC experiments where the kernel often did int-dot then × scales on the SM, or only shrank DRAM bursts while keeping an 8-bit kernel without an explicit Q∥R strip layout.

## Compare vs `mmvq_kquant`

Same shapes (`K`, `N`), same Q type, same thread geometry (4 warps × 32).

```bash
# packed baseline (Q4_K)
../mmvq_kquant/mmvq_kquant 4096 4096 q4_k

# SPEED mode 1: Q4_K only, G=4 strips (R type unused for compute, still allocated)
./mmvq_speed 1 4 q4_k q2_k 4096 4096

# SPEED mode 2: Q4_K + R Q2_K rebuild
./mmvq_speed 2 4 q4_k q2_k 4096 4096
```

| Expect | Why |
|--------|-----|
| Mode 1 ≈ `mmvq_kquant` or slightly slower | Same Q bytes touched; **strided** Q addresses (skip R) |
| Mode 2 ≫ mode 1 bytes | Reads **Q and R**; ~sum of both footprints + extra MACs |
| Verify mode 1 | unit fill → `dst[0] ≈ K` |
| Verify mode 2 | unit fill both → `dst[0] ≈ 2K` |

## Build / run

```bash
make            # sm_90  -> mmvq_speed
# older nvcc:  make ARCH='-gencode=arch=compute_80,code=sm_80'
make exec       # FUNCSIM_SAFE -> mmvq_speed_exec

./mmvq_speed <mode> <G> <q> <r> <K> <N>
./mmvq_speed <mode> <G> <q> <r> 8b|14b <op|all>

# side-by-side vs packed mmvq_kquant (same K,N,q)
../mmvq_kquant/mmvq_kquant 4096 4096 q4_k
./mmvq_speed 1 4 q4_k q2_k 4096 4096    # mode 1 layout tax
./mmvq_speed 2 4 q4_k q2_k 4096 4096    # mode 2 Q+R rebuild

# env (same spirit as mmvq_kquant)
MMVQ_NO_TIME=1 MMVQ_SKIP_FILL=1 MMVQ_TINY=1
```

## Status

- [x] Layout + modes 1 / 2 CUDA prototype
- [x] DESIGN + CLI comparable to `mmvq_kquant`
- [ ] Mode 3 sim hooks (dual region + FR latency + Q8 GEMV binary)
- [ ] Optional: INT8 saturate rebuild in mode 2 to match FR RTL bit-exact
