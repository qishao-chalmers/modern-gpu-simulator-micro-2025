# k3 (`mul_mat_q`) dialogue notes — K nesting, reuse, unrolling, compute intensity

Session notes companion to [`k3-mul_mat_q-kernel-explainer.md`](k3-mul_mat_q-kernel-explainer.md).
Primary source: `llama.cpp/ggml/src/ggml-cuda/mmq.cuh` (`mul_mat_q_process_tile`, `vec_dot_q8_0_q8_1_mma`,
`load_tiles_q8_0`); IMMA lowering in `mma.cuh:879`.

Problem size for the traced k3 launch: **M=N=K axes** with **M=4096, N=512, K=4096**; one CTA owns a
**128×128** output tile and reduces full **K=4096**.

---

## 1. Why `k01` is only 4 — is that “K=4096 / 32”?

**Short answer:** `k01 = 4` because **one `vec_dot` only reduces 128 K**, and each IMMA eats **32 K**
(`m16n8k32`). The rest of K lives in outer loops.

```cpp
// Inside vec_dot (one K-half already in shared):
for (j0 ...)           // 8  — output columns
  for (k01 ...)        // 4  — K within this half only
    load_generic(B, ...);
    for (n ...)        // 2  — row minitiles
      mma(C, A[n][k01/8], B);   // A already in regs
```

`k01` loop: `0 … MMQ_TILE_NE_K` step `QI8_0` → `0,8,16,24` → **4 iters**.
Each IMMA: **32** real K → **4 × 32 = 128 K per `vec_dot`**.

### Three levels of K

```
4096 K  =  16 kb0 steps  ×  2 vec_dot  ×  4 k01  ×  32 K per IMMA
        =  32 vec_dot    ×  4 k01      ×  32 K
```

| Level | Loop | Count | K per iteration | Total K |
|-------|------|------:|-----------------|--------:|
| **1** | `kb0` in `mul_mat_q_process_tile` | **16** | 256 | 4096 |
| **2** | `vec_dot` (2 calls per `kb0`) | **2** | 128 | 256 |
| **3** | `k01` inside `vec_dot` | **4** | 32 | 128 |

**Yes — there are `4096/128 = 32` `vec_dot` calls** for one full 128×128 tile:
`16 kb0 × 2 vec_dot = 32`. All accumulate into the same `sum[]` registers until `write_back`.

`k01` is **not** the whole GEMM K-loop — only the innermost slice over the **128 K already staged
in shared** for this `vec_dot`.

---

## 2. Code map (concepts → source)

Call chain:

```
mul_mat_q (mmq.cuh:3565+)
  └─ mul_mat_q_process_tile (3369+)
        ├─ load_tiles → load_tiles_q8_0 (658+)
        ├─ tile_y fill ×2 (3410+, 3426+)
        ├─ vec_dot → vec_dot_q8_0_q8_1_mma (862+)
        └─ write_back (3442+)
```

k3 binds `vec_dot` via `mmq_type_traits<…, GGML_TYPE_Q8_0>::vec_dot_mma` (`3244–3248`).

| Symbol | Role | Where |
|--------|------|-------|
| `kb0` | Global Q8_0 **group** index along K | `3407`; `load_tiles(…, offset_x + kb0, …)` |
| `k00` | Which shared K-half | `0` / `MMQ_TILE_NE_K` at `3421`, `3437` |
| `k01` | Step within half | `k0 = k00 + k01` (`946`, `948`, `967`) |
| `j0` | Output column band | `965` (8 bands) |
| `n` | Row minitile (`ntx=2`) | `943`, `985` |
| `sum[]` | Full 128×128 fp32 accum | `3403`; `+=` at `991` |

Constants: `MMQ_ITER_K=256`, `blocks_per_iter=8`, `MMQ_TILE_NE_K=32`, `QI8_0=8`, `qk=32`.

Full annotated map: explainer **§2.6**.

---

## 3. Loop unrolling — what, and why

Yes — heavy `#pragma unroll` on **small, compile-time-fixed** loops. The **outer `kb0` is not unrolled**.

| Location | Unrolled? | Trip count (k3) |
|----------|-----------|-----------------|
| `load_tiles_q8_0` row loops | Yes | 16 |
| `tile_y` fill | Yes | 18 (`128×36/256`) |
| All `vec_dot` inner loops (`n`, `k01`, `j0`, `l`) | Yes | 2 / 4 / 8 / 4 |
| **`kb0`** | **No** | runtime (`kb0_start`…`kb0_stop`, stream-K) |

**Function of unrolling:**

1. Remove loop overhead on tiny trip counts
2. Expose ILP — schedule `LDS`/`LDSM`, `IMMA`, dequant (`I2FP`/`FFMA`) together
3. Constant-fold addresses (`j0`, `k01`, `n` become compile-time constants)
4. Trade-off: more registers + I-cache; `kb0` stays a real loop to avoid exploding code size

Unrolling does **not** change the math (still 32 `vec_dot`, 4 `k01`, 64 IMMA/warp/`vec_dot`).
It changes SASS shape: one long straight-line compute block per `vec_dot`.

---

## 4. Shared → register reuse — multiple levels

Within one warp’s `vec_dot`, operands are **not** reloaded from shared for every IMMA.

| Level | Scope | A (weights) | B (activations) |
|-------|-------|-------------|-----------------|
| **L1** | fixed `(j0,k01)`, vary `n` | reload per `n` | **×2** reuse |
| **L2** | fixed `(n,k01)`, vary `j0` | **×8** reuse (preloaded) | reload per `j0` |
| **L3** | `k01` step | fresh | fresh |
| **L4** | 2nd `vec_dot` | fresh (other half of `tile_x`) | fresh (new `tile_y`) |
| **L5** | next `kb0` | fresh (`load_tiles`) | fresh |

Per warp per `vec_dot`: **8 A + 32 B = 40** shared→reg loads for **64** IMMA (avg **1.6 IMMA/load**).

Why the asymmetry (GEMM math \(C_{ij}=\sum_k W_{ik}A_{kj}\)):

- **W** depends on row + K, not output column → preload A, reuse over `j0`
- **A** depends on column + K, not W row minitile → load B per `j0`, reuse over `n`

`sum[]` is the only thing that accumulates across **all** levels (K reduction in registers until
`write_back`).

---

## 5. Cross-warp shared-memory reuse vs warp-private register reuse

Two different layers:

| Layer | What is reused | Who shares it |
|-------|----------------|---------------|
| **Shared → registers** (×8 / ×2) | `A[n][…]`, `B` in RF | **One warp only** — each warp has its own register file |
| **Shared memory** (`tile_x` / `tile_y`) | CTA tile contents | **Whole CTA (8 warps)** — loaded once per phase |

Register reuse does **not** cross warps. Warps that need the same SMEM bytes each issue their own
`LDS`/`LDSM`.

### CTA-level SMEM facts

- **One `tile_x` / `tile_y`** per phase; all 8 warps read it.
- `load_tiles`: warps **cooperatively write** different weight rows (`i = i0 + threadIdx.y`).
- `vec_dot`: warps **read** from the full tile; warp pairs `(0,1)`, `(2,3)`, … share the same
  weight-row band (`i0`) → **duplicate SMEM readers**, not shared registers.
- Activations: warps split **N** via `threadIdx.y % ntx` — different `tile_y` columns per pair.
- Extra global reuse: **`tile_x` kept across both `vec_dot`s** (no second weight `load_tiles`);
  only `tile_y` is overwritten for the 2nd K-half.

```
GLOBAL ──(cooperative)──► SHARED (1× CTA) ──(per-warp LDSM/LDS)──► REGISTERS ──► IMMA
                              ▲                              ▲
                     cross-warp reuse              intra-warp ×8 / ×2
```

---

## 6. How shared / register design raises compute intensity

**Question:** how does this design raise ops/byte vs vanilla “each thread streams its own W and A
from global every K step with no sharing”?

### Naive baseline

Per output \(C[i,j]\) over \(K\): load `W[i,k]` and `A[k,j]` from global **every k**.
Intensity ∝ **1** in K → memory-bound; no tile reuse, no register reuse, no tensor-core batching.

### This kernel — intensity at each storage level

**1. CTA tile in shared (cross-warp)**

One `kb0` slice: stage **128×128×256** into `tile_x` / `tile_y`.

| Data | Global load (slice) | Reused for | Factor along tile |
|------|---------------------|------------|-------------------|
| W | once per row into `tile_x` | all **128** output columns | **×128** along N |
| A | once per col into `tile_y` | all **128** output rows | **×128** along M |

Classic blocked-GEMM intensity:

\[
\frac{2\,B_M B_N B_K}{B_M B_K + B_K B_N}
\qquad (B_M=B_N=128,\; B_K=256)
\]

**2. Keep `tile_x` across two `vec_dot`s**

One `load_tiles` for 256 K of W instead of two 128-K weight passes → ~**2×** less weight global
traffic for the same MACs (weights dominate LDG / outlier barrier at `pc=0x10020`).

**3. Shared → register scheduling (intra-warp)**

40 SMEM→reg loads vs 128 if every IMMA fetched fresh A and B → fewer shared loads per MAC in the
hot unrolled `vec_dot` chain.

**4. Tensor cores**

One `m16n8k32` → **4096 int8 MACs** per warp per instruction on already-amortized fragments
(plus dequant FP without another global read).

### Rough full-tile ratio (order of magnitude)

\[
\frac{M\cdot N\cdot K}{M\cdot K + K\cdot N}
\sim \frac{128\cdot 128\cdot 4096}{128\cdot 4096 + 4096\cdot 128} = 64
\]

Naive streaming has denominator \(\sim M\cdot N\cdot K\) — orders of magnitude worse.

### One-line design answer

> **Shared memory** lifts intensity by staging a **128×128×256** tile once so each global byte of
> W/A is shared across **128 outputs** along N or M, and by keeping weights in `tile_x` across both
> K-halves. **Registers** lift it again by reusing weight fragments across **8** column bands and
> activation fragments across **2** row minitiles. **IMMA** then packs **4096 MACs** per instruction
> on that reused data. Vanilla per-thread global streaming does none of this.

---

## 7. Quick cheat sheet (one warp, one 128×128 tile)

| Quantity | Formula | Value |
|----------|---------|------:|
| `vec_dot` calls | `4096 / 128` | **32** |
| `k01` iters per `vec_dot` | `128 / 32` | **4** |
| IMMA per `vec_dot` | `8 × 4 × 2` | **64** |
| IMMA per `kb0` | `64 × 2` | **128** |
| IMMA per tile | `128 × 16` | **2048** |
| Shared→reg loads per `vec_dot` | `8 A + 32 B` | **40** |

---

## Related docs

- [`k3-mul_mat_q-kernel-explainer.md`](k3-mul_mat_q-kernel-explainer.md) — full walkthrough (§2.4 IMMA,
  shared→reg reuse; §2.6 code map; §3 main loop; §4–6 `load_tiles` / barrier)
- [`prefill-k3-real-hw-correlation.md`](prefill-k3-real-hw-correlation.md) — barrier timing gap
  (`BAR.SYNC` at `pc=0x10020`)
