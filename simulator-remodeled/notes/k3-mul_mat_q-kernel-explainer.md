# k3 (`mul_mat_q<Q8_0, Li128>`) kernel walkthrough: source, SASS, and the barrier-divergence mechanism

Companion to [`prefill-k3-real-hw-correlation.md`](prefill-k3-real-hw-correlation.md) §14. That doc found *where*
the cycle-count gap concentrates (a specific `BAR.SYNC` at `pc=0x10020`, with 10-20x more inter-warp arrival
spread than the loop's other three barriers). This doc explains *why*, by walking the actual llama.cpp source
(`llama.cpp/ggml/src/ggml-cuda/mmq.cuh`) side-by-side with the compiled SASS.

## 1. The kernel in one sentence

`mul_mat_q<GGML_TYPE_Q8_0, mmq_x=128>` is a tiled, quantized GEMM: each CTA computes a `mmq_y × mmq_x` tile of the
output by repeatedly (a) loading a tile of the quantized weight matrix `x` and activation matrix `y` into shared
memory, (b) computing partial dot products from shared memory, and (c) accumulating into registers — looping over
the reduction (`k`) dimension. 256 threads/CTA = 8 warps/CTA (`nwarps = 256/32 = 8`), one CTA per SM for this
kernel (register-pressure-limited occupancy — see the main doc §6).

## 2. What the kernel does — problem size, CTA tile, and per-warp slice

### 2.1 Function

`mul_mat_q<Q8_0, Li128>` is the llama.cpp **fused quantized GEMM** used after `quantize_mmq_q8_1`:

\[
C[M,N] \;=\; \mathrm{dequant}(W_q[M,K]) \times A_q[K,N]
\]

Weights `W_q` are **Q8_0** in global memory; activations `A_q` are **Q8_1** in MMQ tile layout. The kernel
stages both into shared memory, runs **IMMA** int8 tensor-core dots, and applies per-block scales (`I2FP` /
`FMUL` / `FFMA`) into **fp32 accumulators** — it is not a pure int8 GEMM.

For the traced k3 launch (Qwen3-8B prefill ubatch, see [`prefill-k3-correlation.md`](prefill-k3-correlation.md) §4):

| Symbol | Meaning | Value |
|--------|---------|------:|
| **M** | weight rows | 4096 |
| **N** | output columns (tokens) | 512 |
| **K** | inner / hidden dimension | 4096 |

**Launch:** `grid = 132×1×1` (one CTA per H100 SM, stream-K), `block = 32×8×1` (256 threads = 8 warps).

**Work partition (stream-K):** the full GEMM is tiled into **128 output blocks** of 128×128 each
(`nty = ⌈4096/128⌉ = 32` row tiles `it`, `ntx = ⌈512/128⌉ = 4` column tiles `jt`). Each output block
also needs a full **K reduction** (`K/qk = 128` quantized blocks, `qk = 32` for Q8_0). The launch uses
**132 CTAs** (one per SM) — **not** 128 CTAs (one per output tile).

Stream-K linearizes all `(output tile, K progress)` work into one queue and **splits it across the 132 blocks**.
Over the kernel lifetime a single CTA may finish all of K on tile `(it=2, jt=1)`, then move to `(it=2, jt=2)`, or
do only part of K on one tile before another CTA takes over (partial sums go to `mul_mat_q_stream_k_fixup`).

Two levels of loop in source (`mmq.cuh`):

| Level | Function | What it does |
|-------|----------|--------------|
| **Outer** | `mul_mat_q` (`kbc` / `kbc_stop`, ~3565+) | Stream-K scheduler: decodes each CTA's work slice into tile indices `(it, jt)` and a K range `[kb0_start, kb0_stop)`, then calls the inner function |
| **Inner** | `mul_mat_q_process_tile` (§3, ~3407+) | **One output-tile's K-loop** — for a fixed `(it, jt)`, walks K in `kb0` steps (load → sync → `vec_dot` × 2) |

Tile indices from the outer loop (`mmq.cuh:3577-3583`):

- **`it`** — which **128-row** strip of `W` / `C` (M dimension)
- **`jt`** — which **128-column** strip of `C` (N dimension)

§2.2 below fixes `(it, jt)` and zooms in on one call to `mul_mat_q_process_tile`, as if the outer scheduler has
already said "you are on this tile now."

**Example** — tile `(it=0, jt=0)`, the top-left 128×128 corner of `C`:

```
C[0:128, 0:128]  +=  W[0:128, 0:4096] × A[0:4096, 0:128]
                      └── mul_mat_q_process_tile: kb0 = 0, 8, 16, …, 120 ──┘
```

That inner `for (kb0 …)` loop (§3) is what this doc walks through for the barrier analysis; stream-K only decides
**which CTA** runs which slice of those `kb0` steps and **which `(it, jt)`** is active at each moment.

### 2.2 One CTA working on one output tile

Fix tile indices `(it, jt)`. The CTA owns the **`mmq_y × mmq_x = 128 × 128`** fp32 block of `C` at rows
`[it·128, (it+1)·128)` and columns `[jt·128, (jt+1)·128)`.

| | Global inputs (per `kb0` step) | Shared staging | Output |
|--|-------------------------------|----------------|--------|
| **Weights `x`** | `W_q` rows `it·128…`, K-slice `kb0·32…` spanning **ITER_K = 256** elements (`blocks_per_iter = 8` Q8_0 blocks) | `tile_x` — **both** K-halves (offsets `0` and `+MMQ_TILE_NE_K`) written in one `load_tiles` call | — |
| **Activations `y`** | `A_q` cols `jt·128…`, same K-slice, loaded in **two serial halves** into the same `tile_y` | `tile_y` — one K-half at a time (`MMQ_TILE_Y_K` ints per column) | — |
| **Accumulators** | — | — | `sum[]` in registers → full **128×128** fp32 tile when K reduction completes → `write_back` to global `dst` |

#### K-loop arithmetic (per output tile)

Yes — **each outer `kb0` step covers 256 elements along K** (`MMQ_ITER_K = 256`). For the full
`K = 4096` reduction on one 128×128 output tile:

```
4096 / 256 = 16 outer-loop passes
```

`kb0` is indexed in **Q8_0 blocks** — ggml's group-wise quantization units along K. Each
`block_q8_0` (`ggml-common.h`) is one group: **32 int8 values** (`QK8_0 = 32`) sharing **one fp16
scale** `d`. In source this is `qk = 32`; `kb0` is the **group index** along K, not a raw element index.

There are `4096/32 = 128` groups per output tile. The outer loop steps `kb0` by
`blocks_per_iter = ITER_K/qk = 8` groups per pass:

```
kb0 = 0, 8, 16, …, 120
  →  16 steps × 8 groups/step × 32 elements/group = 4096
```

Within **one** `kb0` step (one 256-wide K slice), the kernel still runs **`vec_dot` twice**. That is
not because each `vec_dot` only covers 256/16 — it is because **`tile_y` holds only one activation
K-half at a time** in shared memory, while `tile_x` already has both weight halves from `load_tiles`:

| Step | What happens | K covered |
|------|--------------|-----------|
| `load_tiles` | both weight K-halves → `tile_x` | **256** (full slice) |
| 1st `tile_y` fill + `vec_dot(..., k00=0)` | activations for **1st half** → `tile_y`, then IMMA | **128** |
| 2nd `tile_y` fill + `vec_dot(..., k00=MMQ_TILE_NE_K)` | **overwrite** `tile_y` with 2nd half, then IMMA | **128** |

`k00` is the **starting offset along the shared tile's K axis** (in packed int32 tile indices, not
global K). `MMQ_TILE_NE_K = 32` marks the boundary between the two halves in that layout; with
`QI8_0 = 8`, each half is `32/8 = 4` Q8_0 groups = **4 × 32 = 128** real K elements. Both
`vec_dot` calls accumulate into the same `sum[]` registers. **32 `vec_dot` calls** complete one tile's
full K (`4096/128`); the inner `k01` loop (4 iters) only covers **one** of those 128-K chunks — see §2.6.

```
One output tile, full K = 4096
│
├─ kb0=0    [ K 0:256 )     vec_dot k00=0  → [0:128)   +  vec_dot k00=32 → [128:256)
├─ kb0=8    [ K 256:512 )
├─ …
└─ kb0=120  [ K 3840:4096 )   ← 16th and final slice
```

#### What each `vec_dot` computes

Each `vec_dot` does **not** produce a smaller spatial output tile. Both calls update the same
**128 × 128** `sum[]` registers — only the **K depth** per call differs:

| Call | `k00` | W used from `tile_x` | A used from `tile_y` | K reduced | M×N updated |
|------|-------|----------------------|----------------------|-----------|-------------|
| 1st | `0` | **1st half** only | **1st half** only | 128 | **128 × 128** (partial sum) |
| 2nd | `MMQ_TILE_NE_K` (= 32) | **2nd half** only | **2nd half** only | 128 | **128 × 128** (partial sum) |

Mathematically, for one `kb0` step (one 256-wide K chunk at global offset `kb0·32`):

\[
C \mathrel{+}= W_{:,k:k+128}\,A_{k:k+128,:} \;+\; W_{:,k+128:k+256}\,A_{k+128:k+256,:}
\]

`k00` selects which half of the **shared tile's K axis** `vec_dot` indexes (`mmq.cuh:945-948`):
`k0 = k00 + k01` with `k01` running `0 … MMQ_TILE_NE_K` in steps of `QI8_0`.

**Important:** `load_tiles` puts **both** W halves into `tile_x`, but each `vec_dot` reads **only one**
of them — matched to the corresponding A half in `tile_y`:

```
tile_x after load_tiles:  [ W half-1 | W half-2 ]   ← both in shared, one global load
                              ↓              ↓
vec_dot #1 (k00=0):      [ W half-1 ] × [ A half-1 ]  → sum
vec_dot #2 (k00=32):               [ W half-2 ] × [ A half-2 ]  → sum
                              ↑
                    no second load_tiles — re-read tile_x from shared
```

#### Where the partial results live (registers, not shared/global)

The first `vec_dot` result is **not** written to shared memory or global memory. It stays in
**per-thread register accumulators** until the full K reduction finishes.

```cpp
float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};   // 64 fp32 values per thread

vec_dot(tile_x, tile_y, sum, 0);              // sum[...] +=  (1st K-half contribution)
vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);  // sum[...] +=  (2nd K-half contribution)
// ... repeats for every kb0 step (16 times when K completes) ...

write_back(sum, ..., dst, ...);               // only here: registers → global fp32 C
```

Inside `vec_dot`, each IMMA result is fused with dequant scales and accumulated with **`+=`** into
`sum[]` (`mmq.cuh:991`). So:

| After | Where partial C lives | Visible to 2nd `vec_dot`? |
|-------|----------------------|---------------------------|
| 1st `vec_dot` | **`sum[]` registers** (private per thread) | Yes — same array, in-place `+=` |
| 2nd `vec_dot` | same `sum[]`, now includes both K-halves of this `kb0` step | — |
| Next `kb0` step | same `sum[]`, keeps accumulating across all 16 steps | — |
| `write_back` | **global `dst`** (fp32 output tile) | once per `mul_mat_q_process_tile` call |

Shared memory (`tile_x`, `tile_y`) is only for **input staging**. The running GEMM output is entirely
in **`sum[]` registers** until `write_back` stores the finished 128×128 tile to global memory.

Inside `vec_dot`, each `IMMA` is a **16 × 8** tensor-core micro-tile (`tile_C`); all 8 warps together
cover the full **128 × 128** output tile per call.

#### Why load both W halves once, but A in two passes?

A symmetric design would load **128 K of W + 128 K of A** together, `vec_dot`, repeat — same math, but
llama.cpp does not do that. Reasons from `mmq.cuh`:

**1. Shared-memory layout is asymmetric**

```cpp
int * tile_y = data_mul_mat_q + mmq_x;
int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);
```

| Buffer | Sized for | Per `kb0` step |
|--------|-----------|----------------|
| **`tile_x`** | `mmq_y × MMQ_MMA_TILE_X_K_Q8_0` — layout includes **`2×MMQ_TILE_NE_K`** per row | Holds **both** W K-halves (256 K); `load_tiles` always fills the whole structure |
| **`tile_y`** | `mmq_x × MMQ_TILE_Y_K` ints — **one** activation K-half per column | Holds **one** A K-half (128 K); second half overwrites the buffer |

Shared allocation (`mmq_get_nbytes_shared`, ~3876): `nbs_y = mmq_x × sizeof(block_q8_1_mmq)` — one Q8_1
MMQ chunk per column, not two. k3 already uses **~58 KiB** shared / block; doubling `tile_y` would push
occupancy further (see main doc §23.13 on carveout limits).

**2. Weight reload is the expensive path**

| Design | Global loads per `kb0` step |
|--------|------------------------------|
| **Current** | 1× `load_tiles` (W, 256 K) + 2× `tile_y` fill (A, 128 K each) |
| **Symmetric** | 2× `load_tiles` (W, 128 K each) + 2× `tile_y` fill |

`load_tiles_q8_0` dominates global traffic (~86 LDG/warp, §5) and sits right before the outlier barrier
at `pc=0x10020`. Keeping `tile_x` resident across both `vec_dot` calls **avoids a second weight pass**
from global memory.

**3. Activation global layout matches the split**

The two `tile_y` fills use the same base pointer offset by `sz = sizeof(block_q8_1_mmq)/sizeof(int)` —
Q8_1 activations are already stored as **two MMQ chunks per 256-K step** in global memory
(`mmq.cuh:3410` vs `3426`).

§3 below is the source for one `kb0` step (one row of the diagram).

```
Global W_q [128 rows × K]     Global A_q [K × 128 cols]
         │                              │
    load_tiles (once)              tile_y fill (×2 per kb0)
         └──────────┬───────────────────┘
                    ▼
              tile_x , tile_y   (single shared buffers, ~58 KiB)
                    │
            vec_dot × 2  →  sum[64] per thread  (see §2.4 — not one 16×8 tile per thread)
                    │
                    ▼
         C tile [128 × 128]  (when K complete)
```

### 2.4 Inside `vec_dot`: IMMA tiles → `sum[64]` per thread

The diagram line `sum[64] per thread` means: **each of the 256 threads owns 64 fp32 slots** in the
running output accumulator for the whole 128×128 tile:

```
mmq_x × mmq_y / (nwarps × warp_size) = 128×128 / 256 = 64
```

Those 64 values are **not** one IMMA `tile_C`. They are the **thread-private portion** of the full
output tile, kept in registers across all `kb0` steps and both `vec_dot` calls.

#### Three levels of "result"

| Level | Shape / count | Who holds it | Lifetime |
|-------|---------------|--------------|----------|
| **One `IMMA`** | **A** 16×32 + **B** 32×8 int8 → **D** 16×8 int32 (`D+=A×B`); **16×8×32** MACs/warp | Each lane: **`C.x[0…3]`** (`ne=4`) | One instruction |
| **One `vec_dot` call** | Full **128×128** output tile updated (over **128** K) | **64 fp32 per thread** in `sum[]`; **8 warps** cover the tile cooperatively | One K-half of one `kb0` step |
| **Full tile K reduction** | Same **128×128** `sum[]` | Keeps **`+=`** across 2 `vec_dot` × 16 `kb0` steps | Until `write_back` |

**Glossary (terms in the table):**

- **`ne`** — **n**umber of **e**lements per thread in an MMA `tile` struct (`mma.cuh`). For
  `tile<16,8,int>` (i.e. `tile_C`), `ne = I×J/32 = 16×8/32 = **4**`. Each lane's `C.x[0…ne-1]`
  holds that lane's share of the 16×8 IMMA output. Not related to `kb0`.
- **`kb0`** — loop index in `for (int kb0 = kb0_start; …; kb0 += blocks_per_iter)` (`mmq.cuh:3407`).
  Counts **Q8_0 group index** along K (step 8 groups = 256 elements per iteration). One **`kb0` step**
  = one outer-loop pass: `load_tiles` + 2×(`tile_y` fill + `vec_dot`). Sixteen `kb0` steps complete
  the full K=4096 reduction for one 128×128 output tile.

#### One `IMMA` — hardware inputs/outputs vs C++ `tile` types

On H100, each `mma(C, A, B)` in `vec_dot_q8_0_q8_1_mma` lowers to (`mma.cuh:879`):

```text
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
```

**What the instruction computes (logical shapes, per warp):**

\[
D_{16 \times 8} \mathrel{+}= A_{16 \times 32} \times B_{32 \times 8}
\]

| Operand | Logical shape | Dtype | Role |
|---------|---------------|-------|------|
| **A** | **16 × 32** | **int8** | Weight fragment (from `tile_x` → `LDSM`) |
| **B** | **32 × 8** | **int8** | Activation fragment (from `tile_y` → shared load) |
| **D** | **16 × 8** | **int32** | Accumulator — **read + write** (`D += A×B`) |

**4096 int8 MACs per warp per instruction** (= 16×8×32). Not 16×8×8.

**Per lane:** the warp holds 128 int32 outputs total (16×8); each thread owns **`tile_C::ne = 4`**
values (`C.x[0…3]`). A and B are also **register fragments** spread across the 32 lanes in a fixed
hardware lane map.

**Easy to misread — C++ `tile<I,J,T>` names are fragment helpers, not dense matrix sizes:**

```cpp
typedef tile<16, 8, int> tile_A;   // NOT “16×8 int8 matrix”
typedef tile< 8, 8, int> tile_B;
typedef tile<16, 8, int> tile_C;
```

| C++ type | Looks like | Actually |
|----------|------------|----------|
| `tile_A<16,8,int>` | 16×8 ints | **Register wrapper** for the **16×32 int8** A fragment; `16` aligns with **M** rows of **C**; `int` = packed in 32-bit regs |
| `tile_B<8,8,int>` | 8×8 ints | **Register wrapper** for the **32×8 int8** B fragment; pairs with **N=8** of **C** |
| `tile_C<16,8,int>` | 16×8 ints | **16×8 int32** IMMA output fragment |

```
Logical (hardware IMMA):          C++ typedef (fragment layout):
A: 16 × 32  int8                   tile_A<16, 8, int>
B: 32 ×  8  int8                   tile_B< 8, 8, int>
D: 16 ×  8  int32                  tile_C<16, 8, int>
```

**After IMMA in `vec_dot`:** `D` is not the final GEMM output. Each lane applies dequant scales and
accumulates into fp32 `sum[]`:

```cpp
mma(C, A[n][k01/QI8_0], B);
sum[(j0/tile_C::J + n) * tile_C::ne + l] += C.x[l] * dA[...] * dB[...];
```

| Stage | Output | Where |
|-------|--------|-------|
| **IMMA** | 16×8 int32 partial dots | `tile_C` (registers, per warp) |
| **+ dequant** | fp32 partials | **`sum[64]` per thread** (full 128×128 tile over all IMMAs) |
| **`write_back`** | final fp32 **C** | global `dst` |

#### How one `IMMA` feeds `sum[]`

After each `mma(C, A, B)`, **every lane** does (`mmq.cuh:990-991`):

```cpp
sum[(j0/tile_C::J + n) * tile_C::ne + l] += C.x[l] * dA[...] * dB[...];
```

- `tile_C::ne = 16×8/32 = 4` → **4 `sum[]` slots updated per lane per IMMA**
- `j0` scans output columns in steps of `ntx×8 = 16` (8 bands across 128 cols)
- `k01` scans K in steps of `QI8_0 = 8` (4 times per K-half)
- `n` runs `0…ntx-1` with `ntx = 2` (two 16-row minitiles per warp)

**Result dependency is only along K:** for a fixed output element `C[i,j]`, every `sum[…] += …` across
`k01`, both `vec_dot` calls, and all `kb0` steps is the same register slot — a **K reduction**. The
`j0` and `n` loops touch **different** `sum[]` indices (different `i,j`) and are independent in that
sense.

#### Shared → register loads: operand size and reuse

Each `mma()` needs **operand registers**, but **not** a fresh shared load every time. Within one
`vec_dot`, operands come only from **`tile_x` / `tile_y`** (already staged from global memory).

**Size of each shared → register load** (logical int8 operand for one `m16n8k32`):

| Load | API | From shared | Logical operand | C++ reg tile |
|------|-----|-------------|-----------------|--------------|
| **A** | `load_ldmatrix` | `x_qs` in `tile_x` | **16 × 32** int8 (16 weight rows × 32 K) | `tile_A<16,8,int>` |
| **B** | `load_generic` | `y_qs` in `tile_y` | **32 × 8** int8 (32 K × 8 output cols) | `tile_B<8,8,int>` |

Source (`mmq.cuh:948`, `971`):

```cpp
// A: once per (n, k01), BEFORE j0 loop
load_ldmatrix(A[n][k01/8], x_qs + (i0 + n*16)*MMQ_MMA_TILE_X_K_Q8_0 + k0, ...);

// B: once per (j0, k01), INSIDE j0 loop
load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
```

**Two load schedules (one warp, one `vec_dot`):**

| Load | When | Count | IMMA key | Reuse |
|------|------|------:|----------|-------|
| **A** | preload, per `(n, k01)` | **8** (2×4) | `(n, k01, any j0)` | **×8** — same A regs for all `j0` |
| **B** | per `(j0, k01)` | **32** (8×4) | `(j0, k01, any n)` | **×2** — same B regs for `n=0,1` |

```
A preload (8 loads):  A[n,k01] ──────────────────► reused at all j0=0,16,…,112
B in loop (32 loads): B(j0,k01) ──► mma(n=0) ─┐
                                  mma(n=1) ─┘   same B, two row minitiles

IMMA at (j0, k01, n):  8 × 4 × 2 = 64 per warp per vec_dot
Shared→reg loads:      8 + 32   = 40 per warp per vec_dot  (avg 1.6 IMMA / load)
```

**Why A reuses across `j0`:** weights depend on **row + K**, not output column — same 16×32 W strip
for every column band this warp processes in one `vec_dot`.

**Why B reuses across `n`:** for fixed `(j0, k01)`, activations are the same **32×8** slice; only the
weight row minitile (`A[n]`) changes between the two `mma` calls.

**When loads are always fresh (no reuse across):**

| Change | Reload A from `tile_x`? | Reload B from `tile_y`? |
|--------|-------------------------|-------------------------|
| `j0` (column band) | No | **Yes** |
| `k01` (K step) | **Yes** | **Yes** |
| `n` (row minitile) | **Yes** (different `A[n]`) | No |
| 2nd `vec_dot` (`k00=32`) | **Yes** (other K-half of `tile_x`) | **Yes** (new `tile_y` fill) |
| next `kb0` | **Yes** (`load_tiles`) | **Yes** (`tile_y` refill) |

**IMMA count cheat sheet (one warp):**

| Scope | Spatial patch | K | IMMA (= 2×4×8) |
|-------|---------------|---|----------------|
| one `vec_dot` | 32×64 of `C` | 128 | **64** |
| one `kb0` step | same | 256 | **128** |
| full 128×128 tile | 32×64 (warp slice) | 4096 | **2048** |

#### Why `8 × 4 × 2 = 64` IMMA per warp per `vec_dot`

One `vec_dot` call is a **triple nested loop** in source (`mmq.cuh:965-994`). Each innermost
`mma(C, A[n], B)` becomes **one `IMMA` instruction for the whole warp**:

```cpp
for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {      // 8 iters — column bands
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) { // 4 iters — K along this K-half
        // load activation tile_B (32×8 int8 logical B fragment)
        for (int n = 0; n < ntx; ++n) {                    // 2 iters — row minitiles
            mma(C, A[n][k01/QI8_0], B);                   // ← 1 IMMA (warp-wide)
            // each lane: sum[...] += dequant(C.x[0..3])
        }
    }
}
```

Substituting `mmq_x=128`, `ntx=2`, `tile_C::J=8`, `MMQ_TILE_NE_K=32`, `QI8_0=8`:

| Loop var | Step | Count | What it walks |
|----------|------|------:|---------------|
| **`j0`** | `+= ntx×8 = 16` | **8** | Output **columns** 0–15, 16–31, …, 112–127 (16-col band per iter) |
| **`k01`** | `+= 8` | **4** | **K** within one K-half: 4 groups × 32 = **128** K elements |
| **`n`** | `0, 1` | **2** | Two **row** minitiles of 16 rows (= 32 rows for this warp) |

Same product as **2 × 4 × 8 = 64** (`n` × `k01` × `j0`).

```
One vec_dot (one K-half = 128 along K), ONE warp
═══════════════════════════════════════════════════════════════════

Output patch C  [ 32 rows × 64 cols ]  (this warp's slice of 128×128)

        cols →   0───16───32───48───64───80───96──112─128
              ┌──────┬──────┬──────┬─ ··· ┬──────┐
    rows  0   │ j0=0 │ j0=16│ j0=32│      │j0=112│   8 column bands (j0 loop)
         16   │ n=0  │      │      │      │      │
         32   │ n=1  │      │      │      │      │   2 row minitiles (n loop)
              └──────┴──────┴──────┴─ ··· ┴──────┘
                    each (j0, n) pair × 4 K-steps = IMMAs below

For ONE (j0, n) cell — walk K along this K-half:
        K (128 elts) →
        ├─k01=0─┬─k01=8─┬─k01=16┬─k01=24─┤
        │ IMMA  │ IMMA  │ IMMA  │ IMMA   │     4 IMMA (k01 loop); each = m16n8k32
        └───────┴───────┴───────┴────────┘

IMMA count for ONE vec_dot, ONE warp:  8 × 4 × 2  =  64
```

**One `kb0` step** runs **`vec_dot` twice** (first K-half, then second K-half of the 256-wide slice):

```
one kb0 step (256 along K), ONE warp
═══════════════════════════════════

  load_tiles ──► tile_x [ W half-1 | W half-2 ]     (both in shared)

  tile_y ← A half-1  →  vec_dot(k00=0)   →  64 IMMA ──┐
  tile_y ← A half-2  →  vec_dot(k00=32)  →  64 IMMA ──┴→ 128 IMMA / warp / kb0
                                                         (sum[] += both)

  full K=4096 for one output tile:  16 kb0 steps  →  128 × 16 = 2048 IMMA / warp
```

**Whole CTA (8 warps)** for one `kb0` step: `128 × 8 =` **~1024 IMMA** (each warp runs the same
loop on its own 32×64 patch of the 128×128 tile).

Static SASS in `enhanced_execution_info.json` lists **256 `IMMA` per loop-body template** (correlation
doc §5) — a compressed static count for the whole `kb0` body, not the same as counting dynamic
`mma()` calls from the nested loops above.

#### Per-warp vs per-thread output map

With `granularity = 16`, `rows_per_warp = 32`, `ntx = 2`:

| | Rows of `C` | Cols of `C` | `sum[]` slots |
|--|-------------|-------------|---------------|
| **One warp** | **32** | **64** | 32 threads × **64/thread** = 2,048 outputs (one patch of the tile) |
| **Whole CTA** | **128** | **128** | 256 × 64 = **16,384** (= 128×128) |

Each thread's **64 `sum[]` entries** are a fixed patch of `C` for the life of the tile; every
`IMMA` only touches **4** of those 64 per call.

#### How long does one `vec_dot` take?

There is no single hardcoded latency in source — it is the dynamic instruction span between barriers:

| Segment | Ends at `BAR.SYNC` | Typical inter-warp spread (sim, §14.3) | Dominant ops |
|---------|-------------------|----------------------------------------|--------------|
| `load_tiles` + 1st `tile_y` fill | `0x10020` | **1,273–1,416 cyc** | `LDG`/`STS` (memory) |
| **1st `vec_dot`** | `0x135b0` | **75–520 cyc** | `LDS`/`LDSM`, `IMMA`, `I2FP`/`FFMA` |
| 2nd `tile_y` fill | `0x13a10` | **~75 cyc** | small global→shared copy |
| **2nd `vec_dot`** | `0x17780` | **~75–200 cyc** | same as 1st `vec_dot` |

So **`vec_dot` is compute-bound shared-memory + tensor-core work** — much shorter inter-warp spread
than the weight-load stage, but still hundreds of cycles in the simulator because each call issues
**~64 IMMA per warp** plus dequant (`I2FP`/`FMUL`/`FFMA` — ~3× FP ops per IMMA in the static mix).

Per-IMMA pipeline latency in the sim config is modeled at **~8–32 cycles** depending on
`tensor_rate_per_cycle` (correlation doc §1.1); real H100 ubench ~**24 cyc/IMMA** in a dep chain —
but **`vec_dot` latency is not one IMMA** — it is the full nested loop plus memory reads from shared.

### 2.5 One warp within that CTA

Thread layout: `threadIdx.y` = warp id `w ∈ {0…7}`, `threadIdx.x` = lane `0…31`.

| Phase | What this warp touches | Effective slice |
|-------|------------------------|-----------------|
| **`load_tiles_q8_0`** | `i = i0 + threadIdx.y` over 16 unrolled row steps | **16 weight rows** of the 128-row tile: `w, w+8, w+16, …, w+120` — a fixed, disjoint row-set per warp |
| **`tile_y` fill** | `l = threadIdx.y·32 + threadIdx.x` | Cooperative copy of the activation K-half into shared memory (all warps, all lanes) |
| **`vec_dot_q8_0_q8_1_mma`** | `granularity = 16` → `rows_per_warp = 32`, `ntx = 2` | **32 weight rows × 64 output columns** of the 128×128 tile: warp pairs `(0,1),(2,3),…` share a 32-row band; within a pair, `threadIdx.y % ntx` splits the N columns |
| **Registers** | `float sum[mmq_x·mmq_y / (nwarps·warp_size)]` | **64 fp32 partial sums per thread** (16384 outputs / 256 threads) |

So at the warp level the "input" is not a private matrix — it is a **strided row subset of `tile_x`**, a **shared
`tile_y` column-half**, and the "output" is a **64-element register accumulator slice** that ultimately maps to
**32 rows × 64 columns** of the CTA's 128×128 output tile. All 8 warps must rendezvous at four `__syncthreads()`
per `kb0` step because `tile_x` / `tile_y` are single shared buffers written collectively and read by `vec_dot`.

### 2.6 Code map — concepts → source lines

Primary file: `llama.cpp/ggml/src/ggml-cuda/mmq.cuh` (k3 uses `GGML_TYPE_Q8_0`, `mmq_x=128`, H100 IMMA path).
IMMA lowering: `mma.cuh:879`.

#### Call chain (outer → inner)

```
mul_mat_q<GGML_TYPE_Q8_0, 128>           mmq.cuh:3495+
  └─ stream-K: kbc / kb0_start / kb0_stop  mmq.cuh:3565-3574
  └─ decode (it, jt, …)                  mmq.cuh:3577-3583
  └─ mul_mat_q_process_tile<…>           mmq.cuh:3633-3635  (fixup=false path)
        ├─ load_tiles → load_tiles_q8_0  mmq.cuh:3246-3247, 3408, 658-718
        ├─ tile_y fill (×2)              mmq.cuh:3410-3416, 3426-3432
        ├─ vec_dot → vec_dot_q8_0_q8_1_mma  mmq.cuh:3247, 3386, 3421/3437, 862-997
        └─ write_back                    mmq.cuh:3442-3446
```

`vec_dot` is chosen at compile time via `mmq_type_traits<…, GGML_TYPE_Q8_0>::vec_dot_mma` (`3244-3248`).

#### Compile-time constants (where the numbers come from)

| Symbol | Value (k3 / Q8_0) | Defined | Used for |
|--------|------------------:|---------|----------|
| `QK8_0` / `qk` | **32** | `ggml-common.h:219` | Elements per Q8_0 quant group |
| `QI8_0` | **8** | `ggml-common.h:111` (`QK8_0/(4·QR8_0)`) | `k01` loop step; index into packed `tile_x` |
| `MMQ_ITER_K` | **256** | `mmq.cuh:13` | K elements loaded per `kb0` step |
| `blocks_per_iter` | **8** | `mmq.cuh:3401` (`ITER_K/qk`) | `kb0 += 8` per outer step |
| `MMQ_TILE_NE_K` | **32** | `mmq.cuh:171` | Half-tile width in **packed tile K index**; `k00` for 2nd `vec_dot` |
| `MMQ_TILE_Y_K` | **36** | `mmq.cuh:251` (`32+32/8`) | Activation row stride in shared `tile_y` |
| `mmq_x`, `mmq_y` | **128** | template / `get_mmq_y_device()` | Output tile M×N per CTA |
| `nwarps` | **8** | `mmq_get_nwarps_device()` | `block = 32×8` |
| `sum[]` length | **64** | `mmq.cuh:3403` | `128×128/256` fp32 accumulators per thread |

#### Full K reduction — one formula, four nested loops

For one **128×128** output tile with **K = 4096**:

```
4096 K  =  16 kb0  ×  2 vec_dot  ×  4 k01  ×  32 K per IMMA
        =  32 vec_dot  ×  4 k01  ×  32 K          (since 4096/128 = 32)
```

| Level | Loop variable | Count | K per iter | Source |
|-------|---------------|------:|-----------:|--------|
| **L0** Stream-K / tile scheduler | `kbc`, `(it,jt)` | varies | — | `3565-3642` |
| **L1** Outer K (per output tile) | `kb0` | **16** | **256** | `3407` (`kb0 += blocks_per_iter`) |
| **L2** Activation half / `tile_y` pass | 2× `vec_dot(…, k00)` | **2** | **128** | `3421`, `3437` (`k00=0`, `k00=MMQ_TILE_NE_K`) |
| **L3** IMMA K steps (inside `vec_dot`) | `k01` | **4** | **32** | `945`, `967` (`k01 += QI8_0`; **128/32 = 4**) |
| **L4** One tensor-core op | `mma(C,A,B)` | **1** | **32** | `987` → `mma.cuh:879` (`m16n8k32`) |

**`k01 = 4` is not “K = 4096 / something”.** It only walks the **128 K elements already in shared** for the
current `vec_dot`. The other **3968 K** come from **31 more `vec_dot` calls** (other `k00` halves and `kb0` steps).

#### Symbol glossary ↔ code

| Name | Meaning | Where set / used |
|------|---------|------------------|
| **`kb0`** | Global **Q8_0 group index** along K (0…127 for K=4096) | Loop `3407`; passed to `load_tiles` as `kbx0` at `3408` (`offset_x + kb0`) |
| **`kb0_start` / `kb0_stop`** | This CTA's slice of `kb0` (stream-K) | Computed `3573-3574`, passed `3635` |
| **`k00`** | Base offset on **shared tile K axis** for this `vec_dot` | Arg to `vec_dot` at `3421` (`0`) / `3437` (`MMQ_TILE_NE_K`) |
| **`k01`** | Offset **within one K-half** of shared tile | Loops `945`, `967`; step `QI8_0` |
| **`k0`** | `k00 + k01` — index into `tile_x` / `tile_y` row layout | `946`, `948`, `971` |
| **`j0`** | Output **column band** (steps of 16) | `965` |
| **`n`** | Row **minitile** within warp (`ntx=2`) | `943`, `985` |
| **`sum[]`** | Running fp32 GEMM output (whole 128×128 tile) | Decl `3403`; `+=` at `991`; `write_back` `3445` |

#### L1 — `kb0` loop + global → shared (`3407-3440`)

```cpp
// mmq.cuh:3400-3401, 3407-3440
constexpr int ITER_K          = get_iter_k(type);      // 256
constexpr int blocks_per_iter = ITER_K / qk;           // 8

float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f}; // 64 per thread

for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, ...);      // W: 256 K → tile_x (both halves)

    // 1st activation K-half → tile_y
    const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
    tile_y[l] = by0[l];
    __syncthreads();                                   // BAR #1  pc≈0x10020

    vec_dot(tile_x, tile_y, sum, 0);                   // K-half 1, k00=0
    __syncthreads();                                   // BAR #2  pc≈0x135b0

    // 2nd activation K-half → overwrites tile_y
    by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
    tile_y[l] = by0[l];
    __syncthreads();                                   // BAR #3  pc≈0x13a10

    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);       // K-half 2, k00=32
    __syncthreads();                                   // BAR #4  pc≈0x17780
}
write_back(sum, ..., dst, ...);
```

**`load_tiles_q8_0` — both W halves in one call (`690-691`):**

```cpp
x_qs[row + 0             + txi] = … bxi[0].qs …;              // 1st K-half
x_qs[row + MMQ_TILE_NE_K + txi] = … bxi[MMQ_TILE_NE_K/QI8_0].qs …;  // 2nd K-half
// MMQ_TILE_NE_K/QI8_0 = 4 groups ahead in global W along K
```

#### L2+L3 — `vec_dot_q8_0_q8_1_mma` (`862-997`)

```cpp
// mmq.cuh:925-927, 937-938, 940-994
constexpr int granularity   = mmq_get_granularity_device(mmq_x); // 16
constexpr int rows_per_warp = 2 * granularity;                   // 32
constexpr int ntx           = rows_per_warp / tile_C::I;         // 2

tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];   // A[n][0..3]  — 4 k01 slots per row minitile

// --- A preload (8 loads/warp): reuse across all j0 ---
for (int n = 0; n < ntx; ++n)                    // 2
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0)  // 4
        load_ldmatrix(A[n][k01/QI8_0],
            x_qs + (i0 + n*16)*MMQ_MMA_TILE_X_K_Q8_0 + (k00 + k01), …);

// --- B + IMMA (64 IMMA/warp): j0 × k01 × n ---
for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J)    // 8 column bands
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {  // 4 K steps
        load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, …);   // B: fresh per (j0,k01)
        for (int n = 0; n < ntx; ++n) {                       // 2 row minitiles
            mma(C, A[n][k01/QI8_0], B);                     // A: reused — no load
            for (int l = 0; l < tile_C::ne; ++l)
                sum[(j0/tile_C::J + n)*tile_C::ne + l] +=
                    C.x[l] * dA[n][l/2][k01/QI8_0] * dB[l%2];  // dequant + +=
        }
    }
```

| Inner loop | Lines | Count | Reuse |
|------------|-------|------:|-------|
| `n` (A preload) | `943-949` | 2×4 = **8** `load_ldmatrix` | Each `A[n][k01/8]` used for **8** `j0` values |
| `j0` | `965` | **8** | — |
| `k01` (B+mma) | `967-994` | **4** | Each `k01` = **32** real K elements |
| `n` (mma) | `985-992` | **2** | Same `B` for both `n` |

**Spatial vs K dependency:** `sum[…] +=` at `991` — fixed `(j0,n,l)` slot accumulates across **all `k01`**
in this `vec_dot`, then across the **2nd `vec_dot`**, then across **all `kb0`** (K-only reduction).

#### L4 — `mma` → SASS (`mma.cuh:875-879`)

```cpp
// One warp, one instruction:
// D[16×8] += A[16×32] × B[32×8]   (int8 × int8 → int32 acc)
asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 …");
```

#### Shared-memory pointers (`3381-3383`, `251`, `209`)

```cpp
int * tile_y = data_mul_mat_q + mmq_x;
int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);
// tile_y: mmq_x × MMQ_TILE_Y_K  — one activation K-half at a time
// tile_x: mmq_y × MMQ_MMA_TILE_X_K_Q8_0  — both W K-halves (2×MMQ_TILE_NE_K qs region)
```

#### Stream-K outer loop (`3565-3642`) — when `kb0` ≠ full 16 steps

Not every `mul_mat_q_process_tile` call runs all 16 `kb0` iterations. `kb0_start`/`kb0_stop` bound the
slice for this CTA on tile `(it,jt)`. A CTA may call `process_tile` multiple times on different tiles;
partial K on a tile uses `fixup=true` and `tmp_fixup` (`3699+`).

#### Quick cross-check (one warp, one output tile)

| Quantity | Formula | Value |
|----------|---------|------:|
| `vec_dot` calls | `4096 / 128` | **32** |
| `k01` iters per `vec_dot` | `128 / 32` | **4** |
| IMMA per `vec_dot` | `8 × 4 × 2` | **64** |
| IMMA per `kb0` | `64 × 2` | **128** |
| IMMA per tile (one warp) | `128 × 16` | **2048** |
| Shared→reg loads per `vec_dot` | `8 A + 32 B` | **40** |

## 3. The main loop (source) — `mmq.cuh:3407-3440`

`mul_mat_q_process_tile` uses **one** `tile_x` and **one** `tile_y` in shared memory — no ping-pong buffers
(main doc §5). Each `kb0` step calls `load_tiles` **once** (both W K-halves → `tile_x`), then runs two
`vec_dot` calls that each read **one K-half of `tile_x`** paired with **one K-half of `tile_y`** (§2.2).

```cpp
for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
    // tile_x: LDG/STS — both K-halves of the weight tile (offsets 0 and +MMQ_TILE_NE_K) in one call
    {
        tile_y[l] = by0[l];   // tile_y: first activation K-half only (global → shared)
    }
    __syncthreads();            // sync #1 — pc=0x10020 (outlier BAR.SYNC in §14)

    vec_dot(tile_x, tile_y, sum, 0); // shared → regs → IMMA, k-offset 0
    __syncthreads();            // sync #2 — pc=0x135b0

    {
        tile_y[l] = by0[l];   // overwrites the SAME tile_y with the second activation K-half
    }
    __syncthreads();            // sync #3 — pc=0x13a10

    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);     // same tile_x, k-offset +MMQ_TILE_NE_K
    __syncthreads();            // sync #4 — pc=0x17780; must finish before the next load_tiles
}
```

Four `__syncthreads()` per iteration — the four recurring `BAR.SYNC` sites in §14 (`0x10020`, `0x135b0`,
`0x13a10`, `0x17780`).

Easy to misread:

- **Not double-buffered.** The next `kb0` iteration reloads `tile_x` and `tile_y` only after sync #4; there is no
  `tile_x[0]`/`tile_x[1]` alternation.
- **`tile_x` is not read wholesale by both `vec_dot` calls.** Both halves sit in shared after one `load_tiles`,
  but `vec_dot(..., 0)` uses W half-1 and `vec_dot(..., MMQ_TILE_NE_K)` uses W half-2 — see §2.2.
- **Activations only.** `tile_y` is the buffer that gets **overwritten** between the two `vec_dot` passes; `tile_x`
  is not reloaded from global between them.
- **What sync #1 waits on.** In source, both `load_tiles` and the first `tile_y` fill must complete — not
  `load_tiles` alone. In SASS, the long LDG-heavy stretch before `pc=0x10020` is dominated by `load_tiles_q8_0`
  (§5–6); that is where per-warp timing variance accumulates before the barrier exposes it.

## 4. `load_tiles_q8_0` (source) — `mmq.cuh:658-718`

```cpp
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();      // 8
    constexpr int warp_size = ...;                        // 32
    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;    // 1
    ...
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {     // mmq_y=128, step=8  -> 16 unrolled iterations
        int i = i0 + threadIdx.y;                          // <-- row index depends only on WHICH WARP (threadIdx.y)
        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;
        x_qs[...] = get_int_b2(bxi[0].qs, kqsx);            // global load (LDG) of quantized weight bytes
        x_qs[...] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
    }
    ...
}
```

**Key detail: `i = i0 + threadIdx.y`.** With `mmq_y=128` and the loop stepping by `nrows*nwarps=8`, this unrolls to
16 iterations. Warp `w` (i.e. `threadIdx.y = w`) always reads rows `w, w+8, w+16, ..., w+120` of the weight matrix
`x` — **a fixed, disjoint, strided row-set per warp, different from every other warp's row-set.** Each row read is
a `block_q8_0` struct at `x + kbx0 + i*stride + kbx` — `stride` is the matrix's row stride, so warp `w`'s 16 reads
are spread `8*stride` bytes apart across the weight matrix, a wide address footprint per warp.

## 5. Source → SASS mapping (confirmed via `extra_info/enhanced_execution_info.json`)

The segment from the loop back-edge (`BRA` target `0xf290`) through to `pc=0x10020` (250 static instructions)
disassembles to:

```
LDG.E.U16.CONSTANT  68    <- load_tiles_q8_0's "get_int_b2(bxi[...].qs, ...)" reads (x is `const __restrict__`,
LDG.E.CONSTANT      18       so the compiler routes them through the read-only/constant cache path)
STS                 54    <- "x_qs[i*...+ txi] = ..." writes into shared memory (tile_x)
PRMT                32    <- byte-permute, unpacking int32-packed quantized values (get_int_b2 internals)
IADD3/IMAD.WIDE     ~40   <- address arithmetic for `x + kbx0 + i*stride + kbx`
BAR.SYNC             1    <- the __syncthreads() at line 3419 (pc=0x10020)
```

86 global loads per warp (16 unrolled rows × ~5-6 loads/row for the two `get_int_b2` calls plus the second loop at
line 702-717 that loads the per-block scale factor `bxi->d`), all landing on this one segment, immediately
followed by the barrier. The other three barriers (`0x135b0`, `0x13a10`, `0x17780`) sit after `vec_dot` (pure
`LDS`/`LDSM`-from-shared-memory + `I2FP`/`FFMA` compute, no `LDG`) or the smaller `tile_y` refill — consistent
with their 10-20x smaller observed arrival-spread (§14.3 of the main doc).

## 6. Why this specific load produces per-warp latency variance

Because `i = i0 + threadIdx.y`, **each of the 8 warps reads a different, address-disjoint set of rows from the
same large weight matrix, every iteration.** This isn't a shared, coalesced access where all warps contend for
the *same* cache line — it's 8 independent strided access streams into different regions of `x`. Depending on
GPU's memory address interleaving (`-gpgpu_mem_addr_mapping`, `-gpgpu_n_mem`), different warps' row-sets can map
to different L2 sets and different DRAM/memory-partition queues, each with **independently varying instantaneous
congestion** (other SMs' traffic landing on the same partition this cycle, NoC routing differences, etc.).

This is exactly the kind of access pattern that would experience the *tail* of the interconnect-latency
distribution unevenly: `prefill-k3-real-hw-correlation.md` §12 measured `avg_icnt2sh_latency`=180 cycles
(memory→SM response path) but `max_icnt2sh_latency`=**1,472** cycles. If warp A's 86 loads this iteration happen
to avoid the congested tail while warp B's land on a momentarily-busy partition, warp B simply takes longer to
finish stage (A) — and since `__syncthreads()` forces a rendezvous, **every other warp idles until warp B
arrives.** Which warp is "unlucky" varies round to round (§14.3: the slowest warp at `0x10020` was warp 1 in
round 2, warp 4 in round 6, warp 1 again in round 10) — consistent with *transient* memory-system contention,
not a fixed structural imbalance tied to one warp's row assignment.

## 7. The resulting chain, end to end

```
load_tiles_q8_0's per-warp disjoint row reads (source)
        │  (each warp's 86 LDG.E.CONSTANT/LDG.E.U16.CONSTANT requests hit different L2 sets/DRAM partitions)
        ▼
per-warp variance in memory response latency (some warps see the icnt2sh tail, up to 1,472 cyc; most see ~180 cyc avg)
        │
        ▼
__syncthreads() at line 3419 / BAR.SYNC at pc=0x10020 forces all 8 warps to wait for the slowest one
        │  (spread of 1,273-1,416 cycles observed, vs 75-520 cyc at the other 3 barriers in the same loop)
        ▼
this repeats every loop iteration (the kb0 loop runs many times per kernel invocation)
        │
        ▼
inflates the simulator's "cta_barrier"-correlated starvation stall-reason bucket, without bar.sync
itself, or the occupancy model (8 warps/SM), being miscalibrated — it's a downstream symptom of
memory-subsystem tail-latency variance feeding a hard synchronization point
```

If real H100's interconnect/memory subsystem has a tighter tail (less variance between concurrently-issued
requests to different partitions) than our model's, real warps would stay more synchronized through this exact
load stage — which is consistent with real `ncu`'s measured `barrier` stall reason being only 2.57% of
warp-issue-cycles, versus our model's much larger barrier-correlated share.

## 8. What this suggests as a next lever

Per §12.5/§14.6 of the main doc: widen `-icnt_in_buffer_limit`/`-icnt_out_buffer_limit`/`-icnt_subnets` and re-run
the `[bar_arrival_trace]` capped-cycle capture, checking specifically whether the spread at `pc=0x10020` shrinks
— a more targeted, mechanism-confirmed test than just watching the aggregate `gpu_tot_sim_cycle`.
