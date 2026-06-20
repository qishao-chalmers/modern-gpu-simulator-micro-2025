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

## 2. The main loop (source) — `mmq.cuh:3407-3440`

```cpp
for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);   // (A) global loads: weight tile -> shared
    { /* read y, write tile_y[l] */ }                                    // (B) global/cached loads: activation tile
    __syncthreads();                                                     // sync #1  <-- pc=0x10020, the outlier

    vec_dot(tile_x, tile_y, sum, 0);                                     // (C) compute: shared-mem dot product
    __syncthreads();                                                     // sync #2  <-- pc=0x135b0

    { /* read y, write tile_y[l] (second half) */ }                      // (D) more activation tile loads
    __syncthreads();                                                     // sync #3  <-- pc=0x13a10

    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);                         // (E) compute: shared-mem dot product
    __syncthreads();                                                     // sync #4  <-- pc=0x17780
}
```

Four `__syncthreads()` per iteration, matching the four recurring `BAR.SYNC` sites found in §14 of the main doc.
Stage (A), `load_tiles_q8_0`, is the one immediately before the outlier barrier.

## 3. `load_tiles_q8_0` (source) — `mmq.cuh:658-718`

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

## 4. Source → SASS mapping (confirmed via `extra_info/enhanced_execution_info.json`)

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

## 5. Why this specific load produces per-warp latency variance

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

## 6. The resulting chain, end to end

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

## 7. What this suggests as a next lever

Per §12.5/§14.6 of the main doc: widen `-icnt_in_buffer_limit`/`-icnt_out_buffer_limit`/`-icnt_subnets` and re-run
the `[bar_arrival_trace]` capped-cycle capture, checking specifically whether the spread at `pc=0x10020` shrinks
— a more targeted, mechanism-confirmed test than just watching the aggregate `gpu_tot_sim_cycle`.
