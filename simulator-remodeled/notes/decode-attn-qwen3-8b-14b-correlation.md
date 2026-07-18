# Decode-layer attention correlation — Qwen3-8B vs Qwen3-14B, real H100 vs `SM90_H100` simulator

First document in the decode-attention series. Companion to the prefill k3 series
([`prefill-k3-correlation.md`](prefill-k3-correlation.md) et al.) but for the **decode** phase: one full transformer
decode layer (Q/K/V/QK-norm/RoPE/flash-attention/O-proj/FFN) of Qwen3, llama.cpp + ggml-cuda, on H100. Investigation
started from a per-kernel sim-vs-real diff of one decode layer that surfaced a single large outlier:
`flash_attn_ext_vec`, ~2x slower in simulation than on real hardware, while every other kernel in the same layer is
within ~20%.

| Item | Value |
|------|-------|
| Model / phase | Qwen3-8B and Qwen3-14B, Q8_0, decode (`-p 1024 -n 64 -ngl 99 --flash-attn 1`) |
| Outlier kernel | `flash_attn_ext_vec<128,1,Q8_0,Q8_0,false>` |
| Sim/real ratio for everything else in the layer | 0.63x – 1.21x (within noise) |
| Real H100, live Nsight **Systems** trace (ground truth) | 8B: **4.289 μs** — 14B: **4.352 μs** (nearly identical) |
| **Final, verified sim/real ratio for `flash_attn_ext_vec` in the natural embedded layer sequence, 14B** | **Gap closed.** Was 1.98x–2.07x, robust across the flush-bug fix and a trace-boundary correction (§4.4-§4.5) — found (later session) to be caused by an L2 MSHR merge-cap capacity bottleneck, fixed by `A:192:4`→`A:192:8` (§4.6). Post-fix: **0.9916x** (4.3154 μs vs real 4.352 μs) |
| Real H100 NCU **L2 Hit Rate** for this kernel | **56.30–56.36%** (L1/TEX Hit Rate only 7.44%) — confirmed leading root-cause candidate (§3, §4) |
| Simulator's own L2 hit rate for this kernel | 79.9% (cold/alone) to 86.6% (self-warmed, no flush) — **substantially over-modeled vs. real hardware's 56.3%** |
| Bug found and fixed this session | `gpu-sim.cc`'s L2-flush-between-kernels loop iterated `m_n_mem` (80 channels) instead of `m_n_mem_sub_partition` (160 sub-partitions) — silently never invalidated sub-partitions 80-159 (channels 40-79). Fixed; rebuilt; re-verified. Did **not** explain the core ~2x gap (§4) |
| Trace-boundary mistake found and fixed this session | An "embedded" filter range (`2622-2641`) accidentally included the once-per-decode-step LM-head/vocab-projection kernel (`mul_mat_vec_q`, grid=151936=Qwen vocab size) — not a per-layer kernel at all. Corrected range (`2628-2641`); re-verified the gap is unaffected (§4) |
| Model-size dependence — nuanced finding | Cold/alone condition **is** model-size-dependent (14B's bigger 80-CTA grid costs +85.8% over 8B's 64-CTA grid, vs only +1.5% on real hardware). Genuine steady-state (self-warmed, no artificial flush) condition is **model-size-independent** (8B and 14B converge to within 0.17% of each other) — the original NCU-occupancy-based hypothesis was right about steady state, wrong about why the gap was first noticed (cold-start sensitivity, not model size) |
| Status | **Resolved (§4.6).** Real root cause was an L2 MSHR merge-cap capacity bottleneck (`A:192:4` too small), not the hit-rate-modeling question §6 originally framed as the lead. Fixed via config change; gap closed to 0.99x. Whether the hit-rate-over-crediting observation (§4.5) is fully explained by the same fix has not been independently re-measured — see §6 |

---

## 1. Background: what's in one decode layer

Reverse-engineered from `Qwen_14B_kernel_timing.txt` (tracer's own kernel-id→name dump for a captured Qwen3-14B run)
and confirmed by counting `flash_attn_combine_results` occurrences (exactly 40 = Qwen3-14B's layer count, each
spaced exactly 21 kernels apart in the decode region, kernel_id 2620–3474). One decode layer, in order:

| # | Kernel | Role |
|---|---|---|
| 1 | `rms_norm_f32<1024>` | attention input layernorm |
| 2-3 | `quantize_q8_1` → `mul_mat_vec_q<...,Lb0>` | Q projection |
| 4 | `rms_norm_f32<256>` | Q-norm (Qwen3-specific QK-norm) |
| 5 | `rope_neox<f32,f32>` | RoPE on Q |
| 6-7 | `quantize_q8_1` → `mul_mat_vec_q<...,Lb0>` | K projection |
| 8-9 | `quantize_q8_1` → `mul_mat_vec_q<...,Lb0>` | V projection |
| 10 | `rms_norm_f32<256>` | K-norm |
| 11 | `rope_neox<f32,half>` | RoPE on K |
| 12 | `k_set_rows` | write K/V into KV cache |
| 13 | `flash_attn_ext_vec<128,1,Q8_0,Q8_0,false>` | attention |
| 14 | `flash_attn_combine_results` | combine split-KV partials |
| 15-16 | `quantize_q8_1` → `mul_mat_vec_q<...,Lb1>` | O projection |
| 17 | `rms_norm_f32<1024>` | post-attention / FFN pre-norm |
| 18-19 | `quantize_q8_1` → `mul_mat_vec_q<...,Lb1>` | **fused** gate+up FFN projection + SiLU (see §5) |
| 20-21 | `quantize_q8_1` → `mul_mat_vec_q<...,Lb1>` | down projection (or second fused part) |

`Lb0`/`Lb1` is the `has_fusion` template bool on `mul_mat_vec_q` (confirmed from `ggml-cuda` source, §5) — `Lb1`
calls have a `ggml_cuda_mm_fusion_args_device` payload (gate-projection pointer + GLU op) fused into the same GEMV
launch, which is also why there's no standalone SiLU (`unary_gated_op_kernel`) kernel anywhere in the decode region
even though there is one per layer in prefill.

**Correction (found this session): don't filter a layer window starting from an arbitrary kernel_id without checking
what's actually there.** Inspecting the 14B partial re-trace kernel-by-kernel (`/tmp/pb_py` + `trace_pb2`) around
kernel_id 2622-2641 found that 2622-2627 are *not* part of the attention layer at all — they're the **tail of the
previous decode step**: `rms_norm_f32<1024>` at 2625 is the model-final norm (not a per-layer pre-attention norm),
and 2626-2627 (`quantize_q8_1` → `mul_mat_vec_q`, **grid = 151936**) is the **LM-head / vocabulary-logits
projection** — 151936 is exactly Qwen's vocab size, not a per-layer dimension. This kernel runs once per *decode
step* (once per generated token), not once per layer, and is far more expensive to simulate than anything inside a
single attention block. The real layer boundary for this window is kernel **2628** (`rms_norm_f32<1024>`, grid=1).
Also confirmed directly from trace grid sizes (not assumed from public spec): kernel 2637 (`rms_norm_f32<256>`,
K-norm) has **grid=8 = `num_kv_heads`**, hard-confirming the GQA config used throughout this doc's calculations.

### 1.1 KV-cache size: one token, one layer, all layers

Inputs, with provenance noted since this matters for how much to trust the numbers:

- `head_dim = 128` — **hard fact**, from the kernel template itself (`flash_attn_ext_vec<128,...>`).
- `num_kv_heads = 8` (both models) — **hard fact for 14B** (kernel 2637's grid, above); for 8B this is the published
  Qwen3 GQA config (32 query heads / group size 4 = 8 kv heads), not independently re-derived from a trace this
  session.
- Cache dtype = `Q8_0` — **hard fact**, from the kernel template (`...,Q8_0,Q8_0,...`). One `Q8_0` block packs 32
  elements as 1×fp16 scale (2 B) + 32×int8 (32 B) = 34 bytes / 32 elements = **1.0625 bytes/element**.
- Layer count: 40 for 14B (**hard fact**, confirmed via `flash_attn_combine_results` spacing above); 36 for 8B
  (published Qwen3 spec, **not** independently confirmed — the 8B trace available this session only spans 2 layers,
  too short to count from).

**Per token, per layer** (K and V are separate buffers of identical size):

```
K (or V) alone = num_kv_heads × head_dim × bytes/elem = 8 × 128 × 1.0625 = 1088 bytes
K + V combined  = 2176 bytes/token/layer
```

**Per layer, full context:**

| Context length | Per-layer KV (K+V) |
|---|---|
| 1024 tokens (`-p 1024`, end of prefill) | 2176 × 1024 = 2,228,224 B ≈ **2.13 MiB** (2.23 MB) |
| 1087 tokens (`-p 1024 -n 64`, last decoded token) | 2176 × 1087 = 2,365,312 B ≈ **2.26 MiB** (2.37 MB) |

**All layers, full model:**

| Model | Layers | At 1024 ctx | At 1087 ctx |
|---|---|---|---|
| Qwen3-14B | 40 (confirmed) | 40 × 2.13 MiB = **85.1 MiB** (89.1 MB) | 40 × 2.26 MiB = **90.2 MiB** (94.6 MB) |
| Qwen3-8B | 36 (spec, unconfirmed) | 36 × 2.13 MiB = **76.6 MiB** (80.2 MB) | 36 × 2.26 MiB = **81.2 MiB** (85.2 MB) |

For scale: the simulator's modeled L2 on `SM90_H100_l2norm_l1dnorm` is `gpgpu_n_mem=80` × `n_sub_partition_per_mchannel=2`
= 160 sub-partitions × `N:64:128:40` (64 sets × 128B line × 40-way) = 320 KiB/partition → **≈50 MiB total**, matching
real H100's 50MB L2 almost exactly. So a *single layer's* KV cache (≈2.1-2.3 MiB) fits comfortably in either L2, but
the *full cross-layer* KV cache (≈77-94 MB depending on model/context) does not — relevant context for §4's discussion
of whether L2 capacity could plausibly hold "the whole KV cache" across a continuous multi-layer decode run (it
can't; only one layer's slice at a time could realistically stay resident).

## 2. Per-kernel sim-vs-real, one decode layer (Qwen3-14B)

Real = live Nsight **Systems** trace (not NCU — NCU's kernel-replay/memory-backup overhead inflates duration, see
§3). Sim = `sim_qwen3_14b.log`, first occurrence of each kernel in the decode region.

| Kernel | Real (μs) | Sim (μs) | Sim/Real |
|---|---|---|---|
| `rms_norm_f32<1024>` (attn pre-norm) | 5.952 | 5.1846 | 0.87x |
| `quantize_q8_1` | 2.048 | 1.5809 | 0.77x |
| `mul_mat_vec_q` (Q proj) | 20.256 | 23.4062 | 1.16x |
| `rms_norm_f32<256>` (Q-norm) | 2.304 | 2.1679 | 0.94x |
| `rope_neox` (Q) | 2.048 | 2.2185 | 1.08x |
| `quantize_q8_1` | 2.240 | 1.5815 | 0.71x |
| `mul_mat_vec_q` (K proj) | 6.048 | 6.8883 | 1.14x |
| `quantize_q8_1` | 1.920 | 1.5821 | 0.82x |
| `mul_mat_vec_q` (V proj) | 6.112 | 7.1191 | 1.16x |
| `rms_norm_f32<256>` (K-norm) | 2.304 | 2.0191 | 0.88x |
| `rope_neox` (K) | 2.240 | 2.0790 | 0.93x |
| `k_set_rows` | 1.856 | 1.4895 | 0.80x |
| **`flash_attn_ext_vec`** | **4.352** | **8.8123** | **2.03x** |
| `flash_attn_combine_results` | 2.176 | 1.9414 | 0.89x |
| `quantize_q8_1` | 2.048 | 1.5809 | 0.77x |
| `mul_mat_vec_q` (O proj, fused) | 20.928 | 23.7241 | 1.13x |
| `rms_norm_f32<1024>` (FFN pre-norm) | 5.632 | 5.2074 | 0.93x |
| `quantize_q8_1` | 2.080 | 1.5809 | 0.76x |
| `mul_mat_vec_q` (gate+up fused, Lb1) | 124.096 | 137.8210 | 1.11x |
| `quantize_q8_1` | 2.592 | 1.6265 | 0.63x |
| `mul_mat_vec_q` (Lb1) | 63.808 | 76.9105 | 1.21x |

`flash_attn_ext_vec` is the only kernel outside the **0.63x-1.21x** band every other kernel in the layer falls into.
A second-layer instance shows the gap getting *worse*, not better: sim cycle=16706 (10.3123 μs) vs the same ~4.352
μs real baseline → **2.37x**.

Note: `cpy_scalar_contiguous` appears once per layer in the sim log (cycle=2413, 1.4895 μs) between `k_set_rows` and
`flash_attn_ext_vec` but wasn't in the pasted real-HW excerpt, so it's omitted from the ratio table above rather
than guessed at.

## 3. NCU occupancy is identical between models, but the verified sim-vs-real gap is not

Real H100, `ncu --kernel-name flash_attn_ext_vec --launch-skip 1 --launch-count 1`, `llama-bench -p 1024 -n 64 -b 2048
-ngl 99 --flash-attn 1 --no-warmup -r 1`:

| Metric | Qwen3-8B | Qwen3-14B (run 1) | Qwen3-14B (run 2) |
|---|---|---|---|
| Grid (blocks) | (1,2,32) → **64** | (1,2,40) → **80** | (1,2,40) → **80** |
| Block size | (32,4,1) = 128 threads | 128 | 128 |
| Registers/thread | 168 | 168 | 168 |
| Elapsed Cycles | 10291 | 12435 | 12433 |
| SM Active Cycles | 3193.67 | 4037.05 | 3877.64 |
| NCU Duration | 6.43 μs | 7.78 μs | 7.81 μs |
| Theoretical Occupancy | 18.75% | 18.75% | 18.75% |
| **Achieved Occupancy** | **6.21%** | **6.21%** | **6.21%** |
| **Achieved Active Warps/SM** | **3.97** | **3.97** | **3.97** |
| Compute (SM) Throughput | 5.44% | 5.63% | 5.63% |
| Memory Throughput | 10.43% | 8.68-8.70% | 8.68-8.70% |
| Waves/SM | 0.16 | 0.20 | 0.20 |

Occupancy, register pressure, and active-warps/SM are bit-for-bit identical between 8B and 14B — both are
register-limited to 18.75% theoretical occupancy (12 warps/SM max), achieving only 6.21% / 3.97 warps/SM in practice.
`flash_attn_ext_vec<128,...>` is the same template instantiation regardless of model size (only depends on
`head_dim=128`), so this is expected. Grid size scales with head count, not occupancy: 64 blocks (8B, 32 heads) vs 80
blocks (14B, 40 heads), both well under the H100's 132 SMs.

**But NCU's "Duration" is not directly comparable to live execution**, and once the *actual* live-trace ground truth
is in hand for both models, the occupancy-identity story stops explaining anything. Both NCU runs hit `Backing up
device memory in system memory. Kernel replay might be slow` — NCU's default `--replay-mode kernel` re-executes the
kernel multiple times (9 passes observed) to collect counters, snapshotting/restoring the full GPU memory state
(8-15GB of resident weights/KV-cache) around every replay, and forcibly serializes/isolates the kernel from whatever
concurrent traffic surrounds it in normal execution. This inflates Duration above live execution, and — newly
confirmed — **does so by a different amount for each model**:

| Source | Qwen3-8B | Qwen3-14B | Ratio (14B/8B) | Sim-or-NCU / live-trace |
|---|---|---|---|---|
| NCU Duration | 6.43 μs | 7.78-7.81 μs | 1.21x | 1.50x / 1.79-1.80x |
| **Live Nsight Systems trace (ground truth)** | **4.289 μs** | **4.352 μs** | **1.01x** (~identical) | 1.00x |
| Sim: alone (cold start, zero preceding kernels) | 5.1932 μs | 9.6463 μs | 1.86x | 1.21x / 2.22x |
| Sim: 3-kernel isolated window (`k_set_rows`→`cpy_scalar_contiguous`→this kernel) | 5.2179 μs | 9.5951 μs | 1.84x | 1.22x / 2.21x |
| Sim: self-warmed (identical kernel run twice back-to-back, no flush; 2nd run shown) | 5.3111 μs | 11.9191 μs | 2.24x | 1.24x / 2.74x |

All three sim rows were re-run from scratch this session against the verified `SM90_H100_l2norm_l1dnorm` config (the
config `sim_qwen3_14b.log` and the 8B logs actually used, confirmed by grepping `-gpgpu_n_mem`/`-gpgpu_cache:dl2` out
of the logs) and are mutually consistent (alone ≈ isolated-window within ~0.5-1%, as expected since the two small
preceding kernels barely touch DRAM). They replace an earlier, unreproducible "isolated window" figure of
87297-88056 cycles (47.7-48.1 μs) that does not appear in any saved log and could not be regenerated under the same
config — see §4 for the corrected methodology and numbers.

Real hardware's *live* duration barely moves between the two models (+1.5%, basically noise) — but NCU's replay
overhead inflates 14B's measured duration by nearly double the proportion it inflates 8B's (1.80x vs 1.50x). The
same asymmetry shows up in the simulator (§4): going from 8B's 64-block grid to 14B's 80-block grid (+25% CTAs)
costs the simulator +85.8% more cycles in the "alone" condition, while it costs real hardware nothing. The common
thread across NCU-on-real-hardware and the simulator: **isolating this severely register-starved, ~4-warp/SM kernel
from its natural concurrent execution context penalizes the bigger grid disproportionately**, on both real silicon
and in the model. The simulator's part of this is now precisely quantified in §4; the real-hardware/NCU part is
recorded here as corroborating evidence, not yet explained at a deeper (e.g. memory-controller queueing) level.

**Conclusion of §3 (revised)**: the original "model-size-independent, occupancy explains everything" conclusion does
not survive contact with live-trace data for 8B. Occupancy/register pressure being identical between models was a
red herring — it determines *that* this kernel is latency-sensitive, not *how much* the sim (or NCU) error scales
with grid size. The actual sim-vs-real ratio is **~1.2x for 8B and ~2.0-2.7x for 14B** (§4), confirmed with
reproducible, from-scratch re-runs against the verified `SM90_H100_l2norm_l1dnorm` config and live-trace ground truth
for both models. (§4 later shows the cold-start/grid-size sensitivity, not model size per se, is the actual driver of
this split — see the summary table's "model-size dependence" row.)

### 3b. The real lead: NCU's actual cache hit-rate metrics (found this session, previously missing)

A fuller NCU capture (`--set full` equivalent — includes the **Memory Workload Analysis** section, which the earlier
Speed-of-Light-only captures in the table above didn't have) gives the metric that actually matters for everything
chased in §4 — real cache hit rate, not just throughput-%:

| Metric | Qwen3-14B (instance 1) | Qwen3-14B (instance 2) |
|---|---|---|
| L1/TEX Hit Rate | 7.44% | 7.44% |
| **L2 Hit Rate** | **56.36%** | **56.30%** |
| L2 Compression Success Rate | 0% | 0% |
| Memory Throughput | 151.96 GB/s | 157.66 GB/s |
| Elapsed Cycles | 10111 | 10121 |
| Duration | 7.04 μs | 6.78 μs |

Also from this capture: **Warp State Statistics** show 2.1 of the average 5.5 cycles between issued instructions
(37.7%) spent stalled on an L1TEX scoreboard dependency, and **Scheduler Statistics** show only 18.12% "One or More
Eligible" (each scheduler issues an instruction only every 5.5 cycles, with just 1.01 active warps/scheduler and 0.18
eligible) — consistent with the occupancy story in the table above, but the **L2 Hit Rate of 56.3%** is the new,
concrete number this section was missing. §4 compares this directly against the simulator's own L2 hit rate and
finds a large, unexplained gap — likely the actual root cause of the whole investigation.

## 4. Root-causing the gap: a real simulator bug, a trace-boundary mistake, and the genuine remaining culprit

**This entire section originally reported isolated-window numbers (87297/88056 cycles, "<1% diff, 6.1x cold-cache
sensitivity") that turned out to be unreproducible — no saved log ever supported them, and re-running the same
methodology from scratch gave completely different (much lower) numbers. That whole narrative is retracted below and
replaced with verified, reproducible methodology and results.**

### 4.1 Verified methodology

Two kinds of test, both using the confirmed `SM90_H100_l2norm_l1dnorm` config and `-is_extra_traces_enabled 1`:

1. **Real-trace windows** via `-filter_first_kernel_id`/`-filter_last_kernel_id` on the actual 14B/8B decode traces
   (`qwen14b/decode_traces/`, `qwen8B/modern/new_decode_traces/`) — "alone" (just the kernel itself) and "3-kernel
   isolated window" (`k_set_rows` → `cpy_scalar_contiguous` → `flash_attn_ext_vec`).
2. **Synthetic self-warm traces** — built by hand from `/tmp/pb_py` (`trace_pb2`/`threadblock_pb2`), since no two
   *naturally occurring* kernels in the real trace are identical. Method: load the real trace's `flash_attn_ext_vec`
   kernel proto (`kernel.proto`: id/name/grid/block/regs/shmem), clone it twice with `id=0` and `id=1`, copy its
   per-CTA `threadblock` `.pb` files into new `kernel_0/`/`kernel_1/` directories (renamed to match), and populate
   `cuda_stream.ordered_cuda_events` with `"kernel-0.trace"`/`"kernel-1.trace"` (the actual string format
   `trace_parser.cc`'s `parse_commandlist_file()` expects — verified by reading the parser, not guessed). This runs
   the *exact same* kernel twice, back-to-back, with no cache flush forced between them other than whatever the
   config does on its own. Trace files: `gpu_traces/qwen_fa_selfwarm{,_14b}/traces/dynamic_trace.pb`.

### 4.2 Bug found: L2 is only half-flushed between kernel launches

Diffing the synthetic test's first-kernel vs. second-kernel `L2_cache_bank[N]` stats (`gpu-sim.cc`'s end-of-kernel L2
stat dump) found a **clean, sustained split at exactly sub-partition 80** (out of 160 total): banks 0-79 showed
`Miss` doubling between kernel 1 and kernel 2 (no benefit at all from "self-warming"), while banks 80-159 showed
`Miss` staying *bit-for-bit identical* — meaning kernel 2 hit every single access to those banks. 80 is suspiciously
exactly half of 160 (`gpgpu_n_mem=80` × `gpgpu_n_sub_partition_per_mchannel=2`). Root cause, found in `gpu-sim.cc`
(the `-gpgpu_flush_l2_cache` block, confirmed enabled — `1` — in this config):

```cpp
// before (bug): loop bound is m_n_mem (channel count, 80) but m_memory_sub_partition[]
// is indexed by sub-partition id (0..159) — silently skips sub-partitions 80-159 forever
for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
  dlc = m_memory_sub_partition[i]->invalidateL2();
  ...
}
```

Fixed to `m_memory_config->m_n_mem_sub_partition` (160), rebuilt, and re-verified: channels 40-79's L2 state is no
longer permanently un-flushable. This is a genuine simulator correctness bug, not specific to this investigation —
any config with `gpgpu_n_mem` not equal to `gpgpu_n_mem_sub_partition` and `-gpgpu_flush_l2_cache 1` was affected.

### 4.3 Bug found: the "embedded" filter window included an unrelated, much heavier kernel

§1's correction explains this: the original embedded filter (`2622-2641`) accidentally included the once-per-decode-
step LM-head kernel (`mul_mat_vec_q`, grid=151936=vocab size) at kernel_id 2627 — not part of the attention layer.
Re-running with the corrected range (`2628-2641`) confirmed this didn't change the actual `flash_attn_ext_vec`
number (see 4.5) — it only made the *simulation* far slower to run (151936 CTAs vs. ≤5120 for everything else in the
window), since each kernel's own reported cycle count is unaffected by an earlier kernel's grid size, only by
whatever cache/DRAM state it leaves behind.

### 4.4 Final, verified numbers (post-fix, post-correction)

| Condition | 8B | 14B | 14B/8B ratio |
|---|---|---|---|
| Alone (cold start, zero preceding kernels) | 8413 cyc / 5.1932 μs | 15627 cyc / 9.6463 μs | 1.86x |
| 3-kernel isolated window | 8453 cyc / 5.2179 μs | 14921 cyc / 9.2105 μs | 1.76x |
| Self-warmed, flush bug fixed (2nd of 2 identical launches) | 8436 cyc / 5.2074 μs (+0.27% vs alone) | 13670 cyc / 8.4383 μs (−12.5% vs alone) | — |
| **Self-warmed, `-gpgpu_flush_l2_cache 0`** | **6581 cyc / 4.0623 μs (−21.8%)** | **6592 cyc / 4.0691 μs (−57.8%)** | **1.002x** |
| Embedded in natural layer (corrected range 2628-2641), flush-fixed | — | 14574 cyc / 8.9963 μs | — |
| Embedded in natural layer, no-flush | — | 13943 cyc / 8.6068 μs | — |
| Original (buggy-flush, contaminated-range) embedded baseline | — | 14276 cyc / 8.8123 μs | — |

Two things this settles:

1. **The flush-bug fix barely moves the embedded number** (14276 → 14574/13943, ±2%) — because the natural 12-kernel
   preamble never repeats `flash_attn_ext_vec`'s own addresses, so the bug (which only mattered for literally-
   repeated access patterns) was never actually exercised by the realistic decode sequence. The original §2/§3
   "~2.03x" finding was correct all along, just for the wrong reason (it was never affected by the bug it later
   turned out the synthetic self-warm test surfaced).
2. **Self-warming with the bug fixed barely changes anything** (+0.27% for 8B, even *−12.5%* for 14B — likely
   scheduler-state noise, not cache locality). **Only with flush forced fully off** does self-warming show a large,
   genuine effect — and at that point 8B and 14B converge to within 0.17% of each other, recovering the original
   model-size-independence hypothesis, but only in this idealized "perfect locality" condition that has no real
   on-hardware equivalent of normal decode (real GPUs don't flush L2 between every kernel in a stream, but neither
   does any single real kernel get to replay against its own exact prior footprint the way this synthetic test does).

### 4.5 The real, settled gap — and the actual lead

Comparing the corrected embedded numbers against the one number we trust (4.352 μs, live Nsight Systems, 14B):

- Flush-fixed: 8.9963 / 4.352 = **2.068x**
- No-flush: 8.6068 / 4.352 = **1.978x**
- Original buggy/contaminated baseline: 8.8123 / 4.352 = 2.025x

**All three land within 2% of each other.** Neither the flush bug nor the LM-head contamination explains the ~2x
gap — it is real, robust, and was never actually caused by either artifact found this session.

The genuine lead, found by computing the simulator's own overall L2 hit rate from the `L2_cache_bank` stats and
comparing against NCU's real **L2 Hit Rate** metric (§3b, 56.30-56.36%):

| Condition | Simulator's own L2 hit rate | Real H100 (NCU) |
|---|---|---|
| Alone/cold (14B) | 79.9% (8686 misses / 43200 accesses) | — |
| Self-warmed, no-flush, cumulative (14B) | 86.6% (17372 misses / 129408 accesses) | — |
| — | — | **56.30-56.36%** |

The simulator over-credits L2 hit rate for this kernel's access pattern by **24-30 percentage points**, in every
condition tested. Strikingly, the condition with the *best* cycle-count agreement with real hardware (no-flush
self-warmed, 0.94-0.95x) has the *worst* hit-rate agreement (86.6% vs 56.3%) — strong evidence that close cycle-count
agreement there is **compensating errors**, not correct modeling: the simulator over-estimates hit rate (cheaper
access) while apparently under-modeling some other latency cost (DRAM/L2 access latency, scoreboard stall cost),
and the two errors partially cancel. The real, still-open question is **why does the simulator's L2 model predict
56-87% reuse for this kernel's KV-cache access pattern when real hardware only achieves 56%** — see §6.

### 4.6 Real root cause found and fixed (later session): L2 MSHR merge-cap reservation fails — the ~2x gap, closed

The actual mechanism behind the ~2x gap (§4.4-§4.5) turned out to be neither the L2 flush-loop bug (§4.2,
confirmed to barely move the embedded number) nor the LM-head contamination (§4.3) — it was a structural
capacity limit in the L2 MSHR merge logic, found in a later debugging session via a completely different
diagnostic trail:

- `L2_cache_stats_breakdown[TEXTURE_ACC_R][RESERVATION_FAIL]` for this kernel came back at **1,171,248** —
  an enormous count of L2-side reservation failures, meaning `m_icnt_L2_queue` was repeatedly trying to
  issue requests that L2 couldn't accept.
- Traced into the MSHR logic in `gpu-cache.cc`: the L2 cache's merge cap (`A:192:4` — 192 sets, 4-way merge
  per MSHR entry) couldn't hold enough simultaneously-outstanding *unique* addresses for this kernel's
  access pattern (multiple concurrent long `LDG.E.128.CONSTANT` loads across the kernel's ~4 active
  warps/SM, per the occupancy numbers in §3). Confirmed this was a genuine capacity/serialization
  bottleneck rather than a bandwidth-bound condition by checking DRAM throughput wasn't saturated.
- **Fix**: bumped the L2 MSHR merge cap from `A:192:4` to `A:192:8` (config-only,
  `SM90_H100_l2norm_l1dnorm/gpgpusim.config`'s `-gpgpu_cache:dl2`).

**Result**, embedded natural-layer-boundary run (kernel filter 2628-2641, 14B), measured against the one
number trusted throughout this doc (4.352 μs, live Nsight Systems):

| Config | Sim time | Ratio vs real |
|---|---|---|
| Before (`A:192:4`) | 8.9963 μs | 2.068x |
| **After (`A:192:8`)** | **4.3154 μs** | **0.9916x** |

This closes the entire gap from §2's table — the kernel that was the sole outlier in the whole decode layer
is now within (in fact tighter than) the 0.63x-1.21x band every other kernel already fell into.

**Open thread**: §4.5 separately found the simulator's own L2 hit rate for this kernel (79.9-86.6%) was far
above real hardware's NCU-measured 56.3%. An elevated `RESERVATION_FAIL` count is *plausibly* connected to
that (capacity-starved MSHRs causing retried/merged L1-side accesses that could inflate hit accounting),
but this has **not been independently re-measured post-fix** — no rerun has recomputed this kernel's L2 hit
rate with `A:192:8` to check whether it now lands closer to 56%. Treat the hit-rate question as still open
until that's done (§6).

**Validation scope**: confirmed only for this one kernel and this one trace window so far. `A:192:4`→`8` is
a generic L2 sizing knob affecting every kernel in the config, so it should be re-validated against the
other ~12 already-correlated kernels in the natural layer sequence (§2) and against the 8B trace before
being treated as a settled new default — not yet done.

## 5. ggml-cuda source: confirming the `Lb1`/`Lb0` fusion and the missing SiLU kernel

From `/home/qshao/Project/Fun/llama.cpp/ggml/src/ggml-cuda/`:

- `common.cuh:1480`, `ggml_cuda_mm_fusion_args_device`: `{ x_bias, gate, gate_bias, glu_op, split2_draft,
  q4_k_res_draft, split2_debug_print }`.
- `mmvq.cu`, `mul_mat_vec_q<type, ncols_dst, has_fusion, is_multi_token_id>`: when `has_fusion=true` (`Lb1`), the
  kernel computes **two** dot products per launch — the primary projection (`vx`) and the gate projection
  (`fusion.gate`) — accumulates both, then (mmvq.cu:465-484) applies the GLU activation inline
  (`GGML_GLU_OP_SWIGLU` → `ggml_cuda_op_silu_single`, or GEGLU/SWIGLU_OAI) and multiplies, before writing one
  result. This is exactly why decode has no standalone `unary_gated_op_kernel` (SiLU) the way prefill does (prefill:
  `unary_gated_op_kernel` appears once every 33 kernels/layer in `Qwen_14B_kernel_timing.txt`; decode: zero
  occurrences after kernel_id 2600) — gate+up+SiLU+multiply is fused into one `Lb1` GEMV.
- Fusion eligibility decided graph-side in `ggml-cuda.cu` (`ggml_cuda_should_fuse_mul_mat_vec_q`,
  `ggml_cuda_can_fuse`), not something this repo's tracer/simulator controls.

## 6. Open questions — what "digging deeper" means next

1. **[Resolved differently than expected] Why does the simulator's L2 model over-credit hit rate by 24-30 points
   for this access pattern?** This was originally framed as the central open question, but the actual ~2x cycle
   gap turned out to be an L2 MSHR merge-cap capacity bottleneck (§4.6), not a hit-rate-modeling error per se.
   **Still genuinely open**: whether the 79.9-86.6% vs 56.3% hit-rate mismatch itself is now closed by the same
   `A:192:8` fix, or is a separate, still-real modeling gap — re-measure this kernel's L2 hit rate post-fix and
   compare against NCU's 56.3% before concluding either way.
2. **Extend the `A:192:8` MSHR fix's validation.** Confirmed for `flash_attn_ext_vec` only, on the 14B trace, one
   layer window. Re-run the other ~12 already-correlated kernels in §2's table and the 8B trace to confirm the
   wider merge cap doesn't regress anything currently in the 0.63x-1.21x band before treating it as a new default
   (see `notes/cache-invalidation-l1c-l1t-fix.md` for an unrelated correctness fix found in the same investigative
   arc — general L1C/L1T cache-invalidation bug, not specific to this kernel).
3. **Get a real embedded (natural-layer) number for 8B**, the way §4.4 has for 14B. Currently only alone/isolated-
   window numbers exist for 8B; an embedded 8B run (correct layer-boundary kernel range, analogous to 14B's
   `2628-2641`) would let the model-size comparison in the summary table be made on a fully apples-to-apples basis
   rather than mixing "embedded for 14B" against "alone for 8B."
4. **Real per-SM stall-reason breakdown.** The k3 series (`prefill-k3-real-hw-correlation.md` §6) already built a
   real-vs-sim "No Eligible"/stall-reason validation methodology via NCU's Warp State Statistics / Scheduler
   Statistics sections (§3b above now has these for `flash_attn_ext_vec` on real hardware: 81.88% "No Eligible",
   1.01 active warps/scheduler, 2.1/5.5 cycles stalled on an L1TEX scoreboard dependency) — worth comparing against
   this simulator's own stall-reason instrumentation (`[issue_wait_trace]`, mem-stage latency debug from
   `mem_fetch.h`) for the same kernel, now that the gap is closed and a finer-grained check is possible.
5. **Confirm whether the `-gpgpu_flush_l2_cache 1` semantics themselves are even the right default for this config.**
   Real H100 doesn't flush L2 between kernel launches in the same CUDA stream. The flush-bug fix (§4.2) made the
   *intended* behavior of this flag fully correct, but whether the flag *should* be on at all for steady-state
   decode-loop accuracy (vs. some other use case it was added for) is a separate, unresolved question — worth
   checking other already-validated kernels/configs to see if `-gpgpu_flush_l2_cache 0` makes their sim/real ratios
   better or worse before changing any default.
6. **Rule out KV-cache length as a confound for the 8B/14B comparison.** `flash_attn_ext_vec`'s real-hardware cost
   scales with how many tokens are already in the KV cache. Confirm the captured generation step (`n_past`) was
   comparable between the 8B and 14B captures used in §3b's live-Nsight-Systems numbers (4.289 vs 4.352 μs) — both
   were captured under the same `-p 1024 -n 64` benchmark, which is reassuring, but worth double-checking the exact
   decode step index of each pasted trace excerpt.

## 7. Data sources used in this doc

- `/home/qshao/Project/Fun/gpu_traces/qwen14b/Qwen_14B_kernel_timing.txt` — tracer kernel-id→name→duration_ns dump,
  used to find the decode-layer boundary (kernel_id 2620) and confirm 40 layers × 21 kernels via
  `flash_attn_combine_results` spacing.
- `/home/qshao/Project/Fun/gpu_traces/qwen14b/sim_qwen3_14b.log` — GPGPU-Sim per-kernel `gpu_sim_cycle`/`time_us`
  output for the original (buggy-flush, contaminated-range) 14B decode trace; superseded by §4.4's corrected runs but
  kept as the original §2 per-kernel-ratio source (those numbers are unaffected by either bug).
- `/home/qshao/Project/Fun/gpu_traces/qwen14b/flash_attention_nsight_compute.txt` — first NCU profile of
  `flash_attn_ext_vec`, Qwen3-14B, Speed-of-Light section only (no hit-rate metrics — see below).
- NCU runs with full Memory Workload Analysis (8B, two 14B repeats) and live Nsight Systems "Name/Start/Duration"
  excerpts (8B: 4.289 μs; 14B: 4.352 μs) pasted directly into this conversation — not yet saved to a file in this
  repo; worth copying into `gpu_traces/qwen14b/` (and a new `gpu_traces/qwen8b/`) if this investigation continues.
- `/tmp/pb_py` (`trace_pb2`, `threadblock_pb2`) — generated protobuf Python bindings used throughout this session for
  ad-hoc trace inspection and for constructing the synthetic self-warm traces (§4.1).
- Synthetic self-warm traces built this session:
  `/home/qshao/Project/Fun/gpu_traces/qwen_fa_selfwarm/traces/dynamic_trace.pb` (8B),
  `/home/qshao/Project/Fun/gpu_traces/qwen_fa_selfwarm_14b/traces/dynamic_trace.pb` (14B).
- Re-run logs from this session (all using `SM90_H100_l2norm_l1dnorm`, `-is_extra_traces_enabled 1`):
  `/tmp/fa_selfwarm_{8b,14b}_fixed.log`, `/tmp/fa_real_isolated_{8b,14b}_fixed.log`,
  `/tmp/fa_selfwarm_{8b,14b}_noflush.log`, `/tmp/embedded_14b_{fixed,noflush}_v2.log` (corrected `2628-2641` range).
- Code fix: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc`, the `-gpgpu_flush_l2_cache` block — loop bound changed
  from `m_memory_config->m_n_mem` to `m_memory_config->m_n_mem_sub_partition` (§4.2).
- Config fix (later session): `SM90_H100_l2norm_l1dnorm/gpgpusim.config`, `-gpgpu_cache:dl2` MSHR merge cap
  `A:192:4` → `A:192:8` (§4.6) — the fix that actually closed the ~2x gap.
- Related but separate fix (same later session, general correctness bug, not specific to this kernel):
  `notes/cache-invalidation-l1c-l1t-fix.md` — L1C/L1T caches were never invalidated at kernel boundaries.
