# Prefill k3 — real-H100 microbenchmark correlation, latency-wiring audit, and ncu validation plan

Fourth document in the k3 series. Builds on [`prefill-k3-correlation.md`](prefill-k3-correlation.md) (what k3 is),
[`prefill-k3-debug-investigation.md`](prefill-k3-debug-investigation.md) (two-layer stall model), and
[`prefill-k3-config-audit-and-occupancy.md`](prefill-k3-config-audit-and-occupancy.md) (dead-path audit + starvation
confirmation). This document covers: real H100 microbenchmark numbers vs configured latencies, a full code-level
audit of which config knob actually drives each op-type's latency in the remodeled SM, a source-level confirmation
(direct from `llama.cpp`) of the kernel's load/barrier/prefetch structure, and a concrete plan to validate the
simulator's stall-reason model against the real kernel via Nsight Compute.

| Item | Value |
|------|-------|
| Real H100 target | 102,621 ns → **~166,246 cycles** @ 1.620 GHz |
| Baseline sim k3 | 258,275 cycles (+55%) |
| Best config-only result | 237,536 cycles (+43%) — `SM90_H100_best_optA` |
| Dominant mechanism (prior doc) | 74.6% of subcore-cycles fully starved — occupancy-bound |
| Real H100 validation (§6, this doc) | Mechanism confirmed (51.4% real "No Eligible"), but model overstates severity ~1.45x |

---

## 1. Real H100 microbenchmarks vs configured latencies

Fresh runs on the BSC cluster (`util/tuner/GPU_Microbenchmark/ubench/{core/config_tensor_imma16832, l1_cache/l1_lat,
l2_cache/l2_lat, mem/mem_lat}`), H100 sm_90:

| Microbenchmark | Real measured | Current config | Verdict |
|---|---|---|---|
| L1 latency | **41.4365 cyc** | `gpgpu_l1_latency 41` | **excellent match** (<2% off) — no change needed |
| L2 hit latency | **283.7326 cyc** | `gpgpu_l2_rop_latency 242` | **under by ~15-17%** — fixable, see §4 |
| Full DRAM round trip | **494.99 cyc** | `gpgpu_l2_rop_latency`(242) + `dram_latency`(224) = 466 | within ~9% *before* interconnect transit is added — likely fine once L2 is corrected; don't tune `dram_latency` in isolation (risk of double-counting) |
| IMMA dependent-chain latency | **24.04 cyc** | model's computed `latency` component ≈ 8 cyc (at `tensor_rate_per_cycle=2048`) | see §1.1 — structural limitation, not a simple value fix |
| IMMA warp-issue bandwidth | **0.664 inst/clk/SM** (≈1.5 cyc/inst) | model's computed `initiation` component ≈ 8 cyc | same |

### 1.1 The IMMA result is a structural formula limitation, not a tunable value

Real H100 IMMA behaves like a deeply pipelined unit: **short initiation (~1.5 cyc)**, **long dependent latency (~24
cyc)** — ratio ≈16:1. The model's formula (`abstract_hardware_model.cc:427`,
`warp_inst_t::generate_tensor_core_latencies`) always splits a single computed value 50/50:

```cpp
cycles      = (size_m * size_n * size_k * operand_bit_size) / tensor_rate_per_cycle;
initiation  = cycles / 2;
latency     = cycles - initiation;
```

No choice of `tensor_rate_per_cycle` can hit both real numbers simultaneously — lowering it to match the 24-cycle
latency (`tensor_rate_per_cycle≈1311–1365` → `init=12, latency=12`) would make independent-IMMA throughput **8x
slower** than real hardware, which is the wrong trade for this kernel: the static trace shows 22 independent IMMA
accumulator chains (distinct destination registers) designed to exploit exactly that throughput. Recommendation:
leave `tensor_rate_per_cycle` near its current/`optA` value (favoring throughput) rather than chasing the 24-cycle
latency number; closing this gap properly needs the formula decoupled in code (separate init/latency knobs, or a
hardcoded ratio), not a config sweep.

---

## 2. Latency-wiring audit: which config knob actually drives each op-type, traced through code

Building on the dead-path audit in the prior doc (which established that `trace.config`'s opcode-latency block is
dead for the remodeled SM), this session traced the *opposite* question: for the knobs that **are** live, which
exact CLI flag feeds them? Resolved per op-type, from `subcore.cc`'s functional-unit constructors:

| op_type | `subcore.cc` member used | Source | Current value |
|---|---|---|---|
| `SP_OP` | `max_sp_latency` | `-ptx_opcode_latency_fp <ADD,MAX,MUL,MAD,DIV>` idx[1], via `shader_core_config::set_pipeline_latency()` (`shader.cc:3620`) | 4 cyc (config: `4,4,4,4,39` — idx[1]=4) |
| `INTP_OP` | `max_int_latency` | `max(int_latency[1], int_latency[5], predicate_latency)` from `-ptx_opcode_latency_int` + `-predicate_latency` (`shader.cc:3621-3622`) | 21 cyc (config: `4,4,4,4,21,21` → max(4,21,13)=21) |
| `SFU_OP` | `sfu_latency` (**direct** CLI member, gpu-sim.cc:972) | `-sfu_latency` — **not set** in `SM90_H100/gpgpusim.config`, falls back to registered default | **21 cyc (default)** — see §2.1, this is a bug |
| `TENSOR_CORE_OP` | `tensor_latency` (capacity) + dynamic per-instruction formula | `-tensor_latency` (32, capacity only) + `-tensor_rate_per_cycle` (2048, drives actual latency) | capacity 32, actual ~8 |
| `BRANCH_OP` | `branch_latency` (direct) | `-branch_latency` | 2 cyc |
| `UNIFORM_OP` | `uniform_latency` (direct) | `-uniform_latency` | 2 cyc |
| `DP_OP` | `max_dp_latency` | `dp_latency[1]` from `-ptx_opcode_latency_dp` (`shader.cc:3623`) | 8 cyc |

### 2.1 Found bug: `-sfu_latency` is silently unset, explicit SFU tuning isn't taking effect

There are **two different, confusingly-named CLI flags**:
- `-ptx_opcode_latency_sfu` (cuda-sim.cc:79) — parsed into a *local* variable inside
  `set_pipeline_latency()`, only used to compute the (separately dead-for-SFU) `max_sfu_latency` member, which
  `subcore.cc` never reads.
- `-sfu_latency` (gpu-sim.cc:972) — the **actual** member `subcore.cc`'s `m_sfu_pipeline` consumes
  (`subcore.cc:1716` region).

`SM90_H100/gpgpusim.config` explicitly sets `-ptx_opcode_latency_sfu 23` (clearly an intentional tuning value) but
**never sets `-sfu_latency`**, so the live path silently uses the registered default of **21**, not the intended 23.
Small, easy, real fix: add `-sfu_latency 23` (or whatever the right number turns out to be) explicitly to the config.

### 2.2 Confirmed dead-for-remodeled-SM: `max_sfu_latency`, `max_tensor_core_latency`

Both are computed in `set_pipeline_latency()` but only consumed by `shader.cc`'s classic (non-remodeled,
upstream-Accel-Sim) `pipelined_simd_unit` constructors (`shader.cc:2597`, `shader.cc:2605`) — a parallel legacy SM
model that coexists in this codebase but isn't what executes for `SM90_H100` runs. `max_dp_latency` is the exception:
it's consumed by **both** the classic path (`shader.cc:2712`) and the remodeled path (`subcore.cc:1737`, `sm.cc:791`),
so DP tuning via `-ptx_opcode_latency_dp` is genuinely live.

---

## 3. Static op-mix for k3 (`mul_mat_q<Q8_0, Li128>`, 6,352 static instructions)

Opcode → `op_type` classification resolved from `ISA_Def/hopper_opcode.h`, counts from
`enhanced_execution_info.json`:

| op_type | opcodes (count) | total | latency knob (§2) |
|---|---|---:|---|
| `SP_OP` | FFMA(1024), FMUL(1024), IMAD(290) | **2,338** | 4 cyc |
| `INTP_OP` | IADD3(363), LEA(303), ISETP(128), CS2R(86), LOP3(72), SHF(70), PRMT(66), SEL(28), MOV(147) | **1,263** | 21 cyc |
| `SFU_OP` | I2FP(1024), MUFU(20), F2I(20), I2F(20) | **1,084** | 21 cyc (should be 23, §2.1) |
| `LOAD_OP` | LDS(408), LDG(213), LDSM(32), LDC(21) | **674** | 7 (LDS) / 54 (LDG) / 11 or 0 (LDC) |
| `STORE_OP` | STS(147), STG(128) | **275** | — |
| `TENSOR_CORE_OP` | IMMA(256) | **256** | ~8 cyc (§1.1) |
| `BRANCH_OP` | BSSY(64), BSYNC(64), BRA(105) | **233** | 2 cyc (instruction's own pipe latency — **not** the barrier wait itself, which is data-dependent) |
| `UNIFORM_OP` | ULDC(87), UIMAD(36), UIADD3(15) | **138** | 2 cyc |
| `CALL_OPS` | CALL(14) | 14 | — |

**Correction (superseded by §7):** an earlier draft of this doc claimed `BSSY`/`BSYNC` (64 each) were the compiled
form of `mmq.cuh`'s 4 `__syncthreads()` per `kb0` iteration, "confirming" a 16x fully-unrolled K-loop. That is
**wrong** — `BSSY`/`BSYNC` are warp-level branch-reconvergence stack instructions, unrelated to `__syncthreads()`.
The actual barrier opcode is `BAR.SYNC`, which appears only **13** times total. See §7 for the corrected picture: the
K-loop is very likely a real runtime loop (back-edge branch) executed many times at runtime, not a static 16x
unroll, and the 6,352-instruction static listing instead reflects ~2 compiled passes of the tile-processing body
(e.g. checked/unchecked tile variants).

**Caveat:** count × latency gives a "naive serial" magnitude indicator only (e.g. SFU≈22.8k, INTP≈26.5k cycles) —
not a duration prediction. It ignores cross-warp and cross-iteration overlap entirely, which is the whole subject of
the starvation analysis in the prior doc.

---

## 4. L2 latency sensitivity (existing data, used to predict the real-calibrated fix)

Clean single-knob data point already on hand (`SM90_H100_l2rop300`, no other change):

| `gpgpu_l2_rop_latency` | k3 cycles | Δ |
|---|---|---|
| 242 (baseline) | 258,275 | — |
| 300 (+24%) | 259,633 | +1,358 (+0.53%) |

Sensitivity ≈ 23 k3-cycles per cycle of L2 latency added — shallow, because L2 latency is a small piece of a critical
path dominated by occupancy starvation, not memory exposure. Linear extrapolation to the real-measured 284 (+42 from
baseline): **predicted ~259,250 cycles (+0.4%)** — a small, well-understood, low-risk fix. Not yet run as an isolated
experiment; `SM90_H100_l2rop284` + matching script would be the next one-knob test if/when this gets prioritized.

---

## 5. Source-level confirmation: no shared-memory ping-pong, but register-level prefetch exists

Read `mul_mat_q_process_tile` directly from `llama.cpp/ggml/src/ggml-cuda/mmq.cuh:3407-3440`:

```cpp
for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);   // writes tile_x (fresh, single buffer)
    { tile_y[l] = by0[l]; }                                              // writes tile_y (fresh)
    __syncthreads();              // waits on THIS iteration's writes — no [0]/[1] double buffer exists
    vec_dot(tile_x, tile_y, sum, 0);
    __syncthreads();
    { tile_y[l] = by0[l]; }        // overwrites the SAME tile_y
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();              // protects the NEXT iteration's load_tiles from starting early
}
```

`tile_x`/`tile_y` are single shared buffers (`extern __shared__ int data_mul_mat_q[]`, sliced once). **Confirmed: the
barrier right after `load_tiles()` is waiting on that same loop body's own fresh writes, not a previous iteration's
tile sitting in a second buffer.** `vec_dot` itself (`mmq.cuh:862`, `vec_dot_q8_0_q8_1_mma`) only issues
`LDS`/`LDSM` (shared-memory reads) — no `LDG` — confirming all global traffic happens inside `load_tiles()`.

**But:** the SASS trace shows a run of ~10 consecutive `LDG.E.CONSTANT` instructions late in the current iteration's
dequant phase (after the `FFMA`/`FMUL` chain, before the closing barriers) — `ptxas` hoisting the **next** iteration's
global loads into registers while the current iteration's `vec_dot` is still running, even though the corresponding
shared-memory store must still wait for the barrier. This is real register-level pipelining inserted by the
compiler, distinct from (and not contradicting) the "no double-buffer" finding. Since NVBit records actual dynamic
execution order, this hoisting is already present in the trace, and the simulator's register scoreboard (release
based on actual destination register, not synthetic ordering) should credit the same overlap in principle. The open
question is whether the *prefetch distance* (cycles between the hoisted `LDG` and where it's consumed) is long
enough to fully hide the modeled L1D round-trip (~54 cyc, §1) — directly checkable against real hardware via the ncu
plan in §6.

---

## 6. Real-hardware validation result: mechanism confirmed, severity over-estimated by the model

Ran on the BSC cluster: `ncu --kernel-name regex:mul_mat_q --section Occupancy --section SchedulerStats
--section WarpStateStats --csv ./bin/llama-bench ...` against the real kernel. Raw output:
[`k3_occupancy_starvation_check.csv`](k3_occupancy_starvation_check.csv).

### Occupancy — exact match to the model's assumptions

| Metric | Real H100 |
|---|---|
| Active Warps Per SM (achieved) | **7.99 / 8** |
| Theoretical / Achieved Occupancy | 12.50% / 12.49% |
| Block Limit Registers | **1** |
| Block Limit Shared Mem | **1** (independently binding — we'd only flagged registers before) |
| Active Warps Per Scheduler | **2.00** (= 8 warps / 4 schedulers, exact match to §5 of the prior doc) |

### Scheduler Statistics — the headline number

| Metric | Real H100 | Simulator (90k debug window, prior doc) |
|---|---:|---:|
| **No Eligible** (= starved) | **51.41%** | **74.6%** |
| One or More Eligible | 48.59% | 25.4% |
| Eligible Warps Per Scheduler | 0.78 | not tracked as a continuous average |
| Issued Warp Per Scheduler | 0.49 | ≈0.26 (12.45M issued / (528 subcores × 90,050 cycles), back-of-envelope) |

**Conclusion: the starvation mechanism is real and dominant on actual hardware too — not a simulator artifact.**
More than half of all real cycles genuinely have zero eligible warps, and the structural premises (8 warps/SM, 2/
scheduler) match exactly. But **the simulator overstates the severity by roughly 1.45x** (74.6% vs 51.4%), and real
hardware issues per scheduler roughly **1.9x more often** than the model (0.49 vs ~0.26).

This narrows and quantifies the open question from §5: real hardware achieves nearly double the model's eligible-warp
coverage despite identical occupancy — most likely because the register-level prefetch hoisting found in the SASS
(the ~10 consecutive `LDG.E.CONSTANT` issued ahead of the barrier) is providing real latency-hiding on hardware that
the model's scoreboard/latency timing isn't fully crediting. Leading candidates for *why*, in order of how directly
they connect to existing findings:

1. The L1D/global-load latency feeding that hoisted prefetch (54 cyc modeled, §1) may not match how the prefetch
   distance actually plays out on real hardware.
2. The scoreboard release granularity may be coarser than real per-register hardware barriers — recall the trace's
   `control_bits` (stall_count, wait_barrier_bits) are all zero (prior doc), so the model has no real hardware
   scheduling hints to work from and falls back to a generic register-overlap check.
3. Cross-iteration overlap (next iteration's hoisted load running concurrently with current iteration's compute)
   might not be credited as effectively in the model as on real silicon.

**Bottom line revised from the prior doc:** this is no longer "accept ~75% starvation as correct and stop." It's
"the mechanism is confirmed correct, but real hardware achieves measurably better warp-level overlap (51% vs 75%
starved) than the model does — closing roughly that gap is the highest-value remaining target," and the prefetch-
hoisting/scoreboard-granularity hypotheses above are the concrete next things to check, likely requiring a code-level
look at L1D latency timing relative to issue order, not further config sweeps.

---

## 7. L1D latency vs. prefetch distance: checked and ruled out

Directly tested the §6 leading hypothesis ("the hoisted `LDG.E.CONSTANT` prefetch's latency may not be fully hidden
in the model"). Re-derived the real barrier structure first, since it changed the picture:

**`BAR.SYNC` positions (13 total, the real `__syncthreads()`):**
`[51, 443, 451, 819, 1666, 1748, 2730, 3668, 3671, 4098, 4955, 5025, 6008]` — grouped as
(1, 2, 1, 2, 1) then (2, 1, 2, 1), i.e. the same 5-barrier sub-pattern appears twice across the kernel. This is
consistent with `mul_mat_q_process_tile` being compiled twice (e.g. a checked/unchecked tile-boundary variant, or
the stream-K fixup pass) rather than one logical K-loop iteration repeated 16 static times. The K-loop is most
likely a real runtime back-edge branch, not a static unroll — `BSSY`/`BSYNC` (§3 correction) measure something
else (branch reconvergence), not loop trip count.

**Load → first-use distance, measured directly from the trace** (destination register of every `LDG*`, scanned
forward for first consuming instruction):

| Load class | n | min dist (static instrs) | median | max |
|---|---:|---:|---:|---:|
| `LDG.E.CONSTANT` — bulk tile data | 72/77 (94%) | 91 | 863 | 876 |
| `LDG.E.CONSTANT` — address/bounds setup (e.g. idx 438→440, 448→449) | 5/77 (6%) | 1 | 2 | 3 |
| `LDG.E.U16.CONSTANT` — dequant scale factors | 136/136 | 33 | 95 | 118 |

The 5 short-distance loads are one-time index/bounds-check computations in setup code (confirmed by reading the
surrounding instructions — `IADD3`/`ISETP`/`LEA.HI` address arithmetic, not inside the hot K-loop body), so they
cannot explain a systematic per-iteration gap.

**Conclusion: L1D latency is not the bottleneck.** Even taking the most conservative possible issue pace (1
instruction/cycle/warp), 91–876 static instructions of hoisting dwarfs both the modeled L1D round-trip (~54 cyc,
§1) and the real measured L1D latency (~41 cyc, §1). The scale-factor loads (33–118 instrs) clear the same bar with
real-hardware issue pace factored in (ncu: 4.11 cycles/issued-instruction → 135–485 cycles of margin). This rules
out hypothesis (1) from §6.

**Barrier-mechanics hypothesis also checked and ruled out.** Read `barrier_set_t::warp_reaches_barrier`
(`shader.cc:3871`) and the issue-gate in `subcore.cc:223-244`: `BAR.SYNC` is modeled with standard, architecturally
correct all-or-nothing release (a warp that issues it cannot issue anything else until every active warp in the CTA
arrives) — this matches real hardware by definition, not a place a "mismatch" could hide. There is a relaxation
knob, `-is_relax_barriers_baseline`, that lets non-memory ops issue past a barrier wait; per
[`prefill-k3-config-audit-and-occupancy.md`](prefill-k3-config-audit-and-occupancy.md) §5 it was already tested and
is a **proven no-op** — when all 8 warps are genuinely at the barrier together there is no other ready work,
memory or compute, so relaxing which op types it blocks changes nothing.

**Where this leaves things:** both of the two most direct mechanical explanations (load-latency exposure, barrier
strictness) are now ruled out. The gap must instead live in the *aggregate length of the compute segments between
barriers* — i.e. whether the model's per-instruction scoreboard-wait latencies on the dominant dependency chains
(`SP_OP`/`SFU_OP`/`INTP_OP` — the I2FP→FMUL→FFMA dequant chain, 76-88% starved-when-stalled per the prior doc's
table) collectively make each inter-barrier segment longer in the model than in reality, which would inflate the
*fraction* of total runtime spent barrier-locked even without any difference in barrier mechanics. This points at a
critical-path/ILP question (how much of that dependency chain real hardware overlaps within or across the 8
warps that the model isn't crediting), not a further latency-value or barrier-relaxation experiment.

---

## 7.1 Refined hypotheses for the ~1.45x starvation-severity gap (§6-7)

Two concrete candidate mechanisms for why the model shows 74.6% "no eligible warp" vs real hardware's 51.4%, at
*identical* occupancy (7.99/8 warps, exact match):

1. **Scoreboard release granularity is coarser than real per-register hardware completion.** The trace's
   `control_bits`/`wait_barrier_bits` are all zero (no real scheduling hints survive in the trace), so the model
   falls back to a generic conservative register-dependency check rather than crediting the same fine-grained
   overlap real hardware's scoreboard achieves.
2. **Cross-warp/cross-iteration overlap of the dominant dependency chain is under-credited.** Real hardware's ~2x
   better "eligible warp" coverage at identical occupancy most plausibly comes from the model not crediting
   register-level ILP across warps/iterations as aggressively as real silicon does.

### 7.2 Investigating hypothesis 1 — found a concrete code-level mechanism

`Scoreboard::releaseRegisters_remodeling`/`releaseRegisters` (`scoreboard.cc:178-214`) themselves track at full
per-register granularity (not coarsened) and are only called from `SM::instruction_retirement`
(`sm.cc:303`) — i.e. a register is held reserved from issue until its producing instruction *retires*. The
coarseness isn't in what's tracked; it's in **when retirement is allowed to happen**.

Traced `functional_unit::instruction_finishing_execution` (`functional_unit.cc:218-255`): once an instruction
with destination registers finishes its functional-unit pipeline latency, it does **not** retire directly — it
must first get a slot in `m_rf_write_queue` (`functional_unit.cc:226-242`), and retirement (the `instruction_retirement`
call that releases the scoreboard) only happens when that queue entry is later popped.

**The queue is shared across functional-unit types, and the pop rate is tiny.** From `Subcore::create_pipeline`
(`subcore.cc:1711-1758`), the *same* `m_regular_fixed_latency_rf_write_queue` (capacity
`-max_size_register_file_write_queue_for_fixed_latency_instructions` = **8**, per subcore) is the write-queue for
**`m_int_pipeline` (INTP_OP), `m_sp_pipeline` (SP_OP), `m_tensor_pipeline` (TENSOR_CORE_OP), `m_branch_pipeline`
(BRANCH_OP), and `m_miscellaneous_no_queue_pipeline`** — five different op types funnel into one shared queue. It's
drained by `Subcore::writeback_process_fixed_latency_write_queue` (`subcore.cc:746-757`) at
`-max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions` = **1** entry per cycle, per subcore.
(`SFU_OP` uses a separate queue, `m_EX_WB_sm_variable_latency_latch` — not part of this bottleneck.)

For k3's dequant-heavy op mix (`SP_OP`: FFMA+FMUL+IMAD = 2,338 static instrs; `INTP_OP` = 1,263; `TENSOR_CORE_OP` =
256; `BRANCH_OP` = 233 — all sharing this one port, per subcore, 2 resident warps/subcore), this is a real,
structural, single-wide retirement bottleneck that real hardware does not have (independent per-pipe writeback
resources, not one shared 8-deep/1-wide funnel across SP+INT+TC+branch). This is the most concrete lead yet for
hypothesis 1 and the next thing to test empirically: widen
`max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions` (1→N) and see how much of the
74.6%→51.4% starvation gap it closes.

---

## 8. Open items carried forward

1. **Investigate why the model under-credits warp-level overlap relative to real hardware** (§6) — **both L1D
   latency-vs-prefetch-distance and barrier-mechanics/relaxation have now been checked and ruled out (§7)**. Next:
   measure whether the modeled inter-barrier compute-segment length (critical path through the dominant `SP_OP`/
   `SFU_OP`/`INTP_OP` dequant dependency chain) is longer than real hardware's, and/or whether real hardware overlaps
   that chain across warps more than the model credits — a critical-path/ILP question, not a config sweep.
2. Fix the `-sfu_latency` / `-ptx_opcode_latency_sfu` mismatch (§2.1) — cheap, well-understood, one-line config fix.
3. Run the isolated `SM90_H100_l2rop284` experiment (§4) to confirm the ~+0.4% prediction.
4. Tensor latency/initiation decoupling (§1.1) remains a code-level change, not yet started.
5. Revised from the prior doc: occupancy starvation is confirmed real and dominant on actual hardware (not a
   simulator artifact), but the *severity* the model assigns it (74.6%) measurably exceeds reality (51.4%) — so this
   is not the practical stopping point the prior doc suggested. There is a real, quantified, ~1.45x gap left to
   chase, and item 1/§7 above is where to chase it next.
6. **(New, §9 item 11)** Write-queue/port-contention hypothesis (item 9) is now refuted two independent ways — sim
   (`SM90_H100_rfwbport4`, −0.2%) and real hardware (`math_pipe_throttle`=5.07%, small). Hypothesis 2 (dependency-
   chain/cross-warp-overlap) is now the better-supported direction: real hardware's `wait`(16.19%)+
   `long_scoreboard`(15.13%) are the dominant genuine stall categories, not `barrier`(2.57%, surprisingly small).
7. **Unreconciled: why is real `barrier`=2.57% so small** when §6's `SchedulerStats` "No Eligible"=51.4% was
   attributed mostly to `bar.sync` lockstep? Need to check whether `No Eligible` (joint, across all resident warps)
   and per-warp-average `barrier` attribution are even comparable statistics, or whether `ncu` simply doesn't
   sample "parked at barrier" time the same way our model's `cta_barrier` stall-reason bucket counts it. This
   matters because it changes how much weight the "8 warps + lockstep" framing should get vs. the dequant-chain
   dependency-latency framing.

---

## 9. Session hypothesis log (this investigation pass)

Consolidated record of every hypothesis tested in this pass, on top of the prior docs' P1-P10 / §1-§8 findings.
Status as of this writing: items 1-9 concluded, item 10 still running.

| # | Hypothesis | Test | Result | Conclusion |
|---|---|---|---|---|
| 1 | `BAR.SYNC` release fires before the slowest warp actually arrives (an off-by-something in barrier timing) | Added `bar_arrival_trace`/`bar_release_trace` printf in `sm.cc`/`shader.cc` (`-subcore_issue_debug 1`), SM0/CTA0, all 8 warps | `release_cycle == max_arrival_cycle` in every observed round, exactly | **Ruled out.** Reconfirms §7's code-level read of `barrier_set_t::warp_reaches_barrier` — barrier mechanics are correct by construction |
| 2 | Greedy-then-highest-id subcore issue priority (vs. true round-robin) is suboptimal | Added `-is_subcore_round_robin_issue_scheduler 1` (`subcore.cc`, `gpu-sim.cc`, `shader.h`) | 275,676 cycles vs. 268,513 baseline — **worse** | **Ruled out**, and informative: confirms nothing else is ready to issue regardless of priority order — consistent with the starvation finding (§4 of prior doc) |
| 3 | `int_latency` SHFL=21 (only relevant to a handful of opcodes k3 doesn't use) inflates `max_int_latency` for every `INTP_OP` via `max()` | `-ptx_opcode_latency_int 4,4,4,4,21,4` (SHFL 21→4) | **Byte-identical** 268,513 cycles | **Ruled out** — `predicate_latency`(13) was already the binding term in `max(int_latency[1]=4, int_latency[5], predicate_latency=13)`, not SHFL. Edit kept in config (harmless/correct) but zero performance effect |
| 4 | Subcore-level L0 constant cache (`-perfect_constant_cache`, distinct from the already-perfect SM-level L1C) gates issue and is a hidden bottleneck | `-perfect_constant_cache 1` (was 0) | Byte-identical 268,513 cycles | **Ruled out** — consistent with `nopc_constmshr` (prior doc §1.1): the *SM-level* L1C bypass (`gpgpu_perfect_inst_const_cache 1`, already on) is what matters; the subcore L0C gate isn't binding for k3 |
| 5 | `-gpgpu_perfect_mem` (swap real interconnect for instant response) would show the memory-latency ceiling | `-gpgpu_perfect_mem 1`, k3-only | **Crash**: `gpu-cache.cc:1209` assert in `baseline_cache::fill`, called from `SM::accept_fetch_response` | **Inconclusive — found a real bug**, not a calibration result. `perfect_memory_interface::push()` shoves the same `mf` straight into the response FIFO, bypassing the sectored-cache round trip that populates `m_extra_mf_fields` for `m_L1I_L1_half_C_cache` (SECTOR_ASSOC). Pre-existing incompatibility between `-gpgpu_perfect_mem` and the remodeled SM's sectored L0/L1I/L1C hierarchy, unrelated to anything changed this session. Not fixed (would need either `perfect_memory_interface` to special-case sectored fills, or bypass the L0/L1I/L1C path entirely under perfect_mem) |
| 6 | L1D/L1C MSHR or DRAM-scheduler queue depth is undersized, causing extra queueing latency | Checked DRAM scheduler queue occupancy (`mrqq: max=64 avg=3.42858`, ~63 separate L2 partitions, all similar) in a baseline run's memory-partition dump | Average occupancy ~3.4 of 64 — nowhere near saturated | **Ruled out**, and consistent with the prior doc's three independent MSHR-widening tests (`mq32`/`mq64`/`mshr1024`, `nopc_constmshr`) all showing <1% effect. Queue *capacity* is not the bottleneck anywhere checked |
| 7 | The model's L2 hit/miss latency constants are simply miscalibrated against real H100 | Compared simulated `cache_stats::print_stats()` output (`avg_hit_latency=245.24 cyc` n=3.71M, `avg_miss_latency=658.83 cyc` n=679,984) against real H100 microbenchmarks (L2 hit 273-284 cyc, full DRAM RTT 478-495 cyc, both measured twice this session, consistent) | **Asymmetric result**: simulated hits are *faster* than real (245 < 273-284); simulated misses are *slower* than real (659 > 478-495) | **Important, not a simple fix.** Raising `gpgpu_l2_rop_latency` to match real hit latency would also slow down every miss (rop_latency is paid by both), pushing the already-too-slow total further in the wrong direction. The hit-latency gap and the miss-latency gap need different, independent levers — confirmed analytically (unit-converting `dram_latency`/`rop_latency` across the two configs' different DRAM clocks) that the *configured* static latency sum is already close to real; the ~189-cycle miss-latency excess is an *observed/contended* effect, not a misconfigured constant. Still unresolved which mechanism produces it (ruled out: queue capacity, item 6) |
| 8 | `SM90_H100_best`'s broader memory/topology retuning (n_mem 64→80, dual-bus HBM3, JESD timings, 50MB L2) recovers most of the gap | Ran `SM90_H100_best`, k3-isolated (clean single-kernel run, not the cumulative 3-kernel subtraction the config's own doc comment used) | 262,791 cycles vs. 268,513 baseline — **only −2.1%**; `gpu_occupancy` unchanged at 12.48%, `gpu_ipc` unchanged (~4453 vs ~4531) | **Confirms prior doc's verdict**: memory/topology retuning alone has little headroom left; occupancy/starvation regime is untouched by it. (Also: `SM90_H100_best`'s own header comment claims k3 is "-46% too fast" — **explicitly set aside per user instruction, different baseline/methodology than this investigation's 166,246-cycle target, not reconciled**) |
| 9 | **(Current best lead)** Scoreboard release is gated behind an artificial single-wide cross-op-type retirement port, not real per-pipe hardware writeback — concrete mechanism for refined hypothesis 1 (§7.1) | Code trace: `Scoreboard::releaseRegisters*` (`scoreboard.cc`) only called from `SM::instruction_retirement`, which for instructions with destination registers is only reached after a slot in `functional_unit`'s `m_rf_write_queue` is popped (`functional_unit.cc:218-255`). `Subcore::create_pipeline` (`subcore.cc:1711-1758`) shows **`SP_OP`, `INTP_OP`, `TENSOR_CORE_OP`, `BRANCH_OP`, `MISC_NO_QUEUE_OP` all share one queue** (`m_regular_fixed_latency_rf_write_queue`, capacity 8), drained at `-max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions` = **1 entry/cycle/subcore** (`subcore.cc:746-757`) | Code-level finding, not yet a measured cycle-count result | **Most concrete lead so far** for why the model's inter-barrier compute segments might run longer than real hardware's: k3's dequant chain is exactly `SP_OP`(2,338 static)+`INTP_OP`(1,263)+`TENSOR_CORE_OP`(256)+`BRANCH_OP`(233) — all funneled through this one shared 1-wide-per-cycle port, per subcore (2 resident warps/subcore). Real hardware has independent per-pipe writeback resources, not one shared funnel across these op types |
| 10 | Widening the shared write-queue drain rate (item 9) closes some of the starvation-severity gap | `-max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions 1→4` (`SM90_H100_rfwbport4`), k3-isolated, full run (took 47m34s wall-clock, notably slower than baseline's ~20min for unclear reasons) | **267,986 cycles vs. 268,513 baseline — only −0.2%** (`gpu_ipc` 4367 vs ~4531 baseline, `gpu_occupancy` unchanged 12.48%) | **Ruled out**, in this tested form. The shared write-queue's *width* (1 vs 4 pops/cycle) is not the binding constraint — same pattern as MSHR width and queue depth (items in prior docs): capacity/bandwidth knobs keep not mattering for k3. This means the bottleneck is more likely *latency*-shaped than *throughput*-shaped, which favors refined hypothesis 2 (cross-warp/cross-iteration overlap under-credited) over hypothesis 1's "retirement port too narrow" framing — though the underlying code-level finding (item 9, the shared queue existing at all) may still matter at a different pop-rate or under different contention patterns; not exhaustively ruled out, just this one data point |

| 11 | Real hardware's better warp-level overlap (§6) is visible in `ncu`'s per-reason stall breakdown, not just the aggregate "No Eligible" number | `ncu --target-processes all --kernel-name regex:"mul_mat_q<" --launch-count 1 --metrics "regex:smsp__warp_issue_stalled_.*_per_warp_active.pct" --csv ...` against the real kernel, launch index 0 (confirmed via a separate `gpu__time_duration.sum` pull: launch 0 = 107,936 ns, in the same ~107-110k ns cluster — 144 launches — closest to our 102,621 ns real target; ~4.5% gap is expected `ncu` instrumentation overhead) | **Done.** Full breakdown (% of warp-cycles): `selected`(issued)=24.31, **`wait`=16.19**, **`long_scoreboard`=15.13**, `not_selected`=14.59, `dispatch_stall`=8.15, `math_pipe_throttle`=5.07, `mio_throttle`=4.78, `lg_throttle`=4.45, `short_scoreboard`=3.15, **`barrier`=2.57**, `no_instruction`=0.74, `imc_miss`=0.58, `branch_resolving`=0.46, `drain`/`misc`=0.16 each, `gmma`/`membar`/`sleeping`/`tex_throttle`=0 | **Three findings.** (a) `math_pipe_throttle`=5.07% is small — **independently confirms item 10's refutation** of the write-queue/port-contention hypothesis; real hardware agrees with the simulator that this isn't the dominant mechanism. (b) `wait`(16.19%)+`long_scoreboard`(15.13%) are by far the largest genuine dependency-stall categories — real evidence *for* hypothesis 2 (dependency-chain/cross-warp-overlap framing) over a capacity/throttle story. (c) `barrier`=2.57% is surprisingly small given §6's `SchedulerStats` "No Eligible"=51.4% being attributed mostly to `bar.sync` lockstep — **unreconciled tension, not yet resolved**: either `No Eligible` (a joint statistic across all resident warps) and this per-warp average `barrier` attribution aren't directly comparable denominators, or `ncu`'s per-cycle sampling doesn't attribute time to `barrier` the same way once a warp has already issued `BAR.SYNC` and is parked. `gmma`=0 confirms no `wgmma` usage (consistent with §5/§9) |

**New debug tooling added this session** (all uncommitted C++, all in `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`/`sm.h`):
- `[bar_arrival_trace]` / `[bar_release_trace]` (`-subcore_issue_debug 1`) — per-warp barrier arrival/release cycles, SM0/CTA0.
- `[issue_trace]` / `[commit_trace]` (`-subcore_issue_debug 1`) — per-instruction issue and retirement cycles, SM0/warp0 only, includes PC and `op_type_to_string()` name.
- `[commit_progress]` — **always on, no flag needed** — SM0-only heartbeat, prints cumulative committed-instruction count and current cycle every 100,000 instructions retired. Useful for watching long runs progress without waiting for completion.

**New experimental configs** (`gpu-simulator/gpgpu-sim/configs/tested-cfgs/`): `SM90_H100_rfwbport4` (item 10, write-queue pop rate 1→4, only diff from baseline `SM90_H100`).

---

### 9.1 File index (additions to prior docs, covering §1-9)

| Path | Contents |
|------|----------|
| `notes/prefill-k3-real-hw-correlation.md` | **This document** |
| `util/tuner/GPU_Microbenchmark/ubench/l1_cache/l1_lat/`, `l2_cache/l2_lat/`, `mem/mem_lat/` | Microbenchmark sources used for §1 (run on BSC cluster, results not yet saved as files in-repo — recorded here instead) |
| `gpu-simulator/ISA_Def/hopper_opcode.h` | Opcode → op_type classification table, used for §3 |
| `llama.cpp/ggml/src/ggml-cuda/mmq.cuh:3407-3440` | Source-level confirmation of §5 (external repo, not part of this one) |
| `notes/k3_occupancy_starvation_check.csv` | Real H100 `ncu` Occupancy/SchedulerStats/WarpStateStats raw output, §6 |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`, `sm.h` | §9 items 1, 9-10: `bar_arrival_trace`/`bar_release_trace`/`issue_trace`/`commit_trace`/`commit_progress` printf instrumentation (uncommitted) |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc`, `subcore.h`, `shader.h`, `gpu-sim.cc` | §9 item 2: `-is_subcore_round_robin_issue_scheduler` (uncommitted) |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_rfwbport4/` | §9 item 10: write-queue pop-rate 1→4 experiment config |
| `log/tmp_log/sched_baseline_gto.log`, `sched_round_robin.log` | §9 item 2 raw logs |
| `log/tmp_log/perfect_constcache.log`, `perfect_mem.log`, `intlat_fix.log` | §9 items 3-5 raw logs |
| `log/tmp_log/best_k3_isolated.log`, `rfwbport4_k3_isolated.log` | §9 items 8, 10 raw logs (clean k3-isolated runs, not cumulative-subtraction estimates) |
| `log/tmp_log/commit_trace_5000.log`, `commit_trace_5000_pops4.log` | §9 item 10: 5,000-cycle single-warp `issue_trace`/`commit_trace` comparison (inconclusive by design, see item 10's conclusion) |
| `k3_stall_reasons.csv`, `k3_duration.csv` (repo root, not yet moved into `notes/`) | §9 item 11: real `ncu` per-reason stall breakdown (launch 0) and the duration sweep used to identify it |

---

*Last updated: 2026-06-19 — added §9 session hypothesis log: barrier-mechanics re-confirmation, round-robin scheduler
(worse), int-latency SHFL fix (no-op), subcore constant-cache perfect toggle (no-op), `-gpgpu_perfect_mem` crash
(sectored-cache incompatibility, unfixed), DRAM-scheduler queue occupancy check (not saturated), asymmetric L2
hit/miss latency vs. real microbenchmarks (raising `rop_latency` would worsen the miss-latency direction), a clean
k3-isolated `SM90_H100_best` run (−2.1% only, occupancy/IPC unchanged), the shared register-file write-queue
mechanism for refined hypothesis 1 (item 9) and its empirical test (item 10, concluded: pop-rate 1→4 gives
267,986 vs. 268,513 baseline, only −0.2% — **ruled out**, reinforcing that capacity/bandwidth knobs don't move k3
and the bottleneck is more likely latency-shaped, favoring hypothesis 2), and the completed `ncu` per-reason
stall-breakdown pull (item 11, launch index 0 / 107,936 ns, duration-matched against the 102,621 ns real target):
`math_pipe_throttle`=5.07% independently confirms item 10's refutation on real hardware; `wait`=16.19% and
`long_scoreboard`=15.13% are the dominant genuine dependency-stall categories, supporting hypothesis 2 over a
capacity/throttle story; `barrier`=2.57% is surprisingly small given §6's 51.4% "No Eligible" finding — flagged as
an unreconciled open question (§8 item 7) rather than resolved.*

---

## 10. Bundled depfix experiment (fixes 1–3 combined) — 2026-06-19

Implemented three structural/config changes together in `SM90_H100_k3_depfix` (parent: `SM90_H100_best_optA`):

| Fix | Knob / code | Intent |
|-----|-------------|--------|
| 1 | `-is_scoreboard_release_at_ex 1` + `maybe_release_scoreboard_at_ex()` in `functional_unit.cc` | Release write scoreboard at FU completion (operand-forwarding), not RF writeback |
| 2 | `-sfu_latency 8`, `-ptx_opcode_latency_int 4,4,4,4,21,4` | Align SFU pipe + `max_int_latency=13` with trace.config / ubench |
| 3 | `-tensor_initiation_cycles_override 2`, `-tensor_dependent_latency_override 24` | Decouple IMMA issue vs dependent latency (H100 ubench §1.1) |

Also added: `[commit_progress]` every 1k SM0 retirements (`fflush`), `[ex_release_trace]` under `-subcore_issue_debug 1`, k3 compare scripts.

### 10.1 Results (k3-isolated, `filter_first_kernel_id 3`, parallel run)

| Config | `gpu_tot_sim_cycle` | vs real H100 (166,246) | `gpu_ipc` | `gpu_tot_sim_insn` |
|--------|---------------------|------------------------|-----------|-------------------|
| Real H100 target | **166,246** | — | — | — |
| **baseline** (`SM90_H100_best_optA`) | **265,360** | **+59.6%** | 4411 | 1,170,378,752 |
| **depfix** (fixes 1–3 bundled) | **267,398** | **+60.8%** | 4377 | 1,170,378,752 |

**depfix vs baseline: +2,038 cycles (+0.77% slower)** — no improvement; identical instruction count.

Logs: `log/prefill_k3_compare/baseline_k3.log`, `depfix_k3.log`.

### 10.2 Conclusion

Bundled dependency-chain fixes **do not close the gap** and slightly **hurt** vs `best_optA`. Hypothesis 2 (dependency stalls) remains the best *diagnostic* lead (ncu `wait`+`long_scoreboard` ≈ 31% on real HW), but these three specific levers are ruled out as a bundle.

**Next:** unbundle fixes 1/2/3 individually (`SM90_H100_k3_scoreboard_ex`, `_k3_sfu_int_lat`, `_k3_tensor_decouple`) to see if one helps and another hurts — continued in §11.

---

## 11. Unbundling fixes 1–3, and a baseline-bookkeeping note

### 11.0 Important: `SM90_H100/gpgpusim.config` lost this session's earlier calibration edits

While committing the §10 work, a `git reset` (visible in `git reflog`: `HEAD@{4}: reset: moving to HEAD`) discarded
uncommitted working-tree changes, including §9's edits to the bare `SM90_H100/gpgpusim.config` (`-gpgpu_n_mem`
64→32 reverted, `-gpgpu_clock_domains` 1620:1620:1620:1593→1830:1830:1830:10000 reverted, the SHFL `int_latency`
fix reverted to a 5-field array, `-gpgpu_shader_core_pipeline` 2048:32→1536:32 reverted). Those edits were never
committed, so they're gone from the working tree; **§9's 268,513-cycle baseline no longer exists on disk** and
isn't reproducible against the current `SM90_H100`.

`SM90_H100_best_optA` (a separately, earlier-committed file) is untouched and intact, and is what §10's
`depfix`/`baseline` comparison and all of §11 use. **Decision (user-confirmed): adopt `SM90_H100_best_optA`
(265,360 cycles, +59.6%) as the reference baseline going forward**, rather than recreating §9's bare-`SM90_H100`
numbers. §9's table remains valid as a historical record of what was tested and ruled out, just not numerically
comparable to §10/§11's cycle counts.

### 11.1 Why unbundle instead of trusting the bundled result

The bundled `depfix` result (+0.77% vs `best_optA`) only shows the *net* effect of fixes 1+2+3 together — it
can't distinguish "all three are mildly bad" from "one fix helps but a different one hurts enough to mask it"
from "all three are neutral and +0.77% is noise." Each conclusion implies a different next step, so the three
configs were run individually, same harness as §10 (`./bin/release/accel-sim.out`, `SM90_H100/trace.config`,
`-filter_first_kernel_id 3 -filter_last_kernel_id 3`, same trace; only difference: `OMP_NUM_THREADS=4` instead of
16, run three-way in parallel — a wall-clock-only difference, not a methodology difference):

| Config | Isolates | `gpu_tot_sim_cycle` | vs `best_optA` baseline (265,360) |
|--------|----------|---------------------|-----------------------------------|
| `SM90_H100_k3_scoreboard_ex` | Fix 1 only (`-is_scoreboard_release_at_ex 1`) | killed mid-run, not yet re-run | — |
| `SM90_H100_k3_sfu_int_lat` | Fix 2 only (`-sfu_latency 8` + SHFL `int_latency` fix) | killed mid-run, not yet re-run | — |
| `SM90_H100_k3_tensor_decouple` | Fix 3 only (`-tensor_initiation_cycles_override 2`, `-tensor_dependent_latency_override 24`) | **265,078** | **−282 cycles (−0.11%)** |

All three runs were originally launched 3-way in parallel and two (`scoreboard_ex`, `sfu_int_lat`) were killed
mid-run on 2026-06-20 due to memory thrashing (system down to 177Mi free RAM, 14Gi/19.5Gi swap used, CPU on the
survivors dropped to ~3%). `tensor_decouple` was re-run alone afterward and completed cleanly.

**`tensor_decouple` (Fix 3, IMMA init/latency decoupling) alone is a no-op** — −0.11% is noise, consistent with
the bundled depfix result (§10, +0.77%) and the write-queue width experiment (§9 item 10, −0.2%): none of these
dependency-chain/capacity levers move k3's cold-start cycle count in any meaningful way. `scoreboard_ex` and
`sfu_int_lat` still need to be re-run (one at a time, to avoid repeating the memory-thrashing incident) — but
given §13's finding that cold-start k3 may not even be the right methodology to test fixes against, it's worth
re-running all three (and a fresh baseline) in the warm multi-kernel context first, rather than continuing to
spend wall-clock time perfecting cold-start-only conclusions.

---

## 12. Interconnect response-path congestion — a new, better-supported lead than DRAM bandwidth

### 12.1 Motivation

With dependency-chain levers (§10–11) and occupancy/barrier lockstep (prior doc, §4–5) both exhausted, the
real H100 `mem_bw`/`mem_config`/`mem_lat` microbenchmark numbers pasted earlier this session (never previously
applied — see git history of this doc) were revisited as the next candidate: k3 launches exactly 132 CTAs, one
per SM, all hammering L2/DRAM simultaneously for the initial quantized-weight load — a full-chip-saturation
memory event, distinct from the per-access latency question already covered in §1.

Real H100 (`mem_bw`, `mem_config`, `mem_lat` microbenchmarks, BSC cluster):

```
Mem BW (sustained)        = 950.43 GB/sec   (Max Theoretical = 1631.23 GB/sec, efficiency 54.08%)
Memory channels           = 32   (Bus Width 4096 bit, HBM, clock 1593 MHz)
Mem latency (full RTT)    = 477.63 cycles
L2 Hit Latency            = 273.32 cycles
// suggested Accel-Sim config: -gpgpu_n_mem 32 -gpgpu_dram_buswidth 16 -dram_latency 204
```

### 12.2 First check: is DRAM bandwidth actually saturated? No.

`grep -E "L2_BW|DRAM_BW" log/prefill_k3_compare/{baseline,depfix}_k3.log`:

| Metric | baseline_k3.log | depfix_k3.log |
|---|---|---|
| `L2_BW_total` | 1014.06 GB/Sec | 1005.27 GB/Sec |
| `DRAM_BW_total` | **246.46 GB/Sec** | **244.59 GB/Sec** |

`DRAM_BW_total` (~246 GB/s) is only ~26% of real sustained bandwidth (950 GB/s) and ~15% of real theoretical peak
(1631 GB/s) — **DRAM channel bandwidth has enormous headroom in the model already.** `best_optA`'s
`-gpgpu_n_mem 80` (vs real 32 channels) and `-gpgpu_dram_buswidth 8` (vs real-implied 16) differ from the real
numbers, but **widening them would not help**, because the model isn't bandwidth-bound at the DRAM-channel level
for this kernel — `L2_BW_total` (1014 GB/s) being 4x `DRAM_BW_total` shows most traffic is already served from
L2, not DRAM. This refutes the bandwidth-saturation framing of the hypothesis proposed earlier this session.

### 12.3 Second check: where does the simulated miss latency actually come from — and is it really off vs real?

§9 item 7 (prior session) flagged an asymmetry using the *old, now-abandoned* `SM90_H100` baseline numbers
(`avg_hit_latency`=245, `avg_miss_latency`=659 vs real's hit 273-284 / RTT 478-495). **Re-checked against the
correct current reference (`best_optA`, the config this whole §10-§12 thread actually uses), the picture is
different and the gap is much smaller than previously documented:**

`grep -n "L2_cache_stats_breakdown_avg" log/prefill_k3_compare/{baseline,depfix}_k3.log`:

| Metric | `best_optA` (baseline_k3.log) | Real H100 | Gap |
|---|---|---|---|
| L2 hit latency | **223.22 cyc** (n=3.87M) | 273.32 cyc | sim **18% faster** |
| L2 miss latency | **511.46 cyc** (n=630,800) | 477.63 cyc (full RTT) | sim **7% slower** |
| L2 access (blended) | 263.59 cyc | — | — |

The miss-latency gap under `best_optA` is only **+33.8 cycles (+7%)** — not the dramatic +181-cycle (+38%) gap the
old `SM90_H100` numbers implied. **§9 item 7's "important, not a simple fix" framing was specific to the
already-abandoned baseline; under the current reference config, L2/DRAM latency is already fairly well
calibrated.** This tempers the rest of this section: the interconnect signal below is real and measurable, but
it is not "the missing ~180 cycles" — there isn't a gap that large left to explain on the memory-latency side.

`grep -niE "icnt" log/prefill_k3_compare/baseline_k3.log`:

```
max_icnt2mem_latency  = 372     avg_icnt2mem_latency = 6     (SM → memory, request direction)
max_icnt2sh_latency   = 1472    avg_icnt2sh_latency  = 180   (memory → SM, response direction)
avg_mrq_latency       = 66      (DRAM scheduler request-queue wait)
icnt_total_pkts_simt_to_mem = 1,678,622
icnt_total_pkts_mem_to_simt = 5,190,776   (≈3.1x more packets in the response direction)
```

Configured static latency alone (`l2_rop_latency`+`dram_latency` = 220+170 = 390 cyc) is **121 cycles below** the
*observed* 511.46-cycle average — so there is real dynamic/contention overhead on top of the static config, and
the response-path interconnect (`avg_icnt2sh_latency`=180, far more congested than the request path's
`avg_icnt2mem_latency`=6, with a long tail to `max_icnt2sh_latency`=1472) plus `avg_mrq_latency`=66 queueing are
the visible candidates for it — they don't sum cleanly to 121 (186+66=252 > 121), implying only a fraction of
misses see the full congested path, not all of them. The ≈3.1x response/request packet-count asymmetry is
consistent with large data responses fragmenting into multiple `-icnt_flit_size 40`-byte flits while small read
requests fit in one packet.

**Net assessment: this is a real, measurable contention signal, but a minor one** (~7% on the memory-latency
component, which itself is a smaller contributor to the overall +59.6% cycle gap than the occupancy/barrier
mechanism in §4-5 of the prior doc). Worth a quick, low-cost confirmation experiment (§12.5) since the data and
config knobs are already in hand, but it should not be expected to close more than a small slice of the gap.

### 12.4 Loose end found in passing: `-inter_config_file mesh` does not point to a real file

`best_optA` sets `-network_mode 2` (intersim2) with `-inter_config_file mesh`, but no file literally named `mesh`
exists anywhere in the repo (checked `gpu-simulator/`, repo root, build dirs). The simulation completes without
erroring, so intersim2 is presumably falling back to a built-in default topology/buffer configuration rather than
respecting whatever per-VC buffer/topology tuning the `mesh` filename was meant to supply. **Not yet root-caused or
confirmed as load-bearing** — worth checking only if the buffer-limit experiment in §12.5 doesn't move the needle,
since it's a config-correctness question, not necessarily the cause of the congestion itself.

### 12.5 Proposed next experiment (not yet run — pending confirmation)

Low-cost confirmation check, not expected to close most of the gap (see revised §12.3 assessment): widen
`-icnt_in_buffer_limit`/`-icnt_out_buffer_limit` (e.g. 512→1024) and/or `-icnt_subnets` (2→4), k3-isolated,
one knob group at a time against `best_optA`, and check whether `avg_icnt2sh_latency`/L2 miss latency drop and
by how much `gpu_tot_sim_cycle` moves. This is a different target than the DRAM channel-count/buswidth tuning
originally proposed (refuted by §12.2) — but given §12.3's corrected numbers, the realistic expectation is a
small effect, not a gap-closer. Primary value is closing out whether this category is fully ruled out (like
queue depth/MSHR width in prior docs) or contributes a small, real slice.

---

## 13. Methodology gap: every §10–12 comparison this session used cold-start k3, not the doc's own ceiling number

While cross-checking §12's numbers, found that the **+43% / 237,536-cycle "best config-only result" cited at the
top of this doc** (and in `prefill-k3-debug-investigation.md`, `prefill-k3-config-audit-and-occupancy.md`) does
**not** come from the same methodology as every comparison run this session (§10, §11, §12): it comes from
`log/prefill_best_multiopt/SM90_H100_best_optA.log`, which ran `-filter_first_kernel_id 1 -filter_last_kernel_id 3`
— kernels 1, 2, **and 3 in sequence** — not k3 in isolation.

Per-kernel cycle counts recovered from that log's three `gpu_tot_sim_cycle`/`gpu_tot_sim_insn` checkpoints:

| Checkpoint | cumulative `gpu_tot_sim_cycle` | cumulative `gpu_tot_sim_insn` |
|---|---|---|
| after kernel 1 | 18,969 | 132,169,728 |
| after kernel 2 | 33,695 | 198,623,232 |
| after kernel 3 | 271,231 | 1,369,001,984 |

Kernel 3's own incremental contribution: `271,231 − 33,695 = `**`237,536` cycles**, `1,369,001,984 − 198,623,232 =`
**`1,170,378,752`** instructions — the instruction count matches the k3-isolated run exactly (same workload), but
**the cycle count is 27,824 cycles (−10.5%) lower** than this session's `baseline_k3.log` (k3-isolated,
`-filter_first_kernel_id 3 -filter_last_kernel_id 3`, 265,360 cycles) — purely from running k3 **warm** (caches
already populated, pipeline already full, no cold-start) after kernels 1-2, with **zero code or config
difference**.

### 13.1 Why this matters

**Every comparison in §10, §11, and §12 of this document used the harder, cold-start (`filter_first_kernel_id 3
-filter_last_kernel_id 3`) methodology** — chosen for convenience (faster runs, isolates k3 cleanly for
instrumentation) but not the same target the doc's own historical "+43% ceiling" claim is based on. This means:

- The depfix/scoreboard_ex/sfu_int_lat/tensor_decouple unbundling results (§10-11) were measured against a
  265,360-cycle cold-start baseline, not the 237,536-cycle warm baseline — their *relative* conclusions
  (no improvement, +0.77% regression, etc.) still hold, but the *absolute* "+59.6% vs real" framing overstates
  the gap by about 10.5 points; the real apples-to-apples ceiling is closer to +43%.
- Cold-start vs warm-start is a **pure methodology effect** (pipeline fill latency, cache population time for the
  first wave of CTAs) — not a code bug, not a config knob, and not something `-filter_first_kernel_id 3` cleanly
  isolates away. It is, however, *real simulator behavior* worth understanding: does real H100 also pay a
  cold-start tax for k3 specifically, or does k3 in the real decode/prefill pipeline also always run warm (after
  earlier layers' kernels), making the warm-start 237,536 number the fairer comparison against the
  166,246-cycle real target?

### 13.2 RETRACTED: the warm-context comparison is a kernel-overlap bookkeeping artifact, not a real effect

**Correction (same day):** the warm-vs-cold framing above does not hold up. Two checks against
`SM90_H100_best_optA` config:

1. **L2 miss rate is unchanged.** Isolated k3 (`baseline_k3.log`): 1,138,704/5,190,776 = 21.94% miss rate.
   Kernel 3's incremental segment of the warm multi-kernel run: 1,101,892/5,104,409 = 21.59%. Nearly identical —
   there is no meaningful cache-warming effect from running kernels 1-2 first.
2. **`-gpgpu_max_concurrent_kernel 128` is set.** The simulator allows up to 128 kernels resident simultaneously,
   so in the warm multi-kernel run kernel 3's CTAs can start filling SMs that already finished kernel 2's CTAs
   *before* the "after kernel 2" cumulative-cycle checkpoint is reached. This means the subtraction
   `271,231 − 33,695 = 237,536` undercounts kernel 3's real cycle footprint — part of its execution bled backward
   across the checkpoint boundary due to legitimate kernel overlap, not because kernel 3 itself ran faster.

**Conclusion: the cold-start isolated number (265,360 cycles, +59.6%) is the trustworthy measurement of k3's own
cost, not the 237,536/+43% figure.** The "+43% ceiling" claim at the top of this doc and in
`prefill-k3-debug-investigation.md` / `prefill-k3-config-audit-and-occupancy.md` should be treated as suspect — it
likely understates the true gap by exactly this overlap artifact, not because of a genuine warm-start advantage.
No further action taken on the warm-context methodology; reverted to cold-start as the standard for all subsequent
comparisons (§14).

---

## 14. Where the barrier-correlated starvation actually comes from: per-warp memory-latency variance in the global-load stage

### 14.1 Motivation

§4-5 of the prior doc and §9 item 11(c) of this one flagged an unreconciled tension: our internal
`subcore_cycles_starved` accounting attributes a large share of stalls to `cta_barrier` co-occurring with total
SM idleness, while real `ncu` shows `barrier`=2.57% of warp-issue-cycles — tiny. If barrier waits are genuinely
that brief on real hardware, the question is *why* our model's warps apparently wait so much longer at `bar.sync`,
since the barrier mechanism itself isn't miscalibrated (CTAs really do have 8 co-resident warps that must
rendezvous). The hypothesis: warps accumulate **timing variance** between consecutive barriers (from
memory-latency differences, not from `bar.sync` itself), and the barrier simply exposes that variance as a wait.

### 14.2 Method

Ran `SM90_H100_best_optA`, k3-isolated, debug build, `-subcore_issue_debug 1 -subcore_issue_debug_stop_gpu_cycle 90000`,
capturing the existing `[bar_arrival_trace]` printf (`sm.cc`, gated on SM0/CTA0, all 8 warps) — it fires every
time a warp issues `BAR.SYNC`. Per the `barrier_set_t::warp_reaches_barrier` implementation (`shader.cc:3871`),
**all warps in a CTA are released simultaneously, at the cycle the last (straggler) warp arrives** — so the
spread (max − min) of arrival cycles across the 8 warps for a given barrier instance directly measures how long
the fastest warps sat waiting.

### 14.3 Result: one specific barrier site dominates the spread, every iteration

| Round | static PC | spread (cycles) |
|---|---|---|
| 1 | `0x330` (kernel entry barrier) | 6 |
| 2 | **`0x10020`** | **1,297** |
| 3 | `0x135b0` | 457 |
| 4 | `0x13a10` | 75 |
| 5 | `0x17780` | 198 |
| 6 | **`0x10020`** | **1,416** |
| 7 | `0x135b0` | 333 |
| 8 | `0x13a10` | 75 |
| 9 | `0x17780` | 198 |
| 10 | **`0x10020`** | **1,273** |
| 11 | `0x135b0` | 519 |
| 12 | `0x13a10` | 75 |

The main loop has 4 barriers per iteration. Three of them (`0x135b0`, `0x13a10`, `0x17780`) consistently show
75-520 cycle spread. **`0x10020` consistently shows 1,273-1,416 cycles — 10-20x larger — every single iteration.**

### 14.4 What's in that segment

Disassembled via `extra_info/enhanced_execution_info.json`'s per-instruction `op_code`/`pc_num_dec` fields for
this exact kernel (`mul_mat_q<Q8_0, Li128>`, kernel index 57). The segment causing the spread is the loop
back-edge from `0x17780`'s `BRA` (target `0xf290`) through to the `0x10020` barrier — 250 static instructions,
dominated by:

```
LDG.E.U16.CONSTANT  68   (global loads — quantized weight bytes)
STS                 54   (store into shared memory)
PRMT                32   (byte permute/unpack)
LDG.E.CONSTANT      18   (global loads)
IADD3 / IMAD.WIDE   ~40  (address computation)
```

**86 global loads per warp**, followed by stores into shared memory, followed by the barrier — this is the
per-iteration prefetch stage that loads the next tile of quantized weights. The other three barriers in the loop
sit after compute-only segments (`I2FP`, `FFMA`, `LDS`/`LDSM` from shared memory) with no global memory traffic —
exactly the ones showing small spread.

### 14.5 Mechanism, tying back to §12

§12 found `avg_icnt2sh_latency`=180 cycles (memory→SM response path) but `max_icnt2sh_latency`=**1,472** cycles —
a long tail far above the average. If even a few of a warp's 86 load requests land in that tail while a sibling
warp's loads mostly resolve near the 180-cycle average, that alone produces a 1,000+ cycle gap between warps —
closely matching the observed 1,273-1,416 cycle spread at `0x10020`. The proposed mechanism:

1. Per-iteration global-load stage has per-warp latency variance, driven by interconnect response-path
   congestion (tail, not average — §12.3's "only a fraction of misses see the full congested path" finding).
2. All 8 warps must rendezvous at the barrier immediately after, so the whole CTA pays the slowest warp's
   worst-case latency, every iteration (k3's main loop iterates many times — this compounds).
3. This inflates barrier-correlated stalls in our model's accounting without `bar.sync` release logic itself
   being wrong — it's downstream of memory-subsystem tail-latency variance feeding a hard sync point.
4. If real hardware's interconnect has a tighter tail (less variance between warps' load latencies), warps stay
   more synchronized through this stage, consistent with real `ncu`'s tiny `barrier`=2.57%.

### 14.6 Sharper test for §12.5's interconnect experiment

This gives the icnt buffer-widening experiment proposed in §12.5 a much better success criterion than "did total
cycles drop": **check whether widening `icnt_in_buffer_limit`/`icnt_out_buffer_limit`/`icnt_subnets` specifically
shrinks the spread at `0x10020`** (re-run the same `[bar_arrival_trace]` capped-cycle measurement). A reduction
there, even a partial one, would be the first concrete, mechanism-confirmed lever found this session — as
opposed to the dependency-chain latency fixes (§10-11), which were no-ops because they target the wrong layer
(per-instruction latency, not per-warp memory-timing variance feeding a sync point).

---

## 15. §12.5's icnt experiment: ruled out, but it revealed what the interconnect actually is

### 15.1 Per-partition load-balance check: ruled out

Before re-running §12.5, checked whether the `0x10020` divergence could come from **hotspotting at specific L2/
memory partitions** (some of the 80 partitions getting disproportionate traffic from the 132 concurrently-running
SMs). Parsed every `Memory Partition N:` block in `baseline_k3.log`:

| Metric | Range across all 80 partitions |
|---|---|
| `total_req` | 15,760–15,784 (spread 24, **0.15%**, stdev 7.95) |
| `bwutil` | 0.036737–0.036793 (essentially identical) |
| `mrqq_avg` (DRAM scheduler queue occupancy) | ~1.9–2.1, out of max capacity 64 |

**No imbalance.** Load is almost perfectly uniform across all 80 partitions, and none are remotely close to
saturated. Partition-level hotspotting is ruled out as a contributor.

### 15.2 Buffer/subnet widening: zero effect (and now we know why)

Created `SM90_H100_icnt_wide` (`icnt_in_buffer_limit`/`icnt_out_buffer_limit` 512→1024, `icnt_subnets` 2→4, same
`best_optA` k3-isolated harness as §14.2). Result: **the first 6 `[bar_arrival_trace]` rounds were byte-identical
to the baseline** — same arrival cycles, same spread (round 2 @ `0x10020`: 1,297 cyc; round 6: 1,416 cyc — exact
match). The widened buffers changed nothing at all.

**Root cause, found by reading `icnt_wrapper.h`/`local_interconnect.h`:**

```cpp
enum network_mode { INTERSIM = 1, LOCAL_XBAR = 2, N_NETWORK_MODE };
```

`-network_mode 2` (used by every config in this investigation) selects **`LOCAL_XBAR`** — a simple **single-stage
crossbar** (`local_interconnect.cc`, arbitrated by iSLIP or naive round-robin), not a real multi-hop mesh/booksim
network. `INTERSIM` mode isn't even supported in this build (`icnt_wrapper.cc:126-131` aborts with "Error.
Intersim not supported." if selected). **§12.4's "loose end" is resolved**: `-inter_config_file mesh` pointing to
a nonexistent file is harmless — that file only matters for `INTERSIM` mode, which is unreachable here. There is
no real NoC topology, no hop-count variance, no router chain in this model at all — just one arbitration stage
between every SM and every memory partition.

This also explains why widening buffers did nothing: §15.1 already showed queue occupancy is nowhere near
capacity (avg ~2/64), so there was no buffer-bound congestion to relieve in the first place — consistent, not
contradictory, with the null result.

### 15.3 Updated best guess: arbitration contention, not capacity or topology

With partition hotspotting ruled out (§15.1) and buffer capacity ruled out (§15.2, and explained structurally —
there's no topology for capacity to matter in), the remaining candidate is **contention at the crossbar's
arbitration stage itself**: when responses from multiple memory partitions target the same destination SM's
injection port in the same cycle, the iSLIP/round-robin arbiter (`-icnt_arbiter_algo`, currently iSLIP=1) grants
one at a time — a packet can wait several arbitration rounds behind others with zero queue buildup, which would
produce exactly the pattern observed: low average latency (`avg_icnt2sh_latency`=180) with a long tail
(`max_icnt2sh_latency`=1,472) that buffer size can't fix but arbitration policy or grant-cycle duration might.

**Not yet tested:** `-icnt_arbiter_algo` (try `0`=NAIVE_RR vs current `1`=iSLIP) and `-icnt_grant_cycles`
(currently default `1`).

### 15.4 `-icnt_arbiter_algo` tested: small effect, doesn't close the gap — 2026-06-20

Ran full cold-start k3-isolated (`best_optA` baseline minus `-icnt_arbiter_algo 1` replaced with `0`=NAIVE_RR),
same harness as §10-11 (`-filter_first_kernel_id 3 -filter_last_kernel_id 3`, no cycle cap).

| Config | `gpu_tot_sim_cycle` | vs baseline (iSLIP) | vs real HW (166,246 cyc) |
|---|---|---|---|
| `iSLIP` (`best_optA`, `-icnt_arbiter_algo 1`) | 265,360 | — | +59.62% |
| `NAIVE_RR` (`-icnt_arbiter_algo 0`) | 261,051 | **-1.62%** | +57.03% |

A real but small effect — confirms the arbitration policy is not a major contributor to the gap, consistent with
§15.2/15.3's finding that `-network_mode 2` is `LOCAL_XBAR`, a simple single-stage crossbar with little real
contention for an arbiter to resolve in the first place. `-icnt_grant_cycles` not tested (low expected value given
this result — the underlying arbitration stage clearly isn't where the gap lives). **This closes out the
interconnect-knob-tuning sub-thread of §15-16 entirely**: buffer capacity (§15.2), per-partition load balance
(§15.1, §16.5), DRAM admission "backlog" (§16.5, refuted as a measurement bug), and now arbitration policy have
all been tested and ruled out as material contributors. The best-supported remaining root cause is still §14's
per-warp memory-latency-variance mechanism — no fix attempted yet.

---

## 16. New instrumentation: per-stage `mem_fetch` latency mean/variance — found and fixed a confounding bug, then found something bigger than the original question

### 16.1 What was built

Added real per-stage latency tracking to `mem_fetch` (not just the two aggregate `icnt2mem`/`icnt2sh` numbers
already in §12): `mem_fetch::set_status()` now records, for the *previous* status before overwriting it, a
`(count, sum, sum_sq)` accumulator per `mem_fetch_status` enum value (`IN_PARTITION_ROP_DELAY`,
`IN_PARTITION_ICNT_TO_L2_QUEUE`, `IN_PARTITION_L2_TO_DRAM_QUEUE`, `IN_PARTITION_DRAM_LATENCY_QUEUE`,
`IN_PARTITION_DRAM`, `IN_PARTITION_DRAM_TO_L2_QUEUE`, `IN_PARTITION_L2_TO_ICNT_QUEUE`, `IN_ICNT_TO_SHADER`, etc.),
split into two buckets: `target` (request's PC falls in `[pc_lo, pc_hi]`) vs `other`. New CLI flags:
`-mem_fetch_stage_latency_debug`, `-mem_fetch_stage_latency_period`, `-mem_fetch_stage_latency_pc_lo/_hi`.
Printed periodically (every N cycles) and reset. Files: `mem_fetch.h`/`.cc`, `gpu-sim.cc` (option registration +
periodic print call after `gpu_sim_cycle++`).

Ran against `best_optA`, k3-isolated, capped at 90,000 cycles (same window as §14), targeting
`pc_lo=0xf290, pc_hi=0x10020` (the `load_tiles_q8_0` divergence segment from §14.4).

### 16.2 Bug found: the PC tag is lost once a request crosses into partition-side processing

The `target` bucket only ever populated 3 stages: `MEM_FETCH_INITIALIZED`, `IN_L1T_MISS_QUEUE`,
`IN_ICNT_TO_MEM`. Every later stage (`IN_PARTITION_ROP_DELAY` onward, including the response path
`IN_ICNT_TO_SHADER`) showed *zero* target entries — all of that traffic, including what should have been target
requests, landed in `other`.

Root cause, traced precisely: `mem_fetch::get_pc()` returns `m_inst.empty() ? -1 : m_inst.pc`, and `m_inst` is
only ever set once, at construction, from the `inst` pointer passed to the constructor. The exact loss point is
`memory_sub_partition::push()` (`l2cache.cc:795-821`) — our L2 is a **sectored** cache
(`-gpgpu_cache:dl2 S:64:128:40,...`), so every incoming request goes through
`breakdown_request_to_sector_requests(mf)`, which constructs *new* sector sub-`mem_fetch` objects via
`partition_mf_allocator::alloc(addr, type, ..., original_mf)` (`l2cache.cc:62-74`) — an overload that takes no
instruction pointer at all and passes `NULL` to the `mem_fetch` constructor (line 71), while correctly setting
`original_mf` to the parent request. These sector sub-fetches are what actually get `set_status(IN_PARTITION_ROP_DELAY/...)`
called on them from that point on — so `get_pc()` reverted to `-1` for every stage past the SM boundary.

### 16.3 Fix applied and verified

Fixed at the most general point rather than patching every allocator call site: `mem_fetch::get_pc()`
(`mem_fetch.h`) now falls back through the `original_mf` chain — the field that exists precisely to link a
sector sub-fetch back to its parent — when `m_inst` is empty:

```cpp
address_type get_pc() const {
  if (!m_inst.empty()) return m_inst.pc;
  if (original_mf != NULL) return original_mf->get_pc();
  return (address_type)-1;
}
```

Re-ran the identical capped 90,000-cycle capture after rebuilding. **Confirmed fixed**: `target` now populates
all 17 stages (`IN_PARTITION_ROP_DELAY`, `IN_PARTITION_ICNT_TO_L2_QUEUE`, `IN_PARTITION_L2_TO_DRAM_QUEUE`,
`IN_PARTITION_DRAM_LATENCY_QUEUE`, `IN_PARTITION_DRAM`, `IN_ICNT_TO_SHADER`, etc.), not just the 3 early ones.

### 16.4 What the fixed data actually shows: a chronic, system-wide admission backlog — bigger than the original question

Pooling the per-window stats across the whole 90,000-cycle run, most stages show target and other within a few
percent of each other (e.g. `IN_PARTITION_ROP_DELAY`: target mean 222.18 vs other 222.76 — effectively identical;
`IN_ICNT_TO_SHADER`: target 149.90 vs other 169.41). **No stage shows a clean, large target-specific variance gap**
— the original question's premise (find the one stage where the `0x10020` loads specifically suffer) doesn't
have a clean answer, because of what's actually happening:

**`IN_PARTITION_L2_TO_DRAM_QUEUE`'s wait time grows almost linearly with elapsed cycles, for target and other
alike, throughout the entire window — it never reaches steady state:**

| Window (cycle) | target mean (cyc) | other mean (cyc) |
|---|---|---|
| ~10,000 | 7,628 | 7,596 |
| ~20,000 | 10,292 | 11,388 |
| ~30,000 | 24,774 | 24,445 |
| ~40,000 | 38,493 | 38,514 |
| ~50,000 | 43,782 | 40,691 |
| ~60,000 | 57,475 | 54,884 |
| ~70,000 | 61,310 | 69,427 |
| ~80,000 | 73,617 | 71,236 |
| ~90,000 | 88,182 | 86,208 |

By the end of the window, a *newly arriving* request waits ~88,000 cycles in this one queue alone — comparable to
the entire window's length. This is a **chronic, ever-growing backlog affecting essentially all memory traffic**,
not something specific to the divergence segment — a substantially bigger lever than the original per-barrier
question, if it can be addressed.

**Code traced**: `memory_partition_unit::dram_cycle()` (`l2cache.cc:300`) has `break; // the DRAM should only
accept one request per cycle` — each of the 80 memory partitions admits at most 1 new request per DRAM-clock
tick into the DRAM-latency pipeline. Checked whether this cap is the binding constraint: with `gpgpu_n_mem=80`
and DRAM clock (2619 MHz) ≈1.617x core clock (1620 MHz), the aggregate theoretical admission ceiling
(~80 × 1.617 requests/core-cycle) is far higher than k3's actual demand over a 90k-cycle window — so the simple
per-channel rate cap doesn't look sufficient to explain the backlog by itself. Credit limits
(`m_private_credit_limit=1`, `m_shared_credit_limit`≈64+ from `gpgpu_frfcfs_dram_sched_queue_size`) aren't binding
either (`mrqq_avg`≈2, matches §15.1's full-run average).

**Leading hypothesis (tested in §16.5, REFUTED): transient per-partition hotspotting.** §15.1 showed partitions
are balanced *on average over the full ~265k-cycle kernel run* — that didn't rule out a small subset of the 80
partitions being hit by most of the 132 concurrent SMs *at any given instant*. Tested directly below.

### 16.5 §16.4's "chronic backlog" retracted — it was a measurement-bug artifact, not a real phenomenon

Added a direct instantaneous-occupancy probe: `memory_sub_partition::L2_dram_queue_length()` (`l2cache.h`),
exposing the live FIFO depth of `m_L2_dram_queue` per partition, sampled every 2,000 cycles via a new
`gpgpu_sim::print_mem_partition_queue_snapshot()` (`gpu-sim.cc`/`.h`), gated by
`-mem_partition_queue_snapshot_debug`/`_period`. Ran the same capped 90,000-cycle k3-isolated harness
(`best_optA`) with this enabled.

**Result: the queue is essentially always empty.** Across all 45 samples (every 2,000 cycles), per-partition
depth was 0 or 1 for all but a single transient blip (`max=29` at cycle 8,000); mean depth never exceeded 0.16
after the warm-up. There is no backlog, growing or otherwise, and no hotspotting to find — the premise of
§16.4's finding doesn't hold up under direct measurement.

**Root cause of the §16.4 illusion**: `L2interface::push()` (`l2cache.h:265`):
```cpp
virtual void push(mem_fetch *mf) {
  mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
  m_unit->m_L2_dram_queue->push(mf);
}
```
This hardcodes cycle `0` instead of the real current cycle — a pre-existing bug already flagged `/*FIXME*/` by
the original gpgpu-sim authors, unrelated to our PC-propagation fix in §16.3. Contrast the *other* call site for
the same status, `l2cache.cc:591`, which correctly passes `m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle`. Since
`mem_fetch::set_status()` computes dwell time as `cycle_at_exit - m_status_change`, every request entering this
queue via `L2interface::push()` — the normal L2-miss-to-DRAM path, which is what k3's loads use — got
`m_status_change = 0`. So §16.4's "dwell time" was actually `exit_cycle - 0`, i.e. the absolute simulation clock
reading, not a real queueing delay. That is exactly why it grew ~linearly with elapsed cycles (7,628 → 88,182):
it was measuring wall-clock progress, not contention.

**Conclusion**: §16.4's "chronic, system-wide admission backlog" does not exist. The `IN_PARTITION_L2_TO_DRAM_QUEUE`
stage is not a meaningful lever for the cycle-count gap; this entire sub-thread (interconnect/DRAM-admission
congestion as the explanation for the simulator's overestimate) is now exhausted without finding a real,
actionable mechanism. The remaining open candidate from §15.3 (`-icnt_arbiter_algo`/`-icnt_grant_cycles`) and the
per-warp memory-latency-variance mechanism from §14 (still the best-supported root cause so far) are what's left.

---

## 18. Single-SM isolation experiment: the barrier-arrival variance is mostly *structural*, not contention — 2026-06-20

§15.4's `-icnt_arbiter_algo` test and §16's MSHR-entries test both showed the same pattern: the bar-arrival spread at
`pc=0x10020` barely budged (1297/1416/1273/967/1295 cyc with MSHR 2→32, vs. baseline 1273-1416) even though both
levers should relieve contention if contention were the cause. This raised the question directly: is the variance
even caused by inter-SM/inter-partition contention from the other 131 concurrently-running SMs at all, or is it
inherent to the memory hierarchy's structural address-to-partition mapping regardless of system load?

**Method**: added `-debug_isolate_sm_id <id>` (new CLI flag, `shader.cc`'s `simt_core_cluster::issue_block2core()`)
which skips every SM except the target one *before* it ever calls `get_kernel()`/`set_kernel()` — so only one SM
in the entire 132-SM chip ever binds to a CTA. Combined with `-gpgpu_max_cta 1` (existing flag, caps total CTAs
issued across the whole run to 1) for good measure, though the SM-skip alone is what actually matters here.

(Note: a first attempt using only `-gpgpu_max_cta 1`, without the new SM-skip, did **not** work — all 132 SMs still
bound to a CTA each, producing a 28-million-line, 3.5GB log. `kernel_more_cta_left()`/`hit_max_cta_count()` exist
and look correct on inspection, but something about how/when `m_total_cta_launched` gets folded into
`gpu_tot_issued_cta` clearly doesn't gate fast enough across 132 SMs trying to issue in the same cycle — not fully
root-caused, deprioritized since the new SM-skip flag sidesteps the question entirely and is more direct anyway.)

Verified the fix with a 5,000-cycle capped run: exactly one `bind to kernel` line, on Shader 0. Then ran the full
k3-isolated kernel to completion (`-debug_isolate_sm_id 0 -gpgpu_max_cta 1`, no other CTAs, no other SMs):

| Round (cycle) | Isolated spread (1 SM, 0 contention) | Contended baseline (132 SMs) |
|---|---|---|
| 1 | 1007 | |
| 2 | 716 | |
| 3 | 847 | |
| 4 | 555 | |
| 5 | 749 | |
| 6 | 1170 | |
| 7 | 699 | |
| 8 | 772 | |
| 9 | 1155 | |
| 10 | 898 | |
| 11 | 517 | |
| 12 | 526 | |
| 13 | 721 | |
| 14 | 1008 | |
| avg/max | **~837 avg, 1170 max** | **1273-1416** (§14) |

**Conclusion — this is the key finding of the investigation so far**: the variance does **not** disappear in
isolation. It stays in the same 500-1200 cycle range with zero inter-SM contention, still far larger than this
same kernel's *other* barriers (75-519 cyc, §14) under the *same* isolated conditions implied by the mechanism
(other barriers don't have this divergent-load pattern feeding them). This means the dominant cause of the
`pc=0x10020` barrier variance is **structural**: different warps' disjoint row addresses (from
`load_tiles_q8_0`'s `i = i0 + threadIdx.y` access pattern, §14.4) map to different L2 partitions and DRAM
channels with different *fixed* latencies — not queueing/contention delay. This directly explains why every
contention-relief lever tried (icnt buffer/subnet capacity §15.2, per-partition load balance §15.1/16.5,
DRAM admission "backlog" §16.4-16.5 [retracted, was a bug], arbiter policy §15.4, constant-cache MSHR entries
§16-new) had negligible effect: none of them could touch a structural/fixed-latency difference, only
contention/queueing delay.

The isolated spread (~837 avg, 1170 max) is somewhat smaller than the contended baseline (1273-1416) — so
inter-SM contention is real but secondary, adding roughly 10-40% on top of an irreducible structural floor.

**Bonus result**: the isolated single-CTA run's `gpu_tot_sim_cycle = 218,024` is the time for one output tile
(one CTA) to complete the kernel's full K-loop with zero contention — a new reference point not previously
available, though not yet analyzed further.

**Open next step → resolved, same day**: checked whether `-gpgpu_l2_rop_latency`/DRAM access latency constants
are uniform across partitions. Confirmed yes, both are flat global constants applied identically everywhere,
with zero per-partition/address variation: `rop_latency` (`-gpgpu_l2_rop_latency 220`, `l2cache.cc:814`:
`r.ready_cycle = cycle + m_config->rop_latency`) and `dram_latency` (`-dram_latency 170`, `l2cache.cc:295,357`:
`d.ready_cycle = ... + m_config->dram_latency`) — neither references partition ID, address, or distance.

But after that fixed 170-cycle delay, the request is pushed (`l2cache.cc:374`, `m_dram->push(mf)`) into a
**separate, real address/bank-dependent DRAM timing model**: `-gpgpu_dram_timing_opt
nbk=16:CCD=1:RRD=11:RCD=37:RAS=87:RP=37:RC=124:CL=37:WL=6:CDLR=8:WR=32:nbkgrp=4:CCDL=6:RTPL=11` with
FR-FCFS scheduling (`-gpgpu_frfcfs_dram_sched_queue_size 64`). This is genuinely address-dependent: a request
hitting an already-**open row** in its target bank costs only `CL=37` cycles, while one needing
**precharge+activate** costs `RP+RCD+CL` ≈ 111+ cycles — determined entirely by which bank/row the address maps
to and what was previously accessed there. This is consistent with, and is now the leading explanation for, the
structural (contention-independent) variance found in this section: `load_tiles_q8_0`'s disjoint per-warp row
addresses have different row-buffer locality, so different warps' loads land at different points on the
CL-vs-RAS+RCD+RP cost spectrum — independent of any other SM's traffic. Not yet directly measured (e.g. via a
row-buffer-hit-rate stat correlated with arrival cycle per warp); would be the natural next test if pursued
further.

---

## 20. Per-request `mem_request_trace`/`mshr_probe_trace` instrumentation, and the sectored-L2 finding — HANDOVER, 2026-06-20

Session ending here on quota; handing off to Cursor. This section is the handover summary — read this first before re-reading
§1-19.

### 20.1 New instrumentation built this session (all already built into `gpu-simulator/bin/release/accel-sim.out`)

1. **`-debug_isolate_sm_id <id>`** (`shader.cc`, `simt_core_cluster::issue_block2core()`): skips every SM except
   the target one *before* it touches kernel/CTA state, so only one SM in the whole chip ever binds to a CTA.
   Verified: combine with `-gpgpu_max_cta 1` for a clean single-CTA run (the existing `-gpgpu_max_cta` alone does
   **not** work correctly by itself in this codebase — see the caveat in §18 — always pair it with
   `-debug_isolate_sm_id` if you want true 1-SM isolation).
2. **`-mem_request_trace_debug 1 -mem_request_trace_sm_id <id>`** (`gpu-sim.cc`, inside `gpgpu_sim::cycle()`'s
   ICNT block where `set_return_timestamp`/`IN_ICNT_TO_SHADER` is set): prints one line per memory response
   delivered to the target SM:
   ```
   [mem_request_trace] warp=N pc=0x... addr=0x... type=... send=T resp=T dur=T dram=0|1 [dram_enter=T dram_exit=T dram_dur=T]
   ```
   `dram=1` only if the request actually missed L2 and round-tripped through the real DRAM timing model;
   `dram_dur` = time from entering the DRAM subsystem (`L2_dram_queue` admission) to returning
   (`IN_PARTITION_DRAM_TO_L2_QUEUE`). New `mem_fetch` fields: `m_dram_enter_cycle`/`m_dram_exit_cycle` (sentinel
   `-1` if never entered DRAM), set at all 4 call sites in `l2cache.cc` covering both `dram_cycle()` (active path,
   `simple_dram_model=0` is default) and `simple_dram_model_cycle()` (inactive path, instrumented anyway for
   completeness).
3. **`mshr_probe_trace`** (`gpu-cache.cc`, `baseline_cache::send_read_request()`, reuses the same
   `-mem_request_trace_debug`/`_sm_id` flags — no separate flag): prints `mshr_hit`/`mshr_avail`/`mshr_addr`/
   `block_addr`/`cache=<name>` for every read-miss check on the target SM's requests, at **every** cache level
   (L1C, L1D, L2 sub-partitions all instrumented, distinguished by the `cache=` field, e.g. `L1C_000`,
   `L2_bank_097`).

All three are orthogonal and can be combined freely. Quick recipe for a clean, isolated, per-request trace:
```bash
source ./gpu-simulator/setup_environment_no_git.sh release
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread ./gpu-simulator/bin/release/accel-sim.out \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_best_optA/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config \
  -is_extra_traces_enabled 1 -filter_first_kernel_id 3 -filter_last_kernel_id 3 \
  -debug_isolate_sm_id 0 -gpgpu_max_cta 1 \
  -mem_request_trace_debug 1 -mem_request_trace_sm_id 0 \
  -subcore_issue_debug 1 -subcore_issue_debug_stop_gpu_cycle <N> \
  -trace /home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb \
  > log/tmp_log/<name>.log 2>&1
```
**Caution**: always check `ls -la` on the output log before grepping it. An earlier mistake in this session (using
`-gpgpu_max_cta 1` *without* `-debug_isolate_sm_id`) produced a 3.5GB, 28-million-line log because the CTA cap
silently failed to gate other SMs — always verify with a short capped run first (e.g.
`-subcore_issue_debug_stop_gpu_cycle 5000`) and `grep -c "bind to kernel"` before trusting a long run.

### 20.2 What this instrumentation found: the L2 sectored cache is a real, quantified amplifier

Traced a concrete example: one warp's single `LDG.E.U16.CONSTANT` instruction generates 4-5 coalesced
sub-accesses 32 bytes apart (`0x...0080, 0x...00a0, 0x...00c0, 0x...00e0, 0x...0000`), all landing in the **same**
128-byte L2 cache line (`-gpgpu_cache:dl2 S:64:128:40,...` — SECTOR type, confirmed via `block_addr` identical
across all 4-5 `mshr_probe_trace` lines, `cache=L2_bank_097`). But `mshr_addr` differs per request, because for a
SECTOR cache `m_atom_sz = SECTOR_SIZE = 32` (`gpu-cache.h:684`), not the 128-byte line size — so each 32-byte
sector is tracked, fetched, and DRAM-admitted **independently**, even though they're all needed by the same warp
at the same instant and target the same physical line.

Measured cost of this, in the fully-isolated (1 SM, zero contention) single-CTA run: `dram_enter` cycles for the
5 sectors were `5765, 5766, 5767, 5768, 5769` — exactly one cycle apart — because of the existing
"DRAM accepts one request per cycle per partition" admission cap (`l2cache.cc:300/362`, found earlier in §16.4's
investigation). `dram_dur` grew steadily `221 → 227 → 234 → 240 → 246` (+6-7 cycles per extra sector), purely
from this self-imposed serialization — **not** contention from other SMs, not row-buffer conflicts, just the
per-partition admission gate applied even to one warp's own simultaneous multi-sector burst.

**Architectural read**: `SECTOR_SIZE=32`/`SECTOR_CHUNCK_SIZE=4` are hardcoded `const unsigned` in
`abstract_hardware_model.h:826-827` (not configurable). Sectored caching exists to avoid fetching *unneeded*
sectors for sparse access patterns — but `load_tiles_q8_0` needs every sector of every line it touches (dense
GEMM-style read), so sectoring buys zero bandwidth benefit here while adding the serialization cost above.
**Not yet tested**: switching `-gpgpu_cache:dl2`'s cache-type token from `S` (SECTOR) to `N` (NORMAL) — same
nset/line_sz/assoc, just `m_atom_sz` becomes the full 128-byte line instead of 32 bytes, collapsing 4 sectors into
1 MSHR entry / 1 DRAM admission per line. This is a real, already-supported config-only change (no rebuild
needed), but it's a **global** change (every kernel through L2, not just k3) and moves away from accurately
modeling Hopper's real sectored L2 — frame it as "how much would sectoring's overhead matter if it weren't
there" rather than a production fix. **→ Tested in §22.1 (2026-06-20 Cursor session): −12.1% on isolated 1-SM/1-CTA tile; requires `gpu-cache.cc` fill-path fix when upstream caches remain sectored.**

### 20.3 Where the overall investigation stands (condensed from §1-19)

- **Confirmed root mechanism** (§14, §18): per-warp memory-latency variance at the `pc=0x10020` barrier in
  `load_tiles_q8_0` is mostly **structural** (different warps' disjoint addresses → different DRAM
  banks/rows/sectors with different fixed costs), not contention — confirmed by single-SM isolation (§18: spread
  stayed 517-1170 cyc even with zero contention, vs 1273-1416 cyc contended, vs only 75-519 cyc at the kernel's
  other barriers).
- **Ruled out as major levers** (all tested, all negligible-to-small effect): icnt buffer/subnet capacity (§15.2,
  zero effect), per-partition load imbalance (§15.1/16.5, none found), "DRAM admission backlog" (§16.4, retracted
  — was a `0/*FIXME*/` clock-reset bug in `L2interface::push()`, not real), icnt arbiter policy iSLIP vs NAIVE_RR
  (§15.4, -1.6%), constant-cache MSHR entries 2→32 (this session, -0.34%).
- **New, quantified, not-yet-tested lever**: the L2 sectored-cache per-sector independent-fetch/admission
  serialization (§20.2 above) — structurally sound, partially explains §18's "structural" variance, magnitude not
  yet measured at full-kernel scale (only seen in one example, not yet run as an isolated experiment with N-type
  L2). **→ Now measured: full-chip −18.5% (265,360→216,392); isolated 1-SM/1-CTA −12.1%.**
- **Net cycle gap still open**: baseline `best_optA` k3-isolated cold-start = 265,360 cycles vs real H100 target
  ≈166,246 cycles (+59.6%). **After L2-normal full-chip: 216,392 (+30.2% vs real) — largest single lever so far
  (−18.5%), but ~50k cycles remain.** The §20.2 sector
  experiment, plus directly quantifying what fraction of the total gap is attributable to `pc=0x10020`-style
  barrier stalls at all (proposed in an earlier turn, never run), are the two most promising open threads.
  **→ §22.2 refines the barrier thread: the ~800 cyc/round warp spread at `pc=0x10020` is inter-barrier compute
  skew (scoreboard/pipeline), not memory deps at the sync or sectored-L2 stagger alone.**

---

## 22. L2-normal experiment + combined sync/mem trace analysis — 2026-06-20 (Cursor session)

### 22.1 L2 `S` → `N` experiment (isolated 1-SM / 1-CTA)

**Command recipe** (always pair both isolation flags):
```bash
-debug_isolate_sm_id 0 -gpgpu_max_cta 1 \
-gpgpu_cache:dl2 N:64:128:40,L:B:m:L:P,A:192:4,32:0,32
```

**Crash on first attempts**: CLI-only L2-normal override while L1I/L1C/L1T remain sectored (`SECTOR_ASSOC` /
`SECTOR_TEX_FIFO` MSHR) hit pre-existing fill-path bugs:
- `baseline_cache::fill()` — L0_icnt wrapper `original_mf` not keyed in L1's `m_extra_mf_fields`
- `tex_cache::fill()` — same pattern for texture responses

**Fix** (uncommitted, `gpu-cache.cc`): in both `baseline_cache::fill()` and `tex_cache::fill()`, only enter the
sector-reassembly branch when `get_original_mf() != nullptr` *and* the parent is tracked; if the downstream
returned the tracked miss directly (normal L2), fall through to single-fill.

**Results** (k3-only, cold start, `best_optA` + L2-normal override):

| Config | Log | `gpu_tot_sim_cycle` | vs real H100 |
|--------|-----|---------------------|--------------|
| Sectored L2, **full chip** (132 CTAs) | `log/prefill_k3_compare/baseline_k3.log` | **265,360** | **+59.6%** |
| Normal L2, **full chip** (132 CTAs) | `log/prefill_k3_compare/dl2_normal_k3_fullchip.log` | **216,392** | **+30.2%** |
| **Full-chip Δ (L2 N vs sectored S)** | | **−48,968 (−18.5%)** | closes **~29 pp** of gap |
| Sectored L2, 1-SM/1-CTA (§18) | `log/tmp_log/k3_single_cta_isolated_v2.log` | **218,024** | +31.1% |
| Normal L2, 1-SM/1-CTA | `log/prefill_k3_compare/dl2_normal_k3_isolated_memtrace.log` | **191,627** | +15.3% |
| **Isolated Δ (L2 N vs sectored S)** | | **−26,397 (−12.1%)** | |

At SM0 `[commit_progress] committed_insts=100000`: baseline **99,844** cyc vs full-chip L2-normal **80,276** cyc
(−19.6% at same instruction progress — consistent with final −18.5%).

Full-chip L2-normal also raised `gpu_ipc` (5409 vs 4411 baseline); `gpu_occupancy` unchanged (~12.49%).
Real H100 target remains **166,246** cycles — **~50k cycles (~30%) still open** after L2-normal.

**L1D sectoring tested too — zero effect, closed without a full-chip run.** Switched `-gpgpu_cache:dl1` from
`S:4:128:128,...` to `N:4:128:128,...` *on top of* the L2-normal config
(`SM90_H100_l2norm_l1dnorm/gpgpusim.config`) and re-ran the 1-SM/1-CTA isolated test:
`gpu_tot_sim_cycle = 191,627` — **byte-identical** to L2-normal-alone (0.00% delta, 0 incremental cycles).
Confirms the prediction from the static op-mix (§3): `load_tiles_q8_0`'s loads all route through L1C (already
`N` type, never sectored), so L1D sectoring was never on this kernel's critical load path; whatever `STG`
output-write traffic (128 static instructions) exists doesn't generate enough sector-crossing miss traffic to
matter, at least not within one CTA's window. Given the per-CTA effect is exactly zero, no full-chip run is
needed — **L1D sectoring is closed out as a non-lever for k3.** (L1C, L1T, L1I, L0C were already confirmed `N`
type earlier — §20/§22 — so L2 was the only sectored cache with a real effect on this kernel.)

**L2-normal mem-trace read**: with normal L2, the §20.2 `dram_enter` +1/cycle/sector stepping pattern was **not
observed**. Texture `dram_dur` is bimodal: **220 cyc** (~56%, likely L2-hit/short path) vs **243–272 cyc** (DRAM
path). Adjacent 32B sectors at the same PC still show 243 vs 272 cyc (`pc=0xf350`), but without the sectored
admission stagger.

**Verdict**: sectored L2 is a **real, quantified ~12% cost** on the isolated tile — worth keeping as a modeling
accuracy question — but it is **not** the dominant lever on the +60% real-HW gap and does **not** explain the
`pc=0x10020` warp-arrival spread (see §22.2).

### 22.2 Combined `[sync_*_trace]` + `[mem_request_trace]` analysis

**Log**: `log/prefill_k3_compare/dl2_normal_k3_sync_memtrace.log`  
**Flags**: `-subcore_issue_debug 1 -mem_request_trace_debug 1 -mem_request_trace_sm_id 0` + isolation + L2-normal.

**New sync instrumentation** (uncommitted, `sm.cc` + `warp_dependency_state.{h,cc}`):
- `[sync_issue_trace]` / `[sync_commit_trace]` on SM0 for `BARRIER_OP`, `MEMORY_BARRIER_OP`, `GRID_BARRIER_OP`,
  `LDGDEPBAR_OP`, `DEPBAR_OP`
- Prints: `wait_barrier_bits`, `dep_counters`, `outstanding_mem_pcs` (FIFO of load/store PCs per depbar slot),
  `scoreboard_rd/wr`, `inst_in_pipe`
- `[bar_arrival_trace]` unchanged (all 8 warps, SM0/CTA0)

#### Finding A — large, recurring warp spread at `pc=0x10020`

15 BAR.SYNC rounds at `pc=0x10020`; spread **650–1026 cycles every round** (typical ~750–800 cyc):

| Round | Spread | Cycle range |
|-------|--------|-------------|
| 1 | 762 | 6774–7536 |
| 8 (worst) | **1026** | 91842–92868 |
| 15 | 808 | 178252–179060 |

Early kernel barrier `pc=0x330`: only **6 cycles** spread (warps 4–7 @ 1708 vs 0–3 @ 1714).

#### Finding B — spread is **not** memory deps at the sync instruction

At every `[sync_issue_trace]` for `pc=0x10020`:
- `wait_barrier_bits=0`, all `dep_counters=0`, no `outstanding_mem_pcs`
- `scoreboard_wr=0`; warps are waiting on **each other**, not outstanding traced memory

Last texture `resp` before the worst barrier round: only **38–136 cycles** before BAR.SYNC — far smaller than
the **~800 cyc** arrival spread. The skew accumulates over the **whole inter-barrier compute segment**, not in
the last few memory returns.

#### Finding C — late arrivers carry more in-flight work

At the final `pc=0x10020` round: late warps (1–4, arrive ~179017–179060) have `inst_in_pipe=15`; earlier warps
(0, 6, 7) have `inst_in_pipe=6–12`. Consistent with §9 item 9 (shared RF writeback queue / scoreboard release
gating) as the skew mechanism, not memory-at-barrier.

#### Finding D — barrier mechanics still correct

All 8 warps reach each round; `[bar_arrival_trace]` + prior §7/§14 release checks unchanged. BAR.SYNC **exposes**
skew; it does not **create** the ~800 cyc compute-length difference.

### 22.3 One-sentence summary (current best understanding)

**The +60% vs real-H100 gap is mostly the model stretching inter-barrier compute unevenly across warps
(~750–1000 cyc spread at `pc=0x10020` every loop iteration), which BAR.SYNC then exposes as idle wait — not
broken sync logic, not memory deps at the barrier, and not fully explained by sectored L2 (which adds ~12% on its
own).**

### 22.4 Best next targets

1. **Quantify skew mechanism**: compare `[sync_issue_trace] inst_in_pipe` + issue/commit traces between early vs
   late warps through one full loop body; test `-max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions`
   (§9 item 10 ruled out width=4, but mechanism may be latency-shaped not width-shaped).
2. **Sectored vs normal L2 on warp spread**: **done** — see §22.6. Spread shrinks modestly (~14%) but persists at
   the same order of magnitude → L2 sectoring affects latency far more than skew, confirming the prediction.
3. **Full-chip L2-normal**: **done** — `dl2_normal_k3_fullchip.log`, **216,392 cyc** (−18.5% vs baseline). Remaining
   gap vs real: **+30.2%**.

### 22.6 Warp-arrival spread under L2-normal (isolated 1-SM/1-CTA) — resolves §22.4 item 2

Re-ran the `[bar_arrival_trace]` capture (single-SM isolation, `-debug_isolate_sm_id`/`-gpgpu_max_cta 1`) with the
L2-normal config (`SM90_H100_l2norm_l1dnorm`). `gpu_tot_sim_cycle = 191,627` — matches the known L2-normal isolated
total exactly (sanity check passed).

`pc=0x10020` spread, 15 rounds: 762, 816, 782, 750, 801, 779, 654, 1026, 649, 763, 754, 292, 246, 958, 808
→ avg **722.7**, max **1026**, min **246**.

| | Sectored L2 isolated (§18) | Normal L2 isolated (this run) |
|---|---|---|
| Rounds | 14 | 15 |
| Spread avg | ~837 | **722.7** (−13.7%) |
| Spread max | 1170 | **1026** (−12.3%) |
| Spread min | 517 | **246** |

**Conclusion**: the skew shrinks modestly under L2-normal but is clearly not eliminated — still 246–1026 cyc per
round, still an order of magnitude above this kernel's other barriers (75–519 cyc, §14), still highly
round-to-round variable (not a fixed latency offset). This confirms §22.3's read: L2 sectoring's −12%
(isolated)/−18.5% (full-chip) cycle win comes mainly from cutting absolute per-load DRAM round-trip cost, not
from making warps finish more uniformly. The warp-to-warp skew at `pc=0x10020` remains a largely independent,
unsolved mechanism — §22.4 item 1 (RF-writeback/scoreboard-release timing, `inst_in_pipe` disparity) is still the
best lead.

### 22.5 Code changes this session (uncommitted)

| File | Change |
|------|--------|
| `gpu-cache.cc` | `baseline_cache::fill()` + `tex_cache::fill()` — tolerate normal-L2 response when upstream MSHR is sectored |
| `sm.cc` | `[sync_issue_trace]` / `[sync_commit_trace]` helper |
| `warp_dependency_state.{h,cc}` | FIFO `m_pending_mem_pcs[]` per depbar slot; ldgsts PC tracking |

---

## 21. File index (additions to prior docs)

| Path | Contents |
|------|----------|
| `notes/prefill-k3-real-hw-correlation.md` | **This document** |
| `log/prefill_k3_compare/{baseline,depfix,scoreboard_ex,sfu_int_lat,tensor_decouple}_k3.log` | §10-11 cold-start k3-isolated comparison logs (`scoreboard_ex`/`sfu_int_lat` not yet re-run after the 2026-06-20 memory-thrashing kill) |
| `log/prefill_best_multiopt/SM90_H100_best_optA.log` | §13: source of the now-retracted "+43%/237,536" warm multi-kernel figure |
| `log/tmp_log/bar_arrival_90000.log` | §14: `[bar_arrival_trace]` capture used to find the `0x10020` barrier divergence |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_k3_{scoreboard_ex,sfu_int_lat,tensor_decouple}/` | §11: unbundled fix-isolation configs |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_icnt_wide/` | §15.2: icnt buffer/subnet widening config (zero effect) |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/mem_fetch.{h,cc}` | §16.1-16.3: per-stage `mem_fetch` latency tracking + `get_pc()` `original_mf` fallback fix |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h` | §16.3: `original_mf` accessor context; §16.5: `L2_dram_queue_length()` instantaneous-depth accessor; also where the `0/*FIXME*/` bug in `L2interface::push()` lives |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.{cc,h}` | §16.1: `-mem_fetch_stage_latency_*` CLI flags + periodic print; §16.5: `-mem_partition_queue_snapshot_*` CLI flags + `print_mem_partition_queue_snapshot()` |
| `log/tmp_log/mem_stage_stats_90000_fixed.log` | §16.3-16.4: post-fix per-stage latency capture (source of the now-retracted backlog table) |
| `log/tmp_log/mem_partition_snapshot_90000.log` | §16.5: instantaneous per-partition queue-depth capture that refuted §16.4 |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_constcache_mshr/` | constant-cache MSHR-entries-bump config (2→32), negligible effect |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc` | §18: `g_debug_isolate_sm_id` global + skip-check in `simt_core_cluster::issue_block2core()` (new `-debug_isolate_sm_id` flag) |
| `log/tmp_log/k3_single_cta_isolated_v2.log` | §18: the single-SM isolation run — key evidence that the barrier variance is mostly structural |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/mem_fetch.h` | §20.1: `m_dram_enter_cycle`/`m_dram_exit_cycle` fields + `went_to_dram()`/getters/setters |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc` | §20.1: 4 call sites setting dram enter/exit cycle (covers both `dram_cycle()` and unused `simple_dram_model_cycle()`) |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc` | §20.1: `-mem_request_trace_debug`/`_sm_id` flags + `[mem_request_trace]` print in `cycle()`'s ICNT block |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc` | §20.1: `[mshr_probe_trace]` print in `baseline_cache::send_read_request()` (reuses `-mem_request_trace_debug`/`_sm_id`) |
| `log/tmp_log/mshr_probe_test.log`, `log/tmp_log/mem_request_trace_full.log`, `mem_request_trace_test.log` | §20.2: captures showing the sectored-L2 per-sector independent-fetch/admission-serialization finding |
| `log/prefill_k3_compare/dl2_normal_k3_isolated_memtrace.log` | §22.1: L2-normal isolated 1-SM/1-CTA run (191,627 cyc) + `[mem_request_trace]` only |
| `log/prefill_k3_compare/dl2_normal_k3_sync_memtrace.log` | §22.2: L2-normal isolated + `[sync_*_trace]` + `[mem_request_trace]` — barrier skew analysis |
| `log/prefill_k3_compare/dl2_normal_k3_fullchip.log` | §22.1: L2-normal **full-chip** k3 run — **216,392 cyc** (−18.5% vs baseline) |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc` | §22.1: `baseline_cache::fill()` + `tex_cache::fill()` normal-L2 workaround; §22.5 uncommitted |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc` | §22.5: `[sync_issue_trace]` / `[sync_commit_trace]` (uncommitted) |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/warp_dependency_state.{h,cc}` | §22.5: `outstanding_mem_pcs` FIFO tracking (uncommitted) |
| `gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h` | §23.4: `get_issue_cycle()` getter |
| `log/tmp_log/k3_l2norm_all_traces_v2.log` | §23.4-23.6: all-warps `commit_trace` + `mem_request_trace` + `bar_arrival_trace` capture; source of the barrier-stall-fraction and issue-rate calculations |
| `log/tmp_log/k3_l2norm_rr_issue_bar_arrival.log` | §23.3: round-robin issue-scheduler test (198,541 cyc, worse) |
| `log/tmp_log/k3_l2norm_random_tiebreak.log` | §23.3: greedy-then-random tie-break test (191,943 cyc, spread −8 to −17%) |
| `log/tmp_log/k3_l2norm_scoreboard_ex.log` | §23.9: `-is_scoreboard_release_at_ex 1` validated to completion (191,704 cyc) |
| `log/tmp_log/k3_l2norm_skiprf.log` | §23.8: `-is_skip_rf_limit_enabled 1`, 4 CTAs/SM sensitivity test (794,079 cyc) |
| `log/tmp_log/k3_l2norm_rfwq4.log`, `k3_l2norm_rfwq8.log` | §23.9: write-queue pop-rate 4 vs 8 (identical, saturates at 4) |
| `log/tmp_log/k3_l2norm_rflat8.log` | §23.9: RF-latency-bound 24→8 (byte-identical to baseline, wrong knob) |
| `../h100_real_profile/k3_occupancy_starvation_check.csv` | §23.7: real H100 ncu profile of this exact kernel, used for the occupancy/issue-rate cross-check |

---

## 23. Subcore issue-priority root cause, occupancy cross-check vs real H100, and a string of ruled-out scoreboard-timing knobs — 2026-06-20/21

### 23.1 Memory/compute resource sharing across subcores — reference answer

Checked object lifetimes directly in code (not assumed). Per-subcore (4 independent copies per SM):
tensor core (`m_tensor_pipeline`, `subcore.cc:1724`), SP/INT/SFU/branch/uniform pipelines, register file
(`m_regular_rf`), and the tiny L0 instruction/constant caches (`m_L0I`/`m_L0C_cache`, named e.g. `L0I_000_3`).
SM-wide/shared across all 4 subcores: the entire `ldst_unit_sm` (L1D, L1C, L1T, PRT, MSHR, the one port to
icnt/L2/DRAM — created once in `sm.cc:1009`), shared memory (lives inside `ldst_unit_sm`), the scoreboard
(`m_scoreboard`, one per SM, `shader.cc:322`), and optionally the DP pipeline (`-is_dp_pipeline_shared_for_subcores`).

### 23.2 Root cause of the subcore-pair warp skew: a deterministic fetch/issue-priority seed

`Subcore::finilized_warps_assignation()` (`subcore.cc:1706-1709`) seeds both `m_greedy_pointer_issue` and
`m_greedy_pointer_fetch` to `m_warps_of_subcore.size() - 1` — i.e. local index 1 in each subcore (warps 4-7,
since subcores hold `{warp_id, warp_id+4}` pairs) — **before the kernel runs a single cycle**. Combined with
`order_greedy_then_highest_id()` (`subcore.cc:1415-1442`, ties always favor highest `dynamic_warp_id`) and the
"stick to whoever issued last" rule (`subcore.cc:1096`), this gives warps 4-7 a deterministic, compounding head
start over their subcore partners 0-3.

Confirmed directly via the new all-warps `commit_trace` (§23.4): at the kernel's very first instruction
(`pc=0x10`), warps 4-7 issue at `cycle=1504`, warps 0-3 at `cycle=1511` — a 7-cycle gap **before any compute,
memory access, or divergence has happened**. The gap then rides through unchanged to the first barrier
(`pc=0x330`: warps 4-7 commit `BARRIER_OP` at cycle 1715, warps 0-3 at 1721 — same 6-7 cycle gap, same 7-cycle
BARRIER_OP execution latency for both groups, proving the gap predates the barrier entirely). Later in the
kernel this same seed compounds into the much larger texture-load gaps already documented in §18/§22
(e.g. `pc=0xf470`/`0xf9b0`: 577-647 cycles by mid-kernel) via the same "stick to last issuer" mechanism.

**Initially mis-identified as "subcore-index-dependent" (odd vs even)** from a 5-round sample — ruled out by
extending to all 15 rounds: the pattern (subcore 1/3 large gap, 0/2 small gap) completely breaks down by round
8, with magnitude and sign both drifting unpredictably thereafter. The real, robust signature across all 15
rounds is leadership **drift/sign-flips** (whichever subcore-partner gets ahead first tends to stay ahead for a
while, but can lose that lead when it stalls) — consistent with the deterministic seed plus the self-reinforcing
"sticky greedy" rule, not a fixed hardware asymmetry.

**Addendum (confirmed generalizes to the output epilogue too)**: same mechanism, different code region. In
`k3_all_debug.log`, subcore 1's warp1 shows a 68-cycle gap between issuing `pc=0x180e0` (cycle 182943) and
`pc=0x180f0` (cycle 183011) — but its subcore-mate warp5 issued the *identical* PCs ~450 cycles earlier and was
actively issuing every 5-7 cycles throughout that exact window. Ruled out the L0 I-cache as the cause first
(`L0I_total_cache_misses = 0` for the entire run — zero real misses anywhere) before attributing it to fetch/issue
port sharing: `Subcore::fetch()`/`Subcore::issue()` both allow exactly one warp through their shared per-subcore
port per cycle, prioritized by the same `order_greedy_then_highest_id()`/"stick to whoever issued last" rule. Since
warp5 had a continuous stream of ready instructions, it kept winning both arbitrations every cycle, and warp1's
`is_next_valid()` check on its own IBuffer simply never got serviced — no hazard, no stall reason logged, just
never fetched. This is a second, distinct cause of slow store commits alongside §23.23's `RESERVATION_FAIL`/L1D
contention: one delays *commit* after issue, this one delays *issue* itself.

### 23.3 Scheduler-policy A/B tests on the seed bias

| Policy | `gpu_tot_sim_cycle` (isolated 1-SM/1-CTA, L2-normal) | vs baseline | `pc=0x10020` spread avg/max | Leadership pattern |
|---|---|---|---|---|
| Greedy-then-highest-id (baseline) | 191,627 | — | 722.7 / 1026 | sign-flips, self-correcting |
| `-is_subcore_round_robin_issue_scheduler 1` (true RR) | 198,541 | **+3.6% worse** | 1035.4 / 1174 | **locked-in**, warps 4-7 lead every round, no exceptions |
| **`-is_subcore_random_tiebreak_issue_scheduler 1`** (new, this session) | 191,943 | +0.16% (flat) | **663.9 / 846** | sign-flips preserved |

True round-robin is strictly worse on both axes — it removes the beneficial flexibility of letting a warp that's
genuinely making progress keep going. The new **greedy-then-random tie-break** (randomize only the tie-break in
`order_greedy_then_highest_id` instead of always favoring highest `dynamic_warp_id`, implemented in `subcore.cc`,
flag registered in `gpu-sim.cc`/`shader.h`) is a real, validated improvement: −8.1% avg / −17.5% max barrier
spread at essentially zero cycle cost, while preserving the self-correcting leadership-flip dynamics that
strict RR destroyed. **Uncommitted.**

### 23.4 `commit_trace` rework: removed `issue_trace`, added `issue_cycle`/`commit_cycle` to `commit_trace`, all 8 warps

Per explicit request: deleted the separate `[issue_trace]` print in `SM::issue_warp()`; added a public
`warp_inst_t::get_issue_cycle()` getter (`abstract_hardware_model.h`, exposing the existing protected
`issue_cycle` member set in `warp_inst_t::issue()`); `[commit_trace]` now prints
`issue_cycle=... commit_cycle=...` directly, and the print is no longer gated to `warp_id==0` — fires for all 8
warps on SM0. One line now gives exact per-instruction issue→commit latency per warp, no need to correlate two
separate trace lines. Verified: `gpu_tot_sim_cycle` unchanged (191,627), `commit_trace` fires evenly (32,599 per
warp × 8 = 260,792 total). **Note**: `run_k3_segment_timing.sh:63` has a regex matching the old `[issue_trace]`
format — needs updating to use the new combined field if that script is reused. **Uncommitted.**

### 23.5 Barrier-stall quantified: only ~4.86% of total time — BAR.SYNC is not a major cost driver

Computed directly from `[bar_arrival_trace]` across **all** barrier instances in one full isolated run (61
instances: 4 distinct barrier PCs per loop iteration × 15 rounds + 1 kernel-start barrier — not just the
`pc=0x10020` one focused on in §14/§18/§22): summed `(release_cycle - arrival_cycle)` per warp per barrier
instance = **74,462 total idle warp-cycles**, against `8 warps × 191,627 cycles = 1,533,016` available
warp-cycles → **4.86%**. Per-warp distribution is fairly even (4,216-12,882 each), no outlier.

**This settles the multi-session barrier-skew thread**: however large and real the warp-to-warp arrival skew at
`pc=0x10020` is (§14/§18/§22), it accounts for under 5% of total kernel time. BAR.SYNC *exposes* compute-rate
differences between warps; it doesn't meaningfully *cost* cycles on its own. Confirmed independently by §23.3's
random-tiebreak result: spread shrank 8-17% but total cycles didn't move — further proof spread reduction and
cycle-count reduction are not the same thing here.

### 23.6 The real cost driver: scoreboard stalls, not barriers, not memory

Pre-existing `-subcore_issue_debug` instrumentation (added in an earlier, uncommitted-at-the-time session, never
previously analyzed this way) already tracks per-opcode stall reasons at IBuffer head. Pulled from the isolated
L2-normal run:

```
subcore_cycles_starved = 66.06% of subcore-cycles (no warp issuable)
top (op, reason) pairs:
  47.46%  SP_OP / scoreboard          9.83%  SFU_OP / cta_barrier
   9.53%  TENSOR_CORE_OP / scoreboard 8.63%  SFU_OP / scoreboard
   6.03%  INTP_OP / scoreboard        5.38%  LOAD_OP / scoreboard
   4.86%  LOAD_OP / cta_barrier
```
Grouped by reason: **`scoreboard` ≈ 78.6%** of all stall events; `cta_barrier` ≈ 16.2% (consistent with §23.5's
independent 4.86%-of-wall-clock measurement — most barrier *stall events* don't translate to much *wall-clock*
idle time, since by the time a warp checks-and-stalls repeatedly the cumulative idle window is still small).
Memory (`LOAD_OP`) is a minor contributor. **The dominant inefficiency is plain FP32 (`SP_OP`) instructions
waiting on register-scoreboard dependencies — not memory, not barriers.**

### 23.7 Real-hardware ncu cross-check (`h100_real_profile/k3_occupancy_starvation_check.csv`)

Real ncu profile of the same kernel (llama-bench, qwen3 8B Q8_0, H100), pulled to check whether occupancy or the
scoreboard-stall finding are simulator artifacts:

| Metric | Real H100 (ncu) | Our simulator | Match? |
|---|---|---|---|
| Block Limit Registers | 1 | `CTA/core = 1, limited by: regs` | **exact** |
| Theoretical/Achieved Occupancy | 12.50% / 12.49% | `gpu_occupancy = 12.4856%` | **exact** |
| Active Warps Per Scheduler | 2.00 | 2 (8 warps / 4 subcores) | **exact** |
| Block Limit Shared Mem | 1 | 4 (found via §23.8's skip-RF test) | **mismatch — open, unresolved** |
| Issued Warp Per Scheduler | 0.49 | 0.340 (issued/(cycles×4 subcores)) | **gap: −15 points** |
| Warp Cycles Per Issued Instruction | 4.11 | 5.88 (= 2.00 / issue_rate) | **gap: +43%** |

Register count verified from trace metadata directly (not assumed): max register index referenced by the
kernel's instructions is **R255** — the architectural maximum — confirming the register-bound, 1-CTA/SM,
12.5%-occupancy story is a real, deliberate property of the compiled SASS (register-hungry GEMM kernel trading
occupancy for per-thread ILP via heavy unrolling), present identically on real silicon. **Occupancy itself is
correctly modeled and is not the source of the simulator-vs-real gap.** But the *issue-rate* at that occupancy
is real H100 issuing ~44% more often per scheduler-cycle than our model does — three independent metrics
(issue-rate, warp-cycles-per-instruction, and the §23.6 starvation/no-eligible comparison) agree to within
rounding on the same ~15-percentage-point gap. **This is now the most precisely quantified open thread in the
whole investigation.**

### 23.8 Occupancy/warp-pool-size ruled out as the cause of the issue-rate gap

Tested the obvious hypothesis — maybe the scheduler just doesn't have enough resident warps to find eligible
work — using the pre-existing `-is_skip_rf_limit_enabled 1` escape hatch (`shader.cc:3500`, skips the register
term in `shader_core_config::max_cta()`'s `min()`). Result: shared memory binds next, at **4** CTAs/SM (not 1,
unlike real ncu's `Block Limit Shared Mem=1` — a separate, real, unresolved discrepancy in how this simulator
computes the kernel's shared-memory footprint, flagged but not chased this session).

With 4 CTAs / 32 resident warps (4x the baseline pool):
```
                          1 CTA/SM (baseline)   4 CTAs/SM (skip-RF, unrealistic)
gpu_tot_sim_cycle:             191,627                794,079
issue_rate:                     0.340                  0.347   (+2%, ~nothing)
subcore_cycles_starved:         66.06%                 66.04%  (unchanged)
```
794,079 cycles for 4x the instructions is essentially exactly "4 CTAs running with zero latency-hiding benefit"
(4×191,627 = 766,508 is the no-benefit baseline; actual came out very slightly worse, not better). **Quadrupling
the resident warp pool gave almost no improvement in issue-rate or starvation** — ruling out "not enough warps
available" as the explanation. This is a deliberately unrealistic test (real H100 cannot run 4 CTAs/SM for this
kernel, confirmed by §23.7's register data) — it's a diagnostic probe, not a candidate fix.

### 23.9 Knob-by-knob elimination of SP_OP/scoreboard timing levers (all isolated 1-SM/1-CTA, L2-normal)

| Knob tested | Change | Result | Verdict |
|---|---|---|---|
| `-is_scoreboard_release_at_ex 1` (release at FU completion, not RF writeback) | existing flag, never validated to completion before (earlier session's run was killed mid-way) | 191,704 cyc (+0.04%); SP_OP/scoreboard stalls −14.2%; starvation unchanged (66.06%→66.58%) | **Ruled out** — removes some stall events but warps just hit the next stall instead; critical path unaffected |
| `-max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions` 1→4 | write-queue drain rate | 191,730 cyc (+0.05%); starvation 66.06%→65.58% (−0.48pp) | Small, real, free improvement — **persisted to config (1→4)** |
| same, 4→8 | pushed further | byte-identical to the 4 result (191,730 / 65.58%) | **Saturates at 4**, no further gain |
| `-max_latency_regular_register_file_latency` 24→8 | shrink RF "latency" | **byte-identical** to baseline (191,627 / 66.0612%, exact stall counts) | **Wrong knob** — this is a read-port-conflict bookkeeping array bound, not an actual latency; doesn't affect timing in this range |
| (checked, not changed) `max_sp_latency` | confirmed actual value | `= fp_latency[1] = 4` cycles, from `-ptx_opcode_latency_fp 4,4,4,4,39` | Already small and realistic — **not the inflated value the gap requires** |

Net: no single timing/queue/capacity constant explains the §23.7 issue-rate gap. The ~6-cycle difference between
SP_OP's raw 4-cycle EX latency and its measured ~10-cycle average issue→commit latency is fully accounted for by
known small fixed costs (`num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore=2` + queue insertion), none
of which are large enough to explain a 44%/15-point gap by themselves.

### 23.10 Real ncu profile fully mined — no further granular comparison available

Re-pulled the complete `h100_real_profile/k3_occupancy_starvation_check.csv`: it contains only Scheduler
Statistics, two Warp State Statistics metrics, and Occupancy (already used in §23.7) — no per-reason stall
taxonomy (no `stall_long_scoreboard`/`stall_barrier`/`stall_exec_dependency`-style breakdown as ncu's GUI
sometimes exposes). There is nothing more granular in this specific export to compare against §23.6's
op×reason breakdown.

### 23.11 Where this leaves the investigation

The ~15-percentage-point issue-rate gap vs real H100 (§23.7), at matched occupancy, survives every config-level
lever tried this session: scheduler fairness policy (ruled out, random-tiebreak helps spread only, not cycles),
warp-pool size (ruled out, §23.8), and every SP_OP/scoreboard timing constant found (ruled out, §23.9). It is
likely **not a single mistuned parameter** but something more structural in how dependency chains/instruction
issue are modeled, requiring either a deeper architectural comparison (re-profiling real hardware with a fuller
stall-reason breakdown than the current CSV has) or accepting it as a standing limitation of this model relative
to real Hopper's actual per-scheduler issue logic.

**Separately flagged, not yet chased**: the shared-memory-occupancy-limit mismatch found in §23.8 (our simulator
computes 4 CTAs/SM-worth of shmem headroom for this kernel; real ncu says only 1) — a real, concrete, isolated
discrepancy that's easier to pin down than the issue-rate gap and worth a dedicated look.

### 23.12 Code changes this session (uncommitted — see `git status`)

| File | Change |
|------|--------|
| `gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h` | `warp_inst_t::get_issue_cycle()` public getter |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc` | Removed `[issue_trace]`; `[commit_trace]` now all-warps with `issue_cycle=`/`commit_cycle=` |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h` | `is_subcore_random_tiebreak_issue_scheduler` member |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc` | `-is_subcore_random_tiebreak_issue_scheduler` CLI registration |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc` | `order_greedy_then_highest_id()` — random tie-break among ready warps when flag enabled |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_l2norm_l1dnorm/gpgpusim.config` | `max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions` 1→4 (persisted) |

### 23.13 CONFIRMED: shared-memory occupancy-limit mismatch — driver carveout-tier rounding, not a units bug

From §23.8: our simulator computes `CTA/core = 4, limited by: shmem` for this kernel; real ncu reports
`Block Limit Shared Mem = 1`. Investigated to find the exact numbers (not guessed), then confirmed against the
actual ggml/llama.cpp CUDA source (`/home/qshao/Project/Fun/llama.cpp`):

- **Traced per-block shared memory request**: `57,856 bytes`, read directly out of the kernel-3 entry in
  `dynamic_trace.pb` via the `kernel.size_shared_memory` protobuf field (compiled `kernel.proto` locally with
  `protoc --python_out` and parsed the trace — confirmed value, not estimated).
- **Tracer capture logic verified correct in principle**: `tracer_tool.cu:1301`,
  `ker->set_size_shared_memory(shmem_static_nbytes + cfg.sharedMemBytes)` — sums the function's static shared
  memory (`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`) with the per-launch dynamic `sharedMemBytes` argument. Checked
  all three launch-capture code paths (`extract_launch_config()`, `tracer_tool.cu:1104-1148`: plain
  `cuLaunchKernel`/cooperative/grid-async, `cuLaunchKernelEx`, and `cuGraphAddKernelNode` for CUDA-graph-launched
  kernels — relevant since llama.cpp launches via graphs) — all three correctly read `sharedMemBytes` from the
  appropriate params struct. No obvious tracer bug found by inspection — the 57,856 figure is accurate.
- **Our simulator's arithmetic is internally correct given its inputs**: `233,472 (gpgpu_shmem_size, H100's full
  228KB architectural max) / 57,856 = 4.037 → 4`. Not a units bug (byte vs word) — shared memory is unambiguously
  byte-denominated in the CUDA APIs on both sides; checked and ruled out explicitly.
- **Confirmed root cause**, via `ggml/src/ggml-cuda/mmq.cuh:3886-3899` + `common.cuh:185-208`: the kernel does
  call `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared)` (wrapped in the
  `CUDA_SET_SHARED_MEMORY_LIMIT` macro), raising its allowed ceiling to **exactly** `nbytes_shared` — which
  matches the traced 57,856 bytes precisely. But **`cudaFuncAttributePreferredSharedMemoryCarveout` is never
  called anywhere in `ggml-cuda/`** (grepped the whole directory, no match) — the L1/shared-memory *split* is
  never explicitly requested, only the size *limit*. Without an explicit carveout-preference call, the CUDA
  driver does not grant the full physical maximum — it rounds up to the smallest **discrete carveout tier**
  that satisfies the requested size (tiers like 8/16/32/64/100/132/164/196/227 KB on Hopper, the same kind of
  table visible in our own config's `-gpgpu_shmem_option 0,8,16,32,64,228`). 57,856 bytes exceeds the 32KB tier
  but fits the 64KB tier, so the driver grants only **65,536 bytes**: `65,536 / 57,856 = 1.13 → 1` — **exactly
  matches real ncu's `Block Limit Shared Mem = 1`.** Confirmed, not just plausible.
- **What this means for the simulator**: `-gpgpu_shmem_size 233472` models the GPU's absolute physical maximum,
  but real CUDA only grants that to a kernel that explicitly opts into a larger carveout via
  `cudaFuncAttributePreferredSharedMemoryCarveout` — which this kernel (and, per the grep, every other ggml-cuda
  kernel) never does. A more accurate model would compute the occupancy-limiting shared-memory denominator as
  the smallest carveout tier ≥ the kernel's requested `size_shared_memory`, not the flat architectural maximum.
  **Not fixed this session** — `-gpgpu_shmem_size`/`max_cta()`'s `result_shmem` calculation in `shader.cc:3482-3485`
  would need the tier-rounding logic added.
- **Why this matters**: this CTA admission discrepancy (4 vs 1 — exactly the same register-style mechanism
  investigated at length in §23.2/§23.7/§23.8, but for shared memory instead of registers) doesn't change k3's
  own number (registers already bind at 1 regardless — §23.8), but it would matter for **any other kernel in
  the decode/prefill pipeline where shared memory is the binding constraint and registers aren't** — those
  kernels' simulated occupancy could be silently too high.
- **Not chased further this session** — flagged here as a concrete, scoped, separate thread from the main
  issue-rate gap (§23.7/§23.11), worth a dedicated look later.

### 23.14 BREAKTHROUGH: WAR (anti-dependency) scoreboard hazards are 1/3 of all stalls — disabling WAR checking is a real, validated win

Picking back up the §23.11 "structural, not a single mistuned parameter" thread: built a new live per-cycle
instrumentation flag, **`-issue_wait_trace_debug`**, that reuses the pre-existing `issue_stall_primary_reason()`
classifier but fires every cycle for every warp on SM0 (not just into the aggregate histogram) and, specifically
for `scoreboard`-reason stalls, additionally queries both scoreboards independently
(`find_first_collision_remodeling()` on `m_scoreboard` and `m_scoreboard_WAR`) to report which register and which
*kind* of hazard (RAW vs WAR) is blocking issue. New code in `subcore.cc`'s issue loop (both the issued and
stalled branches) and a new `-issue_wait_trace_debug` flag in `shader.h`/`gpu-sim.cc`.

Ran it on the isolated L2-normal config (`k3_issue_wait_trace.log`, 191,730 cyc, 513,681 `[issue_wait_trace]`
lines) and split the 197,098 `scoreboard`-reason stall events by hazard kind:

| Hazard kind | Count | % of all scoreboard stalls | % of ALL stall events |
|---|---|---|---|
| Pure RAW only | 105,618 | 53.6% | — |
| Pure WAR only (no RAW collision) | 83,950 | **42.6%** | **33.2%** |
| Both RAW and WAR | 7,530 | 3.8% | — |

I.e. **a third of every stall event in this kernel is a pure anti-dependency (WAR) hazard**, not a true data
dependency. GPGPU-Sim's WAR scoreboard mode is a config knob, not a fixed behavior:
`-scoreboard_war_mode {disabled|wb|opc}` — current/default is `opc` (release WAR claim when the instruction
leaves the operand collector; already the less-conservative of the two enabled modes).

**Test: `-scoreboard_war_mode disabled` (turn WAR checking off entirely)**

| Config | Baseline | WAR disabled | Δ |
|---|---|---|---|
| Isolated 1-SM/1-CTA, L2-normal (`k3_war_disabled.log`) | 191,730 cyc | **185,573 cyc** | **−3.21%** |
| Full-chip, 132 SMs, L2-normal (`k3_l2norm_war_disabled_fullchip.log`) | 216,392 cyc | **210,062 cyc** | **−2.93%** |

Total committed instructions unchanged (260,792 in the isolated case) — same work, fewer stalls, faster. The
full-chip result tracks the isolated result almost exactly (−2.93% vs −3.21%), confirming this is a real,
scale-independent effect, not an isolated-config artifact. Scoreboard-reason stall events dropped 197,098→105,390
in the isolated run (i.e. essentially all the pure-WAR stalls disappeared, leaving the RAW ones).

**This is the first genuinely positive lever found across this entire multi-session investigation.** It narrows
the gap to real H100 from +30.2% to **+26.35%** (210,062 vs target 166,246) on the full-chip config.

Caveat — not yet verified with 100% certainty: GPGPU-Sim's scoreboard is a *timing-model* heuristic, decoupled
from the functional value-computation engine (which uses the actual register file / memory state, not the
scoreboard). Disabling WAR tracking should therefore be timing-only and functionally safe in this model, but this
reasoning has not been independently confirmed by, e.g., diffing functional output/correctness checks with WAR
disabled vs enabled. Worth a quick correctness sanity pass before treating this as a permanent config change.

**Persisted**: `-scoreboard_war_mode opc` → `disabled` in `SM90_H100_l2norm_l1dnorm/gpgpusim.config` (the
"current best" config used throughout this investigation). The unmodified base `SM90_H100/gpgpusim.config` was
left at `opc` since it's meant to stay a clean reference config.

**Correctness sanity check — CONFIRMED SAFE.** Two angles:

1. *Architectural*: traced memory addresses/operands come directly from the NVBit-recorded trace
   (`trace_driven.cc:367`, `set_addr(i, trace.memadd_info[0]->addrs[i])`) — pre-computed on real hardware, not
   recomputed by any in-simulator functional/ALU engine. The scoreboard (WAR or RAW) only gates *when* an
   instruction is allowed to issue; it has no path to influence *what* address or operand value an instruction
   uses. In this trace-driven model there is no mechanism by which disabling WAR tracking could corrupt a value
   — it can only change cycle counts.
2. *Empirical*, diffing `k3_l2norm_all_traces_v2.log` (baseline, `opc`) vs `k3_war_disabled.log` (isolated,
   `disabled`):

   | Check | Baseline (`opc`) | WAR disabled | Verdict |
   |---|---|---|---|
   | `gpu_sim_insn`/`gpu_tot_sim_insn` (thread-instructions) | 8,333,696 | 8,333,696 | identical |
   | `gpgpu_n_tot_w_icount` (warp-instructions committed) | 260,792 | 260,792 | identical |
   | Core-level `GLOBAL_ACC_W`/`TEXTURE_ACC_R` issued | 2,048 / 24,928 | 2,048 / 24,928 | identical |
   | Total DRAM reads / writes | 67,360 / 0 | 67,360 / 0 | identical |
   | Deadlock detector / assertions / errors | none fired | none fired | both completed cleanly |
   | L2-level access count, icnt packets | 8441 / 35,828 | 8447 / 35,852 (+0.07%) | tiny, expected cache-timing shift, not corruption |

   Every functional invariant (instruction counts, memory accesses, DRAM traffic, clean completion with no
   deadlock) is byte-identical between the two runs; the only differences are in cache/interconnect counters
   that legitimately shift with issue timing. Treated as confirmed safe — `-scoreboard_war_mode disabled` is a
   pure timing-model change with no functional-correctness risk in this trace-driven simulator.

**Still not done**: investigating *why* WAR hazards are so common in this kernel specifically (mmq's inner loop
reuses register-allocated accumulator/operand slots heavily across unrolled iterations — a plausible source, not
yet confirmed by reading the generated SASS).

### 23.16 Re-tested IMMA init/dependent-latency decouple on the WAR-disabled config — still no improvement

Re-ran the already-built `-tensor_initiation_cycles_override 2 -tensor_dependent_latency_override 24` override
(real H100 ubench numbers, §1.1) on top of today's WAR-disabled best config, to see if the interaction changed now
that WAR stalls no longer dominate:

| Config | Cycles | Δ vs WAR-disabled-only |
|---|---|---|
| Isolated, WAR disabled only (185,573) | 185,573 | — |
| Isolated, WAR disabled + IMMA decouple (`k3_isolated_war_off_imma_decouple.log`) | **186,770** | **+0.65% (worse)** |

Same conclusion as the original §1.1 test on the old sectored-L2 baseline (then: −0.11%, negligible) — decoupling
the IMMA initiation/dependent-latency split toward the real ubench numbers does not help, and now actually
regresses slightly. Not pursued further at full-chip scale (isolated result was already unfavorable). The IMMA
formula split is not on today's critical path; ruled out a second time, on a different baseline.

### 23.17 SECOND POSITIVE LEVER: a real register bypass/forwarding network — new feature, built and validated

Acted on guess #3 from §23.14's list (no bypass/forwarding network was modeled at all — confirmed via grep, zero
bypass logic in `subcore.cc`). Built a real, scoped microarchitecture feature rather than a config-only test:

**Design** (`-is_register_bypass_forwarding_enabled`, `-register_bypass_window_cycles` default 2,
`-register_bypass_ports_per_subcore` default 1): when an instruction with destination registers leaves EX
(`functional_unit::instruction_finishing_execution`, the same hook `maybe_release_scoreboard_at_ex` already uses),
`SM::maybe_record_register_bypass()` marks each destination register as forwardable for `window_cycles`, in a new
per-warp table on `Scoreboard` (`m_bypass_forward_table`) — **without** releasing the real scoreboard claim, so
WAW ordering and full RF-writeback timing stay untouched. At issue time
(`Scoreboard::checkCollision_remodeling_with_bypass`, called from `subcore.cc`'s issue loop in place of the plain
`checkCollision_remodeling` when the flag is on), a scoreboard collision on one of the *consumer's own destination
registers* (WAW) always still blocks — never bypass-eligible. A collision on a *source operand* (true RAW) can
instead be satisfied by the bypass network if a forwarded write for that exact register is still within its
window and a bypass port is still free for that subcore this cycle (tracked per-subcore, reset every cycle).

**Results** (window=2, ports=1 — the defaults, chosen as a conservative/minimal first cut):

| Config | Cycles | Δ |
|---|---|---|
| Isolated, WAR disabled only | 185,573 | — |
| Isolated, WAR disabled + bypass network (`k3_isolated_bypass_w2p1.log`) | **184,904** | **−0.36%** |
| Full-chip, WAR disabled only | 210,062 | — |
| Full-chip, WAR disabled + bypass network (`k3_fullchip_bypass_w2p1.log`) | **208,650** | **−0.67%** |

Smaller than the WAR fix, but real, positive, and — unusually — *larger* proportionally at full-chip scale than in
isolation (−0.67% vs −0.36%), the opposite of the IMMA-decouple result. Instruction counts identical in both
full-chip runs (`gpu_sim_insn=1,170,378,752`, `gpgpu_n_tot_w_icount=36,715,744`) — functionally clean, no
correctness concern, consistent with the design (bypass only ever *shortcuts* a wait, never skips real ordering).

**Cumulative progress vs real H100** (166,246 cycles): 216,392 (L2-normal baseline, +30.2%) → 210,062 (+ WAR
disabled, +26.35%) → **208,650 (+ bypass network, +25.51%)**.

### 23.18 Bypass network window/port sweep — non-monotonic, window=4/ports=1 is the sweet spot

Re-profiled the new best config (`-subcore_issue_debug 1` on WAR-disabled + bypass w2p1, isolated): **SP_OP/scoreboard
is still the #1 stall reason** by a wide margin (44.2% of all 152,869 stall events, 67,559 events) — the narrow
2-cycle/1-port bypass only shaved a fraction of it. New secondary finding: `LOAD_OP/cta_barrier` (12.2%) and
`SFU_OP/cta_barrier` (11.3%) now rank #2/#3 by *stall-event count* — not necessarily by time-share (§23.5's
cycle-time-based barrier analysis found only ~4.86% of total time is barrier-bound; event-count share and
time-share are different metrics, not yet reconciled).

Swept `register_bypass_window_cycles`/`register_bypass_ports_per_subcore` (isolated, 1-SM/1-CTA):

| Window | Ports | Cycles | Δ vs WAR-disabled-only (185,573) |
|---|---|---|---|
| 2 | 1 | 184,904 | −0.36% |
| **4** | **1** | **184,384** | **−0.64% (best found)** |
| 2 | 2 | 184,904 | −0.36% (identical to 2/1 — extra port unused at this window) |
| 4 | 2 | 184,624 | −0.51% |
| 8 | 2 | 184,730 | −0.45% |
| 16 | 4 | 185,339 | −0.13% (worse than even the narrowest 2/1 case) |

**Non-monotonic** — widening past window=4 makes things *worse*, not better, and 16/4 nearly erases the whole
gain. Likely a second-order scheduling effect: shortening more RAW waits shifts which warp the
greedy-then-highest-id priority order picks each cycle, which can ripple into worse downstream decisions
elsewhere (the same kind of sensitivity already seen in the scheduler-policy A/B tests, §23.3). Not a sign of a
bug in the bypass logic itself — confirms this issue scheduler is generally sensitive to small timing
perturbations in non-obvious ways.

**Full-chip validation of window=4/ports=1** (`k3_fullchip_bypass_w4p1.log`): **208,191 cycles** — a further
−0.22% vs the window=2/ports=1 full-chip result (208,650), tracking the isolated improvement's direction.
Instruction counts identical to every other full-chip run this session (`gpu_sim_insn=1,170,378,752`,
`gpgpu_n_tot_w_icount=36,715,744`) — functionally clean.

**Cumulative progress vs real H100, updated**: 216,392 (L2-normal baseline, +30.2%) → 210,062 (+ WAR disabled,
+26.35%) → 208,650 (+ bypass w2/p1, +25.51%) → **208,191 (+ bypass w4/p1, +25.23%)**.

**Window=4/ports=1 persisted** to `SM90_H100_l2norm_l1dnorm/gpgpusim.config`
(`-is_register_bypass_forwarding_enabled 1 -register_bypass_window_cycles 4 -register_bypass_ports_per_subcore 1`).

### 23.19 Re-profile on the new best config, and a `cta_barrier` red herring resolved

Re-ran `-subcore_issue_debug 1` on today's persisted best config (isolated, 184,384 cyc). Stall-event mix barely
moved from §23.14's earlier snapshot: total stall events 152,869→148,231 (−3.0%), but `SP_OP/scoreboard` stayed at
**the same ~44.3% share** (65,607 events) — the bypass network only reaches RAW pairs within its 4-cycle window;
the bulk of `SP_OP/scoreboard` stalls are genuine longer-distance dependency chains a forwarding network
structurally can't shorten (real compute latency elapsing, not a register-file round-trip tax — consistent with
§23.9's earlier finding that SP_OP's raw 4-cycle EX latency is already realistic).

New apparent lead: `LOAD_OP/cta_barrier` + `SFU_OP/cta_barrier` + `BRANCH_OP/cta_barrier` together are 23% of all
stall *events*, and ~99% of those are tagged "starved" (literally nothing else issuable that cycle) — looked
promising. **Dug in and it's a red herring, already settled by §23.5.**

`issue_debug_barrier_kind()`/`is_warp_blocked_by_programmer_barrier_for_issue()` (`subcore.cc:223-266`) confirm
`cta_barrier` = `sm->warp_waiting_at_barrier()` — exactly the same arrival→release window §23.5 already measured
via `[bar_arrival_trace]`. Re-ran §23.5's exact method on today's config (61 barrier instances, pairing every 8
consecutive `[bar_arrival_trace]` lines, summing `max_arrival − own_arrival` per warp):

- **71,692 total idle warp-cycles** → `71,692 / (8 × 184,384) = 4.86%` of total time — **identical to the original
  §23.5 measurement**, unchanged by today's WAR-disable/bypass fixes.
- Reconciles the apparent event-count/time-cost mismatch: 37,135 `cta_barrier` stall *events* is only ≈52% of the
  71,692 true idle warp-cycles — the event log *under*-counts real barrier idle time (a barrier-waiting warp's
  IBuffer entry often isn't even valid/fetched yet for much of its wait, so there's nothing to classify that
  cycle), it doesn't exaggerate it. The near-100% "starved" tag on events that *do* get recorded just confirms
  that whenever a barrier stall is visible in the log, the whole subcore genuinely had nothing else to do.

**Verdict**: barriers remain a small, ~4.86%-of-time cost, exactly as §23.5 found — not a meaningful lever despite
the misleadingly large stall-event share. Re-confirms `SP_OP/scoreboard`'s genuine longer-distance RAW chains as
the one real remaining thread (the L2/DRAM miss-latency excess flagged as the other thread is closed out next,
§23.20 — it turns out to no longer exist on the current config).

### 23.20 CLOSED: the "L2/DRAM miss-latency excess" was stale, and no longer exists on the current config

Dug into the §1/§4/§9-item-7 "miss latency excess" (simulated avg miss latency 658.83 cyc vs real DRAM RTT
478-495 cyc, an apparent +170-190 cyc excess) that this document's earlier sessions had flagged as "important, not
a simple fix" and an open thread. Two findings close it out for good:

1. **The 658.83 cyc figure is stale and unreproducible.** `grep -rn "avg_miss_latency\|avg_hit_latency"` across the
   entire `gpgpu-sim/src/` tree returns **zero matches** — this stat doesn't exist anywhere in the current
   codebase. It must have come from older/temporary instrumentation, not anything still present. (The notes had
   already self-corrected this once before, in §12.3 of an earlier session: re-measured on the `best_optA`
   baseline and found the real gap was only +7%, not +38% — but even that re-measurement predates this session's
   L2-normal/WAR-disable/bypass fixes and is itself now outdated.)
2. **Fresh measurement on today's actual best config**, using the real per-request `[mem_request_trace]`
   instrumentation (`k3_isolated_newbest_memtrace.log`, 8,420 DRAM-bound requests captured, isolated 1-SM/1-CTA):
   avg DRAM round-trip (`send`→`resp`) = **242 cyc**, of which 235 cyc is the DRAM-subsystem span itself
   (`dram_enter`→`dram_exit`) and only ~7 cyc combined is pre/post overhead. **This is nearly 2x faster than real
   H100's measured 478-495 cyc DRAM RTT, not slower.**

**Verdict: this thread is closed, not just deprioritized.** There is no miss-latency excess left to fix on the
current config — if anything, the L2-normal simplification (§20.2/§22, a deliberate working choice, not a
claimed production-accurate setting) has overshot past real DRAM latency in the *fast* direction. Memory latency
is conclusively **not** a contributor to the remaining +25.23% gap; the gap is entirely on the compute/issue side
(`SP_OP/scoreboard`'s genuine longer-distance RAW chains, confirmed in §23.19, is the one real remaining thread).

**Side change, also new this session**: the SM0 commit-progress heartbeat (`[commit_progress] sm=0
committed_insts=... cycle=...`, originally added behind `-subcore_issue_debug` and firing every 1,000
instructions) is now **always on** regardless of `-subcore_issue_debug`, and the interval was changed to every
**10,000** instructions (`sm.cc`, `instruction_retirement()`). This exists purely so long full-chip runs always
show liveness/progress in the log without needing the (expensive) full debug-trace flag.

### 23.21 CLOSED: subcores are single-issue by hard structural design — not a lever, and it matches real H100

With `SP_OP/scoreboard`'s longer-distance RAW chains confirmed as the one real remaining thread (§23.19/§23.20),
checked whether dual-issue (one subcore issuing >1 instruction/cycle to different pipes simultaneously) was even
architecturally possible in this model, as a way to hide those RAW waits with otherwise-independent work. It is
not, and that's correct, not a gap to fix:

1. `Subcore::cycle()` (`subcore.cc:605-616`) calls `issue(m_sm)` **exactly once per subcore per cycle** — a single
   call site, no loop over multiple issue attempts.
2. Inside `issue()`, the priority-ordered warp scan `break`s immediately after the first successful issue
   (`subcore.cc:1108`) — but even without that `break`, there's nowhere for a second instruction to go that
   cycle: `m_ISSUE_CONTROL_latch = register_set_uniptr(1, "ISSUE_CONTROL_latch")` (`subcore.h:131`) is a
   **single-slot** pipeline register. Every other pipeline latch in this design
   (`m_CONTROL_ALLOCATE_latch`, `m_EX_WB_sm_shared_units_latch`, `m_read_stage_aux_latch`, etc.) is likewise
   explicitly sized `1` — single-issue is structural, baked into the pipeline register width throughout, not an
   incidental loop-control artifact.

**Why this isn't a bug to fix**: real Hopper SMs are divided into 4 sub-partitions, each with its own warp
scheduler issuing exactly **1 instruction per cycle** to its private execution pipes — NVIDIA's documented SM
microarchitecture since Volta/Turing (unlike much older architectures with limited dual-issue in some cases).
This model's single-issue-per-subcore is architecturally **correct** for H100, not a simplification costing us
cycles relative to real hardware. **Ruled out as a lever** — no dual-issue headroom exists to recover here.

### 23.22 CONCLUSIVE: the real SASS schedule is already near-optimal — the gap is the simulator's SP_OP latency model, not the kernel's code

Checked whether k3's compiled SASS itself has unexploited ILP that a better instruction ordering could expose
(i.e. is the `SP_OP/scoreboard` bottleneck a property of the kernel's actual code, or of how our simulator models
it). Two approaches:

1. **False lead, caught and discarded**: tried to read this directly off our own trace's `control_bits` field
   (`stall_count`/`wait_barrier_bits`, captured per-instruction in `extra_info/enhanced_execution_info.json`).
   Found all 6,352 of k3's static instructions report `stall_count=0`/`wait_barrier_bits=0` uniformly — including
   memory loads, which on real Volta+ hardware *always* need a wait-barrier (load latency is variable, can't use
   a static stall count). That uniform all-zero pattern is the tell: confirmed via
   `grep -rn "stall_count\|control_bits\|wait_barrier" util/tracer_nvbit/tracer_tool/tracer_tool.cu` → **zero
   matches** — this fork's tracer never actually populates `control_bits` from the real SASS control words. The
   field exists in the schema but is dead/unwired. Caught before reporting it as a finding.
2. **Real measurement**: extracted the actual compiled SASS via `cuobjdump --dump-sass --gpu-architecture sm_90`
   on the real `libggml-cuda.so` (function `_Z9mul_mat_qIL9ggml_type8ELi128ELb0EEv...`, the exact traced k3
   kernel), then decoded the genuine Volta+/Hopper control-word encoding (stall count, yield, write/read barrier
   index, wait-barrier mask — bits 41-57 of each instruction's second 64-bit hex word) with a purpose-built
   decoder. Validated the decode is correct via a self-consistency check: a `MUFU.RCP` (variable-latency SFU op)
   sets `wrtbar=1`, and the next dependent instruction shows `waitmask` with bit 1 set — exactly the real-hardware
   barrier-handshake pattern, not something a wrong bit-offset would produce by chance.

**Result, on the dequant-accumulate chain that IS the kernel's `SP_OP` bottleneck** (`FFMA.FTZ`/`FMUL.FTZ`/
`I2FP.F32.S32`, 1,024 instances each — converts the `IMMA` tensor-core integer accumulator to scaled float and
accumulates into the output, 1,536 total instructions sampled):

- **Average stall = 1.245 cycles, 96.7% of instances at stall ≤ 2.** `IMMA` itself: 86% at stall=1.
- The only high-stall-count code in the whole kernel (stall=4/5/13) is the address-computation/software-emulated
  integer-division preamble — not the hot accumulation loop.

**Verdict**: `ptxas` already achieved near back-to-back issue for this exact instruction sequence on real
hardware — there is no missing ILP for a better instruction ordering to expose; restructuring the SASS is not a
viable lever (and not something achievable within this investigation's scope anyway — it would mean a different
question, asked of ggml's CUDA kernel authors, not us). **This reframes the whole remaining gap**: our simulator
runs this *exact same instruction sequence* (same trace, same register dependencies) but classifies 44% of all
stall events as `SP_OP/scoreboard` (§23.19) — meaning the simulator is inserting far more waiting for this code
than real hardware's own compiler-validated schedule requires. The bottleneck is that this model's SP_OP
dependent-latency (4-cycle EX latency + scoreboard-release timing, even after the §23.17-23.18 bypass-network
fix) is structurally more conservative than what real Hopper's deeply-pipelined FP32 ALU + compiler-driven static
scheduling actually sustains for this code (~1.2-cycle effective dependent latency, not 4+). The path to closing
more of the remaining +25.23% gap is narrowing this model's SP_OP-class dependent latency, not touching the
kernel's code.

### 23.22b Refinement: `SP_OP/scoreboard` is a MIX — one part over-modeled, one part a genuine real-HW cost

Followed up §23.22 by checking which actual instructions occupy the hottest `SP_OP` PCs in the dynamic trace
(every SP_OP PC for warp 0 commits exactly 15 times — once per K-loop round, confirming these are loop-carried,
not unrolled-independent). Cross-referenced against the static instruction JSON
(`extra_info/enhanced_execution_info.json`, `unique_function_id=34`): the hottest PCs are **not** the FFMA
dequant-accumulate epilogue from §23.22 — they're `IMAD.WIDE` **address-computation** instructions, the K-loop's
per-round load-address pointer-increment chain (e.g. `f750: IMAD.WIDE R46,R15,0x22,R44` → `f7a0: IMAD.WIDE
R48,R15,0x22,R46`, an explicit induction-variable RAW chain on the address registers, repeating once per round).

Checked these exact PCs in the real decoded SASS (`k3_sass_decoded.txt`): **`stall=3` and `waitmask=000001` on
nearly every one** — unlike the FFMA chain (no wait-barrier usage at all), these *do* use the dynamic wait-barrier
mechanism on real hardware. Reading the surrounding code explains why: the same address register (e.g. `R46`) is
used as a load address by an `LDG.E.U16.CONSTANT` a few instructions earlier (`f530: LDG ... [R46.64+0x4]`), then
overwritten by the next round's `IMAD.WIDE` (`f590: IMAD.WIDE R46,R15,0x22,R6`) — **a WAR hazard on the address
register itself**: the new address write must wait until the earlier load finishes reading the old value. Real
hardware genuinely respects this with a wait-barrier (load completion time is unpredictable, a static stall count
can't cover it) — a real, necessary cost, not simulator over-conservatism.

**This refines §23.22, doesn't overturn it.** `SP_OP/scoreboard` is a mix of (at least) two different things:
1. FFMA/FMUL/I2FP dequant-accumulate epilogue — real hardware pays almost nothing (avg stall=1.245, §23.22) —
   genuinely an area where this simulator's dynamic scoreboard is more conservative than real hardware needs.
2. `IMAD.WIDE` address-recomputation chain (this section) — real hardware *does* pay a real, comparable cost
   (`stall=3` + wait-barrier, roughly the same ballpark as this model's ~4-cycle EX latency + scoreboard wait) —
   a genuine, hardware-confirmed WAR dependency, not a modeling gap to chase further.

**Net effect**: there is real headroom in #1, but #2 is likely already about as good as it can get without a
real change to the kernel's address-computation strategy (e.g. fewer/cheaper induction-variable updates per round
— an upstream ggml-cuda code question, out of scope here).

### 23.24 Per-PC stall attribution built, and it conclusively resolves #1 vs #2 in favor of #1

Built `-subcore_issue_debug 1`'s new `[pc_stall_stats]` output: a per-static-PC stall histogram (mirrors the
existing per-op-type one, see `PcStallHistogramState`/`pc_stall_histogram_record`/`_print_body` in `subcore.cc`),
recorded at the same two call sites as the existing op-type histogram, printed sorted by stall count descending.
This directly answers which *exact instructions* the §23.22b mix is actually made of, rather than inferring it
from occurrence counts (which is what led §23.22b toward the wrong instruction class — occurrence count picks out
whichever PCs simply execute most often, not whichever ones actually generate the most *stall* time).

**Result, top SP_OP-classified PCs by stall count** (isolated best config, `k3_pc_stall_stats.log`):

| pc | static instruction | our sim stall events | real SASS stall | real waitmask |
|---|---|---|---|---|
| 0xdb70 | `IMAD R4, R5, R4, RZ` | **4,474** (by far the largest single PC) | 4 | `000010` (real barrier wait) |
| 0x10e30 | `FMUL.FTZ R49, R59, R48` | 1,243 | 1 | none |
| 0x13220 | `FMUL.FTZ R50, R46, R97` | 900 | 2 | none |
| 0x12b00 | `FMUL.FTZ R48, R58, R49` | 840 | 1 | none |
| 0x11920 | `FMUL.FTZ R115, R56, R115` | 720 | 1 | none |
| (14 more FFMA/FMUL PCs) | — | 540-690 each | 1-3 | none |

Classified all 43 SP_OP PCs visible in the top-60 printed rows (covering 27,027 of the bucket's 78,893 total
stall events) against the static instruction JSON: **41 of 43 are `FFMA.FTZ`/`FMUL.FTZ`; only 2 are
`IMAD`/`IMAD.WIDE`.** This conclusively resolves the §23.22b open question: **the dequant-accumulate epilogue
(#1) dominates the `SP_OP/scoreboard` stall bucket by every PC that matters except the single `0xdb70` outlier**
(which genuinely does have a real-hardware wait-barrier, i.e. is a legitimate cost, not over-modeling). §23.22b's
IMAD.WIDE address-chain hypothesis doesn't even appear in the top-stalling PCs — it was a red herring produced by
looking at raw occurrence counts rather than actual stall attribution.

**Net conclusion for this whole investigation thread**: the dominant remaining lever, if pursued further, is
narrowing this simulator's SP_OP dependent-latency model specifically for the FFMA/FMUL dequant-accumulate
pattern — real hardware pays almost nothing for it (confirmed twice now: §23.22's aggregate average and this
section's PC-exact cross-check), our simulator pays the most of anything in the kernel for it. Not implemented
this session (would mean changing the 4-cycle EX latency or the scoreboard-release timing specifically for this
op pattern, a more invasive change than the bounded, reversible fixes already shipped today).

**Code change**: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc` — `PcStallHistogramState` struct +
`pc_stall_histogram_record()`/`pc_stall_histogram_print_body()` (mirrors the existing per-op-type histogram),
wired into the same two call sites as `issue_stall_histogram_record()` and the same two print sites as
`issue_stall_histogram_print_body()`/`starvation_histogram_print_body()`. Gated by the existing
`-subcore_issue_debug 1` flag, no new CLI flag needed. Output: `[pc_stall_stats]`, sorted by stall count
descending, top 60 PCs, each showing op type, total stall count, issued count, and a reason breakdown.

### 23.25 FOURTH LEVER, small: early-forward bypass path for SP_OP only (real H100's ~1.2-cycle effective dependent latency)

Acted on §23.24's identified next step: narrow the SP_OP dependent-latency model specifically for the FFMA/FMUL
dequant-accumulate pattern. Tried reducing `-ptx_opcode_latency_fp`'s MUL/MAD entries directly first — already
known to be a dead end (`lat=3` vs `lat=4` was byte-identical, confirmed earlier this session, §23.16's
neighborhood) — so didn't repeat that. Instead built a real, scoped microarchitecture extension to the existing
bypass network (§23.17): a second "early-forward" recording path, specifically for `SP__OP` (FP32 ADD/MUL/MAD)
only, that marks an instruction's destination registers forwardable **at dispatch into the FU** (the moment it
leaves the issue stage), rather than waiting for the existing path's EX-finish trigger. The existing late path
forwards only *after* the full EX latency elapses — useless for closing the literal 4-cycle RAW wait itself, only
the EX-finish-to-WB tail (which is all §23.17-23.18 could ever target). This new path can shorten the wait itself.

**Design**: `Scoreboard::recordBypassWrite()` gained an optional `valid_from_cycle` parameter (default 0 = always
valid, preserving the old call site's behavior exactly); `m_bypass_forward_table`'s value type changed from a
single `expire_cycle` to a `{valid_from_cycle, expire_cycle}` pair, and the bypass-availability check in
`checkCollision_remodeling_with_bypass` now requires `cur_cycle` to be within both bounds.
`SM::maybe_record_register_bypass_early()` (new, `sm.cc`) is called from `functional_unit::issue()` at dispatch
time, gated on `op_pipe == SP__OP`; it records `valid_from = dispatch_cycle + register_early_forward_delay_cycles`,
`expire = valid_from + register_bypass_window_cycles` (reuses the existing window setting). New flags:
`-register_early_forward_sp_op_enabled` (default 0/off), `-register_early_forward_delay_cycles` (default 1).

**Swept `register_early_forward_delay_cycles` × `register_bypass_ports_per_subcore`** (isolated 1-SM/1-CTA, on top
of the persisted window=4/ports=1 best config):

| Delay | Ports | Cycles | Δ vs 184,384 baseline |
|---|---|---|---|
| 1 | 1 | 185,087 | **+0.38% (worse)** |
| 2 | 1 | 184,285 | −0.054% |
| **3** | **1** | 184,242 | −0.077% |
| 4 | 1 | — (not separately run, ≈baseline expected) | — |
| 1 | 2 | 184,683 | still worse than baseline |
| 2 | 2 | 184,592 | −0.054% |
| **3** | **2** | **184,130** | **−0.138% (best found)** |
| 3 | 4 / 8 | 184,130 (identical) | saturates at ports=2 |
| 4 | 2 | 184,624 | worse than delay=3 |

Non-monotonic in both knobs, same as §23.18's bypass-window sweep — delay=1 actively hurts (more bypass attempts
contend for the single port per subcore per cycle, perturbing which warp the greedy scheduler picks that cycle,
the same second-order sensitivity already documented). delay=3/ports=2 is the local optimum; saturates beyond
ports=2 (no further gain at 4 or 8), confirming it's not simply port-starved at that point.

**Full-chip validation** (`k3_fullchip_early_forward_d3p2.log`, `-register_early_forward_sp_op_enabled 1
-register_early_forward_delay_cycles 3 -register_bypass_ports_per_subcore 2`): **208,119 cycles** vs the
208,191-cycle baseline (−0.035%). Instruction counts identical to every other full-chip run this session
(`gpu_sim_insn=1,170,378,752`, `gpgpu_n_tot_w_icount=36,715,744`) — functionally clean, no correctness concern.

**Cumulative progress vs real H100, updated**: 216,392 (L2-normal baseline, +30.2%) → 210,062 (+WAR disabled,
+26.35%) → 208,650 (+bypass w2/p1, +25.51%) → 208,191 (+bypass w4/p1, +25.23%) → **208,119 (+early-forward
d3/p2, +25.19%)**.

**Honest assessment**: real, validated, correctness-clean — but small (0.035% full-chip), much smaller than the
isolated test suggested (−0.138%), the same isolated-vs-full-chip damping pattern seen with the original bypass
network's window/port sweep. The scheduler-sensitivity ceiling documented since §23.3/§23.18 appears to be the
real limiter here, not bypass-port or window capacity (saturates at ports=2) — pushing this lever further (e.g.
extending early-forward to other op-pipe classes) is likely to keep hitting the same wall rather than compound.
**Persisted** to `SM90_H100_l2norm_l1dnorm/gpgpusim.config`: `-register_bypass_ports_per_subcore` raised 1→2,
plus the two new flags (`-register_early_forward_sp_op_enabled 1 -register_early_forward_delay_cycles 3`).

**Code change**: `scoreboard.h`/`scoreboard.cc` (`recordBypassWrite` gains `valid_from_cycle`, bypass table value
type becomes a pair), `sm.h`/`sm.cc` (`maybe_record_register_bypass_early`), `functional_unit.cc` (call site in
`issue()`), `shader.h`/`gpu-sim.cc` (two new CLI flags).

### 23.26 Per-instruction `[commit_trace]` duration confirms §23.23's write-epilogue contention at the instruction level

User asked why specific `STORE_OP` commits in `k3_all_debug.log` took 600-700+ cycles from issue to commit (e.g.
`pc=0x181c0 issue_cycle=183159 commit_cycle=183800`, duration 641). Two things done:

1. **Added a `duration` field directly to `[commit_trace]`** (`sm.cc::instruction_retirement`) — was previously
   issue_cycle/commit_cycle only, requiring manual subtraction. Now prints `duration=<commit-issue>` inline.
2. **Checked what these specific slow commits actually are.** Cross-referenced the hot PCs against the static
   instruction JSON: `pc=0x181c0` etc. (the example in the question) are `STG.E desc[UR10][R6.64+0x20] R174` —
   **the global output epilogue writes**, exactly the `GLOBAL_ACC_W` writes already quantified in §23.23's
   `RESERVATION_FAIL`-based analysis. Computed the full duration distribution for all 9,160 `STORE_OP` commits in
   the run: **p50=40 cyc, p90=99 cyc, but p99=616 cyc** — a sharp bimodal split. Binning by cycle confirms it:
   75% of all >200-cycle-duration stores (420 of 559) land in the 180,000-190,000 cycle bucket, i.e. the kernel's
   final ~4,000 cycles — the simultaneous 8-warp output burst at kernel end, same mechanism §23.23 already found
   via aggregate `RESERVATION_FAIL` counts (6,345 fails / 2,048 successful writes). This is the per-instruction,
   ground-truth confirmation of that same finding, not a new cost.
3. **Smaller, separate observation**: a handful of `STORE_OP` commits scattered throughout the *middle* of the
   kernel (8-22 per 10k-cycle bucket, much rarer than the end-of-kernel spike) also show 200-700+ cycle durations.
   Checked their PCs: these are `STS` (shared-memory stores, e.g. `pc=0xfb00: STS [R20+0x4a00], R55`) — the
   K-loop's IMMA-result shared-memory staging, a structurally different contention path (shared-memory port/bank
   pressure, not the L1D `RESERVATION_FAIL` mechanism §23.23 measured for global stores). Not investigated further
   this session — much smaller magnitude (a few dozen instances total vs. hundreds at the epilogue), and a
   different resource than the one §23.23 already quantified.

### 23.27 RULED OUT: decoupling fetch priority from issue priority makes things worse, not better

Direct follow-up to §23.2's addendum (warp1 starved on fetch for 68 cycles by its subcore-mate warp5 monopolizing
the shared fetch port). Asked: if fetch alternated independently instead of being re-synced to the issue
priority pointer every cycle (`Subcore::cycle()`: `m_greedy_pointer_fetch = m_greedy_pointer_issue;`), would that
help? Built and tested it rather than guessing.

**Design**: new flag `-is_subcore_fetch_round_robin_independent` (default 0/off, preserves exact baseline
behavior). When enabled: `Subcore::cycle()` skips the fetch-pointer resync; instead `Subcore::fetch()` advances
`m_greedy_pointer_fetch` to `(subcore_warp_id+1) % num_warps` round-robin after every fetch attempt, independent
of whatever the issue scheduler is doing. Issue itself is untouched — still greedy-then-highest-id, still rides
the same warp until it stalls.

**Result** (isolated 1-SM/1-CTA, on top of the persisted best config): **184,775 cycles vs 184,130 baseline
(+0.35%, worse)**. `gpu_sim_insn` identical (8,333,696) — correctness-clean, just slower.

**Verdict**: ruled out, same lesson as §23.3's true-round-robin-issue test (+3.6% worse) and consistent with the
broader theme this session (§23.18, §23.25) that this scheduler's greedy "stick with whoever's making progress"
coupling is net beneficial despite causing visible, real, individually-diagnosable starvation in specific spots.
Un-starving the losing warp costs more (in lost momentum for the warp that *was* making progress) than it
recovers. Not persisted; flag left available (default off) for any future, more targeted variant (e.g. only
decouple when the greedy warp's own IBuffer is near-full, rather than every cycle unconditionally).

**Code change**: `shader.h`/`gpu-sim.cc` (new flag), `subcore.cc` (`Subcore::cycle()` conditional resync skip,
`Subcore::fetch()` independent round-robin advance after the existing per-cycle fetch attempt's `break`).

### 23.28 `[commit_trace]` gained live operand annotation + duration field; bulk duration-distribution sanity check

Two small, purely-observational additions to `sm.cc::instruction_retirement`'s `[commit_trace]` print (no cycle-
count change, validated: 184,130 isolated, instruction counts unchanged):
1. **`duration=<commit_cycle-issue_cycle>`** field, computed inline.
2. **`opcode=...` / `operands=[...]`** fields, pulled live from `get_extra_trace_instruction_info()` (the same
   static per-instruction info `checkCollision_remodeling_with_bypass`/the bypass network already use at issue
   time) — each operand tagged `(dst)`/`(src)` by index vs `get_num_destination_registers()`. No new guard
   needed beyond the existing `-subcore_issue_debug` gate; earlier attempt mistakenly gated this on
   `is_captured_from_binary` (which actually means "has real SASS control-bits," confirmed false/unused for this
   trace per §23.22's first finding) and printed empty fields until that guard was removed.

**Built `log/tmp_log/analyze_commit_trace.py`** to bulk-check whether the resulting per-instruction durations are
behaving the way they should — i.e. fixed for ALU pipes, and *properly dynamic* (cache-hit vs DRAM-miss bimodal)
for memory ops — rather than spot-checking a handful of lines. Run against an isolated single-SM/single-CTA
`-subcore_issue_debug 1` capture (`k3_commit_trace_annotated2_subcore0.log`, SM0/subcore0 only, 65,198 lines
parsed):

**Compute pipes — correctly near-static, as they should be (fixed-latency units, no data-dependent timing):**

| op | opcodes | distinct durations | range |
|---|---|---|---|
| `SP_OP` | `FFMA.FTZ`, `FMUL.FTZ`, `IMAD*` | 1-2 | 10 (occasionally 11) |
| `INTP_OP` | `IADD3`, `LEA`, `PRMT`, etc. | 1 each | fixed per opcode (10, or 19 for `ISETP.*`) |
| `UNIFORM_OP` | `UIMAD`, `UIADD3`, etc. | 1 | fixed at 10 |
| `TENSOR_CORE_OP` | `IMMA.16832.S8.S8` | **1 (exactly 14, zero variance across all 3,840 instances)** |
| `SFU_OP` | `I2FP.F32.S32`, `MUFU.RCP` | 9 (mild) | 18-26 |

`TENSOR_CORE_OP`'s perfect zero-variance is notable on its own but not new evidence of a bug — it's the same
conclusion already reached the hard way in §23.16 (IMMA init/dependent-latency decouple tested twice, no effect):
the tensor pipe genuinely never experiences contention in this kernel. `SFU_OP`'s mild 18-26 spread makes sense
too — unlike the fixed-latency ALU pipes, SFU is a queued unit (`is_fixed_latency_unit()` false), so it can show
real queueing delay when multiple warps' conversions compete for it.

**Memory ops — strongly, *appropriately* bimodal (this is what should happen with a real cache hierarchy):**

| opcode | n | p50 | mean | distribution |
|---|---|---|---|---|
| `LDS` (shared mem load) | 3,840 | 22 | 27.7 | 92% in 20-50 cyc (hit), long tail to 184 (bank-conflict/contention) |
| `LDG.E.U16.CONSTANT` (global, dequant-scale table) | 2,040 | 342 | 357.2 | **zero instances under 200 cyc** — every single one takes the long DRAM-class path |
| `LDG.E.CONSTANT` | 1,080 | 380 | 380.2 | same pattern, ~95% in 300-500 |
| `LDSM.16.M88.4` (shared→tensor-core feed) | 480 | 38 | 50.0 | mostly fast, contention tail to 192 |
| `LDC`/`LDC.64` (kernel-param constant loads) | 22 | mixed | mixed | 10/22 fast (20-50), rest at 500+ — consistent with cold first-touch per warp at kernel start, not flagged as a bug |
| `STS` (shared mem store) | 2,162 | 39 | 52.6 | 69% in 20-50, real tail to 275 — matches §23.2's addendum (fetch/issue-port-sharing stalls between subcore-mate warps, not memory contention per se) |
| `STG.E` (global output write) | 128 | 478 | 419.0 | **bimodal**: a handful fast (first writers), ~48% in 500+ (the simultaneous end-of-kernel burst) — direct bulk confirmation of §23.23's `RESERVATION_FAIL` finding |

**Verdict: the dynamicism looks correct, not suspicious.** Every memory opcode's slow tail (300-700 cyc) lines up
with the real DRAM round-trip class already measured directly in §23.20 (242 cyc DRAM-only span, plus realistic
pipeline/queueing overhead to get from "DRAM responds" to "instruction commits"), and every memory opcode's fast
cluster (20-50 cyc) matches L1D/shared-memory hit latency. The one mildly interesting observation for a future
session, not chased further here: `LDG.E.*.CONSTANT` (the dequant-scale-table loads) **never** hit cache at all in
this kernel — every one of 3,120 sampled instances takes the long path. Plausibly just a streamed, low-reuse
access pattern (each output tile reads its own scale values once), but worth a second look if anyone later wants
to check whether there's real, missed reuse across CTAs/warps that L2-normal's simplifications (§20.2/§22) are
masking or destroying.

### 23.29 Correction: prior claim about real-hardware L1D/L2 prefetching was overstated

User pushed back on an earlier answer claiming real NVIDIA GPUs have no hardware data-prefetch mechanism near
L1D/L2 "by deliberate design, to avoid competing with real traffic for bandwidth/MSHRs." That causal explanation
was my own unsourced speculation, and the blanket "no mechanism at all" framing was wrong. Checked via web search:

- A reverse-engineered **"tree-based neighborhood prefetcher"** does exist (Ganguly et al.), operating at the
  Unified-Memory/page-migration level (2MB chunks / 64KB basic blocks) — triggered by page faults under managed
  memory, not a per-access cache-line prefetcher for ordinary `cudaMalloc` buffers.
- **Hopper added `cp.async.bulk.prefetch.L2`** — a real, hardware-supported PTX instruction (Compute Capability
  9.0+) for explicitly prefetching into L2. Hardware-*assisted*, but software-*triggered* (the kernel must issue
  it) — not an automatic, always-on stride detector.

What still holds: no public evidence of a CPU-style automatic stride-predicting hardware prefetcher continuously
watching L1D/L2 access patterns for ordinary compute kernels. For k3 specifically this doesn't change anything
practical — it uses plain `cudaMalloc` (not managed memory, so the UVM neighborhood prefetcher is irrelevant) and
its static opcode list (confirmed via the JSON, §23.27's analysis script and others) contains zero
`cp.async.bulk.prefetch.*` or any other prefetch instruction. This simulator models none of these mechanisms
either way (§ on `-is_instruction_prefetching_enabled`: instruction-cache-only, off in our config). Recorded here
purely so a future session doesn't repeat the overstated framing as settled fact.

### 23.30 Checked: is `LDG.E.*.CONSTANT` reuse actually missed across CTAs? No — it's captured at L2, just not at L1

Follow-up to §23.28's flagged observation ("every `LDG.E.*.CONSTANT` load in the isolated single-CTA test takes
the long path, zero hits"). Checked whether that's a real missed-reuse problem across the kernel's other 131
CTAs, or an artifact of testing with only one CTA.

**First, the correct classification**: `LDG.E.CONSTANT`/`LDG.E.U16.CONSTANT` (".nc"/read-only-global loads) map to
`TEXTURE_ACC_R` in this simulator's access-type enum (`abstract_hardware_model.cc`: `tex_space` → L1TEX path),
**not** `CONST_ACC_R` (that's the separate, small constant-bank `LDC c[0x0][...]` kernel-parameter path, which
behaves completely differently — see below).

**Full-chip cache breakdown for `TEXTURE_ACC_R`** (`k3_fullchip_early_forward_d3p2.log`, all 132 CTAs):

| Level | HIT | MISS | TOTAL_ACCESS | Hit rate |
|---|---|---|---|---|
| L1 (per-SM, private) | 0 | 1,169,859 | 3,407,872 | **0%** |
| L2 (chip-wide, shared) | 929,400 | 157,696 | 1,169,859 | **79.4%** |

**Verdict: real reuse exists and is being captured — just at L2, not L1.** k3 launches exactly 132 CTAs, one per
SM (confirmed: "132 bind to kernel" in the full-chip log) — so no single SM's private L1 ever sees a second CTA
of this kernel to reuse against; 0% L1 hit is expected, not a bug. But L2 is chip-wide and shared, and 79.4% of
`TEXTURE_ACC_R` accesses hit there — the classic GEMM weight/dequant-scale-table sharing pattern: many of the 132
concurrently-running CTAs read overlapping regions of the quantized-weight table, and most of that reuse is
caught by L2.

**This reconciles, not contradicts, §23.28's isolated-test finding.** The isolated 1-SM/1-CTA test runs exactly
one CTA by construction, so it can never observe L2 reuse from a sibling CTA — it was specifically measuring the
unavoidable "first CTA to touch this address" cold-start cost, which really is ~100% of *that* test's accesses,
without that being representative of the full kernel. In the full-chip run, CTAs 2 through 132 get to benefit
from whichever CTA happened to touch a given address first.

**Not a lever**: the remaining 20.6% L2-miss cost is the genuine, unavoidable first-touch tax spread across the
whole kernel (someone has to be first for every address) — already getting nearly 4-in-5 reuse. No evidence of a
real missed-reuse problem to chase here.

**Method note**: used `log/tmp_log/analyze_commit_trace.py` (§23.28) plus a direct grep of
`Total_core_cache_stats_breakdown`/`L2_cache_stats_breakdown` for the correct access-type key — the first attempt
checked `CONST_ACC_R` (wrong access type for these PCs) before correctly identifying `TEXTURE_ACC_R` via
`abstract_hardware_model.cc`'s `generate_mem_accesses()`/`tex_space` mapping.

### 23.31 Real H100 L2 hit rate confirms §23.30's finding — strong match, not a coincidence

User re-profiled the real kernel on the BSC cluster with `ncu --section MemoryWorkloadAnalysis` added (the
command drafted together this session, with a `^mul_mat_q(?!_stream_k_fixup)` regex to exclude the unrelated
stream-k-fixup epilogue kernel — needed because ncu's `--kernel-name regex:` appears to match against the
*mangled* symbol, which has no literal `<` character at all; a plain `mul_mat_q<` pattern returned zero rows for
that reason). Result: `h100_real_profile/k3_occupancy_and_l2_hitrate_filter.csv`, **498 real kernel launches**
captured (the same compiled `mul_mat_q<Q8_0,128,false>` kernel recurs across the model's many linear layers
during prefill — same compiled code, same launch config `(132,1,1)x(32,8,1)` every time, only the underlying
weight-matrix data differs per call).

**Real H100 `lts__t_sector_hit_rate.pct` (L2 Hit Rate) across all 498 launches**: mean=73.09%, median=79.77%,
range 53.62%-82.51% — clearly **bimodal**, most launches cluster at ~79-82%, with a recurring smaller subset
dropping to ~54% (visible directly in the raw per-launch sequence: `81.17, 80.58, 79.77, 81.09, 54.82, 54.4,
78.82, ...` — a periodic dip, plausibly a specific layer-type, e.g. attention vs FFN projection, with worse
weight-matrix reuse at that K-dimension). `l1tex__t_sector_hit_rate.pct` (L1/TEX Hit Rate) is tight and stable:
52.48%-53.21% across all 498 launches — but this is a *unit-wide* metric (L1TEX is the same physical unit serving
both global-memory caching and shared-memory accesses on real GPUs), so it isn't directly comparable to this
simulator's `TEXTURE_ACC_R`-only L1 stat (§23.30's 0%) without conflating in the highly-reused shared-memory
traffic — not pursued further, flagged so it isn't mistaken for a contradiction later.

**Our simulator's aggregate L2 hit rate** (full-chip, summed across `GLOBAL_ACC_W` + `TEXTURE_ACC_R` +
`CONST_ACC_R`, the apples-to-apples equivalent of ncu's unit-wide `lts__t_sector_hit_rate.pct`):
`(380,928 + 929,400 + 512) / (507,904 + 1,169,859 + 528)` = **78.1%**.

**Verdict: strong match.** 78.1% lands almost exactly on real hardware's *dominant mode* (~79-82%, where most of
the 498 launches sit) and well inside the real range. Can't confirm an *exact* single-launch match (no way to
tell which of the 498 differently-shaped real calls corresponds to our one captured trace), so this is a strong
directional/order-of-magnitude validation of the L2-normal cache model's reuse behavior for this kernel, not a
verified one-to-one number match — but it's a real, independent confirmation of §23.30's finding that cross-CTA
L2 reuse is substantial and is being captured correctly, not missed.

### 23.32 Roofline: k3's arithmetic intensity sits just above the ridge point, yet hits neither ceiling — latency-bound, not throughput-bound

Built from numbers already measured this session, not new profiling. H100 SXM5 peak specs confirmed via NVIDIA's
own datasheet (web search, to avoid repeating §23.29's mistake of asserting an unverified hardware claim):
**1,979 TOPS dense INT8** (3,958 with 2:4 sparsity), **3,350 GB/s HBM3**.

**Work done (same for sim and real — identical instruction stream):**
- §23.28's single-subcore (2-warp) commit-trace capture measured **3,840 dynamic `IMMA.16832.S8.S8` instances** →
  1,920/warp → ×8 warps/CTA (12.48% sim occupancy × 64 max warps/SM = 8 active warps = exactly 1 CTA's worth;
  matches real hardware's `Achieved Occupancy` = 12.49% median almost exactly, §23.31's CSV) → 15,360 IMMA/CTA ×
  132 CTAs (grid=132=#SMs, uniform tile work, no remainder tiles) = **2,027,520 dynamic IMMA instructions, full
  kernel**.
- `IMMA.16832` = PTX `mma.m16n8k32` INT8 → ops/instr = 2×16×8×32 = **8,192** (NVIDIA's own TOPS convention) →
  **16.61 GOp total work**.

**Time and bandwidth:**
- Real: 166,246 cycles @ 1.62 GHz = **102.62 μs** (independently reproduces the documented 102,621 ns real
  target — confirms 1.62 GHz, not the baseline config's 1.83 GHz, is the right clock for this comparison).
  `Memory Throughput` (ncu, §23.31's CSV, 498 launches, median) = 243.5 GB/s → bytes moved = 243.5 GB/s ×
  102.62 μs ≈ **25.0 MB**.
- Sim (current best, §23.25): 208,119 cycles @ 1.62 GHz = **128.47 μs**. `DRAM_BW_total` = 314.25 GB/s → bytes
  moved ≈ **40.4 MB** (sim moves more bytes in less cycle-time than real, consistent with its higher overall
  miss/traffic findings elsewhere in this doc).

| | Real H100 | Our sim |
|---|---|---|
| Achieved TOPS | 16.61e9/102.62µs = **161.9 TOPS** | 16.61e9/128.47µs = **129.3 TOPS** |
| % of peak (1979 TOPS) | **8.2%** | **6.5%** |
| Achieved BW | 243.5 GB/s | 314.25 GB/s |
| % of peak (3350 GB/s) | **7.3%** | **9.4%** |
| Arithmetic intensity | 16.61e9/25.0e6 B ≈ **665 ops/byte** | (same work, different byte count — order-of-magnitude similar) |
| Ridge point (AI\* = 1979e12/3350e9) | **591 ops/byte** (HW-fixed, same for both) | |

**Finding**: k3's AI (~665 ops/byte) sits just *above* the ridge point (591) — by naive roofline that nominally
places it on the compute-bound side. But real hardware achieves only 8.2% of peak compute *and* only 7.3% of peak
bandwidth simultaneously — nowhere near either ceiling. That is the signature of a kernel sitting deep inside the
roof, not pinned to either edge: **latency/occupancy-bound, not throughput-bound by compute or memory**. Consistent
with everything else found this session/prior sessions: 12.5% occupancy (real and sim agree almost exactly),
dependency-chain/scoreboard stalls dominating real hardware's own per-warp stall breakdown (item 11: `wait`+
`long_scoreboard` = 31.3%, `math_pipe_throttle` only 5.07%), and every pure-capacity/bandwidth lever tested this
session (MSHR width, write-queue width §item 10, L1D/L2 cache-model normalization) failing to move the cycle gap,
while issue/dependency-latency-shaped levers (early-forward bypass, §23.25) did. Roofline analysis by itself does
not explain k3's sim-vs-real gap — the gap is a scheduling/latency story, not a peak-resource story.

**Caveats**: IMMA dynamic count is extrapolated from one captured subcore/CTA to the full 132-CTA chip (reasonable
given uniform grid/tile shape, not independently re-measured per-CTA). Real bytes-moved figure derives from
`Memory Throughput` × the *documented* real duration (102,621 ns), not a duration column in the same CSV (this
profiling pass collected occupancy/hit-rate/throughput metrics, not `gpu__time_duration.sum`).

### 23.33 General method: estimating peak FLOPS/TOPS and peak bandwidth analytically (no profiling needed)

Recorded for reuse on future chips/configs without re-running anything — both peaks are pure algebra from
published or back-solved hardware constants; the two come from independent parts of the chip and need different
inputs.

**Compute peak:**

```
Peak_ops/s = num_SMs × MMA_ops_retired_per_SM_per_cycle × clock_Hz
```

where `MMA_ops_retired_per_SM_per_cycle = (ops per MMA instruction) × (MMA instructions/cycle/SM)`, and
`ops per MMA instr = 2×M×N×K` for whatever shape the instruction encodes (e.g. `IMMA.16832` = PTX
`mma.m16n8k32` → 2×16×8×32 = **8,192 ops/instr** — NVIDIA's own TOPS-counting convention, mul+add as one pair).

The one factor not derivable from instruction encoding + SM count alone is the **throughput rate** (MMAs/cycle/SM
the tensor cores can actually retire) — a microarchitectural fact. Get it either from a vendor whitepaper, or by
**back-solving from one published peak number**: `1979 TOPS / (132 SMs × 1.83 GHz) = 8192.0 ops/cycle/SM` for
H100 — exactly one 8,192-op IMMA per SM per cycle, clean enough to not be a coincidence. This also explains why
`SM90_H100`'s config uses a 1.83 GHz core clock: almost certainly chosen so the math reproduces NVIDIA's published
spec exactly. Once this per-SM-per-cycle constant is known for a chip, it can be reused to estimate peak for other
MMA shapes/precisions on that same chip with zero additional profiling.

**Bandwidth peak — different inputs, SM/core count is irrelevant:**

```
Peak_BW(bytes/s) = (bus_width_bits / 8) × pin_transfer_rate_Hz
```

H100 SXM5: 5 active HBM3 stacks × 1024-bit/stack = 5,120-bit bus (confirmed via web search). Back-solving from the
published 3,350 GB/s: `3350e9 × 8 / 5120 ≈ 5.23 Gb/s/pin` (HBM3 is rated up to 6.4 Gb/s/pin, so H100 runs it
binned down to land on exactly 3.35 TB/s). Needs memory-subsystem facts (stack count, per-stack bus width, pin
rate) — IMMA shape and SM count don't enter this formula at all.

**Why this matters for roofline**: compute peak scales with SM/tensor-core count and clock; bandwidth peak scales
with HBM stack count and pin rate — independent budgets set independently by the chip designers. Both peaks (and
therefore the ridge point `AI* = peak_ops/peak_bytes`, §23.32's 591 ops/byte) are obtainable with **zero
profiling**. What math alone cannot tell you is whether a *specific kernel* actually reaches either ceiling — that
remains an occupancy/scheduling/latency question requiring real measurement, which is exactly why k3 needed
profiling to reveal it sits at ~8% of *both* peaks despite an arithmetic intensity that nominally classifies it as
compute-bound (§23.32).

Sources: [NVIDIA H100 Datasheet](https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf),
[NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/),
[TechPowerUp H100 SXM5 80GB Specs](https://www.techpowerup.com/gpu-specs/h100-sxm5-80-gb.c3900).

### 23.23 Output-write epilogue: real L1D reservation contention, but a small (~1.24%) tail cost

User noticed varying `GLOBAL_ACC_W` durations (234-422 cyc) in `[mem_request_trace]` output and asked whether
writes reserve a resource and whether the variance is contention. Measured directly (`k3_all_debug.log`, isolated
best config):

- All 2,048 `GLOBAL_ACC_W` writes have `dram=0` — every one resolves inside L2, none ever reach DRAM (output tile
  small enough that writes write-allocate into L2 without evicting to DRAM within the kernel's lifetime).
- `avg dur = 288.6 cyc` (min 234, max 422) — real variance, not noise: confirmed by
  `Total_core_cache_stats_breakdown[GLOBAL_ACC_W][RESERVATION_FAIL] = 6345` against only 2,048 successful writes —
  each write averages ~3 failed reservation attempts (an MSHR/pending-request-table slot at the L1D/core level)
  before succeeding, because all 8 warps hit their write-out epilogue in a tight simultaneous burst at kernel end.
- **Tail-cost measurement**: first write `send=182,094`, last write `resp=184,381`, kernel ends at
  `gpu_tot_sim_cycle=184,384` (only 3 cycles of slack after the last write resolves) — the write-out epilogue is
  the literal last thing in the kernel, with nothing else to overlap/hide it. Tail span
  `184,384 − 182,094 = 2,290 cycles` = **~1.24% of total kernel time**.

**Verdict**: real, measurable contention (confirmed via reservation-fail counts), and it does measurably extend
the kernel's tail — but at ~1.24% it's smaller than the barrier-stall cost (§23.5, 4.86%) and far below
`SP_OP/scoreboard` (§23.19, 44% of stall events) — a small, real, now-quantified contributor, not a lever worth
chasing on its own.

### 23.15 Code changes this session, updated (uncommitted — see `git status`)

| File | Change |
|------|--------|
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h` | `issue_wait_trace_debug` member |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc` | `-issue_wait_trace_debug` CLI registration |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc` | `[issue_wait_trace]` live per-cycle print (issued and stalled branches), with RAW/WAR collision-register lookup on scoreboard stalls |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc` | Commit-progress heartbeat now always-on (not gated by `-subcore_issue_debug`), interval 1,000→10,000 |

---

*Last updated: 2026-06-20 — added §23.14/23.15: found that 33.2% of all kernel stall events are pure WAR
(anti-dependency) scoreboard hazards; `-scoreboard_war_mode disabled` gives a real, validated −3.21%
(isolated)/−2.93% (full-chip) cycle reduction, narrowing the gap to real H100 from +30.2% to +26.35% — the first
genuinely positive lever found this investigation. Not yet persisted to config or correctness-checked.*

*Last updated: 2026-06-20 (Cursor) — added §22: L2-normal isolated experiment (−12.1% vs sectored 218,024→191,627
cyc), `gpu-cache.cc` fill-path fix for L2-N + sectored-upstream mismatch, combined sync/mem trace analysis
confirming ~750–1000 cyc/round warp spread at `pc=0x10020` is inter-barrier compute skew (not memory deps at
sync, not sectored-L2 stagger alone), plus `[sync_issue_trace]` instrumentation.*

*Earlier update, 2026-06-20 — added §13 (methodology gap, then retracted as a kernel-overlap bookkeeping artifact),
§14 (root-caused the barrier-correlated starvation to per-warp memory-latency variance in the global-load
stage at `pc=0x10020`, tied to §12's interconnect tail-latency finding), and §16.5 (direct per-partition queue-depth
measurement refuted §16.4's "chronic backlog" — it was a `0/*FIXME*/` clock-reset bug in `L2interface::push()`,
not a real phenomenon; the interconnect/DRAM-admission congestion sub-thread is now exhausted). `tensor_decouple`
isolated-fix result: 265,078 cycles, -0.11% vs baseline — a no-op, consistent with the bundled depfix result.*
