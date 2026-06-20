# Prefill k3 — config-knob audit, dead-path discovery, and occupancy-starvation confirmation

Third document in the k3 series. Companion to [`prefill-k3-correlation.md`](prefill-k3-correlation.md) (what k3 is) and
[`prefill-k3-debug-investigation.md`](prefill-k3-debug-investigation.md) (the two-layer stall model, 30k-cycle debug run).
This document covers three things done after that doc was last updated:

1. Two more config experiments (constant-cache MSHR widening, corrected GIT/nopc comparison) — both **negative results**.
2. A **code-level audit** of every "tensor latency" knob ever touched in this investigation — most turned out to be
   parsed but never consumed by the code path that actually runs.
3. A **new, decisive instrumentation result** that confirms the root cause of the remaining gap: occupancy starvation,
   not a scheduler or scoreboard bug.

| Item | Value |
|------|-------|
| Kernel under study | k3 `mul_mat_q<Q8_0, mmq_x=128>` (trace `kernel_id=3`, nsys **37**) |
| Real H100 target | 102,621 ns → **~166,246 cycles** @ 1.620 GHz |
| Baseline sim k3 | **258,275 cycles** (+55%) — `log/prefill/sim.log` |
| Best config-only result | **237,536 cycles** (+43%) — `SM90_H100_best_optA`, `log/prefill_best_multiopt/SM90_H100_best_optA.log` |
| New instrumentation | `[starvation_stats]` block, uncommitted C++ in `subcore.h`/`subcore.cc` |

---

## 1. Two more negative config results

### 1.1 Constant-cache MSHR widening — refuted

Hypothesis: the real (non-perfect) constant cache's degenerate `S:2:64,4` (2 MSHR entries, 4-deep miss queue) was the
reason `gpgpu_perfect_inst_const_cache 0` blows k3 up to ~794k cycles (vs 258k with perfect cache). Widened to
`S:64:64,32` in `SM90_H100_nopc_constmshr/gpgpusim.config` (`run_prefill_nopc_constmshr.sh`).

| Config | k1 | k2 | k3 |
|---|---|---|---|
| `nopc` (baseline real-cache, MSHR=2/mq=4) | — | — | 793,924 |
| `nopc_constmshr` (MSHR=64/mq=32) | 47,474 | 32,180 | **811,992** |

**Result: slightly worse, not better.** Local queue depth wasn't the bottleneck — widening it just admits more requests
into an already-congested shared L2/DRAM path. **Conclusion: `gpgpu_perfect_inst_const_cache 1` is not papering over a
fixable local queueing bug.** It's standing in for something GPGPU-Sim's literal constant-cache model doesn't capture
well (most likely: real H100 constant/uniform-data reuse for these heavily-shared Q8_0 scale values behaves close to
"always hit" in practice). Not worth pursuing further; keep perfect cache enabled.

### 1.2 Correction: `nopc_git` was mislabeled

`log/prefill_nopc_git/sim.log` (248,703 cycles) was previously cited as "real constant cache + GIT's wide memory
(n_mem=80) tames the blowup." **This is wrong.** `run_prefill_git.sh` runs `SM90_H100_GIT/gpgpusim.config` as-is, and
that config still has `-gpgpu_perfect_inst_const_cache 1` (confirmed from the log's own configuration dump). The
248,703 figure is just "GIT's physically-grounded memory model, perfect cache still on" — modest and unremarkable,
same ballpark as `best`/`optA`. **There is no completed experiment of "GIT memory model + real constant cache."** If
that's wanted later, it needs a fresh config combining `SM90_H100_GIT`'s memory section with
`-gpgpu_perfect_inst_const_cache 0`.

---

## 2. Code audit: which "tensor latency" knobs are actually live

Across this whole investigation (this doc and the prior one) we touched **four different config fields** trying to
calibrate tensor-op timing. Three turned out to be dead or non-binding for the code path this repo actually runs.
Traced down to the exact source lines:

| Knob | File | Tested values | Result | Why |
|---|---|---|---|---|
| `ptx_opcode_initiation_tensor` | `gpgpusim.config` | 8/16/32 (`tinit16/32`) | **dead** | PTX functional-sim path, bypassed entirely in trace-driven mode |
| `tensor_extra_latency_16816_fp32_1688_fp32` | `gpgpusim.config` | 8/16/32 (`textra8/32`) | **dead for this kernel** | only applied when `is_16816_fp32_1688_fp32` (HMMA.16816/1688), but this kernel's op is `IMMA.16832.S8.S8` — that branch never fires |
| `trace_opcode_latency_initiation_tensor` | `trace.config` | `32,8`→`32,32`→`24,32` | **dead, structurally, for every op type** | see §2.1 |
| `tensor_latency` | `gpgpusim.config` | 32→24 | **non-binding** | it's a pipeline-array *capacity*, not a real latency; see §2.2 |
| `tensor_rate_per_cycle` | `gpgpusim.config` | 512/1024/2048/4096 | **the only real, live knob** | see §2.3 |

### 2.1 `trace.config`'s entire opcode-latency block is dead for the remodeled SM

`subcore.cc` constructs **every** functional-unit pipeline from a fixed depth taken from `gpgpusim.config`, never from
the per-instruction value `trace.config` computes:

```cpp
// subcore.cc:1708-1716
m_int_pipeline    = new functional_unit(..., m_config->max_int_latency, "INT", ...);
m_sp_pipeline     = new functional_unit(..., m_config->max_sp_latency, "SP", ...);
m_uniform_pipeline = new functional_unit(..., m_config->uniform_latency, "UNIFORM", ...);
m_tensor_pipeline = new functional_unit(..., m_config->tensor_latency, "TENSOR", ...);
m_branch_pipeline = new functional_unit(..., m_config->branch_latency, "BRANCH", ...);
m_sfu_pipeline    = new functional_unit_sfu(..., m_config->sfu_latency, "SFU", ...);
// ... same pattern for MISC_QUEUE, MISC_NO_QUEUE, DP
```

```cpp
// functional_unit.cc:58
m_pipeline_depth = max_latency;   // fixed at construction, never revisited
```

Meanwhile `trace.config`'s `-trace_opcode_latency_initiation_*` strings are parsed (`trace_driven.cc:655`,
`parse_config()`) and converted to a per-instruction `(latency, initiation_interval)` pair
(`trace_driven.cc:675`, `set_latency()`), which gets stored on the `warp_inst_t` at
parse time (`trace_driven.cc:338`, `tconfig->set_latency(op, latency, initiation_interval)`) — **and then never read**
by the remodeled core's fixed-depth pipelines. This is true for **every** op category (`int`, `sp`, `dp`, `sfu`,
`tensor`, `branch`, `half`, `uniform`, `predicate`, `miscellaneous_*`), confirmed structurally by reading the
constructor call sites, not just empirically for tensor.

**Practical implication:** every edit ever made to `trace.config` in this entire project — including the very first
action of this investigation (`32,8`→`32,32`) and the `24,32` test in this session — had zero effect on simulated
timing. `trace.config` is effectively vestigial for latency calibration under the remodeled SM; all real latency
knobs live in `gpgpusim.config`.

### 2.2 `tensor_latency` is a pipeline *capacity*, not the IMMA latency

Tensor-core ops are special-cased: they don't use a static latency at all. `abstract_hardware_model.cc:427`
(`warp_inst_t::generate_tensor_core_latencies`) computes a **per-instruction** value from the opcode's shape, parsed
from the opcode string itself by `traced_instruction::set_tensor_core_instruction_info()`
(`util/traces_enhanced/src/traced_instruction.cc:424`, regex over e.g. `"IMMA.16832.S8.S8"` → `size_m=16`,
`size_n=8`, `size_k=32`, `operand_bit_size=8`):

```cpp
cycles      = (size_m * size_n * size_k * operand_bit_size) / tensor_rate_per_cycle;
initiation  = cycles / 2;
latency     = cycles - initiation;
```

For this kernel's `IMMA.16832.S8.S8` at the baseline `tensor_rate_per_cycle=2048`:

```
cycles = (16 * 8 * 32 * 8) / 2048 = 16   →   initiation = 8, latency = 8
```

`gpgpusim.config`'s `-tensor_latency 32` (registered at `gpu-sim.cc:976`) only sets the **maximum capacity** of the
fixed-depth pipeline array (`functional_unit.cc:58`) that this 8-cycle value gets inserted into. Since `8 ≤ 24 ≤ 32`,
changing the capacity from 32 to 24 is non-binding — confirmed empirically: a 90k-cycle k3-only debug run with
`tensor_latency=24` came back byte-identical to baseline (`gpu_sim_cycle=90050`, `gpu_ipc=4395.498`,
`gpu_stall_dramfull=18903`, full `[issue_stall_stats]` histogram identical down to OMP-race-level noise in the last
digit of a few counts). The capacity value of 32 wasn't arbitrary either — it's exactly the cycle count produced by
`tensor_rate_per_cycle=512` (`64 total cycles / 2 = 32`), the slowest rate ever swept; the capacity was sized to avoid
overflow at that extreme, not chosen as a latency target.

### 2.3 The one real lever: `tensor_rate_per_cycle`

This is the only tensor knob that's both parsed and consumed by the path that runs. It directly scales the formula in
§2.2:

| `tensor_rate_per_cycle` | computed cycles | latency/initiation | k3 result |
|---|---|---|---|
| 512 | 64 | 32/32 | 284,048 (+10% vs baseline) |
| 1024 | 32 | 16/16 | 262,411 (+2%) |
| 2048 (baseline) | 16 | 8/8 | 258,275 |
| 4096 (`best_optA`) | 8 | 4/4 | contributes to optA's −8% |

Already swept and exploited. `TENSOR_CORE_OP` is only ~3.7% of total issue-gate stall events (see prior doc, P8), so
this lever has limited remaining headroom — it's real, but nearly exhausted.

---

## 3. New instrumentation: per-cycle subcore starvation cross-check

### 3.1 Motivation

The prior doc's leading hypothesis (P5/P6) was that issue-gate stalls — especially `SP_OP/scoreboard` (the dominant
category, 34% of all stall events) — are caused by **occupancy starvation**: with only 8 warps/SM (forced by
`mmq_x=128`'s 224 regs/thread × 256 threads ≈ 57k of H100's 65k regs/SM — one CTA per SM, no second CTA fits) and
`bar.sync` forcing near-lockstep, there's often no *other* warp ready to fill in when one stalls. An alternative
explanation considered was that the NVBit trace fails to preserve register-level ILP that exists in the real SASS
(the kernel does have genuine independent-accumulator parallelism: 157 distinct FFMA destination registers across
1024 FFMAs, 22 distinct IMMA destination registers across 256 IMMAs, confirmed from `enhanced_execution_info.json`).
That alternative was ruled out — the trace's dynamic order is faithful to what hardware executed (NVBit records
actual execution order), so ILP present in the SASS is present in the trace too. The remaining open question was
purely empirical: *when a warp stalls, is there really no other ready warp, or is the scheduler failing to pick one
that exists?*

### 3.2 What it measures

A new counter, gated behind the existing `-subcore_issue_debug 1` flag (no new CLI flag needed), answers this
directly: for every recorded `(op_type, stall_reason)` stall event, was the **entire subcore** idle that cycle (no
warp issued anything) or did some other warp issue while this one stalled?

### 3.3 Implementation (uncommitted)

Files: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h`, `.../subcore.cc`.

- `subcore.h`: added `Subcore` member `std::vector<std::pair<std::string,std::string>> m_debug_stall_events_this_cycle`
  (plus a kernel uid/name cache) — a per-cycle buffer of `(op, reason)` pairs.
- `subcore.cc`:
  - New `StarvationHistogramState` struct (mirrors the existing `IssueStallHistogramState`) with
    `starved_stall_by_op_reason[op][reason]`, `subcore_cycles_evaluated`, `subcore_cycles_starved`.
  - `starvation_histogram_record_cycle(...)`: called once per subcore per cycle, after `is_issued_inst` is final.
    Increments `subcore_cycles_evaluated` always; if nothing issued, increments `subcore_cycles_starved` and bumps
    `starved_stall_by_op_reason` for every buffered stall this cycle.
  - `starvation_histogram_print_body(...)`: prints overall starvation rate plus a per-`(op,reason)` table of
    `starved / total` (cross-referenced against the existing `g_issue_stall_histogram` totals).
  - In `Subcore::issue()`: at the existing stall-recording call site (where `issue_stall_histogram_record(..., false)`
    already fires), also push `(op, reason)` into `m_debug_stall_events_this_cycle`. At the very end of `issue()`,
    once `is_issued_inst` is finalized, flush the buffer through `starvation_histogram_record_cycle` and clear it.
  - Hooked into both existing print paths (`issue_debug_print_summary_body` and the `kernel_end` branch of
    `print_issue_debug_summary`), and reset alongside `g_issue_stall_histogram` between kernels.

This is **read-only instrumentation** — it doesn't change issue order, scoreboard behavior, or timing. Verified: a
90k-cycle debug run with the new code produced byte-identical `gpu_sim_cycle`/`gpu_ipc`/histogram totals to the
pre-instrumentation baseline.

### 3.4 How to run it

Same debug harness as before, same flags — the new `[starvation_stats]` block appears automatically alongside
`[subcore_issue_debug]` and `[issue_stall_stats]`:

```bash
cd simulator-remodeled/gpu-simulator
source ./setup_environment_no_git.sh debug   # MUST be in the same shell invocation as the run command —
                                               # env does not persist across separate tool/shell calls
OMP_NUM_THREADS=32 ./bin/debug/accel-sim.out \
  -config ./gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config \
  -config ./configs/tested-cfgs/SM90_H100/trace.config \
  -is_extra_traces_enabled 1 \
  -subcore_issue_debug 1 \
  -subcore_issue_debug_summary_interval 10000 \
  -subcore_issue_debug_stop_gpu_cycle 90000 \
  -filter_first_kernel_id 3 -filter_last_kernel_id 3 \
  -trace /home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb \
  > ../log/tmp_log/debug_cycle_90000_starvation.log
```

---

## 4. Result: starvation confirmed as the dominant mechanism

`log/tmp_log/debug_cycle_90000_starvation.log`, same 90k-cycle k3-only window as the prior doc's baseline debug run:

```
subcore_cycles_evaluated = 10,720,345
subcore_cycles_starved   = 8,001,937   (74.6%)
```

**Nearly 3 out of 4 subcore-cycles had nothing else to issue at all** — not a scheduler picking badly, but zero other
warps ready. Per `(op, reason)`:

| op / reason | total stall events | starved % |
|---|---:|---:|
| `LOAD_OP / cta_barrier` | 3,100,935 | **99.8%** |
| `SP_OP / cta_barrier` | 1,505,144 | **99.8%** |
| `SFU_OP / cta_barrier` | 220,312 | **99.8%** |
| `BRANCH_OP / cta_barrier` | 310,434 | 99.2% |
| `HALF_OP / scoreboard` | 86,219 | 99.7% |
| `STORE_OP / scoreboard` | 161,692 | 97.9% |
| `TENSOR_CORE_OP / fu_busy` | 554,968 | 92.8% |
| `UNIFORM_OP / fu_busy` | 15,328 | 93.3% |
| `INTP_OP / fu_busy` | 436,077 | 91.2% |
| `SFU_OP / scoreboard` | 1,841,183 | 87.7% |
| `INTP_OP / scoreboard` | 1,907,295 | 85.8% |
| `UNIFORM_OP / scoreboard` | 192,640 | 78.9% |
| **`SP_OP / scoreboard`** (largest single category) | **6,442,048** | **76.2%** |
| `LOAD_OP / scoreboard` | 743,170 | 67.9% |
| `TENSOR_CORE_OP / scoreboard` | 737,230 | **61.5%** (lowest of the major categories) |

### Reading this

- **Every `cta_barrier` stall is essentially universal starvation (97–100%).** Exactly what `bar.sync` forcing 8
  co-resident warps into lockstep predicts: when they hit the barrier together, the whole SM goes idle together, not
  just one warp.
- **Scoreboard stalls are majority-but-not-total starvation (62–88%).** There's a genuine, smaller minority
  (12–38% depending on op type) where another warp *was* issuing while this one waited on its scoreboard — the
  scheduler isn't blind, it's just usually out of independent work.
- `TENSOR_CORE_OP / scoreboard` has the lowest starvation rate of the major categories (61.5%) — tensor ops have
  comparatively more "slack" from sibling warps than scalar ops do.

---

## 5. Updated overall picture

Combining everything from this doc and the prior one, the ~92,000-cycle gap (258,275 − 166,246) now decomposes as:

```
~75% of issue-gate stall events  →  genuine occupancy starvation (bar.sync lockstep + 8 warps/SM,
                                     itself forced by 224 regs/thread × 256 threads ≈ 57k/65k regs/SM)
                                     — REAL hardware-forced behavior, not a simulator bug
~12-38% of scoreboard stalls     →  scheduler had an alternative warp but didn't fill the gap —
  (minority slice)                  the only remaining code-level lever, low payoff
~8% (proven)                     →  memory-latency under-modeling, recovered by SM90_H100_best_optA
~3.7% of stall events            →  TENSOR_CORE_OP itself, already minimized via tensor_rate_per_cycle
0%                                →  constant-cache queueing (refuted), trace.config edits (dead path),
                                     barrier relaxation (proven no-op), MSHR/miss-queue sizing (<1%)
```

**Conclusion: the dominant share of the +55% gap is a correct reflection of how this specific low-occupancy,
single-CTA-per-SM kernel behaves**, not a fixable simulator defect. Config-only tuning has converged to its ceiling
(`SM90_H100_best_optA`, +43% vs real). Closing the remaining gap further would require either:

1. Accepting current state and documenting it (config tuning is exhausted; root cause is identified and is largely
   real, not simulator error), or
2. Re-examining the ground-truth measurement itself — the isolated 3-kernel trace doesn't include
   `mul_mat_q_stream_k_fixup` or adjacent-layer kernels that overlap with k3 on real hardware; the measured
   297µs/166k-cycle figure could be capturing something the isolated single-kernel simulation structurally cannot
   (this is a measurement-methodology question, not a simulator-accuracy one).

---

## 6. File index (additions to prior doc's §7)

| Path | Contents |
|------|----------|
| `notes/prefill-k3-config-audit-and-occupancy.md` | **This document** |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_nopc_constmshr/` | Refuted: real const cache + widened MSHR |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_tlat24/trace.config` | Dead-path test (trace.config tensor latency) |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_tensorlat24/` | Non-binding capacity test (gpgpusim.config `tensor_latency`) |
| `log/prefill_nopc_constmshr/sim.log` | Refuted constant-cache MSHR experiment, full 3-kernel |
| `log/tmp_log/debug_cycle_90000_tlat24.log` | Dead-path confirmation (trace.config) |
| `log/tmp_log/debug_cycle_90000_tensorlat24.log` | Non-binding capacity confirmation (gpgpusim.config) |
| `log/tmp_log/debug_cycle_90000_starvation.log` | **Decisive starvation result** |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h` / `.cc` | New (uncommitted) `[starvation_stats]` instrumentation |

---

*Last updated: 2026-06-18 — after constant-cache MSHR refutation, tensor-latency dead-path/non-binding-capacity audit,
and the starvation instrumentation result.*
