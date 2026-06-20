# Prefill k3 debug investigation — phenomena, evidence, conclusions

Companion to [`prefill-k3-correlation.md`](prefill-k3-correlation.md). That document covers **what k3 is** (geometry, MACs, trace layout). This document covers **how we diagnosed the +55% cycle gap** and **what we concluded from each observation**.

| Item | Value |
|------|-------|
| Kernel under study | k3 `mul_mat_q<Q8_0, mmq_x=128>` (trace `kernel_id=3`, nsys **37**) |
| Trace | `/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb` |
| Sim config | `SM90_H100/gpgpusim.config` @ **1.620 GHz** |
| Real H100 target | 102,621 ns → **~166,246 cycles** |
| Baseline sim k3 | **258,275 cycles** (+55%) — `log/prefill/sim.log` |
| Debug run (30k early stop) | `log/tmp_log/debug.log` |

---

## 1. What we did (chronology)

### Phase A — Establish ground truth and baseline gap

1. Profiled llama-bench prefill on real H100 (nsys). Identified the correct kernel IDs (**35–37**, not 23–25 or 1–3).
2. Ran full 3-kernel protobuf simulation with `SM90_H100` baseline.
3. Recorded k3 at **258,275 cycles** vs hardware **~166,246 cycles** (+55% too slow). k1/k2 were only ~+8%.

**Artifacts:** `log/prefill/sim.log`, `notes/prefill-k3-correlation.md` §1–7.

### Phase B — Config knob sweeps (hypothesis testing)

Ran scripted experiments (`run_prefill_*.sh`) varying one subsystem at a time:

| Sweep | Knobs | k3 cycles | vs baseline |
|-------|-------|-----------|-------------|
| Tensor throughput | `tensor_rate_per_cycle` 512 / 1024 | 284,048 / 262,411 | **worse** (+10% / +2%) |
| Tensor init / extra latency | `tinit*`, `textra*` | ~258,275 | **no change** |
| L1 MSHR / miss queue | `mq32`, `mq64`, `mshr1024` | ~257,900–259,078 | **<1%** |
| L2 read-only port | `l2rop300`, `l2rop380` | ~259k–263k | **small** |
| Best combined | `SM90_H100_best`, `best_optA` | 243,698 / **237,536** | **−6% / −8%** |
| Bad memory config | old L1D (`sim_another_bk`) | **757,765** | **+193%** |

**Artifacts:** `log/prefill_*`, `log/prefill_best_multiopt/`.

### Phase C — Routing and trace interpretation experiments

1. **LDG.E.CONSTANT → L1TEX** routing change (committed C++): routed constant loads through texture path.
   - **Result:** k3 got **worse** (~+61%), not better. Reverted for calibration purposes.
2. Confirmed trace uses **traditional scoreboarding** (`is_captured_from_binary=false` in enhanced metadata), not full remodeling scoreboard (`is_remodeling_scoreboarding_enabled=0` in config).

### Phase D — Issue-gate debug instrumentation (uncommitted C++)

Added to `subcore.cc` / `gpu-sim.cc` / `main.cc`:

| Flag | Purpose |
|------|---------|
| `-subcore_issue_debug 1` | Enable stall tracing |
| `-subcore_issue_debug_summary_interval N` | Print cumulative summary every N GPU cycles |
| `-subcore_issue_debug_print_period N` | Verbose per-cycle lines on SM0/subcore0/warp0 (optional) |
| `-subcore_issue_debug_stop_gpu_cycle N` | Early stop + final summary |
| Ctrl+C / `request_simulation_stop()` | Stop and print summary |

Two summary layers:

1. **`[subcore_issue_debug]`** — SM0 / subcore0 / dynamic warp 0 only: fraction of logged cycles spent in each stall class.
2. **`[issue_stall_stats]`** — **All SMs, all warps**: histogram of `(op_type, stall_reason)` at IBuffer head when an instruction fails to issue or issues.

Fixed **OpenMP heap corruption**: histogram `unordered_map` was updated from 132 threads without a full critical section → `free(): invalid pointer` at SM131. Fix: single `issue_stall_histogram_record()` under `#pragma omp critical`.

### Phase E — k3-only 30k-cycle debug run

```bash
cd simulator-remodeled/gpu-simulator
source ./setup_environment_no_git.sh debug
OMP_NUM_THREADS=32 ./bin/debug/accel-sim.out \
  -config ./gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config \
  -config ./configs/tested-cfgs/SM90_H100/trace.config \
  -is_extra_traces_enabled 1 \
  -subcore_issue_debug 1 \
  -subcore_issue_debug_summary_interval 10000 \
  -subcore_issue_debug_print_period 0 \
  -subcore_issue_debug_stop_gpu_cycle 30000 \
  -filter_first_kernel_id 3 -filter_last_kernel_id 3 \
  -trace /home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb \
  > log/tmp_log/debug.log
```

**Outcome:** Completed cleanly (no crash), partial summaries at ~10k / ~20k GPU cycles, final summary at 30k (`early_stop`).

**Earlier work (pre-histogram):** per-cycle verbose log on warp0 (`log/tmp_log/detailed.log`) parsed by `log/tmp_log/parse_detailed.py` — showed scoreboard ~51%, programmer/CTA barriers ~40% on a longer single-warp trace.

---

## 2. Phenomenon → evidence → conclusion

Each row is one **observation**, what we **measured**, and what we **infer**. Conclusions at the bottom are the synthesis.

### P1 — Total cycle gap is dominated by k3

| | |
|--|--|
| **Phenomenon** | 3-kernel prefill run is much slower than hardware, but k1/k2 are close. |
| **Evidence** | k1 +8%, k2 +8%, k3 **+55%** (258,275 vs 166,246 target cycles). |
| **Conclusion** | All calibration effort should focus on **k3 `mul_mat_q`**; k1/k2 are acceptable. |

---

### P2 — Tensor-core throughput knobs barely move k3

| | |
|--|--|
| **Phenomenon** | We expected IMMA rate might explain slowness (H100 peak vs achieved TOPS). |
| **Evidence** | `tensor_rate_per_cycle` 2048→512: k3 **+10% worse**. `tinit*` / `textra*`: no effect. IMMA ubench ~24 clk/IMMA; config uses 32-cycle model. |
| **Conclusion** | k3 is **not limited by tensor FU peak throughput** in the simulator. Slowing tensor cores makes the kernel slower; speeding them (within tested range) does not help much. **Do not optimize `tensor_rate` for k3.** |

---

### P3 — Memory subsystem knobs move k3 a lot (but are a separate axis)

| | |
|--|--|
| **Phenomenon** | Changing L1D/L2/MSHR settings swings total k3 cycles wildly. |
| **Evidence** | Bad L1 config: **+193%** (757k cycles). `best_optA`: **−8%** (238k). `mq`/`mshr` sweeps: <1%. Full k3 `gpu_stall_dramfull` = **2,186,115**; L1D miss ~52%, heavy `MISS_QUEUE_FULL`. |
| **Conclusion** | **Memory model bounds** total time (10–50%+), but this is **orthogonal** to the issue-gate story (see P5–P7). Best config so far only recovers ~8%; memory alone cannot close the +55% gap. |

---

### P4 — LDG constant routing is not the fix

| | |
|--|--|
| **Phenomenon** | Constant loads might be miscached (const cache vs global). |
| **Evidence** | LDG.E.CONSTANT → L1TEX routing experiment increased k3 error substantially. |
| **Conclusion** | **Reject** const-cache routing as a calibration lever for k3. |

---

### P5 — Issue-gate scoreboard stalls dominate early and mid kernel

| | |
|--|--|
| **Phenomenon** | Warps often have an instruction at IBuffer head but cannot issue. |
| **Evidence (30k histogram, all SMs)** | Top pairs: SP/INTP/SFU **scoreboard** = 32% + 17% + 17% ≈ **66%** of stall events. SP/INTP issue rate ~24–31%. |
| **Evidence (warp0, ~10k)** | SM0 warp0: **80%** of logged cycles scoreboard-blocked; CTA barrier 0%. |
| **Evidence (earlier full warp0 verbose trace)** | Sole blocker `scoreboards` ~51% of not-ready events. |
| **Conclusion** | The simulator is **too conservative on register / dependency readiness** for compute ops. Config: `is_remodeling_scoreboarding_enabled=0` + trace control bits. This is the **#1 issue-gate bottleneck**. |

---

### P6 — CTA barrier stalls grow through the kernel

| | |
|--|--|
| **Phenomenon** | `bar.sync` / tile synchronization should block some loads and branches, but maybe not as much as modeled. |
| **Evidence (30k, time evolution)** | |

| Checkpoint | warp0 `cta_barrier` | `LOAD_OP` stall reason |
|------------|---------------------|-------------------------|
| ~10k | 0% | 100% scoreboard |
| ~20k | 13.8% | **89% cta_barrier** |
| ~30k | 13.3% | **85% cta_barrier** |

| | |
|--|--|
| **Evidence (30k histogram)** | `LOAD_OP / cta_barrier` = **12.2%** of all stall events; `BRANCH_OP / cta_barrier` appears by 30k. |
| **Evidence (earlier warp0 full trace)** | Programmer/CTA barriers ~**40%** over full kernel. |
| **Conclusion** | Barrier modeling is the **#2 issue-gate bottleneck**, especially for **loads after `bar.sync`** in the LDG→shared→IMMA tile loop. Early 30k **under-represents** barriers vs full kernel. |

---

### P7 — Memory pressure is back-loaded; not visible at issue gate early

| | |
|--|--|
| **Phenomenon** | Full k3 shows huge memory stalls; 30k debug does not. |
| **Evidence** | |

| Metric | 30k debug (k3-only) | Full k3 (`sim.log`) |
|--------|---------------------|---------------------|
| `gpu_sim_cycle` | 30,001 | 258,275 |
| `gpu_ipc` | 3,214 | 4,531 |
| `gpu_stall_dramfull` | 19,903 | 2,186,115 |
| Issue stage issuing | 20.5% | 29.1% |
| Stall: next stage N/A | 57.4% | 47.3% |
| L2 data port util | ~6% | ~high in full run |

| | |
|--|--|
| **Conclusion** | **Two different stall layers:** (A) **front-end** — scoreboard + barriers at IBuffer; (B) **back-end** — memory / pipeline back-pressure (`dramfull`, next-stage N/A). Layer B **ramps up after the first ~10–12%** of kernel cycles. The 30k debug run is **directionally correct** but **not quantitatively representative** of full-kernel stall mix. |

---

### P8 — Tensor ops are a small fraction of issue-gate stalls

| | |
|--|--|
| **Phenomenon** | k3 is an IMMA kernel; we might expect `TENSOR_CORE_OP` to dominate stalls. |
| **Evidence (30k)** | `TENSOR_CORE_OP` stalls = 259k of 7.07M total (**3.7%**). Split: 54% scoreboard, 44% `fu_busy`. |
| **Conclusion** | Most cycles are lost on **scalar FP / int / SFU** around the IMMA loop (dequant, address, uniform), not on the tensor pipe itself. Matches static trace analysis (3072 FP32 ops per 256 IMMA). |

---

### P9 — Kernel micro-pattern is synchronous tiles, not TMA/WGMMA

| | |
|--|--|
| **Phenomenon** | Hopper has TMA/WGMMA; maybe the sim misses async overlap. |
| **Evidence** | NVBit trace: `LDG`/`STS` → `LDS` → `IMMA`, `BSSY`/`BSYNC`. No `UTMA*`, no `cp.async`, no double-buffered `tile_x[0/1]`. |
| **Conclusion** | The sim must get **synchronous tile load + barrier + compute** right. Missing TMA is **not** the explanation — the real kernel does not use TMA on this path. |

---

### P10 — Debug instrumentation validated the failure mode

| | |
|--|--|
| **Phenomenon** | Simulator crashed at SM131 during debug run. |
| **Evidence** | `free(): invalid pointer` in `baseline_cache::fill` during OpenMP `core_cycle`; root cause was racy `issue_stall_histogram` map reset. |
| **Conclusion** | Crash was a **debug-tooling bug**, not a cache model bug. Fixed with OpenMP critical around histogram. Debug runs are **~5× slower** wall-clock due to per-issue histogram locking. |

---

## 3. Comparison tables

### 3.1 Real vs sim (k3 only)

| | Real H100 | Sim baseline | Sim `best_optA` |
|--|-----------|--------------|-----------------|
| Time @ 1.620 GHz | 102,621 ns | 159,429 ns | 146,626 ns |
| Cycles | **~166,246** | **258,275** | **237,536** |
| Error | — | **+55%** | **+43%** |
| Effective INT8 TOPS | ~58 | ~108 | ~118 |

### 3.2 Stall mix — three measurement methods

| Method | Scoreboard / deps | CTA / prog barrier | FU busy | Memory / pipeline |
|--------|-------------------|--------------------|---------|-------------------|
| 30k `[issue_stall_stats]` (all SMs) | **~66%** of issue stalls | **~18%** | ~5% | not in histogram |
| 30k `[subcore_issue_debug]` warp0 @ 30k | **53%** of cycles | **13%** | 3.5% | l1c 0% |
| Earlier warp0 verbose (full kernel) | **~51%** | **~40%** | small | — |
| Full k3 `sim.log` pipeline stats | (issue gate) | — | — | **47%** next-stage N/A, `dramfull` 2.1M |

Methods measure **different things**. Issue-gate histogram counts **per-issue-attempt stall reasons**. Pipeline stats count **cycles** where the pipeline cannot advance. Both point to **front-end pessimism + memory back-pressure**, not tensor peak rate.

### 3.3 30k debug vs full k3 — does partial match full?

| Question | Answer |
|----------|--------|
| Same kernel & config? | **Yes** |
| Same qualitative ranking? | **Yes** — scoreboard #1, barriers #2 |
| Same quantitative stall %? | **No** — barriers still ramping; memory barely started |
| Same IPC? | **No** — 3,214 (30k) vs 4,531 (full); early phase is slower |
| Useful for picking knobs? | **Yes** for issue-gate; **No** for final cycle prediction |

---

## 4. Synthesis — best guess decomposition

Extra **~92,000 cycles** (258k − 166k), rough attribution:

```
~45–55%  Scoreboard / dependency checks too strict (SP, INTP, SFU)
~25–35%  CTA barrier / bar.sync waits (LOAD, BRANCH, SP)
~15–25%  L1 MSHR / memory back-pressure (grows through kernel)
~5%      Tensor FU initiation / width
```

**Causal story:** `mul_mat_q` executes a **stream-K tiled loop** where each K-step does **load tile → barrier → shared read → IMMA → heavy FP dequant**. The simulator:

1. Holds warps on **register deps** too long (scoreboard / control bits).
2. Holds warps at **barriers** too long before issuing loads/branches.
3. Later, **memory ports / MSHRs** add more idle cycles (visible in full run, not in first 30k).

Config can tune (3) and slightly (2) via `is_relax_barriers_baseline`, L2/L1, initiation intervals. **(1) likely needs code changes** to scoreboard release or control-bit interpretation.

**Realistic target with config only:** ~200–220k cycles (~+20–30% vs hardware).  
**To reach ~166k:** front-end modeling fixes, not more `tensor_rate` or TMA work.

---

## 5. What config can and cannot do

| Lever | Can help? | Limit |
|-------|-----------|-------|
| `is_relax_barriers_baseline 1` | **Try** — reduces non-mem ops waiting at barriers | Does not fix scoreboard |
| `SM90_H100_best_optA` memory | **Yes** — proven −8% | Still +43% vs real |
| `int_initiation` / `uniform_initiation` | **Small** — cuts `fu_busy` slice | ~5% of stalls |
| `tensor_rate_per_cycle` | **No** — hurts or neutral | Wrong bottleneck |
| `is_remodeling_scoreboarding_enabled 1` | **Unknown** — may help or hurt | Trace uses traditional scoreboard path |
| LDG constant → L1TEX | **No** — made k3 worse | Rejected |

---

## 6. Open questions and next steps

1. **Full k3-only run** with `[issue_stall_stats]` at kernel end (no `stop_gpu_cycle`) — confirm barrier % at 258k vs 30k.
2. **`is_relax_barriers_baseline 1`** sweep on k3-only — quantify barrier contribution to total cycles.
3. **Scoreboard deep dive** — which registers / op pairs stall most on SP_OP? Extend histogram or revive warp0 verbose with `print_period=100`.
4. **Compare `best_optA` full run + histogram** — does −8% cycles come from memory layer only, or also shift issue-gate mix?
5. **Commit debug C++** (histogram + OMP fix) without `.sh` / experimental configs — user preference from prior session.

---

## 7. File index

| Path | Contents |
|------|----------|
| `notes/prefill-k3-correlation.md` | Kernel geometry, MACs, config sweeps table, microarchitecture context |
| `notes/prefill-k3-debug-investigation.md` | **This document** — debug methodology, phenomenon→conclusion |
| `log/prefill/sim.log` | Full 3-kernel baseline |
| `log/tmp_log/debug.log` | k3-only 30k debug run with histograms |
| `log/tmp_log/parse_detailed.py` | Parser for old per-cycle verbose format |
| `gpu-simulator/.../subcore.cc` | Issue-gate debug + histogram (uncommitted) |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/` | Baseline config |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_best_optA/` | Best k3 cycle count so far |

---

*Last updated: 2026-06-18 — after k3-only 30k debug run (`debug.log`) and OMP histogram fix.*
