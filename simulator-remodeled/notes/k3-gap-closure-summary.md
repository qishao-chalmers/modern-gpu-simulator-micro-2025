# k3 sim-vs-real gap closure: +59.6% → +25.2%

Retrospective summary of the multi-session investigation into why this simulator overestimates cycle count for
kernel **k3** (`mul_mat_q<Q8_0, mmq_x=128>`, llama.cpp prefill GEMM) vs real H100 hardware, and the four fixes
that closed most of the gap. Full session-by-session detail (including everything ruled out) lives in
`notes/prefill-k3-real-hw-correlation.md`; this doc is the condensed "what/why/how" version.

**Target**: real H100 measured **166,246 cycles** (102,621 ns @ 1.62 GHz core clock) for one full-chip launch
(132 CTAs, grid=132=#SMs, 1 CTA/SM, 12.5% occupancy on both real and simulated hardware).

**Starting point**: pre-fix baseline (sectored L2, cold-start k3, full chip) = **265,360 cycles, +59.6%** over
target.

**Current best**: **208,119 cycles, +25.2%** over target — four stacked levers, each independently validated
(instruction counts, DRAM traffic, and completion behavior unchanged before/after every fix; only cycle timing
moved).

| Step | Cycles (full-chip) | Gap vs real | Δ |
|---|---|---|---|
| Baseline (sectored L2) | 265,360 | +59.6% | — |
| + L2 cache type: sectored → normal | 216,392 | +30.2% | **−18.5%** |
| + WAR scoreboard hazard mode disabled | 210,062 | +26.35% | −2.93% |
| + register bypass/forwarding network | 208,650 | +25.51% | −0.67% |
| + bypass window/port sweep (w4/p1) | 208,191 | +25.23% | −0.22% |
| + early-forward bypass (dispatch-time, SP_OP) | **208,119** | **+25.19%** | −0.035% |

---

## Lever 1: L2 cache type, sectored → normal (the big one, −18.5%)

**What**: changed `-gpgpu_cache:dl2`'s type token from `S` (SECTOR) to `N` (NORMAL) — same set count/line
size/associativity, but `m_atom_sz` becomes the full 128-byte line instead of 32-byte sectors, so one MSHR
entry/DRAM admission covers a whole line instead of up to 4 independent sector fetches.

**Why**: `load_tiles_q8_0` (k3's load stage) is a dense GEMM-style read — every CTA touches every sector of every
cache line it accesses, so sectoring buys zero bandwidth benefit here while adding real serialization cost: each
extra sector of the same line independently re-enters the per-partition "one DRAM admission per cycle per
partition" gate (`l2cache.cc`), stretching `dram_dur` ~6-7 cycles per extra sector for what should be a single
fetch. This was a real, structural cost specific to dense access patterns, not a contention artifact — confirmed
on a fully isolated single-SM/single-CTA run (zero cross-CTA contention) where the effect persisted unchanged.

**How**: required a real bug fix to land, not just a config flag. With L2 normal but L1I/L1C/L1T still sectored,
`baseline_cache::fill()` and `tex_cache::fill()` crashed — both unconditionally entered a sector-reassembly branch
keyed on `get_original_mf()`, which is only populated when the *downstream* cache is also sectored. Fixed by
guarding that branch on `get_original_mf() != nullptr` and the parent being tracked; falls through to a plain
single-fill when the downstream response is already a full normal-L2 line. (L1D sectoring was tested too on top
of this and found to be exactly zero-effect for k3 — `load_tiles_q8_0`'s loads route through L1C, which was
already `N` type — so it was closed out as a non-lever, but the combined config is still named
`SM90_H100_l2norm_l1dnorm` for historical reasons.)

**Result**: 265,360 → 216,392 cycles (full-chip), −18.5%, closing ~29 of the original ~60 percentage points.
Confirmed scale-independent (isolated 1-SM/1-CTA showed the matching −12.1%).

---

## Lever 2: disable WAR scoreboard hazard checking (−2.93%)

**What**: `-scoreboard_war_mode opc → disabled`. GPGPU-Sim's scoreboard tracks both true RAW data dependencies
and WAR anti-dependencies (a later instruction can't overwrite a register an earlier, still-in-flight instruction
still needs to read); this flag turns WAR tracking off entirely, leaving only RAW.

**Why**: a new per-event hazard-kind classifier (instrumentation added this session) split every scoreboard stall
in k3 by kind and found **42.6% of all scoreboard stalls were pure WAR with no accompanying RAW collision** — a
full third of every stall event in the kernel was an anti-dependency, not a true data dependency. k3's mmq inner
loop reuses register-allocated accumulator/operand slots heavily across unrolled iterations, which is a plausible
structural source of WAR pressure (not independently confirmed by reading the generated SASS, but consistent with
the measured magnitude).

**How**: existing config knob, no code change. Verified functionally safe before persisting: in this trace-driven
simulator, traced memory addresses/operands come directly from the NVBit-recorded trace, not from any in-simulator
functional engine gated by the scoreboard — so the scoreboard can only ever change *when* an instruction issues,
never *what* value or address it uses. Empirically confirmed too: diffing baseline vs WAR-disabled showed
byte-identical instruction counts, DRAM traffic, and completion behavior; only cache/interconnect timing counters
shifted by the expected sub-0.1% amount from the changed issue timing.

**Result**: 216,392 → 210,062 cycles (full-chip), −2.93%, full-chip and isolated runs tracked almost exactly
(−2.93% vs −3.21%), confirming a real, scale-independent effect.

---

## Lever 3: register bypass/forwarding network (new microarchitecture feature, −0.67%)

**What**: this simulator modeled **no bypass/forwarding network at all** prior to this session (confirmed via
grep — zero bypass logic existed in `subcore.cc`) — every dependent instruction had to wait for the full
register-file writeback, even though real hardware forwards a just-computed result directly to a waiting
consumer well before it's durably written back. Built a real, scoped feature: when an instruction with
destination registers finishes EX, its destination registers are marked forwardable for a configurable window
(`-register_bypass_window_cycles`, default 2) in a new per-warp table on `Scoreboard`
(`m_bypass_forward_table`) — without releasing the real scoreboard claim, so WAW ordering and full RF-writeback
timing stay untouched. At issue time, a collision on a *source* operand (true RAW) can be satisfied by the bypass
network if a forwarded write is still within its window and a bypass port is free for that subcore this cycle; a
collision on the consumer's own destination register (WAW) is never bypass-eligible and always still blocks.

**Why**: real hardware has forwarding paths between execution units precisely so a dependent instruction doesn't
have to wait for the full writeback latency; modeling none of this was a known, plausible source of
over-estimated dependent-latency stalls, and the per-PC stall attribution work this session (see Lever 4) had
already shown the dominant stall class was exactly this kind of RAW wait on FFMA/FMUL-chain accumulators.

**How**: new feature in `scoreboard.h/.cc` (bypass table, availability check) and `sm.cc`
(`maybe_record_register_bypass()`, hooked into `functional_unit::instruction_finishing_execution`); gated behind
`-is_register_bypass_forwarding_enabled` with `-register_bypass_window_cycles`/`-register_bypass_ports_per_subcore`
tunables. A follow-up window/port sweep found window=4/ports=1 as the (non-monotonic) sweet spot — wider windows
and more ports don't help further, sometimes slightly hurt by changing which warp the greedy scheduler favors a
given cycle.

**Result**: 210,062 → 208,650 cycles (w2/p1) → 208,191 cycles (w4/p1 after the sweep), combined −0.77%.
Functionally clean (instruction counts identical before/after) since the bypass only ever shortcuts a wait, never
skips real ordering.

---

## Lever 4: early-forward bypass extension, SP_OP only (−0.035%)

**What**: a second, earlier bypass-recording path specifically for `SP__OP` (FP32 ADD/MUL/MAD) instructions.
Lever 3's bypass network only marks registers forwardable when an instruction *finishes* EX — which can shorten
the EX-finish-to-writeback tail, but does nothing for the literal in-EX dependent latency itself (e.g. a 4-cycle
FFMA RAW wait). This new path marks destination registers forwardable **at dispatch into the functional unit**
instead, with an extra configurable delay before the bypass becomes valid (so it doesn't claim availability sooner
than the pipe could legitimately produce a forwardable value).

**Why**: dedicated per-PC stall-count instrumentation (not just per-op-type) found that the dequant-accumulate
epilogue's `FFMA.FTZ`/`FMUL.FTZ` chain dominates the SP_OP/scoreboard stall bucket almost completely (41 of the
top 43 stalling PCs), and that real hardware pays almost nothing for this exact instruction pattern (confirmed via
both an aggregate stall-time comparison and this PC-exact cross-check) while the simulator pays the most of
anything in the kernel for it. That gap is shaped like a dependent-latency problem specifically on this op class,
which a dispatch-time (rather than EX-finish-time) bypass path can directly target.

**How**: `Scoreboard::recordBypassWrite()` gained a `valid_from_cycle` parameter (the bypass table's value became
a `{valid_from, expire}` pair instead of a single expiry); `SM::maybe_record_register_bypass_early()` is called
from `functional_unit::issue()` at dispatch time, gated to `SP__OP` only, computing `valid_from = dispatch_cycle +
register_early_forward_delay_cycles`. A delay/port sweep found delay=3/ports=2 as the local optimum (delay=1
actively makes things worse — more simultaneous bypass attempts contend for the single port per subcore per
cycle, perturbing scheduler choice).

**Result**: 208,191 → 208,119 cycles (full-chip), −0.035% — real and correctness-clean, but small; the
isolated-test improvement (−0.138%) damped significantly at full-chip scale, the same pattern seen with Lever 3.

---

## What was ruled out along the way (so it isn't re-tried)

A long list of plausible capacity/bandwidth/topology levers were tested and found to have negligible-to-small
effect, consistently pointing toward "the problem is latency-shaped, not throughput-shaped":

- icnt buffer/subnet capacity, icnt arbiter policy (iSLIP vs NAIVE_RR, only −1.6%)
- per-partition DRAM load imbalance (none found)
- a suspected "DRAM admission backlog" — retracted, was a `0/*FIXME*/` clock-reset bug in `L2interface::push()`,
  not a real effect
- constant-cache MSHR entries 2→32 (−0.34%)
- `SM90_H100_best`'s broader memory/topology retuning (n_mem 64→80, dual-bus HBM3, JESD timings, 50MB L2): only
  −2.1%, occupancy/starvation regime untouched
- shared register-file writeback queue width 1→4 pops/cycle: only −0.2%
- IMMA initiation/dependent-latency decouple toward real-hardware ubench numbers: negligible, slightly negative
  on a later baseline
- fetch/issue decoupling experiment (un-starving a subcore-mate warp losing fetch arbitration): +0.35% (worse) —
  the greedy "stick with whoever's making progress" scheduling coupling is net beneficial despite causing
  visible, individually-diagnosable starvation in specific spots

## Where the remaining ~25% gap likely sits

Real H100's own `ncu` per-warp stall breakdown on this kernel shows `wait`(16.2%) + `long_scoreboard`(15.1%) as
by far the largest genuine dependency-stall categories, with `math_pipe_throttle` (5.1%) and other
capacity/throttle reasons small — i.e. real hardware's own profile agrees the bottleneck is dependency-chain
overlap, not a capacity/throughput wall, which is exactly why the four shipped levers (all latency-shaped) moved
the needle while every capacity/bandwidth lever tested didn't.

A roofline check confirms the same story from a different angle: k3's arithmetic intensity (~665 ops/byte) sits
just above the H100 ridge point (591 ops/byte) — nominally "compute-bound" — yet real hardware achieves only
**8.2% of peak INT8 TOPS and 7.3% of peak HBM bandwidth** simultaneously on this kernel (sim: 6.5%/9.4%). Neither
real nor simulated hardware is anywhere near either roofline ceiling; both are sitting deep inside the roof,
consistent with the 12.5% occupancy measured on both. The remaining gap is therefore expected to keep being a
scheduling/dependency-overlap story rather than a capacity story — narrowing it further would mean more
microarchitecture features in the early-forward-bypass family (Levers 3-4), not more cache/bandwidth tuning.

## Source references

All numbers above are traceable to `notes/prefill-k3-real-hw-correlation.md`:
- Baseline + L2-normal: §20.3, §22.1-22.3
- WAR scoreboard disable: §23.15
- Bypass network (Lever 3) + window/port sweep: §23.17-23.18
- Per-PC stall attribution motivating Lever 4: §23.22b, §23.24
- Early-forward bypass (Lever 4): §23.25
- Roofline cross-check: §23.32-23.33
