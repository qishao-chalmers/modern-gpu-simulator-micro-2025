# Simulator debugging toolkit — CLI flags, scripts, and trace-manipulation recipes

Reference doc for tools/scripts discovered or built while debugging the `flash_attn_ext_vec` sim-vs-real gap
(see [`decode-attn-qwen3-8b-14b-correlation.md`](decode-attn-qwen3-8b-14b-correlation.md) for the investigation
itself). This doc is about *how* to dig, not *what was found* — keep it model/investigation-agnostic so it's useful
for the next correlation problem too.

---

## 1. SM0-targeted debug CLI flags (built into `gpu-sim.cc`)

All of these are gated to SM0 only (to keep output volume manageable) and are off by default. Combine with
`-filter_first_kernel_id`/`-filter_last_kernel_id` to isolate a single kernel — if that kernel's grid is smaller than
the GPU's SM count (e.g. 80 CTAs on a 132-SM H100), SM0 will only ever process **one CTA**, so the full trace stays
small enough to read by hand.

| Flag | Default | What it prints |
|---|---|---|
| `-subcore_issue_debug` (bool) | 0 | `[commit_trace]` per retired instruction on SM0: sm/subcore/warp, pc, op_type, opcode, operands (annotated dst/src), issue_cycle, commit_cycle, duration. Plus a per-kernel stall-cycle summary at kernel end. |
| `-subcore_issue_debug_summary_interval` (uint32) | 0 | While the above is on, also print a cumulative stall summary every N `gpu_sim` cycles on SM0 (0 = kernel-end only). |
| `-subcore_issue_debug_print_period` (uint32) | 1 | Emit verbose per-cycle issue lines every N *logged* cycles on SM0/subcore0/warp0 (1 = every cycle). |
| `-subcore_issue_debug_stop_gpu_cycle` (uint64) | 0 | Stop simulation after this `gpu_sim` cycle and print the stall summary (0 = disabled). Useful to bound a long kernel's debug output. |
| `-issue_wait_trace_debug` (bool) | 0 | `[issue_wait_trace]` every cycle, for every warp on SM0 (all subcores): issued or not, and if not, the *specific* reason (scoreboard collision register, barrier kind, fu_busy, result_queue, etc). Independent of `-subcore_issue_debug`'s aggregate stats — much higher volume, use after narrowing down with the flags above. |
| `-debug_isolate_sm_id` (int32) | -1 | If ≥0, no SM other than this one is ever bound to a CTA — isolates the kernel to one SM with **zero inter-SM contention**. Use to rule out cross-SM interconnect/DRAM contention as a confound when characterizing a single SM's behavior. |
| `-mem_fetch_stage_latency_debug` (bool) | 0 | Track mean/variance of time spent in each `mem_fetch_status` stage, split into a target-PC-range bucket vs. everything else. |
| `-mem_fetch_stage_latency_period` (uint64) | 10000 | Print-and-reset period (gpu_sim cycles) for the above. |
| `-mem_fetch_stage_latency_pc_lo` / `-mem_fetch_stage_latency_pc_hi` (uint64) | 0 / 0 | Inclusive PC range that counts as the "target" bucket for `-mem_fetch_stage_latency_debug` (hi=0 means the feature is inert — no requests get bucketed as target until you set a real range). |
| `-mem_partition_queue_snapshot_debug` (bool) | 0 | Print instantaneous L2-to-DRAM queue depth for **every memory partition**, every N cycles — good for spotting transient hotspotting on specific channels. |
| `-mem_partition_queue_snapshot_period` (uint64) | 10000 | Period for the above. |
| `-mem_request_trace_debug` (bool) | 0 | `[mem_request_trace]` for every memory response delivered to the target SM: warp, pc, **addr**, type, send/response/duration cycles. This is the one to reach for when you need real per-request latency + address, not just aggregate stats. |
| `-mem_request_trace_sm_id` (int32) | 0 | Target SM id for the above. |

**Recommended order of attack** for "why is kernel X slow": isolate the kernel via filter flags first, then
`-subcore_issue_debug 1 -mem_request_trace_debug 1` together (both bounded to SM0, moderate output) before reaching
for `-issue_wait_trace_debug 1` (every-cycle, much bigger).

Example — debug `flash_attn_ext_vec` alone, isolated to one kernel:

```bash
cd /home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled && \
source ./gpu-simulator/setup_environment_no_git.sh release && \
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread ./gpu-simulator/bin/release/accel-sim.out \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_l2norm_l1dnorm/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM90_H100_l2norm_l1dnorm/trace.config \
  -is_extra_traces_enabled 1 -filter_first_kernel_id 2641 -filter_last_kernel_id 2642 \
  -subcore_issue_debug 1 -mem_request_trace_debug 1 -mem_request_trace_sm_id 0 \
  -trace /home/qshao/Project/Fun/gpu_traces/qwen14b/decode_traces/dynamic_trace.pb \
  > /tmp/fa_debug_trace.log 2>&1
```

Existing always-on heartbeat (no flag needed, useful to gauge progress on a long-running kernel): SM0 prints
`[commit_progress] sm=0 committed_insts=N cycle=C` every 10,000 committed instructions (`remodeling/sm.cc`).

---

## 2. Quantized-weight DRAM compression (research feature, from an earlier session)

CLI flags (`gpu-sim.cc`/`shader.h`):

- `-is_quantized_weight_dram_compression_enabled 1`
- `-quantized_weight_compression_bits <N>` — target bit-width per element on the DRAM side.
- `-quantized_weight_region_file <path>` — JSON region table (kernel_id → base/size), produced by
  `script/detect_weight_regions.py` (see §3).

Mechanism: `dram_t::push()` shrinks `data_size` for reads landing in a registered weight region (tagged via
`mem_fetch::get_kernel_id()`, which required two separate allocator-class fixes to populate correctly —
`shader_core_mem_fetch_allocator` for the inst-tagged path, `partition_mf_allocator` in `l2cache.cc` for L2's
sector-split path); `dram_t::cycle()` restores the original size on the way out so downstream code never sees the
compressed size. Effect is real but config-dependent — only moves cycle count when `dram_atom_size` is small enough
relative to RAS/RCD/CL latencies that the saved data-bus cycles actually matter.

## 3. `script/detect_weight_regions.py` — find a kernel's weight-matrix DRAM address span from the trace alone

No SASS/binary needed — purely from per-CTA recorded addresses. Heuristic: a PC is "weight-like" if every sampled
CTA produced a *distinct* address (each CTA reads a different row of one large buffer), it's not in the simulator's
shared-memory address family (`(addr>>32)==0x3` or `addr<0x100000`), the PC's static opcode is a load (not a store —
an early version mis-classified a per-CTA-varying *write* address and bridged a real 18MB region into a bogus 15GB
span), and the resulting span is ≥1MB (filters out small per-CTA-varying reads like an `ids` gather-index array).

```bash
python3 script/detect_weight_regions.py <dynamic_trace.pb> <kernel_id|start-end> [...]
# e.g.: python3 script/detect_weight_regions.py trace.pb 2364-2404
```

Non-matching kernels in a given range are safely skipped (printed as "no region detected"), so it's fine to pass a
whole kernel-id range spanning unrelated kernel types. Writes `weight_regions.json` next to the script (override with
`OUT_PATH` env var).

## 4. Checking trace completeness during an in-progress remote→local copy

When a trace directory is still being `rsync`'d/copied, don't guess which kernels are safe to simulate — compare the
*expected* CTA count (from the kernel's `grid_dim.x*y*z` in the small, always-fully-written `dynamic_trace.pb`)
against the *actual* count of per-CTA `.pb` files on disk for that kernel:

```python
import sys
sys.path.insert(0, "/tmp/pb_py")   # see §5 for how to generate these bindings
import trace_pb2, os

t = trace_pb2.Trace()
with open("<dir>/dynamic_trace.pb", "rb") as f:
    t.ParseFromString(f.read())

for did, dev in t.gpu_device.items():
    for sid, stream in dev.streams.items():
        for k in stream.kernels:
            kdir = f"<dir>/threadblocks/device_{did}/stream_{sid}/kernel_{k.id}"
            expected = k.grid_dim.x * k.grid_dim.y * k.grid_dim.z
            actual = len(os.listdir(kdir)) if os.path.isdir(kdir) else 0
            print(k.id, k.name[:30], "expected=", expected, "actual=", actual,
                  "OK" if expected == actual else "INCOMPLETE")
```

Only simulate (or filter-window into) kernel ranges where every kernel reports `OK`.

## 5. Generating Python protobuf bindings for ad-hoc trace inspection

The `.proto` schemas live in `util/traces_enhanced/dynamic_trace/*.proto` (`trace`, `gpu_device`, `cuda_stream`,
`kernel`, `threadblock`, `warp`, `instruction`, `address`, `dim3d`). One-time setup:

```bash
mkdir -p /tmp/pb_py
protoc --python_out=/tmp/pb_py -I /home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/util/traces_enhanced/dynamic_trace \
  /home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/util/traces_enhanced/dynamic_trace/*.proto
```

Then `sys.path.insert(0, "/tmp/pb_py"); import trace_pb2, threadblock_pb2` etc. from any one-off Python script. Key
field paths, all confirmed against the actual `.proto` files (don't re-derive from memory, check the schema):

- `Trace.gpu_device[id].streams[id].kernels` — repeated `kernel` (id, name, grid_dim, block_dim, number_of_registers,
  size_shared_memory, shared/local_memory_base_address, function_unique_id).
- `cuda_stream.ordered_cuda_events` — repeated string, drives actual command-list construction in
  `trace_parser.cc::parse_commandlist_file()` (see §6 — the `kernels` field alone is just metadata; *this* field is
  what determines execution order/inclusion).
- `threadblock.warps[id].instructions[].addresses[].base_address` — per-instruction per-lane-aggregated memory
  addresses (already coalesced to one `base_address` per `address` entry, not per-thread).

## 6. Building a synthetic trace that runs one real kernel twice (self-warm test)

Useful whenever you need to isolate "does this kernel benefit from its own prior execution" from everything else in
a natural sequence. No two kernels in a real trace are byte-identical, so this has to be constructed:

1. Pick a real kernel_id (e.g. the one you want to self-warm-test) and locate its `kernel` proto entry and its
   `threadblocks/device_D/stream_S/kernel_K/*.pb` directory in a *complete* (§4) real trace.
2. Build a new `Trace` with one `gpu_device`/`cuda_stream`, containing **two** `kernel` entries that are `CopyFrom()`
   clones of the original, differing only in `id` (use `0` and `1` — must be contiguous starting at the first
   kernel's id, since `trace_parser.cc::parse_kernel_info()` indexes via
   `stream.kernels(kernelid - stream.kernels(0).id())`).
3. Populate `stream.ordered_cuda_events` with `["kernel-0.trace", "kernel-1.trace"]` — this exact `"kernel-<id>.trace"`
   format is required; `trace_parser.cc`'s `parseKernelAndStreamID()` parses the id back out of this string via
   `extractNumberAfterPattern(s, "kernel-")`, and `command.command_string.substr(0,6)=="kernel"` is what makes
   `parse_commandlist_file()` even register it as a kernel-launch command in the first place — the `kernels` list by
   itself is inert without this.
4. Copy the original kernel's `threadblocks/.../kernel_K/d_D_s_S_k_K_<x>,<y>,<z>.pb` files into two new directories
   `kernel_0/` and `kernel_1/`, renaming each file's `k_K_` segment to `k_0_`/`k_1_` respectively (exact filename
   format confirmed from `trace_parser.cc::get_next_threadblock_traces()`, not guessed).
5. Symlink (or copy) the original trace's `extra_info/enhanced_execution_info.json` into the new trace dir unchanged
   — opcode resolution (`inst_trace_t::parse_from_pb` → `traced_execution::get_kernel_by_unique_function_id`) is
   keyed by **kernel `name`** (kept identical to the original in step 2), so the same static-info JSON entry resolves
   correctly for both synthetic copies. This file is required unconditionally (not just when
   `-is_extra_traces_enabled 1` is set) — every instruction's opcode lookup goes through it.

```python
import sys
sys.path.insert(0, "/tmp/pb_py")
import trace_pb2

src = trace_pb2.Trace()
with open("<real_trace_dir>/dynamic_trace.pb", "rb") as f:
    src.ParseFromString(f.read())

orig_kernel = None
for _, dev in src.gpu_device.items():
    for _, stream in dev.streams.items():
        for k in stream.kernels:
            if k.id == TARGET_KERNEL_ID:
                orig_kernel = k
assert orig_kernel is not None

out = trace_pb2.Trace()
out.name = "selfwarm_synthetic"
out.binary_version = src.binary_version
out.nvbit_version = src.nvbit_version
out.accelsim_version = src.accelsim_version
out.is_gathered_registers_values = src.is_gathered_registers_values

dev = out.gpu_device[0]; dev.id = 0
stream = dev.streams[0]; stream.id = 0
for new_id in (0, 1):
    k = stream.kernels.add()
    k.CopyFrom(orig_kernel)
    k.id = new_id
    stream.ordered_cuda_events.append(f"kernel-{new_id}.trace")

with open("<new_trace_dir>/dynamic_trace.pb", "wb") as f:
    f.write(out.SerializeToString())
```

(Shell-side: `cp`/rename the per-CTA `.pb` files into `kernel_0/`/`kernel_1/`, `ln -sf` the `enhanced_execution_info.json`.)

Run normally with `-trace <new_trace_dir>/dynamic_trace.pb` (no filter flags needed — there are only 2 kernels).
Kernel 0's per-kernel `gpu_sim_cycle` line is the cold/alone baseline; kernel 1's is the self-warmed result. Use
`-gpgpu_flush_l2_cache 0` to additionally disable L2 flush between the two launches for a "genuine best-case
locality" variant (see the correlation doc §4 for what this revealed and why the default-flush variant needed a real
bug fix first).

## 7. Computing which DRAM channel/L2 sub-partition an address maps to (non-power-of-2 channel counts)

For a config where `gpgpu_n_mem` isn't a power of 2 (e.g. 80), `addrdec.cc` takes its "gap" branch and computes the
channel via genuine integer modulo on the upper address bits, **not** a masked-bit trick:

```python
def chip_and_subpart(addr, addr_chip_s, bk_mask, n_channel, n_subpart_per_channel):
    chip = (addr >> addr_chip_s) % n_channel
    quotient = (addr >> addr_chip_s) // n_channel
    rest = (quotient << addr_chip_s) | (addr & ((1 << addr_chip_s) - 1))
    bk = 0
    for bit in range(63, -1, -1):
        if bk_mask & (1 << bit):
            bk = (bk << 1) | ((rest >> bit) & 1)
    sub_partition = chip * n_subpart_per_channel + (bk & (n_subpart_per_channel - 1))
    return chip, sub_partition
```

`addr_chip_s` and `bk_mask` come from the config's `-gpgpu_mem_addr_mapping` string (parsed by
`addrdec.cc::addrdec_parseoption` — `"dramid@N;..."` sets `addr_chip_s=N`; each `R`/`B`/`C`/`S` character in the
remainder marks one address bit, counting down from bit 63, with `B` bits forming `bk_mask`). Read the actual config
string before assuming bit positions — don't hardcode them from one investigation to the next.

Sub-partition array layout (for `m_memory_sub_partition[]` indexing, e.g. when reading `L2_cache_bank[N]` stats):
`submpid = channel_id * n_sub_partition_per_mchannel + local_subpart` (channel-major, confirmed in
`gpu-sim.cc`'s constructor loop) — so `L2_cache_bank[N]` for sub-partition `N` corresponds to channel
`N / n_sub_partition_per_mchannel`.

## 8. Known simulator bug fixed this session: L2 only half-flushed between kernel launches

`gpu-sim.cc`, the `-gpgpu_flush_l2_cache` block: the invalidation loop used `m_memory_config->m_n_mem` (channel
count) as its bound instead of `m_memory_config->m_n_mem_sub_partition` (total L2 sub-partition count), so for any
config where these two differ (i.e. `gpgpu_n_sub_partition_per_mchannel > 1`), only the **first** `m_n_mem`
sub-partitions ever got flushed — the rest silently retained their content across every kernel boundary for the
entire simulation run, no matter how many kernels ran. Fixed to use `m_n_mem_sub_partition`. Full discovery story
and what it did/didn't explain: [`decode-attn-qwen3-8b-14b-correlation.md`](decode-attn-qwen3-8b-14b-correlation.md)
§4.2. **Symptom to watch for in any future investigation using `-gpgpu_flush_l2_cache 1`**: a clean, sustained
miss-rate split in `L2_cache_bank[N]` stats exactly at `N = m_n_mem` (not periodic, not gradual) is this bug, not a
real cache-locality effect — check the fix is actually present in the binary you're running before chasing it again.

## 9. Kernel-name demangling in status-line output (no flag needed, always on)

The `gpu_sim_cycle = ... kernel = <name>` status lines in `gpu-sim.cc` already demangle the kernel name via
`abi::__cxa_demangle`, including handling this tracer's non-standard trailing `___<digits>` suffix (stripped before
demangling is attempted, since it isn't valid Itanium mangling on its own) — no extra flag required to get readable
kernel names instead of raw mangled symbols in any `gpu_sim_cycle`/`gpu_tot_sim_cycle` log line.
