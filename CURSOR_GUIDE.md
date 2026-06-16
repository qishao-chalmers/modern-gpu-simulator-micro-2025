# Cursor / Agent Guide — modern-gpu-simulator-micro-2025

**Read this first** when starting work in this repository. It summarizes project context, what has been done, known pitfalls, and planned next steps.

---

## 1. What this repo is

This is the **MICRO 2025 remodeled GPU simulator** (Accel-Sim lineage) from UPC/BSC-related work. It extends Accel-Sim with:

- Redesigned SM (sub-core pipeline, memory pipeline, L0 I-cache, prefetch)
- **Enhanced NVBit tracer** with control bits + **Protocol Buffers** traces
- OpenMP-parallel simulator, AccelWattch integration
- Hopper/Blackwell opcode headers (`hopper_opcode.h`, `blackwell_opcode.h`)

| Path | Role |
|------|------|
| `simulator-remodeled/` | **Main working tree** — tracer, simulator, configs, examples |
| `APEs/` | Absolute percentage error results (not build-related) |
| `README.md` | Paper citations and high-level feature list |

**Related repos (outside this tree):**

| Path | Notes |
|------|-------|
| `/home/qshao/Project/Fun/accel-sim-framework` | Upstream Accel-Sim; **classic tracer** at `util/tracer_nvbit` @ commit `d1697c7` (PR #427) works with llama.cpp CUDA graphs — use as behavioral reference |
| `/home/qshao/Project/Fun/gpu_traces/decode/` | User's H100 decode-layer traces (classic format — see §6) |

**Cluster path (BSC MN5):**  
`/gpfs/projects/bsc93/bsc747505/modern-gpu-simulator-micro-2025/` (or `~/project/modern-gpu-simulator-micro-2025`)

---

## 2. Repository layout (`simulator-remodeled/`)

```
simulator-remodeled/
├── build.sh                 # One-shot build: tracer + simulator (make or cmake)
├── run_example.sh           # Smoke test: Rodinia backprop protobuf trace
├── run_custom_trace.sh      # Run user protobuf traces (not kernelslist.g)
├── trace_decode_h100.sh     # Re-trace llama.cpp decode on H100 → protobuf
├── run_decode_trace.sh      # Simulate decode protobuf traces (SM90_H100 default)
├── CMakeLists.txt           # Top-level CMake (optional)
├── cmake/
│   ├── ModernGpuSimEnvironment.cmake
│   └── build_gpgpusim.sh    # HPC-safe gpgpu-sim sub-build for CMake
├── util/
│   ├── traces_enhanced/     # Protobuf schemas + traced_* metadata
│   ├── protobuf_deps.mk     # Protobuf/Abseil link flags (Makefile builds)
│   └── tracer_nvbit/        # Modern NVBit tracer → dynamic_trace.pb
├── gpu-simulator/
│   ├── bin/release/accel-sim.out   # Simulator binary (after build)
│   ├── gpgpu-sim/           # Bundled GPGPU-Sim (Makefile only)
│   ├── trace-parser/        # Parses dynamic_trace.pb + threadblock .pb
│   ├── trace-driven/
│   ├── configs/tested-cfgs/ # SM86, SM89, SM90_H100, SM120, …
│   └── ISA_Def/             # hopper_opcode.h, blackwell_opcode.h, …
└── exampleTraces/           # rodinia2Ampere.tar.gz → protobuf example traces
```

---

## 3. Trace formats (critical)

### Modern simulator accepts **protobuf only**

| Artifact | Format | Used by modern simulator? |
|----------|--------|---------------------------|
| `traces/dynamic_trace.pb` | Protobuf index | **Yes** — pass to `-trace` |
| `traces/threadblocks/device_*/stream_*/kernel_*/*.pb` | Per-CTA protobuf | **Yes** |
| `traces/extra_info/enhanced_execution_info.json` | Static metadata | **Yes** if `-is_extra_traces_enabled 1` |
| `kernelslist.g` + `kernel-*.trace.xz` | Classic Accel-Sim text | **No** |

There is **no converter** from classic → protobuf in this repo. H100 traces collected with the **old** accel-sim-framework tracer must be **re-traced** with the modern tracer in `util/tracer_nvbit/`.

### User decode traces (example)

- Location: `/home/qshao/Project/Fun/gpu_traces/decode/`
- Contents: `kernelslist.g` + 41× `kernel-2364…2404*.trace.xz` (classic, **sm_90** / H100)
- Kernel IDs **2364–2404** ≈ llama.cpp decode phase (~2 tg steps × ~21 kernels/step)
- Filter one decode slice: `FILTER_FIRST=2364 FILTER_LAST=2393` (30 kernels)
- **Cannot** run with `./run_custom_trace.sh` until re-traced to `dynamic_trace.pb`

---

## 4. Build system

### Prerequisites

- CUDA + NVCC (11.x+; Hopper tracing needs CUDA 12+)
- Host `g++` / `gcc` (not RISC-V cross-compiler in `$CXX`)
- Protobuf: system 3.x (local) or **module protobuf/24** (cluster) + Abseil
- `make`, `cmake` (3.17+ on cluster: `/gpfs/apps/MN5/GPP/CMAKE/3.25.1/bin/cmake`)

### Local (Debian/Ubuntu-style)

```bash
cd simulator-remodeled
unset CXX CC   # if cross-compiler set
BUILD_SYSTEM=cmake ./build.sh
```

Outputs:

- Tracer: `util/tracer_nvbit/tracer_tool/tracer_tool.so`
- Simulator: `gpu-simulator/bin/release/accel-sim.out`

### Cluster (BSC MN5)

```bash
module load protobuf/24 GCCcore/12 CUDA/12.2
# module load cmake/3.25.1   # if cmake not in PATH
unset CXX CC

cd simulator-remodeled
rm -rf build-cmake   # NEVER copy build-cmake from another machine
BUILD_SYSTEM=cmake ./build.sh
```

`build.sh` now:

- Prefers `$EBROOTGCCCORE/bin/g++` over `/usr/bin/g++`
- Sets protobuf include/lib paths from pkg-config
- Wipes and reconfigures `build-cmake` each cmake build
- Avoids `CPATH=/usr/include` (breaks g++ `#include_next`)

### Makefile-only simulator (no CMake)

```bash
cd simulator-remodeled/gpu-simulator
source ./setup_environment_no_git.sh release
make -j
```

---

## 5. Project state (June 2025)

### Goal

Simulate **llama.cpp decode-layer** workloads on **H100 (sm_90)** using the **modern protobuf tracer** and remodeled accel-sim, starting from classic traces already collected for kernels **2364–2404**.

### What works today

| Component | Status |
|-----------|--------|
| Local CMake/Makefile build (`build.sh`) | ✅ Tracer + `accel-sim.out` build |
| `run_example.sh` (Rodinia protobuf smoke test) | ✅ ~18–22 s locally |
| `SM90_H100` gpgpusim + trace config | ✅ Added; smoke-tested with Rodinia (`CONFIG=SM90_H100 ./run_example.sh`) |
| `run_custom_trace.sh` / `run_decode_trace.sh` | ✅ Ready once protobuf decode traces exist |
| `trace_decode_h100.sh` | ✅ Documents env vars + kernel limits for H100 re-trace |
| Classic decode traces (`gpu_traces/decode/`) | ✅ Present (`kernelslist.g`, 41× `.trace.xz`, `binary version = 90`) |
| Cluster classic tracer (`accel-sim-framework` @ `d1697c7`) | ✅ User-confirmed working with llama.cpp on MN5 |
| **Tracer code review vs d1697c7** | ✅ All 3 CUDA-graph bugs verified present and correct (2025-06-15) |
| **Protobuf re-trace of decode on H100** | ⏳ Needs cluster rebuild + run (`ARCH=sm_90 make -C util/tracer_nvbit/tracer_tool -j`) |
| Full cluster end-to-end (`accel-sim.out` on MN5) | ⏳ Build progressed; not fully confirmed after latest tracer fixes |

### Blocking path

```
Classic traces (gpu_traces/decode/)  ──X──>  modern accel-sim.out
                                              (no converter in repo)

H100 + modern tracer_tool.so  ──>  traces/dynamic_trace.pb  ──>  run_decode_trace.sh
```

---

## 6. Session work log

### Build / portability (earlier sessions)

- [x] `build.sh` — unified tracer + simulator build; host compiler selection; protobuf setup
- [x] CMake support: root + `gpu-simulator/` + `util/traces_enhanced/` + trace-parser/trace-driven
- [x] `util/protobuf_deps.mk` — Abseil link deps for protobuf 24; module `-I` paths; skip `-I/usr/include`
- [x] `protobuf_deps.mk` splits `PROTOBUF_INCLUDES` (-I/-D only) vs `PROTOBUF_COMPILE_FLAGS` (-pthread) for NVCC
- [x] `tracer_tool/Makefile` — `HOST_CC`/`HOST_CXX`; pass correct CC to traces_enhanced
- [x] `traces_enhanced/Makefile` — protobuf dep ordering fix; race fix (`depend: protobuf $(OBJ)`)
- [x] `cmake/build_gpgpusim.sh` — full PATH for cluster; fix `OPENCL_REMOTE_GPU_HOST` unbound var
- [x] `gpgpu-sim/setup_environment` — robust GCC version detect; `${LD_LIBRARY_PATH:-}`; makedepend stub
- [x] `gpgpu-sim/src/cuda-sim/Makefile` — fix `instructions.h` generation (`$@` not `$*.h`)
- [x] `gpgpu-sim/build-tools/bin/makedepend` — no-op when system makedepend missing
- [x] `.gitignore` — `build-cmake/`

### H100 decode workflow (recent sessions)

- [x] **`SM90_H100`** config — `gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config` + `gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config` (132 SMs, sm_90, remodeled SM; IPOLY-safe `gpgpu_n_mem=32`, L2 `S:128`)
- [x] **`trace_decode_h100.sh`** — H100 re-trace wrapper (`ARCH=sm_90`, `DYNAMIC_KERNEL_LIMIT_*`, `LD_PRELOAD`)
- [x] **`run_decode_trace.sh`** — simulation wrapper (default `SM90_H100`, `FILTER_FIRST=2364`, `FILTER_LAST=2393`)
- [x] **`run_custom_trace.sh`** — default config changed to `SM90_H100`
- [x] **`.cursor/agents/h100-decode-trace.md`** — project subagent for this workflow
- [x] **Enhanced tracer fixes** in `util/tracer_nvbit/tracer_tool/tracer_tool.cu` (see §7)

### Verified

- [x] Local: tracer + `accel-sim.out` via CMake; `run_example.sh` completes
- [x] Local: `CONFIG=SM90_H100 ./run_example.sh` completes (after IPOLY L2 fix)
- [x] Cluster: tracer builds with protobuf 24 + Abseil; simulator cmake build progressed through gpgpu-sim

---

## 7. H100 tracer issues and fixes

All tracer fixes are in **one file**:

**`simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu`**

Reference implementation (works on cluster with llama.cpp):

**`accel-sim-framework/util/tracer_nvbit`** @ git commit **`d1697c7`** (PR #427 — CUDA graph + kernel-range support).  
Produces **classic** `kernel-*-ctx_*.trace.xz` + `kernelslist.g` (not protobuf).

### Code review status (2025-06-15)

Code has been refactored relative to the previously-documented state. All three original bugs are verified fixed, and several new improvements were added. Details below.

| # | Symptom | Root cause | Fix / Status |
|---|---------|------------|-------------|
| 1 | `repeated_ptr_field.h: Check failed: index < current_size_` with `DYNAMIC_KERNEL_LIMIT_START=2364` | Recv thread indexed `mutable_kernels()` by global kernel_id but protobuf array only has traced kernels | ✅ `if (stream.kernels_size() == 0) { ...; continue; }` in recv thread |
| 2 | `CUDA error … operation not permitted when stream is capturing` | Per-kernel exit called `cudaDeviceSynchronize` + `flush_channel` during CUDA graph stream capture | ✅ `handle_kernel_launch_exit` returns early when `is_stream_capturing(cfg.hStream)` |
| 3 | Segfault after CUDA graph warmup | `cuGraphLaunch` exit finalized protobuf while holding `pthread_mutex`; mutex deadlocked with CUDA callbacks | ✅ `pthread_mutex_unlock` before `finalize_cuda_graph_launch()` |

### New improvements in current code

| Improvement | Detail |
|-------------|--------|
| **All kernel launch variants** | Now handles `cuLaunchKernelEx`, `cuLaunchCooperativeKernel`, `cuLaunchGridAsync` (was missing these) |
| **`cuGraphAddKernelNode` support** | Manual graph builds now hook kernel enter (passes `build_graph=true`) |
| **`KernelLaunchConfig` + `extract_launch_config`** | Clean helper struct/function shared across all kernel launch callbacks |
| **`handle_kernel_launch_enter/exit`** | Refactored from inline code; separates concerns cleanly |
| **Checkpoint after `cuGraphLaunch`** | `write_dynamic_trace_checkpoint()` writes `dynamic_trace.pb` immediately after each traced graph launch — safe with `TERMINATE_UPON_LIMIT=1` |
| **`traced_kernels_since_graph_launch` counter** | Per-device; guards graph-launch stats/threadblock write more correctly than `!stop_report` |
| **`first_call` init logic** | Fixed operator-precedence bug in `active_region` init; now explicit `if/else` |
| **`TRACE_LOG` macro** | Debug logging gated on `TOOL_VERBOSE=1` (off by default) |

### CUDA graph behavior (llama.cpp)

llama.cpp uses **CUDA graphs** (`ggml_backend_cuda_graph_compute`). The tracer mirrors classic #427:

| Phase | Correct behavior |
|-------|------------------|
| **Stream capture** (graph build / warmup) | On kernel **enter**: instrument + record metadata. On kernel **exit**: **return early** (no sync, no flush). |
| **`cuGraphLaunch` exit** | Release mutex → `cudaStreamSynchronize(stream)` → `flush_channel` → wait recv thread → re-acquire mutex. Write stats + threadblocks only if `had_traced_kernels \|\| had_threadblocks`. Then write checkpoint. |
| **Normal kernel launch** (not capturing) | Full per-kernel finalize: sync → flush → recv drain → stats → threadblock `.pb` (mutex released during CUDA calls). |
| **Manual graph node** (`cuGraphAddKernelNode`) | Kernel enter only; no exit callback; finalized by next `cuGraphLaunch`. |

### Rebuild after tracer changes

```bash
module load protobuf/24 GCCcore/12 CUDA/12.2
unset CXX CC
cd simulator-remodeled
ARCH=sm_90 make -C util/tracer_nvbit/tracer_tool -j
# or full rebuild: BUILD_SYSTEM=cmake ./build.sh
```

Enable verbose logging to trace execution flow:
```bash
TOOL_VERBOSE=1 LD_PRELOAD=./tracer_tool.so ./llama-bench ...
```

### If tracing still fails

```bash
gdb --args env LD_PRELOAD=./tracer_tool.so ./llama-bench ...
# (gdb) run
# (gdb) bt
```

Likely crash locations: `cuGraphLaunch`, `flush_channel`, `recv_thread_fun`, `pc_to_opcode` lookup.

### Not yet resolved

- Protobuf re-trace **validated end-to-end** on MN5 (pending user rebuild + run)

---

## 8. Run simulation

### Bundled example (protobuf)

```bash
cd simulator-remodeled
./run_example.sh
# CONFIG=SM89_RTX4090 ./run_example.sh
```

### H100 re-trace on cluster (MN5)

```bash
module load protobuf/24 GCCcore/12 CUDA/12.2
unset CXX CC
cd simulator-remodeled
ARCH=sm_90 make -C util/tracer_nvbit/tracer_tool -j

export DYNAMIC_KERNEL_LIMIT_START=2364
export DYNAMIC_KERNEL_LIMIT_END=2404
export TERMINATE_UPON_LIMIT=1

mkdir -p ~/decode_retrace && cd ~/decode_retrace
LD_PRELOAD=/path/to/tracer_tool.so \
CUDA_INJECTION64_PATH=/path/to/tracer_tool.so \
  /home/bsc/bsc747505/project/llama.cpp/build_release/bin/llama-bench \
  -m .../Qwen3-8B-Q8_0.gguf -p 1024 -n 64 -b 2048 -ngl 99 --flash-attn 1 --no-warmup -r 1
```

Expected output under `./traces/`:

- `dynamic_trace.pb`
- `threadblocks/device_*/stream_*/kernel_*/*.pb`
- `extra_info/enhanced_execution_info.json`

---

### Simulate decode protobuf traces

```bash
# After successful re-trace:
./run_decode_trace.sh ~/decode_retrace/traces/dynamic_trace.pb

# Or manually:
FILTER_FIRST=2364 FILTER_LAST=2393 CONFIG=SM90_H100 \
  ./run_custom_trace.sh ~/decode_retrace/traces/dynamic_trace.pb
```

### GPU config for H100 / Hopper traces

- **Primary:** `SM90_H100` — configs at `gpgpu-sim/configs/tested-cfgs/SM90_H100/` and `gpu-simulator/configs/tested-cfgs/SM90_H100/` (132 SMs, sm_90, remodeled SM; initial public-spec proxy)
- **Fallback:** `SM89_RTX4090` (Ada) if Hopper timing not yet tuned
- Opcode decode uses `ISA_Def/hopper_opcode.h` when traces report `binary version = 90`
- Timing accuracy for Hopper requires AccelWattch tuner / microbenchmark validation

---

## 9. Planned work (backlog)

### High priority

- [ ] **Rebuild enhanced tracer on MN5** with latest `tracer_tool.cu` and validate llama-bench re-trace (no crash after graph warmup)
- [ ] **Confirm full cluster build** end-to-end (`accel-sim.out` on MN5)
- [ ] **Re-trace decode layer on H100** → `dynamic_trace.pb` (kernels 2364–2404)
- [ ] **Run decode simulation** via `run_decode_trace.sh`; compare with hardware / classic trace kernel list
- [x] **SM90/H100 gpgpusim.config** — initial `SM90_H100` (needs AccelWattch tuning)

### Medium priority

- [x] Document H100 tracing env vars (`trace_decode_h100.sh`, `ARCH=sm_90`, kernel limits)
- [x] Document tracer CUDA-graph fixes and classic reference (`d1697c7`)
- [x] Port remaining classic tracer #427 structure (`cuGraphAddKernelNode`, full `enter/leave_kernel_launch` split) — done in 2025-06 refactor
- [ ] Optional: legacy `kernelslist.g` → protobuf converter (large effort)
- [ ] Sync tracer fixes back to `accel-sim-framework` if desired
- [ ] CI or Slurm script for build + smoke test

### Low priority / research

- [ ] Compare simulated vs H100 hardware for decode layer (APE in `APEs/`)
- [ ] Blackwell (`blackwell_opcode.h`) tracing + config when hardware available

---

## 10. Pitfalls for agents

1. **Do not copy `build-cmake/`** between machines — CMakeCache embeds absolute paths.
2. **Do not pass `kernelslist.g` to `-trace`** — simulator expects `dynamic_trace.pb`.
3. **Cluster protobuf:** always `module load protobuf/24`; ensure `-I$EBROOTPROTOBUF/include` on compile line.
4. **Never set `CPATH=/usr/include`** for protobuf on local Debian — breaks `stdlib.h` via `#include_next`.
5. **NVCC and `-pthread`:** only pass `PROTOBUF_INCLUDES` to NVCC, not full `PROTOBUF_CFLAGS`.
6. **Compiler on cluster:** use GCCcore module (`$EBROOTGCCCORE`), not `/usr/bin/g++` 11.x.
7. **gpgpu-sim is Makefile-only** — CMake builds it via `cmake/build_gpgpusim.sh`.
8. **Cross-compiler in env:** user often has `CXX=riscv64-...-g++`; `build.sh` ignores it for CUDA builds.
9. **Kernel filter + tracer:** `DYNAMIC_KERNEL_LIMIT_START > 1` needs fixed recv thread (see §7). Rebuild `tracer_tool.so` after pull.
10. **CUDA graphs (llama.cpp):** never sync/flush on kernel exit during stream capture; flush on `cuGraphLaunch` exit only; release mutex before CUDA calls. Reference: `accel-sim-framework` @ `d1697c7`.
11. **SM90 IPOLY hashing:** `gpgpu_n_mem` and L2 set count (`gpgpu_cache:dl2 S:N:…`) must be 16, 32, 64, 128, 256, 512, or 1024.
12. **Tracer output path:** protobuf writes to `./traces/` relative to cwd; do not rely on `USER_DEFINED_FOLDERS=1` alone.
13. **Minimize scope** — avoid unrelated refactors in this large fork.

---

## 11. Key files

| File | Purpose |
|------|---------|
| `simulator-remodeled/build.sh` | Main build entry point |
| `simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu` | **Enhanced NVBit tracer** (all H100/graph fixes here) |
| `simulator-remodeled/trace_decode_h100.sh` | H100 re-trace helper |
| `simulator-remodeled/run_decode_trace.sh` | Decode simulation wrapper |
| `simulator-remodeled/run_custom_trace.sh` | Generic protobuf trace runner |
| `simulator-remodeled/run_example.sh` | Rodinia smoke test |
| `simulator-remodeled/util/protobuf_deps.mk` | Protobuf/Abseil flags |
| `simulator-remodeled/cmake/build_gpgpusim.sh` | Cluster-safe gpgpu-sim make |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/` | H100 gpgpusim.config |
| `simulator-remodeled/gpu-simulator/configs/tested-cfgs/SM90_H100/` | H100 trace.config |
| `.cursor/agents/h100-decode-trace.md` | Cursor subagent for this workflow |
| `CURSOR_GUIDE.md` | This file |

---

## 12. Quick diagnostic commands

```bash
# Binary exists?
test -x simulator-remodeled/gpu-simulator/bin/release/accel-sim.out && echo OK

# Protobuf trace layout?
ls traces/dynamic_trace.pb traces/threadblocks/

# Classic trace (wrong format for modern sim)?
head traces/kernelslist.g   # lists kernel-*.trace.xz

# CMake cache from wrong machine?
grep CMAKE_HOME_DIRECTORY build-cmake/CMakeCache.txt

# Compiler used by build.sh
BUILD_SYSTEM=cmake ./build.sh 2>&1 | grep HOST_CC
```

---

## 13. Suggested Cursor session prompts

- *"Build simulator-remodeled on MN5 with module load protobuf/24 GCCcore/12 CUDA/12.2"*
- *"Run run_example.sh and fix any runtime linker errors"*
- *"Help re-trace my decode workload on H100 with the modern NVBit tracer"*
- *"Create SM90 gpgpusim.config based on SM89_RTX4090 and Hopper public specs"*
- *"Run FILTER_FIRST=2364 FILTER_LAST=2393 on my dynamic_trace.pb"*

- *"Use h100-decode-trace subagent to validate protobuf re-trace on MN5"*
- *"Debug segfault in tracer_tool.cu after CUDA graph warmup (compare with accel-sim d1697c7)"*

---

## 14. References

- `simulator-remodeled/README.md` — Accel-Sim trace-driven overview
- `simulator-remodeled/util/tracer_nvbit/README.md` — tracer usage (partially outdated re: kernelslist.g)
- `simulator-remodeled/gpu-simulator/gpgpu-sim4.md` — GPGPU-Sim 4 / trace-driven model
- `accel-sim-framework/util/tracer_nvbit` @ **`d1697c7`** — classic tracer with CUDA graph support (PR #427); behavioral reference for llama.cpp
- Top-level `README.md` — MICRO 2025 / ISPASS citations

---

*Last updated: 2025-06 — H100 decode workflow, SM90_H100 config, enhanced tracer fixes (kernel filter + CUDA graphs), project state summary.*
