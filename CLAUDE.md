# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

MICRO 2025 enhanced GPU simulator (Accel-Sim lineage). Key additions over upstream Accel-Sim:
- Redesigned SM (sub-core pipeline, L0 I-cache, stream-buffer prefetcher)
- Enhanced NVBit tracer with **Protocol Buffers** output (not classic text traces)
- OpenMP-parallel simulator, AccelWattch energy integration
- Hopper/Blackwell opcode headers, SM90_H100 config

Active goal: re-trace llama.cpp decode kernels (IDs 2364–2404) on H100/sm_90 with the protobuf tracer, then simulate with SM90_H100 config.

## Build

All commands run from `simulator-remodeled/`.

### Prerequisites

```bash
sudo apt install protobuf-compiler libprotobuf-dev build-essential g++-10 cmake
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
```

BSC MN5 cluster:
```bash
module load protobuf/24 GCCcore/12 CUDA/12.2
unset CXX CC   # critical: build.sh auto-selects host g++
```

### One-shot build (tracer + simulator)

```bash
cd simulator-remodeled
BUILD_SYSTEM=cmake ./build.sh           # cmake (recommended)
./build.sh                              # Makefile fallback
ARCH=sm_90 BUILD_SYSTEM=cmake ./build.sh  # H100 tracer
```

Outputs:
- `util/tracer_nvbit/tracer_tool/tracer_tool.so`
- `gpu-simulator/bin/release/accel-sim.out`

### Rebuild tracer only (after tracer_tool.cu changes)

```bash
cd simulator-remodeled
ARCH=sm_90 make -C util/tracer_nvbit/tracer_tool -j
```

### Rebuild simulator only (Makefile path)

```bash
cd simulator-remodeled/gpu-simulator
source ./setup_environment_no_git.sh release
make -j
```

## Smoke test

```bash
cd simulator-remodeled
./run_example.sh                    # Rodinia backprop, Ampere
CONFIG=SM90_H100 ./run_example.sh   # same trace, H100 config
```

## Trace format — PROTOBUF ONLY

The modern simulator does **not** accept classic Accel-Sim text traces. Pass only `dynamic_trace.pb`:

```
traces/
  dynamic_trace.pb                      # main protobuf (pass to -trace)
  threadblocks/device_*/stream_*/kernel_*/*.pb   # per-CTA protobuf
  extra_info/enhanced_execution_info.json        # static metadata
```

There is **no converter** from `kernelslist.g` + `kernel-*.trace.xz` to protobuf.

## Running simulations

```bash
cd simulator-remodeled

# Generic protobuf trace
CONFIG=SM90_H100 FILTER_FIRST=2364 FILTER_LAST=2393 \
  ./run_custom_trace.sh /path/to/traces/dynamic_trace.pb

# H100 decode trace (wrapper with defaults)
./run_decode_trace.sh /path/to/traces/dynamic_trace.pb

# Manual invocation
source ./gpu-simulator/setup_environment_no_git.sh release
OMP_NUM_THREADS=32 OMP_PROC_BIND=spread \
  ./gpu-simulator/bin/release/accel-sim.out \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config \
  -is_extra_traces_enabled 1 \
  -trace /path/to/traces/dynamic_trace.pb
```

## Re-tracing on H100 (cluster)

```bash
cd simulator-remodeled

# Build tracer for Hopper
module load protobuf/24 GCCcore/12 CUDA/12.2 && unset CXX CC
ARCH=sm_90 BUILD_SYSTEM=cmake ./build.sh

# Re-trace decode kernels
KERNEL_START=2364 KERNEL_END=2404 \
  ./trace_decode_h100.sh /path/to/llama-bench \
  -m /path/to/Qwen3-8B-Q8_0.gguf \
  -p 1024 -n 64 -b 2048 -ngl 99 --flash-attn 1 --no-warmup -r 1
```

Output lands in `../gpu_traces/decode_retrace/traces/`.

## Architecture

### Data flow

```
GPU app + tracer_tool.so  →  dynamic_trace.pb + threadblocks/*.pb
                                      ↓
                          accel-sim.out (trace-driven mode)
                                      ↓
                              stats, energy output
```

### Key components

| Path | Role |
|------|------|
| `util/tracer_nvbit/tracer_tool/tracer_tool.cu` | **All tracer logic** (NVBit, protobuf serialization, CUDA graph fixes) |
| `util/traces_enhanced/` | Protobuf schemas + static metadata |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/` | Redesigned SM core model |
| `gpu-simulator/trace-parser/trace_parser.cc` | Reads `dynamic_trace.pb` + per-CTA `.pb` files |
| `gpu-simulator/ISA_Def/` | Opcode headers: `hopper_opcode.h`, `blackwell_opcode.h`, etc. |
| `gpu-simulator/configs/tested-cfgs/SM90_H100/` | H100 `trace.config` |
| `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/` | H100 `gpgpusim.config` |

### Tracer internals — CUDA graph handling

llama.cpp uses CUDA graphs (`ggml_backend_cuda_graph_compute`). The tracer must follow accel-sim-framework@d1697c7 (PR #427) behavior:

| Phase | Correct behavior |
|-------|-----------------|
| `cuLaunchKernel enter` (stream capturing) | Instrument + record metadata; set `recv_thread_receiving = true` |
| `cuLaunchKernel exit` (stream capturing) | **Skip** sync/flush (`is_stream_capturing` check) |
| `cuGraphLaunch exit` | Release mutex → `cudaStreamSynchronize` → `flush_channel` → wait recv thread → re-acquire mutex; serialize threadblocks only if `!stop_report` |

Three bugs that were fixed (all in `tracer_tool.cu`):
1. **Kernel filter crash** — recv thread indexed `mutable_kernels()` by global kernel_id; fix: skip when `stream.kernels_size() == 0`
2. **Stream capture CUDA error** — `finalize` called `cudaDeviceSynchronize` during capture; fix: guard with `!is_stream_capturing(p->hStream)`
3. **cuGraphLaunch segfault** — finalize held `pthread_mutex` during CUDA sync calls; fix: `pthread_mutex_unlock` before `finalize_cuda_graph_launch`

## GPU configs

| Config | Location | Notes |
|--------|----------|-------|
| `SM86_RTX3080` | `configs/tested-cfgs/SM86_RTX3080/` | Reference Ampere config |
| `SM86_RTXA6000` | `configs/tested-cfgs/SM86_RTXA6000/` | Ampere professional |
| `SM89_RTX4090` | `configs/tested-cfgs/SM89_RTX4090/` | Ada Lovelace |
| `SM90_H100` | `configs/tested-cfgs/SM90_H100/` | Hopper (132 SMs, sm_90) |

SM90_H100 constraint: `gpgpu_n_mem` and L2 set count must be in {16,32,64,128,256,512,1024} (IPOLY indexing).

## Critical pitfalls

1. **Never pass `kernelslist.g`** to `-trace` — only `dynamic_trace.pb` works.
2. **Never copy `build-cmake/`** between machines — CMakeCache embeds absolute paths; delete and reconfigure.
3. **Protobuf on cluster** — always `module load protobuf/24`; never set `CPATH=/usr/include` (breaks `stdlib.h` via `#include_next`).
4. **Cross-compiler** — `unset CXX CC` before builds; `build.sh` ignores cross-compilers and selects host g++.
5. **`setup_environment_no_git.sh` must be sourced** in the same shell before running `make` or `accel-sim.out` manually.
6. **NVCC and `-pthread`** — only pass `PROTOBUF_INCLUDES` (-I/-D flags) to NVCC, not `PROTOBUF_CFLAGS` (which includes `-pthread`).
7. **gpgpu-sim is Makefile-only** — CMake invokes it via `cmake/build_gpgpusim.sh`.
8. **Tracer output path** — protobuf writes to `./traces/` relative to cwd; do not rely on `USER_DEFINED_FOLDERS=1` for `dynamic_trace.pb`.
9. **`-is_extra_traces_enabled 1`** requires `extra_info/enhanced_execution_info.json` to exist in the trace directory.

## Related paths

- Classic traces (old format, cannot simulate): `/home/qshao/Project/Fun/gpu_traces/decode/`
- Classic tracer reference (CUDA graph support): `/home/qshao/Project/Fun/accel-sim-framework/util/tracer_nvbit` @ commit `d1697c7`
- BSC MN5 cluster path: `/gpfs/projects/bsc93/bsc747505/modern-gpu-simulator-micro-2025/`
