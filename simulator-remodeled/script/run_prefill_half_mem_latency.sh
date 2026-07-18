#!/usr/bin/env bash
# SM90_H100_half_mem_latency: best_optA with L1D/L2/DRAM latencies halved (upper-bound sweep)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE="${TRACE:-/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb}"
FILTER_FIRST="${FILTER_FIRST:-1}"
FILTER_LAST="${FILTER_LAST:-3}"

mkdir -p "$ROOT/log/prefill_half_mem_latency"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
"$ROOT/gpu-simulator/bin/release/accel-sim.out" \
  -config "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_half_mem_latency/gpgpusim.config" \
  -config "$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config" \
  -is_extra_traces_enabled 1 \
  -filter_first_kernel_id "$FILTER_FIRST" -filter_last_kernel_id "$FILTER_LAST" \
  -trace "$TRACE" \
  | tee "$ROOT/log/prefill_half_mem_latency/sim.log"
