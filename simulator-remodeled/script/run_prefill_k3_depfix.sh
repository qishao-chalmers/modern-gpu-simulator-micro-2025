#!/usr/bin/env bash
# k3-only run with dependency-chain fixes (scoreboard@EX, SFU latency, IMMA decouple)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE="${TRACE:-/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb}"
LOG_DIR="${LOG_DIR:-$ROOT/log/prefill_k3_compare}"
mkdir -p "$LOG_DIR"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
"$ROOT/gpu-simulator/bin/release/accel-sim.out" \
  -config "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_k3_depfix/gpgpusim.config" \
  -config "$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config" \
  -is_extra_traces_enabled 1 -filter_first_kernel_id 3 -filter_last_kernel_id 3 \
  -trace "$TRACE" \
  | tee "$LOG_DIR/depfix_k3.log"
