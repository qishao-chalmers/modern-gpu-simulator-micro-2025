#!/usr/bin/env bash
# Experiment: perfect inst/const cache OFF; shorter global path (l2_rop 242→180, l1d_min 13→10)
# Compensates nopc overshoot on the data side while keeping default (undersized) L0I
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT/log/prefill_nopc_l2rop180_l1d10"

OMP_NUM_THREADS=16 OMP_PROC_BIND=spread \
"$ROOT/gpu-simulator/bin/release/accel-sim.out" \
  -config "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_nopc_l2rop180_l1d10/gpgpusim.config" \
  -config "$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config" \
  -is_extra_traces_enabled 1 -filter_first_kernel_id 1 -filter_last_kernel_id 3 \
  -trace /home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb \
  | tee "$ROOT/log/prefill_nopc_l2rop180_l1d10/sim.log"
