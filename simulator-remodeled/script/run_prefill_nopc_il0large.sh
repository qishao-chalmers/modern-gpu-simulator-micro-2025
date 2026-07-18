#!/usr/bin/env bash
# Experiment: perfect inst/const cache OFF; enlarge L0I (il0 8→64 sets) — baseline l2_rop/l1d unchanged
# nopc alone overshoots (+65%); this targets the L0I reservation-fail pathology
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT/log/prefill_nopc_il0large"

OMP_NUM_THREADS=16 OMP_PROC_BIND=spread \
"$ROOT/gpu-simulator/bin/release/accel-sim.out" \
  -config "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_nopc_il0large/gpgpusim.config" \
  -config "$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config" \
  -is_extra_traces_enabled 1 -filter_first_kernel_id 1 -filter_last_kernel_id 3 \
  -trace /home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb \
  | tee "$ROOT/log/prefill_nopc_il0large/sim.log"
