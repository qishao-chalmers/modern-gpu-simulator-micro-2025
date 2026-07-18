#!/usr/bin/env bash
# Experiment: L1D miss queue 16 → 32 (baseline mq=16, N=512 unchanged)
# Compare against log/prefill/sim.log (baseline SM90_H100)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT/log/prefill_mq32"

OMP_NUM_THREADS=16 OMP_PROC_BIND=spread \
"$ROOT/gpu-simulator/bin/release/accel-sim.out" \
  -config "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_mq32/gpgpusim.config" \
  -config "$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config" \
  -is_extra_traces_enabled 1 -filter_first_kernel_id 1 -filter_last_kernel_id 3 \
  -trace /home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb \
  | tee "$ROOT/log/prefill_mq32/sim.log"
