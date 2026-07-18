#!/usr/bin/env bash
# Experiment: ptx_opcode_initiation_tensor 8→32
# Compare gpu_sim_cycle against log/prefill/sim.log (baseline SM90_H100)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT/log/prefill_tinit32"

OMP_NUM_THREADS=16 OMP_PROC_BIND=spread \
"/home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/bin/release/accel-sim.out" \
  -config "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_tinit32/gpgpusim.config" \
  -config "/home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config" \
  -is_extra_traces_enabled 1 -filter_first_kernel_id 1 -filter_last_kernel_id 3 \
  -trace /home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb \
  | tee "$ROOT/log/prefill_tinit32/sim.log"
