#!/usr/bin/env bash
# Experiment: real constant cache (perfect_inst_const_cache=0, same as SM90_H100_nopc)
# PLUS widen its MSHR 2->64 and miss queue 4->32 (was the suspected degenerate bottleneck).
# Compare gpu_sim_cycle against log/prefill_nopc/sim.log (793,924 k3 cycles) and
# log/prefill/sim.log (baseline SM90_H100, perfect cache, 258,275 k3 cycles).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT/log/prefill_nopc_constmshr"

OMP_NUM_THREADS=16 OMP_PROC_BIND=spread \
"$ROOT/gpu-simulator/bin/release/accel-sim.out" \
  -config "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_nopc_constmshr/gpgpusim.config" \
  -config "$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config" \
  -is_extra_traces_enabled 1 -filter_first_kernel_id 1 -filter_last_kernel_id 3 \
  -trace /home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb \
  | tee "$ROOT/log/prefill_nopc_constmshr/sim.log"
