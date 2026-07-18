#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE="/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb"
TRACE_CFG="$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"

configs=(
  "SM90_H100_best"
  "SM90_H100_best_optA"
  "SM90_H100_best_optB"
  "SM90_H100_best_optC"
)

mkdir -p "$ROOT/log/prefill_best_multiopt"

for cfg in "${configs[@]}"; do
  out="$ROOT/log/prefill_best_multiopt/${cfg}.log"
  echo "=== Running ${cfg} → ${out}"
  OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" \
  "$ROOT/gpu-simulator/bin/release/accel-sim.out" \
    -config "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/${cfg}/gpgpusim.config" \
    -config "$TRACE_CFG" \
    -is_extra_traces_enabled 1 -filter_first_kernel_id 1 -filter_last_kernel_id 3 \
    -trace "$TRACE" \
    | tee "$out"
done

echo
echo "Done. To summarize vs real profile (kernel_id 35–37):"
echo "  cd \"$ROOT\" && ./summarize_prefill_results.sh"

