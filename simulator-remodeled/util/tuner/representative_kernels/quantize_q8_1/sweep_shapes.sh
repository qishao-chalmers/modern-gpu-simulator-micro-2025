#!/usr/bin/env bash
# Sweep Qwen3-14B quantize_q8_1 decode shapes under GPGPU-Sim execution-driven mode.
# Usage: ./sweep_shapes.sh [gpgpu-sim lib dir]
set -u
LIB=${1:-/home/qshao/Project/Fun/accel-sim-framework-official/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11050/release}
BIN=./quantize_bench_exec

SHAPES=(
  "5120 1 hidden_5120"
  "17408 1 ffn_mid_17408"
)

for s in "${SHAPES[@]}"; do
  set -- $s; ne0=$1; ne1=$2; name=$3
  log="quantize_exec_${ne0}_${name}.log"
  echo "=== quantize_q8_1 $name  ne0=$ne0 ne1=$ne1 -> $log ==="
  CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH=$LIB $BIN "$ne0" "$ne1" | tee "$log"
  echo "    cycles: $(grep -E 'gpu_sim_cycle = ' "$log" | tail -1)"
done
