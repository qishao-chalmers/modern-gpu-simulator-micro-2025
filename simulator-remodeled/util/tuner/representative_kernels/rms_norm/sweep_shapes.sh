#!/usr/bin/env bash
# Sweep Qwen3-14B rms_norm decode shapes under GPGPU-Sim execution-driven mode.
# Usage: ./sweep_shapes.sh [gpgpu-sim lib dir]
set -u
LIB=${1:-/home/qshao/Project/Fun/accel-sim-framework-official/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11050/release}
BIN=./rms_norm_bench_exec

SHAPES=(
  "5120 1 1024 attn_norm"
  "128 40 256 q_norm"
  "128 8 256 k_norm"
  "5120 1 1024 ffn_norm"
)

for s in "${SHAPES[@]}"; do
  set -- $s; ncols=$1; nrows=$2; bs=$3; name=$4
  log="rms_norm_exec_${ncols}_${nrows}_${name}.log"
  echo "=== rms_norm $name  ncols=$ncols nrows=$nrows block=$bs -> $log ==="
  CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH=$LIB $BIN "$ncols" "$nrows" "$bs" | tee "$log"
  echo "    cycles: $(grep -E 'gpu_sim_cycle = ' "$log" | tail -1)"
done
