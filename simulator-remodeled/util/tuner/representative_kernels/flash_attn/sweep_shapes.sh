#!/usr/bin/env bash
# Sweep flash_attn_ext_vec decode shapes (seq_len) under GPGPU-Sim execution-driven mode.
# Usage: ./sweep_shapes.sh [gpgpu-sim lib dir]
set -u
LIB=${1:-/home/qshao/Project/Fun/accel-sim-framework-official/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11050/release}
BIN=./flash_attn_bench_exec

SHAPES=(
  "512 short_ctx"
  "1024 prefill_only"
  "1088 decode_p1024"
)

for s in "${SHAPES[@]}"; do
  set -- $s; seq=$1; name=$2
  log="flash_attn_exec_${seq}_${name}.log"
  echo "=== flash_attn_ext_vec $name  seq_len=$seq -> $log ==="
  CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH=$LIB $BIN "$seq" | tee "$log"
  echo "    cycles: $(grep -E 'gpu_sim_cycle = ' "$log" | tail -1)"
done
