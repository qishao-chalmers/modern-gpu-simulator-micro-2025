#!/usr/bin/env bash
# Sweep flash_attn decode at multiple PK (split-K) values under GPGPU-Sim.
# Usage: ./sweep_pk.sh [gpgpu-sim lib dir] [seq_len]
set -u
LIB=${1:-/home/qshao/Project/Fun/accel-sim-framework-official/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11050/release}
SEQ=${2:-1024}
BIN=./flash_attn_bench_exec

for PK in 1 2 4 8 16; do
  log="flash_attn_exec_${SEQ}_pk${PK}.log"
  echo "=== flash_attn_ext_vec seq=$SEQ PK=$PK -> $log ==="
  CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH=$LIB PARALLEL_K=$PK $BIN "$SEQ" 1 40 8 "$PK" | tee "$log"
  echo "    attn cycles:  $(grep -E 'gpu_sim_cycle = ' "$log" | head -1)"
  echo "    combine cyc:  $(grep -E 'gpu_sim_cycle = ' "$log" | tail -1)"
  echo "    total cycles: $(grep -E 'gpu_sim_cycle = ' "$log" | awk '{s+=$3} END{print s}')"
done
