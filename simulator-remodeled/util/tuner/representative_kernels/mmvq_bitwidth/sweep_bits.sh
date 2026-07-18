#!/usr/bin/env bash
# Sweep all Qwen3-14B GEMV dimensions at a given weight bit-width under the
# (execution-driven) GPGPU-Sim. Each run -> mmvq_bitwidth_exec_<K>_<N>_<bits>bit.log
#
# Usage:  ./sweep_bits.sh [bits]      (default bits=2)
# Prereq: run from a dir that has gpgpusim.config + config_volta_islip.icnt + *.xml
#         and the mmvq_bitwidth_exec binary (see the one-time setup in the README/commands).
#
# NOTE: functional sim runs every thread, so runtime grows with N (CTAs):
#   kv/qo = minutes, gateup/down = tens of minutes+, lmhead (N=151936) ~ days (impractical).
#   Comment out the big shapes if you only want the feasible ones.
set -u
BITS=${1:-2}
LIB=/home/qshao/Project/Fun/accel-sim-framework-official/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11050/release

# Qwen3-14B GEMV shapes:  "K N name"
SHAPES=(
  "5120 1024 kv"
  "5120 5120 qo"
  "5120 17408 gateup"
  "17408 5120 down"
  "5120 151936 lmhead"
)

for s in "${SHAPES[@]}"; do
  set -- $s; K=$1; N=$2; name=$3
  log="mmvq_bitwidth_exec_${K}_${N}_${BITS}bit.log"
  echo "=== $name  K=$K N=$N  bits=$BITS  ->  $log ==="
  CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH=$LIB ./mmvq_bitwidth_exec "$K" "$N" "$BITS" | tee "$log"
  echo "    cycles: $(grep -E 'gpu_sim_cycle = ' "$log" | tail -1)"
done
