#!/usr/bin/env bash
# Sweep decode GEMV (matrix-vector only) across Qwen3 models × projections × quants.
# Does NOT run layer_decode or other non-GEMV kernels.
#
# Usage:
#   ./sweep_quant.sh
#   MODELS="8b" OPS="q_proj down" QUANTS="q4_k q2_k" ./sweep_quant.sh
#   SKIP_LM_HEAD=1 ./sweep_quant.sh
#
# Requires: make exec in this dir, GPGPU-Sim lib under LIB, config under CFG.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPK="$(cd "$HERE/.." && pwd)"
LIB="${LIB:-/home/qshao/Project/Fun/accel-sim-framework-official/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11050/release}"
CFG="${CFG:-/home/qshao/Project/Fun/accel-sim-framework-official/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100}"

MODELS="${MODELS:-8b 14b}"
OPS="${OPS:-q_proj k_proj v_proj o_proj gate up down lm_head}"
QUANTS="${QUANTS:-q8_0 q4_k q2_k}"
SKIP_LM_HEAD="${SKIP_LM_HEAD:-0}"

EXE="$HERE/mmvq_kquant_exec"
OUT="$REPK/results/gemv_quant"
mkdir -p "$OUT"

run_cfg_dir() {
  local d="$1"
  mkdir -p "$d"
  cp "$CFG/gpgpusim.config" "$CFG/config_volta_islip.icnt" "$CFG"/*.xml "$d/"
}

if [ ! -x "$EXE" ]; then
  echo "Building mmvq_kquant_exec..."
  (cd "$HERE" && make exec) || exit 1
fi

echo "=========== decode GEMV quant sweep -> $OUT ==========="
for m in $MODELS; do
  for op in $OPS; do
    if [ "$SKIP_LM_HEAD" = "1" ] && [ "$op" = "lm_head" ]; then
      continue
    fi
    for q in $QUANTS; do
      tag="${m}_${op}_${q}"
      dir="$OUT/$tag"
      run_cfg_dir "$dir"
      echo "--- $tag ---"
      ( cd "$dir" && CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH="$LIB" \
          MMVQ_NO_TIME=1 "$EXE" "$m" "$op" "$q" > "$tag.log" 2>&1 )
      cyc=$(grep 'gpu_tot_sim_cycle' "$dir/$tag.log" | tail -1 | awk '{print $3}')
      printf '  %-28s %s\n' "$tag" "${cyc:-ERR}"
    done
  done
done

echo
echo "Logs: $OUT/<model>_<op>_<quant>/<tag>.log"
