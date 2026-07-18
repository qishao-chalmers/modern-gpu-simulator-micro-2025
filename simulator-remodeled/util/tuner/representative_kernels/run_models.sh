#!/usr/bin/env bash
# run_models.sh — sweep the representative decode kernels across models × batch sizes.
#
# Runs the compression-INDEPENDENT bucket (rms_norm, quantize, rope, set_rows,
# flash_attn, swiglu, add) at each model's real dimensions, execution-driven on the
# QV100 GPGPU-Sim. These are the kernels whose shape depends on the model + batch but
# NOT on weight bit-width, so one run per (model,batch) feeds every bit-width study.
#
# The weight matmuls (Q/K/V/O/gate/up/down, lm_head) are the QWC-affected part and are
# handled separately by the gemv/ (mmvq_bitwidth) and mmq/ folders with their bit-width
# sweeps — they are also too large for funcsim at full dims. Set WITH_GEMM=1 to also run
# them here at 8-bit baseline (SLOW; reduce with GEMM_N_CAP).
#
# Usage:
#   ./run_models.sh                       # models "14b 8b", batches "1 8 16 32"
#   MODELS="14b" BATCHES="1 32" ./run_models.sh
#   WITH_GEMM=1 ./run_models.sh           # also run the 7 weight matmuls (8-bit)
#
# Output: results/<model>_b<batch>/<kernel>.log  + a printed cycle summary.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LIB="${LIB:-/home/qshao/Project/Fun/accel-sim-framework-official/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11050/release}"
CFG="${CFG:-/home/qshao/Project/Fun/accel-sim-framework-official/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100}"

MODELS="${MODELS:-14b 8b}"
BATCHES="${BATCHES:-1 8 16 32}"
WITH_GEMM="${WITH_GEMM:-0}"

# --- model dimensions --------------------------------------------------------
# fields: H n_q n_kv hd F vocab seq   (QD=n_q*hd, KD=n_kv*hd derived below)
dims_14b="5120 40 8 128 17408 151936 1088"   # Qwen3-14B (40 layers)
dims_8b="4096 32 8 128 12288 151936 1088"    # Qwen3-8B  (36 layers)

run_cfg_dir() {   # stage config + a fresh copy so parallel runs don't clash
  local d="$1"; mkdir -p "$d"
  cp "$CFG/gpgpusim.config" "$CFG/config_volta_islip.icnt" "$CFG"/*.xml "$d/"
}

# run <outdir> <logname> <exec_path> <args...>   -> prints "name  cycles"
run_one() {
  local dir="$1" name="$2" exe="$3"; shift 3
  ( cd "$dir" && CUDA_INSTALL_PATH=/usr LD_LIBRARY_PATH="$LIB" "$exe" "$@" > "$name.log" 2>&1 )
  local cyc; cyc=$(grep 'gpu_tot_sim_cycle' "$dir/$name.log" | tail -1 | awk '{print $3}')
  printf '  %-16s %s\n' "$name" "${cyc:-ERR}"
}

REPK="$HERE"
for m in $MODELS; do
  eval "set -- \$dims_$m"
  H=$1 NQ=$2 NKV=$3 HD=$4 F=$5 VOCAB=$6 SEQ=$7
  QD=$((NQ*HD)); KD=$((NKV*HD))
  for B in $BATCHES; do
    OUT="$HERE/results/${m}_b${B}"; run_cfg_dir "$OUT"
    echo "=========== model=$m  batch=$B  (H=$H n_q=$NQ n_kv=$NKV hd=$HD F=$F seq=$SEQ) ==========="
    export BATCH=$B
    # --- non-gemm + attention bucket (real dims, funcsim-cheap) ---
    run_one "$OUT" rms_norm_attn "$REPK/rms_norm/rms_norm_bench_exec"   "$H"  1    1024
    run_one "$OUT" rms_norm_q    "$REPK/rms_norm/rms_norm_bench_exec"   "$HD" "$NQ" 256
    run_one "$OUT" rms_norm_k    "$REPK/rms_norm/rms_norm_bench_exec"   "$HD" "$NKV" 256
    run_one "$OUT" quant_hidden  "$REPK/quantize_q8_1/quantize_bench_exec" "$H" 1
    run_one "$OUT" quant_ffn     "$REPK/quantize_q8_1/quantize_bench_exec" "$F" 1
    run_one "$OUT" ropeQ         "$REPK/rope/rope_bench_exec"           "$NQ"  "$HD" 0
    run_one "$OUT" ropeK         "$REPK/rope/rope_bench_exec"           "$NKV" "$HD" 1
    run_one "$OUT" set_rows      "$REPK/set_rows/set_rows_bench_exec"   "$KD" "$SEQ"
    run_one "$OUT" flash_attn    "$REPK/flash_attn/flash_attn_bench_exec" "$SEQ" "$B" "$NQ" "$NKV"
    run_one "$OUT" swiglu        "$REPK/swiglu/swiglu_bench_exec"       "$F"
    run_one "$OUT" add           "$REPK/add/add_bench_exec"             "$H"
    # --- weight matmuls (QWC-affected); opt-in, 8-bit baseline, SLOW ---
    if [ "$WITH_GEMM" = "1" ]; then
      G="$REPK/mmvq_bitwidth/mmvq_bitwidth_exec"   # args: K N bits  (dst[N]=W[N x K].y[K])
      run_one "$OUT" gemm_Q    "$G" "$H"  "$QD" 8
      run_one "$OUT" gemm_K    "$G" "$H"  "$KD" 8
      run_one "$OUT" gemm_V    "$G" "$H"  "$KD" 8
      run_one "$OUT" gemm_O    "$G" "$QD" "$H"  8
      run_one "$OUT" gemm_gate "$G" "$H"  "$F"  8
      run_one "$OUT" gemm_up   "$G" "$H"  "$F"  8
      run_one "$OUT" gemm_down "$G" "$F"  "$H"  8
    fi
    unset BATCH
  done
done

echo; echo "Logs under $HERE/results/<model>_b<batch>/.  Summarize with:"
echo "  grep -H gpu_tot_sim_cycle $HERE/results/*/*.log | tail -1 per file"
