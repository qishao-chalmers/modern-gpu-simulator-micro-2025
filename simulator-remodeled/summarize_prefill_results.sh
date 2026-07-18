#!/usr/bin/env bash
# Parse prefill experiment logs and compare against real H100 profiled times.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLK_GHZ="${CLK_GHZ:-1.620}"

# Real H100 profiled durations (ns) for the 3 traced prefill kernels.
# Match nsys/kernel_id 35/36/37 (same names as trace uid 1/2/3), NOT 23/24/25.
# 23/24/25 are a later layer pass (mul_mat_q ~297µs, likely FFN); 1/2/3 are warmup/graph noise.
declare -A REAL_NS=(
  [k1]=16078   # kernel_id=35 rms_norm_f32<1024>
  [k2]=12270   # kernel_id=36 quantize_mmq_q8_1
  [k3]=102621  # kernel_id=37 mul_mat_q<Q8_0, Li128>
)
TARGET_K3_CYCLES=$(python3 -c "print(int(round(${REAL_NS[k3]} * ${CLK_GHZ})))")

experiments=(
  "baseline:log/prefill/sim.log"
  "another_bk:log/prefill/sim_another_bk.log"
  "nopc:log/prefill_nopc/sim.log"
  "l2rop300:log/prefill_l2rop300/sim.log"
  "l2rop380_l1d20:log/prefill_l2rop380_l1d20/sim.log"
  "nopc_il0large:log/prefill_nopc_il0large/sim.log"
  "nopc_l2rop180_l1d10:log/prefill_nopc_l2rop180_l1d10/sim.log"
  "mq32:log/prefill_mq32/sim.log"
  "mq64:log/prefill_mq64/sim.log"
  "mshr1024_mq64:log/prefill_mshr1024_mq64/sim.log"
  "trate512:log/prefill_trate512/sim.log"
  "trate1024:log/prefill_trate1024/sim.log"
  "tinit16:log/prefill_tinit16/sim.log"
  "tinit32:log/prefill_tinit32/sim.log"
  "textra8:log/prefill_textra8/sim.log"
  "textra32:log/prefill_textra32/sim.log"
  "best:log/prefill_best/sim.log"
  "best_multiopt_best:log/prefill_best_multiopt/SM90_H100_best.log"
  "best_multiopt_optA:log/prefill_best_multiopt/SM90_H100_best_optA.log"
  "best_multiopt_optB:log/prefill_best_multiopt/SM90_H100_best_optB.log"
  "best_multiopt_optC:log/prefill_best_multiopt/SM90_H100_best_optC.log"
)

parse_cycles() {
  local log="$1"
  awk '
    /launching kernel name:/ { k++ }
    /^gpu_sim_cycle = / { cycles[k] = $3 }
    END {
      for (i = 1; i <= 3; i++) {
        if (cycles[i] == "") print "NA";
        else print cycles[i];
      }
    }
  ' "$log"
}

pct_err() {
  python3 -c "sim,real=$1,$2; print(f'{(sim-real)/real*100:+.1f}')" 2>/dev/null || echo "NA"
}

cycles_to_ns() {
  python3 -c "print(int(round($1 / ${CLK_GHZ})))"
}

printf '%-18s %10s %10s %10s %10s %10s %10s %10s\n' \
  "Experiment" "k1_cyc" "k2_cyc" "k3_cyc" "k3_ns" "k3_err%" "MISS_Q(k3)" "Status"
printf '%s\n' "$(printf '%.0s-' {1..110})"

for entry in "${experiments[@]}"; do
  name="${entry%%:*}"
  log_rel="${entry#*:}"
  log="$ROOT/$log_rel"

  if [[ ! -f "$log" ]]; then
    printf '%-18s %10s %10s %10s %10s %10s %10s %s\n' \
      "$name" "-" "-" "-" "-" "-" "-" "missing"
    continue
  fi

  mapfile -t cycles < <(parse_cycles "$log")
  k1="${cycles[0]}" k2="${cycles[1]}" k3="${cycles[2]}"

  if [[ "$k3" == "NA" ]]; then
    status="running"
    k3_ns="-"
    k3_err="-"
    miss_q="-"
  else
    status="done"
    k3_ns="$(cycles_to_ns "$k3")"
    k3_err="$(pct_err "$k3_ns" "${REAL_NS[k3]}")"
    # MISS_QUEUE_FULL for kernel 3: last occurrence block in log
    miss_q="$(awk '
      /launching kernel name:.*uid: 3/ { in_k3=1; next }
      /launching kernel name:/ && in_k3 { in_k3=0 }
      in_k3 && /GLOBAL_ACC_R\]\[MISS_QUEUE_FULL\]/ {
        last = $NF
      }
      END { if (last=="") print "-"; else print last }
    ' "$log")"
  fi

  printf '%-18s %10s %10s %10s %10s %10s %10s %s\n' \
    "$name" "$k1" "$k2" "$k3" "$k3_ns" "$k3_err" "$miss_q" "$status"
done

echo
echo "Target k3: ${TARGET_K3_CYCLES} cycles (~${REAL_NS[k3]} ns @ ${CLK_GHZ} GHz)"
echo "Baseline k3: 258275 cycles (~159429 ns) => need +$(python3 -c "print(round((${REAL_NS[k3]}/159429 - 1)*100))")% more cycles"
