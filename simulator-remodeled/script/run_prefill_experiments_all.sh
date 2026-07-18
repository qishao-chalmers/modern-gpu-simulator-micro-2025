#!/usr/bin/env bash
# Launch all prefill calibration experiments with bounded parallelism.
# Usage: MAX_JOBS=2 ./run_prefill_experiments_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_JOBS="${MAX_JOBS:-2}"

scripts=(
  run_prefill_mq32.sh
  run_prefill_mq64.sh
  run_prefill_mshr1024_mq64.sh
  run_prefill_trate512.sh
  run_prefill_trate1024.sh
  run_prefill_tinit16.sh
  run_prefill_tinit32.sh
  run_prefill_textra8.sh
  run_prefill_textra32.sh
)

running=0
pids=()

wait_for_slot() {
  while (( running >= MAX_JOBS )); do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        wait "${pids[$i]}" || true
        unset 'pids[i]'
        ((running--)) || true
      fi
    done
    pids=("${pids[@]:-}")
    sleep 5
  done
}

for script in "${scripts[@]}"; do
  tag="${script#run_prefill_}"
  tag="${tag%.sh}"
  log_dir="$ROOT/log/prefill_${tag}"
  mkdir -p "$log_dir"

  # Skip if k3 already finished
  if [[ -f "$log_dir/sim.log" ]] && grep -q "launching kernel name:.*uid: 3" "$log_dir/sim.log"; then
    k3_cycles="$(awk '/launching kernel name:.*uid: 3/{k=1} k && /^gpu_sim_cycle = /{print $4; exit}' "$log_dir/sim.log" || true)"
    if [[ -n "${k3_cycles:-}" ]]; then
      echo "SKIP $script (k3 done: ${k3_cycles} cycles)"
      continue
    fi
  fi

  wait_for_slot
  echo "START $script (MAX_JOBS=$MAX_JOBS, running=$running)"
  (
    cd "$ROOT"
    exec "./$script"
  ) >"$log_dir/runner.log" 2>&1 &
  pids+=("$!")
  ((running++))
done

echo "Waiting for ${#pids[@]} job(s)..."
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo "All jobs finished. Run: ./summarize_prefill_results.sh"
