#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE="${TRACE:-/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb}"
TRACE_CFG="$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"

OUT_DIR="$ROOT/log/prefill_best_multiopt"
mkdir -p "$OUT_DIR"

configs=(
  "SM90_H100_best"
  "SM90_H100_best_optA"
  "SM90_H100_best_optB"
  "SM90_H100_best_optC"
)

cmd_for_cfg() {
  local cfg="$1"
  cat <<EOF
cd "$ROOT" && OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" \
./gpu-simulator/bin/release/accel-sim.out \
  -config "./gpu-simulator/gpgpu-sim/configs/tested-cfgs/${cfg}/gpgpusim.config" \
  -config "$TRACE_CFG" \
  -is_extra_traces_enabled 1 -filter_first_kernel_id 1 -filter_last_kernel_id 3 \
  -trace "$TRACE" \
  | tee "$OUT_DIR/${cfg}.log"
EOF
}

if command -v tmux >/dev/null 2>&1; then
  session="${TMUX_SESSION:-prefill_multiopt}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session"
    echo "Attach with: tmux attach -t $session"
    exit 0
  fi

  echo "Starting tmux session: $session"
  tmux new-session -d -s "$session" -n "${configs[0]}"
  tmux send-keys -t "$session:0" "$(cmd_for_cfg "${configs[0]}")" C-m

  for i in 1 2 3; do
    tmux new-window -t "$session" -n "${configs[$i]}"
    tmux send-keys -t "$session:$i" "$(cmd_for_cfg "${configs[$i]}")" C-m
  done

  tmux select-window -t "$session:0"
  echo "Attach with: tmux attach -t $session"
  exit 0
fi

echo "tmux not found; running in background (logs in $OUT_DIR/)."
pids=()
for cfg in "${configs[@]}"; do
  log="$OUT_DIR/${cfg}.log"
  echo "Launching $cfg → $log"
  (
    cd "$ROOT"
    OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" \
    ./gpu-simulator/bin/release/accel-sim.out \
      -config "./gpu-simulator/gpgpu-sim/configs/tested-cfgs/${cfg}/gpgpusim.config" \
      -config "$TRACE_CFG" \
      -is_extra_traces_enabled 1 -filter_first_kernel_id 1 -filter_last_kernel_id 3 \
      -trace "$TRACE"
  ) >"$log" 2>&1 &
  pids+=("$!")
done

echo "PIDs: ${pids[*]}"
echo "When all finish, summarize:"
echo "  cd \"$ROOT\" && ./summarize_prefill_results.sh"

