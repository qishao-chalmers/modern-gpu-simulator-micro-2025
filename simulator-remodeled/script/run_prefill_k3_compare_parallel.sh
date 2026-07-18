#!/usr/bin/env bash
# Run baseline (best_optA) and depfix (k3_depfix) k3 simulations IN PARALLEL.
#
# Usage:
#   ./run_prefill_k3_compare_parallel.sh          # launch both, wait, print summary
#   ./run_prefill_k3_compare_parallel.sh --bg   # launch both under nohup, exit immediately
#
# Logs (same paths as sequential compare):
#   log/prefill_k3_compare/baseline_k3.log
#   log/prefill_k3_compare/depfix_k3.log
#   log/prefill_k3_compare/parallel_runner.log   (--bg only)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE="${TRACE:-/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb}"
LOG_DIR="${LOG_DIR:-$ROOT/log/prefill_k3_compare}"
SIM="$ROOT/gpu-simulator/bin/release/accel-sim.out"
TRACE_CFG="$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config"
BASE_CFG="$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_best_optA/gpgpusim.config"
DEP_CFG="$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_k3_depfix/gpgpusim.config"
BG_MODE=0
[[ "${1:-}" == "--bg" ]] && BG_MODE=1

mkdir -p "$LOG_DIR"

if [[ ! -x "$SIM" ]]; then
  echo "error: build simulator first" >&2
  exit 1
fi

run_one() {
  local label="$1"
  local gpgpu_cfg="$2"
  local log="$LOG_DIR/${label}_k3.log"
  echo "[$(date -Iseconds)] starting $label -> $log"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
  stdbuf -oL -eL "$SIM" \
    -config "$gpgpu_cfg" \
    -config "$TRACE_CFG" \
    -is_extra_traces_enabled 1 \
    -filter_first_kernel_id 3 -filter_last_kernel_id 3 \
    -trace "$TRACE" \
    > "$log" 2>&1
  echo "[$(date -Iseconds)] finished $label (exit $?)"
}

print_summary() {
  python3 - "$LOG_DIR" <<'PY'
import re, sys
from pathlib import Path

log_dir = Path(sys.argv[1])
real_target = 166246

def extract(path):
    if not path.exists():
        return None
    txt = path.read_text(errors="replace")
    out = {"file": path.name}
    for key, pat in [
        ("cycles", r"gpu_tot_sim_cycle\s*=\s*(\d+)"),
        ("ipc", r"gpu_ipc\s*=\s*([\d.]+)"),
        ("occupancy", r"gpu_occupancy\s*=\s*([\d.]+)%"),
    ]:
        m = re.search(pat, txt)
        if m:
            out[key] = m.group(1)
    return out

print("\n=== k3 compare summary ===")
rows = []
for label in ("baseline", "depfix"):
    r = extract(log_dir / f"{label}_k3.log")
    if r:
        rows.append(r)
        cyc = int(r["cycles"]) if "cycles" in r else None
        pct = f"{100.0 * (cyc - real_target) / real_target:+.1f}%" if cyc else "n/a"
        print(f"  {r['file']:20s}  cycles={r.get('cycles','?'):>8s}  vs real={pct:>8s}  ipc={r.get('ipc','?')}")
    else:
        print(f"  {label}_k3.log: (missing or incomplete)")

if len(rows) == 2 and "cycles" in rows[0] and "cycles" in rows[1]:
    b, d = int(rows[0]["cycles"]), int(rows[1]["cycles"])
    print(f"  depfix vs baseline: {d - b:+d} cycles ({100.0 * (d - b) / b:+.2f}%)")
print(f"  real H100 target: ~{real_target} cycles")
PY
}

launch_parallel() {
  run_one baseline "$BASE_CFG" &
  local pid_base=$!
  run_one depfix "$DEP_CFG" &
  local pid_dep=$!
  echo "baseline PID=$pid_base  depfix PID=$pid_dep"
  echo "  tail -f $LOG_DIR/baseline_k3.log"
  echo "  tail -f $LOG_DIR/depfix_k3.log"
  wait "$pid_base" || true
  local rc_base=$?
  wait "$pid_dep" || true
  local rc_dep=$?
  echo "baseline exit=$rc_base  depfix exit=$rc_dep"
  print_summary
  [[ $rc_base -eq 0 && $rc_dep -eq 0 ]]
}

if [[ "$BG_MODE" -eq 1 ]]; then
  nohup "$0" >> "$LOG_DIR/parallel_runner.log" 2>&1 &
  echo "Launched parallel compare (nohup PID $!)"
  echo "  monitor: tail -f $LOG_DIR/parallel_runner.log"
  echo "  logs:    $LOG_DIR/baseline_k3.log"
  echo "           $LOG_DIR/depfix_k3.log"
  exit 0
fi

launch_parallel
