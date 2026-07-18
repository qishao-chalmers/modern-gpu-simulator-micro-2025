#!/usr/bin/env bash
# Run k3 baseline (best_optA) and depfix (fixes 1-3) back-to-back with paired logs.
#
# For parallel execution (recommended — ~2x faster wall time), use instead:
#   ./run_prefill_k3_compare_parallel.sh
#   ./run_prefill_k3_compare_parallel.sh --bg
#
# Usage:
#   ./run_prefill_k3_compare.sh              # full k3 runs, logs under log/prefill_k3_compare/
#   TRACE_DEBUG=1 ./run_prefill_k3_compare.sh  # also emit issue/commit/ex_release traces (SM0/warp0)
#
# Compare:
#   grep gpu_tot_sim_cycle log/prefill_k3_compare/baseline_k3.log
#   grep gpu_tot_sim_cycle log/prefill_k3_compare/depfix_k3.log
#   diff -u <(grep -E 'gpu_tot_sim_cycle|gpu_ipc|gpu_occupancy' log/prefill_k3_compare/baseline_k3.log | tail -5) \
#           <(grep -E 'gpu_tot_sim_cycle|gpu_ipc|gpu_occupancy' log/prefill_k3_compare/depfix_k3.log | tail -5)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE="${TRACE:-/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb}"
LOG_DIR="${LOG_DIR:-$ROOT/log/prefill_k3_compare}"
TRACE_DEBUG="${TRACE_DEBUG:-0}"
SIM="$ROOT/gpu-simulator/bin/release/accel-sim.out"
TRACE_CFG="$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config"
mkdir -p "$LOG_DIR"

if [[ ! -x "$SIM" ]]; then
  echo "error: build simulator first (cd $ROOT/gpu-simulator && source setup_environment_no_git.sh release && make -j)" >&2
  exit 1
fi

DEBUG_ARGS=()
if [[ "$TRACE_DEBUG" == "1" ]]; then
  DEBUG_ARGS=(-subcore_issue_debug 1)
  echo "TRACE_DEBUG=1: issue_trace / commit_trace / ex_release_trace enabled (SM0/warp0)"
fi

run_k3() {
  local label="$1"
  local gpgpu_cfg="$2"
  local log="$LOG_DIR/${label}_k3.log"
  echo ""
  echo "========================================"
  echo "  $label"
  echo "  config: $gpgpu_cfg"
  echo "  log:    $log"
  echo "========================================"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
  "$SIM" \
    -config "$gpgpu_cfg" \
    -config "$TRACE_CFG" \
    -is_extra_traces_enabled 1 \
    -filter_first_kernel_id 3 -filter_last_kernel_id 3 \
    "${DEBUG_ARGS[@]}" \
    -trace "$TRACE" \
    2>&1 | tee "$log"
}

BASE_CFG="$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_best_optA/gpgpusim.config"
DEP_CFG="$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_k3_depfix/gpgpusim.config"

run_k3 baseline "$BASE_CFG"
run_k3 depfix "$DEP_CFG"

echo ""
echo "=== summary (extract from logs) ==="
python3 - "$LOG_DIR" <<'PY'
import re, sys
from pathlib import Path

log_dir = Path(sys.argv[1])
real_target = 166246

def extract(path):
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

rows = []
for label in ("baseline", "depfix"):
    p = log_dir / f"{label}_k3.log"
    if p.exists():
        rows.append(extract(p))

for r in rows:
    cyc = int(r["cycles"]) if "cycles" in r else None
    pct = f"{100.0 * (cyc - real_target) / real_target:+.1f}%" if cyc else "n/a"
    print(f"  {r['file']:20s}  cycles={r.get('cycles','?'):>8s}  vs real={pct:>8s}  ipc={r.get('ipc','?')}  occ={r.get('occupancy','?')}%")

if len(rows) == 2 and "cycles" in rows[0] and "cycles" in rows[1]:
    b, d = int(rows[0]["cycles"]), int(rows[1]["cycles"])
    delta = d - b
    pct = 100.0 * delta / b
    print(f"  depfix vs baseline: {delta:+d} cycles ({pct:+.2f}%)")

print(f"\n  real H100 target: ~{real_target} cycles")
print(f"  logs: {log_dir}/baseline_k3.log")
print(f"        {log_dir}/depfix_k3.log")
PY
