#!/usr/bin/env bash
# Unbundle depfix: run baseline + fix1 + fix2 + fix3 in parallel (k3-isolated).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE="${TRACE:-/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb}"
LOG_DIR="${LOG_DIR:-$ROOT/log/prefill_k3_unbundle}"
SIM="$ROOT/gpu-simulator/bin/release/accel-sim.out"
TRACE_CFG="$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config"
CFG="$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs"
mkdir -p "$LOG_DIR"

run_one() {
  local label="$1" gpgpu_cfg="$2"
  local log="$LOG_DIR/${label}_k3.log"
  echo "[$(date -Iseconds)] $label -> $log"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
  stdbuf -oL -eL "$SIM" \
    -config "$gpgpu_cfg" -config "$TRACE_CFG" \
    -is_extra_traces_enabled 1 -filter_first_kernel_id 3 -filter_last_kernel_id 3 \
    -trace "$TRACE" > "$log" 2>&1
}

declare -a PIDS=()
for spec in \
  "baseline:$CFG/SM90_H100_best_optA/gpgpusim.config" \
  "scoreboard_ex:$CFG/SM90_H100_k3_scoreboard_ex/gpgpusim.config" \
  "sfu_int_lat:$CFG/SM90_H100_k3_sfu_int_lat/gpgpusim.config" \
  "tensor_decouple:$CFG/SM90_H100_k3_tensor_decouple/gpgpusim.config" \
  "depfix_bundled:$CFG/SM90_H100_k3_depfix/gpgpusim.config"; do
  label="${spec%%:*}"
  cfg="${spec#*:}"
  run_one "$label" "$cfg" &
  PIDS+=($!)
done

echo "Launched: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do wait "$pid" || true; done

python3 - "$LOG_DIR" <<'PY'
import re, sys
from pathlib import Path
log_dir = Path(sys.argv[1])
real = 166246
print("\n=== k3 unbundle summary ===")
rows = []
for p in sorted(log_dir.glob("*_k3.log")):
    txt = p.read_text(errors="replace")
    m = re.search(r"gpu_tot_sim_cycle\s*=\s*(\d+)", txt)
    ipc = re.search(r"gpu_ipc\s*=\s*([\d.]+)", txt)
    if m:
        cyc = int(m.group(1))
        pct = 100.0 * (cyc - real) / real
        rows.append((p.stem, cyc, pct, ipc.group(1) if ipc else "?"))
for name, cyc, pct, ipc in rows:
    print(f"  {name:22s}  {cyc:>7d}  vs real={pct:+.1f}%  ipc={ipc}")
base = next((c for n,c,_ in rows if n=="baseline_k3"), None)
if base:
    for name, cyc, pct, ipc in rows:
        if name != "baseline_k3":
            d = cyc - base
            print(f"    {name} vs baseline: {d:+d} ({100.0*d/base:+.2f}%)")
print(f"  real H100: {real}")
PY
