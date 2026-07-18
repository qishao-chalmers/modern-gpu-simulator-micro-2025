#!/usr/bin/env bash
# Fix 4: compare baseline (best_optA) vs k3_depfix on segment timing + full k3 cycles.
#
# Usage:
#   ./run_k3_segment_timing.sh              # segment trace (50k cycles) + optional full run
#   FULL_RUN=1 ./run_k3_segment_timing.sh   # also run full k3-isolated cycle counts
#   SEGMENT_CYCLES=100000 ./run_k3_segment_timing.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE="${TRACE:-/home/qshao/Project/Fun/gpu_traces/modern/prefill_traces/dynamic_trace.pb}"
LOG_DIR="${LOG_DIR:-$ROOT/log/k3_segment_timing}"
SEGMENT_CYCLES="${SEGMENT_CYCLES:-50000}"
FULL_RUN="${FULL_RUN:-0}"
SIM="$ROOT/gpu-simulator/bin/release/accel-sim.out"
TRACE_CFG="$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config"
mkdir -p "$LOG_DIR"

if [[ ! -x "$SIM" ]]; then
  echo "error: build simulator first (cd $ROOT && ./build.sh)" >&2
  exit 1
fi

run_segment() {
  local tag="$1"
  local gpgpu_cfg="$2"
  local log="$LOG_DIR/${tag}_segment_${SEGMENT_CYCLES}.log"
  echo "==> segment run: $tag (stop at gpu_cycle=$SEGMENT_CYCLES)"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
  "$SIM" \
    -config "$gpgpu_cfg" \
    -config "$TRACE_CFG" \
    -is_extra_traces_enabled 1 \
    -filter_first_kernel_id 3 -filter_last_kernel_id 3 \
    -subcore_issue_debug 1 \
    -subcore_issue_debug_stop_gpu_cycle "$SEGMENT_CYCLES" \
    -trace "$TRACE" \
    2>&1 | tee "$log"
}

run_full_k3() {
  local tag="$1"
  local gpgpu_cfg="$2"
  local log="$LOG_DIR/${tag}_k3_full.log"
  echo "==> full k3 run: $tag"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
  "$SIM" \
    -config "$gpgpu_cfg" \
    -config "$TRACE_CFG" \
    -is_extra_traces_enabled 1 \
    -filter_first_kernel_id 3 -filter_last_kernel_id 3 \
    -trace "$TRACE" \
    2>&1 | tee "$log"
}

analyze_segment_log() {
  local log="$1" label="$2"
  python3 - "$log" "$label" <<'PY'
import re, sys
from collections import defaultdict

log_path, label = sys.argv[1], sys.argv[2]
issue_re = re.compile(
    r"\[issue_trace\] sm=(\d+) subcore=(\d+) warp=(\d+) pc=0x([0-9a-f]+) op=(\S+) cycle=(\d+)"
)
commit_re = re.compile(
    r"\[commit_trace\] sm=(\d+) subcore=(\d+) warp=(\d+) pc=0x([0-9a-f]+) op=(\S+) cycle=(\d+)"
)
exrel_re = re.compile(
    r"\[ex_release_trace\] sm=(\d+) subcore=(\d+) warp=(\d+) pc=0x([0-9a-f]+) op=(\S+) cycle=(\d+)"
)
barrier_commit_re = re.compile(r"\[commit_trace\].*op=BARRIER_OP cycle=(\d+)")

issues = {}
commits = {}
exrels = {}
barrier_cycles = []

with open(log_path, errors="replace") as f:
    for line in f:
        m = issue_re.search(line)
        if m:
            pc = m.group(4)
            issues[pc] = int(m.group(6))
            continue
        m = commit_re.search(line)
        if m:
            pc, cyc, op = m.group(4), int(m.group(6)), m.group(5)
            commits[pc] = cyc
            if op == "BARRIER_OP":
                barrier_cycles.append(cyc)
            continue
        m = exrel_re.search(line)
        if m:
            exrels[m.group(4)] = int(m.group(6))

print(f"\n=== {label} ===")
print(f"  traced issues: {len(issues)}  commits: {len(commits)}  ex_releases: {len(exrels)}")

# issue->commit latency for dequant chain op types
chain_pcs = []
latencies = defaultdict(list)
for pc, ic in sorted(issues.items(), key=lambda x: int(x[0], 16)):
    if pc not in commits:
        continue
    cc = commits[pc]
    lat = cc - ic
    chain_pcs.append((pc, ic, cc, lat))

# Heuristic: report SFU/SP/INTP issue->commit stats (all PCs in window)
op_by_pc = {}
for line in open(log_path, errors="replace"):
    m = issue_re.search(line)
    if m:
        op_by_pc[m.group(4)] = m.group(5)

for op in ("SFU_OP", "SP_OP", "INTP_OP", "TENSOR_CORE_OP"):
    vals = [cc - issues[pc] for pc in issues if pc in commits and op_by_pc.get(pc) == op]
    if vals:
        vals.sort()
        med = vals[len(vals)//2]
        print(f"  {op} issue->commit: n={len(vals)} min={min(vals)} med={med} max={max(vals)} avg={sum(vals)/len(vals):.1f}")

# inter-BARRIER segment length (commit of barrier -> next barrier commit)
if len(barrier_cycles) >= 2:
    segs = [barrier_cycles[i+1] - barrier_cycles[i] for i in range(len(barrier_cycles)-1)]
    segs.sort()
    print(f"  inter-BARRIER segments (commit->commit): n={len(segs)} min={min(segs)} med={segs[len(segs)//2]} max={max(segs)} avg={sum(segs)/len(segs):.1f}")

# EX release vs commit gap (fix 1 diagnostic)
gaps = []
for pc, ec in exrels.items():
    if pc in commits:
        gaps.append(commits[pc] - ec)
if gaps:
    gaps.sort()
    print(f"  ex_release->commit gap: n={len(gaps)} min={min(gaps)} med={gaps[len(gaps)//2]} max={max(gaps)} avg={sum(gaps)/len(gaps):.1f}")

# Sample dequant chain PCs from first SFU issue in log
sfu_pcs = sorted([pc for pc, op in op_by_pc.items() if op == "SFU_OP"], key=lambda x: int(x, 16))
if len(sfu_pcs) >= 3:
    sample = sfu_pcs[:3]
    print("  sample SFU PCs (issue->commit):")
    for pc in sample:
        if pc in issues and pc in commits:
            ex = f" ex_rel+{commits[pc]-exrels[pc]}" if pc in exrels else ""
            print(f"    pc=0x{pc}: {commits[pc]-issues[pc]} cyc{ex}")

# kernel cycles from sim output
with open(log_path, errors="replace") as f:
    txt = f.read()
m = re.search(r"gpu_sim_cycle\s*=\s*(\d+)", txt)
if not m:
    m = re.search(r"gpu_tot_sim_cycle\s*=\s*(\d+)", txt)
if m:
    print(f"  sim stopped at cycle: {m.group(1)}")
PY
}

BASE_CFG="$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_best_optA/gpgpusim.config"
DEP_CFG="$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_k3_depfix/gpgpusim.config"

echo "Paired logs:"
echo "  baseline segment: $LOG_DIR/baseline_segment_${SEGMENT_CYCLES}.log"
echo "  depfix segment:   $LOG_DIR/depfix_segment_${SEGMENT_CYCLES}.log"
echo "  (full k3 compare: ./run_prefill_k3_compare.sh -> log/prefill_k3_compare/{baseline,depfix}_k3.log)"
echo ""

run_segment baseline "$BASE_CFG"
run_segment depfix "$DEP_CFG"

analyze_segment_log "$LOG_DIR/baseline_segment_${SEGMENT_CYCLES}.log" "baseline (best_optA)"
analyze_segment_log "$LOG_DIR/depfix_segment_${SEGMENT_CYCLES}.log" "depfix (fixes 1-3)"

if [[ "$FULL_RUN" == "1" ]]; then
  run_full_k3 baseline "$BASE_CFG"
  run_full_k3 depfix "$DEP_CFG"
  echo ""
  echo "=== full k3 cycle counts ==="
  for tag in baseline depfix; do
  log="$LOG_DIR/${tag}_k3_full.log"
  python3 - "$log" "$tag" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
for pat in (r"gpu_tot_sim_cycle\s*=\s*(\d+)", r"gpu_sim_cycle\s*=\s*(\d+)"):
    m = re.search(pat, txt)
    if m:
        print(f"  {sys.argv[2]}: {m.group(1)} cycles ({pat.split()[0]})")
        break
else:
    print(f"  {sys.argv[2]}: (cycle count not found in log)")
PY
  done
  echo "  real H100 target: ~166246 cycles"
fi

echo ""
echo "Logs: $LOG_DIR"
