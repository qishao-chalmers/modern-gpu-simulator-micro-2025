#!/usr/bin/env bash
# Generate a weight-region JSON for the quantized-weight DRAM compression experiment.
#
# It wraps detect_weight_regions.py: detects the weight-matrix DRAM address span of
# each GEMV/GEMM kernel (mul_mat_vec_q / mul_mat_q) purely from the trace's own per-CTA
# recorded addresses, and writes a kernel_id -> region map. Non-GEMM kernels are skipped.
# The output feeds accel-sim.out via -quantized_weight_region_file.
#
# INPUTS:
#   $1  TRACE      path to dynamic_trace.pb            (required)
#   $2  KERNELS    kernel id(s) or range, e.g. 2644 or 2622-2669 or "2644 2647 2649"  (required)
#   $3  OUT        output JSON path                    (optional; default: ./weight_regions.json
#                                                       next to detect_weight_regions.py)
#
# detect_weight_regions.py hardcodes sys.path=/tmp/pb_py for the compiled protobufs, so this
# wrapper (re)builds them there from util/traces_enhanced/dynamic_trace/*.proto if missing.
#
# Examples:
#   ./gen_weight_regions.sh /path/qwen14b/decode_traces/dynamic_trace.pb 2622-2669 \
#                           /path/weight_regions_qwen14b.json
#   ./gen_weight_regions.sh /path/dynamic_trace.pb "2644 2647 2668"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/../util/traces_enhanced/dynamic_trace"
DETECT="$SCRIPT_DIR/detect_weight_regions.py"
PB_PY="/tmp/pb_py"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <dynamic_trace.pb> <kernel_id|start-end|\"id id ...\"> [out.json]" >&2
  exit 1
fi
TRACE="$1"
KERNELS="$2"
OUT="${3:-$SCRIPT_DIR/weight_regions.json}"

[[ -f "$TRACE" ]]   || { echo "ERROR: trace not found: $TRACE" >&2; exit 1; }
[[ -f "$DETECT" ]]  || { echo "ERROR: detect_weight_regions.py not found: $DETECT" >&2; exit 1; }

# Ensure the compiled protobufs exist where detect_weight_regions.py expects them.
if [[ ! -f "$PB_PY/trace_pb2.py" ]]; then
  echo "[*] compiling protobufs -> $PB_PY"
  mkdir -p "$PB_PY"
  protoc --python_out="$PB_PY" --proto_path="$PROTO_DIR" "$PROTO_DIR"/*.proto
fi

echo "[*] trace   : $TRACE"
echo "[*] kernels : $KERNELS"
echo "[*] output  : $OUT"
# detect_weight_regions.py takes the trace then space-separated kernel ids/ranges.
OUT_PATH="$OUT" python3 "$DETECT" "$TRACE" $KERNELS
echo "[*] done -> $OUT"
