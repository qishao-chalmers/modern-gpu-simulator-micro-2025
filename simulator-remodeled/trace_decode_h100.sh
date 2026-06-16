#!/usr/bin/env bash
# Re-trace llama.cpp decode kernels on H100 with the modern NVBit tracer (protobuf output).
#
# Run this ON the H100 node after building the tracer (CUDA 12+, sm_90).
# Classic kernelslist.g traces under gpu_traces/decode/ cannot be simulated directly.
#
# Usage:
#   ./trace_decode_h100.sh /path/to/llama-bench [args...]
#
#   OUT_DIR=/scratch/decode_pb ./trace_decode_h100.sh ./llama-bench -m model.gguf ...
#   KERNEL_START=2364 KERNEL_END=2404 ./trace_decode_h100.sh ./llama-bench ...
#
# Output layout (under $OUT_DIR/traces/):
#   dynamic_trace.pb
#   threadblocks/device_*/stream_*/kernel_*/*.pb
#   extra_info/enhanced_execution_info.json
#
# Then simulate locally or on cluster:
#   ./run_decode_trace.sh "$OUT_DIR/traces/dynamic_trace.pb"
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_SO="$ROOT/util/tracer_nvbit/tracer_tool/tracer_tool.so"
OUT_DIR="${OUT_DIR:-$ROOT/../gpu_traces/decode_retrace}"
KERNEL_START="${KERNEL_START:-2364}"
KERNEL_END="${KERNEL_END:-2404}"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/app [app args...]" >&2
    echo "  Example: KERNEL_START=2364 KERNEL_END=2404 $0 ./llama-bench -m model.gguf -p 1024 -n 64 ..." >&2
    exit 1
fi
APP="$1"
shift

if [ ! -x "$APP" ] && [ ! -f "$APP" ]; then
    echo "ERROR: application not found: $APP" >&2
    exit 1
fi
if [ ! -f "$TRACER_SO" ]; then
    echo "ERROR: tracer not built: $TRACER_SO" >&2
    echo "  On H100: module load protobuf/24 GCCcore/12 CUDA/12.2" >&2
    echo "  cd $ROOT && unset CXX CC && ARCH=sm_90 BUILD_SYSTEM=cmake ./build.sh" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

export ARCH="${ARCH:-sm_90}"
export CUDA_VISIBLE_DEVICES="$DEVICE"
export DYNAMIC_KERNEL_LIMIT_START="$KERNEL_START"
export DYNAMIC_KERNEL_LIMIT_END="$KERNEL_END"
export TERMINATE_UPON_LIMIT="${TERMINATE_UPON_LIMIT:-1}"
export EXCLUDE_PRED_OFF="${EXCLUDE_PRED_OFF:-1}"
export TOOL_VERBOSE="${TOOL_VERBOSE:-0}"

# Tracer writes protobuf under ./traces/ relative to cwd (do not set USER_DEFINED_FOLDERS=1;
# that path only retargets intermediate .trace files, not dynamic_trace.pb).
export CUDA_INJECTION64_PATH="$TRACER_SO"
export LD_PRELOAD="$TRACER_SO"

echo "Tracer:       $TRACER_SO"
echo "Output dir:   $OUT_DIR/traces/"
echo "ARCH:         $ARCH"
echo "GPU:          CUDA_VISIBLE_DEVICES=$DEVICE"
echo "Kernel range: $KERNEL_START .. $KERNEL_END (terminate=$TERMINATE_UPON_LIMIT)"
echo "Command:      $APP $*"
echo

"$APP" "$@"

if [ ! -f "$OUT_DIR/traces/dynamic_trace.pb" ]; then
    echo "ERROR: tracer did not produce traces/dynamic_trace.pb under $OUT_DIR" >&2
    echo "  Check stats.csv and that kernel IDs match this workload run." >&2
    exit 1
fi

echo
echo "Protobuf traces ready:"
echo "  $OUT_DIR/traces/dynamic_trace.pb"
echo
echo "Simulate:"
echo "  CONFIG=SM90_H100 FILTER_FIRST=$KERNEL_START FILTER_LAST=$KERNEL_END \\"
echo "    $ROOT/run_custom_trace.sh $OUT_DIR/traces/dynamic_trace.pb"
