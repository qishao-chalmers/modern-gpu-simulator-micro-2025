#!/usr/bin/env bash
# Simulate H100 decode-layer protobuf traces (llama.cpp tg kernels ~2364+).
#
# Usage:
#   ./run_decode_trace.sh /path/to/traces/dynamic_trace.pb
#   ./run_decode_trace.sh   # uses DECODE_TRACE_PB or default path below
#
# Requires modern protobuf traces (see trace_decode_h100.sh). Classic
# gpu_traces/decode/kernelslist.g + kernel-*.trace.xz will NOT work.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PB="$ROOT/../gpu_traces/decode_retrace/traces/dynamic_trace.pb"
TRACE_PB="${1:-${DECODE_TRACE_PB:-$DEFAULT_PB}}"

export CONFIG="${CONFIG:-SM90_H100}"
export FILTER_FIRST="${FILTER_FIRST:-2364}"
export FILTER_LAST="${FILTER_LAST:-2393}"

if [ ! -f "$TRACE_PB" ]; then
    echo "ERROR: protobuf trace not found: $TRACE_PB" >&2
    echo >&2
    echo "Classic traces at gpu_traces/decode/ (kernelslist.g + *.trace.xz) are not supported." >&2
    echo "Re-trace on H100 first:" >&2
    echo "  cd $ROOT" >&2
    echo "  KERNEL_START=2364 KERNEL_END=2404 ./trace_decode_h100.sh /path/to/llama-bench [args]" >&2
    exit 1
fi

exec "$ROOT/run_custom_trace.sh" "$TRACE_PB"
