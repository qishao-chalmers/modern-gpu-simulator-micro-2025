#!/usr/bin/env bash
# Run accel-sim.out in trace mode on enhanced (protobuf) traces from the modern NVBit tracer.
#
# Usage:
#   ./run_custom_trace.sh /path/to/traces/dynamic_trace.pb
#   CONFIG=SM90_H100 ./run_custom_trace.sh /path/to/traces/dynamic_trace.pb
#   CONFIG=SM89_RTX4090 ./run_custom_trace.sh /path/to/traces/dynamic_trace.pb
#   FILTER_FIRST=2364 FILTER_LAST=2393 ./run_custom_trace.sh /path/to/traces/dynamic_trace.pb
#
# Trace directory must contain:
#   dynamic_trace.pb
#   threadblocks/device_*/stream_*/kernel_*/*.pb
#   extra_info/enhanced_execution_info.json  (recommended; enable with -is_extra_traces_enabled 1)
#
# Legacy Accel-Sim traces (kernelslist.g + kernel-*.trace.xz) are NOT supported here.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_PB="${1:-}"
CONFIG="${CONFIG:-SM90_H100}"
FILTER_FIRST="${FILTER_FIRST:-0}"
FILTER_LAST="${FILTER_LAST:-0}"
SIM="$ROOT/gpu-simulator/bin/release/accel-sim.out"

if [ -z "$TRACE_PB" ]; then
    echo "Usage: $0 /path/to/traces/dynamic_trace.pb" >&2
    exit 1
fi
if [ ! -f "$TRACE_PB" ]; then
    echo "ERROR: trace file not found: $TRACE_PB" >&2
    exit 1
fi
if [ ! -x "$SIM" ]; then
    echo "ERROR: $SIM not found. Build first: BUILD_SYSTEM=cmake ./build.sh" >&2
    exit 1
fi

TRACE_DIR="$(dirname "$TRACE_PB")"
if [ ! -d "$TRACE_DIR/threadblocks" ]; then
    echo "ERROR: missing $TRACE_DIR/threadblocks/ (modern protobuf trace layout required)" >&2
    echo "  kernelslist.g + *.trace.xz from the classic tracer cannot be used." >&2
    exit 1
fi

if [ -z "${CUDA_INSTALL_PATH:-}" ]; then
    [ -d /usr/local/cuda ] && export CUDA_INSTALL_PATH=/usr/local/cuda || export CUDA_INSTALL_PATH=/usr
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

GPGPUCFG="$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/$CONFIG/gpgpusim.config"
TRACECFG="$ROOT/gpu-simulator/configs/tested-cfgs/$CONFIG/trace.config"
if [ ! -f "$GPGPUCFG" ] || [ ! -f "$TRACECFG" ]; then
    echo "ERROR: config $CONFIG not found under tested-cfgs/" >&2
    exit 1
fi

cd "$ROOT/gpu-simulator"
source ./setup_environment_no_git.sh release

EXTRA_ARGS=()
EXTRA_ARGS+=(-is_extra_traces_enabled 1)
if [ "$FILTER_LAST" != "0" ]; then
    EXTRA_ARGS+=(-filter_first_kernel_id "$FILTER_FIRST" -filter_last_kernel_id "$FILTER_LAST")
fi

echo "Simulator: $SIM"
echo "Trace:     $TRACE_PB"
echo "Config:    $CONFIG (H100/Hopper: SM90_H100; Ada fallback: SM89_RTX4090)"
echo "Filters:   first=$FILTER_FIRST last=$FILTER_LAST"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OMP_PROC_BIND=spread \
    "$SIM" \
    -config "$GPGPUCFG" \
    -config "$TRACECFG" \
    "${EXTRA_ARGS[@]}" \
    -trace "$TRACE_PB"
