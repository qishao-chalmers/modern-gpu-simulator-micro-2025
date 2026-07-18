#!/usr/bin/env bash
# Run simulation on bundled Rodinia Ampere example traces (protobuf / enhanced traces).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-SM86_RTXA6000}"
APP="${APP:-backprop-rodinia-2.0-ft/4096___data_result_4096_txt}"
SIM="$ROOT/gpu-simulator/bin/release/accel-sim.out"

if [ ! -x "$SIM" ]; then
    echo "ERROR: $SIM not found. Build first: BUILD_SYSTEM=cmake ./build.sh" >&2
    exit 1
fi

if [ -z "${CUDA_INSTALL_PATH:-}" ]; then
    [ -d /usr/local/cuda ] && export CUDA_INSTALL_PATH=/usr/local/cuda || export CUDA_INSTALL_PATH=/usr
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

EXTRACT_DIR="$ROOT/exampleTraces/extracted/rodinia2"
if [ ! -d "$EXTRACT_DIR" ]; then
    if [ ! -f "$ROOT/exampleTraces/rodinia2Ampere.tar.gz" ]; then
        echo "ERROR: missing example traces. Expected:" >&2
        echo "  $ROOT/exampleTraces/rodinia2Ampere.tar.gz" >&2
        exit 1
    fi
    mkdir -p "$ROOT/exampleTraces/extracted"
    tar -xzf "$ROOT/exampleTraces/rodinia2Ampere.tar.gz" -C "$ROOT/exampleTraces/extracted"
fi

TRACE=$(find "$EXTRACT_DIR" -path "*${APP}*/traces/dynamic_trace.pb" | head -1)
if [ -z "$TRACE" ]; then
    echo "ERROR: could not find trace for APP=$APP under $EXTRACT_DIR" >&2
    exit 1
fi

cd "$ROOT/gpu-simulator"
source ./setup_environment_no_git.sh release

GPGPUCFG="$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/$CONFIG/gpgpusim.config"
TRACECFG="$ROOT/gpu-simulator/configs/tested-cfgs/$CONFIG/trace.config"

echo "Running: $APP"
echo "Simulator: $SIM"
echo "Trace:     $TRACE"
echo "Config:    $CONFIG"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OMP_PROC_BIND=spread \
    "$SIM" \
    -config "$GPGPUCFG" \
    -config "$TRACECFG" \
    -trace "$TRACE"
