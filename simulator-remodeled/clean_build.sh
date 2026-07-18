#!/usr/bin/env bash
# Remove simulator + tracer build artifacts (safe to run before a fresh build).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_TRACER="${CLEAN_TRACER:-1}"   # 0 = keep tracer_tool.so build tree

echo "==> Cleaning build artifacts under ${ROOT}"

rm -rf "${ROOT}/build-cmake"

# gpgpu-sim object/libs (debug + release, all gcc/cuda variants)
rm -rf "${ROOT}/gpu-simulator/gpgpu-sim/build"
rm -rf "${ROOT}/gpu-simulator/gpgpu-sim/lib/gcc-"

# accel-sim Makefile build trees
rm -rf "${ROOT}/gpu-simulator/build"

if [ "${CLEAN_TRACER}" = "1" ]; then
    if [ -d "${ROOT}/util/tracer_nvbit/tracer_tool" ]; then
        make -C "${ROOT}/util/tracer_nvbit/tracer_tool" clean 2>/dev/null || true
    fi
    if [ -d "${ROOT}/util/tracer_nvbit" ]; then
        make -C "${ROOT}/util/tracer_nvbit" clean 2>/dev/null || true
    fi
fi

# Optional: remove installed binaries (comment out if you want to keep them)
# rm -f "${ROOT}/gpu-simulator/bin/release/accel-sim.out"
# rm -f "${ROOT}/gpu-simulator/bin/debug/accel-sim.out"

echo "==> Done. Rebuild with:"
echo "    ACCELSIM_CONFIG=debug BUILD_SYSTEM=cmake ./build.sh"
echo "    ACCELSIM_CONFIG=release BUILD_SYSTEM=cmake ./build.sh"
