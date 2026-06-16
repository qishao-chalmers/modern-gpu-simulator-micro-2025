#!/usr/bin/env bash
# Build the modern GPU simulator (tracer + accel-sim.out)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${JOBS:-$(nproc)}"
BUILD_SYSTEM="${BUILD_SYSTEM:-make}"   # make | cmake
ACCELSIM_CONFIG="${ACCELSIM_CONFIG:-release}"   # release | debug

_is_cross_compiler() {
    case "${1:-}" in
        *riscv*|*arm-linux*|*aarch64*|*musl*)
            return 0 ;;
    esac
    return 1
}

_select_host_compiler() {
    local _pick_cc _pick_cxx
    # EasyBuild (BSC MN5): prefer loaded GCCcore module over /usr/bin.
    if [ -n "${EBROOTGCCCORE:-}" ] && [ -x "${EBROOTGCCCORE}/bin/g++" ]; then
        _pick_cxx="${EBROOTGCCCORE}/bin/g++"
        _pick_cc="${EBROOTGCCCORE}/bin/gcc"
    elif command -v g++-10 >/dev/null 2>&1; then
        _pick_cxx="$(command -v g++-10)"
        _pick_cc="$(command -v gcc-10 2>/dev/null || echo "${_pick_cxx/g++/gcc}")"
    elif command -v g++ >/dev/null 2>&1; then
        _pick_cxx="$(command -v g++)"
        _pick_cc="$(command -v gcc 2>/dev/null || echo "${_pick_cxx/g++/gcc}")"
    else
        echo "ERROR: no host g++ found (load GCCcore module, or install g++-10/g++)" >&2
        exit 1
    fi
    if [ -n "${CXX:-}" ] && [ "${CXX}" != "${_pick_cxx}" ] && _is_cross_compiler "${CXX}"; then
        echo "NOTE: Ignoring cross-compiler CXX=${CXX}; using host ${_pick_cxx} for CUDA/NVCC"
    fi
    export CXX="${_pick_cxx}"
    export CC="${_pick_cc}"
    export HOST_CC="${_pick_cc}"
    export HOST_CXX="${_pick_cxx}"
    export PATH="$(dirname "${_pick_cc}"):$(dirname "${_pick_cxx}"):${PATH:-}"
    echo "    HOST_CC=${CC}  HOST_CXX=${HOST_CXX}"
}

_setup_protobuf() {
    if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists protobuf 2>/dev/null; then
        echo "ERROR: protobuf not visible to pkg-config." >&2
        echo "  On BSC MN5 load modules first, e.g.:" >&2
        echo "    module load protobuf/24 GCCcore/12 CUDA/12.2" >&2
        exit 1
    fi

    local prefix lib_ver protoc_ver want stamp te
    prefix=$(pkg-config --variable=prefix protobuf)
    lib_ver=$(pkg-config --modversion protobuf)
    export PROTOC="${PROTOC:-${prefix}/bin/protoc}"

    if [ ! -x "$PROTOC" ]; then
        echo "ERROR: protoc not executable at ${PROTOC}" >&2
        echo "  Load the protobuf module or set PROTOC to the module's protoc." >&2
        exit 1
    fi

    protoc_ver=$("$PROTOC" --version | awk '{print $2}')
    local includedir libdir
    includedir=$(pkg-config --variable=includedir protobuf)
    libdir=$(pkg-config --variable=libdir protobuf)
    echo "    protobuf pkg-config=${lib_ver}  protoc=${protoc_ver}"
    echo "    PROTOC=${PROTOC}"
    echo "    PROTOBUF includedir=${includedir}"
    echo "    PROTOBUF_CFLAGS=$(pkg-config --cflags protobuf) -I${includedir}"

    # Help sub-makes and nvcc find module headers/libs (skip /usr/include — breaks g++).
    if [ "${includedir}" != "/usr/include" ]; then
        export CPATH="${includedir}${CPATH:+:$CPATH}"
        export C_INCLUDE_PATH="${includedir}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
        export CPLUS_INCLUDE_PATH="${includedir}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    fi
    export LIBRARY_PATH="${libdir}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    if [ -f /usr/include/google/protobuf/message.h ] && [ -f "${includedir}/google/protobuf/message.h" ]; then
        echo "    NOTE: both system and module protobuf headers exist; build forces -I${includedir}"
    fi

    # Drop stale generated code (e.g. rsync'd from a machine with another protobuf).
    te="$ROOT/util/traces_enhanced"
    stamp="$te/.protoc_stamp"
    want="${protoc_ver}|${lib_ver}|${prefix}"
    if [ ! -d "$te/pb_trace" ] || [ ! -f "$stamp" ] || [ "$(cat "$stamp" 2>/dev/null)" != "$want" ]; then
        echo "    Regenerating protobuf sources (protoc/lib mismatch or stale pb_trace)"
        rm -rf "$te/pb_trace"
        echo "$want" > "$stamp"
    fi
}

_setup_cmake() {
    if command -v cmake >/dev/null 2>&1; then
        CMAKE_BIN="$(command -v cmake)"
    elif [ -x "/gpfs/apps/MN5/GPP/CMAKE/3.25.1/bin/cmake" ]; then
        CMAKE_BIN="/gpfs/apps/MN5/GPP/CMAKE/3.25.1/bin/cmake"
    else
        echo "ERROR: cmake not found. Load a cmake module or add cmake to PATH." >&2
        exit 1
    fi
    export CMAKE_BIN
    echo "    CMAKE=${CMAKE_BIN}"
}

# build-cmake must be configured on this machine (never rsync from another host).
_prepare_cmake_build_dir() {
    local cache="$ROOT/build-cmake/CMakeCache.txt"
    if [ -f "$cache" ]; then
        local cached_root
        cached_root="$(grep -m1 '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$cache" | cut -d= -f2-)"
        if [ -n "$cached_root" ] && [ "$cached_root" != "$ROOT" ]; then
            echo "    Removing build-cmake (CMake cache from ${cached_root})"
            rm -rf "$ROOT/build-cmake"
        fi
    fi
    if [ -d "$ROOT/build-cmake" ]; then
        if ! grep -q "^CMAKE_COMMAND:INTERNAL=${CMAKE_BIN}$" "$ROOT/build-cmake/CMakeCache.txt" 2>/dev/null; then
            echo "    Removing build-cmake (cmake path changed)"
            rm -rf "$ROOT/build-cmake"
        fi
    fi
}
if [ -z "${CUDA_INSTALL_PATH:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_INSTALL_PATH=/usr/local/cuda
    elif command -v nvcc >/dev/null 2>&1; then
        export CUDA_INSTALL_PATH=/usr
    else
        echo "ERROR: set CUDA_INSTALL_PATH to your CUDA install" >&2
        exit 1
    fi
fi
export PATH="${CUDA_INSTALL_PATH}/bin:/usr/bin:${PATH}"

_select_host_compiler

_setup_protobuf

# NVCC < 11.7 requires explicit ARCH (sm_80=A100, sm_90=Hopper/H100)
if [ -z "${ARCH:-}" ]; then
    NVCC_VER=$(nvcc --version | grep release | sed -re 's/.*release ([0-9]+)\.([0-9]+).*/\1 \2/')
    NVCC_MAJOR=$(echo "$NVCC_VER" | awk '{print $1}')
    NVCC_MINOR=$(echo "$NVCC_VER" | awk '{print $2}')
    if [ "$NVCC_MAJOR" -lt 11 ] || { [ "$NVCC_MAJOR" -eq 11 ] && [ "$NVCC_MINOR" -lt 7 ]; }; then
        export ARCH="${ARCH:-sm_80}"
        echo "NOTE: NVCC < 11.7 detected; defaulting ARCH=$ARCH (override with export ARCH=sm_90 for Hopper)"
    fi
fi

echo "==> Building tracer (util/tracer_nvbit)"
echo "    CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH}  HOST_CXX=${HOST_CXX}  ARCH=${ARCH:-all}"
cd "$ROOT/util/tracer_nvbit"
if [ ! -d nvbit_release ]; then
    ./install_nvbit.sh
fi
make clean
make -j"$JOBS"

echo "==> Building simulator (gpu-simulator) via ${BUILD_SYSTEM} [${ACCELSIM_CONFIG}]"
if [ "$BUILD_SYSTEM" = "cmake" ]; then
    cd "$ROOT"
    _setup_cmake
    _prepare_cmake_build_dir
    # Remove stale gpgpu-sim objects built with wrong gcc-/cuda-* path (common on first cluster build).
    rm -rf "$ROOT/gpu-simulator/gpgpu-sim/build/gcc-" "$ROOT/gpu-simulator/gpgpu-sim/lib/gcc-"
    rm -rf "$ROOT/build-cmake"
    "$CMAKE_BIN" -S "$ROOT" -B "$ROOT/build-cmake" \
        -DCMAKE_CXX_COMPILER="${HOST_CXX}" \
        -DCMAKE_C_COMPILER="${CC}" \
        -DACCELSIM_CONFIG="${ACCELSIM_CONFIG}"
    "$CMAKE_BIN" --build "$ROOT/build-cmake" -j"$JOBS"
else
    cd "$ROOT/gpu-simulator"
    source ./setup_environment_no_git.sh "${ACCELSIM_CONFIG}"
    make clean
    make -j"$JOBS"
fi

echo ""
echo "Build complete:"
echo "  Tracer:    $ROOT/util/tracer_nvbit/tracer_tool/tracer_tool.so"
echo "  Simulator: $ROOT/gpu-simulator/bin/${ACCELSIM_CONFIG}/accel-sim.out"
echo ""
echo "Debug build:   ACCELSIM_CONFIG=debug BUILD_SYSTEM=cmake ./build.sh"
echo "Release build: ACCELSIM_CONFIG=release BUILD_SYSTEM=cmake ./build.sh"
echo "CMake rebuild: BUILD_SYSTEM=cmake ./build.sh"
echo "Makefile only: ./build.sh"
