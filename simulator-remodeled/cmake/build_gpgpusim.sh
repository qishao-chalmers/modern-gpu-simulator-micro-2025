#!/usr/bin/env bash
# Build bundled GPGPU-Sim with a sane PATH for CMake custom commands (HPC-safe).
set -eo pipefail

GPGPUSIM_ROOT="$1"
ACCELSIM_CONFIG="$2"
CUDA_INSTALL_PATH="$3"
CC="$4"
CXX="$5"
JOBS="$6"

export CUDA_INSTALL_PATH
export CC
export CXX

_cc_bin="$(dirname "$CC")"
_cxx_bin="$(dirname "$CXX")"
export PATH="${_cc_bin}:${_cxx_bin}:${CUDA_INSTALL_PATH}/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

if ! command -v make >/dev/null 2>&1; then
    echo "ERROR: make not found in PATH=$PATH" >&2
    exit 127
fi

# shellcheck source=/dev/null
source "${GPGPUSIM_ROOT}/setup_environment" "${ACCELSIM_CONFIG}"
exec make -C "${GPGPUSIM_ROOT}" -j"${JOBS}" CC="${CC}" CXX="${CXX}"
