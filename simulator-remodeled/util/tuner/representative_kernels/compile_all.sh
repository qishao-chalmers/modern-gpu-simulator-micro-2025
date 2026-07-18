#!/usr/bin/env bash
# Build all representative decode kernel benchmarks.
#
# Usage:
#   ./compile_all.sh              # native (GPU arch) + exec (sm_70 GPGPU-Sim)
#   ./compile_all.sh --native     # native only (trace / real GPU runs)
#   ./compile_all.sh --exec       # exec-driven only (GPGPU-Sim under Accel-Sim)
#   ./compile_all.sh --clean      # remove build outputs
#   ARCH=sm_80 ./compile_all.sh   # force native arch (sm_70, sm_80, sm_86, sm_89, sm_90)
#
# Outputs (per subdir):
#   native:  rms_norm_bench, quantize_bench, flash_attn_bench, mmvq_bitwidth, mmvq_bench
#   exec:    *_exec  (sm_70 + FUNCSIM_SAFE where applicable)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${MODE:-both}"   # both | native | exec
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

NATIVE_DIRS=(rms_norm quantize_q8_1 flash_attn mmvq_bitwidth mul_mat_vec_q)
EXEC_DIRS=(rms_norm quantize_q8_1 flash_attn mmvq_bitwidth)

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --native) MODE=native; shift ;;
        --exec)   MODE=exec; shift ;;
        --clean)  MODE=clean; shift ;;
        -j*)      JOBS="${1#-j}"; shift ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

detect_arch() {
    if [[ -n "${ARCH:-}" ]]; then
        case "$ARCH" in
            sm_*) echo "$ARCH"; return ;;
            compute_*) echo "sm_${ARCH#compute_}"; return ;;
            *) echo "$ARCH"; return ;;
        esac
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cap
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
        if [[ -n "$cap" ]]; then
            local major="${cap%%.*}"
            local minor="${cap##*.}"
            echo "sm_${major}${minor}"
            return
        fi
    fi
    if command -v nvcc >/dev/null 2>&1; then
        local ver
        ver="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)"
        # CUDA 12.8+ -> sm_90; 11.x -> sm_80; else sm_70
        if [[ "$ver" -ge 128 ]]; then echo sm_90; return; fi
        if [[ "$ver" -ge 110 ]]; then echo sm_80; return; fi
    fi
    echo sm_80
}

arch_to_gencode() {
    local sm="$1"
    local n="${sm#sm_}"
    local major minor
    if [[ ${#n} -eq 2 ]]; then
        major="${n:0:1}"
        minor="${n:1:1}"
    else
        major="${n:0:2}"
        minor="${n:2:1}"
    fi
    echo "-gencode=arch=compute_${major}${minor},code=sm_${major}${minor}"
}

do_clean() {
    echo "=== clean ==="
    for d in "${NATIVE_DIRS[@]}"; do
        echo "  $d"
        make -C "$d" clean 2>/dev/null || true
    done
}

build_native() {
    local arch gencode
    arch="$(detect_arch)"
    gencode="$(arch_to_gencode "$arch")"
    echo "=== native build  ARCH=$arch  ($gencode) ==="
    for d in "${NATIVE_DIRS[@]}"; do
        echo "--- $d ---"
        make -C "$d" -j"$JOBS" ARCH="$gencode" || {
            echo "warning: native build failed for $d (try ARCH=sm_80)" >&2
            return 1
        }
    done
    echo ""
    echo "Native binaries:"
    find "$ROOT" -maxdepth 2 -type f \( \
        -name 'rms_norm_bench' -o \
        -name 'quantize_bench' -o \
        -name 'flash_attn_bench' -o \
        -name 'mmvq_bitwidth' -o \
        -name 'mmvq_bench' \
    \) -not -name '*_exec' | sort | sed 's/^/  /'
}

build_exec() {
    echo "=== exec build (sm_70, GPGPU-Sim / Accel-Sim) ==="
    for d in "${EXEC_DIRS[@]}"; do
        echo "--- $d ---"
        make -C "$d" exec -j"$JOBS"
    done
    echo ""
    echo "Exec binaries:"
    find "$ROOT" -maxdepth 2 -type f -name '*_exec' | sort | sed 's/^/  /'
}

case "$MODE" in
    clean)
        do_clean
        ;;
    native)
        build_native
        ;;
    exec)
        build_exec
        ;;
    both)
        build_native
        echo ""
        build_exec
        ;;
    *)
        echo "internal error: MODE=$MODE" >&2
        exit 1
        ;;
esac

echo ""
echo "Done. Run under GPGPU-Sim:"
echo "  cd <bench_dir> && ./sweep_shapes.sh"
echo "  # or mmvq: cd mmvq_bitwidth && ./sweep_bits.sh 2"
