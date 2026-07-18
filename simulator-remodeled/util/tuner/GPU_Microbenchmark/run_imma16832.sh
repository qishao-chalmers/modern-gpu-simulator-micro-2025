#!/usr/bin/env bash
# Build and run IMMA.16832.S8.S8 microbenchmarks (llama.cpp mul_mat_q opcode).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOPPER_H100="${HOPPER_H100:-1}"
MAKE_FLAGS=(HOPPER_H100="$HOPPER_H100")

IMMA_BENCHES=(
  tensor_lat_imma16832
  tensor_bw_imma16832
  config_tensor_imma16832
)

mkdir -p bin

for name in "${IMMA_BENCHES[@]}"; do
  dir="ubench/core/${name}"
  echo "=== Building ${name} (HOPPER_H100=${HOPPER_H100}) ==="
  make "${MAKE_FLAGS[@]}" -C "$dir" clean release
done

echo
for name in "${IMMA_BENCHES[@]}"; do
  echo "/////////////////////////////////"
  echo "running ./${name}"
  "./bin/${name}"
  echo "/////////////////////////////////"
done

echo
echo "Optional SASS check (expect IMMA.16832.S8.S8):"
echo "  make -C ubench/core/config_tensor_imma16832 sass"
