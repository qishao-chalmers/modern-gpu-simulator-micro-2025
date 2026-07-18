#!/usr/bin/env bash
# Fast control-bit patch after a trace with TRACER_SKIP_CUBIN_DUMP=1.
#
# Example:
#   export TRACER_SKIP_CUBIN_DUMP=1
#   export DYNAMIC_KERNEL_LIMIT_START=2628 DYNAMIC_KERNEL_LIMIT_END=2750
#   LD_PRELOAD=./tracer_tool.so llama-bench ...
#
#   ./patch_control_bits.sh \
#     /path/to/libggml-cuda.so \
#     traces/extra_info/enhanced_execution_info.json \
#     rope_neox,set_rows

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_SO="${1:?libggml-cuda.so path}"
JSON="${2:?enhanced_execution_info.json path}"
MATCH="${3:-}"

exec python3 "${SCRIPT_DIR}/patch_control_bits_from_cubin.py" patch \
  --cuda-binary "${CUDA_SO}" \
  --json "${JSON}" \
  ${MATCH:+--match "${MATCH}"}
