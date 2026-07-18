#!/bin/bash
# ptxas shim for Accel-Sim + custom PTX opcode dp8a (int4×int4).
#
# GPGPU-Sim shells out to $CUDA_INSTALL_PATH/bin/ptxas -v <file.ptx> to get
# register counts. NVIDIA ptxas does not know dp8a → exit 255 / error 65280.
# This wrapper rewrites dp8a → dp4a (same arity) in a temp copy for that query
# only. The functional simulator still executes the original PTX with dp8a.
#
# Setup (once):
#   REPK=.../mmvq_kquant
#   FAKE=$REPK/cuda_dp8a_shim
#   mkdir -p "$FAKE/bin"
#   REAL_CUDA=$(dirname "$(dirname "$(command -v ptxas)")")
#   ln -sfn "$REAL_CUDA/bin/ptxas"     "$FAKE/bin/ptxas.real"
#   ln -sfn "$REAL_CUDA/bin/cuobjdump" "$FAKE/bin/cuobjdump"
#   ln -sfn "$REAL_CUDA/bin/nvdisasm"  "$FAKE/bin/nvdisasm" 2>/dev/null || true
#   cp "$REPK/ptxas_dp8a_wrap.sh" "$FAKE/bin/ptxas" && chmod +x "$FAKE/bin/ptxas"
#
# Run Accel-Sim with:
#   export CUDA_INSTALL_PATH=$FAKE
#   export PATH=$FAKE/bin:$PATH
#   export REAL_PTXAS=$REAL_CUDA/bin/ptxas

set -euo pipefail
REAL="${REAL_PTXAS:-$(dirname "$0")/ptxas.real}"
if [ ! -x "$REAL" ]; then
  # fall back to next ptxas on PATH that isn't us
  REAL=$(command -v -a ptxas | grep -v "$(dirname "$0")/ptxas" | head -1 || true)
fi
if [ -z "${REAL}" ] || [ ! -x "$REAL" ]; then
  echo "ptxas_dp8a_wrap: cannot find real ptxas (set REAL_PTXAS)" >&2
  exit 127
fi

args=()
tmp=""
for a in "$@"; do
  case "$a" in
    *.ptx)
      tmp=$(mktemp --suffix=.ptx)
      # GNU sed word-boundary rewrite for occupancy-only ptxas
      sed 's/\bdp8a\b/dp4a/g' "$a" > "$tmp"
      args+=("$tmp")
      ;;
    *)
      args+=("$a")
      ;;
  esac
done

set +e
"$REAL" "${args[@]}"
rc=$?
set -e
[ -n "$tmp" ] && rm -f "$tmp"
exit $rc
