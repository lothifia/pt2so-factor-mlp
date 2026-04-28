#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
CONFIG="${CONFIG:-Release}"
JOBS="${JOBS:-$(nproc)}"

if [[ -z "${Torch_DIR:-}" ]]; then
    LOCAL_TORCH_DIR="${ROOT}/external/libtorch/share/cmake/Torch"
    if [[ -d "${LOCAL_TORCH_DIR}" ]]; then
        export Torch_DIR="${LOCAL_TORCH_DIR}"
    fi
fi

if [[ -z "${Torch_DIR:-}" ]]; then
    echo "Torch_DIR is not set. Point it to external/libtorch/share/cmake/Torch." >&2
    exit 1
fi

cmake -S "${ROOT}/runtime" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${CONFIG}"
cmake --build "${BUILD_DIR}" --config "${CONFIG}" -j "${JOBS}"
