#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
CXX="${CXX:-g++}"
CONFIG="${CONFIG:-Debug}"

if ! command -v "${CXX}" >/dev/null 2>&1; then
    echo "C++ compiler not found: ${CXX}" >&2
    echo "Set CXX=/path/to/compiler or install g++/clang++." >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"

case "${CONFIG}" in
    Debug)
        OPT_FLAGS=(-O0 -g)
        ;;
    Release)
        OPT_FLAGS=(-O2 -DNDEBUG)
        ;;
    *)
        echo "unsupported CONFIG=${CONFIG}; use Debug or Release" >&2
        exit 1
        ;;
esac

"${CXX}" \
    -std=c++17 \
    "${OPT_FLAGS[@]}" \
    -Wall \
    -Wextra \
    -I"${ROOT}/runtime/include" \
    "${ROOT}/tools/test_data_source.cpp" \
    -pthread \
    -o "${BUILD_DIR}/test_data_source"

"${BUILD_DIR}/test_data_source"
