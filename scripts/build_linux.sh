#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
CONFIG="${CONFIG:-Release}"
JOBS="${JOBS:-$(nproc)}"
STATIC_OPENSSL="${STATIC_OPENSSL:-ON}"
CHECK_RUNTIME_DEPS="${CHECK_RUNTIME_DEPS:-ON}"

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

cmake_args=(
    -S "${ROOT}/runtime"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${CONFIG}"
    -DPT2SO_STATIC_OPENSSL="${STATIC_OPENSSL}"
)

if [[ -n "${LIBCRYPTO_A:-}" ]]; then
    cmake_args+=("-DPT2SO_LIBCRYPTO_A=${LIBCRYPTO_A}")
fi

if [[ -n "${OPENSSL_INCLUDE_DIR:-}" ]]; then
    cmake_args+=("-DPT2SO_OPENSSL_INCLUDE_DIR=${OPENSSL_INCLUDE_DIR}")
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" --config "${CONFIG}" -j "${JOBS}"

if [[ "${CHECK_RUNTIME_DEPS}" == "ON" && -f "${BUILD_DIR}/libmodel.so" ]]; then
    "${SCRIPT_DIR}/check_runtime_deps_linux.sh" "${BUILD_DIR}/libmodel.so"
fi
