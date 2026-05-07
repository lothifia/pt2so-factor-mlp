#!/usr/bin/env bash
set -euo pipefail

LIB_PATH="${1:-build/libmodel.so}"

if [[ ! -f "${LIB_PATH}" ]]; then
    echo "shared library not found: ${LIB_PATH}" >&2
    exit 1
fi

if ! command -v readelf >/dev/null 2>&1; then
    echo "readelf is required to check runtime dependencies" >&2
    exit 1
fi

needed="$(readelf -d "${LIB_PATH}" | grep '(NEEDED)' || true)"

if echo "${needed}" | grep -Eq 'libcrypto\.so|libssl\.so'; then
    echo "forbidden crypto runtime dependency found in ${LIB_PATH}:" >&2
    echo "${needed}" >&2
    exit 1
fi

echo "runtime_dependency_check=ok"
echo "${needed}"
