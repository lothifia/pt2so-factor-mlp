#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACT_DIR="${ROOT}/artifacts/factor_mlp"

cd "${ROOT}"

python tools/create_factor_mlp.py --output-dir "${ARTIFACT_DIR}"
python tools/extract_weights.py \
    --checkpoint "${ARTIFACT_DIR}/factor_mlp.pt" \
    --model-info "${ARTIFACT_DIR}/model_info.yaml" \
    --output "${ARTIFACT_DIR}/weights.bin"
python tools/encrypt_weights.py \
    --input "${ARTIFACT_DIR}/weights.bin" \
    --output "${ARTIFACT_DIR}/weights.enc" \
    --key-output "${ARTIFACT_DIR}/weights.key"
python tools/generate_blob.py \
    --encrypted "${ARTIFACT_DIR}/weights.enc" \
    --key "${ARTIFACT_DIR}/weights.key" \
    --output "${ROOT}/runtime/src/blob.cpp"

"${SCRIPT_DIR}/build_linux.sh"
python tools/validate_ctypes.py
