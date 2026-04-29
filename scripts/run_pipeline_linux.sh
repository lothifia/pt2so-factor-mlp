#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACT_DIR="${ROOT}/artifacts/factor_mlp"

cd "${ROOT}"

python tools/create_factor_mlp.py --output-dir "${ARTIFACT_DIR}"
python tools/encrypt_model.py \
    --input "${ARTIFACT_DIR}/factor_mlp.pt" \
    --output "${ARTIFACT_DIR}/model.pt.enc" \
    --key-output "${ARTIFACT_DIR}/model.key"
python tools/generate_blob.py \
    --encrypted "${ARTIFACT_DIR}/model.pt.enc" \
    --output "${ROOT}/runtime/src/blob.cpp"

"${SCRIPT_DIR}/build_linux.sh"
export PT2SO_MODEL_KEY_FILE="${ARTIFACT_DIR}/model.key"
python tools/validate_ctypes.py
