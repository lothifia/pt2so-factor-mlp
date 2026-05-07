#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACT_DIR="${ROOT}/artifacts/factor_mlp"
BENCH_DIR="${ROOT}/artifacts/crypto_bench"
BUILD_ROOT="${ROOT}/build/crypto_bench"
REPEAT="${REPEAT:-5}"
WARMUP="${WARMUP:-1}"
CXX="${CXX:-g++}"
MODEL_ARGS="${MODEL_ARGS:-}"
ALGORITHM="gcm-aes-128"

cd "${ROOT}"
mkdir -p "${ARTIFACT_DIR}" "${BENCH_DIR}" "${BUILD_ROOT}"

echo "[1/5] generating one TorchScript model shared by all cases" >&2
# shellcheck disable=SC2086
python tools/create_factor_mlp.py --output-dir "${ARTIFACT_DIR}" ${MODEL_ARGS}

echo "[2/5] compiling dlopen create benchmark helper" >&2
"${CXX}" -std=c++17 -O2 -Iruntime/include tools/benchmark_create.cpp -ldl -o "${BUILD_ROOT}/benchmark_create"

RESULT_CSV="${BENCH_DIR}/results.csv"
echo "algorithm,iteration,encrypted_bytes,plaintext_bytes,decrypt_ms,jit_load_ms,total_create_inner_ms,total_create_wall_ms" > "${RESULT_CSV}"

echo "[3/5] embedding plaintext model.pt into libmodel.so and benchmarking model_create" >&2
PLAIN_EMBED_BUILD_DIR="${BUILD_ROOT}/plaintext-embedded"
mkdir -p "${PLAIN_EMBED_BUILD_DIR}"
python tools/generate_blob.py \
    --plaintext "${ARTIFACT_DIR}/factor_mlp.pt" \
    --output "${ROOT}/runtime/src/blob.cpp"
BUILD_DIR="${PLAIN_EMBED_BUILD_DIR}" "${SCRIPT_DIR}/build_linux.sh" >&2
"${BUILD_ROOT}/benchmark_create" \
    --lib "${PLAIN_EMBED_BUILD_DIR}/libmodel.so" \
    --repeat "${REPEAT}" \
    --warmup "${WARMUP}" >> "${RESULT_CSV}"

echo "[4/5] encrypting, embedding, building, and benchmarking ${ALGORITHM}" >&2
ALGO_ARTIFACT_DIR="${BENCH_DIR}/${ALGORITHM}"
ALGO_BUILD_DIR="${BUILD_ROOT}/${ALGORITHM}"
mkdir -p "${ALGO_ARTIFACT_DIR}" "${ALGO_BUILD_DIR}"

python tools/encrypt_model.py \
    --algorithm "${ALGORITHM}" \
    --input "${ARTIFACT_DIR}/factor_mlp.pt" \
    --output "${ALGO_ARTIFACT_DIR}/weights.enc" \
    --key-source "${ROOT}/runtime/src/aes_key_obfuscated.cpp"

python tools/generate_blob.py \
    --encrypted "${ALGO_ARTIFACT_DIR}/weights.enc" \
    --output "${ROOT}/runtime/src/blob.cpp"

BUILD_DIR="${ALGO_BUILD_DIR}" "${SCRIPT_DIR}/build_linux.sh" >&2

"${BUILD_ROOT}/benchmark_create" \
    --lib "${ALGO_BUILD_DIR}/libmodel.so" \
    --repeat "${REPEAT}" \
    --warmup "${WARMUP}" >> "${RESULT_CSV}"

echo "[5/5] summary" >&2
python tools/summarize_crypto_benchmark.py "${RESULT_CSV}"
echo "crypto_benchmark_results=${RESULT_CSV}"
