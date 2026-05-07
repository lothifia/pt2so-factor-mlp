from __future__ import annotations

import argparse
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "factor_mlp"
DEFAULT_OUTPUT = ROOT / "runtime" / "src" / "blob.cpp"

ENCRYPTED_MAGIC = b"PT2SO_E2"
PLAINTEXT_MAGIC = b"PT2SO_P1"
AES_128_GCM_ALGORITHM_ID = 7
AES_GCM_NONCE_SIZE = 12


def asm_string_literal(path: Path) -> str:
    text = path.resolve().as_posix()
    return text.replace("\\", "\\\\").replace('"', '\\"')


def validate_encrypted_input(encrypted_path: Path) -> None:
    encrypted = encrypted_path.read_bytes()
    if len(encrypted) < 8 or encrypted[:8] != ENCRYPTED_MAGIC:
        raise ValueError(f"invalid encrypted model magic in {encrypted_path}")
    if len(encrypted) < 16:
        raise ValueError(f"truncated encrypted model header in {encrypted_path}")
    algorithm_id, nonce_len = struct.unpack_from("<II", encrypted, 8)
    if algorithm_id != AES_128_GCM_ALGORITHM_ID:
        raise ValueError(
            f"unsupported encrypted model algorithm id {algorithm_id}; expected {AES_128_GCM_ALGORITHM_ID}"
        )
    if nonce_len != AES_GCM_NONCE_SIZE:
        raise ValueError(f"invalid AES-128-GCM nonce length {nonce_len}; expected {AES_GCM_NONCE_SIZE}")

    ciphertext_len_offset = 16 + nonce_len
    if len(encrypted) < ciphertext_len_offset + 8:
        raise ValueError(f"truncated encrypted model before ciphertext length in {encrypted_path}")
    (ciphertext_len,) = struct.unpack_from("<Q", encrypted, ciphertext_len_offset)
    if ciphertext_len < 16:
        raise ValueError("AES-128-GCM ciphertext must include a 16-byte tag")
    expected_size = ciphertext_len_offset + 8 + ciphertext_len
    if len(encrypted) != expected_size:
        raise ValueError(
            f"encrypted model size mismatch: header says {expected_size} bytes, file has {len(encrypted)} bytes"
        )


def validate_plaintext_input(plaintext_path: Path) -> None:
    data = plaintext_path.read_bytes()
    if not data:
        raise ValueError(f"plaintext model is empty: {plaintext_path}")


def generate_blob(input_path: Path, output_path: Path, encrypted: bool) -> None:
    if encrypted:
        validate_encrypted_input(input_path)
    else:
        validate_plaintext_input(input_path)

    model_blob_incbin = asm_string_literal(input_path)
    plaintext_prefix = ""
    if not encrypted:
        plaintext_prefix = f'''
    ".ascii \\"{PLAINTEXT_MAGIC.decode("ascii")}\\"\\n"
    ".quad {input_path.stat().st_size}\\n"'''

    output = f'''#include "blob.h"

extern "C" {{
extern const uint8_t pt2so_model_blob_start[];
extern const uint8_t pt2so_model_blob_end[];
}}

#if !defined(__GNUC__) && !defined(__clang__)
#error "The embedded blob uses GNU-style inline assembly .incbin and requires GCC or Clang."
#endif

__asm__(
    ".section .rodata.pt2so,\\"a\\",@progbits\\n"
    ".balign 16\\n"
    ".global pt2so_model_blob_start\\n"
    ".hidden pt2so_model_blob_start\\n"
    "pt2so_model_blob_start:\\n"
{plaintext_prefix}
    ".incbin \\"{model_blob_incbin}\\"\\n"
    ".global pt2so_model_blob_end\\n"
    ".hidden pt2so_model_blob_end\\n"
    "pt2so_model_blob_end:\\n"
    ".size pt2so_model_blob_start, pt2so_model_blob_end - pt2so_model_blob_start\\n"
    ".previous\\n"
);

const uint8_t* embedded_encrypted_model_data() {{
    return pt2so_model_blob_start;
}}

size_t embedded_encrypted_model_size() {{
    return static_cast<size_t>(
        reinterpret_cast<uintptr_t>(pt2so_model_blob_end) -
        reinterpret_cast<uintptr_t>(pt2so_model_blob_start));
}}
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate runtime/src/blob.cpp that embeds a TorchScript model blob with .incbin."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--encrypted", type=Path, default=ARTIFACT_DIR / "weights.enc")
    group.add_argument("--plaintext", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = args.plaintext if args.plaintext is not None else args.encrypted
    generate_blob(input_path, args.output, encrypted=args.plaintext is None)
    print(f"generated_incbin_blob={args.output}")


if __name__ == "__main__":
    main()
