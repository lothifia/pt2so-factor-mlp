from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "factor_mlp"
DEFAULT_OUTPUT = ROOT / "runtime" / "src" / "blob.cpp"

MAGIC = b"PT2SO_E1"


def asm_string_literal(path: Path) -> str:
    text = path.resolve().as_posix()
    return text.replace("\\", "\\\\").replace('"', '\\"')


def validate_inputs(encrypted_path: Path, key_path: Path) -> None:
    encrypted = encrypted_path.read_bytes()
    if len(encrypted) < 8 or encrypted[:8] != MAGIC:
        raise ValueError(f"invalid encrypted weights magic in {encrypted_path}")

    key = key_path.read_bytes()
    if len(key) != 32:
        raise ValueError("AES-256-GCM key file must contain exactly 32 bytes")


def generate_blob(encrypted_path: Path, key_path: Path, output_path: Path) -> None:
    validate_inputs(encrypted_path, key_path)

    encrypted_incbin = asm_string_literal(encrypted_path)
    key_incbin = asm_string_literal(key_path)

    output = f'''#include "blob.h"

extern "C" {{
extern const uint8_t pt2so_encrypted_weights_pack_start[];
extern const uint8_t pt2so_encrypted_weights_pack_end[];
extern const uint8_t pt2so_weights_key_start[];
extern const uint8_t pt2so_weights_key_end[];
}}

#if !defined(__GNUC__) && !defined(__clang__)
#error "The embedded blob uses GNU-style inline assembly .incbin and requires GCC or Clang."
#endif

__asm__(
    ".section .rodata.pt2so,\"a\",@progbits\n"
    ".balign 16\n"
    ".global pt2so_encrypted_weights_pack_start\n"
    ".hidden pt2so_encrypted_weights_pack_start\n"
    "pt2so_encrypted_weights_pack_start:\n"
    ".incbin \"{encrypted_incbin}\"\n"
    ".global pt2so_encrypted_weights_pack_end\n"
    ".hidden pt2so_encrypted_weights_pack_end\n"
    "pt2so_encrypted_weights_pack_end:\n"
    ".size pt2so_encrypted_weights_pack_start, pt2so_encrypted_weights_pack_end - pt2so_encrypted_weights_pack_start\n"
    ".balign 16\n"
    ".global pt2so_weights_key_start\n"
    ".hidden pt2so_weights_key_start\n"
    "pt2so_weights_key_start:\n"
    ".incbin \"{key_incbin}\"\n"
    ".global pt2so_weights_key_end\n"
    ".hidden pt2so_weights_key_end\n"
    "pt2so_weights_key_end:\n"
    ".size pt2so_weights_key_start, pt2so_weights_key_end - pt2so_weights_key_start\n"
    ".previous\n"
);

const uint8_t* embedded_encrypted_weights_pack_data() {{
    return pt2so_encrypted_weights_pack_start;
}}

size_t embedded_encrypted_weights_pack_size() {{
    return static_cast<size_t>(
        reinterpret_cast<uintptr_t>(pt2so_encrypted_weights_pack_end) -
        reinterpret_cast<uintptr_t>(pt2so_encrypted_weights_pack_start));
}}

const uint8_t* embedded_weights_key_data() {{
    return pt2so_weights_key_start;
}}

size_t embedded_weights_key_size() {{
    return static_cast<size_t>(
        reinterpret_cast<uintptr_t>(pt2so_weights_key_end) -
        reinterpret_cast<uintptr_t>(pt2so_weights_key_start));
}}
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate runtime/src/blob.cpp that embeds encrypted weights with .incbin."
    )
    parser.add_argument("--encrypted", type=Path, default=ARTIFACT_DIR / "weights.enc")
    parser.add_argument("--key", type=Path, default=ARTIFACT_DIR / "weights.key")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    generate_blob(args.encrypted, args.key, args.output)
    print(f"generated_incbin_blob={args.output}")


if __name__ == "__main__":
    main()
