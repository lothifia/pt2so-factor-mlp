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


def validate_inputs(encrypted_path: Path) -> None:
    encrypted = encrypted_path.read_bytes()
    if len(encrypted) < 8 or encrypted[:8] != MAGIC:
        raise ValueError(f"invalid encrypted model magic in {encrypted_path}")


def generate_blob(encrypted_path: Path, output_path: Path) -> None:
    validate_inputs(encrypted_path)

    encrypted_model_incbin = asm_string_literal(encrypted_path)

    output = f'''#include "blob.h"

extern "C" {{
extern const uint8_t pt2so_encrypted_model_start[];
extern const uint8_t pt2so_encrypted_model_end[];
}}

#if !defined(__GNUC__) && !defined(__clang__)
#error "The embedded blob uses GNU-style inline assembly .incbin and requires GCC or Clang."
#endif

__asm__(
    ".section .rodata.pt2so,\"a\",@progbits\n"
    ".balign 16\n"
    ".global pt2so_encrypted_model_start\n"
    ".hidden pt2so_encrypted_model_start\n"
    "pt2so_encrypted_model_start:\n"
    ".incbin \"{encrypted_model_incbin}\"\n"
    ".global pt2so_encrypted_model_end\n"
    ".hidden pt2so_encrypted_model_end\n"
    "pt2so_encrypted_model_end:\n"
    ".size pt2so_encrypted_model_start, pt2so_encrypted_model_end - pt2so_encrypted_model_start\n"
    ".previous\n"
);

const uint8_t* embedded_encrypted_model_data() {{
    return pt2so_encrypted_model_start;
}}

size_t embedded_encrypted_model_size() {{
    return static_cast<size_t>(
        reinterpret_cast<uintptr_t>(pt2so_encrypted_model_end) -
        reinterpret_cast<uintptr_t>(pt2so_encrypted_model_start));
}}
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate runtime/src/blob.cpp that embeds an encrypted TorchScript model with .incbin."
    )
    parser.add_argument("--encrypted", type=Path, default=ARTIFACT_DIR / "model.pt.enc")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    generate_blob(args.encrypted, args.output)
    print(f"generated_incbin_blob={args.output}")


if __name__ == "__main__":
    main()
