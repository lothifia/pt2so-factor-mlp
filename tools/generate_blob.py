from __future__ import annotations

import argparse
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "factor_mlp"
DEFAULT_OUTPUT = ROOT / "runtime" / "src" / "blob.cpp"

MAGIC = b"PT2SO_E1"


def parse_encrypted_pack(path: Path) -> tuple[bytes, bytes]:
    data = path.read_bytes()
    offset = 0

    def take(n: int, what: str) -> bytes:
        nonlocal offset
        if n < 0 or offset + n > len(data):
            raise ValueError(f"truncated encrypted weights while reading {what}")
        out = data[offset : offset + n]
        offset += n
        return out

    magic = take(8, "magic")
    if magic != MAGIC:
        raise ValueError(f"invalid encrypted weights magic: {magic!r}")

    nonce_len = struct.unpack("<I", take(4, "nonce_len"))[0]
    nonce = take(nonce_len, "nonce")
    ciphertext_len = struct.unpack("<Q", take(8, "ciphertext_len"))[0]
    if ciphertext_len > len(data) - offset:
        raise ValueError("ciphertext_len exceeds remaining encrypted weights file size")
    ciphertext_and_tag = take(ciphertext_len, "ciphertext_and_tag")

    if offset != len(data):
        raise ValueError("encrypted weights file has trailing bytes")
    if len(nonce) != 12:
        raise ValueError("AES-GCM nonce must be 12 bytes")
    if len(ciphertext_and_tag) <= 16:
        raise ValueError("ciphertext must include at least one byte plus a 16-byte GCM tag")

    return nonce, ciphertext_and_tag


def format_cpp_array(name: str, data: bytes) -> str:
    if not data:
        return f"alignas(16) const uint8_t {name}[] = {{}};\n"

    lines = [f"alignas(16) const uint8_t {name}[] = {{"]
    for start in range(0, len(data), 12):
        chunk = data[start : start + 12]
        values = ", ".join(f"0x{byte:02x}" for byte in chunk)
        lines.append(f"    {values},")
    lines.append("};")
    return "\n".join(lines) + "\n"


def generate_blob(encrypted_path: Path, key_path: Path, output_path: Path) -> None:
    nonce, ciphertext_and_tag = parse_encrypted_pack(encrypted_path)
    key = key_path.read_bytes()
    if len(key) != 32:
        raise ValueError("AES-256-GCM key file must contain exactly 32 bytes")

    output = [
        '#include "blob.h"',
        "",
        "namespace {",
        "",
        format_cpp_array("kEncryptedWeights", ciphertext_and_tag).rstrip(),
        "",
        format_cpp_array("kWeightsKey", key).rstrip(),
        "",
        format_cpp_array("kWeightsNonce", nonce).rstrip(),
        "",
        "}  // namespace",
        "",
        "const uint8_t* embedded_encrypted_weights_data() {",
        "    return kEncryptedWeights;",
        "}",
        "",
        "size_t embedded_encrypted_weights_size() {",
        "    return sizeof(kEncryptedWeights);",
        "}",
        "",
        "const uint8_t* embedded_weights_key_data() {",
        "    return kWeightsKey;",
        "}",
        "",
        "size_t embedded_weights_key_size() {",
        "    return sizeof(kWeightsKey);",
        "}",
        "",
        "const uint8_t* embedded_weights_nonce_data() {",
        "    return kWeightsNonce;",
        "}",
        "",
        "size_t embedded_weights_nonce_size() {",
        "    return sizeof(kWeightsNonce);",
        "}",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate runtime/src/blob.cpp from encrypted weights.")
    parser.add_argument("--encrypted", type=Path, default=ARTIFACT_DIR / "weights.enc")
    parser.add_argument("--key", type=Path, default=ARTIFACT_DIR / "weights.key")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    generate_blob(args.encrypted, args.key, args.output)
    print(f"generated_blob={args.output}")


if __name__ == "__main__":
    main()
