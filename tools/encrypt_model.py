from __future__ import annotations

import argparse
import secrets
import struct
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "factor_mlp"
DEFAULT_KEY_SOURCE = ROOT / "runtime" / "src" / "aes_key_obfuscated.cpp"

MAGIC_V2 = b"PT2SO_E2"
AAD = b"PT2SO_TORCHSCRIPT_V1"

CANONICAL_ALGORITHM = "gcm-aes-128"
ALGORITHM_ID = 7
KEY_SIZE = 16
NONCE_SIZE = 12
DEFAULT_SHARE_COUNT = 4
MASK64 = 0xFFFFFFFFFFFFFFFF


def rotl64(value: int, shift: int) -> int:
    shift &= 63
    if shift == 0:
        return value & MASK64
    return ((value << shift) | (value >> (64 - shift))) & MASK64


def cpp_u64(value: int) -> str:
    return f"0x{value & MASK64:016X}ULL"


def cpp_u8(value: int) -> str:
    return f"0x{value & 0xFF:02X}"


def key_words(key: bytes) -> list[int]:
    if len(key) != KEY_SIZE:
        raise ValueError(f"AES-128 key must be {KEY_SIZE} bytes")
    return [
        int.from_bytes(key[0:8], "little"),
        int.from_bytes(key[8:16], "little"),
    ]


def split_key_into_shares(key: bytes, share_count: int) -> list[bytes]:
    if share_count < 2:
        raise ValueError("--key-share-count must be at least 2")

    shares = [secrets.token_bytes(KEY_SIZE) for _ in range(share_count - 1)]
    tail = bytearray(key)
    for share in shares:
        for i, byte in enumerate(share):
            tail[i] ^= byte
    shares.append(bytes(tail))
    return shares


def encode_share_words(share: bytes) -> dict[str, list[int]]:
    words = key_words(share)
    order = [0, 1]
    if secrets.randbelow(2) == 1:
        order.reverse()

    encoded: list[int] = []
    mask_a: list[int] = []
    mask_b: list[int] = []
    add_c: list[int] = []
    rotations: list[int] = []
    for original_index in order:
        rotation = 1 + secrets.randbelow(63)
        a = int.from_bytes(secrets.token_bytes(8), "little")
        b = int.from_bytes(secrets.token_bytes(8), "little")
        c = int.from_bytes(secrets.token_bytes(8), "little")
        x = rotl64(words[original_index], rotation)
        x = (x ^ a) & MASK64
        x = (x + c) & MASK64
        x = (x ^ b) & MASK64
        encoded.append(x)
        mask_a.append(a)
        mask_b.append(b)
        add_c.append(c)
        rotations.append(rotation)

    return {
        "encoded": encoded,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "add_c": add_c,
        "rotations": rotations,
        "order": order,
    }


def share_stage_source(stage_index: int, share: bytes) -> str:
    encoded = encode_share_words(share)
    suffix = f"{stage_index:02d}"
    return f'''
void pt2so_key_stage_{suffix}(uint64_t words[2]) {{
    constexpr uint64_t encoded[2] = {{{", ".join(cpp_u64(v) for v in encoded["encoded"])}}};
    constexpr uint64_t mask_a[2] = {{{", ".join(cpp_u64(v) for v in encoded["mask_a"])}}};
    constexpr uint64_t mask_b[2] = {{{", ".join(cpp_u64(v) for v in encoded["mask_b"])}}};
    constexpr uint64_t add_c[2] = {{{", ".join(cpp_u64(v) for v in encoded["add_c"])}}};
    constexpr unsigned rotation[2] = {{{", ".join(str(v) + "U" for v in encoded["rotations"])}}};
    constexpr uint8_t order[2] = {{{", ".join(cpp_u8(v) for v in encoded["order"])}}};

    uint64_t local[2] = {{}};
    for (size_t slot = 0; slot < 2; ++slot) {{
        uint64_t x = encoded[slot];
        x ^= mask_b[slot];
        x -= add_c[slot];
        x ^= mask_a[slot];
        x = rotr64(x, rotation[slot]);
        local[order[slot]] = x;
    }}
    words[0] ^= local[0];
    words[1] ^= local[1];
    cleanse_u64(local, 2);
}}
'''


def write_obfuscated_key_source(key: bytes, output_path: Path, share_count: int) -> None:
    shares = split_key_into_shares(key, share_count)
    order = list(range(share_count))
    for i in range(len(order) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        order[i], order[j] = order[j], order[i]

    stages = "\n".join(share_stage_source(i, share) for i, share in enumerate(shares))
    calls = "\n".join(f"    pt2so_key_stage_{i:02d}(words);" for i in order)

    source = f'''#include "embedded_key.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {{

uint64_t rotr64(uint64_t value, unsigned shift) {{
    shift &= 63U;
    if (shift == 0) {{
        return value;
    }}
    return (value >> shift) | (value << (64U - shift));
}}

void cleanse_u64(uint64_t* words, size_t count) {{
    volatile uint64_t* p = words;
    while (count-- != 0) {{
        *p++ = 0;
    }}
}}

{stages}

}}  // namespace

void pt2so_materialize_aes_128_key(uint8_t* out, size_t out_size) {{
    if (out == nullptr || out_size < 16) {{
        throw std::runtime_error("AES-128 key output buffer is too small");
    }}

    uint64_t words[2] = {{}};
{calls}

    for (size_t word_index = 0; word_index < 2; ++word_index) {{
        uint64_t word = words[word_index];
        for (size_t byte_index = 0; byte_index < 8; ++byte_index) {{
            out[word_index * 8 + byte_index] = static_cast<uint8_t>((word >> (8 * byte_index)) & 0xffU);
        }}
    }}
    cleanse_u64(words, 2);
}}
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")


def encrypt_model(
    input_path: Path,
    output_path: Path,
    key_source_path: Path,
    key_share_count: int,
) -> None:
    plaintext = input_path.read_bytes()
    if not plaintext:
        raise ValueError(f"input model is empty: {input_path}")

    key = secrets.token_bytes(KEY_SIZE)
    nonce = secrets.token_bytes(NONCE_SIZE)
    ciphertext_and_tag = AESGCM(key).encrypt(nonce, plaintext, AAD)
    write_obfuscated_key_source(key, key_source_path, key_share_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(MAGIC_V2)
        f.write(struct.pack("<I", ALGORITHM_ID))
        f.write(struct.pack("<I", len(nonce)))
        f.write(nonce)
        f.write(struct.pack("<Q", len(ciphertext_and_tag)))
        f.write(ciphertext_and_tag)

    print(f"algorithm={CANONICAL_ALGORITHM}")
    print(f"encrypted_model={output_path}")
    print(f"obfuscated_key_source={key_source_path}")
    print(f"key_share_count={key_share_count}")
    print("warning=the AES key is obfuscated inside the shared object, not cryptographically hidden from a determined reverse engineer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encrypt a TorchScript .pt file as AES-128-GCM and generate an obfuscated embedded key source."
    )
    parser.add_argument("--input", type=Path, default=ARTIFACT_DIR / "factor_mlp.pt")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "weights.enc")
    parser.add_argument("--key-source", type=Path, default=DEFAULT_KEY_SOURCE)
    parser.add_argument("--key-share-count", type=int, default=DEFAULT_SHARE_COUNT)
    parser.add_argument(
        "--algorithm",
        choices=[CANONICAL_ALGORITHM, "aes-128-gcm"],
        default=CANONICAL_ALGORITHM,
        help="AES-128-GCM is the only supported encryption mode.",
    )
    args = parser.parse_args()

    encrypt_model(
        input_path=args.input,
        output_path=args.output,
        key_source_path=args.key_source,
        key_share_count=args.key_share_count,
    )


if __name__ == "__main__":
    main()
