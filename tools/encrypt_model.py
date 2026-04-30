from __future__ import annotations

import argparse
import secrets
import struct
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "factor_mlp"

MAGIC = b"PT2SO_E1"
AAD = b"PT2SO_TORCHSCRIPT_V1"


def encrypt_model(input_path: Path, output_path: Path, key_output_path: Path) -> None:
    plaintext = input_path.read_bytes()
    key = secrets.token_bytes(32)
    nonce = secrets.token_bytes(12)
    ciphertext_and_tag = AESGCM(key).encrypt(nonce, plaintext, AAD)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(nonce)))
        f.write(nonce)
        f.write(struct.pack("<Q", len(ciphertext_and_tag)))
        f.write(ciphertext_and_tag)

    key_output_path.write_bytes(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt a TorchScript .pt file for embedding into libmodel.so.")
    parser.add_argument("--input", type=Path, default=ARTIFACT_DIR / "factor_mlp.pt")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "model.pt.enc")
    parser.add_argument("--key-output", type=Path, default=ARTIFACT_DIR / "model.key")
    args = parser.parse_args()

    encrypt_model(args.input, args.output, args.key_output)
    print(f"encrypted_model={args.output}")
    print(f"key={args.key_output}")


if __name__ == "__main__":
    main()
