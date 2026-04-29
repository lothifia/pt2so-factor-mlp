from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Iterable

import torch
import yaml

MAGIC = b"PT2SO_W1"

DTYPE_TO_ID = {
    torch.float32: 1,
    torch.float16: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.uint8: 6,
    torch.bool: 7,
}


def normalize_key(name: str, strip_prefixes: Iterable[str]) -> str:
    for prefix in strip_prefixes:
        if prefix and name.startswith(prefix):
            return name[len(prefix) :]
    return name


def extract_weights(checkpoint_path: Path, model_info_path: Path, output_path: Path) -> None:
    with model_info_path.open("r", encoding="utf-8") as f:
        model_info = yaml.safe_load(f)

    state_cfg = model_info.get("state_dict", {})
    state_key = state_cfg.get("key", "model")
    strip_prefixes = state_cfg.get("strip_prefix", []) or []

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if state_key not in checkpoint:
        raise KeyError(f"checkpoint does not contain state_dict key {state_key!r}")

    state_dict = checkpoint[state_key]
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint[{state_key!r}] is not a state_dict-like mapping")

    tensors = []
    for raw_name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        name = normalize_key(raw_name, strip_prefixes)
        if tensor.dtype not in DTYPE_TO_ID:
            raise TypeError(f"unsupported tensor dtype for {raw_name}: {tensor.dtype}")
        cpu_tensor = tensor.detach().cpu().contiguous()
        tensors.append((name, cpu_tensor))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(tensors)))

        for name, tensor in tensors:
            name_bytes = name.encode("utf-8")
            raw = tensor.numpy().tobytes(order="C")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", DTYPE_TO_ID[tensor.dtype]))
            f.write(struct.pack("<I", tensor.ndim))
            for dim in tensor.shape:
                f.write(struct.pack("<q", int(dim)))
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)


def main() -> None:
    default_dir = Path(__file__).resolve().parents[1] / "artifacts" / "factor_mlp"
    parser = argparse.ArgumentParser(description="Extract a custom binary weight pack from factor_mlp.pt.")
    parser.add_argument("--checkpoint", type=Path, default=default_dir / "factor_mlp.pt")
    parser.add_argument("--model-info", type=Path, default=default_dir / "model_info.yaml")
    parser.add_argument("--output", type=Path, default=default_dir / "weights.bin")
    args = parser.parse_args()

    extract_weights(args.checkpoint, args.model_info, args.output)


if __name__ == "__main__":
    main()
