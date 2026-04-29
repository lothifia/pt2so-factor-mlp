from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.factor_mlp import FactorMLP


def write_model_info(output_dir: Path) -> None:
    model_info = {
        "arch": "factor_mlp_v1",
        "state_dict": {
            "key": "model",
            "strip_prefix": [],
        },
        "model_init": {
            "num_features": 128,
            "hidden1": 256,
            "hidden2": 64,
            "output_dim": 1,
            "dropout": 0.1,
        },
        "input": {
            "count": 1,
            "dtype": "float32",
            "shape": [-1, 128],
            "layout": "BF",
        },
        "output": {
            "count": 1,
            "dtype": "float32",
            "shape": [-1, 1],
        },
        "runtime": {
            "backend": "libtorch",
            "device": "cpu",
            "precision": "fp32",
        },
    }

    with (output_dir / "model_info.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(model_info, f, sort_keys=False)


def create_factor_mlp(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = FactorMLP()
    model.eval()

    sample_input = torch.randn(32, 128, dtype=torch.float32)
    with torch.no_grad():
        expected_output = model(sample_input).contiguous()

    np.save(output_dir / "sample_input.npy", sample_input.numpy())
    np.save(output_dir / "expected_output.npy", expected_output.numpy())
    torch.save(
        {
            "model": model.state_dict(),
            "arch": "factor_mlp_v1",
        },
        output_dir / "factor_mlp.pt",
    )
    write_model_info(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create FactorMLP checkpoint and sample tensors.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "factor_mlp",
        help="Directory for factor_mlp.pt, model_info.yaml, sample_input.npy, and expected_output.npy.",
    )
    args = parser.parse_args()
    create_factor_mlp(args.output_dir)


if __name__ == "__main__":
    main()
