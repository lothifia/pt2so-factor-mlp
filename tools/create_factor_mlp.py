from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.factor_mlp import FactorMLP


def create_factor_mlp(output_dir: Path, depth: int | None, hidden_dim: int | None, batch_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = FactorMLP(depth=depth, hidden_dim=hidden_dim)
    model.eval()

    sample_input = torch.randn(batch_size, 128, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_input)
        traced = torch.jit.freeze(traced.eval())
        expected_output = traced(sample_input).contiguous()

    np.save(output_dir / "sample_input.npy", sample_input.numpy())
    np.save(output_dir / "expected_output.npy", expected_output.numpy())
    traced.save(str(output_dir / "factor_mlp.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create traced TorchScript FactorMLP model and sample tensors.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "factor_mlp",
        help="Directory for TorchScript factor_mlp.pt, sample_input.npy, and expected_output.npy.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Number of hidden Linear/LayerNorm/ReLU/Dropout blocks. Omit to keep the original 256 -> 64 model.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Width for each hidden layer when --depth is used. Defaults to 256.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of random samples to generate for sample_input.npy.",
    )
    args = parser.parse_args()
    create_factor_mlp(args.output_dir, args.depth, args.hidden_dim, args.batch_size)


if __name__ == "__main__":
    main()
