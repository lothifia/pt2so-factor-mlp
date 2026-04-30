from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "factor_mlp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the upstream TorchScript .pt model in Python.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ARTIFACT_DIR / "factor_mlp.pt",
        help="TorchScript .pt file produced by torch.jit.trace.",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=ARTIFACT_DIR / "sample_input.npy",
        help="Shared test sample consumed by both Python and C++ runners.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "python_pt_output.npy",
        help="Output .npy file written by the Python TorchScript reference path.",
    )
    args = parser.parse_args()

    model = torch.jit.load(str(args.model), map_location="cpu")
    model.eval()

    sample_input = np.load(args.sample).astype(np.float32, copy=False)
    sample_input = np.ascontiguousarray(sample_input)
    x = torch.from_numpy(sample_input)

    with torch.inference_mode():
        y = model(x).detach().cpu().contiguous().numpy().astype(np.float32, copy=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, y)
    print(f"python_output={args.output}")
    print(f"shape={tuple(y.shape)}")


if __name__ == "__main__":
    main()
