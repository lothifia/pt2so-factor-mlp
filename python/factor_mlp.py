from __future__ import annotations

import torch
from torch import nn


class FactorMLP(nn.Module):
    def __init__(
        self,
        num_features: int = 128,
        hidden1: int = 256,
        hidden2: int = 64,
        output_dim: int = 1,
        dropout: float = 0.1,
        depth: int | None = None,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if depth is None and hidden_dim is None:
            hidden_sizes = [hidden1, hidden2]
        else:
            depth = 2 if depth is None else depth
            if depth < 1:
                raise ValueError("depth must be >= 1")
            width = hidden1 if hidden_dim is None else hidden_dim
            hidden_sizes = [width] * depth

        layers: list[nn.Module] = []
        in_dim = num_features
        for hidden_size in hidden_sizes:
            if hidden_size < 1:
                raise ValueError("hidden layer size must be >= 1")
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
