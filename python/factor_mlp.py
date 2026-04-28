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
    ):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
