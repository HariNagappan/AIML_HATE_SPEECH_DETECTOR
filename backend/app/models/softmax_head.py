"""Generic softmax classification head (shared base for target/reason heads)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn


class SoftmaxClassificationHead(nn.Module):
    """Linear layer over the final representation with softmax decoding."""

    def __init__(self, input_dim: int, labels: Sequence[str]):
        super().__init__()
        self.labels = [str(label) for label in labels]
        if len(self.labels) < 2:
            raise ValueError("A softmax head needs at least 2 labels")
        self.classifier = nn.Linear(input_dim, len(self.labels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> List[Dict[str, Any]]:
        probabilities = torch.softmax(self.classifier(features), dim=-1)
        results: List[Dict[str, Any]] = []
        for row in probabilities.tolist():
            best = max(range(len(row)), key=lambda index: row[index])
            results.append(
                {
                    "label": self.labels[best],
                    "confidence": float(row[best]),
                    "probabilities": {
                        label: float(p) for label, p in zip(self.labels, row)
                    },
                }
            )
        return results
