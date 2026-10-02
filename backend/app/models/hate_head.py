"""Hate / offensive / normal classification head.

Two modes:

* ``multiclass`` (default) – softmax over the configured label set
  (HateXplain: ``hate`` / ``offensive`` / ``normal``). Multiclass labels are
  never silently collapsed into binary.
* ``binary`` – single logit + sigmoid with a configurable threshold; this
  must be configured explicitly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn


class HateHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        labels: Sequence[str],
        mode: str = "multiclass",
        threshold: float = 0.5,
    ):
        super().__init__()
        if mode not in {"multiclass", "binary"}:
            raise ValueError(f"Unknown hate head mode: {mode!r}")
        self.labels = [str(label) for label in labels]
        if mode == "binary" and len(self.labels) != 2:
            raise ValueError(
                "binary mode requires exactly 2 labels, e.g. ['non_hate', 'hate']"
            )
        if mode == "multiclass" and len(self.labels) < 2:
            raise ValueError("multiclass mode requires at least 2 labels")
        self.mode = mode
        self.threshold = float(threshold)
        out_dim = len(self.labels) if mode == "multiclass" else 1
        self.classifier = nn.Linear(input_dim, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits (shape ``[B, K]`` for multiclass, ``[B, 1]`` for binary)."""
        return self.classifier(features)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> List[Dict[str, Any]]:
        """Decode logits into ``label`` / ``confidence`` / ``probabilities``."""
        logits = self.classifier(features)
        if self.mode == "binary":
            probabilities = torch.sigmoid(logits.squeeze(-1))
            results: List[Dict[str, Any]] = []
            for probability in probabilities.tolist():
                positive = probability >= self.threshold
                results.append(
                    {
                        "label": self.labels[1] if positive else self.labels[0],
                        "confidence": float(
                            probability if positive else 1.0 - probability
                        ),
                        "probabilities": {
                            self.labels[0]: float(1.0 - probability),
                            self.labels[1]: float(probability),
                        },
                    }
                )
            return results

        probabilities = torch.softmax(logits, dim=-1)
        results = []
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
