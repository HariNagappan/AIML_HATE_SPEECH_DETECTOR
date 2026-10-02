"""Supervised token-level rationale component (optional).

A learned token classifier (BCE over the shared encoder's contextual token
representations), trained when token-level rationale annotations are
available — HateXplain provides them.

This is deliberately distinct from the **post-hoc attribution** used at
inference (Integrated Gradients — ``app/reasoning/attribution.py``):

* supervised rationale **prediction** — trained against gold annotations
* attribution-based **explanation** — computed from gradients after the fact

They are different techniques and the project compares them experimentally.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EvidenceHead(nn.Module):
    """Token-level rationale classifier over contextual token states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """``[B, T, H]`` contextual token states → ``[B, T]`` rationale logits."""
        return self.classifier(token_embeddings).squeeze(-1)

    @torch.no_grad()
    def token_probabilities(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities per token (``[B, T]``)."""
        return torch.sigmoid(self.classifier(token_embeddings).squeeze(-1))
