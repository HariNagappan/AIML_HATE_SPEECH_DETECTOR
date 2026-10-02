"""Contrastive learning components (InfoNCE) — **training only**.

The contrastive objective teaches the model that the interpretation of a
comment depends on its context: the context-aware representation formed with
the *correct* preceding comment should be closer to the comment-only anchor
than the representation formed with a *contrasting* context.

IMPORTANT: contrastive learning is not performed at inference time — inference
only produces the context-aware representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveProjection(nn.Module):
    """Projects the final representation into the contrastive space (L2-normalised)."""

    def __init__(self, input_dim: int, projection_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(features), dim=-1)


class InfoNCELoss(nn.Module):
    r"""InfoNCE-style contrastive loss with cosine similarity.

    .. math::

        L = -\log \frac{\exp(\mathrm{sim}(a, p)/\tau)}
                          {\exp(\mathrm{sim}(a, p)/\tau)
                           + \sum_i \exp(\mathrm{sim}(a, n_i)/\tau)}

    ``a`` is the anchor, ``p`` the positive candidate, ``n_i`` the contrasting
    candidates; all vectors are L2-normalised (cosine similarity). The
    temperature :math:`\tau` is configurable.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        anchor:    [B, D]        comment-only representation
        positive:  [B, D]        representation with the correct context
        negatives: [B, N, D]     representations with contrasting contexts
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        positive_similarity = (
            torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature
        )
        negative_similarities = (
            torch.einsum("bd,bnd->bn", anchor, negatives) / self.temperature
        )
        logits = torch.cat([positive_similarity, negative_similarities], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
