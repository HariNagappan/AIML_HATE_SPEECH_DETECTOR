"""Contrastive loss integration for training (thin wrapper over InfoNCE).

The loss itself lives in :mod:`app.models.contrastive`; this module wires it
to a training batch produced by
:class:`app.datasets.collator.ContrastivePairCollator`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from app.models.contrastive import InfoNCELoss


def build_contrastive_loss(temperature: float = 0.07) -> InfoNCELoss:
    return InfoNCELoss(temperature=temperature)


def contrastive_loss_from_batch(
    model,
    batch: Dict[str, Any],
    loss_module: InfoNCELoss,
    negatives_per_anchor: int = 1,
) -> Optional[torch.Tensor]:
    """Compute the InfoNCE loss for a batch (requires a full model).

    Rows without a real context carry no contrastive signal and are excluded.
    Returns ``None`` when the batch has no usable rows.
    """
    if not all(
        key in batch
        for key in (
            "positive_context_input_ids",
            "negative_context_input_ids",
            "positive_context_attention_mask",
            "negative_context_attention_mask",
        )
    ):
        return None

    z_anchor, z_positive, z_negative = model.contrastive_embeddings(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        positive_context_input_ids=batch["positive_context_input_ids"],
        positive_context_attention_mask=batch["positive_context_attention_mask"],
        negative_context_input_ids=batch["negative_context_input_ids"],
        negative_context_attention_mask=batch["negative_context_attention_mask"],
        negatives_per_anchor=negatives_per_anchor,
        context_present=batch.get("context_present"),
    )

    active = batch.get("contrastive_active")
    if active is not None:
        active = active.to(z_anchor.device)
        if int(active.sum()) < 1:
            return None
        z_anchor = z_anchor[active]
        z_positive = z_positive[active]
        z_negative = z_negative[active]

    return loss_module(z_anchor, z_positive, z_negative)
