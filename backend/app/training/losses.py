"""Modular multi-task losses.

Total loss (project spec §18)::

    L_total = λ_hate·L_hate + λ_target·L_target + λ_reason·L_reason
              + λ_contrastive·L_contrastive + λ_evidence·L_evidence

A component is only included when the batch actually contains ground-truth
labels for it (missing labels use ``-100`` / ``IGNORE_INDEX``) — no fake
supervision, no keyword-inferred targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class LossWeights:
    """Configurable weights for every loss component."""

    hate: float = 1.0
    target: float = 1.0
    reason: float = 1.0
    contrastive: float = 0.5
    evidence: float = 1.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LossWeights":
        data = data or {}
        return cls(
            hate=float(data.get("hate", 1.0)),
            target=float(data.get("target", 1.0)),
            reason=float(data.get("reason", 1.0)),
            contrastive=float(data.get("contrastive", 0.5)),
            evidence=float(data.get("evidence", 1.0)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "hate": self.hate,
            "target": self.target,
            "reason": self.reason,
            "contrastive": self.contrastive,
            "evidence": self.evidence,
        }


def _masked_cross_entropy(
    logits: torch.Tensor, labels: Optional[torch.Tensor], ignore_index: int = IGNORE_INDEX
) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    if not bool((labels != ignore_index).any()):
        return None
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def _masked_bce(
    logits: torch.Tensor, labels: Optional[torch.Tensor], ignore_index: int = IGNORE_INDEX
) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    mask = labels != ignore_index
    if not bool(mask.any()):
        return None
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())


def compute_task_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    *,
    hate_mode: str = "multiclass",
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Return the task losses that are actually supervised in this batch."""
    losses: Dict[str, torch.Tensor] = {}

    if "hate_logits" in outputs:
        labels = batch.get("hate_labels")
        if hate_mode == "binary":
            hate_loss = _masked_bce(
                outputs["hate_logits"].squeeze(-1), labels, ignore_index
            )
        else:
            hate_loss = _masked_cross_entropy(outputs["hate_logits"], labels, ignore_index)
        if hate_loss is not None:
            losses["hate"] = hate_loss

    if "target_logits" in outputs:
        target_loss = _masked_cross_entropy(
            outputs["target_logits"], batch.get("target_labels"), ignore_index
        )
        if target_loss is not None:
            losses["target"] = target_loss

    if "reason_logits" in outputs:
        reason_loss = _masked_cross_entropy(
            outputs["reason_logits"], batch.get("reason_labels"), ignore_index
        )
        if reason_loss is not None:
            losses["reason"] = reason_loss

    if "rationale_logits" in outputs:
        evidence_loss = _masked_bce(
            outputs["rationale_logits"], batch.get("rationale_labels"), ignore_index
        )
        if evidence_loss is not None:
            losses["evidence"] = evidence_loss

    return losses


def combine_losses(
    components: Dict[str, torch.Tensor], weights: LossWeights
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Weighted sum of the available components + numeric summary for logging."""
    total: Optional[torch.Tensor] = None
    summary: Dict[str, float] = {}
    for name, loss in components.items():
        weight = float(getattr(weights, name, 1.0))
        summary[f"{name}_loss"] = float(loss.detach().item())
        weighted = weight * loss
        total = weighted if total is None else total + weighted
    summary["total_loss"] = float(total.detach().item()) if total is not None else 0.0
    return total, summary
