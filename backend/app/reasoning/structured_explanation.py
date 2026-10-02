"""Deterministic builder for the project's structured reasoning output.

Combines prediction + confidence + target + reason + evidence + context_used
into one object. Everything is assembled from model outputs and attribution
results — no generated text. Untrained/unavailable heads are reported
explicitly (``*_available: false``) instead of being fabricated.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from app.reasoning.evidence_extractor import EvidenceSpan, evidence_to_dict


def build_explanation(
    *,
    hate: Optional[Dict[str, Any]] = None,
    target: Optional[Dict[str, Any]] = None,
    reason: Optional[Dict[str, Any]] = None,
    evidence: Iterable[Any] = (),
    context_used: bool = False,
    reason_explanation: Optional[Dict[str, Any]] = None,
    evidence_available: bool = False,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the unified structured explanation object.

    ``hate`` / ``target`` / ``reason`` are head prediction dicts
    (``{"label", "confidence", "probabilities"}``) or ``None`` when the head
    is unavailable/untrained. ``evidence`` items may be :class:`EvidenceSpan`
    objects or already-dict shaped.
    """
    evidence_items: List[Dict[str, object]] = []
    for item in evidence or []:
        if isinstance(item, EvidenceSpan):
            evidence_items.append(evidence_to_dict(item))
        elif isinstance(item, dict):
            entry: Dict[str, object] = {
                "text": str(item.get("text", "")),
                "score": float(item.get("score", 0.0)),
            }
            if item.get("type") is not None:
                entry["type"] = item["type"]
            evidence_items.append(entry)

    return {
        "prediction": hate,
        "prediction_available": hate is not None,
        "target": target,
        "target_available": target is not None,
        "reason": reason,
        "reason_available": reason is not None,
        "evidence": evidence_items,
        "reason_explanation": reason_explanation,
        "reasoning": reasoning,
        "context_used": bool(context_used),
        "evidence_available": bool(evidence_available),
    }
