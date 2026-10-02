"""Pydantic schemas for the prediction endpoint."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request body for ``POST /api/v1/predict``."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Current comment to analyse",
    )
    context: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Previous comment / conversational context (optional)",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class HeadPrediction(BaseModel):
    """One head's prediction (label + confidence + optional probabilities)."""

    label: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


class EvidenceItem(BaseModel):
    """One evidence span grounded in the input tokens.

    ``start``/``end`` are character offsets into the original comment
    (end exclusive) and are the primary mechanism for frontend highlighting.
    ``token_start``/``token_end`` are inclusive token indices in the
    tokenized sequence.
    """

    text: str
    score: float
    type: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None


class ReasonExplanationGrounding(BaseModel):
    """The structured fields the reason explanation is grounded in."""

    reason: str
    target: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)


class ReasonExplanation(BaseModel):
    """Deterministic explanation connecting the predicted reason to evidence."""

    summary: str
    details: str
    grounded_in: ReasonExplanationGrounding


class PredictResponse(BaseModel):
    """Structured explanation response for ``/api/v1/predict``.

    Untrained heads are reported explicitly: ``<head>`` is ``null`` and
    ``<head>_available`` is ``false`` — nothing is fabricated.
    """

    prediction: Optional[HeadPrediction] = None
    prediction_available: bool = False
    target: Optional[HeadPrediction] = None
    target_available: bool = False
    reason: Optional[HeadPrediction] = None
    reason_available: bool = False
    evidence: List[EvidenceItem] = Field(default_factory=list)
    reason_explanation: Optional[ReasonExplanation] = None
    context_used: bool = False
    evidence_available: bool = False
