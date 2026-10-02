"""Pydantic schemas for the explanation (attribution) endpoint."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.api.schemas.prediction import EvidenceItem, ReasonExplanation


class ExplainRequest(BaseModel):
    """Request body for ``POST /api/v1/explain``."""

    text: str = Field(..., min_length=1, max_length=5000)
    context: Optional[str] = Field(default=None, max_length=5000)
    target: Literal["hate", "target", "reason"] = Field(
        default="hate",
        description="Which prediction to explain",
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=50, description="Max evidence spans (optional)"
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class TokenScore(BaseModel):
    """Per-token attribution score (for debugging / visualization)."""

    index: int
    token: str
    score: float


class ExplainResponse(BaseModel):
    """Attribution details for one prediction."""

    available: bool
    message: Optional[str] = None
    target: str
    method: str
    steps: int
    predicted_label: Optional[str] = None
    predicted_confidence: Optional[float] = None
    explained_label: Optional[str] = None
    context_used: bool = False
    tokens: List[TokenScore] = Field(default_factory=list)
    spans: List[EvidenceItem] = Field(default_factory=list)
    reason_explanation: Optional[ReasonExplanation] = None
