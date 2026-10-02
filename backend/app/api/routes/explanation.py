"""Explanation endpoint: ``POST /api/v1/explain``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.explanation import ExplainRequest, ExplainResponse
from app.services.inference import get_inference_service

router = APIRouter(prefix="/api/v1", tags=["explanation"])


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    """Attribution-based explanation (token scores + evidence spans) for a head."""
    service = get_inference_service()
    try:
        result = service.explain(
            request.text,
            context=request.context,
            target=request.target,
            top_k=request.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ExplainResponse(**result)
