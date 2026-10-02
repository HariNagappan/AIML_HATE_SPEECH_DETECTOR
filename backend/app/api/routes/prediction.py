"""Prediction endpoint: ``POST /api/v1/predict``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.prediction import PredictRequest, PredictResponse
from app.services.inference import get_inference_service

router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify the comment (with optional context) and return structured evidence."""
    service = get_inference_service()
    try:
        result = service.predict(request.text, request.context)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(**result)
