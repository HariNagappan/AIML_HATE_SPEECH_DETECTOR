"""Model info endpoint: ``GET /api/v1/model/info``."""

from __future__ import annotations

from fastapi import APIRouter

from app.services.inference import get_inference_service

router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model/info")
def model_info() -> dict:
    """Model name, device, hidden size, label counts, trained status, version."""
    return get_inference_service().model_info()
