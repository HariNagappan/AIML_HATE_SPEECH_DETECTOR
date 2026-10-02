"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.services.inference import get_inference_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness probe; also reports whether the model is loaded."""
    service = get_inference_service()
    return {"status": "ok", "model_loaded": service.loaded}
