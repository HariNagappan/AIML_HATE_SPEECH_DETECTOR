"""FastAPI application factory.

Run with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import explanation, health, model, prediction
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the tokenizer + model once at startup; requests reuse them."""
    settings = get_settings()
    setup_logging(settings.log_level)

    from app.services.inference import warmup_inference_service

    error = warmup_inference_service()
    if error:
        logger.warning(
            "Model unavailable at startup (%s). /health stays up; "
            "/predict and /explain will return 503 until fixed.",
            error,
        )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Context-Aware Hate Speech Detection API",
        description=(
            "Structured evidence-based reasoning backend: shared BERT encoder, "
            "context interaction, multi-task heads (hate / target / reason) and "
            "attribution-based evidence (Integrated Gradients)."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health.router)
    application.include_router(prediction.router)
    application.include_router(explanation.router)
    application.include_router(model.router)
    return application


app = create_app()
