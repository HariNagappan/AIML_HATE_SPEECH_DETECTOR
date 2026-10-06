"""API tests: /health, /api/v1/predict, /api/v1/explain, /api/v1/model/info.

Runs with an untrained (random-backbone) model so no downloads/training are
needed; asserts the explicit availability-status behaviour of the spec.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    saved = {
        key: os.environ.get(key)
        for key in (
            "INIT_BACKBONE",
            "MODEL_CHECKPOINT",
            "DEVICE",
            "AUTO_LOAD_CHECKPOINT",
        )
    }
    os.environ["INIT_BACKBONE"] = "random"
    os.environ["DEVICE"] = "cpu"
    os.environ["AUTO_LOAD_CHECKPOINT"] = "false"
    os.environ.pop("MODEL_CHECKPOINT", None)

    from app.core.config import get_settings
    from app.services.inference import reset_inference_service

    get_settings.cache_clear()
    reset_inference_service()

    from app.main import create_app

    application = create_app()
    with TestClient(application) as test_client:
        yield test_client

    get_settings.cache_clear()
    reset_inference_service()
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload


def test_model_info(client):
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["model_name"], str)
    assert payload["device"] in {"cpu", "cuda", "cuda:0"}
    assert "num_labels" in payload and "hate" in payload["num_labels"]
    assert payload["trained"] is False  # untrained by construction
    assert payload["loaded"] is True
    assert "version" in payload
    assert payload["metrics"] is None  # no evaluation file for the dev model


def test_predict_untrained_returns_explicit_status(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "text": "They should all be kicked out.",
            "context": "Those immigrants are ruining everything.",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] is None
    assert payload["prediction_available"] is False
    assert payload["reason"] is None
    assert payload["reason_available"] is False
    assert payload["evidence"] == []
    assert payload["context_used"] is True
    assert payload["reason_explanation"] is None
    assert payload["evidence_available"] is False


def test_predict_without_context(client):
    response = client.post("/api/v1/predict", json={"text": "Just a comment."})
    assert response.status_code == 200
    assert response.json()["context_used"] is False


def test_predict_request_validation(client):
    assert client.post("/api/v1/predict", json={"text": ""}).status_code == 422
    assert client.post("/api/v1/predict", json={"text": "   "}).status_code == 422
    assert client.post("/api/v1/predict", json={}).status_code == 422


def test_explain_untrained_reports_unavailable(client):
    response = client.post("/api/v1/explain", json={"text": "Some comment."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["tokens"] == []
    assert payload["spans"] == []
    assert payload["target"] == "hate"


def test_explain_rejects_invalid_target(client):
    response = client.post(
        "/api/v1/explain", json={"text": "Some comment.", "target": "bogus"}
    )
    assert response.status_code == 422


def test_predict_untrained_reasoning_reports_no_classification(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "previous_comment": "I saw a group of immigrants protesting downtown.",
            "current_comment": "They should all be kicked out.",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    reasoning = payload["reasoning"]
    assert reasoning is not None
    assert reasoning["context_used"] is True
    assert reasoning["context_available"] is True
    assert "no reasoning could be derived" in reasoning["summary"].lower()
    # the reference analysis still runs and is exposed transparently
    assert reasoning["links"]
    link = reasoning["links"][0]
    assert link["from_text"] == "They"
    assert link["to_text"] == "group of immigrants"


def test_predict_accepts_modern_and_legacy_field_names(client):
    modern = client.post(
        "/api/v1/predict",
        json={
            "previous_comment": "Those immigrants were protesting.",
            "current_comment": "They should leave.",
        },
    )
    legacy = client.post(
        "/api/v1/predict",
        json={
            "context": "Those immigrants were protesting.",
            "text": "They should leave.",
        },
    )
    assert modern.status_code == 200 and legacy.status_code == 200
    assert modern.json()["context_used"] is True
    assert legacy.json()["context_used"] is True


def test_predict_without_context_reasoning_says_so(client):
    payload = client.post(
        "/api/v1/predict", json={"current_comment": "You are disgusting."}
    ).json()
    reasoning = payload["reasoning"]
    assert payload["context_used"] is False
    assert reasoning["context_available"] is False
    assert "no previous comment was provided" in reasoning["summary"].lower()


def test_predict_validation_with_modern_names(client):
    assert (
        client.post("/api/v1/predict", json={"current_comment": ""}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/predict", json={"previous_comment": "ctx only"}
        ).status_code
        == 422
    )


def test_predict_reasoning_failure_degrades_gracefully(client, monkeypatch):
    import app.services.inference as inference_module

    def _boom(**kwargs):
        raise RuntimeError("test reasoning crash")

    monkeypatch.setattr(inference_module, "build_context_reasoning", _boom)
    response = client.post(
        "/api/v1/predict",
        json={"current_comment": "Some comment.", "previous_comment": "Some context."},
    )
    assert response.status_code == 200  # classification path unharmed
    reasoning = response.json()["reasoning"]
    assert reasoning["summary"] == "Reasoning unavailable."
    assert reasoning["evidence"] == []
    assert reasoning["links"] == []
