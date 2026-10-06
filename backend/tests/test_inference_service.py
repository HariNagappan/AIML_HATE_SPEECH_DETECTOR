"""Inference service tests: untrained bundle + trained checkpoint path."""

from __future__ import annotations

import json

import pytest
import torch

from app.core.config import Settings
from app.models.full_model import FullModel
from app.services.inference import (
    InferenceService,
    build_model_from_checkpoint,
    build_untrained_bundle,
)

EXPECTED_KWARGS = {
    "architecture": "full",
    "encoder_name": "bert-base-uncased",
    "random_init": True,
    "hidden_size": 64,
    "random_num_layers": 2,
    "random_num_heads": 4,
    "freeze_mode": "full",
    "frozen_layers": 0,
    "hate_labels": ["hate", "offensive", "normal"],
    "target_labels": ["race", "none"],
    "reason_labels": [],
    "interaction_dim": 32,
    "interaction_dropout": 0.1,
    "interaction_mode": "mlp",
    "use_context": True,
    "use_contrastive": False,
    "use_target": True,
    "use_reason": False,
    "use_evidence": False,
    "contrastive_dim": 256,
    "hate_mode": "multiclass",
    "hate_threshold": 0.5,
}


def _tiny_settings(**overrides) -> Settings:
    # NOTE: build a plain Settings and apply overrides via model_copy —
    # init kwargs for alias-carrying fields (e.g. checkpoint_path →
    # MODEL_CHECKPOINT) are not resolved by pydantic-settings and would be
    # silently ignored. auto_load_checkpoint=False keeps the tests hermetic
    # (a test service must never pick up repo checkpoints).
    defaults = {
        "encoder_name": "bert-base-uncased",
        "init_backbone": "random",
        "device": "cpu",
        "evidence_ig_steps": 4,
        "evidence_top_k": 3,
        "auto_load_checkpoint": False,
    }
    defaults.update(overrides)
    return Settings().model_copy(update=defaults)


def test_untrained_bundle_reports_unavailable_heads():
    bundle = build_untrained_bundle(_tiny_settings())
    assert bundle.trained_heads == []
    service = InferenceService(_tiny_settings())
    service._bundle = bundle  # use the prebuilt bundle to skip a second build

    result = service.predict("Those people should leave.", "We discussed immigrants.")
    assert result["prediction"] is None
    assert result["prediction_available"] is False
    assert result["reason"] is None and result["reason_available"] is False
    assert result["reason_explanation"] is None
    assert result["evidence"] == []
    assert result["evidence_available"] is False
    assert result["context_used"] is True

    explanation = service.explain("hello", None, target="hate")
    assert explanation["available"] is False
    assert explanation["tokens"] == []

    info = service.model_info()
    assert info["trained"] is False
    assert info["loaded"] is True
    assert info["hidden_size"] == 768


def test_trained_reason_explanation(tmp_path, small_encoder, tokenizer):
    model = FullModel(
        small_encoder,
        ["hate", "offensive", "normal"],
        ["race", "none"],
        ["insult", "other"],
        use_target=True,
        use_reason=True,
        use_evidence=False,
        use_contrastive=False,
        interaction_dim=32,
    )
    checkpoint_path = tmp_path / "best_reason.pt"
    model_kwargs = dict(EXPECTED_KWARGS)
    model_kwargs.update({"use_reason": True, "reason_labels": ["insult", "other"]})
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "scheduler_state": {},
            "epoch": 0,
            "global_step": 0,
            "config": {},
            "label_maps": {"hate": {}, "target": {}, "reason": {}},
            "metrics": {},
            "model_kwargs": model_kwargs,
            "trained_heads": ["hate", "target", "reason"],
        },
        checkpoint_path,
    )
    service = InferenceService(_tiny_settings(checkpoint_path=str(checkpoint_path)))
    result = service.predict("You are an idiot and should leave", None)

    assert result["reason_available"] is True
    assert result["evidence_available"] is True
    explanation = result["reason_explanation"]
    assert explanation is not None
    assert explanation["grounded_in"]["reason"] == result["reason"]["label"]
    assert explanation["grounded_in"]["target"] == result["target"]["label"]
    assert explanation["summary"]
    assert explanation["details"]
    span_texts = [item["text"] for item in result["evidence"]]
    assert explanation["grounded_in"]["evidence"] == span_texts


def test_trained_checkpoint_predict(tmp_path, small_encoder, tokenizer):
    model = FullModel(
        small_encoder,
        ["hate", "offensive", "normal"],
        ["race", "none"],
        use_target=True,
        use_reason=False,
        use_evidence=False,
        use_contrastive=False,
        interaction_dim=32,
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "scheduler_state": {},
            "epoch": 0,
            "global_step": 0,
            "config": {},
            "label_maps": {"hate": {"hate": 0, "offensive": 1, "normal": 2}, "target": {"race": 0, "none": 1}, "reason": {}},
            "metrics": {},
            "model_kwargs": dict(EXPECTED_KWARGS),
            "trained_heads": ["hate", "target"],
        },
        checkpoint_path,
    )

    bundle = build_model_from_checkpoint(str(checkpoint_path), _tiny_settings())
    assert set(bundle.trained_heads) == {"hate", "target"}

    settings = _tiny_settings(checkpoint_path=str(checkpoint_path))
    service = InferenceService(settings)
    result = service.predict(
        "Those people are disgusting", "We were discussing immigrants"
    )
    assert result["prediction_available"] is True
    assert result["prediction"]["label"] in {"hate", "offensive", "normal"}
    assert 0.0 <= result["prediction"]["confidence"] <= 1.0
    assert result["target_available"] is True
    assert result["reason"] is None and result["reason_available"] is False
    assert result["reason_explanation"] is None
    assert isinstance(result["evidence"], list)
    assert len(result["evidence"]) <= 3
    assert result["context_used"] is True

    explanation = service.explain("Those people are disgusting", None, target="hate", top_k=2)
    assert explanation["available"] is True
    assert len(explanation["spans"]) <= 2
    assert len(explanation["tokens"]) >= 3
    assert explanation["method"] == "integrated_gradients"

    reason_explanation = service.explain("text", None, target="reason")
    assert reason_explanation["available"] is False

    info = service.model_info()
    assert info["trained"] is True
    assert set(info["trained_heads"]) == {"hate", "target"}
    assert info["hidden_size"] == 64
    assert info["num_labels"]["hate"] == 3


def test_trained_reasoning_links_current_to_previous_comment(
    tmp_path, small_encoder, tokenizer
):
    """§17 Test 5: the reasoning block surfaces the pronoun -> context link."""
    model = FullModel(
        small_encoder,
        ["hate", "offensive", "normal"],
        ["race", "none"],
        use_target=True,
        use_reason=False,
        use_evidence=False,
        use_contrastive=False,
        interaction_dim=32,
    )
    checkpoint_path = tmp_path / "best_context.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "scheduler_state": {},
            "epoch": 0,
            "global_step": 0,
            "config": {},
            "label_maps": {
                "hate": {"hate": 0, "offensive": 1, "normal": 2},
                "target": {"race": 0, "none": 1},
                "reason": {},
            },
            "metrics": {},
            "model_kwargs": dict(EXPECTED_KWARGS),
            "trained_heads": ["hate", "target"],
        },
        checkpoint_path,
    )
    service = InferenceService(_tiny_settings(checkpoint_path=str(checkpoint_path)))
    result = service.predict("They should leave.", "Those immigrants were protesting.")

    reasoning = result["reasoning"]
    assert reasoning is not None
    assert reasoning["context_used"] is True
    assert reasoning["context_available"] is True
    assert reasoning["links"]
    link = reasoning["links"][0]
    assert link["from_text"] == "They"
    assert "immigrants" in link["to_text"]
    assert isinstance(reasoning["summary"], str) and reasoning["summary"]
    sources = {item["source"] for item in reasoning["evidence"]}
    assert "previous_comment" in sources


def test_model_info_includes_evaluation_metrics(tmp_path, small_encoder, tokenizer):
    """The metrics file written by scripts/evaluate.py is surfaced by model_info."""
    model = FullModel(
        small_encoder,
        ["hate", "offensive", "normal"],
        ["race", "none"],
        use_target=True,
        use_reason=False,
        use_evidence=False,
        use_contrastive=False,
        interaction_dim=32,
    )
    checkpoint_path = tmp_path / "best_metrics.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "scheduler_state": {},
            "epoch": 0,
            "global_step": 0,
            "config": {},
            "label_maps": {"hate": {}, "target": {}, "reason": {}},
            "metrics": {},
            "model_kwargs": dict(EXPECTED_KWARGS),
            "trained_heads": ["hate", "target"],
        },
        checkpoint_path,
    )
    metrics_payload = {
        "checkpoint": str(checkpoint_path),
        "config": "cc_context",
        "dataset": "counter_context",
        "split": "test",
        "num_examples": 713,
        "evaluated_at": "2026-10-06T12:00:00+00:00",
        "hate": {
            "accuracy": 0.5947,
            "macro_precision": 0.5512,
            "macro_recall": 0.5784,
            "macro_f1": 0.5501,
            "weighted_precision": 0.5901,
            "weighted_recall": 0.5947,
            "weighted_f1": 0.5923,
        },
        "target": None,
        "evidence": None,
    }
    (tmp_path / "best_metrics.metrics.json").write_text(
        json.dumps(metrics_payload), encoding="utf-8"
    )

    service = InferenceService(_tiny_settings(checkpoint_path=str(checkpoint_path)))
    service._bundle = build_model_from_checkpoint(
        str(checkpoint_path), _tiny_settings()
    )

    metrics = service.model_info()["metrics"]
    assert metrics is not None
    assert metrics["split"] == "test"
    assert metrics["dataset"] == "counter_context"
    assert metrics["num_examples"] == 713
    assert metrics["accuracy"] == pytest.approx(0.5947)
    assert metrics["macro_precision"] == pytest.approx(0.5512)
    assert metrics["macro_recall"] == pytest.approx(0.5784)
    assert metrics["macro_f1"] == pytest.approx(0.5501)
    assert metrics["weighted_f1"] == pytest.approx(0.5923)


def test_model_info_metrics_null_without_evaluation_file(
    tmp_path, small_encoder, tokenizer
):
    """No metrics file -> explicit None (availability is never fabricated)."""
    model = FullModel(
        small_encoder,
        ["hate", "offensive", "normal"],
        ["race", "none"],
        use_target=True,
        use_reason=False,
        use_evidence=False,
        use_contrastive=False,
        interaction_dim=32,
    )
    checkpoint_path = tmp_path / "best_no_metrics.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "scheduler_state": {},
            "epoch": 0,
            "global_step": 0,
            "config": {},
            "label_maps": {"hate": {}, "target": {}, "reason": {}},
            "metrics": {},
            "model_kwargs": dict(EXPECTED_KWARGS),
            "trained_heads": ["hate", "target"],
        },
        checkpoint_path,
    )
    service = InferenceService(_tiny_settings(checkpoint_path=str(checkpoint_path)))
    service._bundle = build_model_from_checkpoint(
        str(checkpoint_path), _tiny_settings()
    )
    assert service.model_info()["metrics"] is None
