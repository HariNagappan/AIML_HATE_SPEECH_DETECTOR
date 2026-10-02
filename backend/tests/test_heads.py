"""Classification head tests (hate / target / reason / evidence)."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.models.evidence_head import EvidenceHead
from app.models.hate_head import HateHead
from app.models.reason_head import ReasonHead
from app.models.target_head import TargetHead


def test_hate_head_multiclass_shapes_and_decode():
    head = HateHead(32, ["hate", "offensive", "normal"])
    logits = head(torch.randn(3, 32))
    assert logits.shape == (3, 3)
    predictions = head.predict(torch.randn(3, 32))
    assert len(predictions) == 3
    assert set(p["label"] for p in predictions) <= {"hate", "offensive", "normal"}
    assert 0.0 <= predictions[0]["confidence"] <= 1.0
    assert abs(sum(predictions[0]["probabilities"].values()) - 1.0) < 1e-5


def test_hate_head_binary_threshold_decode():
    head = HateHead(8, ["non_hate", "hate"], mode="binary", threshold=0.5)
    with torch.no_grad():
        head.classifier.weight.copy_(torch.ones(1, 8))
        head.classifier.bias.zero_()
    logits = head(torch.tensor([[1.0] * 8, [-1.0] * 8]))
    assert logits.shape == (2, 1)
    predictions = head.predict(torch.tensor([[1.0] * 8, [-1.0] * 8]))
    assert predictions[0]["label"] == "hate"
    assert predictions[1]["label"] == "non_hate"


def test_hate_head_binary_requires_two_labels():
    import pytest

    with pytest.raises(ValueError):
        HateHead(8, ["a", "b", "c"], mode="binary")


def test_target_head():
    head = TargetHead(16, ["race", "religion", "none"])
    predictions = head.predict(torch.randn(2, 16))
    assert set(p["label"] for p in predictions) <= {"race", "religion", "none"}


def test_reason_head():
    head = ReasonHead(16, ["insult", "threat", "other"])
    logits = head(torch.randn(2, 16))
    assert logits.shape == (2, 3)


def test_evidence_head_token_logits():
    head = EvidenceHead(16)
    logits = head(torch.randn(2, 5, 16))
    assert logits.shape == (2, 5)
    probabilities = head.token_probabilities(torch.randn(2, 5, 16))
    assert probabilities.shape == (2, 5)
    assert bool(((probabilities >= 0) & (probabilities <= 1)).all())
