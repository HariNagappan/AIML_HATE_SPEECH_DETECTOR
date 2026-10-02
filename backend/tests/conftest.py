"""Shared test fixtures.

Design notes
------------
* The default suite runs fast and offline-safe: model tests use small
  **randomly-initialised** BERT configs (hidden size 768 where the project
  spec requires those dimensions) and never download pretrained weights.
* Tests that need the real ``bert-base-uncased`` tokenizer request the
  ``tokenizer`` fixture — it downloads only the small tokenizer files and
  skips cleanly when offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(scope="session")
def tokenizer():
    """Real bert-base-uncased tokenizer (skips when unavailable offline)."""
    from app.datasets.tokenizer import TokenizerWrapper

    try:
        return TokenizerWrapper.from_pretrained("bert-base-uncased")
    except Exception as exc:  # pragma: no cover - offline CI
        pytest.skip(f"bert-base-uncased tokenizer unavailable: {exc}")


@pytest.fixture()
def tiny_encoder_768():
    """Randomly-initialised BERT with hidden size 768 (2 layers) — spec dims."""
    from app.models.bert_encoder import BertEncoder

    return BertEncoder.random_init(
        hidden_size=768, num_hidden_layers=2, num_attention_heads=12
    )


@pytest.fixture()
def small_encoder():
    """Very small random encoder for fast tests."""
    from app.models.bert_encoder import BertEncoder

    return BertEncoder.random_init(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4
    )


@pytest.fixture()
def example_batch():
    """A small batch of UnifiedExample covering context/label/missing cases."""
    from app.datasets.unified import UnifiedExample

    return [
        UnifiedExample(
            id="e1",
            current_text="They are ruining everything.",
            context_text="We were talking about immigrants.",
            hate_label="hate",
            target_label="nationality",
        ),
        UnifiedExample(
            id="e2",
            current_text="I love this weather today.",
            context_text=None,
            hate_label="normal",
            target_label="none",
        ),
        UnifiedExample(
            id="e3",
            current_text="You are an idiot and a loser.",
            context_text="Stop posting nonsense.",
            hate_label="offensive",
            target_label=None,
            rationale_labels=[0, 1, 1, 0, 0, 0],
        ),
    ]
