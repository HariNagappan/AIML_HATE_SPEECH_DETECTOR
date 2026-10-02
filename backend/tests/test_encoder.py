"""BERT encoder tests: shapes, freeze modes, debugging helper."""

from __future__ import annotations

import torch

from app.models.bert_encoder import BertEncoder


def _ids(batch=2, seq=10):
    return torch.randint(1, 100, (batch, seq))


def _mask(batch=2, seq=10):
    return torch.ones(batch, seq, dtype=torch.long)


def test_encoder_forward_shapes_small(small_encoder):
    out = small_encoder(input_ids=_ids(), attention_mask=_mask())
    assert out["cls"].shape == (2, 64)
    assert out["last_hidden_state"].shape == (2, 10, 64)
    assert out["token_embeddings"].shape == out["last_hidden_state"].shape


def test_encoder_hidden_size_768(tiny_encoder_768):
    out = tiny_encoder_768(input_ids=_ids(), attention_mask=_mask())
    # Project spec requires [batch, 768] for the [CLS]/sentence embedding.
    assert out["cls"].shape == (2, 768)
    assert out["last_hidden_state"].shape == (2, 10, 768)


def test_freeze_modes():
    frozen = BertEncoder.random_init(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, freeze_mode="frozen"
    )
    assert all(not parameter.requires_grad for parameter in frozen.bert.parameters())

    partial = BertEncoder.random_init(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        freeze_mode="partial",
        frozen_layers=2,
    )
    frozen_count = sum(
        1 for parameter in partial.bert.parameters() if not parameter.requires_grad
    )
    trainable_count = sum(
        1 for parameter in partial.bert.parameters() if parameter.requires_grad
    )
    assert frozen_count > 0
    assert trainable_count > 0


def test_invalid_freeze_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        BertEncoder.random_init(hidden_size=64, num_hidden_layers=1, num_attention_heads=4, freeze_mode="bogus")


def test_encode_text_debug_helper(small_encoder, tokenizer):
    result = small_encoder.encode_text("Hello [MASK] world", tokenizer, max_length=16)
    assert result["tokens"][0] == "[CLS]"
    assert result["tokens"][-1] == "[SEP]"
    assert result["cls"].shape == (64,)
    assert result["token_embeddings"].shape[0] == len(result["tokens"])


def test_forward_requires_inputs(small_encoder):
    import pytest

    with pytest.raises(ValueError):
        small_encoder()
