"""Full model / baseline tests: forward pass, switches, spec dimensions."""

from __future__ import annotations

import torch

from app.models.baseline import BaselineModel
from app.models.full_model import FullModel


def _tensors(batch: int = 2, seq: int = 8):
    ids = torch.randint(1, 100, (batch, seq))
    mask = torch.ones(batch, seq, dtype=torch.long)
    return ids, mask


def test_full_model_forward_with_all_heads(small_encoder):
    model = FullModel(
        small_encoder,
        ["hate", "offensive", "normal"],
        ["race", "none"],
        ["insult", "other"],
        interaction_dim=32,
        use_reason=True,
    )
    ids, mask = _tensors()
    ctx_ids, ctx_mask = _tensors()
    out = model(
        input_ids=ids,
        attention_mask=mask,
        context_input_ids=ctx_ids,
        context_attention_mask=ctx_mask,
        context_present=torch.ones(2),
    )
    assert out["hate_logits"].shape == (2, 3)
    assert out["target_logits"].shape == (2, 2)
    assert out["reason_logits"].shape == (2, 2)
    assert out["rationale_logits"].shape == (2, 8)
    assert out["projection"].shape[0] == 2


def test_full_model_768_dimension_chain(tiny_encoder_768):
    """Project spec §27: E_c / E_p are [batch, 768] for BERT-base sizes."""
    model = FullModel(tiny_encoder_768, ["hate", "normal"], interaction_dim=1024)
    ids, mask = _tensors()
    out = model(input_ids=ids, attention_mask=mask, return_representations=True)
    assert out["e_c"].shape == (2, 768)
    assert out["e_p"] is None  # no context given → explicit fallback path
    assert out["final_representation"].shape == (2, 1024)

    ctx_ids, ctx_mask = _tensors()
    out2 = model(
        input_ids=ids,
        attention_mask=mask,
        context_input_ids=ctx_ids,
        context_attention_mask=ctx_mask,
        context_present=torch.ones(2),
        return_representations=True,
    )
    assert out2["e_p"].shape == (2, 768)
    assert out2["final_representation"].shape == (2, 1024)


def test_component_switches_disable_heads(small_encoder):
    model = FullModel(
        small_encoder,
        ["hate", "normal"],
        use_context=False,
        use_contrastive=False,
        use_target=False,
        use_reason=False,
        use_evidence=False,
    )
    ids, mask = _tensors()
    out = model(input_ids=ids, attention_mask=mask)
    assert set(out.keys()) == {"hate_logits"}
    assert model.supports_context is False


def test_context_present_mask_zeroes_embedding(small_encoder):
    model = FullModel(small_encoder, ["hate", "normal"], interaction_dim=16)
    ids, mask = _tensors()
    ctx_ids, ctx_mask = _tensors()
    out = model(
        input_ids=ids,
        attention_mask=mask,
        context_input_ids=ctx_ids,
        context_attention_mask=ctx_mask,
        context_present=torch.zeros(2),
        return_representations=True,
    )
    assert torch.allclose(out["e_p"], torch.zeros_like(out["e_p"]))


def test_baseline_no_context(small_encoder):
    model = BaselineModel(small_encoder, ["non_hate", "hate"], mode="binary")
    ids, mask = _tensors()
    out = model(input_ids=ids, attention_mask=mask)
    assert out["hate_logits"].shape == (2, 1)
    assert model.supports_context is False


def test_encoder_is_shared(small_encoder):
    """One encoder module serves both current and context (weight sharing)."""
    model = FullModel(small_encoder, ["hate", "normal"], interaction_dim=16)
    encoder_modules = [name for name, _ in model.named_modules() if name == "encoder"]
    assert len(encoder_modules) == 1
