"""Integration test with the real pretrained model (opt-in).

Enable with::

    RUN_INTEGRATION_TESTS=1 python -m pytest -m integration

Downloads ~440 MB of BERT weights — deliberately excluded from the default
test run.
"""

from __future__ import annotations

import os

import pytest
import torch

RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS") == "1"


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="set RUN_INTEGRATION_TESTS=1 to download and use bert-base-uncased",
)
def test_real_pretrained_bert_forward():
    from app.models.bert_encoder import BertEncoder

    encoder = BertEncoder.from_pretrained("bert-base-uncased")
    outputs = encoder(
        input_ids=torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
    )
    assert outputs["cls"].shape == (1, 768)
    assert outputs["last_hidden_state"].shape == (1, 6, 768)


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="set RUN_INTEGRATION_TESTS=1 to download and use bert-base-uncased",
)
def test_real_pipeline_end_to_end():
    from app.datasets.tokenizer import TokenizerWrapper
    from app.models.bert_encoder import BertEncoder
    from app.models.full_model import FullModel
    from app.reasoning.attribution import compute_attributions
    from app.reasoning.evidence_extractor import extract_evidence

    tokenizer = TokenizerWrapper.from_pretrained("bert-base-uncased")
    encoder = BertEncoder.from_pretrained("bert-base-uncased")
    model = FullModel(encoder, ["hate", "normal"], interaction_dim=256)
    result = compute_attributions(
        model,
        tokenizer,
        "Those people are disgusting",
        context_text="We discussed immigrants",
        steps=4,
    )
    spans = extract_evidence(result, tokenizer, top_k=3)
    assert isinstance(spans, list)
    assert len(result.tokens) > 2
