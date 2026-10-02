"""Evidence extraction + attribution tests."""

from __future__ import annotations

import pytest
import torch

from app.datasets.tokenizer import token_character_offsets
from app.models.full_model import FullModel
from app.reasoning.attribution import AttributionResult, compute_attributions
from app.reasoning.evidence_extractor import EvidenceSpan, evidence_to_dict, extract_evidence

SPECIALS = {"[CLS]", "[SEP]", "[PAD]"}


def make_attribution(tokens, scores):
    return AttributionResult(
        method="test",
        target="hate",
        target_index=2,
        target_label="hate",
        predicted_label="hate",
        predicted_confidence=0.9,
        tokens=tokens,
        token_ids=list(range(len(tokens))),
        scores=scores,
        steps=1,
        context_used=False,
    )


def test_specials_excluded_and_adjacent_merged():
    attribution = make_attribution(
        ["[CLS]", "those", "people", "are", "gross", "[SEP]"],
        [0.0, 0.9, 0.8, 0.05, 0.7, 0.0],
    )
    spans = extract_evidence(attribution, None, top_k=5, min_score_frac=0.5)
    flattened = " ".join(span.text for span in spans)
    assert "[CLS]" not in flattened and "[SEP]" not in flattened
    assert spans[0].text == "those people"  # adjacent tokens merged
    assert {"those people", "gross"} == {span.text for span in spans}


def test_ranking_by_score():
    attribution = make_attribution(
        ["[CLS]", "aa", "bb", "cc", "[SEP]"], [0, 0.2, 0.9, 0.5, 0]
    )
    spans = extract_evidence(
        attribution, None, top_k=5, min_score_frac=0.1, merge_adjacent=False
    )
    scores = [span.score for span in spans]
    assert scores == sorted(scores, reverse=True)
    assert spans[0].text == "bb"


def test_subword_merge_fallback_without_tokenizer():
    attribution = make_attribution(
        ["[CLS]", "disgust", "##ing", "[SEP]"], [0, 0.8, 0.7, 0]
    )
    spans = extract_evidence(attribution, None, top_k=3, min_score_frac=0.5)
    assert spans[0].text == "disgusting"


def test_top_k_cap():
    tokens = ["[CLS]"] + [f"w{index}" for index in range(10)] + ["[SEP]"]
    scores = [0.0] + [1.0 - 0.05 * index for index in range(10)] + [0.0]
    attribution = make_attribution(tokens, scores)
    spans = extract_evidence(attribution, None, top_k=3, min_score_frac=0.0, merge_adjacent=False)
    assert len(spans) == 3


def test_empty_when_all_zero():
    attribution = make_attribution(["[CLS]", "x", "[SEP]"], [0.0, 0.0, 0.0])
    assert extract_evidence(attribution, None) == []


def test_duplicate_evidence_deduplicated():
    attribution = make_attribution(["[CLS]", "bad", "bad", "[SEP]"], [0, 0.9, 0.7, 0])
    spans = extract_evidence(
        attribution, None, top_k=5, min_score_frac=0.1, merge_adjacent=False
    )
    texts = [span.text for span in spans]
    assert len(texts) == len(set(texts))


def test_token_metrics_hand_computed():
    from app.evaluation.rationale import token_binary_metrics

    metrics = token_binary_metrics([1, 0, 1, 0], [1, 1, 0, 0])
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_attribution_end_to_end(small_encoder, tokenizer):
    model = FullModel(small_encoder, ["hate", "normal"], interaction_dim=16)
    result = compute_attributions(
        model,
        tokenizer,
        "Those people are disgusting",
        context_text="We discussed immigrants",
        method="integrated_gradients",
        steps=4,
        max_length=16,
    )
    assert len(result.scores) == len(result.tokens)
    for index, token in enumerate(result.tokens):
        if token in SPECIALS:
            assert result.scores[index] == 0.0
    assert result.context_used is True

    spans = extract_evidence(result, tokenizer, top_k=3)
    assert isinstance(spans, list)

    gradient_result = compute_attributions(
        model,
        tokenizer,
        "Those people are disgusting",
        method="gradient_x_input",
        steps=1,
        max_length=16,
    )
    assert gradient_result.method == "gradient_x_input"
    assert len(gradient_result.scores) == len(gradient_result.tokens)


def test_attribution_specific_class_index(small_encoder, tokenizer):
    model = FullModel(small_encoder, ["hate", "normal"], interaction_dim=16)
    result = compute_attributions(
        model, tokenizer, "hello world", target_index=0, steps=3, max_length=16
    )
    assert result.target_index == 0
    assert result.target_label == "hate"


def test_unknown_method_raises(small_encoder, tokenizer):
    model = FullModel(small_encoder, ["hate", "normal"], interaction_dim=16)
    with pytest.raises(ValueError):
        compute_attributions(model, tokenizer, "x", method="magic", steps=2)


def test_token_character_offsets_align_with_text(tokenizer):
    text = "Those immigrants are disgusting"
    offsets = token_character_offsets(tokenizer, text, max_length=32)
    tokens = tokenizer.token_strings(
        tokenizer.encode(text, max_length=32, truncation=True)["input_ids"]
    )
    assert len(offsets) == len(tokens)
    for (start, end), token in zip(offsets, tokens):
        if token in SPECIALS:
            assert (start, end) == (0, 0)
        else:
            assert 0 <= start < end <= len(text)
            assert text[start:end].strip() != ""


def test_evidence_spans_carry_character_offsets(small_encoder, tokenizer):
    model = FullModel(small_encoder, ["hate", "normal"], interaction_dim=16)
    text = "Those immigrants are disgusting and should leave"
    attribution = compute_attributions(
        model, tokenizer, text, method="integrated_gradients", steps=3, max_length=32
    )
    offsets = token_character_offsets(tokenizer, text, max_length=32)
    spans = extract_evidence(attribution, tokenizer, top_k=3, offsets=offsets)
    for span in spans:
        assert span.start is not None and span.end is not None
        assert 0 <= span.start < span.end <= len(text)
        sliced = text[span.start : span.end].strip().lower().replace(" ", "")
        assert sliced == span.text.lower().replace(" ", "")


def test_evidence_to_dict_includes_offsets():
    span = EvidenceSpan(
        text="kicked out",
        score=0.8,
        raw_score=0.8,
        token_indices=[4, 5],
        start=41,
        end=51,
    )
    payload = evidence_to_dict(span)
    assert payload["start"] == 41
    assert payload["end"] == 51
    assert payload["token_start"] == 4
    assert payload["token_end"] == 5

    bare = EvidenceSpan(text="x", score=0.1, raw_score=0.1, token_indices=[])
    bare_payload = evidence_to_dict(bare)
    assert "start" not in bare_payload
    assert "token_start" not in bare_payload


def test_offsets_absent_without_mapping(small_encoder, tokenizer):
    model = FullModel(small_encoder, ["hate", "normal"], interaction_dim=16)
    attribution = compute_attributions(
        model, tokenizer, "hello world", method="gradient_x_input", steps=1, max_length=16
    )
    spans = extract_evidence(attribution, tokenizer, top_k=2)
    assert all(span.start is None and span.end is None for span in spans)
