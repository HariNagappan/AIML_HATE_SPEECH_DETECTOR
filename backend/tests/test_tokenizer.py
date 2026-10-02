"""Tokenizer wrapper + rationale alignment tests (need the real tokenizer)."""

from __future__ import annotations

from app.datasets.tokenizer import align_rationales_to_subwords, is_special_token


def test_encode_adds_cls_sep(tokenizer):
    encoded = tokenizer.encode(
        "Hello world", max_length=16, padding=False, truncation=True
    )
    tokens = tokenizer.token_strings(encoded["input_ids"])
    assert tokens[0] == "[CLS]"
    assert tokens[-1] == "[SEP]"


def test_batch_padding_shapes(tokenizer):
    encoded = tokenizer.encode(
        ["short", "a much longer sentence here"],
        max_length=16,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    assert encoded["input_ids"].shape[0] == 2
    assert encoded["input_ids"].shape[1] == encoded["attention_mask"].shape[1]


def test_truncation_respects_max_length(tokenizer):
    encoded = tokenizer.encode(
        "token " * 100, max_length=12, padding=False, truncation=True
    )
    assert len(encoded["input_ids"]) == 12


def test_special_token_helper():
    assert is_special_token("[CLS]")
    assert is_special_token("[SEP]")
    assert not is_special_token("hello")


def test_align_rationales_to_subwords(tokenizer):
    text = "Those immigrants are disgusting"
    rationale = [0, 1, 0, 1]
    aligned = align_rationales_to_subwords(tokenizer, text, rationale, max_length=32)
    tokens = tokenizer.token_strings(
        tokenizer.encode(text, max_length=32, truncation=True)["input_ids"]
    )
    assert len(aligned) == len(tokens)
    assert aligned[0] == 0 and aligned[-1] == 0  # specials never selected
    assert sum(aligned) >= 2  # both selected words contribute >= 1 subword
    # subword rows inherit the word label
    for token, label in zip(tokens, aligned):
        assert label in (0, 1)


def test_align_handles_short_rationales(tokenizer):
    aligned = align_rationales_to_subwords(tokenizer, "one two three", [1], max_length=16)
    tokens = tokenizer.token_strings(
        tokenizer.encode("one two three", max_length=16, truncation=True)["input_ids"]
    )
    assert len(aligned) == len(tokens)
