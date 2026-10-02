"""Preprocessing tests."""

from __future__ import annotations

import pytest

from app.datasets.preprocessing import (
    normalize_informals,
    normalize_mentions,
    normalize_repeated_characters,
    normalize_urls,
    normalize_whitespace,
    preprocess_text,
)


def test_urls_are_placeholdered():
    assert normalize_urls("visit https://example.com/x?q=1 now") == "visit [URL] now"
    assert normalize_urls("see www.example.org ok") == "see [URL] ok"


def test_mentions_are_placeholdered():
    assert normalize_mentions("@John hi there") == "@user hi there"
    assert normalize_mentions("ping u/jane_doe please") == "ping @user please"


def test_email_not_treated_as_mention():
    assert normalize_mentions("mail me at bob@example.com") == "mail me at bob@example.com"


def test_letter_floods_collapsed_but_punctuation_kept():
    assert normalize_repeated_characters("sooooo goood") == "soo good"
    assert normalize_repeated_characters("hello!!! 😡😡😡 ...") == "hello!!! 😡😡😡 ..."


def test_digit_and_punctuation_runs_untouched():
    assert normalize_repeated_characters("1000000") == "1000000"
    assert normalize_repeated_characters("........") == "........"


def test_informals_expanded():
    assert normalize_informals("pls tell me cuz I care") == "please tell me because I care"


def test_whitespace_collapsed():
    assert normalize_whitespace("a   b\n\t c ") == "a b c"


def test_pipeline_preserves_signal():
    text = "Check https://t.co/abc @bob you are sooooo dumb 😡!!!"
    out = preprocess_text(text)
    assert "[URL]" in out
    assert "@user" in out
    assert "soo" in out
    assert "!!!" in out          # meaningful punctuation preserved
    assert "😡" in out            # emoji preserved


def test_pipeline_empty_string():
    assert preprocess_text("") == ""


def test_pipeline_type_check():
    with pytest.raises(TypeError):
        preprocess_text(None)  # type: ignore[arg-type]
