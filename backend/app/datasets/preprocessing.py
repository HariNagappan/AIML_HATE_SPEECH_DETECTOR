"""Light-weight, conservative text preprocessing.

Design principles
-----------------
* Keep information that matters for hate-speech detection: punctuation,
  emojis, negation, word order, casing.
* Only normalise what is clearly noise: URLs, user mentions, letter floods,
  whitespace and unicode variants.
* Everything is deterministic and dependency-free so it can be unit tested
  in isolation.

The output is still *the user's text* — nothing here paraphrases meaning or
removes punctuation/emojis (they carry signal for hate-speech detection).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Optional

# --- regexes ------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@[\w.]{1,30}\b")
_REDDIT_USER_RE = re.compile(r"(?<![\w/])u/[\w-]{1,30}\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

#: Small, transparent map of common informal tokens (word-boundary matched).
INFORMAL_EXPRESSIONS = {
    "u": "you",
    "ur": "your",
    "pls": "please",
    "plz": "please",
    "ppl": "people",
    "cos": "because",
    "cuz": "because",
    "thx": "thanks",
    "wanna": "want to",
    "gonna": "going to",
}


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalisation (unifies full-width/fancy unicode variants)."""
    return unicodedata.normalize("NFKC", text)


def strip_control_characters(text: str) -> str:
    """Remove control characters (C0/C1) while keeping newlines/tabs.

    Zero-width joiners used by emoji sequences are preserved on purpose.
    """
    return "".join(
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch) != "Cc"
    )


def normalize_urls(text: str, placeholder: str = "[URL]") -> str:
    """Replace URLs with a placeholder (preserves the signal of a link)."""
    return _URL_RE.sub(placeholder, text)


def normalize_mentions(text: str, placeholder: str = "@user") -> str:
    """Replace @mentions and Reddit-style ``u/username`` with a placeholder."""
    text = _MENTION_RE.sub(placeholder, text)
    return _REDDIT_USER_RE.sub(placeholder, text)


def normalize_repeated_characters(text: str, max_repeats: int = 2) -> str:
    """Collapse letter floods ("sooooo" → "soo") keeping at most ``max_repeats``.

    Only letters are affected; punctuation, digits and emoji repeats are
    preserved because they can carry meaning.
    """
    pattern = re.compile(r"([^\W\d_])\1{%d,}" % (max_repeats,), re.UNICODE)
    return pattern.sub(lambda m: m.group(1) * max_repeats, text)


def normalize_informals(text: str, mapping: Optional[Mapping[str, str]] = None) -> str:
    """Expand a small set of common informal expressions (word-boundary safe)."""
    mapping = INFORMAL_EXPRESSIONS if mapping is None else mapping
    if not mapping:
        return text
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, mapping)) + r")\b", re.IGNORECASE)

    def _replace(match: re.Match) -> str:
        word = match.group(0)
        replacement = mapping[word.lower()]
        return replacement.capitalize() if word[:1].isupper() else replacement

    return pattern.sub(_replace, text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace to single spaces and trim."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def preprocess_text(
    text: str,
    *,
    normalize_unicode_flag: bool = True,
    strip_controls: bool = True,
    normalize_urls_flag: bool = True,
    normalize_mentions_flag: bool = True,
    collapse_char_floods: bool = True,
    max_char_repeats: int = 2,
    normalize_informal_expressions: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Run the full conservative preprocessing pipeline on ``text``.

    Parameters exist for every step so ablations / tests can disable them.
    """
    if not isinstance(text, str):
        raise TypeError(f"preprocess_text expects str, got {type(text).__name__}")

    out = text
    if normalize_unicode_flag:
        out = normalize_unicode(out)
    if strip_controls:
        out = strip_control_characters(out)
    if normalize_urls_flag:
        out = normalize_urls(out)
    if normalize_mentions_flag:
        out = normalize_mentions(out)
    if collapse_char_floods:
        out = normalize_repeated_characters(out, max_repeats=max_char_repeats)
    if normalize_informal_expressions:
        out = normalize_informals(out)
    if collapse_whitespace:
        out = normalize_whitespace(out)
    return out
