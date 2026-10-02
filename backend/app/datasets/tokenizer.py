"""Tokenizer wrapper and subword alignment utilities (Hugging Face tokenizers).

The wrapper keeps the pipeline explicit: raw text → input ids, attention
mask and token strings, plus helpers to map word-level rationale labels onto
subword tokens.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]", "<s>", "</s>", "<pad>", "<unk>", "<mask>"}


def is_special_token(token: str) -> bool:
    return token in SPECIAL_TOKENS


class TokenizerWrapper:
    """Thin wrapper around a Hugging Face tokenizer (fast tokenizer preferred)."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    # --- construction ---------------------------------------------------------
    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> "TokenizerWrapper":
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(name_or_path, **kwargs))

    # --- properties -------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    @property
    def pad_token_id(self) -> int:
        pad_id = self.tokenizer.pad_token_id
        return int(pad_id) if pad_id is not None else 0

    @property
    def is_fast(self) -> bool:
        return bool(getattr(self.tokenizer, "is_fast", False))

    # --- encoding / decoding ------------------------------------------------------
    def encode(
        self,
        texts: Union[str, Sequence[str]],
        max_length: int = 128,
        padding: Union[bool, str] = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tokenize text(s); returns a plain dict of tokenizer outputs."""
        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return dict(enc)

    def encode_with_offsets(
        self, text: str, max_length: int = 128
    ) -> Dict[str, Any]:
        """Tokenize a single text and return character offsets (fast tokenizers)."""
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
        )
        return {
            "input_ids": list(enc["input_ids"]),
            "attention_mask": list(enc["attention_mask"]),
            "offset_mapping": list(enc["offset_mapping"]),
        }

    def token_strings(self, input_ids: Sequence[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens([int(i) for i in input_ids])

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            [int(i) for i in ids], skip_special_tokens=skip_special_tokens
        )

    def decode_tokens(self, tokens: Sequence[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(list(tokens))


def align_rationales_to_subwords(
    tokenizer: TokenizerWrapper,
    text: str,
    rationale_labels: Sequence[int],
    max_length: int = 128,
) -> List[int]:
    """Align word-level rationale labels (0/1 per whitespace token of ``text``)
    to the tokenizer's subword tokens.

    Every subword inherits the label of the word it belongs to; special tokens
    and padding are 0. When the rationale list is shorter than the text, the
    missing words are treated as 0 (never invented).

    Uses the fast tokenizer's offset mapping when available, otherwise a
    ``##``-continuation heuristic.
    """
    rationale = [int(v) for v in rationale_labels]

    if tokenizer.is_fast:
        enc = tokenizer.encode_with_offsets(text, max_length=max_length)
        offsets = enc["offset_mapping"]
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        starts = [w[0] for w in words]
        labels: List[int] = []
        for (start, end) in offsets:
            if start == end:  # special tokens have empty offsets
                labels.append(0)
                continue
            # find the word whose span contains this token's start
            import bisect

            wi = bisect.bisect_right(starts, start) - 1
            if wi < 0 or wi >= len(words) or not (words[wi][0] <= start < words[wi][1]):
                labels.append(0)
            else:
                labels.append(rationale[wi] if wi < len(rationale) else 0)
        return labels

    # Fallback: heuristic based on "##" continuation markers.
    enc = tokenizer.encode(text, max_length=max_length, truncation=True)
    tokens = tokenizer.token_strings(enc["input_ids"])
    labels = []
    word_index = -1
    for token in tokens:
        if is_special_token(token):
            labels.append(0)
        elif token.startswith("##"):
            labels.append(rationale[word_index] if 0 <= word_index < len(rationale) else 0)
        else:
            word_index += 1
            labels.append(rationale[word_index] if 0 <= word_index < len(rationale) else 0)
    return labels


def token_character_offsets(
    tokenizer: TokenizerWrapper, text: str, max_length: int = 128
) -> List[Tuple[int, int]]:
    """Character range ``(start, end)`` in ``text`` for every token of ``text``.

    Uses the fast tokenizer's offset mapping; special tokens get ``(0, 0)``.
    Returns an empty list when the tokenizer cannot produce offsets — callers
    then simply omit character offsets from evidence objects.
    """
    if not tokenizer.is_fast:
        return []
    encoded = tokenizer.encode_with_offsets(text, max_length=max_length)
    return [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
