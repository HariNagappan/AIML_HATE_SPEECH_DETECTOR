"""Convert token attributions into ranked, readable evidence spans.

Pipeline (project spec §12):

1. take per-token attribution scores for the explained prediction
2. rank / threshold tokens (positive attributions support the prediction)
3. merge adjacent tokens
4. decode subword pieces back into readable phrases
5. return the top-k spans

Every span corresponds to an **actual input span** — no generated text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.datasets.tokenizer import TokenizerWrapper
from app.reasoning.attribution import AttributionResult


@dataclass
class EvidenceSpan:
    """One evidence phrase grounded in the input tokens."""

    text: str
    score: float                    # normalised to (0, 1] w.r.t. the strongest token
    raw_score: float                # mean magnitude of the contributing tokens
    token_indices: List[int]        # positions in the tokenized current comment
    type: Optional[str] = None      # span typing is NOT inferred by default
    start: Optional[int] = None     # character offset into the original comment (inclusive)
    end: Optional[int] = None       # character offset into the original comment (exclusive)


def _decode_piece(
    tokens: Sequence[str],
    tokenizer: Optional[TokenizerWrapper],
) -> str:
    """Turn a run of subword tokens back into readable text."""
    if tokenizer is not None:
        try:
            return tokenizer.decode_tokens(tokens)
        except Exception:  # pragma: no cover - decode should not fail
            pass
    # Fallback: join "##" continuations manually.
    text = ""
    for token in tokens:
        if token.startswith("##"):
            text += token[2:]
        else:
            text = f"{text} {token}" if text else token
    return text


def _span_character_range(
    token_indices: Sequence[int],
    offsets: Optional[Sequence[Tuple[int, int]]],
) -> Tuple[Optional[int], Optional[int]]:
    """Character range of a token group, using the tokenizer's offset mapping."""
    if not offsets:
        return None, None
    ranges = [
        offsets[index]
        for index in token_indices
        if 0 <= index < len(offsets) and offsets[index][1] > offsets[index][0]
    ]
    if not ranges:
        return None, None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def extract_evidence(
    attribution: AttributionResult,
    tokenizer: Optional[TokenizerWrapper] = None,
    *,
    top_k: int = 5,
    min_score_frac: float = 0.15,
    merge_adjacent: bool = True,
    use_negative_scores: bool = False,
    offsets: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[EvidenceSpan]:
    """Extract ranked evidence spans from an :class:`AttributionResult`.

    * ``min_score_frac`` – keep tokens whose magnitude is at least this
      fraction of the strongest token (relative threshold, robust to scale).
    * ``use_negative_scores`` – also consider attributions *against* the
      class (off by default: evidence supports the prediction).
    * ``offsets`` – optional per-token character ranges (see
      :func:`app.datasets.tokenizer.token_character_offsets`); when given,
      each span also carries ``start``/``end`` character offsets.
    """
    scores = [float(value) for value in attribution.scores]
    if not scores:
        return []

    magnitudes = [
        abs(value) if use_negative_scores else max(0.0, value) for value in scores
    ]
    max_magnitude = max(magnitudes)
    if max_magnitude <= 0.0:
        return []

    threshold = max_magnitude * float(min_score_frac)
    candidates = [
        index
        for index, magnitude in enumerate(magnitudes)
        if magnitude > 0.0 and magnitude >= threshold
    ]
    if not candidates:
        return []

    # Group adjacent candidate indices.
    groups: List[List[int]] = []
    if merge_adjacent:
        current_group = [candidates[0]]
        for index in candidates[1:]:
            if index == current_group[-1] + 1:
                current_group.append(index)
            else:
                groups.append(current_group)
                current_group = [index]
        groups.append(current_group)
    else:
        groups = [[index] for index in candidates]

    spans: List[EvidenceSpan] = []
    for group in groups:
        pieces = [attribution.tokens[index] for index in group]
        text = _decode_piece(pieces, tokenizer).strip()
        if not text:
            continue
        raw_score = sum(magnitudes[index] for index in group) / len(group)
        span_start, span_end = _span_character_range(group, offsets)
        spans.append(
            EvidenceSpan(
                text=text,
                score=raw_score / max_magnitude,
                raw_score=raw_score,
                token_indices=group,
                start=span_start,
                end=span_end,
            )
        )

    # De-duplicate identical texts (keep the highest scoring occurrence).
    best_by_text: Dict[str, EvidenceSpan] = {}
    for span in spans:
        previous = best_by_text.get(span.text)
        if previous is None or span.score > previous.score:
            best_by_text[span.text] = span

    ordered = sorted(
        best_by_text.values(), key=lambda span: (-span.score, min(span.token_indices))
    )
    return ordered[: max(1, int(top_k))]


def evidence_to_dict(span: EvidenceSpan) -> Dict[str, object]:
    """JSON-ready representation used by the API and structured explanation.

    Includes character offsets (``start``/``end`` into the original comment —
    end exclusive) when available, plus inclusive token indices.
    """
    payload: Dict[str, object] = {"text": span.text, "score": float(span.score)}
    if span.start is not None and span.end is not None:
        payload["start"] = int(span.start)
        payload["end"] = int(span.end)
    if span.token_indices:
        payload["token_start"] = int(min(span.token_indices))
        payload["token_end"] = int(max(span.token_indices))
    if span.type is not None:
        payload["type"] = span.type
    return payload
