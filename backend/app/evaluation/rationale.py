"""Rationale / evidence metrics (token-level, against gold rationales).

Used to evaluate both the supervised token-level rationale head and the
attribution-based evidence extractor against HateXplain rationales. Token-level
precision/recall/F1 is reported; a high attribution score on its own is NOT
treated as proof of correctness (README "Evaluation").
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def token_binary_metrics(
    gold: Sequence[int], predicted: Sequence[int]
) -> Dict[str, float]:
    """Token-level precision / recall / F1 for binary 0/1 masks."""
    gold = [int(value) for value in gold]
    predicted = [int(value) for value in predicted]
    length = min(len(gold), len(predicted))
    gold, predicted = gold[:length], predicted[:length]

    tp = sum(1 for g, p in zip(gold, predicted) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, predicted) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, predicted) if g == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def predicted_token_mask_from_spans(spans: Iterable[Any], seq_len: int) -> List[int]:
    """Build a 0/1 token mask from evidence spans (uses ``token_indices``)."""
    mask = [0] * int(seq_len)
    for span in spans:
        for index in getattr(span, "token_indices", []) or []:
            if 0 <= int(index) < seq_len:
                mask[int(index)] = 1
    return mask
