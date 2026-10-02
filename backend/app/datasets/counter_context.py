"""Adapter for the Counter Context corpus (conversation-context dataset).

Paper: "Hate Speech and Counter Speech Detection: Conversational Context Does
Matter" — Xinchen Yu, Eduardo Blanco, Lingzi Hong (NAACL 2022).
Corpus: https://github.com/xinchenyu/counter_context

Files: ``data/gold/{train,val,test}.jsonl`` and ``data/silver/{train,val}.jsonl``.

Record schema (verified from the repository)::

    {"idx": int, "label": "0"|"1"|"2", "context": str, "target": str}

IMPORTANT naming quirk: here ``context`` is the *preceding comment* and
``target`` is the *current/reply comment being classified*. ``target`` is NOT
a hate-target-group annotation.

The corpus has no target-group and no rationale annotations — those remain
``None`` in the unified format. See README for how it is used for
context-dependent classification and contrastive experiments.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.datasets.label_mapping import (
    get_counter_context_label_map,
    map_counter_context_label,
)
from app.datasets.unified import UnifiedExample


class CounterContextDataset:
    """Adapter exposing Counter Context records as :class:`UnifiedExample` objects."""

    name = "counter_context"

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        split: str = "unspecified",
        label_map: Optional[Dict[str, str]] = None,
    ):
        self.records = list(records)
        self.split = split
        self.label_map = get_counter_context_label_map(label_map)

    def __len__(self) -> int:
        return len(self.records)

    # --- constructors ----------------------------------------------------------
    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        split: str = "unspecified",
        label_map: Optional[Dict[str, str]] = None,
    ) -> "CounterContextDataset":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records, split=split, label_map=label_map)

    # --- conversion --------------------------------------------------------------
    def to_unified(self) -> List[UnifiedExample]:
        examples: List[UnifiedExample] = []
        for index, record in enumerate(self.records):
            raw_label = str(record.get("label", "")).strip()
            canonical, unmapped = map_counter_context_label(raw_label, self.label_map)
            current = str(record.get("target", "") or "").strip()
            context = str(record.get("context", "") or "").strip() or None
            metadata: Dict[str, Any] = {
                "dataset": self.name,
                "split": self.split,
                "original_label": raw_label,
                "field_note": (
                    "'target' is the current comment; "
                    "'context' is the preceding comment"
                ),
            }
            if unmapped:
                metadata["unmapped_label"] = unmapped
            examples.append(
                UnifiedExample(
                    id=f"cc-{self.split}-{record.get('idx', index)}",
                    current_text=current,
                    context_text=context,
                    hate_label=canonical,
                    target_label=None,
                    reason_label=None,
                    rationale_labels=None,
                    metadata=metadata,
                )
            )
        return examples


def build_contrastive_pairs(
    examples: Sequence[UnifiedExample],
    strategy: str = "shuffled_context",
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Build (current, positive_context, negative_context) contrastive pairs.

    ``shuffled_context``: the positive context is the example's real preceding
    comment; the contrasting (negative) context is the preceding comment of a
    random *other* example. This follows the project design: the model should
    learn that swapping the context changes the interpretation of the comment.

    Examples without context are skipped. Returns a list of dicts.
    """
    if strategy != "shuffled_context":
        raise ValueError(f"Unknown contrastive negative strategy: {strategy!r}")
    rng = random.Random(seed)
    with_context = [example for example in examples if example.has_context()]
    if len(with_context) < 2:
        return []
    pairs: List[Dict[str, str]] = []
    for index, example in enumerate(with_context):
        other = index
        while other == index:
            other = rng.randrange(len(with_context))
        pairs.append(
            {
                "id": example.id,
                "current_text": example.current_text,
                "positive_context": example.context_text or "",
                "negative_context": with_context[other].context_text or "",
            }
        )
    return pairs
