"""Dataset-agnostic example container and JSONL helpers.

Every dataset adapter converts raw records into :class:`UnifiedExample`, the
common internal representation. Fields that a dataset does not annotate stay
explicitly ``None`` — nothing is ever fabricated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class UnifiedExample:
    """One instance in the common internal format.

    Attributes
    ----------
    id:
        Stable identifier (prefixed per dataset, e.g. ``cc-train-17``).
    current_text:
        The comment being classified.
    context_text:
        Previous comment / conversational context (``None`` when absent).
    hate_label:
        Canonical hate label for the dataset's label set
        (HateXplain: hate/offensive/normal; Counter Context: its own set).
    target_label / reason_label:
        Canonical target / reason category — ``None`` when not annotated.
    rationale_labels:
        0/1 flags aligned to the **whitespace tokens** of ``current_text``
        (``None`` when not annotated). Subword alignment happens in
        :mod:`app.datasets.tokenizer`.
    metadata:
        Anything else worth keeping (original labels, split, provenance).
    """

    id: str
    current_text: str
    context_text: Optional[str] = None
    hate_label: Optional[str] = None
    target_label: Optional[str] = None
    reason_label: Optional[str] = None
    rationale_labels: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- convenience ----------------------------------------------------------
    def has_context(self) -> bool:
        return bool(self.context_text and self.context_text.strip())

    def has_target(self) -> bool:
        return self.target_label is not None

    def has_reason(self) -> bool:
        return self.reason_label is not None

    def has_rationale(self) -> bool:
        return self.rationale_labels is not None

    # --- (de)serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedExample":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        if not isinstance(payload.get("current_text"), str):
            raise ValueError("UnifiedExample requires a string 'current_text'")
        payload["id"] = str(payload.get("id", ""))
        return cls(**payload)


def write_unified_jsonl(examples: Iterable[UnifiedExample], path: str | Path) -> int:
    """Write examples to a JSONL file; returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_unified_jsonl(path: str | Path) -> List[UnifiedExample]:
    """Load examples from a JSONL file written by :func:`write_unified_jsonl`."""
    path = Path(path)
    examples: List[UnifiedExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(UnifiedExample.from_dict(json.loads(line)))
    return examples


def split_by_metadata_split(
    examples: Sequence[UnifiedExample],
    key: str = "split",
    train_values: Tuple[str, ...] = ("train", "gold-train", "silver-train"),
    val_values: Tuple[str, ...] = ("val", "dev", "validation", "gold-val", "silver-val"),
    test_values: Tuple[str, ...] = ("test", "gold-test"),
) -> Tuple[List[UnifiedExample], List[UnifiedExample], List[UnifiedExample]]:
    """Split examples by ``metadata[key]`` into (train, val, test)."""
    train, val, test = [], [], []
    for example in examples:
        split = str(example.metadata.get(key, "")).lower()
        if split in train_values:
            train.append(example)
        elif split in val_values:
            val.append(example)
        elif split in test_values:
            test.append(example)
    return train, val, test
