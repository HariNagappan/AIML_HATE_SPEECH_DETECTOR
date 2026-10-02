"""Adapter for the HateXplain dataset (primary dataset).

Sources
-------
* https://github.com/hate-alert/HateXplain  (original corpus)
* https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain  (HF mirror)

Schema (original ``dataset.json`` records)
------------------------------------------
* ``post_tokens``  – tokenised tweet text
* ``annotators``   – ``[{"label": "hatespeech"|"offensive"|"normal",
                        "target": ["African", ...], ...}, ...]``
* ``rationales``   – per-annotator token-level 0/1 flags aligned to ``post_tokens``
* ``post_id``      – record id (the file is a mapping ``post_id -> record``)

What HateXplain provides: hate/offensive/normal labels (majority vote over
annotators), target communities, and token-level rationales.

What it does NOT provide: conversational context and reason categories.
Those fields stay ``None`` — they are never invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.datasets.label_mapping import (
    map_hatexplain_hate,
    map_hatexplain_target,
    majority_vote,
)
from app.datasets.unified import UnifiedExample


def merge_rationales(
    rationales: Sequence[Sequence[int]],
    use_flags: Optional[Sequence[bool]] = None,
    expected_len: Optional[int] = None,
) -> Optional[List[int]]:
    """Merge per-annotator 0/1 rationales into one token mask.

    A token is marked 1 when at least half of the selected annotators marked
    it (ties count as 1). Falls back to using all annotators when none is
    selected. Returns ``None`` when there is nothing to merge.
    """
    selected: List[Sequence[int]] = []
    if rationales:
        if use_flags is not None:
            selected = [r for r, use in zip(rationales, use_flags) if use]
        if not selected:
            selected = list(rationales)
    if not selected:
        return None

    length = max(len(r) for r in selected)
    if expected_len:
        length = min(length, expected_len)
    merged: List[int] = []
    for i in range(length):
        votes = [int(r[i]) for r in selected if i < len(r) and int(r[i]) in (0, 1)]
        if votes and sum(votes) * 2 >= len(votes):
            merged.append(1)
        else:
            merged.append(0)
    return merged


class HateXplainDataset:
    """Adapter exposing HateXplain records as :class:`UnifiedExample` objects."""

    name = "hatexplain"

    def __init__(self, records: Sequence[Dict[str, Any]], split: str = "unspecified"):
        self.records = list(records)
        self.split = split

    def __len__(self) -> int:
        return len(self.records)

    # --- constructors ----------------------------------------------------------
    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        split: str = "unspecified",
        keep_ids: Optional[Sequence[str]] = None,
    ) -> "HateXplainDataset":
        """Load the original ``dataset.json`` (a list of records).

        ``keep_ids`` optionally restricts the records (e.g. a split defined by
        ``post_id_divisions.json``).
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # Original corpus format: mapping ``post_id -> record``. Records do
            # not always carry their own "id" — use "post_id" / the key then.
            records = []
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                record = dict(value)
                record.setdefault("id", record.get("post_id", key))
                records.append(record)
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(
                "HateXplain dataset.json must be a list or an id->record mapping"
            )
        if keep_ids is not None:
            wanted = {str(i) for i in keep_ids}
            records = [r for r in records if str(r.get("id")) in wanted]
        return cls(records, split=split)

    @classmethod
    def from_hf(
        cls, split: str = "train", cache_dir: Optional[str] = None
    ) -> "HateXplainDataset":
        """Load via the (optional) Hugging Face ``datasets`` library."""
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install the optional `datasets` package to use "
                "HateXplainDataset.from_hf()."
            ) from exc
        hf_dataset = load_dataset(
            "Hate-speech-CNERG/hatexplain", split=split, cache_dir=cache_dir
        )
        return cls([dict(record) for record in hf_dataset], split=split)

    # --- conversion --------------------------------------------------------------
    def to_unified(self) -> List[UnifiedExample]:
        return [self._convert(record, index) for index, record in enumerate(self.records)]

    def _convert(self, record: Dict[str, Any], index: int) -> UnifiedExample:
        tokens = list(record.get("post_tokens") or [])
        text = " ".join(tokens)
        annotators = list(record.get("annotators") or [])
        raw_labels = [str(annotator.get("label", "")).strip().lower() for annotator in annotators]

        canonical_labels: List[str] = []
        unmapped_labels: List[str] = []
        for raw in raw_labels:
            canonical, unmapped = map_hatexplain_hate(raw)
            if canonical:
                canonical_labels.append(canonical)
            if unmapped:
                unmapped_labels.append(str(unmapped))
        hate_label = majority_vote(canonical_labels)

        # Targets: union of communities proposed by annotators agreeing with
        # the majority hate label.
        target_raws: List[str] = []
        for annotator, raw in zip(annotators, raw_labels):
            canonical, _ = map_hatexplain_hate(raw)
            if canonical != hate_label:
                continue
            for target in annotator.get("target") or []:
                if target:
                    target_raws.append(str(target))

        target_votes: List[str] = []
        unmapped_targets: List[str] = []
        for raw in target_raws:
            canonical, unmapped = map_hatexplain_target(raw)
            if canonical:
                target_votes.append(canonical)
            if unmapped:
                unmapped_targets.append(str(unmapped))
        target_label = majority_vote(target_votes) if target_votes else None

        # Rationales merged from annotators matching the majority label.
        rationales = record.get("rationales") or []
        use_flags = [map_hatexplain_hate(raw)[0] == hate_label for raw in raw_labels]
        merged_rationale = merge_rationales(
            rationales, use_flags, expected_len=len(tokens) or None
        )
        if not tokens:
            merged_rationale = None

        raw_id = record.get("id") or record.get("post_id") or index
        metadata: Dict[str, Any] = {
            "dataset": self.name,
            "split": self.split,
            "original_id": raw_id,
            "source": record.get("source"),
            "num_annotators": len(annotators),
            "annotator_labels": raw_labels,
        }
        if unmapped_labels:
            metadata["unmapped_hate_labels"] = sorted(set(unmapped_labels))
        if unmapped_targets:
            metadata["unmapped_targets"] = sorted(set(unmapped_targets))

        return UnifiedExample(
            id=f"hx-{raw_id}",
            current_text=text,
            context_text=None,
            hate_label=hate_label,
            target_label=target_label,
            reason_label=None,  # HateXplain has no reason categories
            rationale_labels=merged_rationale,
            metadata=metadata,
        )
