"""Collators: convert batches of :class:`UnifiedExample` into model tensors.

* :class:`MultitaskCollator` – tokenizes the current comment (and optional
  context) and prepares all task labels. Missing labels use ``-100``
  (``IGNORE_INDEX``) so losses skip them — no fake supervision.
* :class:`ContrastivePairCollator` – wraps the base collator and additionally
  prepares positive/negative context tensors for InfoNCE training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from app.datasets.tokenizer import TokenizerWrapper, align_rationales_to_subwords
from app.datasets.unified import UnifiedExample

IGNORE_INDEX = -100


@dataclass
class LabelMaps:
    """Canonical label → class index (per head)."""

    hate: Dict[str, int] = field(default_factory=dict)
    target: Dict[str, int] = field(default_factory=dict)
    reason: Dict[str, int] = field(default_factory=dict)


def build_label_maps(
    hate_labels: Sequence[str],
    target_labels: Sequence[str] = (),
    reason_labels: Sequence[str] = (),
) -> LabelMaps:
    return LabelMaps(
        hate={label: index for index, label in enumerate(hate_labels)},
        target={label: index for index, label in enumerate(target_labels)},
        reason={label: index for index, label in enumerate(reason_labels)},
    )


class MultitaskCollator:
    """Tokenize a batch of examples into tensors for the full model."""

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        label_maps: LabelMaps,
        max_length: int = 128,
        include_evidence: bool = False,
        ignore_index: int = IGNORE_INDEX,
    ):
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length
        self.include_evidence = include_evidence
        self.ignore_index = ignore_index

    def _label_id(self, kind: str, label: Optional[str]) -> int:
        if label is None:
            return self.ignore_index
        mapping: Dict[str, int] = getattr(self.label_maps, kind)
        return mapping.get(label, self.ignore_index)

    def __call__(self, batch: List[UnifiedExample]) -> Dict[str, Any]:
        current = [example.current_text for example in batch]
        encoded = self.tokenizer.encode(
            current,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        out: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "example_ids": [example.id for example in batch],
        }

        # Context (only when at least one example carries it).
        if any(example.has_context() for example in batch):
            contexts = [
                example.context_text if example.has_context() else ""
                for example in batch
            ]
            context_encoded = self.tokenizer.encode(
                contexts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            out["context_input_ids"] = context_encoded["input_ids"]
            out["context_attention_mask"] = context_encoded["attention_mask"]
        out["context_present"] = torch.tensor(
            [1.0 if example.has_context() else 0.0 for example in batch],
            dtype=torch.float,
        )

        # Task labels (-100 == ignore).
        out["hate_labels"] = torch.tensor(
            [self._label_id("hate", example.hate_label) for example in batch],
            dtype=torch.long,
        )
        out["target_labels"] = torch.tensor(
            [self._label_id("target", example.target_label) for example in batch],
            dtype=torch.long,
        )
        out["reason_labels"] = torch.tensor(
            [self._label_id("reason", example.reason_label) for example in batch],
            dtype=torch.long,
        )

        # Token-level rationale supervision (subword-aligned).
        if self.include_evidence:
            seq_len = int(out["input_ids"].shape[1])
            rows: List[List[int]] = []
            for example in batch:
                if example.has_rationale():
                    aligned = align_rationales_to_subwords(
                        self.tokenizer,
                        example.current_text,
                        example.rationale_labels or [],
                        max_length=self.max_length,
                    )
                    row = aligned[:seq_len] + [self.ignore_index] * max(
                        0, seq_len - len(aligned)
                    )
                    rows.append(row)
                else:
                    rows.append([self.ignore_index] * seq_len)
            out["rationale_labels"] = torch.tensor(rows, dtype=torch.long)

        return out


class ContrastivePairCollator:
    """Wraps :class:`MultitaskCollator` and adds contrastive context tensors.

    Positive context = the example's own preceding comment.
    Negative context = a preceding comment drawn from ``negative_pool``
    (sampled with a seedable RNG; call :meth:`set_epoch` to vary across epochs).
    """

    def __init__(
        self,
        base_collator: MultitaskCollator,
        negative_pool: Sequence[str],
        seed: int = 42,
        negatives_per_anchor: int = 1,
    ):
        self.base_collator = base_collator
        self.negative_pool = [str(c) for c in negative_pool if c and str(c).strip()]
        if not self.negative_pool:
            raise ValueError("negative_pool must contain at least one context string")
        self.negatives_per_anchor = max(1, int(negatives_per_anchor))
        self.seed = seed
        self._rng = random.Random(seed)

    def set_epoch(self, epoch: int) -> None:
        """Resample negatives deterministically per epoch."""
        self._rng = random.Random(self.seed + int(epoch))

    def __call__(self, batch: List[UnifiedExample]) -> Dict[str, Any]:
        out = self.base_collator(batch)
        tokenizer = self.base_collator.tokenizer
        max_length = self.base_collator.max_length

        positives = [
            example.context_text if example.has_context() else "" for example in batch
        ]
        negatives: List[str] = []
        for _ in batch:
            for _ in range(self.negatives_per_anchor):
                negatives.append(self._rng.choice(self.negative_pool))

        positive_encoded = tokenizer.encode(
            positives,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        negative_encoded = tokenizer.encode(
            negatives,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        out["positive_context_input_ids"] = positive_encoded["input_ids"]
        out["positive_context_attention_mask"] = positive_encoded["attention_mask"]
        out["negative_context_input_ids"] = negative_encoded["input_ids"]
        out["negative_context_attention_mask"] = negative_encoded["attention_mask"]
        out["contrastive_active"] = torch.tensor(
            [example.has_context() for example in batch], dtype=torch.bool
        )
        out["negatives_per_anchor"] = self.negatives_per_anchor
        return out
