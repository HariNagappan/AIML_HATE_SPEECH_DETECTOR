"""Canonical label vocabulary and per-dataset mapping layers.

Datasets annotate different label sets (HateXplain: hate/offensive/normal plus
target communities; Counter Context: hate/counter/neither). This module maps
raw dataset labels onto the internal canonical vocabulary.

Nothing is invented here: a label that cannot be mapped is returned as
``(None, raw_value)`` so callers can keep the raw value in ``metadata`` and
decide explicitly what to do with it.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --- canonical vocabularies ---------------------------------------------------

CANONICAL_HATE = ("hate", "offensive", "normal")
CANONICAL_TARGET = (
    "race",
    "ethnicity",
    "nationality",
    "religion",
    "gender",
    "sexual_orientation",
    "political_group",
    "other",
    "none",
)
CANONICAL_REASON = (
    "insult",
    "dehumanization",
    "negative_stereotyping",
    "threat",
    "exclusion",
    "discrimination",
    "incitement_to_violence",
    "other",
)

# --- HateXplain ----------------------------------------------------------------

HATEXPLAIN_HATE_MAP: Dict[str, str] = {
    "hatespeech": "hate",
    "hate speech": "hate",
    "hate": "hate",
    "offensive": "offensive",
    "normal": "normal",
}

#: Best-effort map from HateXplain's target community strings to the canonical
#: target categories. Extend this when you inspect the real dataset — unknown
#: values are NOT guessed; they fall back to None and are preserved in
#: ``example.metadata['unmapped_targets']``.
HATEXPLAIN_TARGET_MAP: Dict[str, str] = {
    # race
    "african": "race",
    "africans": "race",
    "black": "race",
    "black people": "race",
    "asian": "race",
    "asians": "race",
    "caucasian": "race",
    "white": "race",
    "white people": "race",
    # ethnicity
    "latino": "ethnicity",
    "latina": "ethnicity",
    "hispanic": "ethnicity",
    "arab": "ethnicity",
    "arabs": "ethnicity",
    "indigenous": "ethnicity",
    "native american": "ethnicity",
    "aboriginal": "ethnicity",
    # nationality
    "refugee": "nationality",
    "refugees": "nationality",
    "immigrant": "nationality",
    "immigrants": "nationality",
    "migrant": "nationality",
    "migrants": "nationality",
    "foreign": "nationality",
    "foreigner": "nationality",
    "foreigners": "nationality",
    "iraqi": "nationality",
    "iraq": "nationality",
    "mexican": "nationality",
    "mexicans": "nationality",
    "american": "nationality",
    "americans": "nationality",
    "syrian": "nationality",
    "syrians": "nationality",
    "indian": "nationality",
    # religion
    "islam": "religion",
    "islamic": "religion",
    "muslim": "religion",
    "muslims": "religion",
    "jewish": "religion",
    "jew": "religion",
    "jews": "religion",
    "judaism": "religion",
    "christian": "religion",
    "christians": "religion",
    "christianity": "religion",
    "hindu": "religion",
    "hindus": "religion",
    "sikh": "religion",
    "sikhs": "religion",
    "buddhist": "religion",
    "atheist": "religion",
    "atheism": "religion",
    "nonreligious": "religion",
    "buddhism": "religion",
    # gender
    "women": "gender",
    "woman": "gender",
    "men": "gender",
    "man": "gender",
    "female": "gender",
    "male": "gender",
    "girls": "gender",
    "boys": "gender",
    "transgender": "gender",
    "trans": "gender",
    "trans people": "gender",
    # sexual orientation
    "homosexual": "sexual_orientation",
    "homosexuals": "sexual_orientation",
    "gay": "sexual_orientation",
    "gays": "sexual_orientation",
    "lesbian": "sexual_orientation",
    "lesbians": "sexual_orientation",
    "bisexual": "sexual_orientation",
    "queer": "sexual_orientation",
    "lgbt": "sexual_orientation",
    "lgbtq": "sexual_orientation",
    "heterosexual": "sexual_orientation",
    "asexual": "sexual_orientation",
    # political group
    "political": "political_group",
    "conservative": "political_group",
    "conservatives": "political_group",
    "liberal": "political_group",
    "liberals": "political_group",
    "republican": "political_group",
    "republicans": "political_group",
    "democrat": "political_group",
    "democrats": "political_group",
    # other / none
    "disability": "other",
    "disabled": "other",
    "elderly": "other",
    "age": "other",
    "economic": "other",
    "terrorism": "other",
    "terrorist": "other",
    "miscellaneous": "other",
    "minority": "other",
    "other": "other",
    "none": "none",
}


def normalize_label(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def majority_vote(items: Sequence[str]) -> Optional[str]:
    """Most common value; ties broken deterministically (alphabetical order)."""
    if not items:
        return None
    counts = Counter(items)
    top = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == top)
    return winners[0]


def map_hatexplain_hate(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(canonical, unmapped_raw)`` for a HateXplain hate label."""
    if raw is None:
        return None, None
    key = normalize_label(raw)
    if not key:
        return None, None
    canonical = HATEXPLAIN_HATE_MAP.get(key)
    return (canonical, None) if canonical else (None, raw)


def map_hatexplain_target(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(canonical, unmapped_raw)`` for a HateXplain target community."""
    if raw is None:
        return None, None
    key = normalize_label(raw)
    if not key:
        return None, None
    canonical = HATEXPLAIN_TARGET_MAP.get(key)
    return (canonical, None) if canonical else (None, raw)


# --- Counter Context -------------------------------------------------------------

#: Default mapping from the corpus label ids to internal labels.
#: The corpus of the NAACL 2022 paper "Hate Speech and Counter Speech
#: Detection: Conversational Context Does Matter" (Yu, Blanco & Hong) contains
#: three classes. VERIFY the exact numeric order against the paper before
#: productive training and override via ``configs/*.yaml`` if needed.
COUNTER_CONTEXT_DEFAULT_LABEL_MAP: Dict[str, str] = {
    "0": "hate_speech",
    "1": "counter_speech",
    "2": "neither",
}


def get_counter_context_label_map(
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    mapping = dict(COUNTER_CONTEXT_DEFAULT_LABEL_MAP)
    if overrides:
        mapping.update({str(k): str(v) for k, v in overrides.items()})
    return mapping


def map_counter_context_label(
    raw: Optional[str], mapping: Optional[Dict[str, str]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(canonical, unmapped_raw)`` for a Counter Context label id."""
    if raw is None:
        return None, None
    key = str(raw).strip()
    if not key:
        return None, None
    resolved = get_counter_context_label_map(mapping)
    canonical = resolved.get(key)
    return (canonical, None) if canonical else (None, raw)
