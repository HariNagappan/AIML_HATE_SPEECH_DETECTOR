"""Dataset adapter tests: HateXplain parsing, Counter Context parsing,
label mapping, missing labels (all with small synthetic fixtures — offline)."""

from __future__ import annotations

import json

from app.datasets.counter_context import (
    CounterContextDataset,
    build_contrastive_pairs,
)
from app.datasets.hatexplain import HateXplainDataset, merge_rationales
from app.datasets.label_mapping import (
    majority_vote,
    map_counter_context_label,
    map_hatexplain_target,
)
from app.datasets.unified import (
    UnifiedExample,
    load_unified_jsonl,
    write_unified_jsonl,
)


def _hatexplain_record():
    return {
        "id": 7,
        "post_tokens": ["Those", "immigrants", "are", "disgusting"],
        "annotators": [
            {"label": "hatespeech", "target": ["Immigrants"], "annotator_id": 1},
            {"label": "hatespeech", "target": ["Refugee"], "annotator_id": 2},
            {"label": "offensive", "target": [], "annotator_id": 3},
        ],
        "rationales": [[0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1]],
        "source": "twitter",
    }


def test_hatexplain_majority_targets_rationales():
    dataset = HateXplainDataset([_hatexplain_record()], split="train")
    (example,) = dataset.to_unified()

    assert example.hate_label == "hate"  # 2/3 annotators
    assert example.target_label == "nationality"  # Immigrants + Refugee → nationality
    assert example.reason_label is None  # never invented
    assert example.context_text is None  # HateXplain has no context
    assert example.rationale_labels is not None
    assert len(example.rationale_labels) == 4
    # merged from majority-label annotators: positions 1 and 3 selected
    assert example.rationale_labels == [0, 1, 0, 1]
    assert example.metadata["num_annotators"] == 3


def test_hatexplain_unmapped_targets_preserved():
    record = _hatexplain_record()
    record["annotators"][0]["target"] = ["Blernsday"]
    record["annotators"][1]["target"] = ["Blernsday"]
    record["annotators"][2]["target"] = ["Blernsday"]
    dataset = HateXplainDataset([record], split="train")
    (example,) = dataset.to_unified()
    assert example.target_label is None
    assert example.metadata.get("unmapped_targets") == ["Blernsday"]


def test_counter_context_maps_fields_correctly(tmp_path):
    # NOTE the corpus naming quirk: 'target' is the CURRENT comment,
    # 'context' is the PREVIOUS comment.
    records = [
        {
            "idx": 0,
            "label": "1",
            "context": "The UK is fucked.",
            "target": ">The UK world is fucked FTFY",
        },
        {
            "idx": 1,
            "label": "2",
            "context": "Listen to this wisdom.",
            "target": "Where did you get that?",
        },
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    dataset = CounterContextDataset.from_jsonl(path, split="train")
    examples = dataset.to_unified()

    assert examples[0].current_text == ">The UK world is fucked FTFY"
    assert examples[0].context_text == "The UK is fucked."
    assert examples[0].hate_label == "neither"  # verified map: "1" → neutral
    assert examples[1].hate_label == "counter_speech"  # verified map: "2" → counter
    assert examples[0].target_label is None
    assert examples[0].rationale_labels is None


def test_counter_context_custom_label_map(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"idx": 0, "label": "1", "context": "c", "target": "t"}),
        encoding="utf-8",
    )
    dataset = CounterContextDataset.from_jsonl(
        path, split="train", label_map={"1": "custom_class"}
    )
    (example,) = dataset.to_unified()
    assert example.hate_label == "custom_class"


def test_counter_context_unmapped_label_recorded(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"idx": 0, "label": "9", "context": "c", "target": "t"}),
        encoding="utf-8",
    )
    dataset = CounterContextDataset.from_jsonl(path, split="train")
    (example,) = dataset.to_unified()
    assert example.hate_label is None
    assert example.metadata.get("unmapped_label") == "9"


def test_contrastive_pairs_built_from_contexts():
    examples = [
        UnifiedExample(id="a", current_text="c1", context_text="ctx-1"),
        UnifiedExample(id="b", current_text="c2", context_text="ctx-2"),
        UnifiedExample(id="c", current_text="c3", context_text="ctx-3"),
        UnifiedExample(id="d", current_text="c4", context_text=None),
    ]
    pairs = build_contrastive_pairs(examples, seed=7)
    assert len(pairs) == 3  # example without context skipped
    for pair in pairs:
        assert pair["negative_context"] != pair["positive_context"]
        assert pair["negative_context"] in {"ctx-1", "ctx-2", "ctx-3"}


def test_majority_vote_tie_is_deterministic():
    assert majority_vote(["b", "a"]) == "a"
    assert majority_vote(["a", "a", "b"]) == "a"
    assert majority_vote([]) is None


def test_merge_rationales_flags():
    merged = merge_rationales([[1, 1, 0], [1, 0, 0]], use_flags=[True, True])
    assert merged == [1, 1, 0]  # position 1 is a tie (half) → 1
    assert merge_rationales([], []) is None


def test_label_mapping_helpers():
    assert map_hatexplain_target("Homosexual") == ("sexual_orientation", None)
    assert map_hatexplain_target("Heterosexual") == ("sexual_orientation", None)
    assert map_hatexplain_target("Buddhism") == ("religion", None)
    assert map_hatexplain_target("XYZ-Nonsense") == (None, "XYZ-Nonsense")
    assert map_counter_context_label("0") == ("hate_speech", None)
    assert map_counter_context_label("1") == ("neither", None)
    assert map_counter_context_label("2") == ("counter_speech", None)


def test_unified_roundtrip(tmp_path):
    example = UnifiedExample(
        id="x1",
        current_text="hello there",
        context_text="previous",
        hate_label="normal",
        rationale_labels=[1, 0],
    )
    path = tmp_path / "examples.jsonl"
    assert write_unified_jsonl([example], path) == 1
    loaded = load_unified_jsonl(path)
    assert loaded[0].current_text == "hello there"
    assert loaded[0].rationale_labels == [1, 0]
    assert loaded[0].has_context()


def test_hatexplain_dict_keyed_file(tmp_path):
    """The original corpus stores records as a post_id -> record mapping."""
    payload = {
        "abc_twitter": {
            "post_id": "abc_twitter",
            "post_tokens": ["hi", "there"],
            "annotators": [{"label": "normal", "target": ["None"]}],
            "rationales": [],
        }
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = HateXplainDataset.from_json_file(
        path, split="train", keep_ids=["abc_twitter"]
    )
    (example,) = dataset.to_unified()
    assert example.id == "hx-abc_twitter"
    assert example.hate_label == "normal"
    assert example.target_label == "none"
    assert example.rationale_labels is None
