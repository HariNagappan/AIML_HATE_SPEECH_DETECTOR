"""Tests for the custom reason annotation importer (app/datasets/reason_import)."""

from __future__ import annotations

import json

import pytest

from app.datasets.reason_import import (
    StrictImportError,
    import_annotations,
    load_annotation_file,
    normalize_reason,
)
from app.datasets.unified import UnifiedExample, load_unified_jsonl, write_unified_jsonl


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_base(path):
    """Base processed split for merge tests (ids cc-1..cc-4)."""
    examples = [
        UnifiedExample(
            id="cc-1",
            current_text="They are ruining everything.",
            context_text="We were talking about immigrants.",
            hate_label="hate_speech",
        ),
        UnifiedExample(id="cc-2", current_text="I love sunny days.", hate_label="neither"),
        # whitespace variant of cc-1: must be matched by normalized text too
        UnifiedExample(
            id="cc-3", current_text="They  are   ruining everything.", hate_label="hate_speech"
        ),
        # already labelled: conflict tests target this one
        UnifiedExample(
            id="cc-4",
            current_text="Kick them all out.",
            hate_label="hate_speech",
            reason_label="exclusion",
        ),
    ]
    write_unified_jsonl(examples, path)
    return path


# --- normalisation --------------------------------------------------------------


class TestNormalizeReason:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Exclusion", "exclusion"),
            ("  INSULT ", "insult"),
            ("Incitement to violence", "incitement_to_violence"),
            ("incitement-to-violence", "incitement_to_violence"),
            ("negative stereotyping", "negative_stereotyping"),
            ("Negative-Stereotyping", "negative_stereotyping"),
            ("dehumanization", "dehumanization"),
            ("threat", "threat"),
            ("discrimination", "discrimination"),
            ("other", "other"),
            ("not_a_reason", None),
            ("banana", None),
            ("", None),
            (None, None),
        ],
    )
    def test_normalisation(self, raw, expected):
        assert normalize_reason(raw) == expected


# --- loading --------------------------------------------------------------------


class TestLoadAnnotationFile:
    def test_jsonl_valid_and_invalid(self, tmp_path):
        path = _write_jsonl(
            tmp_path / "a.jsonl",
            [
                {"text": "Those people are disgusting.", "reason": "Exclusion"},
                {"text": "missing reason"},
                {"reason": "insult"},
                {"text": "unknown", "reason": "banana"},
            ],
        )
        load = load_annotation_file(path)
        assert load.total_rows == 4
        assert len(load.annotations) == 1
        assert load.annotations[0].reason == "exclusion"
        assert load.annotations[0].row == 1
        assert [issue.row for issue in load.issues] == [2, 3, 4]
        assert "missing 'reason'" in load.issues[0].message
        assert "missing 'text'" in load.issues[1].message
        assert "banana" in load.issues[2].message

    def test_csv(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text(
            'text,reason,context\n'
            '"You are an idiot",Insult,"Stop posting."\n'
            '"We must act now","Incitement to violence",\n',
            encoding="utf-8",
        )
        load = load_annotation_file(path)
        assert load.total_rows == 2
        assert [a.reason for a in load.annotations] == ["insult", "incitement_to_violence"]
        assert load.annotations[0].context == "Stop posting."
        assert load.annotations[1].context is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_annotation_file(tmp_path / "does-not-exist.jsonl")


# --- standalone -----------------------------------------------------------------


class TestStandaloneImport:
    def test_writes_unified_examples(self, tmp_path):
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl",
            [
                {
                    "text": "Those people are disgusting and should leave.",
                    "reason": "exclusion",
                    "context": "Why are you talking about immigrants?",
                    "target": "nationality",
                },
                {"text": "You are such an idiot.", "reason": "threat", "id": "custom-1"},
            ],
        )
        load = load_annotation_file(ann_path)
        out = tmp_path / "out.jsonl"
        report = import_annotations(load, source=ann_path, output_path=out)

        assert report.mode == "standalone"
        assert report.written == 2
        assert report.by_reason == {"exclusion": 1, "threat": 1}

        examples = load_unified_jsonl(out)
        assert examples[0].id == "reason-1"
        assert examples[0].reason_label == "exclusion"
        assert examples[0].context_text == "Why are you talking about immigrants?"
        assert examples[0].target_label == "nationality"
        assert examples[0].hate_label is None  # nothing fabricated
        assert examples[1].id == "custom-1"
        assert examples[1].reason_label == "threat"

    def test_no_valid_rows_raises_without_writing(self, tmp_path):
        ann_path = _write_jsonl(tmp_path / "bad.jsonl", [{"text": "x", "reason": "nope"}])
        load = load_annotation_file(ann_path)
        out = tmp_path / "out.jsonl"
        with pytest.raises(ValueError):
            import_annotations(load, source=ann_path, output_path=out)
        assert not out.exists()


# --- merge ----------------------------------------------------------------------


class TestMergeImport:
    def test_merge_by_id(self, tmp_path):
        base = _write_base(tmp_path / "base.jsonl")
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl",
            [{"id": "cc-2", "text": "totally different text", "reason": "insult"}],
        )
        out = tmp_path / "merged.jsonl"
        report = import_annotations(
            load_annotation_file(ann_path),
            source=ann_path,
            merge_into=base,
            output_path=out,
        )
        assert report.matched_annotations == 1
        assert report.matched_examples == 1
        merged = {e.id: e for e in load_unified_jsonl(out)}
        assert merged["cc-2"].reason_label == "insult"
        assert merged["cc-2"].metadata.get("reason_source") == "ann.jsonl"
        assert report.written == 4  # base examples only; nothing appended

    def test_merge_by_normalized_text_updates_all_duplicates(self, tmp_path):
        base = _write_base(tmp_path / "base.jsonl")
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl",
            [{"text": "They   are ruining everything.", "reason": "insult"}],
        )
        out = tmp_path / "merged.jsonl"
        report = import_annotations(
            load_annotation_file(ann_path),
            source=ann_path,
            merge_into=base,
            output_path=out,
        )
        assert report.matched_annotations == 1
        assert report.matched_examples == 2  # cc-1 and its whitespace variant cc-3
        merged = {e.id: e for e in load_unified_jsonl(out)}
        assert merged["cc-1"].reason_label == "insult"
        assert merged["cc-3"].reason_label == "insult"

    def test_conflict_is_reported_and_not_overwritten(self, tmp_path):
        base = _write_base(tmp_path / "base.jsonl")
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl",
            [{"text": "Kick them all out.", "reason": "threat"}],
        )
        out = tmp_path / "merged.jsonl"
        report = import_annotations(
            load_annotation_file(ann_path),
            source=ann_path,
            merge_into=base,
            output_path=out,
        )
        assert len(report.conflicts) == 1
        merged = {e.id: e for e in load_unified_jsonl(out)}
        assert merged["cc-4"].reason_label == "exclusion"  # untouched

    def test_same_reason_counts_as_already_present(self, tmp_path):
        base = _write_base(tmp_path / "base.jsonl")
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl",
            [{"id": "cc-4", "text": "Kick them all out.", "reason": "Exclusion"}],
        )
        out = tmp_path / "merged.jsonl"
        report = import_annotations(
            load_annotation_file(ann_path),
            source=ann_path,
            merge_into=base,
            output_path=out,
        )
        assert report.already_present == 1
        assert report.conflicts == []

    def test_unmatched_not_appended_by_default(self, tmp_path):
        base = _write_base(tmp_path / "base.jsonl")
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl",
            [{"text": "not in the base split", "reason": "other"}],
        )
        out = tmp_path / "merged.jsonl"
        report = import_annotations(
            load_annotation_file(ann_path),
            source=ann_path,
            merge_into=base,
            output_path=out,
        )
        assert len(report.unmatched) == 1
        assert report.appended == 0
        assert report.written == 4

    def test_allow_unmatched_appends_reason_only_examples(self, tmp_path):
        base = _write_base(tmp_path / "base.jsonl")
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl",
            [{"text": "not in the base split", "reason": "other"}],
        )
        out = tmp_path / "merged.jsonl"
        report = import_annotations(
            load_annotation_file(ann_path),
            source=ann_path,
            merge_into=base,
            output_path=out,
            allow_unmatched=True,
        )
        assert report.appended == 1
        assert report.written == 5
        merged = load_unified_jsonl(out)
        assert merged[-1].reason_label == "other"
        assert merged[-1].hate_label is None

    def test_missing_base_file_raises(self, tmp_path):
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl", [{"text": "x", "reason": "threat"}]
        )
        with pytest.raises(FileNotFoundError):
            import_annotations(
                load_annotation_file(ann_path),
                source=ann_path,
                merge_into=tmp_path / "nope.jsonl",
                output_path=tmp_path / "out.jsonl",
            )


# --- strict mode ----------------------------------------------------------------


class TestStrictMode:
    def test_strict_blocks_write_on_invalid_rows(self, tmp_path):
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl",
            [
                {"text": "valid one", "reason": "threat"},
                {"text": "bad one", "reason": "banana"},
            ],
        )
        load = load_annotation_file(ann_path)
        out = tmp_path / "out.jsonl"
        with pytest.raises(StrictImportError) as excinfo:
            import_annotations(
                load, source=ann_path, output_path=out, strict=True
            )
        report = excinfo.value.report
        assert len(report.invalid_issues) == 1
        assert report.output_path is None
        assert not out.exists()

    def test_strict_passes_when_clean(self, tmp_path):
        ann_path = _write_jsonl(
            tmp_path / "ann.jsonl", [{"text": "valid one", "reason": "threat"}]
        )
        out = tmp_path / "out.jsonl"
        report = import_annotations(
            load_annotation_file(ann_path), source=ann_path, output_path=out, strict=True
        )
        assert report.written == 1
        assert out.exists()
