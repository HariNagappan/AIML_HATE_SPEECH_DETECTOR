"""Import custom reason annotations into the project's unified JSONL format.

Why this exists
---------------
The reason head is trained **only on real annotations** (see README,
"Supervising the reason head"). The standard splits in this repository
(HateXplain, Counter Context) contain no reason labels, so this module lets
you plug in your own annotation file and produce training-ready data:

* **standalone** — every annotation becomes a ``UnifiedExample`` with
  ``reason_label`` set (backs a reason-only training run), or
* **merge** — annotation labels are attached to the examples of an existing
  processed split (matched by ``id`` first, then by normalized text), so a
  single run can train the hate/evidence heads and the reason head together.

Annotation file format (JSONL; CSV with the same columns also accepted)::

    {"text": "Those people are disgusting and should leave.",
     "reason": "exclusion",
     "id": "optional-match-id",
     "context": "optional previous message",
     "target": "optional target label"}

``text`` and ``reason`` are required. ``reason`` is normalised
(case/space/hyphen insensitive, e.g. ``"Incitement to violence"`` becomes
``incitement_to_violence``) and must be one of the canonical categories.

Nothing is fabricated or guessed: invalid rows are skipped and reported,
merge conflicts are never overwritten, and unmatched annotations are only
added to the dataset when explicitly requested.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.datasets.label_mapping import CANONICAL_REASON
from app.datasets.unified import (
    UnifiedExample,
    load_unified_jsonl,
    write_unified_jsonl,
)

_CANONICAL_REASON_SET = frozenset(CANONICAL_REASON)


# --- normalisation ------------------------------------------------------------


def normalize_reason(value: Any) -> Optional[str]:
    """Normalise a raw reason string to its canonical form, or ``None``.

    Case, surrounding whitespace, hyphens and internal spaces are ignored:
    ``"Incitement to violence"`` → ``incitement_to_violence``.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = text.replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text if text in _CANONICAL_REASON_SET else None


def _clean_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_label(value: Any) -> Optional[str]:
    text = _clean_optional(value)
    if text is None:
        return None
    text = text.lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _normalized_text(text: str) -> str:
    return " ".join(str(text).split())


# --- data containers ------------------------------------------------------------


@dataclass
class ImportIssue:
    """One problem (or noteworthy event) tied to an annotation row."""

    row: int
    message: str


@dataclass
class Annotation:
    """A single validated annotation row."""

    row: int
    text: str
    reason: str
    id: Optional[str] = None
    context: Optional[str] = None
    target: Optional[str] = None


@dataclass
class LoadResult:
    total_rows: int
    annotations: List[Annotation]
    issues: List[ImportIssue]


@dataclass
class ImportReport:
    source: str
    mode: str  # "standalone" | "merge"
    merge_into: Optional[str] = None
    total_rows: int = 0
    valid_annotations: int = 0
    invalid_issues: List[ImportIssue] = field(default_factory=list)
    by_reason: Dict[str, int] = field(default_factory=dict)
    matched_annotations: int = 0
    matched_examples: int = 0
    already_present: int = 0
    conflicts: List[ImportIssue] = field(default_factory=list)
    unmatched: List[ImportIssue] = field(default_factory=list)
    appended: int = 0
    written: int = 0
    output_path: Optional[str] = None


class StrictImportError(Exception):
    """Raised in strict mode when invalid rows or conflicts exist."""

    def __init__(self, report: ImportReport) -> None:
        super().__init__("strict import failed")
        self.report = report


# --- loading --------------------------------------------------------------------


def _annotation_from_mapping(
    row: int, raw: Dict[str, Any]
) -> "Annotation | ImportIssue":
    text = _clean_optional(raw.get("text"))
    if text is None:
        return ImportIssue(row, "missing 'text'")
    raw_reason = raw.get("reason")
    reason = normalize_reason(raw_reason)
    if reason is None:
        if _clean_optional(raw_reason) is None:
            return ImportIssue(row, "missing 'reason'")
        return ImportIssue(row, f"unknown reason {_clean_optional(raw_reason)!r}")
    return Annotation(
        row=row,
        text=text,
        reason=reason,
        id=_clean_optional(raw.get("id")),
        context=_clean_optional(raw.get("context")),
        target=_normalize_label(raw.get("target")),
    )


def load_annotation_file(path: "str | Path") -> LoadResult:
    """Load a JSONL or CSV annotation file.

    Required fields per row: ``text``, ``reason``.
    Optional fields: ``id``, ``context``, ``target``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annotation file not found: {path}")

    total_rows = 0
    annotations: List[Annotation] = []
    issues: List[ImportIssue] = []

    def _consume(row_number: int, raw: Dict[str, Any]) -> None:
        nonlocal total_rows
        total_rows += 1
        item = _annotation_from_mapping(row_number, raw)
        if isinstance(item, ImportIssue):
            issues.append(item)
        else:
            annotations.append(item)

    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                cleaned = {
                    str(key): value
                    for key, value in raw.items()
                    if key is not None
                }
                _consume(reader.line_num, cleaned)
    else:
        with open(path, "r", encoding="utf-8-sig") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    total_rows += 1
                    issues.append(ImportIssue(lineno, f"invalid JSON: {exc.msg}"))
                    continue
                if not isinstance(raw, dict):
                    total_rows += 1
                    issues.append(ImportIssue(lineno, "line is not a JSON object"))
                    continue
                _consume(lineno, raw)

    return LoadResult(total_rows=total_rows, annotations=annotations, issues=issues)


# --- import / merge -------------------------------------------------------------


def _standalone_example(ann: Annotation, source_name: str) -> UnifiedExample:
    return UnifiedExample(
        id=ann.id or f"reason-{ann.row}",
        current_text=ann.text,
        context_text=ann.context,
        reason_label=ann.reason,
        target_label=ann.target,
        metadata={
            "source": "custom-reason-annotations",
            "source_file": source_name,
        },
    )


def import_annotations(
    load: LoadResult,
    *,
    source: "str | Path",
    output_path: "str | Path",
    merge_into: "str | Path | None" = None,
    allow_unmatched: bool = False,
    strict: bool = False,
) -> ImportReport:
    """Write annotations to a processed JSONL file (standalone or merged).

    In merge mode, annotations are matched against the base split by ``id``
    first and then by whitespace-normalized ``current_text``; a label is only
    set where ``reason_label`` is still ``None`` — existing values are never
    overwritten (conflicts are reported instead).
    """
    source_name = Path(source).name
    report = ImportReport(
        source=str(source),
        mode="merge" if merge_into is not None else "standalone",
        merge_into=str(merge_into) if merge_into is not None else None,
        total_rows=load.total_rows,
        valid_annotations=len(load.annotations),
        invalid_issues=list(load.issues),
    )
    for ann in load.annotations:
        report.by_reason[ann.reason] = report.by_reason.get(ann.reason, 0) + 1

    if not load.annotations:
        raise ValueError(
            f"no valid annotations in {source} "
            f"({len(load.issues)} invalid rows); nothing to import"
        )

    if merge_into is None:
        examples = [_standalone_example(a, source_name) for a in load.annotations]
    else:
        base_path = Path(merge_into)
        if not base_path.exists():
            raise FileNotFoundError(f"base processed file not found: {base_path}")
        examples = load_unified_jsonl(base_path)

        by_id: Dict[str, List[UnifiedExample]] = {}
        by_text: Dict[str, List[UnifiedExample]] = {}
        for example in examples:
            if example.id:
                by_id.setdefault(str(example.id), []).append(example)
            by_text.setdefault(_normalized_text(example.current_text), []).append(example)

        for ann in load.annotations:
            matches = by_id.get(ann.id, []) if ann.id else []
            if not matches:
                matches = by_text.get(_normalized_text(ann.text), [])
            if not matches:
                report.unmatched.append(ImportIssue(ann.row, ann.text[:80]))
                if allow_unmatched:
                    examples.append(_standalone_example(ann, source_name))
                    report.appended += 1
                continue
            report.matched_annotations += 1
            for example in matches:
                if example.reason_label is None:
                    example.reason_label = ann.reason
                    example.metadata["reason_source"] = source_name
                    report.matched_examples += 1
                elif normalize_reason(example.reason_label) == ann.reason:
                    report.already_present += 1
                else:
                    report.conflicts.append(
                        ImportIssue(
                            ann.row,
                            f"existing '{example.reason_label}' != annotation "
                            f"'{ann.reason}' (example id={example.id})",
                        )
                    )

    if strict and (report.invalid_issues or report.conflicts):
        raise StrictImportError(report)

    output_path = Path(output_path)
    report.written = write_unified_jsonl(examples, output_path)
    report.output_path = str(output_path)
    return report
