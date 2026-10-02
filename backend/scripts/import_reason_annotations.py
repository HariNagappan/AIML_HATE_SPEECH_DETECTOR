"""Import custom reason annotations into training-ready processed files.

Examples::

    # standalone dataset (trains the reason head on its own)
    python scripts/import_reason_annotations.py \\
        --annotations data/reason_annotations.jsonl \\
        --output data/processed/reason_annotations.jsonl

    # merge into an existing split (keeps hate/evidence labels, adds reasons)
    python scripts/import_reason_annotations.py \\
        --annotations data/reason_annotations.jsonl \\
        --merge-into data/processed/counter_context_train.jsonl \\
        --output data/processed/counter_context_reason_train.jsonl

Format: JSONL (or CSV) with required fields ``text`` and ``reason``, plus
optional ``id``, ``context``, ``target``. Reasons are normalised to the
canonical categories (``"Incitement to violence"`` → ``incitement_to_violence``).

See also: README, "Supervising the reason head", and
``configs/cc_reason.yaml`` for a ready training configuration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import BACKEND_DIR  # noqa: E402
from app.datasets.reason_import import (  # noqa: E402
    ImportReport,
    StrictImportError,
    import_annotations,
    load_annotation_file,
)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BACKEND_DIR / path


def _print_report(report: ImportReport) -> None:
    print(f"source:  {report.source}")
    mode = report.mode + (
        f" (into {report.merge_into})" if report.merge_into else ""
    )
    print(f"mode:    {mode}")
    print(
        f"rows:    {report.total_rows} total · {report.valid_annotations} valid · "
        f"{len(report.invalid_issues)} invalid"
    )
    if report.by_reason:
        dist = ", ".join(f"{key}={value}" for key, value in sorted(report.by_reason.items()))
        print(f"reasons: {dist}")
    for issue in report.invalid_issues[:10]:
        print(f"  ! row {issue.row}: {issue.message}")
    if len(report.invalid_issues) > 10:
        print(f"  ... and {len(report.invalid_issues) - 10} more invalid rows")

    if report.mode == "merge":
        print(
            f"matched: {report.matched_annotations} annotations · "
            f"{report.matched_examples} examples updated · "
            f"{report.already_present} already had the same reason"
        )
        for issue in report.conflicts[:10]:
            print(f"  ! conflict row {issue.row}: {issue.message}")
        for issue in report.unmatched[:10]:
            print(f"  - unmatched row {issue.row}: {issue.message}")
        if len(report.unmatched) > 10:
            print(f"  ... and {len(report.unmatched) - 10} more unmatched")
        if report.appended:
            print(f"appended (--allow-unmatched): {report.appended}")

    if report.output_path:
        print(f"written: {report.written} examples -> {report.output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import custom reason annotations")
    parser.add_argument(
        "--annotations",
        required=True,
        help="JSONL or CSV annotation file (fields: text, reason, id?, context?, target?)",
    )
    parser.add_argument(
        "--merge-into",
        default=None,
        help="existing processed JSONL split to merge reason labels into",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output JSONL (default: data/processed/reason_annotations.jsonl)",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="append unmatched annotations as new reason-only examples",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail without writing when there are invalid rows or conflicts",
    )
    args = parser.parse_args()

    source = _resolve(args.annotations)
    merge_into = _resolve(args.merge_into) if args.merge_into else None
    output = (
        _resolve(args.output)
        if args.output
        else BACKEND_DIR / "data" / "processed" / "reason_annotations.jsonl"
    )

    try:
        load = load_annotation_file(source)
        report = import_annotations(
            load,
            source=source,
            merge_into=merge_into,
            output_path=output,
            allow_unmatched=args.allow_unmatched,
            strict=args.strict,
        )
    except StrictImportError as exc:
        print("strict mode: not writing — resolve the reported issues first")
        _print_report(exc.report)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
