"""Inspect a dataset file: field names, types and label distributions.

Use this to VERIFY actual dataset schemas before trusting the adapters.

Usage::

    python scripts/inspect_dataset.py --path data/raw/counter_context/gold/train.jsonl
    python scripts/inspect_dataset.py --path data/raw/hatexplain/dataset.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, List


def load_records(path: Path) -> List[Any]:
    if path.suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("data", "records", "examples"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    return data


def describe(value: Any) -> str:
    if isinstance(value, list):
        inner = describe(value[0]) if value else "?"
        return f"list[{len(value)}] of {inner}"
    if isinstance(value, dict):
        return "dict{" + ", ".join(str(k) for k in value.keys()) + "}"
    return type(value).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a dataset file")
    parser.add_argument("--path", required=True)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    records = load_records(path)
    print(f"{path}: {len(records)} records")

    sample = records[0]
    print("\nField types (first record):")
    if isinstance(sample, dict):
        for key, value in sample.items():
            print(f"  {key}: {describe(value)}")

    # label-like distributions
    label_keys = [key for key in ("label", "labels", "hate_label", "target", "targets", "annotators") if isinstance(sample, dict) and key in sample]
    for key in label_keys:
        values: List[str] = []
        for record in records:
            value = record.get(key) if isinstance(record, dict) else None
            if key == "annotators" and isinstance(value, list):
                values.extend(str((item or {}).get("label")) for item in value)
            elif isinstance(value, list):
                values.extend(str(item) for item in value)
            elif value is not None:
                values.append(str(value))
        counts = Counter(values)
        print(f"\nValue distribution for '{key}' (top {min(12, len(counts))}):")
        for value, count in counts.most_common(12):
            print(f"  {value}: {count}")

    # annotator sub-structure for HateXplain-style records
    if isinstance(sample, dict) and isinstance(sample.get("annotators"), list) and sample["annotators"]:
        print("\nFirst annotator entry:", json.dumps(sample["annotators"][0], ensure_ascii=False)[:400])
    if isinstance(sample, dict) and "rationales" in sample:
        rationales = sample.get("rationales") or []
        print(f"\nRationales: {len(rationales)} annotator lists, first length = {len(rationales[0]) if rationales else 0}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
