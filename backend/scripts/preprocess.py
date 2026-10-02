"""Convert raw datasets into processed UnifiedExample JSONL files.

Writes (under ``data/processed/``):

* ``hatexplain_{train,val,test}.jsonl`` — split via ``post_id_divisions.json``
* ``counter_context_{train,val,test}.jsonl`` — gold splits
* ``counter_context_silver_{train,val}.jsonl`` — silver split (extra training
  data; not mixed into the gold files)

Usage::

    python scripts/preprocess.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.datasets.counter_context import CounterContextDataset  # noqa: E402
from app.datasets.hatexplain import HateXplainDataset  # noqa: E402
from app.datasets.unified import write_unified_jsonl  # noqa: E402


def _split_ids(divisions_path: Path) -> Dict[str, List[str]]:
    with open(divisions_path, "r", encoding="utf-8") as fh:
        divisions = json.load(fh)
    mapping: Dict[str, List[str]] = {}
    for key, ids in divisions.items():
        normalized = {"validation": "val"}.get(str(key).lower(), str(key).lower())
        mapping[normalized] = [str(item) for item in ids]
    return mapping


def preprocess_hatexplain(settings: Settings) -> None:
    raw_dir = settings.data_dir / "raw" / "hatexplain"
    dataset_path = raw_dir / "dataset.json"
    if not dataset_path.exists():
        print("skip hatexplain: raw files missing (run scripts/download_data.py first)")
        return

    divisions_path = raw_dir / "post_id_divisions.json"
    processed_dir = settings.data_dir / "processed"
    if divisions_path.exists():
        splits = _split_ids(divisions_path)
        for split_name in ("train", "val", "test"):
            ids = splits.get(split_name)
            if not ids:
                print(f"  note: no ids found for split '{split_name}'")
                continue
            dataset = HateXplainDataset.from_json_file(
                dataset_path, split=split_name, keep_ids=ids
            )
            count = write_unified_jsonl(
                dataset.to_unified(), processed_dir / f"hatexplain_{split_name}.jsonl"
            )
            print(f"hatexplain {split_name}: {count} examples")
    else:
        dataset = HateXplainDataset.from_json_file(dataset_path, split="unspecified")
        count = write_unified_jsonl(
            dataset.to_unified(), processed_dir / "hatexplain_unspecified.jsonl"
        )
        print(f"hatexplain (no split file): {count} examples")


def preprocess_counter_context(settings: Settings) -> None:
    raw_dir = settings.data_dir / "raw" / "counter_context"
    processed_dir = settings.data_dir / "processed"

    for split in ("train", "val", "test"):
        path = raw_dir / "gold" / f"{split}.jsonl"
        if not path.exists():
            print(f"skip counter_context gold/{split}")
            continue
        dataset = CounterContextDataset.from_jsonl(path, split=split)
        count = write_unified_jsonl(
            dataset.to_unified(), processed_dir / f"counter_context_{split}.jsonl"
        )
        print(f"counter_context gold/{split}: {count} examples")

    for split in ("train", "val"):
        path = raw_dir / "silver" / f"{split}.jsonl"
        if not path.exists():
            continue
        dataset = CounterContextDataset.from_jsonl(path, split=f"silver-{split}")
        count = write_unified_jsonl(
            dataset.to_unified(),
            processed_dir / f"counter_context_silver_{split}.jsonl",
        )
        print(f"counter_context silver/{split}: {count} examples")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess raw datasets")
    parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)
    (settings.data_dir / "processed").mkdir(parents=True, exist_ok=True)

    preprocess_hatexplain(settings)
    preprocess_counter_context(settings)
    print("Done. Next: python scripts/train.py --config context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
