"""Download the raw datasets.

* **HateXplain** (primary): original corpus JSON from GitHub
  (https://github.com/hate-alert/HateXplain) — ``dataset.json`` plus
  ``post_id_divisions.json`` (the train/val/test split).
* **Counter Context** (context corpus): JSONL files from
  https://github.com/xinchenyu/counter_context — gold and silver splits.

Raw data is written to ``data/raw/<dataset>/`` and is git-ignored.

Usage::

    python scripts/download_data.py --dataset hatexplain
    python scripts/download_data.py --dataset counter_context
    python scripts/download_data.py --dataset all
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402

HATEXPLAIN_BASE = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data"
HATEXPLAIN_FILES = ["dataset.json", "post_id_divisions.json"]

COUNTER_CONTEXT_BASE = (
    "https://raw.githubusercontent.com/xinchenyu/counter_context/main/data"
)
COUNTER_CONTEXT_FILES = [
    "gold/train.jsonl",
    "gold/val.jsonl",
    "gold/test.jsonl",
    "silver/train.jsonl",
    "silver/val.jsonl",
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"  exists, skipping: {destination}")
        return
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=180) as response:
        data = response.read()
    destination.write_bytes(data)
    print(f"  saved {destination} ({len(data) / 1024:.0f} KiB)")


def download_hatexplain(raw_dir: Path) -> None:
    print("HateXplain ->", raw_dir / "hatexplain")
    for name in HATEXPLAIN_FILES:
        download(f"{HATEXPLAIN_BASE}/{name}", raw_dir / "hatexplain" / name)


def download_counter_context(raw_dir: Path) -> None:
    print("Counter Context ->", raw_dir / "counter_context")
    for name in COUNTER_CONTEXT_FILES:
        download(f"{COUNTER_CONTEXT_BASE}/{name}", raw_dir / "counter_context" / name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download raw datasets")
    parser.add_argument(
        "--dataset",
        choices=["hatexplain", "counter_context", "all"],
        default="all",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"hatexplain", "all"}:
        download_hatexplain(raw_dir)
    if args.dataset in {"counter_context", "all"}:
        download_counter_context(raw_dir)
    print("Done. Next: python scripts/preprocess.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
