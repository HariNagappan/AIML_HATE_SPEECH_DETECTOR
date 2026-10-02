"""Draw a random sample of comments for reason annotation.

The annotation workspace in the frontend (``/annotate``) loads the generated
JSON as its starting set. Labels exported from the workspace are merged back
into the same rows via ``scripts/import_reason_annotations.py`` (matching by
``id``), so the draw must come from the processed split you intend to train
on.

Example::

    python scripts/make_annotation_samples.py --n 400 \\
        --out ../frontend/public/annotation-samples.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import BACKEND_DIR  # noqa: E402
from app.datasets.unified import load_unified_jsonl  # noqa: E402


def _normalized(text: str) -> str:
    return " ".join(str(text).split())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build annotation samples JSON")
    parser.add_argument(
        "--input",
        default="data/processed/counter_context_train.jsonl",
        help="processed split to draw from (relative to backend/)",
    )
    parser.add_argument("--n", type=int, default=400, help="how many samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=280)
    parser.add_argument(
        "--out",
        default="../frontend/public/annotation-samples.json",
        help="output JSON (relative to backend/)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = BACKEND_DIR / input_path
    if not input_path.exists():
        print(f"error: input not found: {input_path}")
        return 1

    examples = load_unified_jsonl(input_path)

    seen: set[str] = set()
    candidates = []
    for example in examples:
        text = _normalized(example.current_text)
        if not (args.min_chars <= len(text) <= args.max_chars):
            continue
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        candidates.append(
            {
                "id": example.id,
                "text": example.current_text.strip(),
                **(
                    {"context": example.context_text.strip()}
                    if example.context_text and example.context_text.strip()
                    else {}
                ),
            }
        )

    if len(candidates) < args.n:
        print(f"error: only {len(candidates)} eligible comments for n={args.n}")
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(candidates, args.n)
    sample.sort(key=lambda item: item["id"])

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = BACKEND_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")

    with_context = sum(1 for item in sample if item.get("context"))
    print(
        f"sampled {len(sample)} comments from {input_path.name} "
        f"({len(candidates)} eligible, seed={args.seed}, {with_context} with context)"
    )
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
