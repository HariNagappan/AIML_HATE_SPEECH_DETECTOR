"""Export a serve-lean checkpoint (drops optimizer / scheduler state).

Full training checkpoints keep optimizer + scheduler state so training can
resume; the inference service never uses those tensors, and for the shipped
configurations they are roughly two thirds of the ≈1.29 GB file size.

Examples::

    # writes checkpoints/cc_context/best.serving.pt next to the source
    python scripts/export_serving_checkpoint.py \\
        --checkpoint checkpoints/cc_context/best.pt

    # explicit output path (overwrite with --force)
    python scripts/export_serving_checkpoint.py \\
        --checkpoint checkpoints/hx_full/best.pt \\
        --output checkpoints/hx_full/serving.pt

The exported file can be served directly:

    MODEL_CHECKPOINT=checkpoints/cc_context/best.serving.pt uvicorn app.main:app
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import BACKEND_DIR  # noqa: E402
from app.training.checkpoint_io import DROP_KEYS, export_serving_checkpoint  # noqa: E402


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BACKEND_DIR / path


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a serve-lean checkpoint (drops optimizer/scheduler state)"
    )
    parser.add_argument("--checkpoint", required=True, help="training checkpoint (.pt)")
    parser.add_argument(
        "--output",
        default=None,
        help="output path (default: <checkpoint stem>.serving.pt next to the source)",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite the output if it exists"
    )
    args = parser.parse_args()

    source = _resolve(args.checkpoint)
    destination = _resolve(args.output) if args.output else None

    try:
        info = export_serving_checkpoint(source, destination, force=args.force)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    saved = info["size_before"] - info["size_after"]
    saved_pct = (saved / info["size_before"] * 100) if info["size_before"] else 0.0
    print(f"source:  {info['source']} ({_mb(info['size_before'])})")
    print(f"serving: {info['destination']} ({_mb(info['size_after'])})")
    print(
        f"dropped: {', '.join(info['dropped_keys']) or '(none)'} "
        f"— saved {_mb(saved)} ({saved_pct:.1f}%)"
    )
    print(f"kept:    {', '.join(info['kept_keys'])}")
    print("export OK — serve it with MODEL_CHECKPOINT=<serving path> uvicorn app.main:app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
