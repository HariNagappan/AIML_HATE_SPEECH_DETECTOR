"""Explain a single comment (structured, evidence-based) via a trained checkpoint.

Example::

    python scripts/explain.py --checkpoint checkpoints/cc_context/best.pt \\
        --text "They should all be kicked out." \\
        --context "Those immigrants are ruining everything."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured explanation for one comment")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path (default: auto-resolved from checkpoints/)")
    parser.add_argument("--text", required=True, help="current comment")
    parser.add_argument("--context", default=None, help="previous comment (optional)")
    parser.add_argument("--target", default="hate", choices=["hate", "target", "reason"])
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    # None -> the inference service auto-resolves the checkpoint
    # (checkpoints/default.json, then cc_context / hx_full best files).
    checkpoint = args.checkpoint

    from app.services.inference import InferenceService

    service_settings = settings.model_copy(update={"checkpoint_path": checkpoint})
    service = InferenceService(service_settings)

    print("== structured prediction ==")
    print(json.dumps(service.predict(args.text, args.context), indent=2))
    print("== attribution details ==")
    print(
        json.dumps(
            service.explain(args.text, args.context, target=args.target, top_k=args.top_k),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
