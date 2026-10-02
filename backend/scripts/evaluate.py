"""Evaluate a trained checkpoint on the test split.

Examples::

    python scripts/evaluate.py --checkpoint checkpoints/full/best.pt
    python scripts/evaluate.py --config context --dataset counter_context
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.training.configs import TrainConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path (default: best.pt of the config)")
    parser.add_argument("--config", default="full")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max-evidence-examples", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    overrides = {"dataset": args.dataset} if args.dataset else None
    config = TrainConfig.from_yaml(args.config, overrides=overrides)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        best = config.resolved_output_dir() / "best.pt"
        checkpoint = best if best.exists() else config.resolved_output_dir() / "last.pt"
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return 1

    from app.evaluation.ablation import evaluate_checkpoint, evaluate_evidence

    test_examples = config.load_examples("test")
    metrics = evaluate_checkpoint(str(checkpoint), test_examples, config, settings)
    print(json.dumps(metrics, indent=2))

    evidence = evaluate_evidence(
        str(checkpoint),
        test_examples,
        config,
        settings,
        max_examples=args.max_evidence_examples,
    )
    if evidence:
        print("evidence metrics:", json.dumps(evidence, indent=2))
    else:
        print("evidence metrics: not available (no rationales / no trained hate head)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
