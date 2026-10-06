"""Evaluate a trained checkpoint on the test split.

Examples::

    python scripts/evaluate.py --checkpoint checkpoints/cc_context/best.pt
    python scripts/evaluate.py --config context --dataset counter_context

The script also writes a machine-readable copy of the results next to the
evaluated checkpoint (``<checkpoint-stem>.metrics.json``; override with
``--output``). ``GET /api/v1/model/info`` picks that file up automatically and
the frontend renders it in the "Evaluation metrics" card.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "where to write the metrics JSON "
            "(default: <checkpoint-stem>.metrics.json next to the checkpoint)"
        ),
    )
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

    # Persist machine-readable results for the API/frontend. The UI never
    # hardcodes scores - it renders exactly what a real evaluation produced.
    payload = {
        "checkpoint": str(checkpoint),
        "config": config.config_name,
        "dataset": config.dataset_name,
        "split": "test",
        "num_examples": metrics.get("num_examples", len(test_examples)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "hate": metrics.get("hate"),
        "target": metrics.get("target"),
        "evidence": evidence,
    }
    output_path = (
        Path(args.output)
        if args.output
        else checkpoint.with_name(checkpoint.stem + ".metrics.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"Metrics written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
