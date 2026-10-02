"""Train a model.

Examples::

    python scripts/train.py --config baseline
    python scripts/train.py --config context --dataset counter_context
    python scripts/train.py --config cc_context --resume checkpoints/cc_context/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings, resolve_device  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.datasets.collator import build_label_maps  # noqa: E402
from app.datasets.tokenizer import TokenizerWrapper  # noqa: E402
from app.training.configs import TrainConfig  # noqa: E402
from app.training.trainer import Trainer, build_model, set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the context-aware hate speech model"
    )
    parser.add_argument(
        "--config",
        default="full",
        help="config name in configs/ (baseline | context | full | base)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="override dataset (hatexplain | counter_context)",
    )
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    overrides = {"dataset": args.dataset} if args.dataset else None
    config = TrainConfig.from_yaml(args.config, overrides=overrides)
    device = resolve_device(settings.device)
    set_seed(config.seed)

    print(
        f"Training '{config.config_name}': mode={config.mode} "
        f"dataset={config.dataset_name} device={device}"
    )

    train_examples = config.load_examples("train")
    val_examples = config.load_examples("val")
    print(f"Loaded {len(train_examples)} train / {len(val_examples)} val examples")

    labels = config.resolved_label_sets()
    label_maps = build_label_maps(labels["hate"], labels["target"], labels["reason"])
    tokenizer = TokenizerWrapper.from_pretrained(settings.encoder_name)
    model, model_kwargs = build_model(config, settings, device)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        val_examples=val_examples,
        config=config,
        device=device,
        label_maps=label_maps,
        model_kwargs=model_kwargs,
    )
    if args.resume:
        trainer.load_checkpoint(Path(args.resume), resume_training=True)

    result = trainer.train()
    print(
        "Training finished:",
        {
            key: result[key]
            for key in ("best_epoch", "best_metric", "trained_heads", "output_dir")
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
