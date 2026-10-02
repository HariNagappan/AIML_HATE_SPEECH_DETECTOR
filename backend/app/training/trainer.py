"""Training loop: multi-task objective, AMP, gradient accumulation,
checkpointing (save / load / resume), early stopping, reproducibility.

The trainer is independent from FastAPI: inference rebuilds the model from a
saved checkpoint via :mod:`app.services.inference`.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.config import get_settings
from app.core.logging import get_logger
from app.datasets.collator import (
    ContrastivePairCollator,
    LabelMaps,
    MultitaskCollator,
)
from app.datasets.tokenizer import TokenizerWrapper
from app.datasets.unified import UnifiedExample
from app.training.configs import TrainConfig
from app.training.contrastive_loss import (
    build_contrastive_loss,
    contrastive_loss_from_batch,
)
from app.training.losses import LossWeights, combine_losses, compute_task_losses

logger = get_logger("training.trainer")

CHECKPOINT_FORMAT_VERSION = 1
HEAD_NAMES = ("hate", "target", "reason", "evidence")


def set_seed(seed: int) -> None:
    """Reproducibility: python / numpy / torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_grad_scaler(enabled: bool):
    """Create an AMP GradScaler compatible with the installed torch version."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _linear_warmup_decay(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    remaining = max(0, total_steps - step)
    denominator = max(1, total_steps - warmup_steps)
    return max(0.0, min(1.0, remaining / denominator))


def build_model(
    config: TrainConfig, settings=None, device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build the full/baseline model for a training run + its rebuild kwargs."""
    from app.models.baseline import BaselineModel
    from app.models.bert_encoder import BertEncoder
    from app.models.full_model import FullModel

    settings = settings or get_settings()
    labels = config.resolved_label_sets()

    random_init = str(settings.init_backbone).lower() == "random"
    if random_init:
        encoder = BertEncoder.random_init(
            freeze_mode=config.freeze_mode, frozen_layers=config.frozen_layers
        )
    else:
        encoder = BertEncoder.from_pretrained(
            settings.encoder_name,
            freeze_mode=config.freeze_mode,
            frozen_layers=config.frozen_layers,
        )

    model_kwargs: Dict[str, Any] = {
        "architecture": "baseline" if config.mode == "baseline" else "full",
        "encoder_name": settings.encoder_name,
        "random_init": random_init,
        "hidden_size": encoder.hidden_size,
        "hate_labels": labels["hate"],
        "target_labels": labels["target"],
        "reason_labels": labels["reason"],
        "freeze_mode": config.freeze_mode,
        "frozen_layers": config.frozen_layers,
        "interaction_dim": settings.interaction_hidden_dim,
        "interaction_dropout": settings.interaction_dropout,
        "interaction_mode": config.interaction_mode,
        "use_context": config.use_context,
        "use_contrastive": config.use_contrastive,
        "use_target": config.use_target,
        "use_reason": config.use_reason,
        "use_evidence": config.use_evidence,
        "contrastive_dim": config.contrastive_projection_dim,
        "hate_mode": "multiclass",
        "hate_threshold": settings.hate_threshold,
    }
    if random_init:
        model_kwargs["random_num_layers"] = int(encoder.bert.config.num_hidden_layers)
        model_kwargs["random_num_heads"] = int(encoder.bert.config.num_attention_heads)

    if model_kwargs["architecture"] == "baseline":
        model: nn.Module = BaselineModel(
            encoder,
            labels["hate"],
            mode=model_kwargs["hate_mode"],
            threshold=settings.hate_threshold,
        )
    else:
        model = FullModel(
            encoder,
            labels["hate"],
            labels["target"],
            labels["reason"],
            interaction_dim=model_kwargs["interaction_dim"],
            interaction_dropout=model_kwargs["interaction_dropout"],
            interaction_mode=config.interaction_mode,
            use_context=config.use_context,
            use_contrastive=config.use_contrastive,
            use_target=config.use_target,
            use_reason=config.use_reason,
            use_evidence=config.use_evidence,
            contrastive_dim=config.contrastive_projection_dim,
            hate_mode=model_kwargs["hate_mode"],
            hate_threshold=settings.hate_threshold,
        )

    if device is not None:
        model.to(device)
    return model, model_kwargs


class Trainer:
    """Multi-task trainer with checkpointing, AMP and early stopping."""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        train_examples: List[UnifiedExample],
        val_examples: List[UnifiedExample],
        config: TrainConfig,
        device: torch.device,
        label_maps: LabelMaps,
        model_kwargs: Dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_examples = list(train_examples)
        self.val_examples = list(val_examples)
        self.config = config
        self.device = device
        self.label_maps = label_maps
        self.model_kwargs = dict(model_kwargs)

        self.global_step = 0
        self.start_epoch = 0
        self.observed_components: set = set()

        include_evidence = config.use_evidence and any(
            example.has_rationale() for example in self.train_examples
        )
        self.include_evidence = include_evidence
        base_collator = MultitaskCollator(
            tokenizer,
            label_maps,
            max_length=config.max_seq_length,
            include_evidence=include_evidence,
        )

        self.use_contrastive = bool(
            config.use_contrastive and getattr(model, "supports_context", False)
        )
        if self.use_contrastive:
            negative_pool = [
                example.context_text
                for example in self.train_examples
                if example.has_context()
            ]
            if negative_pool:
                self.collator = ContrastivePairCollator(
                    base_collator,
                    negative_pool,
                    seed=config.seed,
                    negatives_per_anchor=config.negatives_per_anchor,
                )
            else:  # no contexts available → no contrastive supervision
                self.use_contrastive = False
                self.collator = base_collator
        else:
            self.collator = base_collator

        self.contrastive_loss = build_contrastive_loss(config.contrastive_temperature)

        self.trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not self.trainable_parameters:
            raise ValueError("Model has no trainable parameters (check freeze settings).")
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters, lr=config.lr, weight_decay=config.weight_decay
        )

        self.train_loader = DataLoader(
            self.train_examples,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )
        val_collator = MultitaskCollator(
            tokenizer,
            label_maps,
            max_length=config.max_seq_length,
            include_evidence=include_evidence,
        )
        self.val_loader = (
            DataLoader(
                self.val_examples,
                batch_size=config.eval_batch_size,
                shuffle=False,
                collate_fn=val_collator,
            )
            if self.val_examples
            else None
        )

        updates_per_epoch = math.ceil(
            len(self.train_loader) / max(1, config.grad_accum_steps)
        )
        total_updates = max(1, updates_per_epoch * max(1, config.epochs))
        warmup_updates = int(round(total_updates * config.warmup_ratio))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _linear_warmup_decay(step, warmup_updates, total_updates),
        )
        self.scaler = build_grad_scaler(enabled=self._amp_enabled())

        self.best_metric: Optional[float] = None
        self.best_epoch = -1
        self.patience_left = config.early_stopping_patience

    # --- helpers -------------------------------------------------------------------
    def _amp_enabled(self) -> bool:
        amp = str(self.config.amp).lower()
        if amp in {"false", "0", "no"}:
            return False
        return self.device.type == "cuda"

    def _hate_mode(self) -> str:
        return str(getattr(self.model.hate_head, "mode", "multiclass"))

    def _model_kwargs_for_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if getattr(self.model, "supports_context", False):
            kwargs.update(
                {
                    "context_input_ids": batch.get("context_input_ids"),
                    "context_attention_mask": batch.get("context_attention_mask"),
                    "context_present": batch.get("context_present"),
                }
            )
        return kwargs

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    # --- checkpointing ----------------------------------------------------------------
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> None:
        from app import __version__ as app_version

        trained_heads = sorted(self.observed_components & set(HEAD_NAMES))
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "app_version": app_version,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "config": self.config.to_dict(),
            "label_maps": {
                "hate": dict(self.label_maps.hate),
                "target": dict(self.label_maps.target),
                "reason": dict(self.label_maps.reason),
            },
            "metrics": metrics or {},
            "model_kwargs": self.model_kwargs,
            "trained_heads": trained_heads,
            "is_best": bool(is_best),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        logger.info("Saved checkpoint %s (epoch %d, best=%s)", path, epoch, is_best)

    def load_checkpoint(self, path: Path, resume_training: bool = True) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        if resume_training:
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scheduler.load_state_dict(payload["scheduler_state"])
            self.global_step = int(payload.get("global_step", 0))
            self.start_epoch = int(payload.get("epoch", -1)) + 1
            self.observed_components.update(payload.get("trained_heads", []))
            logger.info(
                "Resumed from %s: starting at epoch %d", path, self.start_epoch
            )

    # --- training --------------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        output_dir = self.config.resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

        history: List[Dict[str, Any]] = []
        for epoch in range(self.start_epoch, self.config.epochs):
            if hasattr(self.collator, "set_epoch"):
                self.collator.set_epoch(epoch)
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._evaluate(self.val_loader) if self.val_loader else {}

            epoch_record: Dict[str, Any] = {"epoch": epoch}
            epoch_record.update({f"train_{key}": value for key, value in train_metrics.items()})
            epoch_record.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(epoch_record)

            metric = self._selection_metric(val_metrics)
            is_best = False
            if metric is not None:
                if self.best_metric is None or metric > self.best_metric:
                    self.best_metric = float(metric)
                    self.best_epoch = epoch
                    self.patience_left = self.config.early_stopping_patience
                    is_best = True
                else:
                    self.patience_left -= 1

            logger.info(
                "epoch %d done | train_total_loss=%.4f | val=%s",
                epoch,
                train_metrics.get("total_loss", float("nan")),
                {key: round(value, 4) for key, value in val_metrics.items() if isinstance(value, float)},
            )

            metrics_payload = {"train": train_metrics, "val": val_metrics}
            self.save_checkpoint(output_dir / "last.pt", epoch, metrics_payload, is_best=False)
            if is_best:
                self.save_checkpoint(output_dir / "best.pt", epoch, metrics_payload, is_best=True)

            if metric is not None and self.patience_left <= 0:
                logger.info("Early stopping after epoch %d", epoch)
                break

        result: Dict[str, Any] = {
            "history": history,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "trained_heads": sorted(self.observed_components & set(HEAD_NAMES)),
            "output_dir": str(output_dir),
        }
        summary_path = output_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("Training summary written to %s", summary_path)
        return result

    def _train_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        accumulation = max(1, self.config.grad_accum_steps)
        amp_enabled = self.scaler.is_enabled()
        running: Dict[str, float] = {}
        logged_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                outputs = self.model(**self._model_kwargs_for_forward(batch))
                components = compute_task_losses(
                    outputs, batch, hate_mode=self._hate_mode()
                )
                if self.use_contrastive:
                    contrastive_value = contrastive_loss_from_batch(
                        self.model,
                        batch,
                        self.contrastive_loss,
                        negatives_per_anchor=self.config.negatives_per_anchor,
                    )
                    if contrastive_value is not None:
                        components["contrastive"] = contrastive_value
                total, summary = combine_losses(components, self.config.loss_weights)

            if total is None:
                continue

            for name in components:
                if name in HEAD_NAMES:
                    self.observed_components.add(name)

            self.scaler.scale(total / accumulation).backward()

            if (step + 1) % accumulation == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters, self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

            for key, value in summary.items():
                running[key] = running.get(key, 0.0) + float(value)
            logged_batches += 1

            if (
                self.config.log_every_n_steps
                and (step + 1) % self.config.log_every_n_steps == 0
            ):
                averages = {
                    key: round(value / logged_batches, 4)
                    for key, value in running.items()
                }
                logger.info("epoch %d | step %d | %s", epoch, step + 1, averages)

        epoch_summary = {
            key: value / max(1, logged_batches) for key, value in running.items()
        }
        epoch_summary["num_batches"] = float(logged_batches)
        return epoch_summary

    def _evaluate(self, loader: Optional[DataLoader]) -> Dict[str, Any]:
        if loader is None:
            return {}
        from app.evaluation.classification import classification_metrics

        self.model.eval()
        running: Dict[str, float] = {}
        logged_batches = 0
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                outputs = self.model(**self._model_kwargs_for_forward(batch))
                components = compute_task_losses(
                    outputs, batch, hate_mode=self._hate_mode()
                )
                _, summary = combine_losses(components, self.config.loss_weights)
                for key, value in summary.items():
                    running[key] = running.get(key, 0.0) + float(value)
                logged_batches += 1
                all_logits.append(outputs["hate_logits"].detach().cpu())
                all_labels.append(batch["hate_labels"].detach().cpu())

        metrics: Dict[str, Any] = {
            key: value / max(1, logged_batches) for key, value in running.items()
        }

        if all_logits:
            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)
            mask = labels != -100
            if bool(mask.any()):
                head = self.model.hate_head
                if logits.shape[-1] > 1:
                    predictions = logits.argmax(dim=-1)
                else:
                    threshold = float(getattr(head, "threshold", 0.5))
                    probabilities = torch.sigmoid(logits.squeeze(-1))
                    predictions = (probabilities >= threshold).long()
                y_true = labels[mask].tolist()
                y_pred = predictions[mask].tolist()
                label_names = list(getattr(head, "labels", []))
                head_metrics = classification_metrics(
                    y_true, y_pred, labels=label_names
                )
                metrics["accuracy"] = head_metrics.get("accuracy")
                metrics["macro_f1"] = head_metrics.get("macro_f1")
                metrics["weighted_f1"] = head_metrics.get("weighted_f1")
                if logits.shape[-1] > 1:
                    probabilities_all = torch.softmax(logits, dim=-1)
                else:
                    single = torch.sigmoid(logits.squeeze(-1))
                    probabilities_all = torch.stack([1.0 - single, single], dim=-1)
                try:
                    from sklearn.metrics import roc_auc_score

                    if probabilities_all.shape[-1] == 2:
                        roc = float(
                            roc_auc_score(
                                labels[mask].tolist(),
                                probabilities_all[mask, 1].tolist(),
                            )
                        )
                        metrics["roc_auc"] = roc
                except ValueError:
                    metrics["roc_auc"] = None
        return metrics

    @staticmethod
    def _selection_metric(val_metrics: Dict[str, Any]) -> Optional[float]:
        for key in ("macro_f1", "accuracy"):
            value = val_metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None
