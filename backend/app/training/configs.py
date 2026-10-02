"""Training configuration.

YAML files under ``configs/`` are merged (``base.yaml`` + the named config)
and mapped onto :class:`TrainConfig`. Relative paths are resolved against the
backend directory. Nothing about hyper-parameters is hardcoded in the model
classes themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import BACKEND_DIR, deep_merge, get_settings, load_yaml_config
from app.training.losses import LossWeights


@dataclass
class TrainConfig:
    # experiment identity
    mode: str = "full"                    # baseline | context | full
    config_name: str = "base"

    seed: int = 42

    # encoder
    freeze_mode: str = "full"
    frozen_layers: int = 4
    max_seq_length: int = 128
    interaction_mode: str = "mlp"         # mlp | simple

    # component switches (ablation experiments rely on these)
    use_context: bool = True
    use_contrastive: bool = True
    use_target: bool = True
    use_reason: bool = False
    use_evidence: bool = True

    # optimisation
    lr: float = 2e-5
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    amp: str = "auto"                     # auto | true | false
    early_stopping_patience: int = 2
    log_every_n_steps: int = 50
    output_dir: str = "checkpoints"

    # loss weights / contrastive
    loss_weights: LossWeights = field(default_factory=LossWeights)
    contrastive_temperature: float = 0.07
    contrastive_projection_dim: int = 256
    negatives_per_anchor: int = 1
    negative_strategy: str = "shuffled_context"

    # dataset
    dataset_name: str = "hatexplain"
    train_file: str = "data/processed/hatexplain_train.jsonl"
    val_file: str = "data/processed/hatexplain_val.jsonl"
    test_file: str = "data/processed/hatexplain_test.jsonl"

    # label sets (None → sensible defaults from the dataset / runtime settings)
    hate_labels: Optional[List[str]] = None
    target_labels: Optional[List[str]] = None
    reason_labels: Optional[List[str]] = None

    # --- resolution helpers -----------------------------------------------------
    def resolved_output_dir(self) -> Path:
        path = Path(self.output_dir)
        return path if path.is_absolute() else BACKEND_DIR / path

    def resolved_file(self, which: str) -> Path:
        """Resolve ``train`` / ``val`` / ``test`` file against the backend dir."""
        raw = {"train": self.train_file, "val": self.val_file, "test": self.test_file}[which]
        path = Path(raw)
        return path if path.is_absolute() else BACKEND_DIR / path

    def resolved_label_sets(self) -> Dict[str, List[str]]:
        """Label sets for the heads (explicit config > dataset default > settings)."""
        settings = get_settings()
        hate = self.hate_labels
        if hate is None:
            hate = (
                ["hate_speech", "counter_speech", "neither"]
                if "counter" in self.dataset_name
                else settings.hate_label_list
            )
        target = self.target_labels if self.target_labels is not None else settings.target_label_list
        reason = self.reason_labels if self.reason_labels is not None else settings.reason_label_list
        return {"hate": list(hate), "target": list(target), "reason": list(reason)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "config_name": self.config_name,
            "seed": self.seed,
            "freeze_mode": self.freeze_mode,
            "frozen_layers": self.frozen_layers,
            "max_seq_length": self.max_seq_length,
            "interaction_mode": self.interaction_mode,
            "use_context": self.use_context,
            "use_contrastive": self.use_contrastive,
            "use_target": self.use_target,
            "use_reason": self.use_reason,
            "use_evidence": self.use_evidence,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "grad_accum_steps": self.grad_accum_steps,
            "max_grad_norm": self.max_grad_norm,
            "amp": self.amp,
            "early_stopping_patience": self.early_stopping_patience,
            "output_dir": self.output_dir,
            "loss_weights": self.loss_weights.to_dict(),
            "contrastive_temperature": self.contrastive_temperature,
            "contrastive_projection_dim": self.contrastive_projection_dim,
            "negatives_per_anchor": self.negatives_per_anchor,
            "negative_strategy": self.negative_strategy,
            "dataset_name": self.dataset_name,
            "train_file": self.train_file,
            "val_file": self.val_file,
            "test_file": self.test_file,
            "hate_labels": self.hate_labels,
            "target_labels": self.target_labels,
            "reason_labels": self.reason_labels,
        }

    # --- loaders ---------------------------------------------------------------------
    def load_examples(self, which: str) -> List[Any]:
        """Load one processed split (``train`` / ``val`` / ``test``) as examples."""
        from app.datasets.unified import load_unified_jsonl

        path = self.resolved_file(which)
        if not path.exists():
            raise FileNotFoundError(
                f"Processed data file not found: {path}. "
                f"Run `python scripts/download_data.py` and `python scripts/preprocess.py` first."
            )
        return load_unified_jsonl(path)

    @classmethod
    def from_yaml(
        cls, name: str = "base", overrides: Optional[Dict[str, Any]] = None
    ) -> "TrainConfig":
        base = load_yaml_config("base")
        specific = {} if name == "base" else load_yaml_config(name)
        merged = deep_merge(base, specific)
        if overrides:
            merged = deep_merge(merged, overrides)
        return cls.from_merged(merged, config_name=name)

    @classmethod
    def from_merged(cls, merged: Dict[str, Any], config_name: str = "custom") -> "TrainConfig":
        encoder_cfg = merged.get("encoder", {}) or {}
        model_cfg = merged.get("model", {}) or {}
        training_cfg = merged.get("training", {}) or {}
        contrastive_cfg = merged.get("contrastive", {}) or {}
        loss_weights = LossWeights.from_dict(merged.get("loss_weights"))
        datasets_cfg = merged.get("datasets", {}) or {}
        labels_cfg = merged.get("labels", {}) or {}

        dataset_name = str(merged.get("dataset", "hatexplain"))
        dataset_files = datasets_cfg.get(dataset_name, {}) or {}
        default_files = {
            "train": f"data/processed/{dataset_name}_train.jsonl",
            "val": f"data/processed/{dataset_name}_val.jsonl",
            "test": f"data/processed/{dataset_name}_test.jsonl",
        }

        def label_list(key: str) -> Optional[List[str]]:
            value = labels_cfg.get(key)
            if value:
                return [str(item) for item in value]
            return None

        return cls(
            mode=str(merged.get("mode", "full")),
            config_name=config_name,
            seed=int(merged.get("seed", 42)),
            freeze_mode=str(encoder_cfg.get("freeze_mode", "full")),
            frozen_layers=int(encoder_cfg.get("frozen_layers", 4)),
            max_seq_length=int(encoder_cfg.get("max_seq_length", 128)),
            interaction_mode=str(model_cfg.get("interaction_mode", "mlp")),
            use_context=bool(model_cfg.get("use_context", True)),
            use_contrastive=bool(model_cfg.get("use_contrastive", True)),
            use_target=bool(model_cfg.get("use_target", True)),
            use_reason=bool(model_cfg.get("use_reason", False)),
            use_evidence=bool(model_cfg.get("use_evidence", True)),
            lr=float(training_cfg.get("lr", 2e-5)),
            batch_size=int(training_cfg.get("batch_size", 16)),
            eval_batch_size=int(training_cfg.get("eval_batch_size", 32)),
            epochs=int(training_cfg.get("epochs", 5)),
            weight_decay=float(training_cfg.get("weight_decay", 0.01)),
            warmup_ratio=float(training_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(training_cfg.get("grad_accum_steps", 1)),
            max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
            amp=str(training_cfg.get("amp", "auto")),
            early_stopping_patience=int(training_cfg.get("early_stopping_patience", 2)),
            log_every_n_steps=int(training_cfg.get("log_every_n_steps", 50)),
            output_dir=str(training_cfg.get("output_dir", "checkpoints")),
            loss_weights=loss_weights,
            contrastive_temperature=float(contrastive_cfg.get("temperature", 0.07)),
            contrastive_projection_dim=int(contrastive_cfg.get("projection_dim", 256)),
            negatives_per_anchor=int(contrastive_cfg.get("negatives_per_anchor", 1)),
            negative_strategy=str(contrastive_cfg.get("negative_strategy", "shuffled_context")),
            dataset_name=dataset_name,
            train_file=str(dataset_files.get("train_file", default_files["train"])),
            val_file=str(dataset_files.get("val_file", default_files["val"])),
            test_file=str(dataset_files.get("test_file", default_files["test"])),
            hate_labels=label_list("hate"),
            target_labels=label_list("target"),
            reason_labels=label_list("reason"),
        )
