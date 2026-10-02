"""Ablation experiment definitions and runner (project spec §21).

==============================================  ======================================
Experiment                                      Description
==============================================  ======================================
``exp1_baseline``                               BERT only (no context)
``exp2_bert_context``                           BERT + context (concatenation, no D/M interaction)
``exp3_interaction``                            BERT + context interaction ``[E_c;E_p;D;M]`` → MLP
``exp4_contrastive``                            + contrastive learning (InfoNCE)
``exp5_full``                                   Full multi-task model (hate + target [+ reason])
``exp6_evidence``                               Full model + attribution-based evidence evaluation
==============================================  ======================================

Results are stored as machine-readable JSON in ``experiments/`` plus a CSV
summary. The runner only *reports* numbers — it never claims that context or
contrastive learning improved anything without experimental evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_settings, resolve_device
from app.core.logging import get_logger
from app.training.configs import TrainConfig

logger = get_logger("evaluation.ablation")

EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "exp1_baseline",
        "description": "BERT only (current comment, no context)",
        "config": "baseline",
        "overrides": {},
    },
    {
        "name": "exp2_bert_context",
        "description": "BERT + context concatenation (no interaction features)",
        "config": "context",
        "overrides": {"model": {"interaction_mode": "simple"}},
    },
    {
        "name": "exp3_interaction",
        "description": "BERT + context interaction ([E_c;E_p;D;M] MLP)",
        "config": "context",
        "overrides": {"model": {"interaction_mode": "mlp"}},
    },
    {
        "name": "exp4_contrastive",
        "description": "BERT + context interaction + contrastive learning",
        "config": "context",
        "overrides": {"model": {"interaction_mode": "mlp", "use_contrastive": True}},
    },
    {
        "name": "exp5_full",
        "description": "Full multi-task model (hate + target [+ reason when annotated])",
        "config": "full",
        "overrides": {},
    },
    {
        "name": "exp6_evidence",
        "description": "Full model + attribution-based evidence evaluation against rationales",
        "config": "full",
        "overrides": {},
        "evaluate_evidence": True,
    },
]


def get_experiment(name: str) -> Dict[str, Any]:
    for experiment in EXPERIMENTS:
        if experiment["name"] == name:
            return experiment
    raise KeyError(
        f"Unknown experiment {name!r}. Available: {[e['name'] for e in EXPERIMENTS]}"
    )


def build_experiment_config(
    name: str, dataset: Optional[str] = None
) -> TrainConfig:
    definition = get_experiment(name)
    overrides: Dict[str, Any] = json.loads(
        json.dumps(definition.get("overrides") or {})
    )
    if dataset:
        overrides.setdefault("dataset", dataset)
    training_overrides = dict(overrides.get("training", {}))
    training_overrides.setdefault("output_dir", f"checkpoints/{name}")
    overrides["training"] = training_overrides
    return TrainConfig.from_yaml(definition["config"], overrides=overrides)


def run_experiment(
    name: str,
    dataset: Optional[str] = None,
    max_evidence_examples: int = 100,
) -> Dict[str, Any]:
    """Train + evaluate one experiment and persist its results."""
    from app.datasets.collator import build_label_maps
    from app.datasets.tokenizer import TokenizerWrapper
    from app.training.trainer import Trainer, build_model, set_seed

    settings = get_settings()
    config = build_experiment_config(name, dataset=dataset)
    device = resolve_device(settings.device)
    set_seed(config.seed)

    train_examples = config.load_examples("train")
    val_examples = config.load_examples("val")
    test_examples = (
        config.load_examples("test") if config.resolved_file("test").exists() else []
    )

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
    training_result = trainer.train()

    result: Dict[str, Any] = {
        "experiment": name,
        "description": get_experiment(name)["description"],
        "dataset": config.dataset_name,
        "training": {
            "best_epoch": training_result["best_epoch"],
            "best_metric": training_result["best_metric"],
            "trained_heads": training_result["trained_heads"],
            "history": training_result["history"],
        },
        "test_metrics": {},
        "evidence_metrics": None,
    }

    output_dir = config.resolved_output_dir()
    best_path = output_dir / "best.pt"
    checkpoint_path = best_path if best_path.exists() else output_dir / "last.pt"

    if test_examples:
        result["test_metrics"] = evaluate_checkpoint(
            str(checkpoint_path), test_examples, config, settings
        )
    if get_experiment(name).get("evaluate_evidence") and test_examples:
        result["evidence_metrics"] = evaluate_evidence(
            str(checkpoint_path),
            test_examples,
            config,
            settings,
            max_examples=max_evidence_examples,
        )

    experiments_dir = settings.experiments_dir
    experiments_dir.mkdir(parents=True, exist_ok=True)
    output_path = experiments_dir / f"{name}.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    _append_summary_csv(experiments_dir / "summary.csv", name, result)
    logger.info("Experiment %s written to %s", name, output_path)
    return result


def evaluate_checkpoint(
    checkpoint_path: str,
    test_examples: List[Any],
    config: TrainConfig,
    settings,
) -> Dict[str, Any]:
    """Evaluate a trained checkpoint on a split (hate + target heads)."""
    import torch
    from torch.utils.data import DataLoader

    from app.datasets.collator import MultitaskCollator, build_label_maps
    from app.evaluation.classification import classification_metrics
    from app.services.inference import build_model_from_checkpoint

    bundle = build_model_from_checkpoint(checkpoint_path, settings)
    model = bundle.model
    tokenizer = bundle.tokenizer
    model_kwargs = bundle.model_kwargs
    labels = {
        "hate": model_kwargs.get("hate_labels") or [],
        "target": model_kwargs.get("target_labels") or [],
        "reason": model_kwargs.get("reason_labels") or [],
    }
    label_maps = build_label_maps(labels["hate"], labels["target"], labels["reason"])
    collator = MultitaskCollator(
        tokenizer, label_maps, max_length=config.max_seq_length
    )
    loader = DataLoader(
        test_examples,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model.eval()
    hate_true: List[int] = []
    hate_pred: List[int] = []
    target_true: List[int] = []
    target_pred: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(bundle.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            forward_kwargs: Dict[str, Any] = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            if getattr(model, "supports_context", False):
                forward_kwargs.update(
                    {
                        "context_input_ids": batch.get("context_input_ids"),
                        "context_attention_mask": batch.get("context_attention_mask"),
                        "context_present": batch.get("context_present"),
                    }
                )
            outputs = model(**forward_kwargs)

            hate_logits = outputs["hate_logits"]
            hate_labels = batch["hate_labels"]
            mask = hate_labels != -100
            if bool(mask.any()):
                if hate_logits.shape[-1] > 1:
                    predictions = hate_logits.argmax(dim=-1)
                else:
                    threshold = float(getattr(model.hate_head, "threshold", 0.5))
                    predictions = (
                        torch.sigmoid(hate_logits.squeeze(-1)) >= threshold
                    ).long()
                hate_true.extend(hate_labels[mask].tolist())
                hate_pred.extend(predictions[mask].tolist())

            if "target_logits" in outputs and "target_labels" in batch:
                target_logits = outputs["target_logits"]
                target_labels = batch["target_labels"]
                target_mask = target_labels != -100
                if bool(target_mask.any()):
                    predictions = target_logits.argmax(dim=-1)
                    target_true.extend(target_labels[target_mask].tolist())
                    target_pred.extend(predictions[target_mask].tolist())

    metrics: Dict[str, Any] = {}
    if hate_true:
        metrics["hate"] = classification_metrics(hate_true, hate_pred, labels["hate"])
    if target_true:
        metrics["target"] = classification_metrics(
            target_true, target_pred, labels["target"]
        )
    metrics["num_examples"] = len(test_examples)
    return metrics


def evaluate_evidence(
    checkpoint_path: str,
    test_examples: List[Any],
    config: TrainConfig,
    settings,
    max_examples: int = 100,
) -> Optional[Dict[str, Any]]:
    """Attribution-based evidence vs gold rationales (token-level P/R/F1)."""
    from app.datasets.tokenizer import align_rationales_to_subwords
    from app.evaluation.rationale import token_binary_metrics
    from app.reasoning.attribution import compute_attributions
    from app.reasoning.evidence_extractor import extract_evidence
    from app.services.inference import build_model_from_checkpoint

    bundle = build_model_from_checkpoint(checkpoint_path, settings)
    if "hate" not in bundle.trained_heads:
        return None
    examples = [
        example for example in test_examples if example.has_rationale()
    ][: max(1, int(max_examples))]
    if not examples:
        return None

    per_example: List[Dict[str, float]] = []
    for example in examples:
        attribution = compute_attributions(
            bundle.model,
            bundle.tokenizer,
            example.current_text,
            example.context_text,
            target="hate",
            method=settings.evidence_method,
            steps=settings.evidence_ig_steps,
            max_length=config.max_seq_length,
            device=bundle.device,
        )
        spans = extract_evidence(
            attribution,
            bundle.tokenizer,
            top_k=settings.evidence_top_k,
            min_score_frac=settings.evidence_min_score,
        )
        predicted_mask = [0] * len(attribution.tokens)
        for span in spans:
            for index in span.token_indices:
                if 0 <= index < len(predicted_mask):
                    predicted_mask[index] = 1
        gold_mask = align_rationales_to_subwords(
            bundle.tokenizer,
            example.current_text,
            example.rationale_labels or [],
            max_length=config.max_seq_length,
        )
        per_example.append(token_binary_metrics(gold_mask, predicted_mask))

    if not per_example:
        return None
    aggregated = {
        key: sum(item[key] for item in per_example) / len(per_example)
        for key in ("precision", "recall", "f1")
    }
    aggregated["num_examples"] = float(len(per_example))
    return aggregated


def _append_summary_csv(path: Path, name: str, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    hate_metrics = (result.get("test_metrics") or {}).get("hate") or {}
    evidence = result.get("evidence_metrics") or {}
    row = {
        "experiment": name,
        "dataset": result.get("dataset"),
        "best_epoch": result.get("training", {}).get("best_epoch"),
        "best_metric": result.get("training", {}).get("best_metric"),
        "test_accuracy": hate_metrics.get("accuracy"),
        "test_macro_f1": hate_metrics.get("macro_f1"),
        "evidence_precision": evidence.get("precision"),
        "evidence_recall": evidence.get("recall"),
        "evidence_f1": evidence.get("f1"),
    }
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
