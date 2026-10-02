"""Trainer smoke test: 1 tiny epoch, checkpoint save + resume (fast, CPU)."""

from __future__ import annotations

import torch

from app.datasets.collator import build_label_maps
from app.datasets.unified import UnifiedExample
from app.models.full_model import FullModel
from app.training.configs import TrainConfig
from app.training.trainer import Trainer


def _toy_examples():
    rows = [
        ("They should all be kicked out.", "Those immigrants are ruining everything.", "hate"),
        ("I love this weather today.", "What a beautiful morning.", "normal"),
        ("You are an idiot and a loser.", "Stop posting nonsense.", "offensive"),
        ("Go back to your country.", "We were talking about immigration.", "hate"),
    ]
    return [
        UnifiedExample(id=f"toy-{index}", current_text=text, context_text=context, hate_label=label)
        for index, (text, context, label) in enumerate(rows)
    ]


def _model_kwargs():
    return {
        "architecture": "full",
        "encoder_name": "bert-base-uncased",
        "random_init": True,
        "hidden_size": 64,
        "random_num_layers": 2,
        "random_num_heads": 4,
    }


def test_trainer_smoke_and_resume(tmp_path, small_encoder, tokenizer):
    examples = _toy_examples()
    label_maps = build_label_maps(["hate", "offensive", "normal"])

    def make_trainer(output_dir: str) -> Trainer:
        model = FullModel(
            small_encoder,
            ["hate", "offensive", "normal"],
            interaction_dim=32,
            use_target=False,
            use_reason=False,
            use_evidence=False,
            use_contrastive=True,
        )
        config = TrainConfig(
            mode="context",
            config_name="test",
            epochs=1,
            batch_size=2,
            eval_batch_size=2,
            lr=5e-4,
            log_every_n_steps=0,
            early_stopping_patience=5,
            warmup_ratio=0.0,
            amp="false",
            max_seq_length=32,
            seed=42,
            output_dir=output_dir,
            use_contrastive=True,
            use_target=False,
            use_reason=False,
            use_evidence=False,
        )
        return Trainer(
            model=model,
            tokenizer=tokenizer,
            train_examples=examples,
            val_examples=examples,
            config=config,
            device=torch.device("cpu"),
            label_maps=label_maps,
            model_kwargs=_model_kwargs(),
        )

    output_dir = str(tmp_path / "ckpt")
    trainer = make_trainer(output_dir)
    result = trainer.train()

    assert (tmp_path / "ckpt" / "last.pt").exists()
    assert "hate" in result["trained_heads"]
    assert result["history"], "training history must not be empty"
    first_epoch = result["history"][0]
    assert "train_hate_loss" in first_epoch
    assert "train_total_loss" in first_epoch

    # resume path: a fresh trainer loads weights + epoch bookkeeping
    resumed = make_trainer(output_dir)
    resumed.load_checkpoint(tmp_path / "ckpt" / "last.pt", resume_training=True)
    assert resumed.start_epoch == 1
    assert "hate" in resumed.observed_components


def test_reason_loss_only_when_labels_present():
    """Reason supervision is present iff the batch carries reason labels."""
    from app.training.losses import compute_task_losses

    outputs = {
        "hate_logits": torch.randn(2, 3),
        "reason_logits": torch.randn(2, 8),
    }
    with_labels = compute_task_losses(
        outputs,
        {"hate_labels": torch.tensor([0, 2]), "reason_labels": torch.tensor([4, 0])},
    )
    assert "reason" in with_labels and "hate" in with_labels

    masked = compute_task_losses(
        outputs,
        {"hate_labels": torch.tensor([0, 2]), "reason_labels": torch.tensor([-100, -100])},
    )
    assert "reason" not in masked  # fully masked batch → no fake supervision


def test_trainer_trains_reason_head_when_labels_exist(tmp_path, small_encoder, tokenizer):
    """End-to-end trainer smoke with reason labels: loss + trained_heads."""
    rows = [
        ("They should all be kicked out.", "Those immigrants are ruining everything.", "hate", "exclusion"),
        ("I love this weather today.", "What a beautiful morning.", "normal", "other"),
        ("You are an idiot and a loser.", "Stop posting nonsense.", "offensive", "insult"),
        ("Go back to your own country.", "We were talking about immigration.", "hate", "exclusion"),
    ]
    examples = [
        UnifiedExample(
            id=f"reason-{index}",
            current_text=text,
            context_text=context,
            hate_label=hate,
            reason_label=reason,
        )
        for index, (text, context, hate, reason) in enumerate(rows)
    ]
    reason_labels = [
        "insult",
        "dehumanization",
        "negative_stereotyping",
        "threat",
        "exclusion",
        "discrimination",
        "incitement_to_violence",
        "other",
    ]
    label_maps = build_label_maps(["hate", "offensive", "normal"], (), reason_labels)

    model = FullModel(
        small_encoder,
        ["hate", "offensive", "normal"],
        interaction_dim=32,
        reason_labels=reason_labels,
        use_target=False,
        use_reason=True,
        use_evidence=False,
        use_contrastive=True,
    )
    config = TrainConfig(
        mode="context",
        config_name="test-reason",
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        lr=5e-4,
        log_every_n_steps=0,
        early_stopping_patience=5,
        warmup_ratio=0.0,
        amp="false",
        max_seq_length=32,
        seed=42,
        output_dir=str(tmp_path / "ckpt-reason"),
        use_contrastive=True,
        use_target=False,
        use_reason=True,
        use_evidence=False,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_examples=examples,
        val_examples=examples,
        config=config,
        device=torch.device("cpu"),
        label_maps=label_maps,
        model_kwargs={**_model_kwargs(), "use_reason": True, "reason_labels": reason_labels},
    )
    result = trainer.train()

    assert "reason" in result["trained_heads"]
    assert "hate" in result["trained_heads"]
    first_epoch = result["history"][0]
    assert "train_reason_loss" in first_epoch
    assert first_epoch["train_reason_loss"] > 0.0
