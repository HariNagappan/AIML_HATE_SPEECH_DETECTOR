"""Tests for the serve-lean checkpoint export (app/training/checkpoint_io)."""

from __future__ import annotations

import torch
import pytest

from app.training.checkpoint_io import export_serving_checkpoint


def _fake_payload():
    linear = torch.nn.Linear(4, 3)
    return {
        "format_version": 1,
        "app_version": "0.1.0",
        "model_state": linear.state_dict(),
        "optimizer_state": {
            "state": {"0": {"momentum": torch.zeros(3, 4)}},
            "param_groups": [{"lr": 2e-5}],
        },
        "scheduler_state": {"last_epoch": 10, "base_lrs": [2e-5]},
        "epoch": 3,
        "global_step": 120,
        "config": {"mode": "full"},
        "label_maps": {"hate": {"hate": 0, "normal": 1}},
        "metrics": {"val": {"macro_f1": 0.5}},
        "model_kwargs": {"architecture": "full", "encoder_name": "bert-base-uncased"},
        "trained_heads": ["hate"],
        "is_best": True,
    }


def test_export_drops_training_state_and_preserves_weights(tmp_path):
    payload = _fake_payload()
    src = tmp_path / "best.pt"
    torch.save(payload, src)

    info = export_serving_checkpoint(src)

    dst = tmp_path / "best.serving.pt"
    assert info["destination"] == str(dst)
    assert dst.exists()
    assert sorted(info["dropped_keys"]) == ["optimizer_state", "scheduler_state"]

    out = torch.load(dst, map_location="cpu", weights_only=False)
    assert "optimizer_state" not in out
    assert "scheduler_state" not in out
    assert out["trained_heads"] == ["hate"]
    assert out["model_kwargs"] == payload["model_kwargs"]
    assert out["label_maps"] == payload["label_maps"]
    assert out["epoch"] == 3
    assert out["format_version"] == 1
    for key, value in payload["model_state"].items():
        assert torch.equal(out["model_state"][key], value)
    assert out["serving_export"]["exported_from"] == str(src)


def test_output_path_default_and_custom(tmp_path):
    src = tmp_path / "last.pt"
    torch.save(_fake_payload(), src)
    custom = tmp_path / "nested" / "serving.pt"

    info = export_serving_checkpoint(src, custom)
    assert info["destination"] == str(custom)
    assert custom.exists()  # parents created


def test_overwrite_requires_force(tmp_path):
    src = tmp_path / "best.pt"
    torch.save(_fake_payload(), src)
    export_serving_checkpoint(src)

    with pytest.raises(FileExistsError):
        export_serving_checkpoint(src)

    info = export_serving_checkpoint(src, force=True)
    assert info["destination"].endswith("best.serving.pt")


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_serving_checkpoint(tmp_path / "nope.pt")


def test_rejects_non_checkpoint(tmp_path):
    src = tmp_path / "junk.pt"
    torch.save({"foo": 1}, src)
    with pytest.raises(ValueError):
        export_serving_checkpoint(src)


def test_note_is_recorded(tmp_path):
    src = tmp_path / "best.pt"
    torch.save(_fake_payload(), src)
    export_serving_checkpoint(src, note="deploy build A")
    out = torch.load(tmp_path / "best.serving.pt", map_location="cpu", weights_only=False)
    assert out["serving_export"]["note"] == "deploy build A"
