"""Application configuration.

Two layers:

* **runtime** — environment variables / ``.env`` → :class:`Settings` (this module)
* **training** — YAML files under ``configs/`` → see :mod:`app.training.configs`

Nothing in the codebase should hardcode model names, dimensions, thresholds or
paths; pull them from here (or from a training config).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.datasets.label_mapping import (
    CANONICAL_HATE,
    CANONICAL_REASON,
    CANONICAL_TARGET,
)

#: Project root (this file lives in ``backend/app/core/config.py``).
BACKEND_DIR = Path(__file__).resolve().parents[2]


def _csv(value: str) -> List[str]:
    """Split a comma separated string into a clean list."""
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- model / encoder -----------------------------------------------------
    encoder_name: str = Field(
        default="bert-base-uncased",
        validation_alias=AliasChoices("MODEL_NAME", "ENCODER_NAME"),
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("MODEL_CHECKPOINT", "CHECKPOINT_PATH"),
    )
    # When MODEL_CHECKPOINT is unset, auto-load a previously trained checkpoint
    # from checkpoints/ (default.json pointer, else conventional paths).
    # Set AUTO_LOAD_CHECKPOINT=false to always serve an untrained model.
    auto_load_checkpoint: bool = True
    device: str = "auto"                 # auto | cuda | cpu
    init_backbone: str = "pretrained"    # pretrained | random (tests/dev)
    max_seq_length: int = 128
    encoder_freeze_mode: str = "full"    # full | frozen | partial
    encoder_frozen_layers: int = 4
    interaction_hidden_dim: int = 1024
    interaction_dropout: float = 0.2

    # --- heads / label sets ---------------------------------------------------
    # Defaults derive from the canonical vocabularies in
    # ``app.datasets.label_mapping`` (single source of truth); override per
    # deployment via env / .env when a dataset uses a different label set.
    hate_labels: str = ",".join(CANONICAL_HATE)
    target_labels: str = ",".join(CANONICAL_TARGET)
    reason_labels: str = ",".join(CANONICAL_REASON)
    hate_threshold: float = 0.5

    # --- multi-task loss weights ---------------------------------------------
    lambda_hate: float = 1.0
    lambda_target: float = 1.0
    lambda_reason: float = 1.0
    lambda_contrastive: float = 0.5
    lambda_evidence: float = 1.0

    # --- contrastive learning --------------------------------------------------
    contrastive_temperature: float = 0.07
    contrastive_projection_dim: int = 256

    # --- evidence / attribution ------------------------------------------------
    evidence_method: str = "integrated_gradients"  # integrated_gradients | gradient_x_input
    evidence_top_k: int = 5
    evidence_min_score: float = 0.15
    evidence_ig_steps: int = 32

    # --- misc -------------------------------------------------------------------
    seed: int = 42
    log_level: str = "INFO"
    cors_origins: str = "*"

    # --- derived paths -----------------------------------------------------------
    @property
    def backend_dir(self) -> Path:
        return BACKEND_DIR

    @property
    def configs_dir(self) -> Path:
        return BACKEND_DIR / "configs"

    @property
    def checkpoints_dir(self) -> Path:
        return BACKEND_DIR / "checkpoints"

    @property
    def data_dir(self) -> Path:
        return BACKEND_DIR / "data"

    @property
    def experiments_dir(self) -> Path:
        return BACKEND_DIR / "experiments"

    # --- helpers ------------------------------------------------------------------
    @property
    def hate_label_list(self) -> List[str]:
        return _csv(self.hate_labels)

    @property
    def target_label_list(self) -> List[str]:
        return _csv(self.target_labels)

    @property
    def reason_label_list(self) -> List[str]:
        return _csv(self.reason_labels)

    @property
    def cors_origin_list(self) -> List[str]:
        return _csv(self.cors_origins)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def load_yaml_config(name: str | Path) -> Dict[str, Any]:
    """Load a YAML config from ``configs/<name>`` (extension optional)."""
    path = Path(name)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        path = path.with_suffix(".yaml")
    if not path.is_absolute():
        path = BACKEND_DIR / "configs" / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_device(preference: str = "auto"):
    """Return a ``torch.device`` honouring the configured preference.

    ``auto`` selects CUDA when available, otherwise CPU. The backend never
    assumes a GPU exists.
    """
    import torch

    preference = (preference or "auto").lower()
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(preference: str = "auto") -> str:
    """Human-readable compute-device description for startup logs.

    Reports CPU vs GPU explicitly and, when CUDA is used, the GPU name —
    e.g. ``"GPU (CUDA) — NVIDIA GeForce RTX 4060 Laptop GPU"`` or
    ``"CPU (CUDA GPU not available)"``.
    """
    import torch

    device = resolve_device(preference)
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:  # noqa: BLE001 - name lookup is best-effort
            name = "unknown CUDA device"
        return f"GPU (CUDA) — {name}"
    return "CPU (CUDA GPU not available)"
