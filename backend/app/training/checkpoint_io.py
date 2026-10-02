"""Checkpoint utilities: serve-lean exports.

Full training checkpoints carry optimizer and scheduler state so training can
resume at any time. The inference service never uses those tensors — for the
configurations shipped with this project they account for roughly two thirds
of the file size (≈840 MB of ≈1.29 GB). :func:`export_serving_checkpoint`
produces a copy that keeps everything serving consumes:

* ``model_kwargs`` — how to rebuild the architecture,
* ``model_state`` — the trained weights,
* ``trained_heads`` — which heads may be presented as available,

plus the light metadata fields (label maps, metrics, epoch, ...) and a
``serving_export`` provenance record. Training-only tensors
(``optimizer_state`` / ``scheduler_state``) are dropped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

#: Large training-only payload keys that serving exports drop.
DROP_KEYS = ("optimizer_state", "scheduler_state")


def export_serving_checkpoint(
    source: "str | Path",
    destination: "str | Path | None" = None,
    *,
    force: bool = False,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a serve-lean copy of a training checkpoint.

    Parameters
    ----------
    source:
        Path to a checkpoint saved by the trainer (must contain ``model_state``).
    destination:
        Output path. Defaults to ``<source-stem>.serving<source-suffix>`` next
        to the source (e.g. ``best.pt`` → ``best.serving.pt``).
    force:
        Overwrite an existing destination (refused by default).
    note:
        Optional free-form note stored in the ``serving_export`` metadata.

    Returns
    -------
    dict
        Summary with ``source``, ``destination``, ``size_before``,
        ``size_after``, ``dropped_keys`` and ``kept_keys``.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint not found: {source}")

    if destination is None:
        destination = source.with_name(source.stem + ".serving" + source.suffix)
    destination = Path(destination)

    if destination.exists() and not force:
        raise FileExistsError(
            f"destination already exists: {destination} (pass force=True to overwrite)"
        )

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"not a training checkpoint (no model_state): {source}")

    kept: Dict[str, Any] = {}
    dropped = []
    for key, value in payload.items():
        if key in DROP_KEYS:
            dropped.append(key)
            continue
        kept[key] = value

    serving_meta: Dict[str, Any] = {
        "exported_from": str(source),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    if note:
        serving_meta["note"] = str(note)
    kept["serving_export"] = serving_meta

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kept, destination)

    return {
        "source": str(source),
        "destination": str(destination),
        "size_before": source.stat().st_size,
        "size_after": destination.stat().st_size,
        "dropped_keys": dropped,
        "kept_keys": sorted(kept.keys()),
    }
