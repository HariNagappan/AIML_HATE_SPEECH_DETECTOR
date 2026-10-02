"""Logging setup shared by the API, training scripts and tests."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
