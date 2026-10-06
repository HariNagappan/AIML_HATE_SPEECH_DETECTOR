"""Core config tests: device resolution + human-readable device description."""

from __future__ import annotations

from app.core.config import describe_device, resolve_device


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_describe_device_reports_cpu_or_gpu():
    # Environment-independent: on a CUDA machine this must name the GPU, on a
    # CPU-only machine it must say CPU — an empty description is never valid.
    text = describe_device("auto")
    assert text.startswith("CPU") or text.startswith("GPU (CUDA) —")


def test_describe_device_cpu_string():
    assert describe_device("cpu") == "CPU (CUDA GPU not available)"
