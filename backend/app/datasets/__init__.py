"""Dataset adapters, label mapping, preprocessing and collators.

Datasets are never merged blindly: each dataset has its own adapter that
converts raw records into :class:`app.datasets.unified.UnifiedExample`, with
missing annotations represented explicitly as ``None``.
"""
