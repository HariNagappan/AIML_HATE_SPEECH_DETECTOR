"""Target-group classification head.

Predicts the target category (race / ethnicity / nationality / religion /
gender / sexual_orientation / political_group / other / none — configurable).
Trained only where the dataset provides target annotations (HateXplain does);
label mapping from dataset communities lives in ``app.datasets.label_mapping``.
"""

from __future__ import annotations

from app.models.softmax_head import SoftmaxClassificationHead


class TargetHead(SoftmaxClassificationHead):
    """Softmax head over the configured target categories."""
