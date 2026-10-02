"""Reason-category classification head.

The head is a learned classifier over the final representation — **not** a
keyword lookup. The architecture always contains this head, but it is trained
**only** when real reason annotations are available (a reason-annotated
dataset or a custom annotation file). It is never trained on fabricated
labels — see README "Reason classification" and project spec §11.
"""

from __future__ import annotations

from app.models.softmax_head import SoftmaxClassificationHead


class ReasonHead(SoftmaxClassificationHead):
    """Softmax head over the configured reason categories."""
