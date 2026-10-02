"""Classification metrics for the hate / target / reason heads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Accuracy, macro/weighted precision-recall-F1, per-class F1, confusion matrix.

    ``labels`` is the readable class-name list; when omitted, raw class ids
    are used as names. Metrics that cannot be computed for the given data are
    omitted (e.g. ROC-AUC needs scores, not hard labels) — nothing is guessed.
    """
    y_true = [int(value) for value in y_true]
    y_pred = [int(value) for value in y_pred]
    if not y_true:
        return {}

    if labels is not None:
        label_ids = list(range(len(labels)))
        names: List[str] = [str(label) for label in labels]
    else:
        label_ids = sorted(set(y_true) | set(y_pred))
        names = [str(idx) for idx in label_ids]

    result: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average="weighted", zero_division=0
    )
    result.update(
        {
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
        }
    )

    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average=None, zero_division=0
    )
    result["per_class_precision"] = {
        name: float(value) for name, value in zip(names, per_class[0])
    }
    result["per_class_recall"] = {
        name: float(value) for name, value in zip(names, per_class[1])
    }
    result["per_class_f1"] = {
        name: float(value) for name, value in zip(names, per_class[2])
    }
    result["support"] = {
        name: int(value) for name, value in zip(names, per_class[3])
    }

    result["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=label_ids
    ).tolist()
    return result
