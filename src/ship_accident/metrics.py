from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[int] | None = None,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    labels = labels if labels is not None else sorted(np.unique(np.concatenate([y_true, y_pred])))
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if target_names is None:
        out["classification_report"] = classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        )
    else:
        out["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[target_names[i] for i in labels],
            zero_division=0,
        )
    return out
