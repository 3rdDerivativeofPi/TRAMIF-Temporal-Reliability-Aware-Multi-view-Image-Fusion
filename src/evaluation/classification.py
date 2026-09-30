"""Classification metrics for the MalCSBSV reproduction track."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def _to_numpy_1d(values: object, *, name: str) -> np.ndarray:
    """Convert NumPy-like or torch.Tensor-like input to a 1-D NumPy array."""

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    return array


def compute_classification_metrics(
    y_true: object,
    y_pred: object,
    *,
    labels: Sequence[int] | None = None,
) -> dict[str, float]:
    """Compute accuracy and macro-averaged precision, recall, and F1.

    Parameters
    ----------
    y_true:
        Ground-truth class IDs.
    y_pred:
        Predicted class IDs.
    labels:
        Optional fixed class ordering. For the 10-family reproduction, pass
        ``range(10)`` so all intended classes are evaluated consistently.

    Notes
    -----
    ``zero_division=0`` makes undefined class-wise precision/recall explicit
    as zero instead of emitting warnings or silently dropping classes.
    """

    true = _to_numpy_1d(y_true, name="y_true")
    pred = _to_numpy_1d(y_pred, name="y_pred")

    if len(true) != len(pred):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples"
        )

    if len(true) == 0:
        raise ValueError("Metric inputs must not be empty")

    metric_labels = None if labels is None else list(labels)

    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(
            precision_score(
                true,
                pred,
                labels=metric_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                true,
                pred,
                labels=metric_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                true,
                pred,
                labels=metric_labels,
                average="macro",
                zero_division=0,
            )
        ),
    }
