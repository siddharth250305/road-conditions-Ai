"""Classification metrics (PRD C5): macro-F1 headline, accuracy, per-class precision/recall.

Pure numpy, so the metrics are testable without scikit-learn. Divisions by zero give 0.0,
matching scikit-learn's zero_division=0.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, n_classes: int) -> np.ndarray:
    """Count matrix with rows = true class and columns = predicted class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must be 1-D and equally long, got {y_true.shape} and {y_pred.shape}"
        )
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    if y_true.size == 0:
        return cm
    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= n_classes:
            raise ValueError(f"{name} must hold integer class ids in [0, {n_classes})")
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def precision_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall and F1 from a confusion matrix."""
    tp = np.diag(cm)
    precision = _safe_divide(tp, cm.sum(axis=0))
    recall = _safe_divide(tp, cm.sum(axis=1))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike, n_classes: int) -> float:
    """Unweighted mean F1 over all n_classes; a class never predicted correctly counts as 0."""
    return float(precision_recall_f1(confusion_matrix(y_true, y_pred, n_classes))[2].mean())


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("accuracy needs non-empty y_true and y_pred of equal shape")
    return float(np.mean(y_true == y_pred))


def classification_report(
    y_true: ArrayLike, y_pred: ArrayLike, class_names: Sequence[str]
) -> dict[str, Any]:
    """JSON-ready report: accuracy, macro-F1, per-class metrics and the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    precision, recall, f1 = precision_recall_f1(cm)
    total = int(cm.sum())
    return {
        "accuracy": float(np.trace(cm) / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for name, p, r, f, s in zip(
                class_names, precision, recall, f1, cm.sum(axis=1), strict=True
            )
        },
        "confusion_matrix": cm.tolist(),
    }


def mean_std(values: Sequence[float], digits: int = 3) -> str:
    """Format seeds' results as 'mean ± sample std' (just the mean for a single value)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return f"{arr[0]:.{digits}f}"
    return f"{arr.mean():.{digits}f} ± {arr.std(ddof=1):.{digits}f}"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a Markdown table that also reads well in a terminal."""
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in cells
    ]
    lines.insert(1, "|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(lines)
