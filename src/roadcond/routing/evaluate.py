"""Evaluation of a routing run: mode counts, errors per mode, catch rate, calibration.

expected_calibration_error mirrors roadcond.calibration.expected_calibration_error, which cannot
be imported without torch; tests check that both agree.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from roadcond.routing.policy import ReviewMode

STRICT_MODES = (ReviewMode.FULL, ReviewMode.SECOND)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Sample-weighted mean |accuracy - confidence| over equal-width confidence bins (lo, hi]."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels)
    if probs.ndim != 2 or labels.shape != probs.shape[:1]:
        raise ValueError(f"Need probs (N, C) and labels (N,), got {probs.shape}, {labels.shape}")
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    index = np.clip(np.ceil(confidence * n_bins).astype(int) - 1, 0, n_bins - 1)
    count = np.bincount(index, minlength=n_bins)
    safe = np.maximum(count, 1)
    mean_conf = np.bincount(index, weights=confidence, minlength=n_bins) / safe
    accuracy = np.bincount(index, weights=correct, minlength=n_bins) / safe
    return float((count * np.abs(accuracy - mean_conf)).sum() / count.sum())


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def mode_counts(modes: Sequence[str]) -> dict[str, dict[str, float]]:
    """Count and percentage of images per review mode (every mode listed, even if empty)."""
    modes = np.asarray(modes)
    n = len(modes)
    out = {}
    for mode in ReviewMode:
        count = int((modes == mode).sum())
        out[str(mode)] = {"count": count, "pct": 100.0 * count / n if n else 0.0}
    return out


def error_rate_by_mode(modes: Sequence[str], wrong: np.ndarray) -> dict[str, dict[str, Any]]:
    """The primary model's error rate among the images routed to each mode."""
    modes, wrong = np.asarray(modes), np.asarray(wrong, dtype=bool)
    out = {}
    for mode in ReviewMode:
        inside = modes == mode
        errors, count = int(wrong[inside].sum()), int(inside.sum())
        out[str(mode)] = {"errors": errors, "count": count, "error_rate": _rate(errors, count)}
    return out


def error_catch_rate(modes: Sequence[str], wrong: np.ndarray) -> dict[str, Any]:
    """Share of the primary model's wrong predictions routed to full or second review."""
    modes, wrong = np.asarray(modes), np.asarray(wrong, dtype=bool)
    caught = int((wrong & np.isin(modes, [str(m) for m in STRICT_MODES])).sum())
    errors = int(wrong.sum())
    return {"caught": caught, "errors": errors, "rate": _rate(caught, errors)}


def class_to_second_review(
    modes: Sequence[str], true_labels: Sequence[str], cls: str
) -> dict[str, Any]:
    """How many images truly of class cls were routed to second_review."""
    modes, true_labels = np.asarray(modes), np.asarray(true_labels)
    of_class = true_labels == cls
    count = int(of_class.sum())
    second = int((of_class & (modes == ReviewMode.SECOND)).sum())
    return {"count": count, "second_review": second, "share": _rate(second, count)}


def light_review_stats(modes: Sequence[str], wrong: np.ndarray) -> dict[str, Any]:
    """Share of images in light_review and the primary model's error rate among them."""
    modes, wrong = np.asarray(modes), np.asarray(wrong, dtype=bool)
    light = modes == ReviewMode.LIGHT
    count = int(light.sum())
    return {
        "light_review_count": count,
        "light_review_pct": 100.0 * count / len(modes) if len(modes) else 0.0,
        "light_review_error_rate": _rate(int(wrong[light].sum()), count),
    }
