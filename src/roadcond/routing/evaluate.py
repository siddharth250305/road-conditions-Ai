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
    """Share of the primary model's wrong predictions routed to full or second review.

    Read it against strict_share, the share of all images sent there: routing everything to
    full review catches every error. lift = rate / strict_share, where 1.0 is what random
    routing of the same volume would catch.
    """
    modes, wrong = np.asarray(modes), np.asarray(wrong, dtype=bool)
    strict = np.isin(modes, [str(m) for m in STRICT_MODES])
    caught, errors = int((wrong & strict).sum()), int(wrong.sum())
    rate = _rate(caught, errors)
    strict_share = _rate(int(strict.sum()), len(modes))
    lift = rate / strict_share if rate is not None and strict_share else None
    return {
        "caught": caught,
        "errors": errors,
        "rate": rate,
        "strict_share": strict_share,
        "lift": lift,
    }


def class_to_second_review(
    modes: Sequence[str],
    true_labels: Sequence[str],
    reasons: Sequence[Sequence[str]],
    cls: str,
) -> dict[str, Any]:
    """How images truly of class cls were routed, and how precise the cls safety rule is.

    share: true-cls images sent to second_review (compare with the overall second_review share).
    rule_precision: of the images the "safety_class:<cls>" rule fired on, the share truly cls.
    """
    modes, true_labels = np.asarray(modes), np.asarray(true_labels)
    of_class = true_labels == cls
    count = int(of_class.sum())
    second = int((of_class & (modes == ReviewMode.SECOND)).sum())
    fired = np.array([f"safety_class:{cls}" in r for r in reasons], dtype=bool)
    return {
        "count": count,
        "second_review": second,
        "share": _rate(second, count),
        "left_in_light_review": int((of_class & (modes == ReviewMode.LIGHT)).sum()),
        "rule_fired": int(fired.sum()),
        "rule_precision": _rate(int((fired & of_class).sum()), int(fired.sum())),
    }


def mode_shares(modes: Sequence[str], wrong: np.ndarray) -> dict[str, Any]:
    """Percentage of images per mode, and the primary model's error rate in light_review."""
    modes, wrong = np.asarray(modes), np.asarray(wrong, dtype=bool)
    light = modes == ReviewMode.LIGHT
    count = int(light.sum())
    out: dict[str, Any] = {
        f"{m}_pct": 100.0 * int((modes == m).sum()) / len(modes) if len(modes) else 0.0
        for m in ReviewMode
    }
    out["light_review_count"] = count
    out["light_review_error_rate"] = _rate(int(wrong[light].sum()), count)
    return out
