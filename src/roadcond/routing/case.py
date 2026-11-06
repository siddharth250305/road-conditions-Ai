"""The router's input for one image, and the read-only view providers and policies get."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class Case:
    """One image: class names and each model's probability vector over them.

    true_label is for evaluation only. The router hands providers and policies a CaseView, which
    has neither true_label nor image_id (RSCD file names contain the label), so routing can never
    depend on the answer.
    """

    image_id: str
    classes: Sequence[str]
    probs: Mapping[str, Sequence[float]]
    true_label: str | None = None

    def view(self) -> "CaseView":
        return CaseView.of(self)


def top_class(classes: Sequence[str], probs: np.ndarray) -> str | None:
    """Class with the highest probability, or None if the vector is unusable."""
    if probs.shape != (len(classes),) or not np.isfinite(probs).all():
        return None
    return classes[int(probs.argmax())]


def _as_vector(values: object) -> np.ndarray:
    """A read-only float vector; an empty one when the input is not numeric (shape check fails)."""
    try:
        arr = np.array(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        arr = np.empty(0)
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, slots=True)
class CaseView:
    """What providers and policies may read: no label, no file name, nothing they can modify."""

    classes: tuple[str, ...]
    probs: Mapping[str, np.ndarray] = field(repr=False)
    predictions: Mapping[str, str | None]  # model -> top class (None if its vector is unusable)

    @classmethod
    def of(cls, case: Case) -> "CaseView":
        classes = tuple(case.classes)
        probs = {model: _as_vector(p) for model, p in case.probs.items()}
        predictions = {m: top_class(classes, p) for m, p in probs.items()}
        return cls(
            classes=classes,
            probs=MappingProxyType(probs),
            predictions=MappingProxyType(predictions),
        )


def is_valid_signal(value: object) -> bool:
    """A usable confidence signal: a real number in [0, 1] (bools and NaN/inf are not)."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.floating | np.integer):
        return False
    try:
        number = float(value)
    except OverflowError:  # e.g. 10**400
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0
