"""The router's input for one image, and the read-only view providers and policies get."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class Case:
    """One image: class names and each model's probability vector over them.

    true_label is for evaluation only. The router hands providers and policies a CaseView,
    which has no true_label, so routing can never depend on it.
    """

    image_id: str
    classes: Sequence[str]
    probs: Mapping[str, Sequence[float]]
    true_label: str | None = None

    def view(self) -> "CaseView":
        return CaseView.of(self)


def top_class(classes: Sequence[str], probs: Sequence[float]) -> str | None:
    """Class with the highest probability, or None if the vector is unusable."""
    p = np.asarray(probs, dtype=float)
    if p.shape != (len(classes),) or not np.isfinite(p).all():
        return None
    return classes[int(p.argmax())]


@dataclass(frozen=True, slots=True)
class CaseView:
    """What providers and policies may read: no true_label, and nothing they can modify."""

    image_id: str
    classes: tuple[str, ...]
    probs: Mapping[str, np.ndarray] = field(repr=False)
    predictions: Mapping[str, str | None]  # model -> top class (None if its vector is unusable)

    @classmethod
    def of(cls, case: Case) -> "CaseView":
        classes = tuple(case.classes)
        probs = {}
        for model, p in case.probs.items():
            arr = np.array(p, dtype=float)
            arr.flags.writeable = False
            probs[model] = arr
        predictions = {m: top_class(classes, p) for m, p in probs.items()}
        return cls(
            image_id=case.image_id,
            classes=classes,
            probs=MappingProxyType(probs),
            predictions=MappingProxyType(predictions),
        )


def is_valid_signal(value: object) -> bool:
    """A usable confidence signal: a real number in [0, 1] (bools and NaN/inf are not)."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.floating | np.integer):
        return False
    return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
