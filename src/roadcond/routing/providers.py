"""Confidence providers: each turns one case into a single signal in [0, 1].

Temperature scaling lives here in NumPy so routing needs no torch: probabilities are scaled as
softmax(log p / T), which equals softmax(z / T) for the logits z the probabilities came from.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from roadcond.routing.case import CaseView

EPS = 1e-12  # floor for log(0): random-forest vote fractions are often exactly 0
SUM_TOLERANCE = 1e-3


class ProviderError(RuntimeError):
    """A provider could not produce a signal for this case."""


@runtime_checkable
class ConfidenceProvider(Protocol):
    name: str

    def score(self, case: CaseView) -> float:
        """A signal in [0, 1]; raises ProviderError when it cannot be computed."""
        ...


def scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled probabilities, row-wise: softmax(log(probs) / temperature)."""
    logits = np.log(np.clip(np.asarray(probs, dtype=float), EPS, None)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true labels."""
    picked = np.asarray(probs)[np.arange(len(labels)), np.asarray(labels)]
    return float(-np.log(np.clip(picked, EPS, None)).mean())


def fit_temperature(
    probs: np.ndarray, labels: np.ndarray, t_min: float, t_max: float, steps: int
) -> float:
    """Temperature minimising NLL over a log-spaced grid of `steps` values in [t_min, t_max].

    Fit on validation data only; ties go to the smallest temperature.
    """
    if not 0 < t_min < t_max or steps < 2:
        raise ValueError(f"Need 0 < t_min < t_max and steps >= 2, got {t_min}, {t_max}, {steps}")
    grid = np.geomspace(t_min, t_max, steps)
    losses = [nll(scale_probs(probs, t), labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def _probs_for(case: CaseView, model: str) -> np.ndarray:
    if model not in case.probs:
        raise ProviderError(f"no probabilities for model {model!r}")
    p = case.probs[model]
    if p.shape != (len(case.classes),):
        raise ProviderError(f"{model}: {p.shape} probabilities for {len(case.classes)} classes")
    if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1.0) > SUM_TOLERANCE:
        raise ProviderError(f"{model}: not a probability vector")
    return p


class CalibratedConfidenceProvider:
    """Top-class probability of one model after temperature scaling (1.0 = uncalibrated)."""

    def __init__(self, model: str, temperature: float, name: str = "calibrated_confidence"):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be a positive number, got {temperature}")
        self.model = model
        self.temperature = float(temperature)
        self.name = name

    def score(self, case: CaseView) -> float:
        return float(scale_probs(_probs_for(case, self.model), self.temperature).max())


class AgreementProvider:
    """Share of the other listed models whose top class matches the first model's top class.

    With two models: 1.0 when they agree, 0.0 when they do not. With three: 0, 0.5 or 1.
    """

    def __init__(self, models: Sequence[str], name: str = "agreement"):
        if len(models) < 2 or len(set(models)) != len(models):
            raise ValueError(f"Agreement needs at least two distinct models, got {list(models)}")
        self.models = list(models)
        self.name = name

    def score(self, case: CaseView) -> float:
        tops = [int(_probs_for(case, m).argmax()) for m in self.models]
        return sum(t == tops[0] for t in tops[1:]) / (len(tops) - 1)
