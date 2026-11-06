"""Review policies: turn confidence signals into one of three review modes.

There is deliberately no automatic-accept mode: light_review, where a person glances at the
model's label, is the lightest outcome.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from roadcond.routing.case import CaseView


class ReviewMode(StrEnum):
    LIGHT = "light_review"  # a person glances at the model's label
    FULL = "full_review"  # a person checks the image carefully
    SECOND = "second_review"  # two people check it independently

    @property
    def rank(self) -> int:
        return _RANK[self]


_RANK = {ReviewMode.LIGHT: 0, ReviewMode.FULL: 1, ReviewMode.SECOND: 2}


def strictest(*modes: ReviewMode) -> ReviewMode:
    return max(modes, key=lambda m: m.rank)


@dataclass(frozen=True)
class Decision:
    mode: ReviewMode
    reasons: list[str] = field(default_factory=list)  # short rule ids, e.g. "conf<0.85"


@runtime_checkable
class ReviewPolicy(Protocol):
    name: str
    version: str

    def decide(self, signals: Mapping[str, float | None], case_view: CaseView) -> Decision:
        """signals: provider name -> value in [0, 1], or None when the provider failed."""
        ...


class ThresholdPolicy:
    """Fixed rules applied in order; the strictest mode any rule triggers wins.

    1. any model predicts a safety-critical class  -> second_review   "safety_class:<c>"
    2. agreement < 1.0                             -> second_review   "models_disagree"
    3. confidence < low                            -> second_review   "conf<low"
       confidence < high                           -> full_review     "conf<high"
       otherwise                                   -> light_review    "conf>=high"
    4. a needed signal is missing                  -> at least full   "missing_signal:<name>"
    """

    name = "threshold"

    def __init__(
        self,
        high: float,
        low: float,
        safety_critical_classes: Sequence[str],
        version: str,
        confidence_signal: str = "calibrated_confidence",
        agreement_signal: str | None = "agreement",
    ):
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"Need 0 <= low <= high <= 1, got low={low}, high={high}")
        self.high = float(high)
        self.low = float(low)
        self.safety_critical_classes = list(safety_critical_classes)
        self.version = str(version)
        self.confidence_signal = confidence_signal
        self.agreement_signal = agreement_signal

    @classmethod
    def from_config(cls, routing: Mapping[str, Any]) -> "ThresholdPolicy":
        """Build from the `routing:` section of the YAML config."""
        return cls(
            high=routing["thresholds"]["high"],
            low=routing["thresholds"]["low"],
            safety_critical_classes=routing["safety_critical_classes"],
            version=routing["policy_version"],
            agreement_signal="agreement" if len(routing["agreement_models"]) >= 2 else None,
        )

    def decide(self, signals: Mapping[str, float | None], case_view: CaseView) -> Decision:
        mode, reasons = ReviewMode.LIGHT, []

        def trigger(new: ReviewMode, reason: str) -> None:
            nonlocal mode
            mode = strictest(mode, new)
            if reason not in reasons:
                reasons.append(reason)

        for predicted in case_view.predictions.values():
            if predicted in self.safety_critical_classes:
                trigger(ReviewMode.SECOND, f"safety_class:{predicted}")

        missing = []
        if self.agreement_signal is not None:
            agreement = signals.get(self.agreement_signal)
            if agreement is None:
                missing.append(self.agreement_signal)
            elif agreement < 1.0:
                trigger(ReviewMode.SECOND, "models_disagree")

        confidence = signals.get(self.confidence_signal)
        if confidence is None:
            missing.append(self.confidence_signal)
        elif confidence < self.low:
            trigger(ReviewMode.SECOND, f"conf<{self.low:g}")
        elif confidence < self.high:
            trigger(ReviewMode.FULL, f"conf<{self.high:g}")
        else:
            trigger(ReviewMode.LIGHT, f"conf>={self.high:g}")

        for name in missing:
            trigger(ReviewMode.FULL, f"missing_signal:{name}")
        return Decision(mode=mode, reasons=reasons)
