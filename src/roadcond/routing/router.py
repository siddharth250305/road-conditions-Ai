"""The review router: collect signals, ask the policy, record the decision.

It knows only the ConfidenceProvider and ReviewPolicy interfaces; every concrete provider and
policy comes in through the constructor. A failure can make the review stricter, never lighter.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from roadcond.routing.case import Case, is_valid_signal
from roadcond.routing.policy import ReviewMode, ReviewPolicy, strictest
from roadcond.routing.providers import ConfidenceProvider

if TYPE_CHECKING:
    from roadcond.routing.log import DecisionLog

POLICY_ERROR_REASON = "fallback:policy_error"
PROVIDER_ERROR_REASON = "fallback:provider_error"


@dataclass(frozen=True)
class RoutedDecision:
    """Everything one routing decision rests on; the log writes one line from it."""

    decision_id: str
    image_id: str
    predictions: dict[str, str | None]
    signals: dict[str, float | None]
    provider_errors: dict[str, str]
    policy_name: str
    policy_version: str
    mode: ReviewMode
    reasons: list[str] = field(default_factory=list)
    fallback: bool = False


class ReviewRouter:
    def __init__(
        self,
        providers: Sequence[ConfidenceProvider],
        policy: ReviewPolicy,
        log: "DecisionLog | None" = None,
    ):
        names = [p.name for p in providers]
        if len(set(names)) != len(names):
            raise ValueError(f"Provider names must be unique, got {names}")
        self.providers = list(providers)
        self.policy = policy
        self.log = log

    def _signals(self, view) -> tuple[dict[str, float | None], dict[str, str]]:
        signals: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for provider in self.providers:
            try:
                value = provider.score(view)
            except Exception as err:  # ProviderError or anything else: the signal is missing
                signals[provider.name] = None
                errors[provider.name] = f"{type(err).__name__}: {err}"
                continue
            if is_valid_signal(value):
                signals[provider.name] = float(value)
            else:
                signals[provider.name] = None
                errors[provider.name] = f"invalid signal {value!r}: need a number in [0, 1]"
        return signals, errors

    def route(self, case: Case) -> RoutedDecision:
        view = case.view()  # providers and the policy never see true_label
        signals, errors = self._signals(view)
        fallback = bool(errors)
        try:
            decision = self.policy.decide(dict(signals), view)
            mode = ReviewMode(decision.mode)
            reasons = [str(r) for r in decision.reasons]
        except Exception as err:
            mode, reasons, fallback = ReviewMode.FULL, [POLICY_ERROR_REASON], True
            errors = {**errors, "policy": f"{type(err).__name__}: {err}"}
        if errors and mode.rank < ReviewMode.FULL.rank:
            # Guard against a policy that ignores missing signals.
            mode = strictest(mode, ReviewMode.FULL)
            reasons.append(PROVIDER_ERROR_REASON)
        routed = RoutedDecision(
            decision_id=str(uuid.uuid4()),
            image_id=case.image_id,
            predictions=dict(view.predictions),
            signals=signals,
            provider_errors=errors,
            policy_name=self.policy.name,
            policy_version=self.policy.version,
            mode=mode,
            reasons=reasons,
            fallback=fallback,
        )
        if self.log is not None:
            self.log.write(routed)
        return routed
