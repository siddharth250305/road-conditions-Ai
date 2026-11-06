"""Confidence-based review routing: how much human review each image needs, with an audit log.

Standard library and NumPy only (no torch, sklearn or onnxruntime), so it can be tested and
reused anywhere.

    providers (signals in [0, 1]) -> policy (review mode + reasons) -> DecisionLog (JSONL)
                              \\________ ReviewRouter ________/
"""

from roadcond.routing.case import Case, CaseView
from roadcond.routing.log import DecisionLog, config_hash, read_decisions
from roadcond.routing.policy import Decision, ReviewMode, ReviewPolicy, ThresholdPolicy
from roadcond.routing.providers import (
    AgreementProvider,
    CalibratedConfidenceProvider,
    ConfidenceProvider,
    ProviderError,
)
from roadcond.routing.router import ReviewRouter, RoutedDecision

__all__ = [
    "AgreementProvider",
    "CalibratedConfidenceProvider",
    "Case",
    "CaseView",
    "ConfidenceProvider",
    "Decision",
    "DecisionLog",
    "ProviderError",
    "ReviewMode",
    "ReviewPolicy",
    "ReviewRouter",
    "RoutedDecision",
    "ThresholdPolicy",
    "config_hash",
    "read_decisions",
]
