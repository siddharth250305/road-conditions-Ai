"""A learned review policy: predicted risk that the primary model is wrong picks the mode.

    signals (confidence, margin, agreement) -> logistic regression -> risk in [0, 1]
    hard rules first (safety class, disagreement, missing signal), then risk >= cut_second ->
    second_review, risk >= cut_full -> full_review, otherwise light_review

The weights are fitted on the validation split only (fit_policy) and saved as JSON. The policy
version carries a short sha256 of that file, so every logged decision names the exact weights.
The cutoffs follow the equal-workload rule: on val, the learned policy sends as many images to
each mode as the threshold policy does, so a comparison is about which images get the extra
review, not how many.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from roadcond.config import resolve_path
from roadcond.routing.case import CaseView
from roadcond.routing.policy import Decision, ReviewMode, strictest

FORMAT_VERSION = 1
FIT_SPLIT = "val"  # the only split the policy may learn from
CONFIDENCE_SIGNAL = "calibrated_confidence"
MARGIN_SIGNAL = "margin"
AGREEMENT_SIGNAL = "agreement"
LEARNED_PROVIDERS = (MARGIN_SIGNAL,)  # run.PROVIDERS entries the features need
RANKS = {m: m.rank for m in ReviewMode}
ABOVE_ONE = math.nextafter(1.0, 2.0)  # a cutoff no risk reaches


class LearnedPolicyError(ValueError):
    """The weights file is missing, malformed, or was fitted for another setup."""


# --- the model: NumPy logistic regression ---------------------------------------------------


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """1 / (1 + exp(-z)) without overflow."""
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


@dataclass(frozen=True)
class LogisticFit:
    intercept: float
    coef: np.ndarray
    iterations: int
    converged: bool
    log_loss: float  # mean, on the fitted rows


def fit_logistic(
    X: np.ndarray, y: np.ndarray, l2: float, max_iter: int = 100, tol: float = 1e-9
) -> LogisticFit:
    """Logistic regression by Newton's method (IRLS) with step halving; deterministic.

    Minimises sum(log loss) + l2 / 2 * ||coef||^2; the intercept is not penalised.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or not len(X):
        raise ValueError(f"Need X (N, D) and y (N,) with N > 0, got {X.shape}, {y.shape}")
    if not np.isfinite(X).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("X must be finite and y must be 0 or 1")
    if not (math.isfinite(l2) and l2 >= 0):
        raise ValueError(f"l2 must be a number >= 0, got {l2}")
    A = np.hstack([np.ones((len(X), 1)), X])
    penalty = np.full(A.shape[1], float(l2))
    penalty[0] = 0.0

    def objective(w: np.ndarray) -> float:
        z = A @ w
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(penalty * w * w))

    w = np.zeros(A.shape[1])
    loss = objective(w)
    converged, iterations = False, 0
    while iterations < max_iter:
        iterations += 1
        p = sigmoid(A @ w)
        grad = A.T @ (p - y) + penalty * w
        hessian = (A * (p * (1 - p))[:, None]).T @ A + np.diag(penalty)
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            raise ValueError("Singular Hessian: raise l2 or drop a constant feature") from None
        t = 1.0
        while (new_loss := objective(w - t * step)) > loss and t > 1e-10:
            t /= 2
        w, loss = w - t * step, new_loss
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
    z = A @ w
    log_loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    return LogisticFit(float(w[0]), w[1:].copy(), iterations, converged, log_loss)


def standardisation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and stds; a constant column gets std 1 so it standardises to 0."""
    X = np.asarray(X, dtype=float)
    means, stds = X.mean(axis=0), X.std(axis=0)
    return means, np.where(stds > 0, stds, 1.0)


# --- choosing the cutoffs (on val) ----------------------------------------------------------


def _levels(risks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct risks, highest first, and how many risks are >= each."""
    values, counts = np.unique(np.asarray(risks, dtype=float), return_counts=True)
    return values[::-1], np.cumsum(counts[::-1])


def _cut_at(values: np.ndarray, i: int) -> float:
    """A cutoff that keeps values[:i] (the i highest levels) at or above it: a midpoint, so
    a recomputed risk that differs in the last bit cannot cross it."""
    if i == 0:
        return float((values[0] + 1.0) / 2) if len(values) and values[0] < 1.0 else ABOVE_ONE
    if i == len(values):
        return float(values[-1] / 2)
    return float((values[i - 1] + values[i]) / 2)


def cutoff_for_count(risks: np.ndarray, k: int) -> float:
    """A cutoff c with #(risks >= c) as close to k as ties allow (a tie goes to the larger)."""
    values, at_or_above = _levels(risks)
    counts = np.concatenate([[0], at_or_above])
    best = np.abs(counts - k).min()
    i = int(np.flatnonzero(np.abs(counts - k) == best)[-1])  # the larger count on a tie
    return _cut_at(values, i)


def modes_from_risk(
    risk: np.ndarray, floor: np.ndarray, cut_full: float, cut_second: float
) -> np.ndarray:
    """Mode ranks (0 light, 1 full, 2 second) as LearnedPolicy.decide gives them, vectorised.

    floor: the rank the hard rules alone give; risk: NaN where a signal was missing.
    """
    risk, floor = np.asarray(risk, dtype=float), np.asarray(floor)
    known = np.isfinite(risk)
    by_risk = np.where(risk >= cut_second, 2, np.where(risk >= cut_full, 1, 0))
    return np.maximum(floor, np.where(known, by_risk, 1))


def equal_workload_cutoffs(
    risk: np.ndarray, floor: np.ndarray, target: Mapping[str, int]
) -> tuple[float, float]:
    """(cut_full, cut_second) giving per-mode counts as close to target as ties allow."""
    risk, floor = np.asarray(risk, dtype=float), np.asarray(floor)
    known = np.isfinite(risk)
    second_pool = known & (floor < RANKS[ReviewMode.SECOND])
    forced_second = int((floor == RANKS[ReviewMode.SECOND]).sum())
    k_second = max(target[str(ReviewMode.SECOND)] - forced_second, 0)
    cut_second = cutoff_for_count(risk[second_pool], k_second)
    light_pool = known & (floor == RANKS[ReviewMode.LIGHT])
    k_strict = max(int(light_pool.sum()) - target[str(ReviewMode.LIGHT)], 0)
    cut_full = cutoff_for_count(risk[light_pool], k_strict)
    return min(cut_full, cut_second), cut_second


def equal_catch_cut_full(
    risk: np.ndarray, floor: np.ndarray, wrong: np.ndarray, target_rate: float
) -> float:
    """The highest cut_full (most light reviews) whose error catch rate is >= target_rate.

    Catch rate = share of wrong predictions routed to full or second review; it depends on
    cut_full only (cut_second just splits the strict modes).
    """
    risk, floor = np.asarray(risk, dtype=float), np.asarray(floor)
    wrong = np.asarray(wrong, dtype=bool)
    if not wrong.any():
        return ABOVE_ONE
    light_pool = np.isfinite(risk) & (floor == RANKS[ReviewMode.LIGHT])
    forced_caught = int((wrong & ~light_pool).sum())
    pool_risk, pool_wrong = risk[light_pool], wrong[light_pool]
    values, at_or_above = _levels(pool_risk)
    wrong_by_risk = np.cumsum(pool_wrong[np.argsort(-pool_risk, kind="stable")])
    errors_at = wrong_by_risk[at_or_above - 1] if len(values) else np.empty(0, dtype=int)
    caught = np.concatenate([[0], errors_at]) + forced_caught
    reached = np.flatnonzero(caught / wrong.sum() >= target_rate - 1e-12)
    return _cut_at(values, int(reached[0]) if len(reached) else len(values))


# --- the policy -----------------------------------------------------------------------------


def feature_names(routing: Mapping[str, Any]) -> list[str]:
    """The signals the risk model reads: agreement only when two or more models are listed."""
    names = [CONFIDENCE_SIGNAL, MARGIN_SIGNAL]
    if len(routing["agreement_models"]) >= 2:
        names.append(AGREEMENT_SIGNAL)
    return names


def with_learned_signals(routing: Mapping[str, Any]) -> dict[str, Any]:
    """The routing config with the providers the learned features need switched on."""
    extra = list(routing.get("extra_providers", []))
    return {**routing, "extra_providers": extra + [p for p in LEARNED_PROVIDERS if p not in extra]}


def hard_rules(
    signals: Mapping[str, float | None],
    case_view: CaseView,
    safety_critical_classes: Sequence[str],
    required: Sequence[str],
    agreement_signal: str | None,
) -> tuple[ReviewMode, list[str]]:
    """The rules no model can override, and the lightest mode they leave.

    any model predicts a safety-critical class -> second_review   "safety_class:<c>"
    any model's probabilities are unusable     -> at least full   "unusable_prediction:<m>"
    agreement < 1.0                            -> second_review   "models_disagree"
    a required signal is missing               -> at least full   "missing_signal:<name>"
    """
    mode, reasons = ReviewMode.LIGHT, []

    def trigger(new: ReviewMode, reason: str) -> None:
        nonlocal mode
        mode = strictest(mode, new)
        if reason not in reasons:
            reasons.append(reason)

    for model, predicted in case_view.predictions.items():
        if predicted is None:
            trigger(ReviewMode.FULL, f"unusable_prediction:{model}")
        elif predicted in safety_critical_classes:
            trigger(ReviewMode.SECOND, f"safety_class:{predicted}")
    if agreement_signal is not None:
        agreement = signals.get(agreement_signal)
        if agreement is not None and agreement < 1.0:
            trigger(ReviewMode.SECOND, "models_disagree")
    for name in dict.fromkeys([*([agreement_signal] if agreement_signal else []), *required]):
        if signals.get(name) is None:
            trigger(ReviewMode.FULL, f"missing_signal:{name}")
    return mode, reasons


@dataclass(frozen=True)
class RiskModel:
    """Standardised logistic regression over named signals."""

    features: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    intercept: float
    coef: np.ndarray

    def risk_matrix(self, X: np.ndarray) -> np.ndarray:
        """Risk per row of X (columns in self.features order)."""
        z = self.intercept + ((np.asarray(X, dtype=float) - self.means) / self.stds) @ self.coef
        return sigmoid(z)

    def risk(self, signals: Mapping[str, float | None]) -> float | None:
        values = [signals.get(f) for f in self.features]
        if any(v is None for v in values):
            return None
        return float(self.risk_matrix(np.array([values], dtype=float))[0])


class LearnedPolicy:
    """Hard safety rules, then predicted risk against two cutoffs (see the module docstring)."""

    name = "learned"

    def __init__(
        self,
        model: RiskModel,
        cut_full: float,
        cut_second: float,
        safety_critical_classes: Sequence[str],
        version: str,
        agreement_signal: str | None = AGREEMENT_SIGNAL,
    ):
        if not 0.0 <= cut_full <= cut_second <= ABOVE_ONE:
            raise ValueError(f"Need 0 <= cut_full <= cut_second, got {cut_full}, {cut_second}")
        self.model = model
        self.cut_full = float(cut_full)
        self.cut_second = float(cut_second)
        self.safety_critical_classes = list(safety_critical_classes)
        self.version = str(version)
        self.agreement_signal = agreement_signal

    def with_cutoffs(self, cut_full: float, cut_second: float, tag: str) -> "LearnedPolicy":
        """The same weights with other cutoffs; tag goes into the version."""
        return LearnedPolicy(
            self.model,
            cut_full,
            cut_second,
            self.safety_critical_classes,
            f"{self.version}+{tag}",
            self.agreement_signal,
        )

    def floor(
        self, signals: Mapping[str, float | None], case_view: CaseView
    ) -> tuple[ReviewMode, list[str]]:
        return hard_rules(
            signals,
            case_view,
            self.safety_critical_classes,
            self.model.features,
            self.agreement_signal,
        )

    def decide(self, signals: Mapping[str, float | None], case_view: CaseView) -> Decision:
        mode, reasons = self.floor(signals, case_view)
        risk = self.model.risk(signals)
        if risk is None:  # a missing signal: hard_rules already made it at least full
            return Decision(mode=mode, reasons=reasons)
        if risk >= self.cut_second:
            by_risk, reason = ReviewMode.SECOND, f"risk>={self.cut_second:.3g}"
        elif risk >= self.cut_full:
            by_risk, reason = ReviewMode.FULL, f"risk>={self.cut_full:.3g}"
        else:
            by_risk, reason = ReviewMode.LIGHT, f"risk<{self.cut_full:.3g}"
        return Decision(mode=strictest(mode, by_risk), reasons=[*reasons, reason])

    @classmethod
    def from_weights(
        cls, weights: Mapping[str, Any], routing: Mapping[str, Any], weights_sha256: str
    ) -> "LearnedPolicy":
        model = RiskModel(
            features=tuple(weights["features"]),
            means=np.array(weights["standardisation"]["means"], dtype=float),
            stds=np.array(weights["standardisation"]["stds"], dtype=float),
            intercept=float(weights["intercept"]),
            coef=np.array(weights["coefficients"], dtype=float),
        )
        cuts = weights["cutoffs"]
        return cls(
            model,
            cuts["cut_full"],
            cuts["cut_second"],
            routing["safety_critical_classes"],
            f"{routing['policy_version']}+{weights_sha256[:12]}",
            AGREEMENT_SIGNAL if len(routing["agreement_models"]) >= 2 else None,
        )

    @classmethod
    def from_config(cls, routing: Mapping[str, Any]) -> "LearnedPolicy":
        """Load routing.learned_weights, refusing a file fitted for another setup."""
        path = resolve_path(routing["learned_weights"])
        weights, sha = load_weights(path)
        check_weights(weights, routing, file_sha256(resolve_path(routing["probs_file"])), path)
        return cls.from_weights(weights, routing, sha)


# --- the weights file -----------------------------------------------------------------------


@dataclass(frozen=True)
class SplitFeatures:
    """Signals and outcomes of one split, row-aligned; built by roadcond.routing.compare."""

    split: str
    X: np.ndarray  # (N, D) signal values in feature order; NaN where a signal is missing
    wrong: np.ndarray  # (N,) the primary model's prediction is wrong
    floor: np.ndarray  # (N,) mode rank the hard rules alone give
    threshold_modes: np.ndarray  # (N,) the threshold policy's mode on this split


def mode_counts_of(ranks: np.ndarray) -> dict[str, int]:
    ranks = np.asarray(ranks)
    return {str(m): int((ranks == m.rank).sum()) for m in ReviewMode}


def fit_policy(
    val: SplitFeatures, features: Sequence[str], l2: float, max_iter: int = 100
) -> dict[str, Any]:
    """Fit the risk model and the equal-workload cutoffs on val; the core of the weights file.

    Refuses any split but val, so test data can never shape the weights or the cutoffs.
    """
    if val.split != FIT_SPLIT:
        raise ValueError(f"The learned policy is fitted on {FIT_SPLIT!r} only, got {val.split!r}")
    complete = np.isfinite(val.X).all(axis=1)
    X, y = val.X[complete], val.wrong[complete].astype(float)
    means, stds = standardisation(X)
    fit = fit_logistic((X - means) / stds, y, l2, max_iter)
    model = RiskModel(tuple(features), means, stds, fit.intercept, fit.coef)
    risk = np.full(len(val.X), np.nan)
    risk[complete] = model.risk_matrix(X)
    target = {str(m): int((val.threshold_modes == str(m)).sum()) for m in ReviewMode}
    cut_full, cut_second = equal_workload_cutoffs(risk, val.floor, target)
    achieved = mode_counts_of(modes_from_risk(risk, val.floor, cut_full, cut_second))
    return {
        "features": list(features),
        "standardisation": {"means": means.tolist(), "stds": stds.tolist()},
        "intercept": fit.intercept,
        "coefficients": fit.coef.tolist(),
        "l2": float(l2),
        "fitted_on": FIT_SPLIT,
        "cutoffs": {"rule": "equal_workload", "cut_full": cut_full, "cut_second": cut_second},
        "fit": {
            "iterations": fit.iterations,
            "converged": fit.converged,
            "log_loss_val": fit.log_loss,
        },
        "val": {
            "n_images": len(val.X),
            "n_fitted": int(complete.sum()),
            "n_errors": int(val.wrong.sum()),
            "threshold_counts": target,
            "learned_counts": achieved,
            "max_count_gap": max(abs(achieved[m] - target[m]) for m in target),
        },
    }


@lru_cache(maxsize=16)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """sha256 of a file, cached while its size and modification time stay the same."""
    stat = Path(path).stat()
    return _sha256(str(path), stat.st_size, stat.st_mtime_ns)


def matched_setup(routing: Mapping[str, Any]) -> dict[str, Any]:
    """The routing settings the weights and cutoffs depend on; a change needs a refit."""
    return {
        "primary_model": routing["primary_model"],
        "agreement_models": list(routing["agreement_models"]),
        "safety_critical_classes": list(routing["safety_critical_classes"]),
        "temperature_search": dict(routing["temperature_search"]),
        "l2": float(routing["l2"]),
    }


def write_weights(
    path: str | Path,
    core: Mapping[str, Any],
    routing: Mapping[str, Any],
    temperature: float,
    probs_path: Path,
) -> str:
    """Write the weights file; returns its sha256."""
    weights = {
        "format_version": FORMAT_VERSION,
        "policy": LearnedPolicy.name,
        "created_at": datetime.now(UTC).isoformat(),
        "setup": matched_setup(routing),
        # The threshold workload the cutoffs were matched to (recorded, not enforced: the
        # learned policy does not use the thresholds).
        "matched_thresholds": dict(routing["thresholds"]),
        "temperature": temperature,
        **core,
        "probs_file": str(probs_path),
        "probs_sha256": file_sha256(probs_path),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2, allow_nan=False) + "\n")
    return file_sha256(path)


def load_weights(path: str | Path) -> tuple[dict[str, Any], str]:
    """(weights, sha256 of the file); LearnedPolicyError if it is missing or malformed."""
    path = Path(path)
    if not path.is_file():
        raise LearnedPolicyError(
            f"{path} not found: fit it first with python scripts/fit_learned_policy.py --config ..."
        )
    try:
        weights = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise LearnedPolicyError(f"{path} is not valid JSON: {err}") from None
    if not isinstance(weights, dict) or weights.get("format_version") != FORMAT_VERSION:
        raise LearnedPolicyError(f"{path}: not a learned-policy file of format {FORMAT_VERSION}")
    return weights, file_sha256(path)


def check_weights(
    weights: Mapping[str, Any], routing: Mapping[str, Any], probs_sha256: str, path: Path
) -> None:
    """Refuse weights fitted on other probabilities, other features or another setup."""
    if weights.get("probs_sha256") != probs_sha256:
        raise LearnedPolicyError(
            f"{path} was fitted on a different probabilities file than {routing['probs_file']}; "
            "refit it"
        )
    if weights.get("setup") != matched_setup(routing):
        raise LearnedPolicyError(f"{path} was fitted for other routing settings; refit it")
    if weights.get("features") != feature_names(routing):
        raise LearnedPolicyError(f"{path} uses features {weights.get('features')}; refit it")
    providers = {CONFIDENCE_SIGNAL, *routing.get("extra_providers", [])}
    if len(routing["agreement_models"]) >= 2:
        providers.add(AGREEMENT_SIGNAL)
    absent = [f for f in weights["features"] if f not in providers]
    if absent:
        raise LearnedPolicyError(
            f"The learned policy needs signals {absent}: add them to routing.extra_providers"
        )
