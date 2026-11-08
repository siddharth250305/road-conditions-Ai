"""Review routing inside the live API: one decision per uploaded photo, reviews recorded later.

The offline check (`python scripts/route_images.py --config configs/routing_cnn.yaml
--bundle-dir serving`) routes the exported model's test probabilities and, only then, writes
<bundle>/routing.json: the routing config, the temperature fitted on val and the check's key
numbers, stamped with the model it was checked on. The API routes live photos only when that
file matches the model it serves, so it never applies thresholds that were checked on another
model.

Every decision and every review goes to the append-only DecisionLog. A light or full review is
done after one reviewer; a second review after two reviewers agree on the label.
"""

import json
import logging
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roadcond.routing.case import Case
from roadcond.routing.log import DecisionLog, config_hash, read_log
from roadcond.routing.policy import ReviewMode
from roadcond.routing.providers import CalibratedConfidenceProvider
from roadcond.routing.router import ReviewRouter
from roadcond.routing.run import build_policy

log = logging.getLogger(__name__)

ROUTING_FILE = "routing.json"
LIVE_FORMAT_VERSION = 1
LOG_UNAVAILABLE_REASON = "fallback:log_unavailable"
REVIEWS_NEEDED = {ReviewMode.LIGHT: 1, ReviewMode.FULL: 1, ReviewMode.SECOND: 2}
OFFLINE_KEYS = (
    "label",
    "probs_sha256",
    "n_images",
    "primary_accuracy",
    "modes",
    "error_rate_by_mode",
    "error_catch_rate",
    "safety_classes_to_second_review",
)


class LiveRoutingError(RuntimeError):
    """routing.json is missing, malformed, or was checked on a different model."""


class UnknownDecisionError(KeyError):
    pass


class ReviewConflictError(ValueError):
    """The review is not wanted: the decision is done, or this reviewer already reviewed it."""


def write_live_routing(
    bundle_dir: str | Path,
    routing: Mapping[str, Any],
    classes: Sequence[str],
    summary: Mapping[str, Any],
) -> Path:
    """Write <bundle>/routing.json after an offline check of the bundle's own model."""
    bundle = Path(bundle_dir)
    meta = json.loads((bundle / "meta.json").read_text())
    if meta.get("mock"):
        raise LiveRoutingError("The bundle is a mock model; routing is checked on real models only")
    if routing["primary_model"] != meta["model_name"]:
        raise LiveRoutingError(
            f"Checked {routing['primary_model']!r} but the bundle serves {meta['model_name']!r}"
        )
    if len(routing["agreement_models"]) >= 2:
        raise LiveRoutingError(
            "The API serves one model, so agreement_models must list fewer than 2"
        )
    if list(classes) != list(meta["classes"]):
        raise LiveRoutingError(
            f"Checked classes {list(classes)} but the bundle has {meta['classes']}"
        )
    spec = {
        "format_version": LIVE_FORMAT_VERSION,
        "model_name": meta["model_name"],
        "onnx_sha256": meta["provenance"]["onnx_sha256"],
        "classes": list(classes),
        "temperature": summary["calibration"]["temperature"],
        "routing": dict(routing),
        "config_hash": config_hash(routing),
        "offline_check": {k: summary[k] for k in OFFLINE_KEYS},
    }
    path = bundle / ROUTING_FILE
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return path


def load_live_routing(bundle_dir: str | Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    """routing.json, checked against the served model's meta.json."""
    path = Path(bundle_dir) / ROUTING_FILE
    if not path.is_file():
        raise LiveRoutingError(f"{path} not found: run the offline routing check with --bundle-dir")
    try:
        spec = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise LiveRoutingError(f"{path} is not valid JSON: {err}") from None
    if spec.get("format_version") != LIVE_FORMAT_VERSION:
        raise LiveRoutingError(f"Unsupported routing.json format {spec.get('format_version')!r}")
    served_sha = meta.get("provenance", {}).get("onnx_sha256")
    if spec.get("onnx_sha256") != served_sha or spec.get("model_name") != meta.get("model_name"):
        raise LiveRoutingError(
            "routing.json was checked on a different model; rerun the offline routing check"
        )
    if spec.get("classes") != list(meta.get("classes", [])):
        raise LiveRoutingError("routing.json classes differ from the served model's")
    if spec.get("config_hash") != config_hash(spec.get("routing", {})):
        raise LiveRoutingError("routing.json config_hash does not match its routing section")
    return spec


def review_status(mode: str, labels: Sequence[str]) -> tuple[str, str | None]:
    """("pending" | "disputed" | "done", the agreed label once done)."""
    needed = REVIEWS_NEEDED[ReviewMode(mode)]
    if not labels:
        return "pending", None
    label, votes = Counter(labels).most_common(1)[0]
    if votes >= needed:
        return "done", label
    return ("disputed" if len(labels) >= needed else "pending"), None


def outcome_label(outcome: str) -> str:
    """ "confirmed:wet" or "corrected:wet" -> "wet"."""
    return outcome.split(":", 1)[-1]


class LiveRouting:
    """Routes one photo at a time and records reviews; safe to share between threads."""

    def __init__(self, bundle_dir: str | Path, meta: Mapping[str, Any], log_path: str | Path):
        spec = load_live_routing(bundle_dir, meta)
        routing = spec["routing"]
        self.spec = spec
        self.model = spec["model_name"]
        self.classes: list[str] = list(spec["classes"])
        self.log_path = Path(log_path)
        self.log = DecisionLog(self.log_path, spec["config_hash"])
        self.router = ReviewRouter(
            [CalibratedConfidenceProvider(self.model, float(spec["temperature"]))],
            build_policy(routing),
            self.log,
        )
        self._lock = threading.Lock()
        self._decisions: dict[str, dict[str, Any]] = {}
        if self.log_path.exists():
            try:
                decisions, outcomes = read_log(self.log_path)
            except ValueError as err:
                raise LiveRoutingError(
                    f"Decision log {self.log_path} is unreadable: {err}"
                ) from None
            for d in decisions:
                self._decisions[d["decision_id"]] = {
                    "decision_id": d["decision_id"],
                    "routed_at": d["timestamp"],
                    "prediction": d["predictions"].get(self.model),
                    "mode": d["mode"],
                    "reasons": d["reasons"],
                    "reviews": [
                        {"reviewer": o["reviewer"], "label": outcome_label(o["outcome"])}
                        for o in outcomes[d["decision_id"]]
                    ],
                }

    @property
    def info(self) -> dict[str, Any]:
        routing = self.spec["routing"]
        return {
            "policy": {"name": routing["policy"], "version": routing["policy_version"]},
            "thresholds": dict(routing["thresholds"]),
            "safety_critical_classes": list(routing["safety_critical_classes"]),
            "temperature": self.spec["temperature"],
            "config_hash": self.spec["config_hash"],
            "offline_check": self.spec["offline_check"],
        }

    def route(self, image_id: str, probabilities: Mapping[str, float]) -> dict[str, Any]:
        """Route one photo. If the log cannot be written, it gets an unrecorded full review."""
        case = Case(image_id, self.classes, {self.model: [probabilities[c] for c in self.classes]})
        with self._lock:
            try:
                routed = self.router.route(case)
            except Exception as err:  # an unrecorded decision must not be a light one
                log.error("Routing decision not recorded: %s", err)
                return {
                    "decision_id": None,
                    "mode": str(ReviewMode.FULL),
                    "reasons": [LOG_UNAVAILABLE_REASON],
                    "fallback": True,
                    "reviews_needed": REVIEWS_NEEDED[ReviewMode.FULL],
                    "recorded": False,
                }
            entry = {
                "decision_id": routed.decision_id,
                "routed_at": datetime.now(UTC).isoformat(),
                "prediction": routed.predictions.get(self.model),
                "mode": str(routed.mode),
                "reasons": list(routed.reasons),
                "reviews": [],
            }
            self._decisions[routed.decision_id] = entry
        return {
            "decision_id": routed.decision_id,
            "mode": entry["mode"],
            "reasons": entry["reasons"],
            "fallback": routed.fallback,
            "reviews_needed": REVIEWS_NEEDED[routed.mode],
            "recorded": True,
        }

    def _state(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        status, label = review_status(entry["mode"], [r["label"] for r in entry["reviews"]])
        return {
            "decision_id": entry["decision_id"],
            "routed_at": entry["routed_at"],
            "prediction": entry["prediction"],
            "mode": entry["mode"],
            "reasons": list(entry["reasons"]),
            "status": status,
            "agreed_label": label,
            "reviews_needed": REVIEWS_NEEDED[ReviewMode(entry["mode"])],
            "reviewers": [r["reviewer"] for r in entry["reviews"]],  # never the labels
        }

    def review(self, decision_id: str, label: str, reviewer: str) -> dict[str, Any]:
        """Record one reviewer's label for a decision; returns its new state."""
        if label not in self.classes:
            raise ValueError(f"Unknown class {label!r}")
        with self._lock:
            entry = self._decisions.get(decision_id)
            if entry is None:
                raise UnknownDecisionError(decision_id)
            if self._state(entry)["status"] == "done":
                raise ReviewConflictError("This photo has already been reviewed")
            if any(r["reviewer"] == reviewer for r in entry["reviews"]):
                raise ReviewConflictError("You have already reviewed this photo")
            verb = "confirmed" if label == entry["prediction"] else "corrected"
            self.log.record_outcome(decision_id, f"{verb}:{label}", reviewer)
            entry["reviews"].append({"reviewer": reviewer, "label": label})
            return self._state(entry)

    def queue(self, reviewer: str, limit: int = 50) -> list[dict[str, Any]]:
        """Open decisions this reviewer has not reviewed: second, then full, then light; oldest
        first within a mode."""
        order = {str(m): -m.rank for m in ReviewMode}
        with self._lock:
            open_items = [
                self._state(e)
                for e in self._decisions.values()
                if all(r["reviewer"] != reviewer for r in e["reviews"])
            ]
        open_items = [s for s in open_items if s["status"] != "done"]
        open_items.sort(key=lambda s: (order[s["mode"]], s["routed_at"]))
        return open_items[:limit]

    def counts(self) -> dict[str, dict[str, int]]:
        """Open and done decisions per mode."""
        with self._lock:
            states = [self._state(e) for e in self._decisions.values()]
        out = {str(m): {"open": 0, "done": 0} for m in ReviewMode}
        for s in states:
            out[s["mode"]]["done" if s["status"] == "done" else "open"] += 1
        return out
