"""Append-only JSONL audit log of routing decisions and the human outcomes recorded later.

Lines are only ever appended. A decision line has human_outcome null; a reviewer's verdict is a
separate {"event": "outcome", ...} line, and read_decisions joins the two. One DecisionLog may be
shared by the threads of a web server: each line is written whole, under a lock.
"""

import hashlib
import json
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roadcond.routing.router import RoutedDecision

SCHEMA_VERSION = 1
OUTCOME_EVENT = "outcome"
DECISION_FIELDS = (
    "schema_version",
    "decision_id",
    "timestamp",
    "image_id",
    "predictions",
    "signals",
    "provider_errors",
    "policy",
    "config_hash",
    "mode",
    "reasons",
    "fallback",
    "human_outcome",
)


def config_hash(routing_config: Mapping[str, Any]) -> str:
    """sha256 of the routing config section, independent of key order."""
    text = json.dumps(routing_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DecisionLog:
    """Writes one JSON line per decision or outcome to path; never rewrites a line."""

    def __init__(self, path: str | Path, config_hash: str):
        self.path = Path(path)
        self.config_hash = config_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, allow_nan=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)

    def write(self, decision: RoutedDecision) -> dict[str, Any]:
        record = {
            "schema_version": SCHEMA_VERSION,
            "decision_id": decision.decision_id,
            "timestamp": _now(),
            "image_id": decision.image_id,
            "predictions": decision.predictions,
            "signals": decision.signals,
            "provider_errors": decision.provider_errors,
            "policy": {"name": decision.policy_name, "version": decision.policy_version},
            "config_hash": self.config_hash,
            "mode": str(decision.mode),
            "reasons": decision.reasons,
            "fallback": decision.fallback,
            "human_outcome": None,
        }
        self._append(record)
        return record

    def record_outcome(self, decision_id: str, outcome: str, reviewer: str) -> dict[str, Any]:
        """Append a reviewer's verdict for an earlier decision."""
        if not outcome or not reviewer:
            raise ValueError("outcome and reviewer must be non-empty")
        record = {
            "schema_version": SCHEMA_VERSION,
            "event": OUTCOME_EVENT,
            "decision_id": decision_id,
            "timestamp": _now(),
            "outcome": outcome,
            "reviewer": reviewer,
        }
        self._append(record)
        return record


def read_log(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """(decisions in log order, every outcome per decision_id in log order).

    Raises ValueError, naming the line, for a malformed line, a duplicate decision, or an
    outcome that names no decision in the log.
    """
    decisions: dict[str, dict[str, Any]] = {}
    outcomes: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:  # e.g. a write cut off by a crash
                raise ValueError(f"line {number}: not valid JSON ({err})") from None
            if not isinstance(record, dict) or "decision_id" not in record:
                raise ValueError(f"line {number}: not a decision or outcome record")
            if record.get("event") == OUTCOME_EVENT:
                outcomes.append(record)
            elif record["decision_id"] in decisions:
                raise ValueError(f"line {number}: duplicate decision {record['decision_id']}")
            else:
                decisions[record["decision_id"]] = dict(record)
    by_decision: dict[str, list[dict[str, Any]]] = {d: [] for d in decisions}
    for outcome in outcomes:
        if outcome["decision_id"] not in decisions:
            raise ValueError(f"Outcome for unknown decision {outcome['decision_id']}")
        by_decision[outcome["decision_id"]].append(
            {k: outcome[k] for k in ("outcome", "reviewer", "timestamp")}
        )
    return list(decisions.values()), by_decision


def read_decisions(path: str | Path) -> list[dict[str, Any]]:
    """Decisions in log order, each with human_outcome set to its latest outcome (or None).

    Raises ValueError as read_log does.
    """
    decisions, outcomes = read_log(path)
    for decision in decisions:
        history = outcomes[decision["decision_id"]]
        decision["human_outcome"] = history[-1] if history else None
    return decisions
