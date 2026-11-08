"""How live review routing is doing, from the API's decision log and the reviewers' labels.

    python scripts/review_report.py [--log logs/decisions.jsonl] [--bundle serving] [--out FILE]

The offline check measured the model against RSCD's labels; live, the reviewers' agreed label
is the reference. The main question is the light-review error rate: those photos get only a
glance, so if the model is wrong there more often than the offline check said (new cameras, a
new season), the temperature and `high` threshold need refitting. Only finished reviews count,
and only decisions made with the current routing.json config.
"""

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roadcond.routing.live import ROUTING_FILE, outcome_label, review_status
from roadcond.routing.log import read_log
from roadcond.routing.policy import ReviewMode

MIN_REVIEWS = 30  # fewer finished light reviews than this: too few to judge


def wilson_interval(errors: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a rate of errors/n; None when n is 0."""
    if n == 0:
        return None
    p = errors / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(errors: int, n: int) -> float | None:
    return errors / n if n else None


def review_report(
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, Sequence[Mapping[str, Any]]],
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarise live decisions and finished reviews; compare with spec's offline check."""
    hashes = Counter(d["config_hash"] for d in decisions)
    current = spec["config_hash"] if spec else None
    if current is not None:
        decisions = [d for d in decisions if d["config_hash"] == current]
    model = spec["model_name"] if spec else None
    safety = list(spec["routing"]["safety_critical_classes"]) if spec else []
    modes = [str(m) for m in ReviewMode]

    per_mode = {m: {"open": 0, "done": 0, "errors": 0} for m in modes}
    caught = errors = 0
    safety_seen = {c: {"count": 0, "second_review": 0} for c in safety}
    second_pairs = second_split = 0
    for d in decisions:
        labels = [outcome_label(o["outcome"]) for o in outcomes.get(d["decision_id"], [])]
        status, agreed = review_status(d["mode"], labels)
        if d["mode"] == str(ReviewMode.SECOND) and len(labels) >= 2:  # done or disputed
            second_pairs += 1
            second_split += labels[0] != labels[1]
        row = per_mode[d["mode"]]
        if status != "done":
            row["open"] += 1
            continue
        row["done"] += 1
        predictions = d["predictions"]
        predicted = predictions.get(model) if model else next(iter(predictions.values()), None)
        if predicted != agreed:
            row["errors"] += 1
            errors += 1
            caught += d["mode"] != str(ReviewMode.LIGHT)
        if agreed in safety_seen:
            safety_seen[agreed]["count"] += 1
            safety_seen[agreed]["second_review"] += d["mode"] == str(ReviewMode.SECOND)

    reviews = {}
    for m, row in per_mode.items():
        ci = wilson_interval(row["errors"], row["done"])
        reviews[m] = {**row, "error_rate": _rate(row["errors"], row["done"]), "ci95": ci}

    light = reviews[str(ReviewMode.LIGHT)]
    offline_rate = None
    if spec:
        offline = spec["offline_check"]["error_rate_by_mode"].get(str(ReviewMode.LIGHT), {})
        offline_rate = offline.get("error_rate")
    enough = light["done"] >= MIN_REVIEWS
    alert = bool(
        enough and offline_rate is not None and light["ci95"] and light["ci95"][0] > offline_rate
    )
    if not enough:
        verdict = f"too few finished light reviews to judge ({light['done']} < {MIN_REVIEWS})"
    elif offline_rate is None:
        verdict = "no offline check to compare with"
    elif alert:
        verdict = (
            "light-review error rate is above the offline check: refit the temperature and "
            "recheck `high` (make route-cnn) on fresh labelled photos"
        )
    else:
        verdict = "light-review error rate is consistent with the offline check"

    return {
        "config_hash": current,
        "config_hashes": dict(hashes),
        "decisions": len(decisions),
        "modes": {m: sum(d["mode"] == m for d in decisions) for m in modes},
        "fallbacks": sum(bool(d["fallback"]) for d in decisions),
        "reviews": reviews,
        "light_review_check": {
            "live_error_rate": light["error_rate"],
            "live_ci95": light["ci95"],
            "finished_reviews": light["done"],
            "offline_error_rate": offline_rate,
            "alert": alert,
            "verdict": verdict,
        },
        "error_catch_rate": {"caught": caught, "errors": errors, "rate": _rate(caught, errors)},
        "safety_classes_to_second_review": {
            c: {**v, "share": _rate(v["second_review"], v["count"])} for c, v in safety_seen.items()
        },
        "second_review_disagreement": {
            "pairs": second_pairs,
            "disagreed": second_split,
            "rate": _rate(second_split, second_pairs),
        },
        "note": (
            "Reference labels are the reviewers' agreed labels; a light review is a glance, so "
            "its errors are found less reliably than in a full or second review."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report on live review routing.")
    parser.add_argument("--log", default="logs/decisions.jsonl", help="the API's decision log")
    parser.add_argument("--bundle", default="serving", help="bundle with routing.json")
    parser.add_argument("--out", help="also write the report to this JSON file")
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    if not log_path.is_file():
        raise SystemExit(f"{log_path} not found: the API writes it once review routing is on")
    spec_path = Path(args.bundle) / ROUTING_FILE
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else None
    decisions, outcomes = read_log(log_path)
    report = review_report(decisions, outcomes, spec)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    check = report["light_review_check"]
    print(f"\n{report['decisions']} decisions; light review: {check['verdict']}")


if __name__ == "__main__":
    main()
