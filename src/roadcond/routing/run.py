"""Route every test image of a saved probabilities file and summarise the run.

    python -m roadcond.routing.run [--probs FILE.npz] [--out-dir DIR] [--archive-existing]
                                   [--bundle-dir serving]

The temperature of the primary model is fitted on val only; every test image is routed and
logged to <out_dir>/decisions.jsonl, and <out_dir>/summary.json holds the evaluation. This is
the one place that picks concrete providers and policies; the router itself does not.

With --bundle-dir, a check of the exported model's own probabilities also writes
<bundle>/routing.json, which switches routing on in the live API (roadcond.routing.live).
"""

import argparse
import copy
import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from roadcond.config import load_config, resolve_path
from roadcond.routing.case import Case
from roadcond.routing.evaluate import (
    class_to_second_review,
    error_catch_rate,
    error_rate_by_mode,
    expected_calibration_error,
    mode_counts,
    mode_shares,
)
from roadcond.routing.log import DecisionLog, config_hash
from roadcond.routing.policy import ReviewPolicy, ThresholdPolicy
from roadcond.routing.probs import ProbsFile, load_probs
from roadcond.routing.providers import (
    AgreementProvider,
    CalibratedConfidenceProvider,
    ConfidenceProvider,
    fit_temperature,
    nll,
    scale_probs,
)
from roadcond.routing.router import ReviewRouter

log = logging.getLogger(__name__)

POLICIES: dict[str, Callable[[Mapping[str, Any]], ReviewPolicy]] = {
    "threshold": ThresholdPolicy.from_config,
}
MODEL_LABELS = {
    "random_forest": "random forest",
    "logreg": "logistic regression",
    "resnet50": "ResNet-50",
}
NO_SEED = -1  # the model's training seed was not recorded


def build_providers(routing: Mapping[str, Any], temperature: float) -> list[ConfidenceProvider]:
    providers: list[ConfidenceProvider] = [
        CalibratedConfidenceProvider(routing["primary_model"], temperature)
    ]
    if len(routing["agreement_models"]) >= 2:
        providers.append(AgreementProvider(routing["agreement_models"]))
    return providers


def build_policy(routing: Mapping[str, Any]) -> ReviewPolicy:
    if routing["policy"] not in POLICIES:
        raise ValueError(f"Unknown policy {routing['policy']!r}; known: {sorted(POLICIES)}")
    return POLICIES[routing["policy"]](routing)


def split_cases(probs: ProbsFile) -> list[Case]:
    test = probs.splits["test"]
    return [
        Case(
            image_id=str(test.image_ids[i]),
            classes=probs.classes,
            probs={m: p[i] for m, p in test.probs.items()},
            true_label=probs.classes[int(test.labels[i])],
        )
        for i in range(len(test.image_ids))
    ]


def sweep_values(sweep: Mapping[str, float]) -> list[float]:
    start, stop, step = sweep["high_start"], sweep["high_stop"], sweep["high_step"]
    return [round(float(v), 10) for v in np.arange(start, stop + step / 2, step)]


def run_label(probs: ProbsFile, primary: str) -> str:
    """E.g. "random forest + logistic regression, RSCD test set, seed 0" (primary model first)."""
    ordered = [primary, *(m for m in probs.models if m != primary)]
    models = " + ".join(MODEL_LABELS.get(m, m) for m in ordered)
    seed = "seed not recorded" if probs.seed == NO_SEED else f"seed {probs.seed}"
    return f"{models}, RSCD test set, {seed}"


def archive_log(path: Path) -> Path:
    """Move an existing log aside under a UTC timestamp; nothing recorded is ever deleted."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archived = path.with_name(f"{path.stem}-{stamp}{path.suffix}")
    path.rename(archived)
    return archived


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_all(
    routing: Mapping[str, Any],
    probs: ProbsFile,
    out_dir: Path,
    archive_existing: bool = False,
    probs_path: Path | None = None,
) -> dict[str, Any]:
    """Route the test split, write decisions.jsonl and summary.json, return the summary.

    An existing decisions.jsonl is never overwritten: it is refused, or with archive_existing
    moved aside (outcomes recorded in it are kept there).
    """
    primary = routing["primary_model"]
    needed = {primary, *routing["agreement_models"]}
    if not needed <= set(probs.models):
        raise ValueError(f"Models {sorted(needed - set(probs.models))} not in the probs file")
    unknown = set(routing["safety_critical_classes"]) - set(probs.classes)
    if unknown:  # a typo would silently switch the safety rule off
        raise ValueError(f"Safety classes {sorted(unknown)} are not in {probs.classes}")
    decisions_path = out_dir / "decisions.jsonl"
    archived = None
    if decisions_path.exists():
        if not archive_existing:
            raise FileExistsError(
                f"{decisions_path} exists; pass --archive-existing to move it aside"
            )
        archived = archive_log(decisions_path)
        log.info("Moved the previous log to %s", archived)

    val, test = probs.splits["val"], probs.splits["test"]
    ts = routing["temperature_search"]
    temperature = fit_temperature(
        val.probs[primary], val.labels, ts["t_min"], ts["t_max"], ts["steps"]
    )
    at_edge = bool(np.isclose(temperature, [ts["t_min"], ts["t_max"]]).any())
    if at_edge:
        log.warning("Temperature %.4f is at the edge of the search grid; widen it", temperature)
    log.info("Temperature for %s fitted on val: %.4f", primary, temperature)

    chash = config_hash(routing)
    policy = build_policy(routing)
    router = ReviewRouter(
        build_providers(routing, temperature), policy, DecisionLog(decisions_path, chash)
    )
    cases = split_cases(probs)
    routed = [router.route(case) for case in cases]
    modes = [str(r.mode) for r in routed]

    true_labels = [c.true_label for c in cases]
    wrong = test.probs[primary].argmax(axis=1) != test.labels

    sweep = []
    for high in sweep_values(routing["sweep"]):
        swept = copy.deepcopy(dict(routing))
        swept["thresholds"] = {**routing["thresholds"], "high": high}
        sweep_router = ReviewRouter(build_providers(swept, temperature), build_policy(swept))
        sweep_modes = [str(sweep_router.route(c).mode) for c in cases]
        sweep.append({"high": high, **mode_shares(sweep_modes, wrong)})

    bins = routing["ece_bins"]
    raw_val, raw_test = val.probs[primary], test.probs[primary]
    summary = {
        "label": run_label(probs, primary),
        "probs_file": str(probs_path) if probs_path else None,
        "probs_sha256": sha256_file(probs_path) if probs_path else None,
        "probs_seed": probs.seed,
        "numpy_version": np.__version__,
        "primary_model": primary,
        "agreement_models": list(routing["agreement_models"]),
        "policy": {"name": policy.name, "version": policy.version},
        "config_hash": chash,
        "thresholds": dict(routing["thresholds"]),
        "safety_critical_classes": list(routing["safety_critical_classes"]),
        "n_images": len(cases),
        "modes": mode_counts(modes),
        "fallbacks": sum(r.fallback for r in routed),
        "primary_accuracy": float(1.0 - wrong.mean()),
        "calibration": {
            "temperature": temperature,
            "fitted_on": "val",
            "temperature_at_grid_edge": at_edge,
            "ece_bins": bins,
            "nll_val_before": nll(raw_val, val.labels),
            "nll_val_after": nll(scale_probs(raw_val, temperature), val.labels),
            "ece_val_before": expected_calibration_error(raw_val, val.labels, bins),
            "ece_val_after": expected_calibration_error(
                scale_probs(raw_val, temperature), val.labels, bins
            ),
            "ece_test_before": expected_calibration_error(raw_test, test.labels, bins),
            "ece_test_after": expected_calibration_error(
                scale_probs(raw_test, temperature), test.labels, bins
            ),
        },
        "error_rate_by_mode": error_rate_by_mode(modes, wrong),
        "error_catch_rate": error_catch_rate(modes, wrong),
        "safety_classes_to_second_review": {
            cls: class_to_second_review(modes, true_labels, [r.reasons for r in routed], cls)
            for cls in routing["safety_critical_classes"]
        },
        "threshold_sweep_note": (
            "Descriptive only: computed on the test split. The thresholds in use were set in "
            "the config before this run, not chosen from this sweep. Only `high` varies, so "
            "second_review (safety class, disagreement, `low`) barely moves."
        ),
        "threshold_sweep": sweep,
        "archived_previous_log": str(archived) if archived else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Route test images to review modes.")
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--probs", help="probabilities file (default: routing.probs_file)")
    parser.add_argument("--out-dir", help="output folder (default: routing.out_dir)")
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="move an existing decisions.jsonl aside (timestamped) instead of refusing",
    )
    parser.add_argument(
        "--bundle-dir",
        help="after the check, write <DIR>/routing.json so the live API routes with this config",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    routing = load_config(args.config)["routing"]
    cwd = Path.cwd()

    def pick(arg: str | None, default: str) -> Path:
        return resolve_path(arg, cwd) if arg else resolve_path(default)

    probs_path = pick(args.probs, routing["probs_file"])
    if not probs_path.is_file():
        raise SystemExit(
            f"{probs_path} not found. Save the baseline probabilities first:\n"
            "  python scripts/train_baseline.py --root <RSCD> --eval-test --save-probs"
        )
    out_dir = pick(args.out_dir, routing["out_dir"])
    probs = load_probs(probs_path)
    summary = route_all(routing, probs, out_dir, args.archive_existing, probs_path)
    modes = ", ".join(f"{m} {v['pct']:.1f}%" for m, v in summary["modes"].items())
    print(f"{summary['label']}: {summary['n_images']} images; {modes}")
    print(f"Saved {out_dir / 'decisions.jsonl'} and {out_dir / 'summary.json'}")
    if args.bundle_dir:
        from roadcond.routing.live import write_live_routing  # live imports this module

        written = write_live_routing(
            resolve_path(args.bundle_dir, cwd), routing, probs.classes, summary
        )
        print(f"Saved {written}: the API now routes live photos with this config")


if __name__ == "__main__":
    main()
