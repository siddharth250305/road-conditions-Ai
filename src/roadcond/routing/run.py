"""Route every test image of a saved probabilities file and summarise the run.

    python -m roadcond.routing.run [--probs artifacts/results/baseline_probs.npz] [--overwrite]

The temperature of the primary model is fitted on val only; every test image is routed and
logged to <out_dir>/decisions.jsonl, and <out_dir>/summary.json holds the evaluation. This is
the one place that picks concrete providers and policies; the router itself does not.
"""

import argparse
import copy
import json
import logging
from collections.abc import Callable, Mapping, Sequence
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
    light_review_stats,
    mode_counts,
)
from roadcond.routing.log import DecisionLog, config_hash
from roadcond.routing.policy import ReviewPolicy, ThresholdPolicy
from roadcond.routing.probs import ProbsFile, load_probs
from roadcond.routing.providers import (
    AgreementProvider,
    CalibratedConfidenceProvider,
    ConfidenceProvider,
    fit_temperature,
    scale_probs,
)
from roadcond.routing.router import ReviewRouter

log = logging.getLogger(__name__)

POLICIES: dict[str, Callable[[Mapping[str, Any]], ReviewPolicy]] = {
    "threshold": ThresholdPolicy.from_config,
}
MODEL_LABELS = {"random_forest": "random forest", "logreg": "logistic regression"}


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


def run_label(probs: ProbsFile) -> str:
    models = " + ".join(MODEL_LABELS.get(m, m) for m in probs.models)
    return f"{models}, RSCD test set, seed {probs.seed}"


def route_all(
    routing: Mapping[str, Any],
    probs: ProbsFile,
    out_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Route the test split, write decisions.jsonl and summary.json, return the summary."""
    primary = routing["primary_model"]
    needed = {primary, *routing["agreement_models"]}
    if not needed <= set(probs.models):
        raise ValueError(f"Models {sorted(needed - set(probs.models))} not in the probs file")
    decisions_path = out_dir / "decisions.jsonl"
    if decisions_path.exists():
        if not overwrite:
            raise FileExistsError(f"{decisions_path} exists; pass --overwrite to start a new log")
        decisions_path.unlink()

    val, test = probs.splits["val"], probs.splits["test"]
    ts = routing["temperature_search"]
    temperature = fit_temperature(
        val.probs[primary], val.labels, ts["t_min"], ts["t_max"], ts["steps"]
    )
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
    predicted = test.probs[primary].argmax(axis=1)
    wrong = predicted != test.labels

    sweep = []
    for high in sweep_values(routing["sweep"]):
        swept = copy.deepcopy(dict(routing))
        swept["thresholds"] = {**routing["thresholds"], "high": high}
        sweep_router = ReviewRouter(build_providers(swept, temperature), build_policy(swept))
        sweep_modes = [str(sweep_router.route(c).mode) for c in cases]
        sweep.append({"high": high, **light_review_stats(sweep_modes, wrong)})

    bins = routing["ece_bins"]
    summary = {
        "label": run_label(probs),
        "probs_seed": probs.seed,
        "primary_model": primary,
        "agreement_models": list(routing["agreement_models"]),
        "policy": {"name": policy.name, "version": policy.version},
        "config_hash": chash,
        "thresholds": dict(routing["thresholds"]),
        "n_images": len(cases),
        "modes": mode_counts(modes),
        "fallbacks": sum(r.fallback for r in routed),
        "primary_accuracy": float(1.0 - wrong.mean()),
        "calibration": {
            "temperature": temperature,
            "fitted_on": "val",
            "ece_bins": bins,
            "ece_test_before": expected_calibration_error(test.probs[primary], test.labels, bins),
            "ece_test_after": expected_calibration_error(
                scale_probs(test.probs[primary], temperature), test.labels, bins
            ),
        },
        "error_rate_by_mode": error_rate_by_mode(modes, wrong),
        "error_catch_rate": error_catch_rate(modes, wrong),
        "safety_classes_to_second_review": {
            cls: class_to_second_review(modes, true_labels, cls)
            for cls in routing["safety_critical_classes"]
        },
        "threshold_sweep": sweep,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Route test images to review modes.")
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--probs", help="probabilities file (default: routing.probs_file)")
    parser.add_argument("--out-dir", help="output folder (default: routing.out_dir)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing log")
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
    summary = route_all(routing, load_probs(probs_path), out_dir, args.overwrite)
    modes = ", ".join(f"{m} {v['pct']:.1f}%" for m, v in summary["modes"].items())
    print(f"{summary['label']}: {summary['n_images']} images; {modes}")
    print(f"Saved {out_dir / 'decisions.jsonl'} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
