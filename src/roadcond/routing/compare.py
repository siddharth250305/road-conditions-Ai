"""Fit the learned review policy on val, then compare it with the threshold policy on test.

    python scripts/fit_learned_policy.py --config configs/default.yaml
    python scripts/compare_policies.py --config configs/default.yaml [--archive-existing]

Both policies get the same providers (the learned features' margin switched on) and go through
the same ReviewRouter, one decision log each. Two views:

  equal workload  the learned cutoffs give the threshold policy's per-mode counts on val, so
                  the question is which images reach careful review
  equal catch     a learned cut_full chosen on val to match the threshold policy's val error
                  catch rate; the question is how many light reviews each policy keeps

Every fitted or chosen number comes from val; test data only measures. Differences carry a
paired bootstrap 95% interval over test images.
"""

import argparse
import json
import logging
import time
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
    mode_counts,
)
from roadcond.routing.learned import (
    AGREEMENT_SIGNAL,
    FIT_SPLIT,
    LearnedPolicy,
    SplitFeatures,
    equal_catch_cut_full,
    feature_names,
    fit_policy,
    hard_rules,
    load_weights,
    modes_from_risk,
    with_learned_signals,
    write_weights,
)
from roadcond.routing.log import DecisionLog, config_hash
from roadcond.routing.policy import ReviewMode, ReviewPolicy, ThresholdPolicy
from roadcond.routing.probs import ProbsFile, load_probs
from roadcond.routing.providers import fit_temperature
from roadcond.routing.router import ReviewRouter
from roadcond.routing.run import archive_log, build_providers, run_label, sha256_file

log = logging.getLogger(__name__)

RESAMPLES = 1000
SEED = 0
COMPARISON_JSON = "policy_comparison.json"
COMPARISON_MD = "policy_comparison.md"
LOG_DIR = "comparison_logs"
LIGHT, STRICT = str(ReviewMode.LIGHT), (str(ReviewMode.FULL), str(ReviewMode.SECOND))
MODE_BY_RANK = np.array([str(m) for m in sorted(ReviewMode, key=lambda m: m.rank)])


def cases_for(probs: ProbsFile, split: str) -> list[Case]:
    sp = probs.splits[split]
    return [
        Case(
            image_id=str(sp.image_ids[i]),
            classes=probs.classes,
            probs={m: p[i] for m, p in sp.probs.items()},
            true_label=probs.classes[int(sp.labels[i])],
        )
        for i in range(len(sp.image_ids))
    ]


def val_temperature(routing: Mapping[str, Any], probs: ProbsFile) -> float:
    """The primary model's temperature, fitted on val exactly as run.route_all fits it."""
    val, ts = probs.splits[FIT_SPLIT], routing["temperature_search"]
    return fit_temperature(
        val.probs[routing["primary_model"]], val.labels, ts["t_min"], ts["t_max"], ts["steps"]
    )


def split_features(
    routing: Mapping[str, Any], probs: ProbsFile, split: str, temperature: float
) -> SplitFeatures:
    """Route a split with the threshold policy and keep each image's signals and hard-rule floor.

    The signals come from the router itself, so the learned policy is fitted on exactly the
    values it sees when routing.
    """
    shared = with_learned_signals(routing)
    router = ReviewRouter(build_providers(shared, temperature), ThresholdPolicy.from_config(shared))
    features = feature_names(routing)
    agreement = AGREEMENT_SIGNAL if len(routing["agreement_models"]) >= 2 else None
    safety = routing["safety_critical_classes"]
    rows, floors, modes = [], [], []
    for case in cases_for(probs, split):
        routed = router.route(case)
        values = [routed.signals.get(f) for f in features]
        rows.append([np.nan if v is None else v for v in values])
        floor, _ = hard_rules(routed.signals, case.view(), safety, features, agreement)
        floors.append(floor.rank)
        modes.append(str(routed.mode))
    sp = probs.splits[split]
    return SplitFeatures(
        split=split,
        X=np.array(rows, dtype=float).reshape(len(rows), len(features)),
        wrong=sp.probs[routing["primary_model"]].argmax(axis=1) != sp.labels,
        floor=np.array(floors, dtype=int),
        threshold_modes=np.array(modes),
    )


def fit(routing: Mapping[str, Any], probs: ProbsFile, probs_path: Path) -> dict[str, Any]:
    """Fit on val and write routing.learned_weights; returns the written weights."""
    temperature = val_temperature(routing, probs)
    val = split_features(routing, probs, FIT_SPLIT, temperature)
    core = fit_policy(val, feature_names(routing), routing["l2"])
    if not core["fit"]["converged"]:
        raise RuntimeError("The logistic regression did not converge; raise routing.l2")
    path = resolve_path(routing["learned_weights"])
    write_weights(path, core, routing, temperature, probs_path)
    return load_weights(path)[0]


# --- measuring ------------------------------------------------------------------------------


def policy_metrics(
    modes: np.ndarray,
    wrong: np.ndarray,
    true_labels: Sequence[str],
    reasons: Sequence[Sequence[str]],
    safety: Sequence[str],
) -> dict[str, Any]:
    return {
        "modes": mode_counts(modes),
        "error_rate_by_mode": error_rate_by_mode(modes, wrong),
        "error_catch_rate": error_catch_rate(modes, wrong),
        "safety_classes": {
            c: class_to_second_review(modes, true_labels, reasons, c) for c in safety
        },
    }


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, NaN where den is 0 (a resample with no images in that group)."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def catch_rate(w: np.ndarray, modes: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    """Error catch rate for each row of per-image resampling weights w (B, N)."""
    return _ratio(w @ (np.isin(modes, STRICT) & wrong), w @ wrong)


def light_error_rate(w: np.ndarray, modes: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    light = modes == LIGHT
    return _ratio(w @ (light & wrong), w @ light)


def light_share(w: np.ndarray, modes: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    return _ratio(w @ (modes == LIGHT), w.sum(axis=1))


Statistic = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
STATISTICS: dict[str, Statistic] = {
    "error_catch_rate": catch_rate,
    "light_review_error_rate": light_error_rate,
    "light_review_share": light_share,
}


def paired_bootstrap(
    modes_a: np.ndarray,
    modes_b: np.ndarray,
    wrong: np.ndarray,
    statistics: Mapping[str, Statistic] = STATISTICS,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    batch: int = 50,
) -> dict[str, dict[str, Any]]:
    """95% percentile interval of statistic(b) - statistic(a), for each statistic.

    Paired: each resample draws test images with replacement and scores both policies on the
    same draw. Resamples where a statistic is undefined (an empty group) are dropped and counted.
    """
    modes_a, modes_b = np.asarray(modes_a), np.asarray(modes_b)
    wrong = np.asarray(wrong, dtype=bool)
    n = len(wrong)
    rng = np.random.default_rng(seed)
    diffs: dict[str, list[np.ndarray]] = {name: [] for name in statistics}
    done = 0
    while done < resamples:
        size = min(batch, resamples - done)
        w = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(size)])
        for name, stat in statistics.items():
            diffs[name].append(stat(w, modes_b, wrong) - stat(w, modes_a, wrong))
        done += size
    ones = np.ones((1, n))
    out = {}
    for name, stat in statistics.items():
        d = np.concatenate(diffs[name])
        ok = d[np.isfinite(d)]
        estimate = float((stat(ones, modes_b, wrong) - stat(ones, modes_a, wrong))[0])
        ci = [float(v) for v in np.percentile(ok, [2.5, 97.5])] if len(ok) else None
        out[name] = {
            "estimate": estimate,
            "ci95": ci,
            "includes_zero": ci is None or ci[0] <= 0.0 <= ci[1],
            "resamples_used": int(len(ok)),
        }
    return out


def describe(name: str, result: Mapping[str, Any], higher_is_better: bool) -> str:
    """One plain sentence about a learned - threshold difference and its interval."""
    est, ci = result["estimate"], result["ci95"]
    if ci is None:
        return f"{name}: no interval (too few images)."
    text = (
        f"{name}: learned - threshold = {100 * est:+.3f} points "
        f"(95% CI {100 * ci[0]:+.3f} to {100 * ci[1]:+.3f})"
    )
    if result["includes_zero"]:
        return f"{text}; the interval includes 0, so this test set shows no clear difference."
    better = (est > 0) == higher_is_better
    return (
        f"{text}; the interval excludes 0: the learned policy is {'better' if better else 'worse'}."
    )


def _route(
    router: ReviewRouter, cases: Sequence[Case]
) -> tuple[np.ndarray, list[list[str]], float, int]:
    started = time.perf_counter()
    routed = [router.route(c) for c in cases]
    seconds = time.perf_counter() - started
    modes = np.array([str(r.mode) for r in routed])
    return modes, [r.reasons for r in routed], seconds, sum(r.fallback for r in routed)


def compare(
    routing: Mapping[str, Any],
    probs: ProbsFile,
    probs_path: Path,
    out_dir: Path,
    archive_existing: bool = False,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Route test with both policies, write policy_comparison.json and .md, return the JSON."""
    temperature = val_temperature(routing, probs)
    shared = with_learned_signals(routing)
    configs = {
        "threshold": {**shared, "policy": "threshold"},
        "learned": {**shared, "policy": "learned"},
    }
    learned = LearnedPolicy.from_config(configs["learned"])  # refuses stale weights
    weights, weights_sha = load_weights(resolve_path(routing["learned_weights"]))
    if weights["temperature"] != temperature:
        raise ValueError(
            f"The weights were fitted with T={weights['temperature']}, val gives {temperature}; "
            "refit them"
        )
    policies: dict[str, ReviewPolicy] = {
        "threshold": ThresholdPolicy.from_config(configs["threshold"]),
        "learned": learned,
    }
    primary, safety = routing["primary_model"], list(routing["safety_critical_classes"])
    test = probs.splits["test"]
    wrong = test.probs[primary].argmax(axis=1) != test.labels
    cases = cases_for(probs, "test")
    true_labels = [c.true_label for c in cases]

    results, modes = {}, {}
    for name, policy in policies.items():
        path = out_dir / LOG_DIR / f"decisions_{name}.jsonl"
        if path.exists():
            if not archive_existing:
                raise FileExistsError(f"{path} exists; pass --archive-existing to move it aside")
            log.info("Moved the previous log to %s", archive_log(path))
        chash = config_hash(configs[name])
        router = ReviewRouter(
            build_providers(shared, temperature), policy, DecisionLog(path, chash)
        )
        modes[name], reasons, seconds, fallbacks = _route(router, cases)
        results[name] = {
            "policy": {"name": policy.name, "version": policy.version},
            "config_hash": chash,
            "decision_log": str(path),
            **policy_metrics(modes[name], wrong, true_labels, reasons, safety),
            "fallbacks": fallbacks,
            "routing_ms_per_1000_images": 1e6 * seconds / len(cases),
        }
        log.info("%s: routed %d test images in %.1f s", name, len(cases), seconds)
    workload_ci = paired_bootstrap(
        modes["threshold"], modes["learned"], wrong, STATISTICS, resamples, seed
    )

    # Equal catch: cut_full chosen on val so the learned policy catches as many val errors.
    val = split_features(routing, probs, FIT_SPLIT, temperature)
    val_risk = learned.model.risk_matrix(val.X)
    target = error_catch_rate(val.threshold_modes, val.wrong)["rate"]
    cut_full = equal_catch_cut_full(val_risk, val.floor, val.wrong, target)
    cut_second = max(learned.cut_second, cut_full)
    val_modes = MODE_BY_RANK[modes_from_risk(val_risk, val.floor, cut_full, cut_second)]
    catch_policy = learned.with_cutoffs(cut_full, cut_second, "equal_catch")
    catch_router = ReviewRouter(build_providers(shared, temperature), catch_policy)
    catch_modes, catch_reasons, _, catch_fallbacks = _route(catch_router, cases)
    catch_ci = paired_bootstrap(modes["threshold"], catch_modes, wrong, STATISTICS, resamples, seed)

    summary = {
        "label": run_label(probs, primary),
        "probs_file": str(probs_path),
        "probs_sha256": sha256_file(probs_path),
        "n_test_images": len(cases),
        "n_val_images": len(val.X),
        "temperature": temperature,
        "providers": [p.name for p in build_providers(shared, temperature)],
        "weights": {
            "file": str(resolve_path(routing["learned_weights"])),
            "sha256": weights_sha,
            "created_at": weights["created_at"],
            "features": weights["features"],
            "coefficients": weights["coefficients"],
            "intercept": weights["intercept"],
            "l2": weights["l2"],
            "fitted_on": weights["fitted_on"],
        },
        "equal_workload": {
            "rule": "learned cutoffs give the threshold policy's per-mode counts on val",
            "cutoffs": {"cut_full": learned.cut_full, "cut_second": learned.cut_second},
            "val_threshold_counts": weights["val"]["threshold_counts"],
            "val_learned_counts": weights["val"]["learned_counts"],
            "val_max_count_gap": weights["val"]["max_count_gap"],
            "policies": results,
            "paired_bootstrap": {"resamples": resamples, "seed": seed, **workload_ci},
            "statements": [
                describe("Error catch rate", workload_ci["error_catch_rate"], True),
                describe("Light-review error rate", workload_ci["light_review_error_rate"], False),
            ],
        },
        "equal_catch": {
            "rule": (
                "learned cut_full = the highest cutoff whose val error catch rate reaches the "
                "threshold policy's val catch rate; cut_second kept (raised to cut_full if lower)"
            ),
            "cutoffs": {"cut_full": cut_full, "cut_second": cut_second},
            "policy_version": catch_policy.version,
            "val_target_catch_rate": target,
            "val_learned_catch_rate": error_catch_rate(val_modes, val.wrong)["rate"],
            "val_threshold_light_share": float(np.mean(val.threshold_modes == LIGHT)),
            "val_learned_light_share": float(np.mean(val_modes == LIGHT)),
            "learned_test": {
                **policy_metrics(catch_modes, wrong, true_labels, catch_reasons, safety),
                "fallbacks": catch_fallbacks,
            },
            "paired_bootstrap": {"resamples": resamples, "seed": seed, **catch_ci},
            "statements": [
                describe("Light-review share", catch_ci["light_review_share"], True),
                describe("Error catch rate", catch_ci["error_catch_rate"], True),
            ],
        },
        "notes": [
            "Fitted on val only: the risk model, its standardisation, the temperature and both "
            "sets of cutoffs. Test images are only routed and measured.",
            "Paired percentile bootstrap over test images; one trained model, so training "
            "variance is not in the intervals.",
            "Routing time covers computing signals and writing the decision log, one run per "
            "policy on the same machine: a rough cost, not a benchmark.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / COMPARISON_JSON).write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / COMPARISON_MD).write_text(comparison_markdown(summary))
    return summary


# --- the markdown table ---------------------------------------------------------------------


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{100 * rate:.1f}%"


def _diff(result: Mapping[str, Any]) -> str:
    ci = result["ci95"]
    if ci is None:
        return "n/a"
    zero = " (includes 0)" if result["includes_zero"] else ""
    return f"{100 * result['estimate']:+.3f} pts [{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}]{zero}"


def _of(part: int, whole: int) -> str:
    return f"{part:,} / {whole:,}"


def _rows(
    metrics: Sequence[Mapping[str, Any]], ci: Mapping[str, Any], safety: Sequence[str]
) -> list[str]:
    def row(label: str, cells: list[str], diff: str = "") -> str:
        return f"| {label} | {' | '.join(cells)} | {diff} |"

    light = str(ReviewMode.LIGHT)
    rows = []
    for mode in map(str, ReviewMode):
        cells = [f"{m['modes'][mode]['pct']:.1f}% ({m['modes'][mode]['count']:,})" for m in metrics]
        rows.append(
            row(f"`{mode}` share", cells, _diff(ci["light_review_share"]) * (mode == light))
        )
    for mode in map(str, ReviewMode):
        cells = []
        for m in metrics:
            e = m["error_rate_by_mode"][mode]
            cells.append(f"{_pct(e['error_rate'])} ({_of(e['errors'], e['count'])})")
        diff = _diff(ci["light_review_error_rate"]) * (mode == light)
        rows.append(row(f"Error rate in `{mode}`", cells, diff))
    cells = [
        f"{_pct(m['error_catch_rate']['rate'])} "
        f"({_of(m['error_catch_rate']['caught'], m['error_catch_rate']['errors'])})"
        for m in metrics
    ]
    rows.append(row("Error catch rate", cells, _diff(ci["error_catch_rate"])))
    for c in safety:
        stats = [m["safety_classes"][c] for m in metrics]
        cells = [f"{_pct(s['share'])} ({_of(s['second_review'], s['count'])})" for s in stats]
        rows.append(row(f"True-`{c}` images in `second_review`", cells))
        rows.append(
            row(
                f"True-`{c}` images left in `light_review`",
                [f"{s['left_in_light_review']:,}" for s in stats],
            )
        )
    return rows


def comparison_markdown(s: Mapping[str, Any]) -> str:
    wl, ec = s["equal_workload"], s["equal_catch"]
    t, lp = wl["policies"]["threshold"], wl["policies"]["learned"]
    safety = list(t["safety_classes"])
    boot = wl["paired_bootstrap"]
    lines = [
        f"# Policy comparison: {s['label']}",
        "",
        f"{s['n_test_images']:,} test images. Learned policy `{lp['policy']['version']}`, fitted "
        f"on val ({s['n_val_images']:,} images). Differences are learned - threshold in "
        f"percentage points, with a paired bootstrap 95% interval ({boot['resamples']:,} "
        f"resamples, seed {boot['seed']}).",
        "",
        "## Equal workload",
        "",
        "Learned cutoffs chosen on val so each mode gets the threshold policy's val count "
        f"(largest gap on val: {wl['val_max_count_gap']} images; cut_full "
        f"{wl['cutoffs']['cut_full']:.4f}, cut_second {wl['cutoffs']['cut_second']:.4f}).",
        "",
        "| Measure (test) | threshold | learned | learned - threshold (95% CI) |",
        "|---|---:|---:|---:|",
        *_rows([t, lp], boot, safety),
        f"| Routing time per 1,000 images | {t['routing_ms_per_1000_images']:.0f} ms | "
        f"{lp['routing_ms_per_1000_images']:.0f} ms | |",
        "",
        *[f"- {line}" for line in wl["statements"]],
        "",
        "## Equal catch rate",
        "",
        "Learned cut_full chosen on val to reach the threshold policy's val catch rate "
        f"({_pct(ec['val_target_catch_rate'])}; learned on val "
        f"{_pct(ec['val_learned_catch_rate'])}; cut_full {ec['cutoffs']['cut_full']:.4f}, "
        f"cut_second {ec['cutoffs']['cut_second']:.4f}).",
        "",
        "| Measure (test) | threshold | learned (equal catch) | learned - threshold (95% CI) |",
        "|---|---:|---:|---:|",
        *_rows([t, ec["learned_test"]], ec["paired_bootstrap"], safety),
        "",
        *[f"- {line}" for line in ec["statements"]],
        "",
    ]
    return "\n".join(lines)


# --- command lines --------------------------------------------------------------------------


def _load(config: str | None, probs_arg: str | None) -> tuple[dict[str, Any], ProbsFile, Path]:
    routing = load_config(config)["routing"]
    cwd = Path.cwd()
    probs_path = resolve_path(probs_arg, cwd) if probs_arg else resolve_path(routing["probs_file"])
    if not probs_path.is_file():
        raise SystemExit(f"{probs_path} not found: save the val/test probabilities first")
    return {**routing, "probs_file": str(probs_path)}, load_probs(probs_path), probs_path


def fit_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the learned review policy on val.")
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--probs", help="probabilities file (default: routing.probs_file)")
    parser.add_argument("--weights", help="output file (default: routing.learned_weights)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    routing, probs, probs_path = _load(args.config, args.probs)
    if args.weights:
        routing["learned_weights"] = str(resolve_path(args.weights, Path.cwd()))
    weights = fit(routing, probs, probs_path)
    print(
        f"Fitted on val ({weights['val']['n_images']} images), features {weights['features']}: "
        f"cut_full {weights['cutoffs']['cut_full']:.4f}, "
        f"cut_second {weights['cutoffs']['cut_second']:.4f}; val counts "
        f"{weights['val']['learned_counts']} (threshold {weights['val']['threshold_counts']})"
    )
    print(f"Saved {resolve_path(routing['learned_weights'])}")


def compare_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare the learned and threshold policies.")
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--probs", help="probabilities file (default: routing.probs_file)")
    parser.add_argument("--weights", help="learned weights (default: routing.learned_weights)")
    parser.add_argument("--out-dir", help="output folder (default: routing.out_dir)")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="move existing comparison decision logs aside (timestamped) instead of refusing",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    routing, probs, probs_path = _load(args.config, args.probs)
    cwd = Path.cwd()
    if args.weights:
        routing["learned_weights"] = str(resolve_path(args.weights, cwd))
    out_dir = resolve_path(args.out_dir, cwd) if args.out_dir else resolve_path(routing["out_dir"])
    compare(routing, probs, probs_path, out_dir, args.archive_existing, args.resamples, args.seed)
    print((out_dir / COMPARISON_MD).read_text())
    print(f"Saved {out_dir / COMPARISON_JSON} and {out_dir / COMPARISON_MD}")
