"""Review routing (roadcond.routing): providers, policy, router fallbacks, audit log, script.

No torch needed; the ECE cross-check and the baseline end-to-end test skip without torch /
scikit-learn.
"""

import copy
import json
import math

import numpy as np
import pytest
from tests.helpers import CLASSES

from roadcond.config import load_config
from roadcond.routing import (
    AgreementProvider,
    CalibratedConfidenceProvider,
    Case,
    CaseView,
    ConfidenceProvider,
    Decision,
    DecisionLog,
    ProviderError,
    ReviewMode,
    ReviewPolicy,
    ReviewRouter,
    ThresholdPolicy,
    config_hash,
    read_decisions,
)
from roadcond.routing.evaluate import (
    class_to_second_review,
    error_catch_rate,
    error_rate_by_mode,
    expected_calibration_error,
    mode_counts,
)
from roadcond.routing.log import DECISION_FIELDS
from roadcond.routing.probs import SplitProbs, load_probs, save_probs
from roadcond.routing.providers import fit_temperature, scale_probs
from roadcond.routing.run import main as route_main
from roadcond.routing.run import route_all, sweep_values

ROUTING = load_config()["routing"]
LIGHT, FULL, SECOND = ReviewMode.LIGHT, ReviewMode.FULL, ReviewMode.SECOND


def probs_for(top: str, p: float) -> list[float]:
    """A probability vector with p on class `top` and the rest spread evenly."""
    rest = (1.0 - p) / (len(CLASSES) - 1)
    return [p if c == top else rest for c in CLASSES]


def case(rf=("wet", 0.9), lr=("wet", 0.8), true_label="wet", image_id="img-1") -> Case:
    return Case(
        image_id=image_id,
        classes=CLASSES,
        probs={"random_forest": probs_for(*rf), "logreg": probs_for(*lr)},
        true_label=true_label,
    )


def policy(**overrides) -> ThresholdPolicy:
    return ThresholdPolicy.from_config({**ROUTING, **overrides})


def router(log=None, temperature=1.0, providers=None, pol=None) -> ReviewRouter:
    providers = providers or [
        CalibratedConfidenceProvider("random_forest", temperature),
        AgreementProvider(["random_forest", "logreg"]),
    ]
    return ReviewRouter(providers, pol or policy(), log)


class Fixed:
    """A provider returning a fixed value (or raising), for fallback tests."""

    def __init__(self, name, value=None, error=None):
        self.name, self.value, self.error = name, value, error

    def score(self, case_view):
        if self.error is not None:
            raise self.error
        return self.value


# --- providers ------------------------------------------------------------------------------


def test_calibrated_confidence_temperature_one_is_the_raw_top_probability():
    view = case(rf=("dry", 0.7)).view()
    assert CalibratedConfidenceProvider("random_forest", 1.0).score(view) == pytest.approx(0.7)


def test_calibrated_confidence_temperature_scaling():
    p = np.array(probs_for("dry", 0.7))
    view = Case("x", CLASSES, {"random_forest": p}).view()
    expected = np.exp(np.log(p) / 2.0)
    expected = expected.max() / expected.sum()
    softer = CalibratedConfidenceProvider("random_forest", 2.0).score(view)
    sharper = CalibratedConfidenceProvider("random_forest", 0.5).score(view)
    assert softer == pytest.approx(expected)
    assert softer < 0.7 < sharper <= 1.0


def test_calibrated_confidence_handles_zero_probabilities_and_bad_input():
    p = [1.0, 0, 0, 0, 0, 0]  # random-forest votes are often exactly 0
    score = CalibratedConfidenceProvider("random_forest", 3.0).score(
        Case("x", CLASSES, {"random_forest": p}).view()
    )
    assert 0 < score <= 1.0
    provider = CalibratedConfidenceProvider("random_forest", 1.0)
    for bad in ({"logreg": probs_for("dry", 0.9)}, {"random_forest": [0.5, 0.5]}):
        with pytest.raises(ProviderError):
            provider.score(Case("x", CLASSES, bad).view())
    with pytest.raises(ProviderError):
        provider.score(Case("x", CLASSES, {"random_forest": [0.9] * 6}).view())  # sums to 5.4
    for t in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError):
            CalibratedConfidenceProvider("random_forest", t)


def test_agreement_with_two_models():
    agree = AgreementProvider(["random_forest", "logreg"])
    assert agree.score(case(rf=("wet", 0.9), lr=("wet", 0.4)).view()) == 1.0
    assert agree.score(case(rf=("wet", 0.9), lr=("dry", 0.9)).view()) == 0.0


def test_agreement_with_three_models_is_a_share_of_the_others():
    def view(*tops):
        names = ["a", "b", "c"]
        probs = {n: probs_for(t, 0.8) for n, t in zip(names, tops, strict=True)}
        return Case("x", CLASSES, probs).view()

    agree = AgreementProvider(["a", "b", "c"])
    assert agree.score(view("wet", "wet", "wet")) == 1.0
    assert agree.score(view("wet", "wet", "dry")) == 0.5
    assert agree.score(view("wet", "dry", "ice")) == 0.0
    assert agree.score(view("wet", "dry", "dry")) == 0.0  # matching each other does not count
    with pytest.raises(ValueError):
        AgreementProvider(["a"])
    with pytest.raises(ValueError):
        AgreementProvider(["a", "a"])


def test_providers_satisfy_the_protocol():
    assert isinstance(CalibratedConfidenceProvider("m", 1.0), ConfidenceProvider)
    assert isinstance(AgreementProvider(["a", "b"]), ConfidenceProvider)
    assert isinstance(policy(), ReviewPolicy)


def test_fit_temperature_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4000, 6)) * 3
    true_t = 2.5
    labels = np.array([rng.choice(6, p=p) for p in scale_probs(np.exp(logits), true_t)])
    overconfident = scale_probs(np.exp(logits), 1.0)  # the "model" is too sharp by 2.5x
    t = fit_temperature(overconfident, labels, 0.05, 20.0, 2000)
    assert t == pytest.approx(true_t, rel=0.1)
    with pytest.raises(ValueError):
        fit_temperature(overconfident, labels, 0.0, 1.0, 10)


# --- policy ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("confidence", "mode", "reason"),
    [
        (0.85, LIGHT, "conf>=0.85"),
        (0.8500001, LIGHT, "conf>=0.85"),
        (0.8499999, FULL, "conf<0.85"),
        (0.50, FULL, "conf<0.85"),
        (0.5000001, FULL, "conf<0.85"),
        (0.4999999, SECOND, "conf<0.5"),
        (0.0, SECOND, "conf<0.5"),
        (1.0, LIGHT, "conf>=0.85"),
    ],
)
def test_threshold_boundaries(confidence, mode, reason):
    decision = policy().decide(
        {"calibrated_confidence": confidence, "agreement": 1.0}, case().view()
    )
    assert decision.mode == mode
    assert decision.reasons == [reason]


@pytest.mark.parametrize("which", ["rf", "lr"])
def test_safety_class_predicted_by_either_model_gives_second_review(which):
    kwargs = {which: ("ice", 0.99)}
    view = case(**kwargs).view()
    decision = policy().decide({"calibrated_confidence": 0.99, "agreement": 1.0}, view)
    assert decision.mode == SECOND
    assert "safety_class:ice" in decision.reasons


def test_disagreement_gives_second_review_even_when_confident():
    decision = policy().decide(
        {"calibrated_confidence": 0.99, "agreement": 0.0},
        case(rf=("wet", 0.99), lr=("dry", 0.99)).view(),
    )
    assert decision.mode == SECOND
    assert decision.reasons == ["models_disagree", "conf>=0.85"]


def test_strictest_rule_wins_and_reasons_list_every_trigger():
    decision = policy().decide(
        {"calibrated_confidence": 0.6, "agreement": 0.0}, case(rf=("ice", 0.6)).view()
    )
    assert decision.mode == SECOND
    assert decision.reasons == ["safety_class:ice", "models_disagree", "conf<0.85"]


@pytest.mark.parametrize("missing", ["calibrated_confidence", "agreement"])
def test_missing_signal_is_never_lighter_than_full_review(missing):
    signals = {"calibrated_confidence": 0.99, "agreement": 1.0, missing: None}
    decision = policy().decide(signals, case().view())
    assert decision.mode == FULL
    assert f"missing_signal:{missing}" in decision.reasons
    # Absent from the dict counts as missing too.
    del signals[missing]
    assert policy().decide(signals, case().view()).mode == FULL


def test_thresholds_come_from_config():
    strict = policy(thresholds={"high": 0.95, "low": 0.9}, safety_critical_classes=["water"])
    view = case(rf=("water", 0.92), lr=("water", 0.9)).view()
    decision = strict.decide({"calibrated_confidence": 0.92, "agreement": 1.0}, view)
    assert decision.mode == SECOND
    assert decision.reasons == ["safety_class:water", "conf<0.95"]
    assert strict.version == ROUTING["policy_version"]
    with pytest.raises(ValueError):
        policy(thresholds={"high": 0.4, "low": 0.5})


# --- router ---------------------------------------------------------------------------------


def test_router_routes_a_confident_agreeing_case_to_light_review(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", "abc")
    routed = router(log).route(case())
    assert routed.mode == LIGHT
    assert routed.signals == {"calibrated_confidence": pytest.approx(0.9), "agreement": 1.0}
    assert routed.predictions == {"random_forest": "wet", "logreg": "wet"}
    assert not routed.fallback and routed.provider_errors == {}


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (Fixed("calibrated_confidence", error=ProviderError("model offline")), "model offline"),
        (Fixed("calibrated_confidence", error=KeyError("boom")), "KeyError"),
        (Fixed("calibrated_confidence", value=math.nan), "nan"),
        (Fixed("calibrated_confidence", value=math.inf), "inf"),
        (Fixed("calibrated_confidence", value=1.7), "1.7"),
        (Fixed("calibrated_confidence", value=-0.1), "-0.1"),
        (Fixed("calibrated_confidence", value="0.9"), "'0.9'"),
        (Fixed("calibrated_confidence", value=True), "True"),
    ],
)
def test_failed_provider_falls_back_to_full_review_or_stricter(tmp_path, bad, message):
    path = tmp_path / "d.jsonl"
    providers = [bad, AgreementProvider(["random_forest", "logreg"])]
    routed = router(DecisionLog(path, "abc"), providers=providers).route(case())
    assert routed.mode.rank >= FULL.rank
    assert routed.fallback is True
    assert routed.signals["calibrated_confidence"] is None
    assert "missing_signal:calibrated_confidence" in routed.reasons
    [line] = read_decisions(path)
    assert line["fallback"] is True
    assert message in line["provider_errors"]["calibrated_confidence"]
    assert line["signals"]["calibrated_confidence"] is None


def test_policy_that_raises_falls_back_to_full_review(tmp_path):
    class Broken:
        name, version = "broken", "0"

        def decide(self, signals, case_view):
            raise RuntimeError("bug in policy")

    path = tmp_path / "d.jsonl"
    routed = router(DecisionLog(path, "abc"), pol=Broken()).route(case())
    assert routed.mode == FULL
    assert routed.reasons == ["fallback:policy_error"]
    assert routed.fallback is True
    [line] = read_decisions(path)
    assert "bug in policy" in line["provider_errors"]["policy"]
    assert line["policy"] == {"name": "broken", "version": "0"}


def test_policy_returning_an_unknown_mode_is_a_policy_error():
    class Bogus:
        name, version = "bogus", "0"

        def decide(self, signals, case_view):
            return Decision(mode="auto_accept", reasons=[])

    routed = router(pol=Bogus()).route(case())
    assert routed.mode == FULL and routed.fallback


def test_router_never_lets_a_failure_make_review_lighter():
    class IgnoresMissing:  # a careless policy that returns light_review no matter what
        name, version = "careless", "0"

        def decide(self, signals, case_view):
            return Decision(mode=LIGHT, reasons=["always_light"])

    providers = [Fixed("calibrated_confidence", error=ProviderError("down"))]
    routed = router(providers=providers, pol=IgnoresMissing()).route(case())
    assert routed.mode == FULL
    assert routed.reasons == ["always_light", "fallback:provider_error"]
    # A stricter decision is kept as it is.
    strict = ReviewRouter(providers, policy()).route(case(rf=("ice", 0.99)))
    assert strict.mode == SECOND


def test_swapping_in_a_dummy_policy_needs_no_router_change():
    class AlwaysSecond:
        name, version = "always_second", "9.9"

        def __init__(self):
            self.seen = []

        def decide(self, signals, case_view):
            self.seen.append(dict(signals))
            return Decision(mode=SECOND, reasons=["dummy"])

    dummy = AlwaysSecond()
    routed = router(pol=dummy).route(case())
    assert (routed.mode, routed.reasons) == (SECOND, ["dummy"])
    assert (routed.policy_name, routed.policy_version) == ("always_second", "9.9")
    assert set(dummy.seen[0]) == {"calibrated_confidence", "agreement"}


def test_a_third_provider_appears_in_the_log_with_no_router_change(tmp_path):
    class Entropy:
        name = "normalised_entropy"

        def score(self, case_view):
            p = np.clip(case_view.probs["random_forest"], 1e-12, 1)
            return float(1 + (p * np.log(p)).sum() / np.log(len(p)))

    path = tmp_path / "d.jsonl"
    providers = [
        CalibratedConfidenceProvider("random_forest", 1.0),
        AgreementProvider(["random_forest", "logreg"]),
        Entropy(),
    ]
    router(DecisionLog(path, "abc"), providers=providers).route(case())
    [line] = read_decisions(path)
    assert set(line["signals"]) == {"calibrated_confidence", "agreement", "normalised_entropy"}
    assert 0 <= line["signals"]["normalised_entropy"] <= 1


def test_router_rejects_duplicate_provider_names():
    with pytest.raises(ValueError):
        ReviewRouter([Fixed("a", 0.5), Fixed("a", 0.6)], policy())


def test_providers_and_policies_never_see_true_label():
    seen = []

    class Spy:
        name = "spy"

        def score(self, case_view):
            seen.append(case_view)
            return 0.5

    class SpyPolicy:
        name, version = "spy", "0"

        def decide(self, signals, case_view):
            seen.append(case_view)
            return Decision(mode=FULL, reasons=[])

    ReviewRouter([Spy()], SpyPolicy()).route(case(true_label="ice"))
    assert len(seen) == 2
    for view in seen:
        assert isinstance(view, CaseView)
        assert not hasattr(view, "true_label")
        with pytest.raises((AttributeError, TypeError)):
            view.true_label = "ice"  # slots: no attribute can be added either
        with pytest.raises(ValueError):
            view.probs["random_forest"][0] = 1.0  # read-only arrays


def test_true_label_does_not_change_the_decision():
    a = router().route(case(true_label="wet"))
    b = router().route(case(true_label="ice"))
    c = router().route(case(true_label=None))
    assert a.mode == b.mode == c.mode and a.reasons == b.reasons == c.reasons


# --- audit log ------------------------------------------------------------------------------


def test_log_lines_are_valid_json_with_every_schema_field(tmp_path):
    path = tmp_path / "d.jsonl"
    r = router(DecisionLog(path, config_hash(ROUTING)))
    for i, c in enumerate([case(), case(rf=("ice", 0.9)), case(rf=("wet", 0.6))]):
        r.route(Case(f"img-{i}", c.classes, c.probs, c.true_label))
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    ids = set()
    for line in lines:
        record = json.loads(line)
        assert tuple(record) == DECISION_FIELDS
        assert record["schema_version"] == 1
        assert record["config_hash"] == config_hash(ROUTING)
        assert record["policy"] == {"name": "threshold", "version": ROUTING["policy_version"]}
        assert record["mode"] in {m.value for m in ReviewMode}
        assert record["human_outcome"] is None
        assert record["timestamp"].endswith("+00:00")  # ISO-8601 UTC
        ids.add(record["decision_id"])
    assert len(ids) == 3
    modes = [json.loads(x)["mode"] for x in lines]
    assert modes == ["light_review", "second_review", "full_review"]


def test_record_outcome_appends_and_reader_joins(tmp_path):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(path, "abc")
    r = router(log)
    first = r.route(Case("a", CLASSES, case().probs))
    second = r.route(Case("b", CLASSES, case().probs))
    before = path.read_text()

    log.record_outcome(first.decision_id, "confirmed:wet", "reviewer-1")
    log.record_outcome(first.decision_id, "corrected:water", "reviewer-2")
    after = path.read_text()
    assert after.startswith(before)  # earlier lines untouched
    new = [json.loads(x) for x in after[len(before) :].splitlines()]
    assert [x["event"] for x in new] == ["outcome", "outcome"]

    joined = {d["image_id"]: d for d in read_decisions(path)}
    assert list(joined) == ["a", "b"]
    assert joined["a"]["human_outcome"]["outcome"] == "corrected:water"  # latest wins
    assert joined["a"]["human_outcome"]["reviewer"] == "reviewer-2"
    assert joined["b"]["human_outcome"] is None
    assert joined["b"]["decision_id"] == second.decision_id

    log.record_outcome("no-such-id", "confirmed:wet", "reviewer-1")
    with pytest.raises(ValueError, match="unknown decision"):
        read_decisions(path)
    with pytest.raises(ValueError):
        log.record_outcome(first.decision_id, "", "reviewer-1")


def test_same_input_and_config_give_same_modes_and_hash(tmp_path):
    cases = [case(), case(rf=("dry", 0.55)), case(rf=("ice", 0.9)), case(lr=("dry", 0.7))]
    runs = []
    for k in range(2):
        path = tmp_path / f"{k}.jsonl"
        r = router(DecisionLog(path, config_hash(copy.deepcopy(ROUTING))), temperature=1.3)
        for c in cases:
            r.route(c)
        runs.append(read_decisions(path))
    for key in ("mode", "reasons", "signals", "config_hash"):
        assert [d[key] for d in runs[0]] == [d[key] for d in runs[1]]
    reordered = dict(reversed(list(ROUTING.items())))
    assert config_hash(reordered) == config_hash(ROUTING)
    assert config_hash({**ROUTING, "policy_version": "2.0.0"}) != config_hash(ROUTING)


# --- evaluation -----------------------------------------------------------------------------


def test_summary_metrics_on_a_hand_made_run():
    modes = ["light_review", "light_review", "full_review", "second_review", "second_review"]
    wrong = np.array([False, True, True, True, False])
    counts = mode_counts(modes)
    assert counts["light_review"] == {"count": 2, "pct": 40.0}
    by_mode = error_rate_by_mode(modes, wrong)
    assert by_mode["light_review"]["error_rate"] == 0.5
    assert by_mode["full_review"]["error_rate"] == 1.0
    catch = error_catch_rate(modes, wrong)
    assert (catch["caught"], catch["errors"], catch["rate"]) == (2, 3, 2 / 3)
    assert catch["strict_share"] == 0.6
    assert catch["lift"] == pytest.approx((2 / 3) / 0.6)
    assert error_catch_rate(modes, np.zeros(5, bool))["rate"] is None
    assert error_catch_rate(modes, np.zeros(5, bool))["lift"] is None

    true = ["ice", "ice", "wet", "ice", "dry"]
    reasons = [[], ["safety_class:ice"], ["safety_class:ice"], ["safety_class:ice"], []]
    ice = class_to_second_review(modes, true, reasons, "ice")
    assert (ice["count"], ice["second_review"], ice["share"]) == (3, 1, 1 / 3)
    assert ice["left_in_light_review"] == 2
    assert (ice["rule_fired"], ice["rule_precision"]) == (3, 2 / 3)


def test_ece_matches_roadcond_calibration():
    pytest.importorskip("torch")
    from roadcond.calibration import expected_calibration_error as reference

    rng = np.random.default_rng(1)
    probs = scale_probs(np.exp(rng.normal(size=(500, 6)) * 2), 1.0)
    labels = rng.integers(0, 6, 500)
    for bins in (10, 15):
        assert expected_calibration_error(probs, labels, bins) == pytest.approx(
            reference(probs, labels, bins), abs=1e-12
        )


def test_sweep_values_from_config():
    assert sweep_values(ROUTING["sweep"]) == [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


# --- end to end -----------------------------------------------------------------------------


def mock_probs_file(path, n=60, seed=0):
    """A probabilities file shaped like the baseline's, with made-up (mock) predictions."""
    rng = np.random.default_rng(seed)
    splits = {}
    for split in ("val", "test"):
        labels = rng.integers(0, len(CLASSES), n)
        probs = {}
        for model, noise in (("random_forest", 1.0), ("logreg", 1.5)):
            logits = rng.normal(size=(n, len(CLASSES))) * noise
            logits[np.arange(n), labels] += 2.0
            probs[model] = scale_probs(np.exp(logits), 1.0)
        ids = np.array([f"{split}_{i:03d}.jpg" for i in range(n)])
        splits[split] = SplitProbs(image_ids=ids, labels=labels, probs=probs)
    save_probs(path, CLASSES, 0, splits)
    return load_probs(path)


def check_summary(summary, decisions_path, n):
    assert summary["n_images"] == n
    assert sum(v["count"] for v in summary["modes"].values()) == n
    assert sum(v["pct"] for v in summary["modes"].values()) == pytest.approx(100.0)
    assert summary["calibration"]["fitted_on"] == "val"
    assert summary["calibration"]["temperature"] > 0
    for key in ("ece_test_before", "ece_test_after"):
        assert 0 <= summary["calibration"][key] <= 1
    assert [s["high"] for s in summary["threshold_sweep"]] == sweep_values(ROUTING["sweep"])
    shares = [s["light_review_pct"] for s in summary["threshold_sweep"]]
    assert shares == sorted(shares, reverse=True)  # a higher bar leaves fewer light reviews
    for row in summary["threshold_sweep"]:
        total = row["light_review_pct"] + row["full_review_pct"] + row["second_review_pct"]
        assert total == pytest.approx(100.0)
    assert "Descriptive only" in summary["threshold_sweep_note"]
    decisions = read_decisions(decisions_path)
    assert len(decisions) == n
    by_mode = {m: sum(d["mode"] == m for d in decisions) for m in summary["modes"]}
    assert by_mode == {m: v["count"] for m, v in summary["modes"].items()}
    catch = summary["error_catch_rate"]
    errors = sum(v["errors"] for v in summary["error_rate_by_mode"].values())
    assert catch["errors"] == errors
    assert catch["caught"] == errors - summary["error_rate_by_mode"]["light_review"]["errors"]


def test_route_images_end_to_end_on_mock_probabilities(tmp_path):
    probs = mock_probs_file(tmp_path / "probs.npz")
    out = tmp_path / "routing"
    summary = route_all(ROUTING, probs, out)
    check_summary(summary, out / "decisions.jsonl", 60)
    assert json.loads((out / "summary.json").read_text()) == summary
    assert summary["label"] == "random forest + logistic regression, RSCD test set, seed 0"

    # Every image predicted ice by either model went to second review.
    test = probs.splits["test"]
    ice = CLASSES.index("ice")
    either_ice = {
        str(i)
        for k, i in enumerate(test.image_ids)
        if any(p[k].argmax() == ice for p in test.probs.values())
    }
    assert either_ice
    decisions = read_decisions(out / "decisions.jsonl")
    assert all(d["mode"] == "second_review" for d in decisions if d["image_id"] in either_ice)

    with pytest.raises(FileExistsError):
        route_all(ROUTING, probs, out)  # the audit log is never silently replaced
    again = route_all(ROUTING, probs, out, archive_existing=True)
    assert again["modes"] == summary["modes"] and again["config_hash"] == summary["config_hash"]
    assert again["archived_previous_log"] is not None


def test_route_images_command_line(tmp_path, capsys):
    mock_probs_file(tmp_path / "probs.npz")
    out = tmp_path / "routing"
    route_main(["--probs", str(tmp_path / "probs.npz"), "--out-dir", str(out)])
    assert (out / "summary.json").is_file() and (out / "decisions.jsonl").is_file()
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["probs_sha256"]) == 64
    args = ["--probs", str(tmp_path / "probs.npz"), "--out-dir", str(out)]
    with pytest.raises(FileExistsError):
        route_main(args)
    route_main([*args, "--archive-existing"])
    assert len(list(out.glob("decisions-*.jsonl"))) == 1
    assert "light_review" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--save-probs"):
        route_main(["--probs", str(tmp_path / "missing.npz"), "--out-dir", str(out)])


def test_baseline_probabilities_route_end_to_end(tmp_path):
    """Mock RSCD -> classical baseline with save_probs -> router: the real pipeline's path."""
    pytest.importorskip("sklearn")
    pytest.importorskip("skimage")
    from roadcond.data.mock import make_mock_rscd
    from roadcond.data.rscd import load_splits, prepare
    from roadcond.models.baseline import run_baseline

    cfg = load_config()
    root = tmp_path / "rscd"
    make_mock_rscd(root, cfg["rscd"]["split_dirs"], n_per_label=(6, 3, 3))
    prepare(cfg, root, tmp_path / "splits", train_size=36)
    cfg["project"]["seeds"] = [0]
    cfg["baseline"]["image_size"] = [48, 72]
    cfg["baseline"]["models"] = {
        "random_forest": {"n_estimators": 10, "grid": {"max_depth": [None]}},
        "logreg": {"max_iter": 200, "grid": {"C": [1.0]}},
    }
    probs_path = tmp_path / "baseline_probs.npz"
    splits = load_splits(tmp_path / "splits")
    run_baseline(cfg, splits, root, eval_test=True, save_probs=probs_path)
    probs = load_probs(probs_path)
    out = tmp_path / "routing"
    summary = route_all(ROUTING, probs, out)
    check_summary(summary, out / "decisions.jsonl", len(probs.splits["test"].image_ids))


# --- review findings: leaks, hostile input, log safety ---------------------------------------


def test_view_hides_the_image_id_because_rscd_file_names_carry_the_label():
    leaky = Case("test_50k/2022012523413511-ice-asphalt-smooth.jpg", CLASSES, case().probs, "ice")
    view = leaky.view()
    assert not hasattr(view, "image_id")
    assert "ice-asphalt" not in repr(view)
    routed = ReviewRouter([Fixed("s", 0.9)], policy(agreement_models=["random_forest"])).route(
        leaky
    )
    assert routed.image_id == leaky.image_id  # the log still records which image it was


@pytest.mark.parametrize(
    "bad_vector",
    [["a", "b", "c", "d", "e", "f"], [None] * 6, [[0.5, 0.5], [1.0]], [0.2] * 5, [math.nan] * 6],
)
def test_malformed_probabilities_never_crash_and_never_route_lighter(tmp_path, bad_vector):
    path = tmp_path / "d.jsonl"
    bad = Case("x", CLASSES, {"random_forest": bad_vector, "logreg": probs_for("wet", 0.95)})
    routed = router(DecisionLog(path, "abc")).route(bad)
    assert routed.mode.rank >= FULL.rank
    assert routed.fallback is True
    assert routed.predictions["random_forest"] is None
    assert len(read_decisions(path)) == 1


def test_huge_signal_is_rejected_not_raised():
    routed = router(providers=[Fixed("calibrated_confidence", 10**400)]).route(case())
    assert routed.mode.rank >= FULL.rank and routed.fallback


def test_unusable_prediction_from_an_unused_model_still_blocks_light_review():
    c = Case(
        "x",
        CLASSES,
        {
            "random_forest": probs_for("wet", 0.95),
            "logreg": probs_for("wet", 0.95),
            "cnn": [math.nan] * 6,  # not read by any provider
        },
    )
    routed = router().route(c)
    assert routed.mode == FULL
    assert "unusable_prediction:cnn" in routed.reasons


def test_policy_whose_name_raises_still_gets_logged(tmp_path):
    class NoName:
        version = "0"

        @property
        def name(self):
            raise RuntimeError("no name")

        def decide(self, signals, case_view):
            return Decision(mode=LIGHT, reasons=[])

    routed = router(DecisionLog(tmp_path / "d.jsonl", "abc"), pol=NoName()).route(case())
    assert routed.policy_name == "unknown"


def test_misspelled_safety_class_is_refused(tmp_path):
    probs = mock_probs_file(tmp_path / "probs.npz")
    with pytest.raises(ValueError, match="Safety classes"):
        route_all({**ROUTING, "safety_critical_classes": ["Ice"]}, probs, tmp_path / "out")


def test_archiving_keeps_recorded_outcomes(tmp_path):
    probs = mock_probs_file(tmp_path / "probs.npz")
    out = tmp_path / "out"
    route_all(ROUTING, probs, out)
    first = read_decisions(out / "decisions.jsonl")[0]
    DecisionLog(out / "decisions.jsonl", "abc").record_outcome(
        first["decision_id"], "confirmed:wet", "reviewer-1"
    )
    summary = route_all(ROUTING, probs, out, archive_existing=True)
    archived = read_decisions(summary["archived_previous_log"])
    assert archived[0]["human_outcome"]["outcome"] == "confirmed:wet"
    assert all(d["human_outcome"] is None for d in read_decisions(out / "decisions.jsonl"))


def test_reader_names_the_bad_line(tmp_path):
    path = tmp_path / "d.jsonl"
    router(DecisionLog(path, "abc")).route(case())
    with path.open("a") as f:
        f.write('{"schema_version": 1, "decision_id": "cut-off')  # crash mid-write
    with pytest.raises(ValueError, match="line 2: not valid JSON"):
        read_decisions(path)
    path.write_text('{"no": "id"}\n')
    with pytest.raises(ValueError, match="line 1"):
        read_decisions(path)


def test_routing_core_imports_no_heavy_frameworks():
    import subprocess
    import sys

    code = (
        "import sys, roadcond.routing, roadcond.routing.run; "
        "print(sorted(m for m in ('torch', 'sklearn', 'onnxruntime') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
