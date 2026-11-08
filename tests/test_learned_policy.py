"""The margin provider and the learned review policy (roadcond.routing). NumPy only."""

import json
import math

import numpy as np
import pytest
import yaml
from tests.helpers import CLASSES
from tests.test_routing import mock_probs_file, probs_for

from roadcond.config import load_config
from roadcond.routing import (
    CalibratedConfidenceProvider,
    Case,
    CaseView,
    ConfidenceProvider,
    DecisionLog,
    MarginProvider,
    ReviewMode,
    ReviewPolicy,
    ReviewRouter,
    ThresholdPolicy,
    read_decisions,
)
from roadcond.routing.compare import (
    COMPARISON_JSON,
    COMPARISON_MD,
    compare,
    compare_main,
    fit,
    fit_main,
    paired_bootstrap,
)
from roadcond.routing.learned import (
    LearnedPolicy,
    LearnedPolicyError,
    RiskModel,
    SplitFeatures,
    cutoff_for_count,
    equal_catch_cut_full,
    fit_logistic,
    fit_policy,
    modes_from_risk,
    sigmoid,
    write_weights,
)
from roadcond.routing.probs import SplitProbs, load_probs, save_probs
from roadcond.routing.providers import scale_probs
from roadcond.routing.run import POLICIES, build_policy, build_providers, route_all
from roadcond.routing.run import main as route_main

ROUTING = load_config()["routing"]
LIGHT, FULL, SECOND = ReviewMode.LIGHT, ReviewMode.FULL, ReviewMode.SECOND


def view(vector, model="random_forest") -> CaseView:
    return Case("img-1", CLASSES, {model: vector}).view()


# --- margin provider ------------------------------------------------------------------------


def test_margin_at_temperature_one_is_the_raw_top_two_gap():
    p = [0.6, 0.25, 0.05, 0.05, 0.03, 0.02]
    margin = MarginProvider("random_forest", 1.0)
    assert margin.score(view(p)) == pytest.approx(0.35)
    assert margin.name == "margin" and isinstance(margin, ConfidenceProvider)


@pytest.mark.parametrize("temperature", [0.5, 2.0])
def test_margin_uses_the_temperature_scaled_probabilities(temperature):
    p = np.array([0.6, 0.25, 0.05, 0.05, 0.03, 0.02])
    scaled = np.sort(scale_probs(p, temperature))
    margin = MarginProvider("random_forest", temperature).score(view(p))
    assert margin == pytest.approx(scaled[-1] - scaled[-2])
    assert margin != pytest.approx(0.35)
    assert (margin > 0.35) == (temperature < 1)  # T < 1 sharpens, T > 1 flattens


def test_margin_is_zero_for_a_tie_and_one_for_a_certain_model():
    assert MarginProvider("m", 1.0).score(view([0.4, 0.4, 0.1, 0.1, 0, 0], "m")) == 0.0
    assert MarginProvider("m", 1.0).score(view([1, 0, 0, 0, 0, 0], "m")) == pytest.approx(1.0)


def test_margin_refuses_a_bad_temperature():
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            MarginProvider("m", bad)


@pytest.mark.parametrize(
    "vector", [[float("nan")] * 6, [0.5, 0.5], [2.0, -1.0, 0, 0, 0, 0], "not numbers"]
)
def test_invalid_margin_input_goes_to_the_routers_fallback(tmp_path, vector):
    path = tmp_path / "d.jsonl"
    router = ReviewRouter(
        [MarginProvider("random_forest", 1.0)],
        ThresholdPolicy.from_config({**ROUTING, "agreement_models": ["random_forest"]}),
        DecisionLog(path, "abc"),
    )
    routed = router.route(Case("img-1", CLASSES, {"random_forest": vector}))
    assert routed.fallback and routed.mode.rank >= FULL.rank
    [line] = read_decisions(path)
    assert line["signals"]["margin"] is None and "margin" in line["provider_errors"]


def test_a_margin_outside_zero_one_is_refused_by_the_router():
    class Broken(MarginProvider):
        def score(self, case):
            return 1.5

    pol = ThresholdPolicy.from_config({**ROUTING, "agreement_models": ["random_forest"]})
    providers = [CalibratedConfidenceProvider("random_forest", 1.0), Broken("random_forest", 1.0)]
    routed = ReviewRouter(providers, pol).route(
        Case("img-1", CLASSES, {"random_forest": probs_for("wet", 0.99)})
    )
    assert routed.signals["margin"] is None
    assert routed.mode == FULL and "missing_signal:margin" not in routed.reasons
    assert routed.reasons[-1] == "fallback:provider_error"


def test_margin_never_sees_the_label_or_the_file_name():
    seen = []

    class Spy(MarginProvider):
        def score(self, case):
            seen.append(case)
            return super().score(case)

    case = Case("ice/2021_ice_001.jpg", CLASSES, {"m": probs_for("ice", 0.9)}, true_label="ice")
    ReviewRouter(
        [Spy("m", 1.0)], ThresholdPolicy.from_config({**ROUTING, "agreement_models": ["m"]})
    ).route(case)
    [seen_view] = seen
    assert isinstance(seen_view, CaseView)
    assert not hasattr(seen_view, "true_label") and not hasattr(seen_view, "image_id")


def test_config_switches_the_margin_provider_on_by_name():
    names = [p.name for p in build_providers(ROUTING, 1.0)]
    assert names == ["calibrated_confidence", "agreement"]  # off by default
    providers = build_providers({**ROUTING, "extra_providers": ["margin"]}, 0.7)
    assert [p.name for p in providers] == ["calibrated_confidence", "agreement", "margin"]
    assert providers[-1].model == ROUTING["primary_model"] and providers[-1].temperature == 0.7
    with pytest.raises(ValueError, match="Unknown provider"):
        build_providers({**ROUTING, "extra_providers": ["entropy"]}, 1.0)


def test_margin_is_logged_and_leaves_threshold_decisions_unchanged(tmp_path):
    probs = mock_probs_file(tmp_path / "probs.npz")
    plain = route_all(ROUTING, probs, tmp_path / "plain")
    with_margin = route_all({**ROUTING, "extra_providers": ["margin"]}, probs, tmp_path / "m")
    assert with_margin["modes"] == plain["modes"]
    a = read_decisions(tmp_path / "plain" / "decisions.jsonl")
    b = read_decisions(tmp_path / "m" / "decisions.jsonl")
    assert [d["mode"] for d in a] == [d["mode"] for d in b]
    assert [d["reasons"] for d in a] == [d["reasons"] for d in b]
    assert all(0 <= d["signals"]["margin"] <= 1 for d in b)
    assert "margin" not in a[0]["signals"]
    assert json.loads((tmp_path / "m" / "summary.json").read_text())["n_images"] == 60


# --- logistic regression --------------------------------------------------------------------


def test_fit_logistic_recovers_known_coefficients_and_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100_000, 3))
    true_coef, true_intercept = np.array([2.0, -1.0, 0.5]), -1.0
    y = (rng.uniform(size=len(X)) < sigmoid(true_intercept + X @ true_coef)).astype(float)
    fit = fit_logistic(X, y, l2=0.0)
    assert fit.converged and fit.iterations < 20
    assert fit.intercept == pytest.approx(true_intercept, abs=0.05)
    np.testing.assert_allclose(fit.coef, true_coef, atol=0.05)
    again = fit_logistic(X, y, l2=0.0)
    assert again.intercept == fit.intercept and np.array_equal(again.coef, fit.coef)


def test_l2_shrinks_the_coefficients_but_not_the_intercept():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 2))
    y = (X[:, 0] + rng.normal(size=500) > 0).astype(float)
    small, big = fit_logistic(X, y, 0.0), fit_logistic(X, y, 1e4)
    assert np.linalg.norm(big.coef) < 0.1 * np.linalg.norm(small.coef)
    assert big.intercept == pytest.approx(math.log(y.mean() / (1 - y.mean())), abs=0.05)


def test_fit_logistic_refuses_bad_input():
    X, y = np.zeros((4, 1)), np.array([0.0, 1.0, 0.0, 1.0])
    for bad_x, bad_y, l2 in [
        (np.zeros(4), y, 1.0),
        (X, np.array([0.0, 2.0, 0.0, 1.0]), 1.0),
        (np.full((4, 1), np.nan), y, 1.0),
        (X, y, -1.0),
        (np.zeros((0, 1)), np.zeros(0), 1.0),
    ]:
        with pytest.raises(ValueError):
            fit_logistic(bad_x, bad_y, l2)


# --- the learned policy ---------------------------------------------------------------------

FEATURES = ("calibrated_confidence", "margin", "agreement")
SIGNALS = {"calibrated_confidence": 0.95, "margin": 0.9, "agreement": 1.0}


def two_models(rf=("wet", 0.95), lr=("wet", 0.9)) -> CaseView:
    probs = {"random_forest": probs_for(*rf), "logreg": probs_for(*lr)}
    return Case("img-1", CLASSES, probs).view()


def learned(risk=0.01, cut_full=0.2, cut_second=0.6, agreement=True) -> LearnedPolicy:
    """A policy whose predicted risk is the constant `risk`."""
    features = FEATURES if agreement else FEATURES[:2]
    d = len(features)
    model = RiskModel(features, np.zeros(d), np.ones(d), math.log(risk / (1 - risk)), np.zeros(d))
    return LearnedPolicy(
        model, cut_full, cut_second, ["ice"], "1.0.0+test", "agreement" if agreement else None
    )


@pytest.mark.parametrize(
    "risk, mode, reason",
    [
        (0.7, SECOND, "risk>=0.6"),
        (0.6, SECOND, "risk>=0.6"),
        (0.3, FULL, "risk>=0.2"),
        (0.2, FULL, "risk>=0.2"),
        (0.1, LIGHT, "risk<0.2"),
    ],
)
def test_predicted_risk_picks_the_mode(risk, mode, reason):
    decision = learned(risk).decide(SIGNALS, two_models())
    assert (decision.mode, decision.reasons) == (mode, [reason])


@pytest.mark.parametrize("which", ["rf", "lr"])
def test_safety_class_wins_even_at_almost_zero_risk(which):
    view = two_models(**{which: ("ice", 0.99)})
    decision = learned(1e-9).decide({**SIGNALS, "agreement": 0.0}, view)
    assert decision.mode == SECOND
    assert decision.reasons[0] == "safety_class:ice" and decision.reasons[-1] == "risk<0.2"


def test_disagreement_wins_even_at_almost_zero_risk():
    decision = learned(1e-9).decide({**SIGNALS, "agreement": 0.0}, two_models())
    assert decision.mode == SECOND and "models_disagree" in decision.reasons


@pytest.mark.parametrize("missing", ["calibrated_confidence", "margin", "agreement"])
def test_a_missing_signal_gives_at_least_full_review(missing):
    signals = {**SIGNALS, missing: None}
    decision = learned(1e-9).decide(signals, two_models())
    assert decision.mode == FULL and decision.reasons == [f"missing_signal:{missing}"]
    assert learned(0.9).decide(signals, two_models()).mode == FULL  # no risk without the signal
    ice = learned(1e-9).decide(signals, two_models(rf=("ice", 0.9)))
    assert ice.mode == SECOND


def test_an_unusable_prediction_gives_at_least_full_review():
    view = Case("img-1", CLASSES, {"random_forest": [np.nan] * 6, "logreg": probs_for("wet", 0.9)})
    decision = learned(1e-9).decide(SIGNALS, view.view())
    assert decision.mode == FULL and "unusable_prediction:random_forest" in decision.reasons


def test_one_model_needs_no_agreement_signal():
    signals = {"calibrated_confidence": 0.95, "margin": 0.9}
    decision = learned(0.01, agreement=False).decide(signals, view(probs_for("wet", 0.95)))
    assert decision.mode == LIGHT


def test_a_learned_policy_that_raises_falls_back_to_full_review(tmp_path):
    class Exploding(RiskModel):
        def risk(self, signals):
            raise RuntimeError("bad weights")

    policy = learned()
    policy.model = Exploding(
        *(getattr(policy.model, f) for f in ("features", "means", "stds", "intercept", "coef"))
    )
    path = tmp_path / "d.jsonl"
    providers = build_providers({**ROUTING, "extra_providers": ["margin"]}, 1.0)
    routed = ReviewRouter(providers, policy, DecisionLog(path, "abc")).route(
        Case(
            "img-1",
            CLASSES,
            {"random_forest": probs_for("wet", 0.99), "logreg": probs_for("wet", 0.99)},
        )
    )
    assert (routed.mode, routed.reasons, routed.fallback) == (FULL, ["fallback:policy_error"], True)
    [line] = read_decisions(path)
    assert line["policy"] == {"name": "learned", "version": "1.0.0+test"}
    assert "bad weights" in line["provider_errors"]["policy"]


def test_the_learned_policy_never_sees_the_label_or_the_file_name():
    seen = []

    class Spy(LearnedPolicy):
        def decide(self, signals, case_view):
            seen.append((dict(signals), case_view))
            return super().decide(signals, case_view)

    base = learned()
    spy = Spy(base.model, 0.2, 0.6, ["ice"], "1.0.0+test")
    providers = build_providers({**ROUTING, "extra_providers": ["margin"]}, 1.0)
    case = Case(
        "ice/2021_ice_001.jpg",
        CLASSES,
        {"random_forest": probs_for("wet", 0.9), "logreg": probs_for("wet", 0.9)},
        true_label="ice",
    )
    ReviewRouter(providers, spy).route(case)
    [(signals, seen_view)] = seen
    assert set(signals) == set(FEATURES)
    assert isinstance(seen_view, CaseView)
    assert not hasattr(seen_view, "true_label") and not hasattr(seen_view, "image_id")


# --- cutoffs --------------------------------------------------------------------------------


def test_cutoff_for_count_comes_as_close_as_ties_allow():
    risks = np.array([0.9, 0.9, 0.5, 0.5, 0.1, 0.1])
    assert cutoff_for_count(risks, 2) == pytest.approx(0.7)
    assert cutoff_for_count(risks, 3) == pytest.approx(0.3)  # 2 or 4: the tie goes to more review
    assert cutoff_for_count(risks, 0) == pytest.approx(0.95)
    assert cutoff_for_count(risks, 6) == pytest.approx(0.05)
    for k in range(7):
        c = cutoff_for_count(risks, k)
        assert abs(int((risks >= c).sum()) - k) <= 1


def synthetic_split(n=3000, seed=0, split="val") -> SplitFeatures:
    """Signals, errors and hard-rule floors shaped like a two-model run (made-up numbers)."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.2, 1.0, n)
    margin = conf * rng.uniform(0.2, 1.0, n)
    agreement = (rng.uniform(size=n) > 0.1).astype(float)
    ice = rng.uniform(size=n) < 0.05
    wrong = rng.uniform(size=n) < sigmoid(2.5 - 6 * conf + 1.5 * (1 - agreement))
    floor = np.where(ice | (agreement < 1), 2, 0)
    modes = np.where(
        floor == 2,
        "second_review",
        np.where(conf < 0.5, "second_review", np.where(conf < 0.85, "full_review", "light_review")),
    )
    X = np.column_stack([conf, margin, agreement])
    return SplitFeatures(split, X, wrong, floor, modes), ice


def test_equal_workload_cutoffs_match_the_threshold_shares_to_one_image():
    val, _ = synthetic_split()
    core = fit_policy(val, FEATURES, l2=1.0)
    assert core["fit"]["converged"]
    assert core["val"]["max_count_gap"] <= 1
    assert sum(core["val"]["learned_counts"].values()) == len(val.X)
    assert core["cutoffs"]["cut_full"] <= core["cutoffs"]["cut_second"]
    assert core["coefficients"][0] < 0  # higher confidence, lower risk of being wrong


def test_decide_gives_the_modes_the_cutoffs_were_chosen_for():
    val, ice = synthetic_split(n=800)
    core = fit_policy(val, FEATURES, l2=1.0)
    policy = LearnedPolicy.from_weights(core, ROUTING, "f" * 64)
    model = policy.model
    risk = model.risk_matrix(val.X)
    expected = modes_from_risk(risk, val.floor, policy.cut_full, policy.cut_second)
    views = {False: two_models(), True: two_models(rf=("ice", 0.9))}
    for i, row in enumerate(val.X):
        signals = dict(zip(FEATURES, map(float, row), strict=True))
        assert policy.decide(signals, views[bool(ice[i])]).mode.rank == expected[i]


def test_equal_catch_cut_full_is_the_lightest_cut_that_reaches_the_target():
    val, _ = synthetic_split()
    core = fit_policy(val, FEATURES, l2=1.0)
    risk = LearnedPolicy.from_weights(core, ROUTING, "f" * 64).model.risk_matrix(val.X)
    target = 0.9
    cut = equal_catch_cut_full(risk, val.floor, val.wrong, target)

    def catch(c):
        strict = modes_from_risk(risk, val.floor, c, 2.0) >= 1
        return (strict & val.wrong).sum() / val.wrong.sum()

    assert catch(cut) >= target
    higher = np.unique(risk[(val.floor == 0) & (risk >= cut)])
    assert catch(higher.min() + 1e-12) < target  # one level lighter misses the target


def test_the_fit_refuses_any_split_but_val():
    test, _ = synthetic_split(split="test")
    with pytest.raises(ValueError, match="'val' only"):
        fit_policy(test, FEATURES, l2=1.0)


# --- the weights file -----------------------------------------------------------------------


def learned_setup(tmp_path, fit_split=None):
    """A mock probabilities file, a weights file fitted for it, and a routing config."""
    probs_path = tmp_path / "probs.npz"
    mock_probs_file(probs_path)
    weights = tmp_path / "learned_policy.json"
    routing = {
        **ROUTING,
        "policy": "learned",
        "extra_providers": ["margin"],
        "learned_weights": str(weights),
        "probs_file": str(probs_path),
    }
    core = fit_policy(fit_split or synthetic_split()[0], FEATURES, routing["l2"])
    sha = write_weights(weights, core, routing, 1.0, probs_path)
    return routing, sha


def test_the_weights_hash_is_in_the_version_and_every_logged_decision(tmp_path):
    routing, sha = learned_setup(tmp_path)
    policy = build_policy(routing)
    assert isinstance(policy, LearnedPolicy) and POLICIES["learned"] == LearnedPolicy.from_config
    assert policy.version == f"1.0.0+{sha[:12]}"
    probs = mock_probs_file(tmp_path / "probs.npz")
    summary = route_all(routing, probs, tmp_path / "out")
    assert summary["policy"] == {"name": "learned", "version": f"1.0.0+{sha[:12]}"}
    decisions = read_decisions(tmp_path / "out" / "decisions.jsonl")
    assert len(decisions) == 60
    assert {d["policy"]["version"] for d in decisions} == {f"1.0.0+{sha[:12]}"}
    assert all("margin" in d["signals"] for d in decisions)
    saved = json.loads((tmp_path / "learned_policy.json").read_text())
    for key in (
        "coefficients",
        "standardisation",
        "features",
        "cutoffs",
        "val",
        "probs_sha256",
        "created_at",
    ):
        assert key in saved
    assert saved["val"]["n_images"] == 3000 and saved["fitted_on"] == "val"


def test_edited_weights_get_a_new_version(tmp_path):
    routing, sha = learned_setup(tmp_path)
    path = tmp_path / "learned_policy.json"
    weights = json.loads(path.read_text())
    weights["intercept"] += 1.0
    path.write_text(json.dumps(weights))
    assert build_policy(routing).version != f"1.0.0+{sha[:12]}"


def test_a_missing_or_mismatched_weights_file_is_refused(tmp_path):
    routing, _ = learned_setup(tmp_path)
    with pytest.raises(LearnedPolicyError, match="not found"):
        build_policy({**routing, "learned_weights": str(tmp_path / "none.json")})
    other = tmp_path / "other.npz"
    mock_probs_file(other, seed=1)
    with pytest.raises(LearnedPolicyError, match="different probabilities file"):
        build_policy({**routing, "probs_file": str(other)})
    for change in (
        {"l2": 10.0},
        {"safety_critical_classes": ["ice", "water"]},
        {"agreement_models": ["random_forest"]},
        {"primary_model": "logreg"},
    ):
        with pytest.raises(LearnedPolicyError, match="other routing settings"):
            build_policy({**routing, **change})
    # The thresholds only set the workload the cutoffs were matched to; they are recorded.
    assert build_policy({**routing, "thresholds": {"high": 0.9, "low": 0.5}})
    with pytest.raises(LearnedPolicyError, match="extra_providers"):
        build_policy({**routing, "extra_providers": []})
    (tmp_path / "learned_policy.json").write_text("{not json")
    with pytest.raises(LearnedPolicyError, match="not valid JSON"):
        build_policy(routing)


def test_the_live_api_is_offered_the_threshold_policy_only(tmp_path):
    routing, _ = learned_setup(tmp_path)
    config = tmp_path / "learned.yaml"
    config.write_text(yaml.safe_dump({"routing": routing}))
    with pytest.raises(SystemExit, match="policy: threshold"):
        route_main(["--config", str(config), "--bundle-dir", str(tmp_path / "serving")])


def test_the_learned_policy_satisfies_the_review_policy_protocol():
    assert isinstance(learned(), ReviewPolicy) and learned().name == "learned"


# --- fit and compare end to end -------------------------------------------------------------


def mock_setup(tmp_path, n=300, seed=0):
    probs_path = tmp_path / "probs.npz"
    probs = mock_probs_file(probs_path, n=n, seed=seed)
    routing = {
        **ROUTING,
        "learned_weights": str(tmp_path / "learned_policy.json"),
        "probs_file": str(probs_path),
    }
    return routing, probs, probs_path


def test_fit_and_compare_on_mock_data_write_every_field(tmp_path):
    routing, probs, probs_path = mock_setup(tmp_path)
    weights = fit(routing, probs, probs_path)
    assert weights["fitted_on"] == "val" and weights["val"]["n_images"] == 300
    assert weights["features"] == ["calibrated_confidence", "margin", "agreement"]
    out = tmp_path / "out"
    summary = compare(routing, probs, probs_path, out, resamples=200)
    assert json.loads((out / COMPARISON_JSON).read_text()) == summary
    md = (out / COMPARISON_MD).read_text()
    assert "## Equal workload" in md and "## Equal catch rate" in md and "95% CI" in md

    wl, ec = summary["equal_workload"], summary["equal_catch"]
    for name in ("threshold", "learned"):
        result = wl["policies"][name]
        assert sum(v["count"] for v in result["modes"].values()) == 300
        for key in (
            "error_rate_by_mode",
            "error_catch_rate",
            "safety_classes",
            "fallbacks",
            "routing_ms_per_1000_images",
            "config_hash",
            "decision_log",
        ):
            assert key in result
        assert {"share", "left_in_light_review"} <= set(result["safety_classes"]["ice"])
        decisions = read_decisions(result["decision_log"])
        assert len(decisions) == 300
        assert {d["policy"]["version"] for d in decisions} == {result["policy"]["version"]}
    assert wl["policies"]["learned"]["policy"]["version"].endswith(
        summary["weights"]["sha256"][:12]
    )
    for view_ in (wl, ec):
        boot = view_["paired_bootstrap"]
        assert boot["resamples"] == 200 and boot["seed"] == 0
        for stat in ("error_catch_rate", "light_review_error_rate", "light_review_share"):
            ci = boot[stat]["ci95"]
            assert ci[0] <= ci[1] and isinstance(boot[stat]["includes_zero"], bool)
        assert len(view_["statements"]) == 2
    assert ec["val_learned_catch_rate"] >= ec["val_target_catch_rate"]
    assert ec["cutoffs"]["cut_full"] <= ec["cutoffs"]["cut_second"]
    assert ec["policy_version"].endswith("+equal_catch")
    assert set(summary["hard_rules_second_review_share"]) == {"val", "test"}

    # The threshold policy routes exactly as `make route` does: margin is only logged.
    plain = route_all(routing, probs, tmp_path / "plain")
    assert wl["policies"]["threshold"]["modes"] == plain["modes"]

    with pytest.raises(FileExistsError):
        compare(routing, probs, probs_path, out, resamples=10)
    compare(routing, probs, probs_path, out, archive_existing=True, resamples=10)
    assert len(list((out / "comparison_logs").glob("decisions_*-*.jsonl"))) == 2


def test_test_data_never_shapes_the_weights_or_any_cutoff(tmp_path):
    routing, probs, probs_path = mock_setup(tmp_path)
    rng = np.random.default_rng(99)
    test = probs.splits["test"]
    scrambled = SplitProbs(
        image_ids=test.image_ids,
        labels=rng.permutation(test.labels),
        probs={m: scale_probs(rng.uniform(size=p.shape), 1.0) for m, p in test.probs.items()},
    )
    other_path = tmp_path / "other" / "probs.npz"
    save_probs(
        other_path, probs.classes, probs.seed, {"val": probs.splits["val"], "test": scrambled}
    )
    other = load_probs(other_path)
    other_routing = {
        **routing,
        "probs_file": str(other_path),
        "learned_weights": str(tmp_path / "other" / "learned_policy.json"),
    }

    a, b = fit(routing, probs, probs_path), fit(other_routing, other, other_path)
    differ = {"created_at", "probs_file", "probs_sha256"}
    assert {k: v for k, v in a.items() if k not in differ} == {
        k: v for k, v in b.items() if k not in differ
    }
    sa = compare(routing, probs, probs_path, tmp_path / "a", resamples=10)
    sb = compare(other_routing, other, other_path, tmp_path / "b", resamples=10)
    for view_ in ("equal_workload", "equal_catch"):
        assert sa[view_]["cutoffs"] == sb[view_]["cutoffs"]
    assert sa["equal_workload"]["policies"] != sb["equal_workload"]["policies"]  # test differs


def test_a_stale_weights_file_stops_the_comparison(tmp_path):
    routing, probs, probs_path = mock_setup(tmp_path)
    with pytest.raises(LearnedPolicyError, match="not found"):
        compare(routing, probs, probs_path, tmp_path / "out")
    fit(routing, probs, probs_path)
    with pytest.raises(LearnedPolicyError, match="other routing settings"):
        compare({**routing, "l2": 5.0}, probs, probs_path, tmp_path / "out")


def test_fit_and_compare_command_lines(tmp_path, capsys):
    mock_probs_file(tmp_path / "probs.npz", n=200)
    weights = tmp_path / "w.json"
    common = ["--probs", str(tmp_path / "probs.npz"), "--weights", str(weights)]
    fit_main(common)
    assert json.loads(weights.read_text())["fitted_on"] == "val"
    compare_main([*common, "--out-dir", str(tmp_path / "out"), "--resamples", "50"])
    out = capsys.readouterr().out
    assert "Saved" in out and "| Error catch rate |" in out
    assert (tmp_path / "out" / COMPARISON_JSON).is_file()
    with pytest.raises(SystemExit, match="not found"):
        fit_main(["--probs", str(tmp_path / "missing.npz")])


# --- paired bootstrap -----------------------------------------------------------------------


def test_bootstrap_of_identical_policies_is_exactly_zero_and_deterministic():
    rng = np.random.default_rng(0)
    modes = rng.choice(["light_review", "full_review", "second_review"], 500)
    wrong = rng.uniform(size=500) < 0.2
    result = paired_bootstrap(modes, modes, wrong, resamples=100, seed=3)
    for stat in result.values():
        assert stat["estimate"] == 0 and stat["ci95"] == [0.0, 0.0] and stat["includes_zero"]
    other = rng.choice(["light_review", "full_review"], 500)
    assert paired_bootstrap(modes, other, wrong, resamples=100, seed=3) == paired_bootstrap(
        modes, other, wrong, resamples=100, seed=3
    )


def test_bootstrap_separates_a_clearly_better_policy():
    rng = np.random.default_rng(1)
    wrong = rng.uniform(size=2000) < 0.2
    random_modes = np.where(rng.uniform(size=2000) < 0.5, "light_review", "full_review")
    oracle = np.where(wrong, "full_review", "light_review")  # sends exactly the errors
    result = paired_bootstrap(random_modes, oracle, wrong, resamples=200)
    catch = result["error_catch_rate"]
    assert catch["estimate"] > 0.4 and not catch["includes_zero"] and catch["ci95"][0] > 0
    assert result["light_review_error_rate"]["ci95"][1] < 0
