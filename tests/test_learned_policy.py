"""The margin provider and the learned review policy (roadcond.routing). NumPy only."""

import json

import numpy as np
import pytest
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
    ReviewRouter,
    ThresholdPolicy,
    read_decisions,
)
from roadcond.routing.providers import scale_probs
from roadcond.routing.run import build_providers, route_all

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
