import json
import shutil
import threading

import pytest

from roadcond.routing.live import (
    LOG_UNAVAILABLE_REASON,
    OFFLINE_KEYS,
    ROUTING_FILE,
    LiveRouting,
    LiveRoutingError,
    ReviewConflictError,
    UnknownDecisionError,
    load_live_routing,
    review_status,
    write_live_routing,
)
from roadcond.routing.log import read_log

CLASSES = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]
SHA = "a" * 64


def routing_section(model="cnn", high=0.85, low=0.50, safety=("ice",), agreement=None):
    return {
        "policy": "threshold",
        "policy_version": "1.0.0",
        "primary_model": model,
        "agreement_models": agreement if agreement is not None else [model],
        "thresholds": {"high": high, "low": low},
        "safety_critical_classes": list(safety),
        "probs_file": "artifacts/results/cnn_probs.npz",
        "out_dir": "artifacts/routing_cnn",
        "temperature_search": {"t_min": 0.05, "t_max": 20.0, "steps": 50},
        "ece_bins": 15,
        "sweep": {"high_start": 0.6, "high_stop": 0.95, "high_step": 0.05},
    }


def summary(temperature=1.0):
    return {"calibration": {"temperature": temperature}, **{k: f"<{k}>" for k in OFFLINE_KEYS}}


def make_bundle(tmp_path, model="cnn", mock=False, sha=SHA):
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": model,
        "mock": mock,
        "classes": CLASSES,
        "provenance": {"onnx_sha256": sha},
    }
    (bundle / "meta.json").write_text(json.dumps(meta))
    return bundle, meta


def live(tmp_path, **section):
    bundle, meta = make_bundle(tmp_path)
    write_live_routing(bundle, routing_section(**section), CLASSES, summary())
    return LiveRouting(bundle, meta, tmp_path / "logs" / "decisions.jsonl")


def probs(top="dry", p=0.95):
    rest = (1.0 - p) / (len(CLASSES) - 1)
    return {c: (p if c == top else rest) for c in CLASSES}


# --- routing.json --------------------------------------------------------------------------


def test_routing_json_records_the_check_and_the_model_it_was_checked_on(tmp_path):
    bundle, meta = make_bundle(tmp_path)
    path = write_live_routing(bundle, routing_section(), CLASSES, summary(temperature=0.8))
    spec = json.loads(path.read_text())
    assert path == bundle / ROUTING_FILE
    assert spec["model_name"] == "cnn" and spec["onnx_sha256"] == SHA
    assert spec["temperature"] == 0.8 and spec["classes"] == CLASSES
    assert set(spec["offline_check"]) == set(OFFLINE_KEYS)
    assert load_live_routing(bundle, meta) == spec


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"mock": True}, "mock"),
        ({"model": "other"}, "serves 'other'"),
    ],
)
def test_routing_json_is_refused_for_a_mock_or_different_model(tmp_path, kwargs, match):
    bundle, _ = make_bundle(tmp_path, **kwargs)
    with pytest.raises(LiveRoutingError, match=match):
        write_live_routing(bundle, routing_section(), CLASSES, summary())


def test_routing_json_needs_one_model_and_the_bundle_classes(tmp_path):
    bundle, _ = make_bundle(tmp_path)
    with pytest.raises(LiveRoutingError, match="one model"):
        write_live_routing(bundle, routing_section(agreement=["cnn", "rf"]), CLASSES, summary())
    with pytest.raises(LiveRoutingError, match="classes"):
        write_live_routing(bundle, routing_section(), CLASSES[::-1], summary())


def test_loading_refuses_a_missing_broken_stale_or_edited_routing_json(tmp_path):
    bundle, meta = make_bundle(tmp_path)
    with pytest.raises(LiveRoutingError, match="not found"):
        load_live_routing(bundle, meta)
    (bundle / ROUTING_FILE).write_text("{not json")
    with pytest.raises(LiveRoutingError, match="not valid JSON"):
        load_live_routing(bundle, meta)

    write_live_routing(bundle, routing_section(), CLASSES, summary())
    retrained = {**meta, "provenance": {"onnx_sha256": "b" * 64}}
    with pytest.raises(LiveRoutingError, match="different model"):
        load_live_routing(bundle, retrained)

    spec = json.loads((bundle / ROUTING_FILE).read_text())
    spec["routing"]["thresholds"]["high"] = 0.1  # edited by hand after the check
    (bundle / ROUTING_FILE).write_text(json.dumps(spec))
    with pytest.raises(LiveRoutingError, match="config_hash"):
        load_live_routing(bundle, meta)


# --- review status -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, labels, expected",
    [
        ("light_review", [], ("pending", None)),
        ("light_review", ["wet"], ("done", "wet")),
        ("full_review", ["dry"], ("done", "dry")),
        ("second_review", ["ice"], ("pending", None)),
        ("second_review", ["ice", "ice"], ("done", "ice")),
        ("second_review", ["ice", "wet"], ("disputed", None)),
        ("second_review", ["ice", "wet", "ice"], ("done", "ice")),
    ],
)
def test_review_status(mode, labels, expected):
    assert review_status(mode, labels) == expected


# --- LiveRouting ---------------------------------------------------------------------------


def test_each_photo_gets_a_logged_decision(tmp_path):
    routing = live(tmp_path)
    confident = routing.route("sha-1", probs("dry", 0.95))
    unsure = routing.route("sha-2", probs("wet", 0.60))
    icy = routing.route("sha-3", probs("ice", 0.99))
    assert [confident["mode"], unsure["mode"], icy["mode"]] == [
        "light_review",
        "full_review",
        "second_review",
    ]
    assert confident["recorded"] and confident["reviews_needed"] == 1
    assert icy["reviews_needed"] == 2 and "safety_class:ice" in icy["reasons"]
    decisions, outcomes = read_log(routing.log_path)
    assert [d["image_id"] for d in decisions] == ["sha-1", "sha-2", "sha-3"]
    assert decisions[0]["config_hash"] == routing.spec["config_hash"]
    assert all(v == [] for v in outcomes.values())


def test_the_fitted_temperature_is_applied_to_the_served_probabilities(tmp_path):
    bundle, meta = make_bundle(tmp_path)
    write_live_routing(bundle, routing_section(), CLASSES, summary(temperature=1.5))
    routing = LiveRouting(bundle, meta, tmp_path / "d.jsonl")
    # 0.95 is light at T=1; T=1.5 flattens it to about 0.81, a full review
    assert routing.route("x", probs("dry", 0.95))["mode"] == "full_review"


def test_one_review_finishes_light_and_full_reviews(tmp_path):
    routing = live(tmp_path)
    decision = routing.route("x", probs("dry", 0.95))["decision_id"]
    state = routing.review(decision, "wet", "alex")
    assert state["status"] == "done" and state["agreed_label"] == "wet"
    assert state["reviewers"] == ["alex"]
    with pytest.raises(ReviewConflictError, match="already been reviewed"):
        routing.review(decision, "wet", "sam")
    _, outcomes = read_log(routing.log_path)
    assert outcomes[decision][0]["outcome"] == "corrected:wet"  # the model said dry


def test_a_second_review_needs_two_reviewers_who_agree(tmp_path):
    routing = live(tmp_path)
    decision = routing.route("x", probs("ice", 0.99))["decision_id"]
    assert routing.review(decision, "ice", "alex")["status"] == "pending"
    with pytest.raises(ReviewConflictError, match="already reviewed"):
        routing.review(decision, "ice", "alex")
    assert routing.review(decision, "wet", "sam")["status"] == "disputed"
    final = routing.review(decision, "ice", "kim")
    assert final["status"] == "done" and final["agreed_label"] == "ice"
    _, outcomes = read_log(routing.log_path)
    assert [o["outcome"] for o in outcomes[decision]] == [
        "confirmed:ice",
        "corrected:wet",
        "confirmed:ice",
    ]


def test_reviews_are_checked(tmp_path):
    routing = live(tmp_path)
    decision = routing.route("x", probs("dry", 0.95))["decision_id"]
    with pytest.raises(ValueError, match="Unknown class"):
        routing.review(decision, "gravel", "alex")
    with pytest.raises(UnknownDecisionError):
        routing.review("no-such-decision", "dry", "alex")


def test_the_queue_puts_the_strictest_first_and_hides_what_you_reviewed(tmp_path):
    routing = live(tmp_path)
    light = routing.route("a", probs("dry", 0.95))["decision_id"]
    full = routing.route("b", probs("wet", 0.60))["decision_id"]
    second = routing.route("c", probs("ice", 0.99))["decision_id"]
    assert [s["decision_id"] for s in routing.queue("alex")] == [second, full, light]
    routing.review(second, "ice", "alex")
    assert [s["decision_id"] for s in routing.queue("alex")] == [full, light]
    sams_view = routing.queue("sam")
    assert sams_view[0]["decision_id"] == second and sams_view[0]["reviewers"] == ["alex"]
    assert "ice" not in json.dumps(sams_view[0]["reviewers"])  # names only, never labels
    assert routing.queue("sam", limit=1) == sams_view[:1]
    assert routing.counts()["second_review"] == {"open": 1, "done": 0}


def test_decisions_and_reviews_survive_a_restart(tmp_path):
    routing = live(tmp_path)
    decision = routing.route("x", probs("ice", 0.99))["decision_id"]
    routing.review(decision, "ice", "alex")
    meta = json.loads((tmp_path / "bundle" / "meta.json").read_text())
    restarted = LiveRouting(tmp_path / "bundle", meta, routing.log_path)
    [item] = restarted.queue("sam")
    assert item["decision_id"] == decision and item["reviewers"] == ["alex"]
    assert restarted.review(decision, "ice", "sam")["status"] == "done"


def test_an_unreadable_log_stops_routing_instead_of_losing_history(tmp_path):
    bundle, meta = make_bundle(tmp_path)
    write_live_routing(bundle, routing_section(), CLASSES, summary())
    log_path = tmp_path / "d.jsonl"
    log_path.write_text("{cut off\n")
    with pytest.raises(LiveRoutingError, match="unreadable"):
        LiveRouting(bundle, meta, log_path)


def test_a_failed_log_write_gives_an_unrecorded_full_review(tmp_path):
    routing = live(tmp_path)

    def broken(_decision):
        raise OSError("disk full")

    routing.log.write = broken
    review = routing.route("x", probs("dry", 0.99))
    assert review["mode"] == "full_review" and review["reasons"] == [LOG_UNAVAILABLE_REASON]
    assert review["decision_id"] is None and review["recorded"] is False
    assert routing.queue("alex") == []


def test_concurrent_requests_write_whole_lines(tmp_path):
    routing = live(tmp_path)
    errors = []

    def worker(n):
        try:
            for i in range(50):
                d = routing.route(f"{n}-{i}", probs("wet", 0.6))["decision_id"]
                routing.review(d, "wet", f"reviewer-{n}")
        except Exception as err:  # pragma: no cover - reported below
            errors.append(err)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    decisions, outcomes = read_log(routing.log_path)  # raises on a torn line
    assert len(decisions) == 400 and sum(len(o) for o in outcomes.values()) == 400


# --- the API ---------------------------------------------------------------------------------

TOKENS = {"alex-token-0123456789": "alex", "sam-token-0123456789ab": "sam"}


@pytest.fixture
def routed_bundle(bundle_factory, tmp_path):
    """A copy of the confident test bundle (predicts dry) with routing.json added."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    def build(**section):
        bundle = tmp_path / "routed"
        if bundle.exists():
            shutil.rmtree(bundle)
        shutil.copytree(bundle_factory("small_cnn", confident=True), bundle)
        meta = json.loads((bundle / "meta.json").read_text())
        meta["mock"] = False  # write_live_routing refuses mock bundles
        (bundle / "meta.json").write_text(json.dumps(meta))
        write_live_routing(bundle, routing_section("small_cnn", **section), CLASSES, summary())
        return bundle

    return build


def app_client(bundle, tmp_path, tokens=TOKENS, images=True):
    from api.main import create_app
    from fastapi.testclient import TestClient

    return TestClient(
        create_app(
            bundle,
            decision_log=tmp_path / "logs" / "decisions.jsonl",
            reviewer_tokens=tokens,
            review_image_dir=tmp_path / "photos" if images else None,
        )
    )


def post_photo(client):
    from tests.helpers import image_bytes

    return client.post("/predict", files={"file": ("road.png", image_bytes(), "image/png")})


def auth(name):
    token = next(t for t, n in TOKENS.items() if n == name)
    return {"Authorization": f"Bearer {token}"}


def test_predict_carries_a_logged_review_decision(routed_bundle, tmp_path):
    client = app_client(routed_bundle(high=0.0, low=0.0), tmp_path)  # everything confident
    body = post_photo(client).json()
    review = body["review"]
    assert body["prediction"] == "dry"  # the rest of the answer is unchanged
    assert review["mode"] == "light_review" and review["recorded"] is True
    decisions, _ = read_log(tmp_path / "logs" / "decisions.jsonl")
    assert [d["decision_id"] for d in decisions] == [review["decision_id"]]
    assert len(decisions[0]["image_id"]) == 64  # sha256 of the upload, not a file name
    health = client.get("/health").json()["routing"]
    assert health["enabled"] and health["reviewing"] and health["stores_images"]
    assert health["thresholds"] == {"high": 0.0, "low": 0.0}


def test_routing_is_off_when_routing_json_was_checked_on_another_model(routed_bundle, tmp_path):
    bundle = routed_bundle()
    spec = json.loads((bundle / ROUTING_FILE).read_text())
    spec["onnx_sha256"] = "0" * 64
    (bundle / ROUTING_FILE).write_text(json.dumps(spec))
    client = app_client(bundle, tmp_path)
    assert post_photo(client).json()["review"] is None
    health = client.get("/health").json()["routing"]
    assert health["enabled"] is False and "different model" in health["detail"]
    assert client.get("/review/queue", headers=auth("alex")).status_code == 503


def test_review_endpoints_need_a_reviewer_token(routed_bundle, tmp_path):
    client = app_client(routed_bundle(), tmp_path)
    assert client.get("/review/queue").status_code == 401
    bad = {"Authorization": "Bearer wrong-token-0123456789"}
    assert client.get("/review/queue", headers=bad).status_code == 401
    no_tokens = app_client(routed_bundle(), tmp_path, tokens={})
    assert no_tokens.get("/review/queue", headers=auth("alex")).status_code == 503


def test_two_reviewers_finish_a_second_review_through_the_api(routed_bundle, tmp_path):
    client = app_client(routed_bundle(high=1.0, low=1.0), tmp_path)  # everything strict
    decision = post_photo(client).json()["review"]["decision_id"]

    queue = client.get("/review/queue", headers=auth("alex")).json()
    assert queue["reviewer"] == "alex"
    [item] = queue["items"]
    assert item["mode"] == "second_review" and item["has_image"] is True
    image = client.get(f"/review/{decision}/image", headers=auth("alex"))
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert image.headers["x-content-type-options"] == "nosniff"

    first = client.post(f"/review/{decision}", json={"label": "wet"}, headers=auth("alex"))
    assert first.status_code == 200 and first.json()["status"] == "pending"
    again = client.post(f"/review/{decision}", json={"label": "wet"}, headers=auth("alex"))
    assert again.status_code == 409
    assert client.get("/review/queue", headers=auth("alex")).json()["items"] == []
    second = client.post(f"/review/{decision}", json={"label": "wet"}, headers=auth("sam"))
    assert second.json()["status"] == "done" and second.json()["agreed_label"] == "wet"
    assert second.json()["reviewers"] == ["alex", "sam"]


def test_review_errors_are_specific(routed_bundle, tmp_path):
    client = app_client(routed_bundle(), tmp_path, images=False)
    decision = post_photo(client).json()["review"]["decision_id"]
    headers = auth("alex")
    wrong_label = client.post(f"/review/{decision}", json={"label": "gravel"}, headers=headers)
    assert wrong_label.status_code == 422
    missing = "00000000-0000-4000-8000-000000000000"
    unknown = client.post(f"/review/{missing}", json={"label": "dry"}, headers=headers)
    assert unknown.status_code == 404
    assert client.get(f"/review/{decision}/image", headers=headers).status_code == 404  # none kept
    assert client.get("/review/..%2Fmeta.json/image", headers=headers).status_code == 404
    assert client.get("/review/queue", headers=headers).json()["items"][0]["has_image"] is False


def test_reviewer_tokens_from_the_environment():
    pytest.importorskip("fastapi")
    from api.main import parse_reviewer_tokens

    tokens = parse_reviewer_tokens("alex:alex-token-0123456789, sam:short,:no-name-token-0123456")
    assert tokens == {"alex-token-0123456789": "alex"}
    assert parse_reviewer_tokens("") == {}


# --- monitoring ------------------------------------------------------------------------------


def test_wilson_interval():
    from roadcond.routing.monitor import wilson_interval

    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    assert wilson_interval(0, 50)[0] == 0.0


def reviewed_log(tmp_path, light_wrong, light_right):
    routing = live(tmp_path)
    for i in range(light_wrong + light_right):
        d = routing.route(f"light-{i}", probs("dry", 0.95))["decision_id"]
        routing.review(d, "wet" if i < light_wrong else "dry", "alex")
    icy = routing.route("icy", probs("ice", 0.99))["decision_id"]
    routing.review(icy, "ice", "alex")
    routing.review(icy, "water", "sam")  # disputed: still open
    unsure = routing.route("unsure", probs("wet", 0.6))["decision_id"]
    routing.review(unsure, "water", "alex")  # a model error the full review caught
    routing.route("open", probs("dry", 0.95))
    spec = {
        **routing.spec,
        "offline_check": {"error_rate_by_mode": {"light_review": {"error_rate": 0.1}}},
    }
    return routing, spec


def test_the_report_compares_live_light_reviews_with_the_offline_check(tmp_path):
    from roadcond.routing.monitor import review_report

    routing, spec = reviewed_log(tmp_path, light_wrong=15, light_right=25)
    report = review_report(*read_log(routing.log_path), spec)
    light = report["reviews"]["light_review"]
    assert light == {**light, "open": 1, "done": 40, "errors": 15, "error_rate": 15 / 40}
    assert report["reviews"]["full_review"]["errors"] == 1
    assert report["reviews"]["second_review"] == {
        **report["reviews"]["second_review"],
        "open": 1,
        "done": 0,
    }
    assert report["error_catch_rate"] == {"caught": 1, "errors": 16, "rate": 1 / 16}
    assert report["second_review_disagreement"] == {"pairs": 1, "disagreed": 1, "rate": 1.0}
    check = report["light_review_check"]
    assert check["alert"] is True and "refit" in check["verdict"]  # 37.5% vs 10% offline


def test_the_report_does_not_alert_on_too_few_or_consistent_reviews(tmp_path):
    from roadcond.routing.monitor import review_report

    few, spec = reviewed_log(tmp_path / "few", light_wrong=5, light_right=5)
    assert review_report(*read_log(few.log_path), spec)["light_review_check"]["alert"] is False
    fine, spec = reviewed_log(tmp_path / "fine", light_wrong=3, light_right=37)
    check = review_report(*read_log(fine.log_path), spec)["light_review_check"]
    assert check["alert"] is False and "consistent" in check["verdict"]


def test_the_report_only_counts_decisions_made_with_the_current_config(tmp_path):
    from roadcond.routing.monitor import review_report

    routing, spec = reviewed_log(tmp_path, light_wrong=1, light_right=1)
    decisions, outcomes = read_log(routing.log_path)
    old = {**decisions[0], "decision_id": "old", "config_hash": "previous"}
    report = review_report([old, *decisions], outcomes | {"old": []}, spec)
    assert report["config_hashes"] == {"previous": 1, spec["config_hash"]: len(decisions)}
    assert report["decisions"] == len(decisions)


def test_the_report_script_reads_the_log_and_routing_json(tmp_path, capsys):
    from roadcond.routing.monitor import main

    routing, spec = reviewed_log(tmp_path, light_wrong=0, light_right=2)
    (tmp_path / "bundle" / ROUTING_FILE).write_text(json.dumps(spec))
    out = tmp_path / "report.json"
    main(["--log", str(routing.log_path), "--bundle", str(tmp_path / "bundle"), "--out", str(out)])
    assert json.loads(out.read_text())["decisions"] == 5
    assert "too few finished light reviews" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not found"):
        main(["--log", str(tmp_path / "missing.jsonl")])
