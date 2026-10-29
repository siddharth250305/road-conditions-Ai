"""The built frontend is served by the same app, after (and never in place of) the API routes."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("onnxruntime")

from api.main import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from tests.helpers import CLASSES, image_bytes  # noqa: E402

from roadcond.serving import ServingSettings  # noqa: E402

PAGE = "<!doctype html><title>Road Condition Survey Sheet</title><div id=root></div>"


@pytest.fixture
def dist(tmp_path):
    folder = tmp_path / "dist"
    (folder / "assets").mkdir(parents=True)
    (folder / "index.html").write_text(PAGE)
    (folder / "assets" / "app.js").write_text("console.log('hi')")
    (tmp_path / "secret.txt").write_text("not for the web")
    return folder


@pytest.fixture
def client(bundle_factory, dist):
    app = create_app(bundle_factory("small_cnn", confident=True), ServingSettings(), dist)
    return TestClient(app)


def test_the_page_and_its_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "Road Condition Survey Sheet" in page.text
    assert client.get("/assets/app.js").text == "console.log('hi')"


def test_api_routes_are_never_shadowed_by_the_page(client):
    health = client.get("/health")
    assert health.headers["content-type"].startswith("application/json")
    assert health.json()["status"] == "ok"
    assert client.get("/openapi.json").json()["info"]["title"] == "Road Condition AI"
    assert client.get("/docs").status_code == 200
    study = client.get("/study")  # no study.json in this bundle: still the API's JSON 404
    assert study.status_code == 404 and study.headers["content-type"].startswith("application/json")
    answer = client.post("/predict", files={"file": ("a.png", image_bytes(), "image/png")})
    assert answer.status_code == 200 and answer.json()["prediction"] in CLASSES


def test_files_outside_the_build_folder_are_not_reachable(client):
    for path in (
        "/../secret.txt",
        "/..%2fsecret.txt",
        "/%2e%2e/secret.txt",
        "/assets/../../secret.txt",
    ):
        response = client.get(path)
        assert "not for the web" not in response.text, path


def test_unknown_paths_are_a_404_not_the_page(client):
    assert client.get("/nope").status_code == 404


def test_without_a_build_the_api_still_runs(bundle_factory, tmp_path):
    app = create_app(
        bundle_factory("small_cnn", confident=True), ServingSettings(), tmp_path / "none"
    )
    client = TestClient(app)
    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 200


def test_the_frontend_folder_can_come_from_the_environment(bundle_factory, dist, monkeypatch):
    monkeypatch.setenv("ROADCOND_FRONTEND_DIR", str(dist))
    client = TestClient(create_app(bundle_factory("small_cnn", confident=True), ServingSettings()))
    assert "Road Condition Survey Sheet" in client.get("/").text


def test_health_describes_calibration_training_and_provenance(client):
    body = client.get("/health").json()
    assert body["uncertainty_threshold"] == 0.5
    assert body["calibration"]["temperature"] > 0 and "ece_after" in body["calibration"]
    assert body["training"]["dataset"] == "RSCD" and body["training"]["seed"] == 0
    assert len(body["provenance"]["checkpoint_sha256"]) == 64
    assert len(body["provenance"]["onnx_sha256"]) == 64
