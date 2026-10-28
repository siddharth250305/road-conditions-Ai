import base64
import io
import shutil

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
ort = pytest.importorskip("onnxruntime")

from api.main import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from tests.helpers import CLASSES, TEMPERATURE, image_bytes  # noqa: E402

from roadcond.serving import (  # noqa: E402
    ImageTooLargeError,
    InvalidImageError,
    ModelUnavailableError,
    Predictor,
    ServingSettings,
    cam_to_heatmap,
    decode_image,
    limits_note,
    preprocess,
    render_overlay,
    softmax,
)

PREDICT_KEYS = {
    "prediction", "confidence", "is_uncertain", "display_label", "top2", "probabilities",
    "heatmap_base64", "heatmap_media_type", "heatmap_class", "limits_note", "temperature",
    "uncertainty_threshold", "model",
}  # fmt: skip


def upload(client, data, name="road.png", content_type="image/png", **params):
    return client.post("/predict", files={"file": (name, data, content_type)}, params=params)


@pytest.fixture(scope="module")
def client(bundle_factory):
    return TestClient(create_app(bundle_factory("small_cnn", confident=True), ServingSettings()))


@pytest.fixture(scope="module")
def unsure_client(bundle_factory):
    return TestClient(create_app(bundle_factory("small_cnn", confident=False), ServingSettings()))


# --- /health and /predict -------------------------------------------------------------------


def test_health_reports_the_loaded_model(client):
    body = client.get("/health").json()
    assert client.get("/health").status_code == 200
    assert body["status"] == "ok" and body["model"] == "small_cnn" and body["classes"] == CLASSES
    assert body["mock"] is True and body["version"]


def test_predict_returns_the_documented_answer(client):
    response = upload(client, image_bytes())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == PREDICT_KEYS
    assert body["prediction"] == "dry" and body["display_label"] == "dry"
    assert body["is_uncertain"] is False and 0.5 <= body["confidence"] <= 1.0
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert set(body["probabilities"]) == set(CLASSES)
    top2 = body["top2"]
    assert len(top2) == 2 and top2[0]["class"] == body["prediction"]
    assert top2[0]["probability"] == body["confidence"] >= top2[1]["probability"]
    assert body["temperature"] == TEMPERATURE and body["model"] == {
        "name": "small_cnn",
        "mock": True,
    }
    assert body["heatmap_media_type"] == "image/png" and body["heatmap_class"] == "dry"


def test_confidence_is_the_temperature_scaled_probability(client, bundle_factory):
    bundle = bundle_factory("small_cnn", confident=True)
    data = image_bytes((300, 200), "JPEG")
    session = ort.InferenceSession(str(bundle / "model.onnx"), providers=["CPUExecutionProvider"])
    logits = session.run(None, {"image": preprocess(decode_image(data), 240, 360)[0]})[0][0]
    expected = softmax(logits / TEMPERATURE)
    body = upload(client, data, "road.jpg", "image/jpeg").json()
    assert body["confidence"] == pytest.approx(float(expected.max()), abs=1e-6)
    for name, p in zip(CLASSES, expected, strict=True):
        assert body["probabilities"][name] == pytest.approx(float(p), abs=1e-6)
    assert not np.allclose(expected, softmax(logits))  # the temperature really changes them


def test_heatmap_is_a_png_overlay_at_model_size(client):
    body = upload(client, image_bytes((500, 300))).json()
    overlay = Image.open(io.BytesIO(base64.b64decode(body["heatmap_base64"])))
    assert overlay.format == "PNG" and overlay.size == (360, 240) and overlay.mode == "RGB"
    assert np.asarray(overlay).std() > 1  # not a blank picture


def test_heatmap_can_be_skipped(client):
    body = upload(client, image_bytes(), heatmap="false").json()
    assert body["heatmap_base64"] is None and body["prediction"] == "dry"


def test_low_confidence_gives_not_sure_with_the_top_two_classes(unsure_client):
    body = upload(unsure_client, image_bytes()).json()
    assert body["confidence"] < 0.5
    assert body["is_uncertain"] is True and body["display_label"] == "Not sure"
    assert body["prediction"] == body["top2"][0]["class"]  # the best guess is still reported
    assert (
        len(body["top2"]) == 2 and body["top2"][0]["probability"] >= body["top2"][1]["probability"]
    )
    assert body["heatmap_base64"]  # the heatmap is still available


def test_the_threshold_decides_what_is_uncertain(bundle_factory):
    confident = bundle_factory("small_cnn", confident=True)
    strict = TestClient(create_app(confident, ServingSettings(uncertainty_threshold=0.999)))
    assert upload(strict, image_bytes()).json()["is_uncertain"] is True
    unsure = bundle_factory("small_cnn", confident=False)
    lenient = TestClient(create_app(unsure, ServingSettings(uncertainty_threshold=0.0)))
    body = upload(lenient, image_bytes()).json()
    assert body["is_uncertain"] is False and body["uncertainty_threshold"] == 0.0


def test_the_limits_note_states_training_data_and_calibration(client):
    note = upload(client, image_bytes()).json()["limits_note"]
    for phrase in (
        "RSCD",
        "China",
        "unknown",
        "calibrated on held-out",
        "1.70",
        "0.040",
        "not a safety",
    ):
        assert phrase in note, phrase
    assert note.startswith("DEMO MODEL BUILT FROM MOCK DATA")  # the fixture bundle is a mock


def test_the_limits_note_without_calibration_numbers_or_mock_flag():
    meta = {
        "classes": CLASSES,
        "calibration": {"temperature": 2.0, "split": None, "ece_after": None},
    }
    note = limits_note(meta)
    assert "temperature 2.00" in note and "expected calibration error" not in note
    assert not note.startswith("DEMO")


@pytest.mark.parametrize(
    ("mode", "fmt"), [("RGB", "JPEG"), ("RGB", "PNG"), ("L", "PNG"), ("RGBA", "PNG"), ("P", "GIF")]
)
def test_common_image_formats_and_modes_are_accepted(client, mode, fmt):
    image = Image.new("RGB", (120, 80), (90, 100, 110)).convert(mode)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    assert upload(client, buffer.getvalue(), "x", "application/octet-stream").status_code == 200


def test_the_client_supplied_name_and_type_do_not_matter(client):
    body = upload(client, image_bytes(), "secret_name.exe", "text/plain")
    assert body.status_code == 200 and "secret_name" not in body.text


def test_phone_photos_are_rotated_by_their_exif_flag():
    image = Image.new("RGB", (100, 50), (10, 20, 30))
    exif = image.getexif()
    exif[0x0112] = 6  # "rotate 270 degrees to display"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    assert decode_image(buffer.getvalue()).size == (50, 100)


# --- errors ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"this is not an image",
        b"\x89PNG\r\n\x1a\n" + b"garbage" * 20,
        image_bytes((360, 240), "JPEG")[:900],
        image_bytes((360, 240), "PNG")[:-40],
    ],
    ids=["empty", "text", "png-signature-only", "truncated-jpeg", "truncated-png"],
)
def test_corrupt_uploads_are_a_400_without_internals(client, payload):
    response = upload(client, payload, "bad.jpg", "image/jpeg")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert isinstance(detail, str) and detail
    assert "Traceback" not in detail and "/" not in detail and "bad.jpg" not in detail


def test_images_with_too_many_pixels_are_a_413_before_decoding(bundle_factory):
    app = create_app(
        bundle_factory("small_cnn", confident=True), ServingSettings(max_pixels=10_000)
    )
    response = upload(TestClient(app), image_bytes((200, 200)))
    assert response.status_code == 413 and "pixels" in response.json()["detail"]
    with pytest.raises(ImageTooLargeError):
        decode_image(image_bytes((200, 200)), max_pixels=10_000)


def test_too_many_bytes_are_a_413(bundle_factory):
    bundle = bundle_factory("small_cnn", confident=True)
    small = TestClient(create_app(bundle, ServingSettings(max_upload_bytes=1000)))
    assert upload(small, image_bytes((360, 240), "PNG")).status_code == 413  # in the handler
    assert upload(small, b"x" * 200_000).status_code == 413  # refused from Content-Length


def test_a_missing_file_field_is_a_422(client):
    assert (
        client.post(
            "/predict", files={"picture": ("a.png", image_bytes(), "image/png")}
        ).status_code
        == 422
    )
    assert client.post("/predict").status_code == 422


def test_no_model_means_503_but_the_app_still_starts(tmp_path):
    client = TestClient(create_app(tmp_path / "nothing"))
    health = client.get("/health")
    assert health.status_code == 503 and health.json()["status"] == "unavailable"
    assert "not found" in health.json()["detail"]
    assert upload(client, image_bytes()).status_code == 503
    assert client.get("/study").status_code == 404


def test_a_broken_bundle_is_refused_with_a_clear_reason(bundle_factory, tmp_path):
    source = bundle_factory("small_cnn", confident=True)
    shutil.copytree(source, tmp_path / "b")
    (tmp_path / "b" / "meta.json").write_text("{not json")
    with pytest.raises(ModelUnavailableError, match="not valid JSON"):
        Predictor(tmp_path / "b")
    (tmp_path / "b" / "meta.json").write_text('{"format_version": 99}')
    with pytest.raises(ModelUnavailableError, match="Unsupported bundle format"):
        Predictor(tmp_path / "b")


# --- /study ---------------------------------------------------------------------------------


def test_study_serves_the_saved_benchmark(bundle_factory, study_results, tmp_path):
    from roadcond.export import build_study

    shutil.copytree(bundle_factory("small_cnn", confident=True), tmp_path / "bundle")
    build_study(study_results, tmp_path / "bundle" / "study.json", mock=True)
    response = TestClient(create_app(tmp_path / "bundle")).get("/study")
    assert response.status_code == 200
    study = response.json()
    assert study["mock"] is True and study["source_country"] == "Japan"
    assert [row["country"] for row in study["slope"]] == ["Norway", "Czech", "United_States"]
    assert {p["labels"] for p in study["learning_curve"]["Norway"]} == {0, 10, 50}
    assert study["learning_curve"]["Norway"][0]["map50"]["std"] is None


def test_study_is_a_404_not_invented_data_when_absent(client):
    response = client.get("/study")
    assert response.status_code == 404 and "No study results" in response.json()["detail"]


# --- the rest of the surface ----------------------------------------------------------------


def test_openapi_lists_the_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/predict", "/study", "/health"}
    assert "post" in paths["/predict"] and "get" in paths["/study"] and "get" in paths["/health"]


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ROADCOND_UNCERTAINTY_THRESHOLD", "0.7")
    monkeypatch.setenv("ROADCOND_MAX_UPLOAD_MB", "2")
    monkeypatch.setenv("ROADCOND_MAX_PIXELS", "1000")
    monkeypatch.setenv("ROADCOND_HEATMAP_ALPHA", "0.3")
    settings = ServingSettings.from_env()
    assert settings.uncertainty_threshold == 0.7 and settings.max_upload_bytes == 2 * 1024 * 1024
    assert settings.max_pixels == 1000 and settings.heatmap_alpha == 0.3
    monkeypatch.delenv("ROADCOND_UNCERTAINTY_THRESHOLD")
    assert ServingSettings.from_env().uncertainty_threshold is None


def test_cors_is_off_unless_configured(bundle_factory, monkeypatch):
    bundle = bundle_factory("small_cnn", confident=True)
    headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    assert (
        "access-control-allow-origin"
        not in TestClient(create_app(bundle)).options("/predict", headers=headers).headers
    )
    monkeypatch.setenv("ROADCOND_CORS_ORIGINS", "http://localhost:5173")
    allowed = TestClient(create_app(bundle)).options("/predict", headers=headers)
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_overlay_and_heatmap_helpers():
    cam = np.random.default_rng(0).normal(size=(8, 12)).astype(np.float32)
    heat = cam_to_heatmap(cam, 240, 360)
    assert (
        heat.shape == (240, 360)
        and heat.min() == 0.0
        and heat.max() == pytest.approx(1.0, abs=1e-5)
    )
    rgb = np.full((240, 360, 3), 100, dtype=np.uint8)
    png = render_overlay(rgb, heat, alpha=0.5)
    assert Image.open(io.BytesIO(png)).size == (360, 240)
    with pytest.raises(ValueError, match="sizes differ"):
        render_overlay(rgb, heat[:10], 0.5)
    assert render_overlay(rgb, heat, 0.0) == render_overlay(
        rgb, np.zeros_like(heat), 0.0
    )  # alpha 0: picture only


def test_decode_errors_have_distinct_types():
    with pytest.raises(InvalidImageError, match="empty"):
        decode_image(b"")
    with pytest.raises(InvalidImageError, match="not a readable image"):
        decode_image(b"abc")
    assert issubclass(ImageTooLargeError, InvalidImageError)
