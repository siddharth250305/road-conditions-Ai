import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("timm")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from PIL import Image  # noqa: E402
from tests.helpers import CLASSES, TEMPERATURE, image_bytes  # noqa: E402

from roadcond.config import load_config  # noqa: E402
from roadcond.data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms  # noqa: E402
from roadcond.export import (  # noqa: E402
    ServingModel,
    build_bundle,
    build_study,
    calibration_meta,
    export_onnx,
    feature_layer,
    load_checkpoint,
    pick_run,
    validate_onnx,
)
from roadcond.serving import cam_to_heatmap, decode_image, preprocess  # noqa: E402
from roadcond.train import MODEL_NAMES, build_model  # noqa: E402

CFG = load_config()
SIZE = tuple(CFG["rscd"]["image_size"])


def made(name):
    torch.manual_seed(0)
    model = build_model(name, CFG, pretrained=False).eval()
    return model, ServingModel(model, name).eval()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_onnx_runtime_matches_pytorch_for_any_batch(name, tmp_path):
    model, wrapper = made(name)
    path = export_onnx(wrapper, tmp_path / "model.onnx", SIZE)
    report = validate_onnx(wrapper, path, SIZE, tolerance=1e-4, batches=(1, 3))
    assert max(report["max_abs_diff"].values()) < 1e-4
    assert [p.name for p in tmp_path.iterdir()] == ["model.onnx"]  # one self-contained file

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[1:] == [3, *SIZE]
    x = torch.rand(5, 3, *SIZE)  # a batch size the export never saw
    logits, cam = session.run(None, {"image": x.numpy()})
    with torch.no_grad():
        ref_logits, ref_cam = wrapper(x)
    np.testing.assert_allclose(logits, ref_logits.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cam, ref_cam.numpy(), atol=1e-4, rtol=1e-4)
    assert logits.shape == (5, len(CLASSES)) and cam.shape[:2] == (5, len(CLASSES))


def test_normalisation_is_inside_the_graph(tmp_path):
    model, wrapper = made("small_cnn")
    path = export_onnx(wrapper, tmp_path / "model.onnx", SIZE)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.rand(2, 3, *SIZE)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    with torch.no_grad():
        expected = model((x - mean) / std).numpy()  # the plain model wants normalised input
    logits = session.run(None, {"image": x.numpy()})[0]  # the graph takes raw [0, 1] pixels
    np.testing.assert_allclose(logits, expected, atol=1e-4, rtol=1e-4)


def test_validation_fails_loudly_when_the_model_differs(tmp_path):
    _, wrapper = made("small_cnn")
    path = export_onnx(wrapper, tmp_path / "model.onnx", SIZE)
    with torch.no_grad():
        wrapper.model.classifier.weight.add_(0.5)  # PyTorch drifts away from the exported file
    with pytest.raises(ValueError, match="differs from PyTorch"):
        validate_onnx(wrapper, path, SIZE)


@pytest.mark.parametrize("name", ["small_cnn", "resnet50"])
def test_served_heatmap_equals_gradcam_at_the_last_feature_map(name, tmp_path):
    """The CAM output needs no gradients, yet equals Grad-CAM where the network pools."""
    pytest.importorskip("pytorch_grad_cam")
    from roadcond.explain import gradcam

    model, wrapper = made(name)
    path = export_onnx(wrapper, tmp_path / "model.onnx", SIZE)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.rand(2, 3, *SIZE)
    logits, cam = session.run(None, {"image": x.numpy()})
    top = logits.argmax(axis=1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    reference = gradcam(model, (x - mean) / std, feature_layer(model, name), list(top))
    served = np.stack([cam_to_heatmap(cam[i, c], *SIZE) for i, c in enumerate(top)])
    np.testing.assert_allclose(served, reference, atol=1e-4)


def test_server_preprocessing_matches_the_training_pipeline(bundle_factory):
    """Pillow resizing on the server versus torchvision in training: same answers, to a hair."""
    bundle = bundle_factory("small_cnn")
    session = ort.InferenceSession(str(bundle / "model.onnx"), providers=["CPUExecutionProvider"])
    model, _ = load_checkpoint(bundle.parent / "best.pt", CFG)
    transform = build_transforms(False, SIZE)
    for size in ((720, 480), (150, 100), (900, 1200)):  # downscale, upscale, large downscale
        data = image_bytes(size, fmt="PNG")
        image = decode_image(data)
        served_logits = session.run(None, {"image": preprocess(image, *SIZE)[0]})[0]
        with torch.no_grad():
            reference = model(transform(image)[None]).numpy()
        assert np.abs(served_logits - reference).max() < 1e-4, size  # measured: about 1e-7
        # The bound is meaningful: a wrong resize (nearest neighbour) misses it by a wide margin.
        nearest = np.asarray(image.resize((SIZE[1], SIZE[0]), Image.Resampling.NEAREST), np.float32)
        wrong = np.ascontiguousarray(nearest.transpose(2, 0, 1)[None] / 255.0)
        wrong_logits = session.run(None, {"image": wrong})[0]
        assert np.abs(wrong_logits - reference).max() > 1e-3, size


def test_bundle_metadata_describes_the_export(bundle_factory):
    bundle = bundle_factory("small_cnn")
    meta = json.loads((bundle / "meta.json").read_text())
    assert meta["classes"] == CLASSES and meta["model_name"] == "small_cnn" and meta["mock"]
    assert meta["input"]["height"] == SIZE[0] and meta["input"]["width"] == SIZE[1]
    assert meta["input"]["range"] == [0.0, 1.0] and meta["normalization"]["embedded_in_graph"]
    assert meta["outputs"]["cam_shape"] == [6, 15, 22]
    cal = meta["calibration"]
    assert cal["temperature"] == TEMPERATURE and cal["split"] == "test" and cal["ece_after"] == 0.04
    assert meta["training"]["seed"] == 0 and meta["training"]["test_macro_f1"] == 0.45
    assert meta["serving"]["uncertainty_threshold"] == 0.5
    prov = meta["provenance"]
    assert prov["onnx_sha256"] == hashlib.sha256((bundle / "model.onnx").read_bytes()).hexdigest()
    assert prov["onnx_validation"]["tolerance"] == 1e-4
    assert set(prov["exported_with"]) == {"python", "torch", "onnx", "onnxruntime"}


def test_build_bundle_refuses_bad_inputs(tmp_path, bundle_factory):
    bundle_factory("small_cnn")  # makes sure the optional stack is present
    with pytest.raises(FileNotFoundError):
        build_bundle(tmp_path / "missing.pt", tmp_path / "out", CFG, temperature=1.0)
    torch.save({"weights": 1}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="not a roadcond checkpoint"):
        build_bundle(tmp_path / "bad.pt", tmp_path / "out", CFG, temperature=1.0)
    model = build_model("small_cnn", CFG, pretrained=False)
    state = {"model": model.state_dict(), "model_name": "small_cnn", "classes": CLASSES}
    torch.save(state, tmp_path / "ok.pt")
    with pytest.raises(ValueError, match="calibration"):  # never serve an uncalibrated model
        build_bundle(tmp_path / "ok.pt", tmp_path / "out", CFG)
    with pytest.raises(ValueError, match="positive"):
        build_bundle(tmp_path / "ok.pt", tmp_path / "out", CFG, temperature=-1.0)
    (tmp_path / "other.json").write_text(json.dumps({"models": {"resnet50": []}}))
    with pytest.raises(ValueError, match="no results"):
        build_bundle(tmp_path / "ok.pt", tmp_path / "out", CFG, tmp_path / "other.json")


def test_temperature_alone_is_enough_and_pick_run_is_strict():
    assert calibration_meta(None, 2.0)["temperature"] == 2.0
    runs = [
        {"seed": 0, "temperature": 1.0, "val": {"macro_f1": 0.5}},
        {"seed": 1, "temperature": 2.0, "val": {"macro_f1": 0.7}},
    ]
    assert pick_run(runs, {"val_macro_f1": 0.7}, None)["seed"] == 1  # matched by macro-F1
    assert pick_run(runs, {"val_macro_f1": 0.7}, 0)["seed"] == 0  # an explicit seed wins
    with pytest.raises(ValueError, match="--seed"):
        pick_run(runs, {"val_macro_f1": 0.123}, None)


def test_study_is_built_only_from_result_files(study_results, tmp_path):
    out = tmp_path / "study.json"
    study = build_study(study_results, out, mock=True)
    assert json.loads(out.read_text()) == study
    assert study["mock"] and study["source_country"] == "Japan" and study["metric"] == "map50"
    by_country = {row["country"]: row for row in study["slope"]}
    assert set(by_country) == {"Norway", "Czech", "United_States"}
    assert by_country["Norway"]["relative_gap_map50"] == pytest.approx(0.25)
    assert by_country["United_States"]["relative_gap_map50"] < 0  # target above source
    curve = {p["labels"]: p for p in study["learning_curve"]["Norway"]}
    assert sorted(curve) == [0, 10, 50]
    assert curve[0]["map50"]["std"] is None and curve[10]["map50"]["n"] == 3
    assert curve[10]["recovered"]["mean"] is not None
    us = {p["labels"]: p for p in study["learning_curve"]["United_States"]}
    assert us[10]["recovered"]["mean"] is None  # no gap to recover
    recovery = study["recovery"]["Norway"]
    assert recovery["augmentation"]["map50"] == pytest.approx(0.62)
    assert recovery["pseudo_labels"]["threshold"] == 0.5


def test_study_needs_the_source_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="detection_source"):
        build_study(tmp_path, tmp_path / "study.json")


def test_study_without_optional_results_still_builds(study_results, tmp_path):
    (tmp_path / "fewshot").mkdir()
    (tmp_path / "detection_source.json").write_text(
        (study_results / "detection_source.json").read_text()
    )
    study = build_study(tmp_path, tmp_path / "study.json")
    assert study["learning_curve"] == {} and "augmentation" not in study["recovery"]["Norway"]


def test_image_modes_survive_decoding():
    for mode, fmt in (("RGB", "JPEG"), ("L", "PNG"), ("RGB", "PNG")):
        image = decode_image(image_bytes((64, 48), fmt, mode))
        assert image.mode == "RGB" and image.size == (64, 48)
