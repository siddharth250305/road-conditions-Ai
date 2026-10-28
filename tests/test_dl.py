import copy
import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("timm")
pytest.importorskip("pytorch_grad_cam")

import torch.nn.functional as F  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from roadcond.calibration import (  # noqa: E402
    TemperatureScaler,
    expected_calibration_error,
    nll,
    reliability_bins,
)
from roadcond.config import load_config  # noqa: E402
from roadcond.data.dataset import RoadDataset, build_transforms, make_loader  # noqa: E402
from roadcond.data.mock import make_mock_rscd  # noqa: E402
from roadcond.data.rscd import load_splits, prepare  # noqa: E402
from roadcond.explain import gradcam, overlay, select_examples  # noqa: E402
from roadcond.models.pretrained import (  # noqa: E402
    build_pretrained,
    count_parameters,
    freeze_batchnorm_stats,
    set_backbone_trainable,
)
from roadcond.models.small_cnn import SmallCNN  # noqa: E402
from roadcond.train import EarlyStopper, build_model, fit, main, select_device  # noqa: E402

CFG = load_config()
CLASSES = CFG["rscd"]["classes"]
MODELS = ["small_cnn", "resnet50", "efficientnet"]
SMALL = (48, 72)  # mock image size (height, width)


@pytest.fixture(scope="module")
def mock_splits(tmp_path_factory):
    base = tmp_path_factory.mktemp("dl")
    make_mock_rscd(base / "rscd", CFG["rscd"]["split_dirs"], n_per_label=(6, 2, 2))
    prepare(CFG, base / "rscd", base / "splits", train_size=36)
    return base / "rscd", base / "splits"


def small_loaders(mock_splits, batch_size=16):
    root, splits_dir = mock_splits
    splits = load_splits(splits_dir)
    train = RoadDataset(splits["train"], root, CLASSES, build_transforms(True, SMALL))
    val = RoadDataset(splits["val"], root, CLASSES, build_transforms(False, SMALL))
    return make_loader(train, batch_size, shuffle=True), make_loader(val, batch_size)


# --- models ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_forward_loss_and_backward(name):
    torch.manual_seed(0)
    model = build_model(name, CFG, pretrained=False)
    x, y = torch.randn(2, 3, 240, 360), torch.tensor([0, 5])
    logits = model(x)
    assert logits.shape == (2, len(CLASSES))
    loss = F.cross_entropy(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_small_cnn_is_small_and_has_four_stages():
    model = SmallCNN(6)
    assert len(model.features) == 4
    assert 100_000 < count_parameters(model) < 1_000_000


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("vgg", CFG)


def test_freezing_leaves_only_the_head_trainable():
    model = build_pretrained("resnet50", 6, pretrained=False, freeze_backbone=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}
    assert model.fc.out_features == 6
    set_backbone_trainable(model, True)
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_batchnorm_stats_survives_train_mode():
    model = build_pretrained("resnet50", 6, pretrained=False, freeze_backbone=True)
    model.train()
    freeze_batchnorm_stats(model)
    assert not model.bn1.training
    assert model.fc.training


# --- training engine ------------------------------------------------------------------------


def test_early_stopper_stops_after_patience_and_keeps_best():
    stopper = EarlyStopper(patience=2)
    assert stopper.update(0.5, 0)
    assert stopper.update(0.7, 1)
    assert not stopper.update(0.6, 2) and not stopper.should_stop
    assert not stopper.update(0.7, 3)  # a tie is not an improvement
    assert stopper.should_stop
    assert (stopper.best, stopper.best_epoch) == (0.7, 1)
    restored = EarlyStopper(patience=2)
    restored.load_state_dict(stopper.state_dict())
    assert restored.should_stop and restored.best_epoch == 1


def test_fit_resumes_from_last_checkpoint(mock_splits, tmp_path):
    train_loader, val_loader = small_loaders(mock_splits)
    mcfg = {"epochs": 1, "patience": 3, "lr": 1e-3}
    ccfg = {"amp": False, "weight_decay": 0.01}
    device = torch.device("cpu")
    first = fit(SmallCNN(6), train_loader, val_loader, mcfg, ccfg, device, tmp_path)
    assert len(first["history"]) == 1 and (tmp_path / "best.pt").is_file()

    second = fit(
        SmallCNN(6),
        train_loader,
        val_loader,
        {**mcfg, "epochs": 2},
        ccfg,
        device,
        tmp_path,
        resume=True,
    )
    assert len(second["history"]) == 2
    assert second["history"][0] == first["history"][0]
    assert json.loads((tmp_path / "history.json").read_text())[1]["epoch"] == 1


def test_frozen_epoch_trains_only_the_head_and_keeps_bn_statistics(mock_splits, tmp_path):
    train_loader, val_loader = small_loaders(mock_splits)
    model = build_pretrained("resnet50", 6, pretrained=False, freeze_backbone=True)
    conv_before = model.conv1.weight.detach().clone()
    head_before = model.fc.weight.detach().clone()
    bn_before = model.bn1.running_mean.detach().clone()
    mcfg = {"epochs": 1, "patience": 3, "lr": 1e-2, "backbone_lr": 1e-3}
    ccfg = {"amp": False, "weight_decay": 0.01}
    result = fit(
        model, train_loader, val_loader, mcfg, ccfg, torch.device("cpu"), tmp_path, freeze_epochs=1
    )
    assert result["history"][0]["frozen_backbone"]
    assert torch.equal(model.conv1.weight, conv_before)
    assert torch.equal(model.bn1.running_mean, bn_before)
    assert not torch.equal(model.fc.weight, head_before)


def test_select_device_cpu():
    assert select_device("cpu").type == "cpu"


def test_cli_end_to_end_on_mock(mock_splits, tmp_path, capsys):
    root, splits_dir = mock_splits
    cfg = copy.deepcopy(CFG)
    cfg["rscd"]["image_size"] = list(SMALL)
    cfg["classifier"].update(batch_size=16, num_workers=0, amp=False)
    cfg["classifier"]["models"]["small_cnn"].update(epochs=2, patience=2)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"

    main(
        [
            *("--model", "small_cnn", "--config", str(config_path), "--root", str(root)),
            *("--splits-dir", str(splits_dir), "--out-dir", str(out), "--seeds", "0", "1"),
            *("--eval-test", "--gradcam", "2", "--device", "cpu"),
        ]
    )
    results = json.loads((out / "results" / "small_cnn.json").read_text())
    assert results["seeds"] == [0, 1] and results["eval_test"]
    runs = results["models"]["small_cnn"]
    assert len(runs) == 2
    for run in runs:
        assert 0.0 <= run["test"]["macro_f1"] <= 1.0
        assert set(run["test"]["per_class"]) == set(CLASSES)
        cal = run["test"]["calibration"]
        assert cal["temperature"] > 0 and run["temperature"] == cal["temperature"]
        assert 0.0 <= cal["ece_after"] <= 1.0
        assert run["gradcam"]
    run_dir = out / "checkpoints" / "small_cnn_seed0"
    for name in (
        "best.pt",
        "last.pt",
        "history.json",
        "loss_curves.png",
        "reliability_test.png",
        "gradcam_test.png",
    ):
        assert (run_dir / name).is_file(), name
    assert "small_cnn" in capsys.readouterr().out


# --- dataset --------------------------------------------------------------------------------


def test_transforms_resize_and_jitter_options():
    train = build_transforms(True, (240, 360), strength=1.0)
    off = build_transforms(True, (240, 360), strength=0)
    evaluation = build_transforms(False, (240, 360))
    assert any(isinstance(t, v2.ColorJitter) for t in train.transforms)
    assert not any(isinstance(t, v2.ColorJitter) for t in off.transforms)
    assert not any(isinstance(t, v2.RandomHorizontalFlip) for t in evaluation.transforms)


def test_road_dataset_items_and_loader(mock_splits):
    root, splits_dir = mock_splits
    splits = load_splits(splits_dir)
    dataset = RoadDataset(splits["train"], root, CLASSES, build_transforms(False, SMALL))
    image, target = dataset[0]
    assert image.shape == (3, *SMALL) and image.dtype == torch.float32
    assert target == CLASSES.index(splits["train"][0].friction)
    assert dataset.load_raw(0, SMALL).shape == (*SMALL, 3)
    x, y = next(iter(make_loader(dataset, 8)))
    assert x.shape == (8, 3, *SMALL) and y.shape == (8,)


def test_road_dataset_rejects_unknown_class(mock_splits):
    root, splits_dir = mock_splits
    with pytest.raises(ValueError, match="not in"):
        RoadDataset(load_splits(splits_dir)["train"], root, ["dry"])


def test_eval_transform_is_deterministic_and_train_is_seeded(mock_splits):
    root, splits_dir = mock_splits
    records = load_splits(splits_dir)["train"]
    evaluation = RoadDataset(records, root, CLASSES, build_transforms(False, SMALL))
    assert torch.equal(evaluation[3][0], evaluation[3][0])
    training = RoadDataset(records, root, CLASSES, build_transforms(True, SMALL))
    torch.manual_seed(1)
    first = training[3][0]
    torch.manual_seed(1)
    assert torch.equal(first, training[3][0])


# --- calibration ----------------------------------------------------------------------------


def one_hot_probs(n_classes, predicted, confidence):
    probs = np.full((len(predicted), n_classes), (1 - confidence) / (n_classes - 1))
    probs[np.arange(len(predicted)), predicted] = confidence
    return probs


def test_ece_is_zero_for_perfect_confident_predictions():
    labels = np.array([0, 1, 2, 3, 4, 5])
    assert expected_calibration_error(one_hot_probs(6, labels, 1.0), labels) == pytest.approx(0.0)


def test_ece_hand_example():
    # Confidence 0.9 everywhere but only half the predictions are right: ECE = |0.5 - 0.9|.
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probs = one_hot_probs(6, np.zeros(10, dtype=int), 0.9)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.4)


def test_reliability_bins_count_every_sample():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(6), size=200)
    bins = reliability_bins(probs, rng.integers(0, 6, 200), n_bins=10)
    assert sum(bins["count"]) == 200
    assert len(bins["edges"]) == 11 and len(bins["accuracy"]) == 10


def test_temperature_scaling_fixes_overconfident_logits():
    rng = np.random.default_rng(0)
    n, k = 4000, 6
    labels = rng.integers(0, k, n)
    wrong = (labels + rng.integers(1, k, n)) % k
    predicted = np.where(rng.random(n) < 0.7, labels, wrong)  # 70% accurate...
    logits = rng.normal(0, 0.5, (n, k))
    logits[np.arange(n), predicted] += 6.0  # ...but extremely confident
    logits, labels_t = torch.tensor(logits, dtype=torch.float32), torch.tensor(labels)

    scaler = TemperatureScaler()
    assert scaler.temperature == pytest.approx(1.0)
    assert torch.equal(scaler(logits), logits)
    scaler.fit(logits, labels_t)
    assert scaler.temperature > 1.5
    scaled = scaler(logits).detach()
    assert nll(scaled, labels_t) < nll(logits, labels_t)
    assert torch.equal(scaled.argmax(1), logits.argmax(1))
    ece_before = expected_calibration_error(F.softmax(logits, 1).numpy(), labels)
    ece_after = expected_calibration_error(F.softmax(scaled, 1).numpy(), labels)
    assert ece_after < ece_before / 2


def test_temperature_scaler_validates_shapes():
    with pytest.raises(ValueError):
        TemperatureScaler().fit(torch.zeros(4, 6), torch.zeros(3, dtype=torch.long))


# --- Grad-CAM -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_gradcam_shape_range_and_determinism(name):
    torch.manual_seed(0)
    model = build_model(name, CFG, pretrained=False)
    x = torch.randn(2, 3, 240, 360)
    cam = gradcam(model, x)
    assert cam.shape == (2, 240, 360)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(gradcam(model, x), cam, atol=1e-6)


def test_gradcam_explains_the_requested_class():
    torch.manual_seed(0)
    model = SmallCNN(6)
    x = torch.randn(1, 3, 96, 144)
    assert gradcam(model, x, class_ids=[0]).shape == (1, 96, 144)


def test_gradcam_works_with_frozen_parameters():
    model = build_pretrained("resnet50", 6, pretrained=False)
    for p in model.parameters():
        p.requires_grad = False
    assert gradcam(model, torch.randn(1, 3, 96, 144)).shape == (1, 96, 144)


def test_overlay_returns_uint8_image_and_checks_sizes():
    raw = np.random.default_rng(0).integers(0, 256, (240, 360, 3), dtype=np.uint8)
    blended = overlay(raw, np.random.default_rng(1).random((240, 360)).astype(np.float32))
    assert blended.shape == raw.shape and blended.dtype == np.uint8
    with pytest.raises(ValueError):
        overlay(raw, np.zeros((10, 10), dtype=np.float32))


def test_select_examples_takes_most_confident_mistakes():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.55, 0.45]])
    labels = np.array([0, 1, 1, 1, 1])  # samples 0 and 2 are correct; 1, 3 and 4 are wrong
    chosen = select_examples(probs, labels, n_each=2)
    assert chosen["wrong"] == [1, 3]  # confidences 0.8 and 0.6, most confident first
    assert set(chosen["correct"]) == {0, 2}
