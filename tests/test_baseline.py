import copy

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skimage")

from roadcond.config import load_config  # noqa: E402
from roadcond.data.mock import make_mock_rscd  # noqa: E402
from roadcond.data.rscd import load_splits, prepare  # noqa: E402
from roadcond.models.baseline import extract_features, run_baseline, summary_tables  # noqa: E402

CFG = load_config()


def test_extract_features_length_and_values():
    bcfg = CFG["baseline"]
    height, width = bcfg["image_size"]
    image = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    features = extract_features(image, bcfg["color_bins"], bcfg["hog"])

    hog = bcfg["hog"]
    cells = (height // hog["pixels_per_cell"][0], width // hog["pixels_per_cell"][1])
    blocks = [c - b + 1 for c, b in zip(cells, hog["cells_per_block"], strict=True)]
    hog_len = blocks[0] * blocks[1] * np.prod(hog["cells_per_block"]) * hog["orientations"]
    assert features.shape == (6 * bcfg["color_bins"] + hog_len,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    # Each of the 6 colour histograms sums to 1.
    color = features[: 6 * bcfg["color_bins"]].reshape(6, -1)
    np.testing.assert_allclose(color.sum(axis=1), 1.0, rtol=1e-5)


def test_run_baseline_end_to_end_on_mock(tmp_path):
    root = tmp_path / "rscd"
    make_mock_rscd(root, CFG["rscd"]["split_dirs"], n_per_label=(6, 2, 2))
    prepare(CFG, root, tmp_path / "splits", train_size=36)

    cfg = copy.deepcopy(CFG)
    cfg["project"]["seeds"] = [0, 1]
    cfg["baseline"]["image_size"] = [48, 72]
    cfg["baseline"]["models"] = {
        "logreg": {"max_iter": 200, "grid": {"C": [1.0]}},
        "random_forest": {"n_estimators": 10, "grid": {"max_depth": [None, 5]}},
    }
    splits = load_splits(tmp_path / "splits")
    results = run_baseline(cfg, splits, root, cache_dir=tmp_path / "cache", eval_test=True)

    assert set(results["models"]) == {"logreg", "random_forest"}
    for runs in results["models"].values():
        assert [r["seed"] for r in runs] == [0, 1]
        for run in runs:
            assert 0.0 <= run["test"]["macro_f1"] <= 1.0
            assert set(run["test"]["per_class"]) == set(cfg["rscd"]["classes"])
    assert "random_forest" in summary_tables(results)
    assert (tmp_path / "cache" / "rscd_features_test.npz").is_file()

    # Second run reuses the cache and gives identical results.
    again = run_baseline(cfg, splits, root, cache_dir=tmp_path / "cache", eval_test=True)
    assert again["models"]["logreg"][0]["test"] == results["models"]["logreg"][0]["test"]
