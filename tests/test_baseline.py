import copy
import json

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


def test_save_probs_is_opt_in_and_leaves_results_unchanged(tmp_path):
    from roadcond.models.baseline import main
    from roadcond.routing.probs import load_probs

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
    cache = tmp_path / "cache"
    plain = run_baseline(cfg, splits, root, cache_dir=cache, eval_test=True)
    probs_path = tmp_path / "probs.npz"
    with_probs = run_baseline(
        cfg, splits, root, cache_dir=cache, eval_test=True, save_probs=probs_path
    )
    # baseline.json is written as json.dumps(results, indent=2): compare the serialised bytes.
    assert json.dumps(with_probs, indent=2) == json.dumps(plain, indent=2)

    saved = load_probs(probs_path)
    assert saved.classes == cfg["rscd"]["classes"]
    assert saved.seed == 0
    assert saved.models == ["logreg", "random_forest"]
    for split in ("val", "test"):
        sp = saved.splits[split]
        assert list(sp.image_ids) == [r.path for r in splits[split]]
        assert [saved.classes[i] for i in sp.labels] == [r.friction for r in splits[split]]
        for p in sp.probs.values():
            np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)  # float32 features
    # The saved probabilities reproduce the reported first-seed test predictions.
    rf_test = plain["models"]["random_forest"][0]["test"]
    pred = saved.splits["test"].probs["random_forest"].argmax(axis=1)
    assert np.mean(pred == saved.splits["test"].labels) == pytest.approx(rf_test["accuracy"])

    with pytest.raises(ValueError, match="eval_test"):
        run_baseline(cfg, splits, root, cache_dir=cache, save_probs=probs_path)

    # Command line: off by default, and refused without --eval-test.
    config = tmp_path / "cfg.yaml"
    import yaml

    config.write_text(yaml.safe_dump(cfg))
    common = ["--config", str(config), "--root", str(root), "--cache-dir", str(cache)]
    common += ["--splits-dir", str(tmp_path / "splits"), "--workers", "1"]
    default_probs = tmp_path / "out" / "baseline_probs.npz"
    main([*common, "--eval-test", "--results", str(tmp_path / "out" / "baseline.json")])
    assert not default_probs.exists()
    with pytest.raises(SystemExit):
        main([*common, "--save-probs", "--probs-path", str(default_probs)])
