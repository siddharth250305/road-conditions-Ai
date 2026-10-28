"""Shared fixtures for the export and API tests: serving bundles without any training."""

import json
from pathlib import Path

import pytest
from tests.helpers import CLASSES, TEMPERATURE


@pytest.fixture(scope="session")
def bundle_factory(tmp_path_factory):
    """build(model_name, confident) -> a serving bundle folder made from random weights.

    confident=True gives the classifier a large bias towards class 0 ("dry"), so predictions
    are confident and deterministic; otherwise the probabilities are close to uniform.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    cache: dict = {}

    def build(model_name: str = "small_cnn", confident: bool = False) -> Path:
        key = (model_name, confident)
        if key in cache:
            return cache[key]
        import torch

        from roadcond.config import load_config
        from roadcond.export import build_bundle
        from roadcond.train import build_model

        cfg = load_config()
        torch.manual_seed(0)
        model = build_model(model_name, cfg, pretrained=False)
        if confident:
            head = model.classifier if model_name == "small_cnn" else model.get_classifier()
            head.bias.data = torch.tensor([6.0, 0, 0, 0, 0, 0])
        base = tmp_path_factory.mktemp(f"bundle_{model_name}_{int(confident)}")
        checkpoint = base / "best.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": 3,
                "val_macro_f1": 0.5,
                "model_name": model_name,
                "classes": CLASSES,
            },
            checkpoint,
        )
        cal = {"ece_before": 0.25, "ece_after": 0.04}
        results = {
            "classes": CLASSES,
            "models": {
                model_name: [
                    {
                        "seed": 0,
                        "temperature": TEMPERATURE,
                        "val": {"macro_f1": 0.5, "calibration": cal},
                        "test": {"macro_f1": 0.45, "calibration": cal},
                    }
                ]
            },
        }
        (base / "results.json").write_text(json.dumps(results))
        build_bundle(checkpoint, base / "serving", cfg, base / "results.json", mock=True)
        cache[key] = base / "serving"
        return cache[key]

    return build


def _scores(map50: float) -> dict:
    return {"map50": map50, "map50_95": map50 / 2, "f1": map50 * 0.9}


@pytest.fixture(scope="session")
def study_results(tmp_path_factory):
    """Synthetic Phase 3 result files (the shape the detection CLI writes), no detector needed."""
    root = tmp_path_factory.mktemp("detection_results")
    countries = {"Japan": 0.80, "Norway": 0.60, "Czech": 0.70, "United_States": 0.85}
    gaps = {}
    for country, value in countries.items():
        if country == "Japan":
            continue
        gaps[country] = {
            m: {
                "source": _scores(0.80)[m],
                "target": _scores(value)[m],
                "relative_gap": (_scores(0.80)[m] - _scores(value)[m]) / _scores(0.80)[m],
            }
            for m in ("map50", "map50_95")
        }
    source = {
        "run": "source",
        "source_country": "Japan",
        "f1_conf": 0.25,
        "countries": {c: _scores(v) for c, v in countries.items()},
        "gaps": gaps,
    }
    (root / "detection_source.json").write_text(json.dumps(source))
    aug = {"countries": {c: _scores(v + 0.02) for c, v in countries.items()}}
    (root / "detection_source_aug.json").write_text(json.dumps(aug))
    (root / "fewshot").mkdir()
    for country in ("Norway", "Czech", "United_States"):
        for budget in (10, 50):
            for seed in (0, 1, 2):
                value = countries[country] + 0.001 * budget + 0.01 * seed
                run = {"country": country, "budget": budget, "seed": seed, **_scores(value)}
                (root / "fewshot" / f"{country}_n{budget}_s{seed}.json").write_text(json.dumps(run))
    pseudo = {
        "countries": {
            c: {
                "threshold": 0.5,
                "map50": {"mean": countries[c] + 0.03, "std": 0.01, "n": 2},
                "runs": [{"map50": countries[c] + 0.02}, {"map50": countries[c] + 0.04}],
            }
            for c in ("Norway", "Czech", "United_States")
        }
    }
    (root / "detection_pseudo.json").write_text(json.dumps(pseudo))
    return root
