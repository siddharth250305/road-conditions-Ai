"""Build a demo serving bundle from random weights, to try the web app without any training.

python scripts/make_mock_bundle.py --out serving            # confident: always answers "wet"
python scripts/make_mock_bundle.py --out serving --unsure   # near-uniform: always "Not sure"

The bundle is marked as built from mock data, and the page says so on every screen. Its answers
mean nothing. A real model comes from scripts/export_model.py after training on Colab.
Needs: pip install -e ".[export]"
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CLASSES = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]


def synthetic_detection_results(folder: Path) -> None:
    """Result files in the shape the Phase 3 scripts write, with invented numbers."""

    def scores(v: float) -> dict[str, float]:
        return {"map50": v, "map50_95": v / 2, "f1": v * 0.9}

    countries = {"Japan": 0.80, "Norway": 0.45, "Czech": 0.62, "United_States": 0.38}
    gaps = {
        c: {
            m: {
                "source": scores(0.8)[m],
                "target": scores(v)[m],
                "relative_gap": (scores(0.8)[m] - scores(v)[m]) / scores(0.8)[m],
            }
            for m in ("map50", "map50_95")
        }
        for c, v in countries.items()
        if c != "Japan"
    }
    (folder / "fewshot").mkdir(parents=True, exist_ok=True)
    source = {
        "run": "source",
        "source_country": "Japan",
        "f1_conf": 0.25,
        "countries": {c: scores(v) for c, v in countries.items()},
        "gaps": gaps,
    }
    (folder / "detection_source.json").write_text(json.dumps(source))
    for country in ("Norway", "Czech", "United_States"):
        for budget in (10, 50, 200):
            for seed in (0, 1, 2):
                value = countries[country] + 0.0008 * budget + 0.01 * seed
                run = {"country": country, "budget": budget, "seed": seed, **scores(value)}
                (folder / "fewshot" / f"{country}_n{budget}_s{seed}.json").write_text(
                    json.dumps(run)
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="serving", help="bundle folder (default: serving)")
    parser.add_argument("--unsure", action="store_true", help="near-uniform answers: 'Not sure'")
    args = parser.parse_args()

    import torch

    from roadcond.config import load_config
    from roadcond.export import build_bundle, build_study
    from roadcond.train import build_model

    cfg = load_config()
    torch.manual_seed(0)
    model = build_model("small_cnn", cfg, pretrained=False)
    if not args.unsure:
        model.classifier.bias.data = torch.tensor([0.0, 6.0, 0, 0, 0, 0])  # always "wet"

    calibration = {"ece_before": 0.25, "ece_after": 0.04}
    run = {
        "seed": 0,
        "temperature": 1.7,
        "val": {"macro_f1": 0.5, "calibration": calibration},
        "test": {"macro_f1": 0.45, "calibration": calibration},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        checkpoint = tmp_dir / "best.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": 3,
                "val_macro_f1": 0.5,
                "model_name": "small_cnn",
                "classes": CLASSES,
            },
            checkpoint,
        )
        results = tmp_dir / "results.json"
        results.write_text(json.dumps({"classes": CLASSES, "models": {"small_cnn": [run]}}))
        out = Path(args.out)
        build_bundle(checkpoint, out, cfg, results, mock=True)
        detection = tmp_dir / "detection"
        synthetic_detection_results(detection)
        build_study(detection, out / "study.json", mock=True)
    print(f"Mock bundle written to {out} (small_cnn, random weights, marked as mock).")


if __name__ == "__main__":
    main()
