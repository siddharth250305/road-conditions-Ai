"""ONNX export (PRD W3): a classifier checkpoint becomes a small serving bundle.

The bundle is a folder the demo API reads and nothing else (PRD: research and serving meet
only at the exported files):

    model.onnx   raw RGB in [0, 1] -> logits and per-class activation maps
    meta.json    classes, input size, temperature, calibration, training provenance
    study.json   the cross-country benchmark for the demo's study view (optional)

The graph includes the mean/std normalisation, so the server only resizes the picture. It has a
second output, `cam`: for a network that ends in global average pooling and one linear layer
(SmallCNN, ResNet-50, EfficientNet-B0) Grad-CAM at the last feature map is exactly the class
activation map sum_k W[c, k] * A_k, so a heatmap needs no gradients at serving time.

    python -m roadcond.export bundle --checkpoint best.pt --results resnet50.json --out serving
    python -m roadcond.export study --results-dir artifacts/results --out serving/study.json
"""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from roadcond.config import load_config, resolve_path
from roadcond.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from roadcond.detection.metrics import gap_recovered, learning_curve, summarize
from roadcond.train import MODEL_NAMES, build_model

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
INPUT_NAME = "image"
OUTPUT_NAMES = ("logits", "cam")
MAX_EXPORT_BATCH = 64


class ServingModel(nn.Module):
    """Raw RGB in [0, 1] (N, 3, H, W) -> (logits (N, C), cam (N, C, h, w)).

    cam holds, for every class, the class activation map at the layer that feeds global
    pooling. The classifier bias is left out on purpose: Grad-CAM ignores it too.
    """

    def __init__(self, model: nn.Module, name: str):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model {name!r}; use one of {MODEL_NAMES}")
        self.model, self.name = model, name
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        head = model.classifier if name == "small_cnn" else model.get_classifier()
        if not isinstance(head, nn.Linear):
            raise ValueError(f"{name} does not end in one linear layer: no class activation map")
        self.register_buffer("cam_weight", head.weight.detach().clone()[:, :, None, None])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.mean) / self.std
        if self.name == "small_cnn":
            features = self.model.features(x)
            logits = self.model.classifier(features.mean(dim=(2, 3)))
        else:
            features = self.model.forward_features(x)
            logits = self.model.forward_head(features)
        return logits, F.conv2d(features, self.cam_weight)


def feature_layer(model: nn.Module, name: str) -> nn.Module:
    """The layer whose output feeds global pooling: where Grad-CAM equals the served heatmap."""
    if name == "small_cnn":
        return model.features[-1]
    if name == "resnet50":
        return model.layer4
    if name == "efficientnet":
        return model.bn2
    raise ValueError(f"Unknown model {name!r}; use one of {MODEL_NAMES}")


def export_onnx(
    wrapper: ServingModel,
    path: str | Path,
    input_size: Sequence[int] = (240, 360),
    opset: int | None = None,
) -> Path:
    """Write one self-contained .onnx file with a dynamic batch dimension and a fixed H x W."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = input_size
    # A batch of 2 keeps torch.export from specialising the batch axis to 1.
    example = torch.rand(2, 3, height, width)
    kwargs: dict[str, Any] = {"opset_version": opset} if opset else {}
    batch = torch.export.Dim("batch", min=1, max=MAX_EXPORT_BATCH)
    with contextlib.redirect_stdout(io.StringIO()):  # the exporter prints progress marks
        torch.onnx.export(
            wrapper.eval(),
            (example,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_shapes={"x": {0: batch}},
            dynamo=True,
            external_data=False,  # the weights must live inside the one file
            **kwargs,
        )
    return path


def validate_onnx(
    wrapper: ServingModel,
    path: str | Path,
    input_size: Sequence[int] = (240, 360),
    tolerance: float = 1e-4,
    batches: Sequence[int] = (1, 3),
    seed: int = 0,
) -> dict[str, Any]:
    """ONNX Runtime must reproduce PyTorch (logits and maps) within tolerance, per batch size."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(seed)
    worst = dict.fromkeys(OUTPUT_NAMES, 0.0)
    cam_shape: list[int] = []
    wrapper.eval()
    for batch in batches:
        x = torch.rand(batch, 3, *input_size, generator=generator)
        with torch.no_grad():
            reference = wrapper(x)
        outputs = session.run(None, {INPUT_NAME: x.numpy()})
        for name, expected, got in zip(OUTPUT_NAMES, reference, outputs, strict=True):
            expected = expected.numpy()
            if got.shape != expected.shape:
                raise ValueError(f"ONNX {name} has shape {got.shape}, PyTorch {expected.shape}")
            worst[name] = max(worst[name], float(np.abs(got - expected).max()))
            if not np.allclose(got, expected, rtol=tolerance, atol=tolerance):
                raise ValueError(
                    f"ONNX Runtime differs from PyTorch on {name} by {worst[name]:.2e} "
                    f"(tolerance {tolerance:.0e}, batch {batch})"
                )
        cam_shape = list(outputs[1].shape[1:])
    return {
        "max_abs_diff": worst,
        "tolerance": tolerance,
        "batches": list(batches),
        "cam_shape": cam_shape,
    }


def load_checkpoint(path: str | Path, cfg: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model a Phase 2 checkpoint (best.pt) was saved from: (model, checkpoint)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "model_name", "classes"):
        if key not in state:
            raise ValueError(f"{path} is not a roadcond checkpoint: missing {key!r}")
    n_config = len(cfg["rscd"]["classes"])
    if len(state["classes"]) != n_config:
        raise ValueError(f"Checkpoint has {len(state['classes'])} classes, config has {n_config}")
    model = build_model(state["model_name"], cfg, pretrained=False)
    model.load_state_dict(state["model"])
    return model.eval(), state


def pick_run(
    runs: Sequence[dict[str, Any]], checkpoint: dict[str, Any], seed: int | None
) -> dict[str, Any]:
    """The results entry that belongs to a checkpoint: by seed, else by its validation macro-F1."""
    if seed is not None:
        chosen = [r for r in runs if r["seed"] == seed]
    else:
        best = checkpoint.get("val_macro_f1", -1.0)
        chosen = [r for r in runs if abs(r["val"]["macro_f1"] - best) < 1e-6]
        if not chosen and len(runs) == 1:
            chosen = list(runs)
    if len(chosen) != 1:
        raise ValueError(
            f"Could not tell which of the {len(runs)} runs in the results file belongs to this "
            "checkpoint; pass --seed (or --temperature)"
        )
    return chosen[0]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_meta(run: dict[str, Any] | None, temperature: float | None) -> dict[str, Any]:
    """Temperature and the ECE it achieved, from a Phase 2 run (test split when it was scored)."""
    if temperature is None and run is None:
        raise ValueError("Need --results or --temperature: serving without calibration is refused")
    cal: dict[str, Any] = {
        "method": "temperature scaling fitted on validation logits",
        "temperature": float(temperature if temperature is not None else run["temperature"]),
        "split": None,
        "ece_before": None,
        "ece_after": None,
    }
    if run is not None:
        split = "test" if "test" in run else "val"
        info = run[split]["calibration"]
        cal.update(split=split, ece_before=info["ece_before"], ece_after=info["ece_after"])
    if cal["temperature"] <= 0:
        raise ValueError(f"Temperature must be positive, got {cal['temperature']}")
    return cal


def build_bundle(
    checkpoint: str | Path,
    out_dir: str | Path,
    cfg: dict[str, Any] | None = None,
    results: str | Path | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    threshold: float | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Export a checkpoint to out_dir/model.onnx, check it against PyTorch, write meta.json."""
    cfg = cfg or load_config()
    scfg = cfg["serving"]
    model, state = load_checkpoint(checkpoint, cfg)
    name, classes = state["model_name"], list(state["classes"])

    run = None
    if results is not None:
        data = json.loads(Path(results).read_text())
        if name not in data["models"]:
            raise ValueError(f"{results} has no results for {name!r}: {sorted(data['models'])}")
        run = pick_run(data["models"][name], state, seed)
    calibration = calibration_meta(run, temperature)

    size = tuple(cfg["rscd"]["image_size"])
    wrapper = ServingModel(model, name).eval()
    out_dir = Path(out_dir)
    onnx_path = export_onnx(wrapper, out_dir / "model.onnx", size)
    validation = validate_onnx(wrapper, onnx_path, size, scfg["onnx_tolerance"])
    import onnx
    import onnxruntime

    meta = {
        "format_version": FORMAT_VERSION,
        "mock": bool(mock),
        "model_name": name,
        "classes": classes,
        "input": {
            "name": INPUT_NAME,
            "height": size[0],
            "width": size[1],
            "channels": 3,
            "layout": "NCHW",
            "dtype": "float32",
            "colour": "RGB",
            "range": [0.0, 1.0],
        },
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "embedded_in_graph": True,
        },
        "outputs": {"logits": "logits", "cam": "cam", "cam_shape": validation["cam_shape"]},
        "calibration": calibration,
        "serving": {
            "uncertainty_threshold": float(
                threshold if threshold is not None else scfg["uncertainty_threshold"]
            ),
        },
        "training": {
            "dataset": "RSCD",
            "labels": "friction, 6 classes",
            "subset_size_config": cfg["rscd"]["subset"]["train_size"],
            "best_epoch": state.get("epoch"),
            "val_macro_f1": state.get("val_macro_f1"),
            "test_macro_f1": run["test"]["macro_f1"] if run and "test" in run else None,
            "seed": run["seed"] if run else None,
        },
        "provenance": {
            "checkpoint_sha256": sha256(checkpoint),
            "onnx_sha256": sha256(onnx_path),
            "onnx_validation": validation,
            "exported_with": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": onnxruntime.__version__,
            },
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# --- the study bundle -----------------------------------------------------------------------


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _recovered(source: dict[str, Any], zero_shot: dict[str, Any], method: float) -> float | None:
    return gap_recovered(source["map50"], zero_shot["map50"], method)


def build_study(
    results_dir: str | Path, out_path: str | Path, mock: bool = False
) -> dict[str, Any]:
    """Compact cross-country benchmark for the demo, built only from saved Phase 3 results."""
    results_dir = Path(results_dir)
    base = _read(results_dir / "detection_source.json")
    if base is None:
        raise FileNotFoundError(f"No detection_source.json in {results_dir}: run the 'source' step")
    aug = _read(results_dir / "detection_source_aug.json")
    pseudo = _read(results_dir / "detection_pseudo.json")
    runs = [json.loads(p.read_text()) for p in sorted((results_dir / "fewshot").glob("*.json"))]

    source = base["source_country"]
    src = base["countries"][source]
    targets = [c for c in base["countries"] if c != source]
    zero = {c: base["countries"][c] for c in targets}
    curve = learning_curve(runs, zero, src) if runs else {}

    slope, recovery = [], {}
    for country in targets:
        gap = base["gaps"][country]
        slope.append(
            {
                "country": country,
                "source_map50": gap["map50"]["source"],
                "target_map50": gap["map50"]["target"],
                "relative_gap_map50": gap["map50"]["relative_gap"],
                "source_map50_95": gap["map50_95"]["source"],
                "target_map50_95": gap["map50_95"]["target"],
                "relative_gap_map50_95": gap["map50_95"]["relative_gap"],
            }
        )
        entry: dict[str, Any] = {"zero_shot_map50": zero[country]["map50"]}
        if aug is not None:
            a = aug["countries"][country]["map50"]
            entry["augmentation"] = {"map50": a, "recovered": _recovered(src, zero[country], a)}
        if pseudo is not None and country in pseudo["countries"]:
            p = pseudo["countries"][country]
            recovered = summarize([_recovered(src, zero[country], r["map50"]) for r in p["runs"]])
            entry["pseudo_labels"] = {
                "threshold": p["threshold"],
                "map50": p["map50"],
                "recovered": recovered,
            }
        recovery[country] = entry

    study = {
        "format_version": FORMAT_VERSION,
        "mock": bool(mock),
        "metric": "map50",
        "f1_conf": base["f1_conf"],
        "source_country": source,
        "source": {k: src[k] for k in ("map50", "map50_95", "f1")},
        "slope": slope,
        "learning_curve": {
            country: [
                {
                    "labels": labels,
                    "map50": cell["map50"],
                    "map50_95": cell["map50_95"],
                    "f1": cell["f1"],
                    "relative_gap": cell["relative_gap"],
                    "recovered": cell["recovered"],
                }
                for labels, cell in cells.items()
            ]
            for country, cells in curve.items()
        },
        "recovery": recovery,
        "notes": [
            "Scores are on our own held-out split of each country's labelled images.",
            "Spread is over seeds for label budgets above 0; the source detector is one run.",
            "A gap that is zero or negative means the target scored as well as the source, "
            "and nothing is recovered.",
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n")
    return study


# --- command line ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export classifier checkpoints and study data.")
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bundle", help="checkpoint -> model.onnx + meta.json")
    p.add_argument("--checkpoint", required=True, help="best.pt from scripts/train_classifier.py")
    p.add_argument("--results", help="results/<model>.json written next to it (for T and ECE)")
    p.add_argument("--out", help="bundle folder (default: serving.bundle_dir)")
    p.add_argument(
        "--seed", type=int, help="which run of the results file (default: match best.pt)"
    )
    p.add_argument("--temperature", type=float, help="override the temperature from the results")
    p.add_argument("--threshold", type=float, help="'Not sure' below this calibrated probability")
    p.add_argument("--mock", action="store_true", help="mark the bundle as built from mock data")
    p = sub.add_parser("study", help="Phase 3 results -> study.json")
    p.add_argument("--results-dir", required=True, help="folder with detection_source.json etc.")
    p.add_argument("--out", help="default: <serving.bundle_dir>/study.json")
    p.add_argument("--mock", action="store_true", help="mark the study as built from mock data")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("roadcond").setLevel(logging.INFO)  # the exporter libraries are chatty

    cfg = load_config(args.config)
    cwd = Path.cwd()
    default_dir = resolve_path(cfg["serving"]["bundle_dir"])
    if args.command == "bundle":
        out = resolve_path(args.out, cwd) if args.out else default_dir
        results = resolve_path(args.results, cwd) if args.results else None
        meta = build_bundle(
            resolve_path(args.checkpoint, cwd), out, cfg, results, args.seed,
            args.temperature, args.threshold, args.mock,
        )  # fmt: skip
        cal, val = meta["calibration"], meta["provenance"]["onnx_validation"]
        print(f"Exported {meta['model_name']} to {out / 'model.onnx'}")
        print(
            f"  ONNX Runtime vs PyTorch: max |diff| {val['max_abs_diff']}, "
            f"tolerance {val['tolerance']}"
        )
        print(
            f"  temperature {cal['temperature']:.3f}; "
            f"ECE {cal['ece_before']} -> {cal['ece_after']} ({cal['split']})"
        )
    else:
        out = resolve_path(args.out, cwd) if args.out else default_dir / "study.json"
        study = build_study(resolve_path(args.results_dir, cwd), out, args.mock)
        print(f"Wrote {out}: {len(study['slope'])} target countries, mock={study['mock']}")


if __name__ == "__main__":
    main()
