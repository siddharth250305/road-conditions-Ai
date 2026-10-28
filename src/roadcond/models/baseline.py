"""Classical baseline (PRD C2): colour histograms + HOG into logistic regression and random forest.

Hyperparameters are chosen on validation macro-F1. The test split is scored only with
--eval-test, once per final model and seed (PRD reporting rules).

    python -m roadcond.models.baseline --root /path/to/RSCD            # tune on val
    python -m roadcond.models.baseline --root /path/to/RSCD --eval-test
"""

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from roadcond.config import load_config, resolve_path
from roadcond.data.rscd import Record, load_splits
from roadcond.metrics import classification_report, format_table, mean_std

log = logging.getLogger(__name__)

FEATURE_KEYS = ("image_size", "color_bins", "hog")


def load_image(path: str | Path, size: Sequence[int]) -> np.ndarray:
    """RGB uint8 array resized to size = (height, width)."""
    height, width = size
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))


def extract_features(image: np.ndarray, color_bins: int, hog_cfg: dict[str, Any]) -> np.ndarray:
    """RGB and HSV histograms (fraction of pixels per bin), then HOG of the grey image."""
    hists = [np.histogram(image[..., c], bins=color_bins, range=(0, 256))[0] for c in range(3)]
    hsv = rgb2hsv(image)
    hists += [np.histogram(hsv[..., c], bins=color_bins, range=(0.0, 1.0))[0] for c in range(3)]
    color = np.concatenate(hists).astype(np.float32) / (image.shape[0] * image.shape[1])
    shape = hog(
        rgb2gray(image),
        orientations=hog_cfg["orientations"],
        pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
        cells_per_block=tuple(hog_cfg["cells_per_block"]),
        feature_vector=True,
    )
    return np.concatenate([color, shape.astype(np.float32)])


def _features_for_path(rel_path: str, root: str, fcfg: dict[str, Any]) -> np.ndarray:
    image = load_image(Path(root) / rel_path, fcfg["image_size"])
    return extract_features(image, fcfg["color_bins"], fcfg["hog"])


def _cache_key(records: Sequence[Record], root: Path, fcfg: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps({"root": str(root), **fcfg}, sort_keys=True).encode())
    for r in records:
        digest.update(r.path.encode() + b"\0")
    return digest.hexdigest()


def build_features(
    records: Sequence[Record],
    root: str | Path,
    fcfg: dict[str, Any],
    cache_path: Path | None = None,
    workers: int = 1,
) -> np.ndarray:
    """Feature matrix for records, reusing cache_path when files and settings are unchanged."""
    root = Path(root)
    key = _cache_key(records, root, fcfg)
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path) as cached:
            if str(cached["key"]) == key:
                log.info("Loaded cached features %s", cache_path)
                return cached["X"]
        log.info("Feature cache %s is stale; recomputing", cache_path)
    fn = partial(_features_for_path, root=str(root), fcfg=fcfg)
    paths = [r.path for r in records]
    log.info("Extracting features for %d images", len(paths))
    if workers <= 1:
        features = [fn(p) for p in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            features = list(pool.map(fn, paths, chunksize=64))
    X = np.stack(features)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, key=np.array(key))
    return X


def make_model(name: str, params: dict[str, Any], seed: int) -> ClassifierMixin:
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **params))
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    raise ValueError(f"Unknown baseline model {name!r}; use logreg or random_forest")


def run_baseline(
    cfg: dict[str, Any],
    splits: dict[str, list[Record]],
    root: str | Path,
    cache_dir: Path | None = None,
    workers: int = 1,
    eval_test: bool = False,
) -> dict[str, Any]:
    """Tune each model on val per seed; optionally score the chosen model on test once."""
    classes = cfg["rscd"]["classes"]
    bcfg = cfg["baseline"]
    fcfg = {k: bcfg[k] for k in FEATURE_KEYS}
    used = ("train", "val", "test") if eval_test else ("train", "val")
    X, y = {}, {}
    for split in used:
        cache = cache_dir / f"rscd_features_{split}.npz" if cache_dir else None
        X[split] = build_features(splits[split], root, fcfg, cache, workers)
        y[split] = np.array([classes.index(r.friction) for r in splits[split]])

    seeds = cfg["project"]["seeds"]
    results: dict[str, Any] = {"classes": classes, "seeds": seeds, "eval_test": eval_test}
    results["models"] = {}
    for name, mcfg in bcfg["models"].items():
        fixed = {k: v for k, v in mcfg.items() if k != "grid"}
        runs = []
        for seed in seeds:
            best = None
            for params in ParameterGrid(mcfg.get("grid", {})):
                model = make_model(name, {**fixed, **params}, seed).fit(X["train"], y["train"])
                val = classification_report(y["val"], model.predict(X["val"]), classes)
                log.info("%s seed=%d %s: val macro-F1 %.4f", name, seed, params, val["macro_f1"])
                if best is None or val["macro_f1"] > best[1]["macro_f1"]:
                    best = (params, val, model)
            params, val, model = best
            run = {"seed": seed, "params": params, "val": val}
            if eval_test:
                run["test"] = classification_report(y["test"], model.predict(X["test"]), classes)
            runs.append(run)
        results["models"][name] = runs
    return results


def summary_tables(results: dict[str, Any]) -> str:
    """Headline table (macro-F1, accuracy) and per-class precision / recall table."""
    split = "test" if results["eval_test"] else "val"
    headers = ["model", "val macro-F1"]
    if split == "test":
        headers += ["test macro-F1"]
    headers += [f"{split} accuracy", "chosen params (per seed)"]
    rows = []
    for name, runs in results["models"].items():
        row = [name, mean_std([r["val"]["macro_f1"] for r in runs])]
        if split == "test":
            row.append(mean_std([r["test"]["macro_f1"] for r in runs]))
        row += [mean_std([r[split]["accuracy"] for r in runs]), [r["params"] for r in runs]]
        rows.append(row)

    per_class_headers = ["class"] + [f"{name} P / R" for name in results["models"]]
    per_class_rows = []
    for c in results["classes"]:
        row = [c]
        for runs in results["models"].values():
            p = np.mean([r[split]["per_class"][c]["precision"] for r in runs])
            r_ = np.mean([r[split]["per_class"][c]["recall"] for r in runs])
            row.append(f"{p:.3f} / {r_:.3f}")
        per_class_rows.append(row)

    n = len(results["seeds"])
    return "\n\n".join(
        [
            f"Classical baseline, mean ± std over {n} seeds ({split} split):",
            format_table(headers, rows),
            f"Per-class precision / recall on {split}, mean over seeds:",
            format_table(per_class_headers, per_class_rows),
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the classical baseline.")
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--root", help="RSCD folder (default: rscd.root in the config)")
    parser.add_argument("--splits-dir", help="split files folder (default: paths.splits)")
    parser.add_argument("--cache-dir", help="feature cache folder (default: paths.processed)")
    parser.add_argument("--results", help="results JSON (default: artifacts/results/baseline.json)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="score the final models on the test split (do this once)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    paths = cfg["paths"]
    cwd = Path.cwd()

    def pick(arg: str | None, default: str | Path) -> Path:
        return resolve_path(arg, cwd) if arg else resolve_path(default)

    root = pick(args.root, cfg["rscd"]["root"])
    splits = load_splits(pick(args.splits_dir, paths["splits"]))
    results_path = pick(args.results, Path(paths["artifacts"]) / "results" / "baseline.json")
    results = run_baseline(
        cfg, splits, root, pick(args.cache_dir, paths["processed"]), args.workers, args.eval_test
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(summary_tables(results))
    print(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()
