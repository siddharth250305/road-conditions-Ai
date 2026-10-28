"""Pseudo-labelling (PRD D4): self-training on unlabelled target images.

The source detector labels a pool of target images; boxes at or above a confidence threshold
become training labels. The detector is then fine-tuned from the source weights on those
images plus a replay sample of source images (so it does not forget the source domain).

No target label is used for training. The pool's hidden labels serve one diagnostic only: how
precise the pseudo-labels were. The threshold is chosen on target *validation* images and the
test split is scored only for the chosen threshold, so this selection does use target labels
and is recorded as such in the results.
"""

import json
import logging
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from roadcond.data.rdd import label_path_for, read_name_list, slug, tree_image
from roadcond.detection.metrics import f1_at_conf, summarize
from roadcond.detection.train_yolo import (
    Env,
    collect_predictions,
    evaluate_detector,
    last_weights,
    load_ground_truth,
    results_dir,
    save_json,
    split_images,
    subset_yaml,
    train_detector,
    tree_paths,
)

log = logging.getLogger(__name__)


def pseudo_rows(pred: dict[str, np.ndarray], conf: float) -> np.ndarray:
    """YOLO rows (cls cx cy w h) for the predicted boxes with score >= conf."""
    keep = np.asarray(pred["scores"]) >= conf
    boxes = np.asarray(pred["boxes"], dtype=float).reshape(-1, 4)[keep]
    cls = np.asarray(pred["cls"])[keep].astype(float).reshape(-1, 1)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    return np.hstack([cls, np.stack([cx, cy, w, h], axis=1)])


def write_pseudo_tree(
    env: Env, run_name: str, country: str, rows_by_name: dict[str, np.ndarray]
) -> list[Path]:
    """Symlink the pseudo-labelled images into a fresh tree and write their pseudo labels.

    Only images with at least one pseudo box are kept: an empty pseudo label would claim
    "no damage here", which is a much stronger statement than "nothing was confident".
    """
    root = env.work_dir / "pseudo" / run_name
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    images = []
    for name, rows in sorted(rows_by_name.items()):
        if not len(rows):
            continue
        image = root / "images" / name
        if not (image.exists() or image.is_symlink()):
            os.symlink(tree_image(env.work_dir, country, name).resolve(), image)
        lines = [f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows]
        (root / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
        images.append(image)
    return images


class PoolPredictor:
    """Runs the source detector over a seeded sample of a country's pool, once per sample."""

    def __init__(self, env: Env, source_weights: str | Path, conf_min: float):
        from ultralytics import YOLO

        self.env = env
        self.model = YOLO(str(source_weights))
        self.conf_min = conf_min
        self._cache: dict[tuple[str, int], dict[str, dict[str, np.ndarray]]] = {}

    def sample(self, country: str, seed: int) -> list[str]:
        pool = read_name_list(self.env.splits_dir / f"rdd_{slug(country)}_pool.txt")
        size = min(self.env.cfg["rdd2022"]["pseudo_label"]["pool_size"], len(pool))
        return sorted(random.Random(seed).sample(pool, size))

    def predict(self, country: str, seed: int) -> dict[str, dict[str, np.ndarray]]:
        key = (country, seed)
        if key not in self._cache:
            names = self.sample(country, seed)
            images = tree_paths(self.env, country, names)
            preds = collect_predictions(self.model, images, self.env, self.conf_min)
            self._cache[key] = dict(zip(names, preds, strict=True))
        return self._cache[key]


def run_pseudo_one(
    env: Env,
    predictor: PoolPredictor,
    source_weights: str | Path,
    country: str,
    threshold: float,
    seed: int,
    split: str,
    force: bool = False,
) -> dict[str, Any]:
    """One pseudo-label fine-tune, scored on `split` ('val' to pick a threshold, 'test' final)."""
    rdd = env.cfg["rdd2022"]
    pl = rdd["pseudo_label"]
    name = f"pseudo_{slug(country)}_tau{threshold:g}_s{seed}"
    result_path = results_dir(env) / "pseudo" / f"{name}_{split}.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text())

    preds = predictor.predict(country, seed)
    rows = {n: pseudo_rows(p, threshold) for n, p in preds.items()}
    pseudo_images = write_pseudo_tree(env, name, country, rows)

    # Diagnostic: how good were the pseudo-labels against the pool's (hidden) labels?
    names = sorted(preds)
    truths = load_ground_truth(tree_paths(env, country, names))
    quality = f1_at_conf([preds[n] for n in names], truths, threshold)

    source = rdd["source_country"]
    replay_pool = read_name_list(env.splits_dir / f"rdd_{slug(source)}_train.txt")
    replay = random.Random(seed).sample(replay_pool, min(pl["source_replay"], len(replay_pool)))
    train_images = [*pseudo_images, *tree_paths(env, source, sorted(replay))]
    data = subset_yaml(
        env,
        name,
        train_images,
        split_images(env, country, "val"),
        split_images(env, country, "test"),
    )
    train_detector(
        env,
        name,
        source_weights,
        data,
        epochs=pl["epochs"],
        batch=pl["batch"],
        seed=seed,
        val=False,
        optimizer="AdamW",
        lr0=pl["lr0"],
        freeze=pl["freeze"],
        force=force,
    )
    scores = evaluate_detector(env, last_weights(env, name), data, split, f"{name}_{split}")
    result = {
        "country": country,
        "threshold": threshold,
        "seed": seed,
        "split": split,
        "pool_images": len(names),
        "pseudo_images": len(pseudo_images),
        "pseudo_boxes": int(sum(len(r) for r in rows.values())),
        "no_pseudo_labels": not pseudo_images,
        "pseudo_label_precision": quality["precision"],
        "pseudo_label_recall": quality["recall"],
        "replay_images": len(replay),
        **scores,
    }
    save_json(result_path, result)
    return result


def run_pseudo(
    env: Env,
    source_weights: str | Path,
    countries: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    thresholds: Sequence[float] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Pick a threshold per country on target val (first seed), then score it on test per seed."""
    rdd = env.cfg["rdd2022"]
    pl = rdd["pseudo_label"]
    countries = list(countries or rdd["target_countries"])
    seeds = list(seeds if seeds is not None else pl["seeds"])
    thresholds = list(thresholds if thresholds is not None else pl["sweep"])
    predictor = PoolPredictor(env, source_weights, conf_min=min(thresholds))

    summary: dict[str, Any] = {"countries": {}, "threshold_selected_on": "target validation images"}
    for country in countries:
        val_scores = {}
        if len(thresholds) > 1:
            for tau in thresholds:
                run = run_pseudo_one(
                    env, predictor, source_weights, country, tau, seeds[0], "val", force
                )
                val_scores[tau] = run["map50"]
            default = pl["confidence_threshold"]
            chosen = max(val_scores, key=lambda t: (val_scores[t], -abs(t - default)))
        else:
            chosen = thresholds[0]
        log.info("%s: threshold %g (val mAP50 %s)", country, chosen, val_scores or "n/a")
        runs = [
            run_pseudo_one(env, predictor, source_weights, country, chosen, seed, "test", force)
            for seed in seeds
        ]
        summary["countries"][country] = {
            "threshold": chosen,
            "val_map50_by_threshold": {str(t): v for t, v in val_scores.items()},
            "runs": runs,
            "map50": summarize([r["map50"] for r in runs]),
            "map50_95": summarize([r["map50_95"] for r in runs]),
            "f1": summarize([r["f1"] for r in runs]),
            "pseudo_label_precision": summarize([r["pseudo_label_precision"] for r in runs]),
        }
    save_json(results_dir(env) / "detection_pseudo.json", summary)
    return summary


def label_files_exist(env: Env, run_name: str) -> int:
    """How many pseudo label files a run wrote (for tests and sanity checks)."""
    root = env.work_dir / "pseudo" / run_name
    return sum(1 for p in (root / "images").glob("*") if label_path_for(p).is_file())
