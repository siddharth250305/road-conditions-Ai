"""Source detector (PRD D1): Ultralytics YOLOv8 training and per-country evaluation.

The source detector is trained on one country (Japan) and scored on a held-out test split of
every country. Training and evaluation go through Ultralytics; F1 at a fixed confidence comes
from roadcond.detection.metrics, because Ultralytics reports F1 at its own best threshold.
Ultralytics is AGPL-3.0, so it is imported only inside the functions that need it.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from roadcond.data.rdd import (
    class_names,
    label_path_for,
    read_image_list,
    read_yolo_labels,
    slug,
    tree_image,
    write_data_yaml,
    write_image_list,
    yolo_to_xyxy,
)
from roadcond.detection.metrics import f1_at_conf, relative_gap

log = logging.getLogger(__name__)

METRICS = ("map50", "map50_95")


@dataclass
class Env:
    """Where things live and how to run: shared by every detection command."""

    cfg: dict[str, Any]
    work_dir: Path
    splits_dir: Path
    out_dir: Path
    device: str = "cpu"
    pretrained: bool = True
    workers: int | None = None  # None: use rdd2022.yolo.workers

    @property
    def yolo(self) -> dict[str, Any]:
        return self.cfg["rdd2022"]["yolo"]

    @property
    def n_workers(self) -> int:
        return self.yolo["workers"] if self.workers is None else self.workers


def pick_device(name: str = "auto") -> str:
    """'auto' -> first CUDA GPU if there is one, else cpu; anything else is passed through."""
    if name != "auto":
        return name
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def all_countries(cfg: dict[str, Any]) -> list[str]:
    rdd = cfg["rdd2022"]
    return [rdd["source_country"], *rdd["target_countries"]]


def country_yaml(env: Env, country: str) -> Path:
    return env.splits_dir / f"rdd_{slug(country)}.yaml"


def run_dir(env: Env, name: str) -> Path:
    return env.out_dir / "detection" / name


def results_dir(env: Env) -> Path:
    return env.out_dir / "results"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def base_weights(env: Env) -> str:
    """Ultralytics name of the starting weights: COCO-pretrained, or random init for tests."""
    model = env.yolo["model"]
    return f"{model}.pt" if env.pretrained else f"{model}.yaml"


def best_weights(env: Env, name: str) -> Path:
    return run_dir(env, name) / "weights" / "best.pt"


def last_weights(env: Env, name: str) -> Path:
    return run_dir(env, name) / "weights" / "last.pt"


def subset_yaml(
    env: Env,
    name: str,
    train_images: list[Path],
    val_images: list[Path],
    test_images: list[Path],
) -> Path:
    """Data config for one experiment: its own list files, written under the work tree."""
    lists = {}
    for key, images in (("train", train_images), ("val", val_images), ("test", test_images)):
        lists[key] = env.work_dir / "lists" / f"{name}_{key}.txt"
        write_image_list(lists[key], images)
    path = env.work_dir / "yaml" / f"{name}.yaml"
    write_data_yaml(path, env.work_dir, lists, class_names(env.cfg))
    return path


def split_images(env: Env, country: str, split: str) -> list[Path]:
    """Work-tree image paths of a country's saved split (from the country's data config)."""
    data = yaml.safe_load(country_yaml(env, country).read_text())
    return read_image_list(data[split])


def tree_paths(env: Env, country: str, names: list[str]) -> list[Path]:
    return [tree_image(env.work_dir, country, n) for n in names]


def train_detector(
    env: Env,
    name: str,
    weights: str | Path,
    data_yaml: Path,
    *,
    epochs: int,
    batch: int,
    seed: int = 0,
    val: bool = True,
    augmentations: list | None = None,
    optimizer: str | None = None,
    lr0: float | None = None,
    freeze: int | None = None,
    pretrained: bool = True,
    resume: bool = False,
    force: bool = False,
) -> Path:
    """Train or fine-tune with Ultralytics; returns the run folder.

    A finished run leaves done.json and is skipped on the next call (unless force). With
    resume, an interrupted run continues from its last.pt. Ultralytics ignores lr0 under its
    default 'auto' optimizer, so callers that set lr0 must also name an optimizer.
    """
    from ultralytics import YOLO

    run = run_dir(env, name)
    done = run / "done.json"
    if done.is_file() and not force:
        log.info("%s is already trained; skipping", name)
        return run
    last = run / "weights" / "last.pt"
    if resume and last.is_file() and not force:
        try:
            log.info("Resuming %s from %s", name, last)
            YOLO(str(last)).train(resume=True)
        except AssertionError as err:  # Ultralytics: "training ... is finished, nothing to resume"
            if "finished" not in str(err):
                raise
    else:
        args: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": env.yolo["imgsz"],
            "batch": batch,
            "workers": env.n_workers,
            "device": env.device,
            "seed": seed,
            "plots": False,
            "val": val,
            "project": str(env.out_dir / "detection"),
            "name": name,
            "exist_ok": True,
            "verbose": False,
            "pretrained": pretrained,
        }
        for key, value in (("optimizer", optimizer), ("lr0", lr0), ("freeze", freeze)):
            if value is not None:
                args[key] = value
        if augmentations:
            args["augmentations"] = augmentations
        YOLO(str(weights)).train(**args)
    save_json(done, {"name": name, "epochs": epochs, "seed": seed, "data": str(data_yaml)})
    return run


def collect_predictions(
    model: Any, images: list[Path], env: Env, conf: float
) -> list[dict[str, np.ndarray]]:
    """Predictions per image as normalised corner boxes, in the order of images."""
    if not images:
        return []
    out = []
    for result in model.predict(
        source=[str(p) for p in images],
        conf=conf,
        imgsz=env.yolo["imgsz"],
        batch=env.yolo["batch"],
        device=env.device,
        verbose=False,
        stream=True,
    ):
        boxes = result.boxes
        out.append(
            {
                "boxes": boxes.xyxyn.cpu().numpy(),
                "scores": boxes.conf.cpu().numpy(),
                "cls": boxes.cls.cpu().numpy().astype(int),
            }
        )
    return out


def load_ground_truth(images: list[Path]) -> list[dict[str, np.ndarray]]:
    """Ground-truth boxes (normalised corners) read from each image's YOLO label file."""
    truths = []
    for path in images:
        rows = read_yolo_labels(label_path_for(path))
        truths.append({"boxes": yolo_to_xyxy(rows), "cls": rows[:, 0].astype(int)})
    return truths


def evaluate_detector(
    env: Env, weights: str | Path, data_yaml: Path, split: str = "test", tag: str = "eval"
) -> dict[str, Any]:
    """mAP@0.5, mAP@0.5:0.95 (Ultralytics) and F1 at the fixed confidence, on one split."""
    from ultralytics import YOLO

    conf = env.cfg["metrics"]["detection_f1_conf"]
    names = class_names(env.cfg)
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=env.yolo["imgsz"],
        batch=env.yolo["batch"],
        workers=env.n_workers,
        device=env.device,
        conf=0.001,
        iou=0.6,
        plots=False,
        verbose=False,
        project=str(env.out_dir / "detection" / "_val"),
        name=tag,
        exist_ok=True,
    )
    box = metrics.box
    ap50: dict[str, float | None] = {names[i]: None for i in sorted(names)}
    ap: dict[str, float | None] = dict(ap50)
    for idx, a50, a in zip(box.ap_class_index, box.ap50, box.ap, strict=True):
        ap50[names[int(idx)]], ap[names[int(idx)]] = float(a50), float(a)

    images = read_image_list(yaml.safe_load(Path(data_yaml).read_text())[split])
    truths = load_ground_truth(images)
    f1 = f1_at_conf(collect_predictions(model, images, env, conf), truths, conf)
    return {
        "split": split,
        "eval_images": len(images),
        "eval_boxes": int(sum(len(t["cls"]) for t in truths)),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": f1["precision"],
        "recall": f1["recall"],
        "f1": f1["f1"],
        "f1_conf": conf,
        "tp": f1["tp"],
        "fp": f1["fp"],
        "fn": f1["fn"],
        "ap50_per_class": ap50,
        "ap50_95_per_class": ap,
    }


def evaluate_countries(env: Env, weights: str | Path, split: str = "test", tag: str = "eval"):
    """Evaluate on the source country and every target country."""
    return {
        c: evaluate_detector(env, weights, country_yaml(env, c), split, f"{tag}_{slug(c)}")
        for c in all_countries(env.cfg)
    }


def domain_gaps(results: dict[str, dict[str, Any]], source: str) -> dict[str, Any]:
    """Per target: source score, target score and relative gap for mAP@0.5 and mAP@0.5:0.95."""
    gaps = {}
    for country, scores in results.items():
        if country == source:
            continue
        gaps[country] = {
            m: {
                "source": results[source][m],
                "target": scores[m],
                "relative_gap": relative_gap(results[source][m], scores[m]),
            }
            for m in METRICS
        }
    return gaps


def train_source(
    env: Env, *, augment: bool = False, resume: bool = False, force: bool = False
) -> Path:
    """D1 (and D2 with augment=True): train on the source country; returns best.pt.

    best.pt is picked on the *source* validation split, which is allowed: no target data.
    """
    from roadcond.detection.augment import weather_transforms

    rdd = env.cfg["rdd2022"]
    name = "source_aug" if augment else "source"
    augmentations = weather_transforms(rdd["augment"]["strength"]) if augment else None
    train_detector(
        env,
        name,
        base_weights(env),
        country_yaml(env, rdd["source_country"]),
        epochs=env.yolo["epochs"],
        batch=env.yolo["batch"],
        seed=env.yolo["seed"],
        val=True,
        augmentations=augmentations,
        pretrained=env.pretrained,
        resume=resume,
        force=force,
    )
    return best_weights(env, name)


def evaluate_source(env: Env, name: str = "source", force: bool = False) -> dict[str, Any]:
    """Score a trained source detector (name = run) on every country's test split."""
    path = results_dir(env) / f"detection_{name}.json"
    if path.is_file() and not force:
        return json.loads(path.read_text())
    source = env.cfg["rdd2022"]["source_country"]
    countries = evaluate_countries(env, best_weights(env, name), "test", name)
    result = {
        "run": name,
        "weights": str(best_weights(env, name)),
        "split": "test",
        "f1_conf": env.cfg["metrics"]["detection_f1_conf"],
        "source_country": source,
        "countries": countries,
        "gaps": domain_gaps(countries, source),
    }
    save_json(path, result)
    return result


def default_workers() -> int:
    return os.cpu_count() or 1
