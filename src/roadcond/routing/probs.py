"""Saved class probabilities: the file the review router reads.

One .npz per run, written with plain arrays (no pickles). Any model can be routed once its
probabilities are saved in this format, so a calibrated CNN can join later without code changes:

    format_version          1
    classes                 (C,) class names, in column order
    models                  (M,) model names
    seed                    model seed the probabilities come from
    {split}_image_ids       (N,) image ids (RSCD paths), for split in val and test
    {split}_labels          (N,) true class index
    {split}_probs__{model}  (N, C) probabilities, each row sums to 1
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
SPLITS = ("val", "test")
_SEP = "__"


@dataclass(frozen=True)
class SplitProbs:
    """Every model's probabilities on one split, row-aligned with image ids and labels."""

    image_ids: np.ndarray  # (N,) str
    labels: np.ndarray  # (N,) int, index into classes
    probs: dict[str, np.ndarray]  # model name -> (N, C)


@dataclass(frozen=True)
class ProbsFile:
    classes: list[str]
    seed: int
    splits: dict[str, SplitProbs]

    @property
    def models(self) -> list[str]:
        return list(self.splits[SPLITS[0]].probs)


def _check(classes: list[str], splits: Mapping[str, SplitProbs]) -> None:
    if set(splits) != set(SPLITS):
        raise ValueError(f"Need probabilities for splits {SPLITS}, got {sorted(splits)}")
    models = None
    for split, sp in splits.items():
        n = len(sp.image_ids)
        if sp.labels.shape != (n,):
            raise ValueError(f"{split}: {n} image ids but labels of shape {sp.labels.shape}")
        if len(sp.labels) and (sp.labels.min() < 0 or sp.labels.max() >= len(classes)):
            raise ValueError(f"{split}: labels must index into the {len(classes)} classes")
        if models is not None and list(sp.probs) != models:
            raise ValueError(f"{split}: models {list(sp.probs)} differ from {models}")
        models = list(sp.probs)
        if not models:
            raise ValueError(f"{split}: no model probabilities")
        for name, p in sp.probs.items():
            if _SEP in name:
                raise ValueError(f"Model name {name!r} must not contain {_SEP!r}")
            if p.shape != (n, len(classes)):
                raise ValueError(f"{split}/{name}: probs {p.shape}, expected {(n, len(classes))}")


def save_probs(
    path: str | Path, classes: list[str], seed: int, splits: Mapping[str, SplitProbs]
) -> None:
    """Write the probabilities file; validates shapes and labels first."""
    _check(classes, splits)
    arrays: dict[str, np.ndarray] = {
        "format_version": np.array(FORMAT_VERSION),
        "classes": np.array(classes),
        "models": np.array(list(splits[SPLITS[0]].probs)),
        "seed": np.array(seed),
    }
    for split, sp in splits.items():
        arrays[f"{split}_image_ids"] = np.asarray(sp.image_ids, dtype=str)
        arrays[f"{split}_labels"] = np.asarray(sp.labels, dtype=np.int64)
        for name, p in sp.probs.items():
            arrays[f"{split}_probs{_SEP}{name}"] = np.asarray(p, dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_probs(path: str | Path) -> ProbsFile:
    """Read and validate a probabilities file written by save_probs."""
    with np.load(path, allow_pickle=False) as z:
        version = int(z["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"{path}: probabilities format {version}, expected {FORMAT_VERSION}")
        classes = [str(c) for c in z["classes"]]
        models = [str(m) for m in z["models"]]
        splits = {
            split: SplitProbs(
                image_ids=z[f"{split}_image_ids"],
                labels=z[f"{split}_labels"],
                probs={m: z[f"{split}_probs{_SEP}{m}"] for m in models},
            )
            for split in SPLITS
        }
        seed = int(z["seed"])
    _check(classes, splits)
    return ProbsFile(classes=classes, seed=seed, splits=splits)
