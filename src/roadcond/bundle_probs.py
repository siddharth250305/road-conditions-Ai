"""Save the served model's probabilities on RSCD val and test, for the review router.

    python scripts/save_bundle_probs.py --root <RSCD> [--bundle serving] [--out FILE.npz]

It runs the exported ONNX bundle with the server's own decoding, resizing and temperature
(roadcond.serving), so the offline routing check measures exactly what the API serves. The
output is a roadcond.routing.probs file with one model, named after the bundle's model_name.
"""

import argparse
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from roadcond.config import load_config, resolve_path
from roadcond.data.rscd import Record, load_splits
from roadcond.routing.probs import SplitProbs, save_probs
from roadcond.routing.run import NO_SEED
from roadcond.serving import Predictor, decode_image, preprocess, softmax

log = logging.getLogger(__name__)


def _load(path: Path, predictor: Predictor) -> np.ndarray:
    image = decode_image(path.read_bytes(), predictor.settings.max_pixels)
    return preprocess(image, predictor.height, predictor.width)[0]


def served_probs(
    predictor: Predictor, paths: Sequence[Path], batch_size: int = 32, workers: int = 8
) -> np.ndarray:
    """(N, C) probabilities, as Predictor.predict computes them, for each image path."""
    logits_name = predictor.meta["outputs"]["logits"]
    out = np.empty((len(paths), len(predictor.classes)), dtype=np.float64)
    started = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            batch = np.concatenate(list(pool.map(lambda p: _load(p, predictor), chunk)))
            (logits,) = predictor.session.run([logits_name], {predictor.input_name: batch})
            for i, row in enumerate(logits):
                out[start + i] = softmax(row.astype(np.float64) / predictor.temperature)
            done = start + len(chunk)
            if done % (batch_size * 100) < batch_size or done == len(paths):
                rate = done / max(time.monotonic() - started, 1e-9)
                log.info("%d/%d images (%.0f/s)", done, len(paths), rate)
    return out


def split_probs(
    predictor: Predictor, records: Sequence[Record], root: Path, batch_size: int, workers: int
) -> SplitProbs:
    index = {c: i for i, c in enumerate(predictor.classes)}
    paths = [root / r.path for r in records]
    return SplitProbs(
        image_ids=np.array([r.path for r in records]),
        labels=np.array([index[r.friction] for r in records], dtype=np.int64),
        probs={predictor.meta["model_name"]: served_probs(predictor, paths, batch_size, workers)},
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Save the served model's val/test probabilities.")
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--root", help="RSCD root (default: rscd.root)")
    parser.add_argument("--splits-dir", help="saved split lists (default: paths.splits)")
    parser.add_argument("--bundle", default="serving", help="exported model bundle folder")
    parser.add_argument("--out", help="output .npz (default: <artifacts>/results/cnn_probs.npz)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8, help="threads decoding images")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    cwd = Path.cwd()

    def pick(arg: str | None, default: str | Path) -> Path:
        return resolve_path(arg, cwd) if arg else resolve_path(default)

    predictor = Predictor(pick(args.bundle, "serving"))
    if predictor.meta.get("mock"):
        raise SystemExit("The bundle is a mock model; export a trained model first")
    if predictor.classes != list(cfg["rscd"]["classes"]):
        raise SystemExit(f"Bundle classes {predictor.classes} differ from the config's")
    root = pick(args.root, cfg["rscd"]["root"])
    splits = load_splits(pick(args.splits_dir, cfg["paths"]["splits"]))
    out = pick(args.out, Path(cfg["paths"]["artifacts"]) / "results" / "cnn_probs.npz")

    probs = {}
    for split in ("val", "test"):
        log.info("%s: %d images", split, len(splits[split]))
        probs[split] = split_probs(predictor, splits[split], root, args.batch_size, args.workers)
    seed = predictor.meta.get("training", {}).get("seed")
    save_probs(out, predictor.classes, NO_SEED if seed is None else int(seed), probs)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
