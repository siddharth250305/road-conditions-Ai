"""Synthetic RSCD-like data for tests and local dry runs. These are not road images.

The layout matches what configs/default.yaml assumes about the real download:
train/<label>/*.jpg (label from the folder) and flat val/test folders with the label in the
file name. A few test images are copied into train as planted leaks.

    python -m roadcond.data.mock --out data/raw/rscd_mock
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from roadcond.config import load_config, resolve_path
from roadcond.data.rscd import SPLITS, to_friction

# The 27 RSCD labels as assumed here: friction x material x unevenness, plus snow and ice.
# Only the friction prefix is relied on by roadcond.data.rscd.
RSCD_LABELS = (
    [
        f"{f}_{m}_{u}"
        for f in ("dry", "wet", "water")
        for m in ("asphalt", "concrete")
        for u in ("smooth", "slight", "severe")
    ]
    + [f"{f}_{m}" for f in ("dry", "wet", "water") for m in ("gravel", "mud")]
    + ["fresh_snow", "melted_snow", "ice"]
)

FRICTION_COLORS = {
    "dry": (150, 140, 125),
    "wet": (90, 100, 115),
    "water": (50, 70, 110),
    "fresh_snow": (235, 235, 240),
    "melted_snow": (170, 180, 190),
    "ice": (190, 225, 235),
}
_STRIPES = {"smooth": 0, "slight": 4, "severe": 10}  # horizontal bands per image height


def _make_image(label: str, rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    base = np.array(FRICTION_COLORS[to_friction(label, FRICTION_COLORS)], dtype=float)
    # A random low-frequency field makes every image (and its dHash) distinct.
    coarse = rng.normal(0.0, 1.0, (6, 9)).astype(np.float32)
    blobs = np.asarray(Image.fromarray(coarse).resize((width, height), Image.Resampling.BILINEAR))
    freq = _STRIPES.get(label.rsplit("_", 1)[-1], 0)
    stripes = 15 * np.sin(2 * np.pi * freq * np.arange(height) / height)
    noise = rng.normal(0, 6, (height, width, 3))
    img = base + 25 * blobs[..., None] + stripes[:, None, None] + noise
    return img.clip(0, 255).astype(np.uint8)


def make_mock_rscd(
    root: str | Path,
    split_dirs: dict[str, str] | None = None,
    n_per_label: Sequence[int] = (20, 4, 4),
    size: tuple[int, int] = (48, 72),
    n_leaks: int = 2,
    seed: int = 0,
) -> list[str]:
    """Write a mock dataset; return the train paths that copy a test image (planted leaks)."""
    root = Path(root)
    split_dirs = split_dirs or load_config()["rscd"]["split_dirs"]
    rng = np.random.default_rng(seed)
    serial = 20220101000000000
    test_images: list[tuple[str, Path]] = []
    for split, n in zip(SPLITS, n_per_label, strict=True):
        split_dir = root / split_dirs[split]
        for label in RSCD_LABELS:
            for i in range(n):
                if split == "train":
                    path = split_dir / label / f"{i:06d}.jpg"
                else:
                    serial += 1
                    path = split_dir / f"{serial}-{label.replace('_', '-')}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(_make_image(label, rng, size)).save(path, quality=95)
                if split == "test":
                    test_images.append((label, path))

    leaks = []
    for k, idx in enumerate(rng.choice(len(test_images), n_leaks, replace=False)):
        label, src = test_images[idx]
        dst = root / split_dirs["train"] / label / f"leak_{k:03d}.jpg"
        with Image.open(src) as img:
            # Every other leak is brightened and re-encoded: a near, not exact, duplicate.
            pixels = np.asarray(img, dtype=np.int16) + (0 if k % 2 == 0 else 6)
        Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(dst, quality=90)
        leaks.append(dst.relative_to(root).as_posix())
    return leaks


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write a small synthetic RSCD-like dataset.")
    parser.add_argument("--out", default="data/raw/rscd_mock")
    parser.add_argument(
        "--per-label", type=int, nargs=3, default=(20, 4, 4), metavar=("TRAIN", "VAL", "TEST")
    )
    args = parser.parse_args(argv)
    out = resolve_path(args.out, Path.cwd())
    leaks = make_mock_rscd(out, n_per_label=args.per_label)
    print(f"Wrote mock RSCD to {out} ({len(leaks)} planted leaks: {leaks})")


if __name__ == "__main__":
    main()
