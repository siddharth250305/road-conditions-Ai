"""RDD2022 detection data (PRD D1-D5): parsers, per-country held-out splits, YOLO trees.

RDD2022 ships Pascal-VOC XML for the training images, but no public test labels, so every
country gets its own seeded train-pool / val / test split from the labelled images (PRD data
rules). Pool images that nearly duplicate a val/test image are dropped, as in Phase 1.

For Ultralytics, each country gets a work tree of symlinked images next to YOLO label files
(it finds labels by swapping `images` for `labels` in the path). Subsets are plain list
files, so a few-shot subset or a pseudo-labelled mix needs no copying.

    python -m roadcond.data.rdd --root /path/to/RDD2022
    python -m roadcond.data.rdd --make-mock data/raw/rdd_mock
"""

import argparse
import json
import logging
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from roadcond.config import load_config, resolve_path
from roadcond.data.rscd import NearDuplicateIndex, hash_images

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
MAX_REPORT_EXAMPLES = 20
MIN_BOX_PIXELS = 1.0


@dataclass(frozen=True)
class Box:
    cls: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class Annotation:
    name: str  # image file name, e.g. Japan_000001.jpg
    width: int
    height: int
    boxes: list[Box]


def class_ids(cfg: dict[str, Any]) -> dict[str, int]:
    """Damage code -> class id, from the config (D00 -> 0, D10 -> 1, D20 -> 2, D40 -> 3)."""
    return {c["code"]: c["id"] for c in cfg["rdd2022"]["classes"]}


def class_names(cfg: dict[str, Any]) -> dict[int, str]:
    return {c["id"]: c["code"] for c in cfg["rdd2022"]["classes"]}


def slug(country: str) -> str:
    return country.lower()


# --- parsing --------------------------------------------------------------------------------


def _to_int(text: str | None) -> int:
    try:
        return int(float(text)) if text is not None else 0
    except ValueError:
        return 0


def parse_voc_xml(
    source: str | bytes | Path, ids: dict[str, int], image_path: Path | None = None
) -> tuple[int, int, list[Box], Counter]:
    """Pascal-VOC XML -> (width, height, boxes, dropped-label counts).

    Objects whose name is not in ids (other RDD codes such as D43) are dropped and counted.
    Boxes are clamped to the image; degenerate ones are dropped. If the XML carries no image
    size, it is read from image_path. XML with a DTD or entities is rejected: the files come
    from a download, and entity expansion can blow up memory.
    """
    data = source.read_bytes() if isinstance(source, Path) else source
    if isinstance(data, str):
        data = data.encode()
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("XML with a DTD or entities is not accepted")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as err:
        raise ValueError(f"Invalid XML: {err}") from None

    width = _to_int(root.findtext("size/width"))
    height = _to_int(root.findtext("size/height"))
    if width <= 0 or height <= 0:
        if image_path is None:
            raise ValueError("XML has no image size and no image was given")
        with Image.open(image_path) as img:
            width, height = img.size

    boxes: list[Box] = []
    dropped: Counter = Counter()
    for obj in root.iter("object"):
        code = (obj.findtext("name") or "").strip()
        if code not in ids:
            dropped[code or "<empty>"] += 1
            continue
        bndbox = obj.find("bndbox")
        try:
            xmin, ymin, xmax, ymax = (
                float(bndbox.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")
            )
        except (AttributeError, TypeError, ValueError):
            dropped["<bad bndbox>"] += 1
            continue
        xmin, xmax = max(0.0, xmin), min(float(width), xmax)
        ymin, ymax = max(0.0, ymin), min(float(height), ymax)
        if xmax - xmin < MIN_BOX_PIXELS or ymax - ymin < MIN_BOX_PIXELS:
            dropped["<degenerate>"] += 1
            continue
        boxes.append(Box(ids[code], xmin, ymin, xmax, ymax))
    return width, height, boxes, dropped


def boxes_to_yolo(boxes: Iterable[Box], width: int, height: int) -> list[str]:
    """YOLO label lines `cls cx cy w h`, normalised to [0, 1]."""
    lines = []
    for b in boxes:
        cx, cy = (b.xmin + b.xmax) / 2 / width, (b.ymin + b.ymax) / 2 / height
        w, h = (b.xmax - b.xmin) / width, (b.ymax - b.ymin) / height
        lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def parse_yolo_text(text: str) -> np.ndarray:
    """YOLO label text -> array (N, 5) of [cls, cx, cy, w, h]; blank text gives (0, 5)."""
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"YOLO line {number} needs 5 values, got {len(parts)}: {line!r}")
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            raise ValueError(f"YOLO line {number} is not numeric: {line!r}") from None
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def read_yolo_labels(path: str | Path) -> np.ndarray:
    """Labels of one image; a missing file counts as no objects."""
    path = Path(path)
    return parse_yolo_text(path.read_text()) if path.is_file() else np.zeros((0, 5))


def yolo_to_xyxy(rows: np.ndarray, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """(N, 5) YOLO rows -> (N, 4) corner boxes; width = height = 1 keeps them normalised."""
    rows = np.asarray(rows, dtype=float).reshape(-1, 5)
    cx, cy, w, h = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    return np.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
        axis=1,
    )


def yolo_to_boxes(rows: np.ndarray, width: int, height: int) -> list[Box]:
    rows = np.asarray(rows, dtype=float).reshape(-1, 5)
    corners = yolo_to_xyxy(rows, width, height)
    return [Box(int(r[0]), *map(float, c)) for r, c in zip(rows, corners, strict=True)]


# --- indexing and splitting -----------------------------------------------------------------


def index_country(
    root: str | Path, country: str, cfg: dict[str, Any]
) -> tuple[list[Annotation], dict[str, Any]]:
    """Parse every labelled training image of a country; returns annotations and a report."""
    rdd = cfg["rdd2022"]
    base = Path(root) / rdd["countries"][country]
    image_dir, xml_dir = base / rdd["images_dir"], base / rdd["annotations_dir"]
    for folder in (image_dir, xml_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"RDD2022 folder for {country} not found: {folder}")
    ids = class_ids(cfg)
    images = {p.stem: p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES}
    xmls = {p.stem: p for p in sorted(xml_dir.glob("*.xml"))}

    annotations: list[Annotation] = []
    dropped: Counter = Counter()
    bad_xml: list[str] = []
    for stem, image in images.items():
        if stem not in xmls:
            continue
        try:
            width, height, boxes, drop = parse_voc_xml(xmls[stem], ids, image)
        except (ValueError, OSError) as err:
            bad_xml.append(f"{xmls[stem].name}: {err}")
            continue
        dropped.update(drop)
        annotations.append(Annotation(image.name, width, height, boxes))
    if not annotations:
        raise ValueError(f"No labelled images found for {country} under {base}")

    per_class = Counter(b.cls for a in annotations for b in a.boxes)
    names = class_names(cfg)
    sizes = Counter((a.width, a.height) for a in annotations)
    report = {
        "images": len(images),
        "labelled_images": len(annotations),
        "images_without_xml": len(images.keys() - xmls.keys()),
        "xml_without_image": len(xmls.keys() - images.keys()),
        "unreadable_xml": len(bad_xml),
        "unreadable_examples": bad_xml[:MAX_REPORT_EXAMPLES],
        "images_with_damage": sum(1 for a in annotations if a.boxes),
        "boxes_per_class": {names[i]: per_class.get(i, 0) for i in sorted(names)},
        "dropped_labels": dict(dropped.most_common()),
        "image_sizes": [list(s) for s, _ in sizes.most_common(3)],
    }
    return annotations, report


def split_annotations(
    annotations: Sequence[Annotation], split_cfg: dict[str, Any]
) -> dict[str, list[Annotation]]:
    """Seeded train_pool / val / test split. Val is capped; the excess joins the pool."""
    fractions = {k: split_cfg[k] for k in ("train_pool", "val", "test")}
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    order = sorted(annotations, key=lambda a: a.name)
    random.Random(split_cfg["seed"]).shuffle(order)
    n_test = round(len(order) * fractions["test"])
    n_val = min(round(len(order) * fractions["val"]), split_cfg.get("max_val_images", len(order)))
    test, val = order[:n_test], order[n_test : n_test + n_val]
    pool = order[n_test + n_val :]
    return {
        "train_pool": sorted(pool, key=lambda a: a.name),
        "val": sorted(val, key=lambda a: a.name),
        "test": sorted(test, key=lambda a: a.name),
    }


def drop_near_duplicates(
    pool: Sequence[Annotation],
    held_out: Sequence[Annotation],
    path_of: Callable[[Annotation], Path],
    dedup_cfg: dict[str, Any],
    workers: int = 1,
) -> tuple[list[Annotation], dict[str, Any]]:
    """Remove pool images within max_hamming dHash bits of any held-out (val/test) image."""
    if not pool or not held_out:
        return list(pool), {"dropped": 0, "examples": []}
    size = dedup_cfg["hash_size"]
    held_hashes = hash_images([path_of(a) for a in held_out], size, workers)
    index = NearDuplicateIndex(held_hashes, size**2, dedup_cfg["max_hamming"])
    pool_hashes = hash_images([path_of(a) for a in pool], size, workers)
    kept, examples, dropped = [], [], 0
    for annotation, h in zip(pool, pool_hashes, strict=True):
        matches = index.query(h)
        if not matches:
            kept.append(annotation)
            continue
        dropped += 1
        if len(examples) < MAX_REPORT_EXAMPLES:
            item, distance = matches[0]
            examples.append(
                {"pool": annotation.name, "match": held_out[item].name, "distance": distance}
            )
    return kept, {"dropped": dropped, "examples": examples}


# --- work tree, lists and data configs ------------------------------------------------------


def label_path_for(image_path: str | Path) -> Path:
    """Where Ultralytics looks for an image's labels: last `images` folder -> `labels`, .txt."""
    parts = list(Path(image_path).parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        raise ValueError(f"{image_path} has no 'images' folder in its path")
    return Path(*parts).with_suffix(".txt")


def tree_image(work_dir: Path, country: str, name: str) -> Path:
    return Path(work_dir) / country / "images" / name


def write_yolo_tree(
    images_dir: Path,
    work_dir: Path,
    country: str,
    annotations: Iterable[Annotation],
    copy: bool = False,
) -> None:
    """Symlink (or copy) each image into <work>/<country>/images and write its YOLO labels."""
    images_dir = Path(os.path.abspath(images_dir))
    image_out = Path(work_dir) / country / "images"
    label_out = Path(work_dir) / country / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    for a in annotations:
        src, dst = images_dir / a.name, image_out / a.name
        if not (dst.exists() or dst.is_symlink()):
            if copy:
                shutil.copy2(src, dst)
            else:
                try:
                    os.symlink(src, dst)
                except OSError:
                    log.warning("Cannot symlink on this system; copying images instead")
                    copy = True
                    shutil.copy2(src, dst)
        lines = boxes_to_yolo(a.boxes, a.width, a.height)
        label_file = label_out / (Path(a.name).stem + ".txt")
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_name_list(path: Path, names: Iterable[str]) -> None:
    """Committed subset list: image file names, one per line, sorted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{n}\n" for n in sorted(names)))


def read_name_list(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split list not found: {path}; run the prepare step first")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_image_list(path: Path, images: Iterable[str | Path]) -> None:
    """Ultralytics list file: absolute image paths, one per line (machine-specific)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p}\n" for p in images))


def read_image_list(path: str | Path) -> list[Path]:
    return [Path(line.strip()) for line in Path(path).read_text().splitlines() if line.strip()]


def write_data_yaml(
    path: Path, work_dir: Path, lists: dict[str, Path], names: dict[int, str]
) -> None:
    """Ultralytics data config; train, val and test are list files."""
    data = {
        "path": str(work_dir),
        "train": str(lists["train"]),
        "val": str(lists["val"]),
        "test": str(lists["test"]),
        "nc": len(names),
        "names": {int(i): n for i, n in sorted(names.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def annotated_names(work_dir: Path, country: str, names: Iterable[str]) -> list[str]:
    """Names of the images that have at least one box (read from the YOLO label files)."""
    out = []
    for name in names:
        if len(read_yolo_labels(label_path_for(tree_image(work_dir, country, name)))):
            out.append(name)
    return out


def prepare_country(
    cfg: dict[str, Any],
    root: Path,
    country: str,
    splits_dir: Path,
    work_dir: Path,
    workers: int = 1,
    max_train_images: int | None = None,
    copy: bool = False,
) -> dict[str, Any]:
    """Index, split, dedup and write everything for one country; returns its report."""
    rdd = cfg["rdd2022"]
    is_source = country == rdd["source_country"]
    annotations, report = index_country(root, country, cfg)
    images_dir = Path(root) / rdd["countries"][country] / rdd["images_dir"]
    splits = split_annotations(annotations, rdd["split"])
    held_out = splits["val"] + splits["test"]
    pool, dedup = drop_near_duplicates(
        splits["train_pool"], held_out, lambda a: images_dir / a.name, rdd["dedup"], workers
    )
    kept = {"pool": pool, "val": splits["val"], "test": splits["test"]}

    train = pool
    if is_source:
        cap = max_train_images if max_train_images is not None else rdd["yolo"]["max_train_images"]
        if cap and len(pool) > cap:
            sample = random.Random(rdd["split"]["seed"]).sample(pool, cap)
            train = sorted(sample, key=lambda a: a.name)

    write_yolo_tree(images_dir, work_dir, country, pool + held_out, copy=copy)
    prefix = f"rdd_{slug(country)}"
    for split, items in kept.items():
        write_name_list(splits_dir / f"{prefix}_{split}.txt", (a.name for a in items))
    if is_source:
        write_name_list(splits_dir / f"{prefix}_train.txt", (a.name for a in train))

    lists = {}
    for key, items in (("train", train), ("val", splits["val"]), ("test", splits["test"])):
        lists[key] = Path(work_dir) / "lists" / f"{slug(country)}_{key}.txt"
        write_image_list(lists[key], (tree_image(work_dir, country, a.name) for a in items))
    write_data_yaml(splits_dir / f"{prefix}.yaml", Path(work_dir), lists, class_names(cfg))

    report.update(
        {
            "split_sizes": {
                "train_pool": len(pool),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
            },
            "train_images": len(train),
            "near_duplicates_dropped": dedup["dropped"],
            "near_duplicate_examples": dedup["examples"],
        }
    )
    return report


def prepare(
    cfg: dict[str, Any],
    root: str | Path,
    splits_dir: str | Path,
    work_dir: str | Path,
    workers: int = 1,
    countries: Sequence[str] | None = None,
    max_train_images: int | None = None,
    copy: bool = False,
) -> dict[str, Any]:
    """Prepare the source and target countries; writes lists, YAMLs and rdd_report.json."""
    rdd = cfg["rdd2022"]
    todo = list(countries) if countries else [rdd["source_country"], *rdd["target_countries"]]
    root, splits_dir = Path(root), Path(splits_dir)
    work_dir = Path(os.path.abspath(work_dir))
    reports = {}
    for country in todo:
        log.info("Preparing %s", country)
        reports[country] = prepare_country(
            cfg, root, country, splits_dir, work_dir, workers, max_train_images, copy
        )
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "rdd_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


# --- mock data ------------------------------------------------------------------------------

# Per-country look: a different road colour, noise and image size gives the mock a domain gap.
MOCK_STYLES = {
    "Japan": {"bg": (112, 112, 116), "noise": 6, "size": (128, 128)},
    "Norway": {"bg": (176, 180, 196), "noise": 10, "size": (144, 192)},
    "Czech": {"bg": (96, 102, 86), "noise": 8, "size": (128, 160)},
    "United_States": {"bg": (140, 126, 108), "noise": 12, "size": (128, 128)},
}
DEFAULT_STYLE = {"bg": (120, 120, 120), "noise": 8, "size": (128, 128)}


def _draw_damage(
    draw: ImageDraw.ImageDraw, cls: int, width: int, height: int, rng: np.random.Generator
) -> tuple[int, int, int, int]:
    """Draw one synthetic defect and return its box (xmin, ymin, xmax, ymax) in pixels."""
    dark = (28, 28, 30)
    if cls == 0:  # longitudinal crack: thin, mostly vertical
        w, h = int(rng.integers(4, 9)), int(rng.integers(height // 3, height // 2))
        x, y = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
        draw.line([(x + w // 2, y), (x + int(rng.integers(0, w)), y + h)], fill=dark, width=2)
    elif cls == 1:  # transverse crack: thin, mostly horizontal
        w, h = int(rng.integers(width // 3, width // 2)), int(rng.integers(4, 9))
        x, y = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
        draw.line([(x, y + h // 2), (x + w, y + int(rng.integers(0, h)))], fill=dark, width=2)
    elif cls == 2:  # alligator crack: a mesh of crossing lines
        w = int(rng.integers(width // 5, width // 3))
        h = int(rng.integers(height // 5, height // 3))
        x, y = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
        for k in range(1, 4):
            draw.line([(x + k * w // 4, y), (x + k * w // 4, y + h)], fill=dark, width=1)
            draw.line([(x, y + k * h // 4), (x + w, y + k * h // 4)], fill=dark, width=1)
        draw.rectangle([x, y, x + w, y + h], outline=dark)
    else:  # pothole: dark filled ellipse
        w = int(rng.integers(width // 8, width // 4))
        h = int(rng.integers(height // 8, height // 4))
        x, y = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
        draw.ellipse([x, y, x + w, y + h], fill=(18, 18, 20))
    return x, y, x + w, y + h


def _render_mock(
    rng: np.random.Generator, style: dict[str, Any], codes: Sequence[str], extra_codes: bool
) -> tuple[Image.Image, list[tuple[str, tuple[int, int, int, int]]]]:
    """One synthetic road image and its objects as (damage code, pixel box)."""
    height, width = style["size"]
    noise = rng.normal(0, style["noise"], (height, width, 3))
    # Low-frequency texture makes every image (and its perceptual hash) distinct;
    # a flat image would hash to all zeros and look like a duplicate of any other.
    coarse = rng.normal(0, 1, (6, 8)).astype(np.float32)
    blobs = np.asarray(Image.fromarray(coarse).resize((width, height), Image.Resampling.BICUBIC))
    pixels = np.clip(np.array(style["bg"]) + noise + 14 * blobs[..., None], 0, 255)
    image = Image.fromarray(pixels.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    objects = []
    if rng.random() > 0.15:
        for _ in range(int(rng.integers(1, 4))):
            cls = int(rng.integers(0, 4))
            objects.append((codes[cls], _draw_damage(draw, cls, width, height, rng)))
    if extra_codes and rng.random() < 0.1:
        objects.append(("D43", _draw_damage(draw, 1, width, height, rng)))
    return image, objects


def make_mock_rdd(
    root: str | Path,
    cfg: dict[str, Any] | None = None,
    n_images: dict[str, int] | int = 40,
    seed: int = 0,
    n_duplicates: int = 0,
    n_test_images: int = 3,
) -> dict[str, int]:
    """Write a small synthetic RDD2022-like dataset (Pascal-VOC XML, per-country folders).

    Images are noise on a per-country road colour with drawn cracks and potholes: not road
    photos. About 15% have no damage and a few XML objects use an unknown code (D43), so the
    parser's dropping logic gets exercised. Returns the labelled images written per country.
    """
    cfg = cfg or load_config()
    rdd = cfg["rdd2022"]
    codes = [c["code"] for c in sorted(rdd["classes"], key=lambda c: c["id"])]
    rng = np.random.default_rng(seed)
    written = {}
    for country, folder in rdd["countries"].items():
        style = MOCK_STYLES.get(country, DEFAULT_STYLE)
        count = n_images[country] if isinstance(n_images, dict) else n_images
        height, width = style["size"]
        base = Path(root) / folder
        image_dir, xml_dir = base / rdd["images_dir"], base / rdd["annotations_dir"]
        test_dir = base / "test" / "images"
        for folder_path in (image_dir, xml_dir, test_dir):
            folder_path.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            stem = f"{country}_{i:06d}"
            image, objects = _render_mock(rng, style, codes, extra_codes=True)
            image.save(image_dir / f"{stem}.jpg", quality=95)
            body = "".join(
                f"<object><name>{code}</name><pose>Unspecified</pose><truncated>0</truncated>"
                f"<difficult>0</difficult><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
                f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
                for code, b in objects
            )
            (xml_dir / f"{stem}.xml").write_text(
                f"<annotation><folder>{folder}</folder><filename>{stem}.jpg</filename>"
                f"<size><width>{width}</width><height>{height}</height><depth>3</depth></size>"
                f"{body}</annotation>\n"
            )
        for k in range(n_duplicates):  # slightly brightened copies, for the dedup guard
            src = image_dir / f"{country}_{k:06d}.jpg"
            stem = f"{country}_dup{k:03d}"
            with Image.open(src) as img:
                brighter = np.asarray(img, dtype=np.int16) + 5
            Image.fromarray(brighter.clip(0, 255).astype(np.uint8)).save(
                image_dir / f"{stem}.jpg", quality=90
            )
            xml = (xml_dir / f"{country}_{k:06d}.xml").read_text()
            (xml_dir / f"{stem}.xml").write_text(xml.replace(f"{country}_{k:06d}", stem))
        for j in range(n_test_images):  # the real test folder has images only
            image = _render_mock(rng, style, codes, extra_codes=False)[0]
            image.save(test_dir / f"{country}_test{j:03d}.jpg", quality=95)
        written[country] = count + n_duplicates
    return written


# --- command line ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split RDD2022 per country and write YOLO trees, list files and data YAMLs."
    )
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--root", help="RDD2022 folder (default: rdd2022.root in the config)")
    parser.add_argument("--splits-dir", help="list and YAML folder (default: paths.splits)")
    parser.add_argument("--work-dir", help="YOLO work tree (default: rdd2022.work_dir)")
    parser.add_argument("--countries", nargs="+", help="default: source and all targets")
    parser.add_argument("--max-train-images", type=int, help="cap on the source training pool")
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--make-mock", metavar="DIR", help="write a synthetic dataset and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    cwd = Path.cwd()
    if args.make_mock:
        out = resolve_path(args.make_mock, cwd)
        written = make_mock_rdd(out, cfg)
        print(f"Wrote mock RDD2022 to {out}: {written}")
        return

    def pick(arg: str | None, default: str | Path) -> Path:
        return resolve_path(arg, cwd) if arg else resolve_path(default)

    reports = prepare(
        cfg,
        pick(args.root, cfg["rdd2022"]["root"]),
        pick(args.splits_dir, cfg["paths"]["splits"]),
        pick(args.work_dir, cfg["rdd2022"]["work_dir"]),
        args.workers,
        args.countries,
        args.max_train_images,
        args.copy,
    )
    for country, r in reports.items():
        print(
            f"{country}: {r['labelled_images']} labelled images, splits {r['split_sizes']}, "
            f"{r['near_duplicates_dropped']} near-duplicates dropped, "
            f"dropped labels {r['dropped_labels']}"
        )


if __name__ == "__main__":
    main()
