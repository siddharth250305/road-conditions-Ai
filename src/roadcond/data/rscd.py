"""RSCD file lists: balanced train subset, untouched val/test, cross-split leak check.

The official val and test splits are only listed, never sampled or changed (PRD data rules).
A train image that nearly duplicates a val/test image (dHash within max_hamming bits), or
shares a drive id with one, is dropped from the subset and replaced from the same class.

    python -m roadcond.data.rscd --root /path/to/RSCD
"""

import argparse
import csv
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from roadcond.config import load_config, resolve_path

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
SPLIT_FILES = {"train": "rscd_train_subset.csv", "val": "rscd_val.csv", "test": "rscd_test.csv"}
LEAK_REPORT = "rscd_leak_report.json"
CSV_FIELDS = ("path", "friction", "label")
MAX_REPORT_EXAMPLES = 50

_SEPARATORS = re.compile(r"[\s\-]+")
_LEADING_ID = re.compile(r"^\d+[_\-]?")
_TRAILING_ID = re.compile(r"[_\-]?\d+$")


@dataclass(frozen=True)
class Record:
    path: str  # relative to the RSCD root, POSIX style
    friction: str
    label: str  # full RSCD label, e.g. dry_asphalt_smooth
    split: str


def normalize_label(text: str) -> str:
    return _SEPARATORS.sub("_", text.strip().lower()).strip("_")


def to_friction(label: str, classes: Iterable[str]) -> str:
    """Map a full RSCD label (e.g. wet_concrete_slight) to its friction class (wet)."""
    label = normalize_label(label)
    for name in sorted(classes, key=len, reverse=True):
        if label == name or label.startswith(name + "_"):
            return name
    raise ValueError(f"Label {label!r} does not start with a friction class {sorted(classes)}")


def label_from_path(path: Path, split_dir: Path, source: str = "auto") -> str:
    """Full label from the class folder(s) or, for a flat split folder, from the file name.

    "auto" uses the folders when the image sits in a sub-folder of the split, otherwise
    the file name with leading/trailing numeric ids removed (20220101-dry-asphalt-smooth.jpg).
    """
    folders = path.relative_to(split_dir).parent.parts
    if source == "auto":
        source = "folder" if folders else "filename"
    if source == "folder":
        if not folders:
            raise ValueError(f"{path} is not inside a class folder")
        return normalize_label("_".join(folders))
    if source == "filename":
        return normalize_label(_TRAILING_ID.sub("", _LEADING_ID.sub("", path.stem)))
    raise ValueError(f"Unknown label_source {source!r}; use auto, folder or filename")


def index_split(root: str | Path, split: str, rscd_cfg: dict[str, Any]) -> list[Record]:
    """List every image of an official split, sorted by path."""
    root = Path(root)
    split_dir = root / rscd_cfg["split_dirs"][split]
    if not split_dir.is_dir():
        raise FileNotFoundError(f"RSCD {split} folder not found: {split_dir}")
    source = rscd_cfg.get("label_source", "auto")
    records = []
    for path in sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        label = label_from_path(path, split_dir, source)
        try:
            friction = to_friction(label, rscd_cfg["classes"])
        except ValueError as err:
            raise ValueError(f"{path}: {err}") from None
        records.append(Record(path.relative_to(root).as_posix(), friction, label, split))
    if not records:
        raise ValueError(f"No images found under {split_dir}")
    return records


def dhash(path: str | Path, hash_size: int = 8) -> int:
    """Difference hash with hash_size**2 bits; stable under small brightness/JPEG changes."""
    with Image.open(path) as img:
        small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    value = 0
    for bit in (pixels[:, 1:] > pixels[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return value


def hash_images(paths: Sequence[Path], hash_size: int, workers: int = 1) -> list[int]:
    fn = partial(dhash, hash_size=hash_size)
    if workers <= 1 or len(paths) < 256:
        return [fn(p) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, paths, chunksize=256))


class NearDuplicateIndex:
    """Finds stored hashes within max_distance bits of a query.

    The bits are split into max_distance + 1 bands. Two hashes within max_distance bits
    agree exactly on at least one band (pigeonhole), so only bucket-mates are compared.
    Identical hashes are stored once, which keeps low-texture images (snow) cheap.
    """

    def __init__(self, hashes: Iterable[int], n_bits: int, max_distance: int):
        if not 0 <= max_distance < n_bits:
            raise ValueError(f"max_distance must be in [0, {n_bits}), got {max_distance}")
        edges = np.linspace(0, n_bits, max_distance + 2).round().astype(int)
        self._bands = [
            (int(lo), int(hi - lo)) for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        self.max_distance = max_distance
        self._items: dict[int, list[int]] = defaultdict(list)  # unique hash -> item ids
        for item, h in enumerate(hashes):
            self._items[h].append(item)
        self._tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in self._bands]
        for h in self._items:
            for table, key in zip(self._tables, self._keys(h), strict=True):
                table[key].append(h)

    def _keys(self, h: int) -> list[int]:
        return [(h >> lo) & ((1 << width) - 1) for lo, width in self._bands]

    def query(self, h: int) -> list[tuple[int, int]]:
        """(item id, distance) of every stored hash within max_distance, closest first."""
        candidates: set[int] = set()
        for table, key in zip(self._tables, self._keys(h), strict=True):
            candidates.update(table.get(key, ()))
        matches = []
        for other in candidates:
            distance = (h ^ other).bit_count()
            if distance <= self.max_distance:
                matches.extend((item, distance) for item in self._items[other])
        return sorted(matches, key=lambda m: (m[1], m[0]))


def drive_id(path: str, pattern: str) -> str | None:
    match = re.search(pattern, Path(path).name)
    if match is None:
        return None
    return match.group(1) if match.groups() else match.group(0)


class LeakChecker:
    """Rejects train records that nearly duplicate, or share a drive with, a held-out image."""

    def __init__(
        self,
        root: str | Path,
        held_out: Sequence[Record],
        dedup_cfg: dict[str, Any],
        workers: int = 1,
    ):
        self.root = Path(root)
        self.held_out = list(held_out)
        self.hash_size = dedup_cfg["hash_size"]
        self.workers = workers
        self.hashes = hash_images(
            [self.root / r.path for r in self.held_out], self.hash_size, workers
        )
        self.index = NearDuplicateIndex(self.hashes, self.hash_size**2, dedup_cfg["max_hamming"])
        self.pattern = dedup_cfg.get("drive_id_regex")
        self.held_out_drives = (
            {drive_id(r.path, self.pattern) for r in self.held_out} - {None}
            if self.pattern
            else set()
        )
        self.checked = 0
        self.n_near_duplicate = 0
        self.n_shared_drive = 0
        self.examples: list[dict[str, Any]] = []

    def __call__(self, batch: Sequence[Record]) -> set[str]:
        """Return the paths in batch that must not enter the train subset."""
        self.checked += len(batch)
        rejected = set()
        if self.pattern:
            for r in batch:
                if drive_id(r.path, self.pattern) in self.held_out_drives:
                    rejected.add(r.path)
                    self.n_shared_drive += 1
        todo = [r for r in batch if r.path not in rejected]
        hashes = hash_images([self.root / r.path for r in todo], self.hash_size, self.workers)
        for record, h in zip(todo, hashes, strict=True):
            matches = self.index.query(h)
            if not matches:
                continue
            rejected.add(record.path)
            self.n_near_duplicate += 1
            if len(self.examples) < MAX_REPORT_EXAMPLES:
                item, distance = matches[0]
                match = self.held_out[item]
                self.examples.append(
                    {
                        "train": record.path,
                        "match": match.path,
                        "match_split": match.split,
                        "distance": distance,
                        "n_matches": len(matches),
                    }
                )
        return rejected


@dataclass
class SubsetResult:
    records: list[Record]
    rejected: list[Record] = field(default_factory=list)
    shortfall: dict[str, int] = field(default_factory=dict)


def class_quotas(train_size: int, classes: Sequence[str]) -> dict[str, int]:
    """Split train_size over the classes as evenly as possible (earlier classes get +1)."""
    base, extra = divmod(train_size, len(classes))
    return {c: base + (i < extra) for i, c in enumerate(classes)}


def interleave_by_label(records: Sequence[Record], rng: random.Random) -> list[Record]:
    """Shuffle within each full label, then round-robin the labels.

    Taking a prefix of the result spreads a class quota evenly over its materials and
    unevenness levels instead of following the raw (asphalt-heavy) distribution.
    """
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.label].append(r)
    lists = []
    for label in sorted(groups):
        group = list(groups[label])
        rng.shuffle(group)
        lists.append(group)
    rng.shuffle(lists)
    longest = max((len(g) for g in lists), default=0)
    return [g[i] for i in range(longest) for g in lists if i < len(g)]


def balanced_subset(
    train: Sequence[Record],
    classes: Sequence[str],
    train_size: int,
    seed: int,
    reject: Callable[[Sequence[Record]], set[str]] | None = None,
) -> SubsetResult:
    """Draw a class-balanced subset; records named by reject() are skipped and replaced."""
    rng = random.Random(seed)
    by_class: dict[str, list[Record]] = defaultdict(list)
    for r in train:
        by_class[r.friction].append(r)
    result = SubsetResult(records=[])
    for name, quota in class_quotas(train_size, classes).items():
        order = interleave_by_label(by_class[name], rng)
        taken: list[Record] = []
        pos = 0
        while len(taken) < quota and pos < len(order):
            batch = order[pos : pos + quota - len(taken)]
            pos += len(batch)
            bad = reject(batch) if reject else set()
            taken += [r for r in batch if r.path not in bad]
            result.rejected += [r for r in batch if r.path in bad]
        if len(taken) < quota:
            result.shortfall[name] = quota - len(taken)
            log.warning("Class %s: only %d of %d images available", name, len(taken), quota)
        result.records += taken
    return result


def write_split_csv(records: Iterable[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        writer.writerows((r.path, r.friction, r.label) for r in records)


def read_split_csv(path: str | Path, split: str) -> list[Record]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}; run `python -m roadcond.data.rscd` first"
        )
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return [Record(row["path"], row["friction"], row["label"], split) for row in rows]


def load_splits(splits_dir: str | Path) -> dict[str, list[Record]]:
    """Saved train subset, val and test lists, keyed by split."""
    return {s: read_split_csv(Path(splits_dir) / name, s) for s, name in SPLIT_FILES.items()}


def _shared_file_names(records: dict[str, list[Record]]) -> dict[str, int]:
    names = {s: {Path(r.path).name for r in records[s]} for s in SPLITS}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    return {f"{a}/{b}": len(names[a] & names[b]) for a, b in pairs}


def prepare(
    cfg: dict[str, Any],
    root: str | Path,
    splits_dir: str | Path,
    workers: int = 1,
    train_size: int | None = None,
) -> dict[str, Any]:
    """Index RSCD, write the three split files and the leak report; return the report."""
    rscd = cfg["rscd"]
    classes = rscd["classes"]
    root, splits_dir = Path(root), Path(splits_dir)
    records = {s: index_split(root, s, rscd) for s in SPLITS}
    for s in SPLITS:
        log.info("%s: %d images", s, len(records[s]))

    log.info("Hashing %d val/test images", len(records["val"]) + len(records["test"]))
    checker = LeakChecker(root, records["val"] + records["test"], rscd["dedup"], workers)
    size = train_size or rscd["subset"]["train_size"]
    seed = rscd["subset"]["seed"]
    result = balanced_subset(records["train"], classes, size, seed, reject=checker)

    # Report-only: val/test overlap cannot be fixed without touching the official splits.
    n_val = len(records["val"])
    val_index = NearDuplicateIndex(
        checker.hashes[:n_val], checker.hash_size**2, checker.index.max_distance
    )
    val_test = sum(1 for h in checker.hashes[n_val:] if val_index.query(h))

    write_split_csv(result.records, splits_dir / SPLIT_FILES["train"])
    write_split_csv(records["val"], splits_dir / SPLIT_FILES["val"])
    write_split_csv(records["test"], splits_dir / SPLIT_FILES["test"])

    subset_counts = Counter(r.friction for r in result.records)
    report = {
        "seed": seed,
        "train_size_requested": size,
        "train_size": len(result.records),
        "split_sizes": {s: len(records[s]) for s in SPLITS},
        "subset_per_class": {c: subset_counts[c] for c in classes},
        "shortfall": result.shortfall,
        "dedup": {"hash": "dhash", **rscd["dedup"]},
        "train_candidates_checked": checker.checked,
        "train_rejected_near_duplicate": checker.n_near_duplicate,
        "train_rejected_shared_drive": checker.n_shared_drive,
        "test_images_near_duplicate_of_val": val_test,
        "shared_file_names": _shared_file_names(records),
        "examples": checker.examples,
        "full_labels": {
            s: dict(Counter(r.label for r in records[s]).most_common()) for s in SPLITS
        },
    }
    (splits_dir / LEAK_REPORT).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write the RSCD split files (balanced train subset, val, test) and leak report."
    )
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--root", help="RSCD folder (default: rscd.root in the config)")
    parser.add_argument("--splits-dir", help="output folder (default: paths.splits)")
    parser.add_argument("--train-size", type=int, help="override rscd.subset.train_size")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    cwd = Path.cwd()
    root = resolve_path(args.root, cwd) if args.root else resolve_path(cfg["rscd"]["root"])
    splits_dir = (
        resolve_path(args.splits_dir, cwd)
        if args.splits_dir
        else resolve_path(cfg["paths"]["splits"])
    )
    report = prepare(cfg, root, splits_dir, args.workers, args.train_size)
    summary = {k: v for k, v in report.items() if k not in ("examples", "full_labels")}
    print(json.dumps(summary, indent=2))
    print(f"Wrote split files and {LEAK_REPORT} to {splits_dir}")


if __name__ == "__main__":
    main()
