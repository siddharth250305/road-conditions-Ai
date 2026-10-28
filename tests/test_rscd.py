import json
import random
from pathlib import Path

import pytest
from PIL import Image

from roadcond.config import load_config
from roadcond.data.mock import RSCD_LABELS, make_mock_rscd
from roadcond.data.rscd import (
    LEAK_REPORT,
    SPLITS,
    NearDuplicateIndex,
    Record,
    balanced_subset,
    class_quotas,
    dhash,
    index_split,
    label_from_path,
    load_splits,
    prepare,
    to_friction,
)

CFG = load_config()
CLASSES = CFG["rscd"]["classes"]
PER_LABEL = (6, 2, 2)  # train, val, test images per full label


@pytest.fixture(scope="module")
def mock_rscd(tmp_path_factory):
    root = tmp_path_factory.mktemp("rscd")
    leaks = make_mock_rscd(root, CFG["rscd"]["split_dirs"], n_per_label=PER_LABEL, n_leaks=2)
    return root, leaks


@pytest.mark.parametrize(
    ("label", "friction"),
    [
        ("dry_asphalt_smooth", "dry"),
        ("wet-concrete-slight", "wet"),
        ("water_gravel", "water"),
        ("fresh_snow", "fresh_snow"),
        ("Melted Snow", "melted_snow"),
        ("ice", "ice"),
    ],
)
def test_to_friction(label, friction):
    assert to_friction(label, CLASSES) == friction


@pytest.mark.parametrize("label", ["dryer_asphalt", "slush", "snow"])
def test_to_friction_rejects_unknown(label):
    with pytest.raises(ValueError):
        to_friction(label, CLASSES)


def test_label_from_path():
    split = Path("/r/train")
    assert label_from_path(split / "dry_asphalt_smooth" / "001.jpg", split) == "dry_asphalt_smooth"
    flat = Path("/r/test_50k")
    path = flat / "20220101000000001-wet-mud.jpg"
    assert label_from_path(path, flat) == "wet_mud"
    assert label_from_path(flat / "ice_000123.jpg", flat, "filename") == "ice"


def test_index_split_reads_both_layouts(mock_rscd):
    root, leaks = mock_rscd
    rscd = CFG["rscd"]
    train = index_split(root, "train", rscd)
    test = index_split(root, "test", rscd)
    assert len(train) == PER_LABEL[0] * len(RSCD_LABELS) + len(leaks)
    assert len(test) == PER_LABEL[2] * len(RSCD_LABELS)
    assert {r.label for r in test} == set(RSCD_LABELS)
    assert {r.friction for r in train} == set(CLASSES)


def test_index_split_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_split(tmp_path, "train", CFG["rscd"])


def test_class_quotas_sum_exactly():
    quotas = class_quotas(40000, CLASSES)
    assert sum(quotas.values()) == 40000
    assert max(quotas.values()) - min(quotas.values()) <= 1


def test_balanced_subset_spreads_over_sub_labels():
    train = [
        Record(f"{label}/{i}.jpg", to_friction(label, CLASSES), label, "train")
        for label in RSCD_LABELS
        for i in range(10)
    ]
    result = balanced_subset(train, CLASSES, 60, seed=0)
    dry = [r for r in result.records if r.friction == "dry"]
    assert len(dry) == 10
    # dry has 8 sub-labels and a quota of 10: the round-robin covers all of them.
    assert len({r.label for r in dry}) == 8
    again = balanced_subset(train, CLASSES, 60, seed=0)
    assert [r.path for r in again.records] == [r.path for r in result.records]
    other = balanced_subset(train, CLASSES, 60, seed=1)
    assert [r.path for r in other.records] != [r.path for r in result.records]


def test_balanced_subset_reports_shortfall():
    train = [Record(f"ice/{i}.jpg", "ice", "ice", "train") for i in range(3)]
    result = balanced_subset(train, CLASSES, 12, seed=0)
    assert "ice" not in result.shortfall
    assert result.shortfall["dry"] == 2


def test_near_duplicate_index_matches_brute_force():
    rng = random.Random(0)
    stored = [rng.getrandbits(64) for _ in range(300)]
    stored += [stored[0] ^ (1 << 5), stored[1] ^ 0b111]  # planted neighbours
    index = NearDuplicateIndex(stored, 64, 4)
    for query in stored[:20] + [rng.getrandbits(64) for _ in range(20)]:
        expected = sorted(
            (i, (query ^ h).bit_count())
            for i, h in enumerate(stored)
            if (query ^ h).bit_count() <= 4
        )
        assert sorted(index.query(query)) == expected


def test_dhash_survives_reencode_and_brightness(mock_rscd, tmp_path):
    root, _ = mock_rscd
    src = next((root / CFG["rscd"]["split_dirs"]["test"]).glob("*.jpg"))
    with Image.open(src) as img:
        brighter = img.point(lambda v: min(255, v + 6))
        brighter.save(tmp_path / "copy.jpg", quality=85)
    assert (dhash(src) ^ dhash(tmp_path / "copy.jpg")).bit_count() <= 4


def test_prepare_writes_balanced_subset_and_untouched_splits(mock_rscd, tmp_path):
    root, _ = mock_rscd
    report = prepare(CFG, root, tmp_path, train_size=24)
    saved = load_splits(tmp_path)
    assert report["subset_per_class"] == dict.fromkeys(CLASSES, 4)
    assert len(saved["train"]) == 24
    for split in ("val", "test"):
        official = index_split(root, split, CFG["rscd"])
        assert [r.path for r in saved[split]] == [r.path for r in official]
    held_out = {r.path for r in saved["val"] + saved["test"]}
    assert not held_out & {r.path for r in saved["train"]}
    assert json.loads((tmp_path / LEAK_REPORT).read_text())["seed"] == 0
    again = prepare(CFG, root, tmp_path / "again", train_size=24)
    assert again["subset_per_class"] == report["subset_per_class"]
    assert [r.path for r in load_splits(tmp_path / "again")["train"]] == [
        r.path for r in saved["train"]
    ]


def test_prepare_rejects_planted_leaks(mock_rscd, tmp_path):
    root, leaks = mock_rscd
    # Ask for more than exists, so every train image is checked.
    report = prepare(CFG, root, tmp_path, train_size=10_000)
    subset = {r.path for r in load_splits(tmp_path)["train"]}
    assert report["train_candidates_checked"] == len(index_split(root, "train", CFG["rscd"]))
    assert report["train_rejected_near_duplicate"] == len(leaks)
    assert not subset & set(leaks)
    assert {ex["train"] for ex in report["examples"]} == set(leaks)
    assert all(ex["match_split"] == "test" for ex in report["examples"])
    assert report["shortfall"]  # 10k requested from a tiny mock set
    assert set(report["split_sizes"]) == set(SPLITS)
