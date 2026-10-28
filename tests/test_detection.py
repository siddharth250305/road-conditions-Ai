import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

# Ultralytics writes settings on import; keep them out of the user's home during tests.
os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.mkdtemp(prefix="yolo_cfg_"))

from roadcond.config import load_config  # noqa: E402
from roadcond.data.rdd import (  # noqa: E402
    Annotation,
    Box,
    annotated_names,
    boxes_to_yolo,
    class_ids,
    drop_near_duplicates,
    index_country,
    label_path_for,
    make_mock_rdd,
    parse_voc_xml,
    parse_yolo_text,
    prepare,
    read_image_list,
    read_name_list,
    read_yolo_labels,
    split_annotations,
    tree_image,
    yolo_to_boxes,
    yolo_to_xyxy,
)
from roadcond.detection.few_shot import fewshot_epochs, sample_budgets  # noqa: E402
from roadcond.detection.metrics import (  # noqa: E402
    box_iou,
    f1_at_conf,
    format_curve_table,
    gap_recovered,
    learning_curve,
    match_boxes,
    prf1,
    relative_gap,
    summarize,
)
from roadcond.detection.pseudo_label import pseudo_rows  # noqa: E402

CFG = load_config()
IDS = class_ids(CFG)

SAMPLE_XML = """<annotation><folder>Japan</folder><filename>a.jpg</filename>
<size><width>200</width><height>100</height><depth>3</depth></size>
<object><name>D00</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object>
<object><name>D40</name><bndbox><xmin>-5</xmin><ymin>80</ymin><xmax>300</xmax><ymax>120</ymax></bndbox></object>
<object><name>D43</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
<object><name>D10</name><bndbox><xmin>30</xmin><ymin>30</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
<object><name>D20</name><bndbox><xmin>x</xmin></bndbox></object>
</annotation>"""

ONE_BOX_XML = (
    "<annotation><object><name>D00</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
    "<xmax>9</xmax><ymax>9</ymax></bndbox></object></annotation>"
)


# --- annotation parsing ---------------------------------------------------------------------


def test_class_ids_match_the_prd():
    assert IDS == {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


def test_parse_voc_xml_keeps_known_classes_and_clamps():
    width, height, boxes, dropped = parse_voc_xml(SAMPLE_XML, IDS)
    assert (width, height) == (200, 100)
    assert boxes == [Box(0, 10, 20, 50, 60), Box(3, 0, 80, 200, 100)]  # D40 clamped to the image
    assert dropped == {"D43": 1, "<degenerate>": 1, "<bad bndbox>": 1}


def test_parse_voc_xml_reads_size_from_the_image_when_missing(tmp_path):
    Image.new("RGB", (64, 32)).save(tmp_path / "a.jpg")
    assert parse_voc_xml(ONE_BOX_XML, IDS, tmp_path / "a.jpg")[:2] == (64, 32)
    with pytest.raises(ValueError, match="no image size"):
        parse_voc_xml(ONE_BOX_XML, IDS)


@pytest.mark.parametrize(
    "xml",
    [
        '<!DOCTYPE a [<!ENTITY x "y">]><annotation/>',
        '<!entity x "y"><annotation/>',
        "<annotation><size>",
    ],
)
def test_parse_voc_xml_rejects_entities_and_bad_xml(xml):
    with pytest.raises(ValueError):
        parse_voc_xml(xml, IDS)


def test_yolo_round_trip():
    boxes = [Box(0, 10, 20, 50, 60), Box(3, 0, 80, 200, 100)]
    lines = boxes_to_yolo(boxes, 200, 100)
    assert lines[0] == "0 0.150000 0.400000 0.200000 0.400000"
    rows = parse_yolo_text("\n".join(lines) + "\n")
    assert rows.shape == (2, 5)
    for a, b in zip(yolo_to_boxes(rows, 200, 100), boxes, strict=True):
        assert a.cls == b.cls
        np.testing.assert_allclose(
            [a.xmin, a.ymin, a.xmax, a.ymax], [b.xmin, b.ymin, b.xmax, b.ymax], atol=1e-3
        )
    np.testing.assert_allclose(yolo_to_xyxy(rows)[0], [0.05, 0.2, 0.25, 0.6])


@pytest.mark.parametrize("text", ["0 0.5 0.5 0.1", "a 0.5 0.5 0.1 0.1"])
def test_parse_yolo_text_rejects_malformed_lines(text):
    with pytest.raises(ValueError):
        parse_yolo_text(text)


def test_read_yolo_labels_missing_and_empty(tmp_path):
    assert read_yolo_labels(tmp_path / "missing.txt").shape == (0, 5)
    (tmp_path / "empty.txt").write_text("")
    assert read_yolo_labels(tmp_path / "empty.txt").shape == (0, 5)


def test_label_path_for_swaps_only_the_last_images_folder():
    assert label_path_for("/w/Japan/images/a.jpg") == Path("/w/Japan/labels/a.txt")
    assert label_path_for("/images/Japan/images/a.jpeg") == Path("/images/Japan/labels/a.txt")
    with pytest.raises(ValueError):
        label_path_for("/w/Japan/pics/a.jpg")


# --- mock data, splits, data configs --------------------------------------------------------


@pytest.fixture(scope="module")
def mock_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("rdd")
    make_mock_rdd(root, CFG, n_images=40, n_duplicates=2)
    return root


def test_mock_dataset_layout_and_boxes(mock_root):
    for folder in CFG["rdd2022"]["countries"].values():
        base = mock_root / folder
        assert len(list((base / "train" / "images").glob("*.jpg"))) == 42
        assert len(list((base / "train" / "annotations" / "xmls").glob("*.xml"))) == 42
        assert (
            len(list((base / "test" / "images").glob("*.jpg"))) == 3
        )  # images only, as in RDD2022
    annotations, _ = index_country(mock_root, "Norway", CFG)
    assert all(
        0 <= b.xmin < b.xmax <= a.width and 0 <= b.ymin < b.ymax <= a.height
        for a in annotations
        for b in a.boxes
    )


def test_mock_countries_differ_in_look(mock_root):
    means = {}
    for country, folder in CFG["rdd2022"]["countries"].items():
        path = next((mock_root / folder / "train" / "images").glob("*.jpg"))
        means[country] = np.asarray(Image.open(path), dtype=float).mean(axis=(0, 1))
    assert not np.allclose(means["Japan"], means["Norway"], atol=15)  # a domain gap to learn


def test_index_country_report(mock_root):
    annotations, report = index_country(mock_root, "Japan", CFG)
    assert len(annotations) == report["labelled_images"] == 42
    assert report["images_without_xml"] == report["xml_without_image"] == 0
    assert set(report["boxes_per_class"]) == {"D00", "D10", "D20", "D40"}
    assert sum(report["boxes_per_class"].values()) == sum(len(a.boxes) for a in annotations)
    assert report["images_with_damage"] < report["labelled_images"]  # some images show no damage


def test_index_country_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Japan"):
        index_country(tmp_path, "Japan", CFG)


def _annotations(n):
    return [Annotation(f"img_{i:03d}.jpg", 10, 10, [Box(0, 1, 1, 5, 5)]) for i in range(n)]


def test_split_is_seeded_disjoint_and_complete():
    items = _annotations(100)
    split = {"seed": 0, "train_pool": 0.7, "val": 0.1, "test": 0.2, "max_val_images": 500}
    a, b = split_annotations(items, split), split_annotations(items, split)
    assert [x.name for x in a["test"]] == [x.name for x in b["test"]]
    assert (len(a["train_pool"]), len(a["val"]), len(a["test"])) == (70, 10, 20)
    names = [x.name for part in a.values() for x in part]
    assert len(names) == len(set(names)) == 100
    other = split_annotations(items, {**split, "seed": 1})
    assert [x.name for x in other["test"]] != [x.name for x in a["test"]]


def test_split_caps_val_and_moves_the_excess_to_the_pool():
    split = {"seed": 0, "train_pool": 0.5, "val": 0.3, "test": 0.2, "max_val_images": 5}
    parts = split_annotations(_annotations(100), split)
    assert len(parts["val"]) == 5 and len(parts["test"]) == 20 and len(parts["train_pool"]) == 75
    with pytest.raises(ValueError, match="sum to 1"):
        split_annotations(_annotations(10), {**split, "val": 0.5})


def test_dedup_guard_drops_pool_images_that_copy_held_out_images(mock_root):
    annotations, _ = index_country(mock_root, "Japan", CFG)
    by_name = {a.name: a for a in annotations}
    original, dup = by_name["Japan_000000.jpg"], by_name["Japan_dup000.jpg"]
    others = [a for a in annotations if a.name not in (original.name, dup.name)][:6]
    images = mock_root / "Japan" / "train" / "images"
    kept, report = drop_near_duplicates(
        [dup, *others], [original], lambda a: images / a.name, CFG["rdd2022"]["dedup"]
    )
    assert dup.name not in {a.name for a in kept}
    assert report["dropped"] == 1 and report["examples"][0]["pool"] == dup.name
    assert len(kept) == len(others)  # the others are not near-duplicates of it


def test_prepare_writes_lists_yaml_and_yolo_tree(mock_root, tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["rdd2022"]["yolo"]["max_train_images"] = 15
    splits, work = tmp_path / "splits", tmp_path / "work"
    reports = prepare(cfg, mock_root, splits, work, countries=["Japan", "Norway"])
    assert set(reports) == {"Japan", "Norway"}
    assert reports["Japan"]["near_duplicates_dropped"] >= 1  # the planted brightened copies

    for country, slug in (("Japan", "japan"), ("Norway", "norway")):
        lists = {s: read_name_list(splits / f"rdd_{slug}_{s}.txt") for s in ("pool", "val", "test")}
        everything = [n for names in lists.values() for n in names]
        assert len(everything) == len(set(everything))  # no image in two splits
        for name in everything:
            image = tree_image(work, country, name)
            assert image.exists() and label_path_for(image).is_file()
        data = yaml.safe_load((splits / f"rdd_{slug}.yaml").read_text())
        assert data["nc"] == 4 and data["names"] == {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        for key in ("train", "val", "test"):
            assert Path(data[key]).is_file() and len(read_image_list(data[key])) > 0
        assert [p.name for p in read_image_list(data["test"])] == lists["test"]
    assert len(read_name_list(splits / "rdd_japan_train.txt")) == 15  # source pool is capped
    assert not (splits / "rdd_norway_train.txt").exists()  # targets keep the whole pool
    assert json.loads((splits / "rdd_report.json").read_text())["Japan"]["train_images"] == 15

    # Labels in the work tree equal the parsed XML (Norway mock images are 144 x 192).
    name = lists["test"][0]
    xml = mock_root / "Norway" / "train" / "annotations" / "xmls" / f"{Path(name).stem}.xml"
    expected = parse_voc_xml(xml, IDS)[2]
    saved = yolo_to_boxes(
        read_yolo_labels(label_path_for(tree_image(work, "Norway", name))), 192, 144
    )
    assert [b.cls for b in saved] == [b.cls for b in expected]
    annotated = annotated_names(work, "Norway", lists["pool"])
    assert 0 < len(annotated) <= len(lists["pool"])


# --- metrics on hand-made boxes -------------------------------------------------------------


def test_box_iou_values():
    a = np.array([[0, 0, 2, 2]])
    b = np.array([[0, 0, 2, 2], [1, 0, 3, 2], [5, 5, 6, 6]])
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 1 / 3, 0.0]])
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 3)
    np.testing.assert_allclose(box_iou([[0, 0, 0, 0]], [[0, 0, 0, 0]]), [[0.0]])  # empty boxes


def test_iou_ignores_per_axis_scaling():
    a, b = np.array([[0.0, 0, 4, 2]]), np.array([[1.0, 0, 5, 2]])
    scale = np.array([4.0, 2, 4, 2])
    np.testing.assert_allclose(box_iou(a / scale, b / scale), box_iou(a, b))


def test_match_boxes_is_greedy_by_score_and_class_aware():
    gt = np.array([[0, 0, 10, 10]])
    preds = np.array([[0, 0, 10, 10], [1, 1, 10, 10]])
    # Both overlap the single ground truth; only the higher-scoring one may claim it.
    assert match_boxes(preds, [0.9, 0.5], [0, 0], gt, [0]) == (1, 1, 0)
    assert match_boxes(preds, [0.5, 0.9], [0, 0], gt, [0]) == (1, 1, 0)
    # A perfect box of the wrong class does not match.
    assert match_boxes(preds[:1], [0.9], [1], gt, [0]) == (0, 1, 1)
    assert match_boxes(np.zeros((0, 4)), [], [], gt, [0]) == (0, 0, 1)
    assert match_boxes(preds, [0.9, 0.5], [0, 0], np.zeros((0, 4)), []) == (0, 2, 0)


def test_prf1_handles_zero_denominators():
    assert prf1(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    out = prf1(2, 2, 6)
    assert out["precision"] == 0.5 and out["recall"] == 0.25
    assert out["f1"] == pytest.approx(2 * 0.5 * 0.25 / 0.75)


def test_f1_at_conf_hand_example():
    preds = [
        {"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "scores": [0.9, 0.3], "cls": [0, 1]},
        {"boxes": [[0, 0, 5, 5]], "scores": [0.6], "cls": [3]},
    ]
    truths = [
        {"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "cls": [0, 1]},
        {"boxes": [[50, 50, 60, 60]], "cls": [3]},
    ]
    low = f1_at_conf(preds, truths, conf=0.25)  # keeps all 3: 2 hits, 1 false alarm, 1 miss
    assert (low["tp"], low["fp"], low["fn"]) == (2, 1, 1)
    assert low["f1"] == pytest.approx(2 / 3)
    high = f1_at_conf(preds, truths, conf=0.5)  # drops the 0.3 box: 1 hit, 1 false alarm, 2 misses
    assert (high["tp"], high["fp"], high["fn"]) == (1, 1, 2)
    assert high["precision"] == 0.5 and high["recall"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        f1_at_conf(preds, truths[:1], 0.5)


def test_relative_gap_and_gap_recovered():
    assert relative_gap(0.8, 0.6) == pytest.approx(0.25)
    assert relative_gap(0.5, 0.6) == pytest.approx(-0.2)  # target better than source
    assert relative_gap(0.0, 0.1) is None
    assert gap_recovered(0.8, 0.6, 0.6) == pytest.approx(0.0)
    assert gap_recovered(0.8, 0.6, 0.7) == pytest.approx(0.5)
    assert gap_recovered(0.8, 0.6, 0.8) == pytest.approx(1.0)
    assert gap_recovered(0.8, 0.6, 0.9) == pytest.approx(1.5)  # can overshoot the source
    assert gap_recovered(0.5, 0.6, 0.7) is None  # no gap to recover
    assert gap_recovered(0.5, 0.5, 0.7) is None


def test_summarize_ignores_none_and_needs_two_values_for_a_spread():
    out = summarize([0.2, 0.4, 0.6])
    assert out == {"mean": pytest.approx(0.4), "std": pytest.approx(0.2), "n": 3}
    assert summarize([0.5])["std"] is None
    assert summarize([None, 0.5, None, 0.7])["n"] == 2
    assert summarize([None]) == {"mean": None, "std": None, "n": 0}


def test_learning_curve_aggregates_seeds_and_adds_budget_zero():
    runs = [
        {"country": "Norway", "budget": 10, "seed": s, "map50": m, "map50_95": m / 2, "f1": m}
        for s, m in enumerate([0.5, 0.6, 0.7])
    ] + [{"country": "Norway", "budget": 50, "seed": 0, "map50": 0.8, "map50_95": 0.4, "f1": 0.8}]
    zero = {"Norway": {"map50": 0.4, "map50_95": 0.2, "f1": 0.4}}
    source = {"map50": 0.8, "map50_95": 0.4, "f1": 0.8}
    curve = learning_curve(runs, zero, source)["Norway"]
    assert sorted(curve) == [0, 10, 50]
    assert curve[0]["map50"] == {"mean": 0.4, "std": None, "n": 1}
    assert curve[0]["recovered"]["mean"] == pytest.approx(0.0)
    assert curve[0]["relative_gap"]["mean"] == pytest.approx(0.5)
    assert curve[10]["map50"]["mean"] == pytest.approx(0.6)
    assert curve[10]["map50"]["std"] == pytest.approx(0.1)
    assert curve[10]["recovered"]["mean"] == pytest.approx(0.5)  # (0.6 - 0.4) / (0.8 - 0.4)
    assert curve[50]["recovered"]["mean"] == pytest.approx(1.0)
    table = format_curve_table({"Norway": curve})
    assert "Norway" in table and "0.600 ± 0.100" in table and "50.0%" in table


def test_learning_curve_reports_no_recovery_when_there_is_no_gap():
    runs = [{"country": "Czech", "budget": 10, "seed": 0, "map50": 0.9, "map50_95": 0.5, "f1": 0.9}]
    zero = {"Czech": {"map50": 0.85, "map50_95": 0.45, "f1": 0.85}}
    curve = learning_curve(runs, zero, {"map50": 0.8, "map50_95": 0.4, "f1": 0.8})["Czech"]
    assert curve[10]["recovered"]["mean"] is None
    assert "n/a" in format_curve_table({"Czech": curve})


# --- augmentation ---------------------------------------------------------------------------


def test_weather_augmentation_changes_pixels_but_never_boxes():
    pytest.importorskip("albumentations")
    from roadcond.detection.augment import apply_to_sample, weather_transforms

    rng = np.random.default_rng(0)
    image = rng.integers(60, 200, (96, 128, 3), dtype=np.uint8)
    rows = [[0, 0.5, 0.5, 0.2, 0.3], [3, 0.2, 0.7, 0.1, 0.1]]
    changed = 0
    for seed in range(10):
        out, boxes = apply_to_sample(image, rows, strength=2.0, seed=seed)
        np.testing.assert_allclose(boxes, np.array(rows), atol=1e-6)
        assert out.shape == image.shape and out.dtype == np.uint8
        changed += not np.array_equal(out, image)
    assert changed >= 8
    first, _ = apply_to_sample(image, rows, 2.0, seed=3)
    again, _ = apply_to_sample(image, rows, 2.0, seed=3)
    np.testing.assert_array_equal(first, again)  # seeded: the same draw
    assert weather_transforms(0) == []
    untouched, _ = apply_to_sample(image, rows, 0, seed=0)
    np.testing.assert_array_equal(untouched, image)


def test_weather_augmentation_can_be_saved_in_checkpoints():
    A = pytest.importorskip("albumentations")
    from roadcond.detection.augment import weather_transforms

    spec = A.to_dict(A.Compose(weather_transforms(1.0)))  # what Ultralytics stores
    assert len(A.from_dict(spec).transforms) == 3


# --- budgets and pseudo-labels --------------------------------------------------------------


def test_budget_subsets_are_nested_seeded_and_validated():
    names = [f"img_{i:03d}.jpg" for i in range(300)]
    subsets = sample_budgets(names, [10, 50, 200], seed=0)
    assert [len(subsets[n]) for n in (10, 50, 200)] == [10, 50, 200]
    assert set(subsets[10]) < set(subsets[50]) < set(subsets[200])
    assert subsets == sample_budgets(names, [10, 50, 200], seed=0)
    assert subsets[10] != sample_budgets(names, [10], seed=1)[10]
    independent = sample_budgets(names, [10, 50], seed=0, nested=False)
    assert len(independent[10]) == 10 and len(independent[50]) == 50
    with pytest.raises(ValueError, match="larger than"):
        sample_budgets(names[:20], [50], seed=0)


def test_fewshot_epochs_give_tiny_budgets_enough_updates():
    assert fewshot_epochs(10, 8, epochs=30, min_iterations=100) == 50  # 2 steps per epoch
    assert fewshot_epochs(200, 8, epochs=30, min_iterations=100) == 30  # already 750 steps
    assert fewshot_epochs(5, 5, epochs=3, min_iterations=10) == 10


def test_pseudo_rows_filter_by_confidence_and_convert_to_yolo():
    pred = {
        "boxes": np.array([[0.1, 0.2, 0.3, 0.6], [0.5, 0.5, 0.9, 0.7]]),
        "scores": np.array([0.8, 0.3]),
        "cls": np.array([2, 1]),
    }
    np.testing.assert_allclose(pseudo_rows(pred, 0.5), [[2, 0.2, 0.4, 0.2, 0.4]])
    assert pseudo_rows(pred, 0.9).shape == (0, 5)
    assert pseudo_rows(pred, 0.1).shape == (2, 5)


# --- Ultralytics end to end (random-init YOLOv8n, tiny images: nothing is downloaded) ---------


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    pytest.importorskip("ultralytics")
    from roadcond.detection.train_yolo import Env, evaluate_source, train_source

    base = tmp_path_factory.mktemp("p3")
    cfg = copy.deepcopy(CFG)
    r = cfg["rdd2022"]
    r["yolo"].update(imgsz=64, epochs=1, batch=8, workers=0, max_train_images=20)
    r["few_shot"].update(epochs=1, min_iterations=2, batch=4, seeds=[0, 1])
    r["label_budgets"] = [0, 5]
    r["pseudo_label"].update(
        pool_size=10, source_replay=6, epochs=1, batch=8, sweep=[0.001, 0.01], seeds=[0, 1]
    )
    make_mock_rdd(base / "raw", cfg, n_images=40)
    env = Env(
        cfg, base / "work", base / "splits", base / "art", device="cpu", pretrained=False, workers=0
    )
    prepare(cfg, base / "raw", env.splits_dir, env.work_dir)
    train_source(env)
    result = evaluate_source(env)
    (base / "config.yaml").write_text(yaml.safe_dump(cfg))
    return env, result, base


def test_source_detector_is_scored_on_every_country(pipeline):
    env, result, _ = pipeline
    assert list(result["countries"]) == ["Japan", "Norway", "Czech", "United_States"]
    assert set(result["gaps"]) == {"Norway", "Czech", "United_States"}
    for scores in result["countries"].values():
        assert 0.0 <= scores["map50"] <= 1.0 and 0.0 <= scores["f1"] <= 1.0
        assert scores["tp"] + scores["fn"] == scores["eval_boxes"]  # every GT box is counted
        assert set(scores["ap50_per_class"]) == {"D00", "D10", "D20", "D40"}
    assert (env.out_dir / "detection" / "source" / "weights" / "best.pt").is_file()
    assert (env.out_dir / "results" / "detection_source.json").is_file()


def test_finished_runs_are_skipped_on_the_second_call(pipeline):
    from roadcond.detection.train_yolo import best_weights, train_source

    env, _, _ = pipeline
    best = best_weights(env, "source")
    before = best.stat().st_mtime_ns
    assert train_source(env) == best
    assert best.stat().st_mtime_ns == before  # not retrained


def test_few_shot_runs_resume_and_feed_the_learning_curve(pipeline):
    from roadcond.detection.few_shot import build_curve, load_runs, run_few_shot
    from roadcond.detection.train_yolo import best_weights

    env, _, _ = pipeline
    runs = run_few_shot(env, best_weights(env, "source"), countries=["Norway"])
    keys = [(r["country"], r["budget"], r["seed"]) for r in runs]
    assert keys == [("Norway", 5, 0), ("Norway", 5, 1)]
    assert all(len(r["images"]) == 5 for r in runs)
    saved = read_name_list(env.splits_dir / "rdd_norway_fewshot_n5_seed0.txt")
    assert saved == sorted(runs[0]["images"])
    test_names = set(read_name_list(env.splits_dir / "rdd_norway_test.txt"))
    assert not test_names & set(saved)  # few-shot images never come from the test split

    again = run_few_shot(env, best_weights(env, "source"), countries=["Norway"])
    assert again == runs and len(load_runs(env)) == 2  # second call reads the saved results
    curve = build_curve(env)["Norway"]
    assert sorted(curve) == [0, 5] and curve[5]["map50"]["n"] == 2


def test_pseudo_labelling_sweeps_thresholds_and_scores_test_once(pipeline):
    from roadcond.detection.pseudo_label import run_pseudo
    from roadcond.detection.train_yolo import best_weights

    env, _, _ = pipeline
    summary = run_pseudo(env, best_weights(env, "source"), countries=["Czech"])
    entry = summary["countries"]["Czech"]
    assert set(entry["val_map50_by_threshold"]) == {"0.001", "0.01"}
    assert entry["threshold"] in (0.001, 0.01)
    assert [r["seed"] for r in entry["runs"]] == [0, 1]
    for run in entry["runs"]:
        assert run["split"] == "test" and run["pool_images"] == 10
        assert run["pseudo_images"] <= run["pool_images"]
        assert run["no_pseudo_labels"] == (run["pseudo_images"] == 0)
    assert summary["threshold_selected_on"] == "target validation images"
    results = env.out_dir / "results" / "pseudo"
    assert len(list(results.glob("*_val.json"))) == 2  # the sweep never touched test
    assert len(list(results.glob("*_test.json"))) == 2  # only the chosen threshold, per seed


def test_report_and_cli_over_saved_results(pipeline, capsys):
    from roadcond.detection import cli

    env, _, base = pipeline
    cli.main(
        [
            *("report", "--config", str(base / "config.yaml"), "--splits-dir", str(env.splits_dir)),
            *("--work-dir", str(env.work_dir), "--out-dir", str(env.out_dir), "--device", "cpu"),
        ]
    )
    out = capsys.readouterr().out
    for heading in ("D1", "D2", "D3 and D5", "D4", "side by side"):
        assert heading in out
    assert "Not run yet (`source --augment`)" in out  # D2 was not run in this fixture
    assert (env.out_dir / "results" / "detection_report.md").is_file()
    assert (env.out_dir / "results" / "learning_curve.png").is_file()


def test_cli_overrides_apply_to_the_step_being_run():
    from roadcond.detection.cli import apply_overrides, make_parser

    parser = make_parser()
    args = parser.parse_args(["fewshot", "--epochs", "7", "--imgsz", "96", "--model", "yolov8s"])
    cfg = apply_overrides(CFG, args)
    assert cfg["rdd2022"]["few_shot"]["epochs"] == 7 and cfg["rdd2022"]["yolo"]["epochs"] == 30
    assert cfg["rdd2022"]["yolo"]["imgsz"] == 96 and cfg["rdd2022"]["yolo"]["model"] == "yolov8s"
    assert CFG["rdd2022"]["yolo"]["imgsz"] == 640  # the loaded config is not mutated
