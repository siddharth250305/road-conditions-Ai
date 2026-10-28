"""Command line for the cross-country study (PRD D1-D5).

    python -m roadcond.detection.cli prepare  --root /path/to/RDD2022
    python -m roadcond.detection.cli source                  # D1: train on Japan, score everywhere
    python -m roadcond.detection.cli source --augment        # D2: weather augmentation
    python -m roadcond.detection.cli fewshot                 # D3, D5: 10 / 50 / 200 labels
    python -m roadcond.detection.cli pseudo                  # D4: pseudo-labelling
    python -m roadcond.detection.cli report                  # tables and learning curve

Every step skips work that already finished, so rerun a command after a Colab timeout (add
--resume to continue a half-trained source detector). The test split of each country is
scored once per final model.
"""

import argparse
import copy
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from roadcond.config import load_config, resolve_path
from roadcond.data import rdd
from roadcond.detection.few_shot import build_curve, run_few_shot
from roadcond.detection.metrics import fmt, format_curve_table
from roadcond.detection.pseudo_label import run_pseudo
from roadcond.detection.report import build_report
from roadcond.detection.train_yolo import (
    Env,
    best_weights,
    evaluate_countries,
    evaluate_source,
    pick_device,
    results_dir,
    save_json,
    train_source,
)
from roadcond.metrics import format_table


def make_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    common.add_argument("--root", help="RDD2022 folder (default: rdd2022.root)")
    common.add_argument("--splits-dir", help="list and YAML folder (default: paths.splits)")
    common.add_argument("--work-dir", help="YOLO work tree (default: rdd2022.work_dir)")
    common.add_argument("--out-dir", help="runs and results folder (default: artifacts)")
    common.add_argument("--device", default="auto", help="auto, cpu or a CUDA index such as 0")
    common.add_argument("--no-pretrained", action="store_true", help="random init (no download)")
    common.add_argument("--model", help="yolov8n or yolov8s")
    common.add_argument("--imgsz", type=int)
    common.add_argument("--batch", type=int)
    common.add_argument("--workers", type=int)
    common.add_argument("--epochs", type=int, help="epochs of the step being run")
    common.add_argument("--resume", action="store_true", help="continue an interrupted source run")
    common.add_argument("--force", action="store_true", help="redo steps that already finished")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", parents=[common], help="split countries, write lists and YAMLs")
    p.add_argument("--countries", nargs="+")
    p.add_argument("--max-train-images", type=int, help="cap on the source training pool")
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    p = sub.add_parser(
        "source", parents=[common], help="D1 / D2: train and score the source detector"
    )
    p.add_argument("--augment", action="store_true", help="D2: weather augmentation")
    p = sub.add_parser("evaluate", parents=[common], help="score any weights on every test split")
    p.add_argument("--weights", required=True)
    p.add_argument("--name", default="custom")
    p = sub.add_parser("fewshot", parents=[common], help="D3 / D5: few-shot fine-tuning")
    p.add_argument("--countries", nargs="+")
    p.add_argument("--budgets", type=int, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p = sub.add_parser("pseudo", parents=[common], help="D4: pseudo-labelling")
    p.add_argument("--countries", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--thresholds", type=float, nargs="+", help="one value skips the sweep")
    sub.add_parser("report", parents=[common], help="tables, learning curve and report.md")
    return parser


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """A copy of the config with the command-line overrides applied to the step being run."""
    cfg = copy.deepcopy(cfg)
    rdd_cfg = cfg["rdd2022"]
    if args.model:
        rdd_cfg["yolo"]["model"] = args.model
    if args.imgsz:
        rdd_cfg["yolo"]["imgsz"] = args.imgsz
    if args.batch:
        rdd_cfg["yolo"]["batch"] = args.batch
    if args.epochs:
        section = {"source": "yolo", "fewshot": "few_shot", "pseudo": "pseudo_label"}.get(
            args.command
        )
        if section:
            rdd_cfg[section]["epochs"] = args.epochs
    return cfg


def make_env(args: argparse.Namespace) -> Env:
    cfg = apply_overrides(load_config(args.config), args)
    cwd = Path.cwd()

    def pick(arg: str | None, default: str | Path) -> Path:
        return resolve_path(arg, cwd) if arg else resolve_path(default)

    return Env(
        cfg=cfg,
        work_dir=Path(os.path.abspath(pick(args.work_dir, cfg["rdd2022"]["work_dir"]))),
        splits_dir=pick(args.splits_dir, cfg["paths"]["splits"]),
        out_dir=pick(args.out_dir, cfg["paths"]["artifacts"]),
        device=pick_device(args.device),
        pretrained=not args.no_pretrained,
        workers=args.workers,
    )


def print_source_table(result: dict) -> None:
    rows = [
        [c, f"{s['map50']:.3f}", f"{s['map50_95']:.3f}", f"{s['f1']:.3f}"]
        for c, s in result["countries"].items()
    ]
    print(format_table(["country", "mAP@0.5", "mAP@0.5:0.95", "F1"], rows))
    for country, gap in result["gaps"].items():
        value = gap["map50"]["relative_gap"]
        print(f"relative gap {country}: {'n/a' if value is None else f'{value * 100:.1f}%'}")


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    env = make_env(args)
    cfg = env.cfg
    source = cfg["rdd2022"]["source_country"]

    if args.command == "prepare":
        root = (
            resolve_path(args.root, Path.cwd())
            if args.root
            else resolve_path(cfg["rdd2022"]["root"])
        )
        reports = rdd.prepare(
            cfg, root, env.splits_dir, env.work_dir, args.workers or os.cpu_count() or 1,
            args.countries, args.max_train_images, args.copy,
        )  # fmt: skip
        for country, r in reports.items():
            print(
                f"{country}: {r['labelled_images']} labelled, splits {r['split_sizes']}, "
                f"{r['near_duplicates_dropped']} near-duplicates dropped, "
                f"dropped labels {r['dropped_labels']}"
            )
    elif args.command == "source":
        name = "source_aug" if args.augment else "source"
        train_source(env, augment=args.augment, resume=args.resume, force=args.force)
        result = evaluate_source(env, name, force=args.force)
        print_source_table(result)
        print(f"Saved {results_dir(env) / f'detection_{name}.json'}")
    elif args.command == "evaluate":
        countries = evaluate_countries(env, args.weights, "test", args.name)
        path = results_dir(env) / f"detection_eval_{args.name}.json"
        save_json(path, {"weights": args.weights, "countries": countries})
        print_source_table({"countries": countries, "gaps": {}})
        print(f"Saved {path}")
    elif args.command == "fewshot":
        runs = run_few_shot(
            env, best_weights(env, "source"), args.countries, args.budgets, args.seeds, args.force
        )
        curve = build_curve(env)
        print(format_curve_table(curve) if curve else f"{len(runs)} runs finished")
    elif args.command == "pseudo":
        summary = run_pseudo(
            env,
            best_weights(env, "source"),
            args.countries,
            args.seeds,
            args.thresholds,
            args.force,
        )
        for country, entry in summary["countries"].items():
            print(
                f"{country}: threshold {entry['threshold']}, test mAP@0.5 {fmt(entry['map50'])}, "
                f"pseudo-label precision {fmt(entry['pseudo_label_precision'])}"
            )
    elif args.command == "report":
        text = build_report(env)
        path = results_dir(env) / "detection_report.md"
        path.write_text(f"# Cross-country detection study ({source} as source)\n\n{text}")
        print(text)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
