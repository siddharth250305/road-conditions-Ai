"""Markdown results tables for the README and the write-up (PRD R1, R2), from saved result files.

Nothing here is typed in by hand: every number comes from the JSON files the training and
detection scripts wrote, and a table whose files do not exist yet says "pending" instead of
showing a made-up value.
"""

import json
from pathlib import Path
from typing import Any

from roadcond.detection.metrics import fmt, learning_curve
from roadcond.metrics import format_table, mean_std

START = "<!-- results:start -->"
END = "<!-- results:end -->"
CLASSIFIERS = ("logreg", "random_forest", "small_cnn", "resnet50", "efficientnet")
BUDGETS = (10, 50, 200)


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def classification_table(results_dir: str | Path) -> str:
    """One row per classifier: macro-F1 (the headline), accuracy and calibration, over seeds."""
    results_dir = Path(results_dir)
    runs_by_model: dict[str, list[dict[str, Any]]] = {}
    seeds = 0
    for name in ("baseline", "small_cnn", "resnet50", "efficientnet"):
        data = _read(results_dir / f"{name}.json")
        if data is None:
            continue
        seeds = max(seeds, len(data.get("seeds", [])))
        for model, runs in data["models"].items():
            runs_by_model[model] = runs
    rows = []
    for model in CLASSIFIERS:
        runs = runs_by_model.get(model)
        if not runs:
            rows.append([model, "pending", "pending", "pending", "pending"])
            continue
        tested = [r for r in runs if r.get("test")]
        val = mean_std([r["val"]["macro_f1"] for r in runs])
        test = mean_std([r["test"]["macro_f1"] for r in tested]) if tested else "not run"
        acc = mean_std([r["test"]["accuracy"] for r in tested]) if tested else "not run"
        cal = "-"
        if tested and "calibration" in tested[0]["test"]:
            before = sum(r["test"]["calibration"]["ece_before"] for r in tested) / len(tested)
            after = sum(r["test"]["calibration"]["ece_after"] for r in tested) / len(tested)
            cal = f"{before:.3f} -> {after:.3f}"
        rows.append([model, val, test, acc, cal])
    table = format_table(
        ["model", "val macro-F1", "test macro-F1", "test accuracy", "test ECE before -> after"],
        rows,
    )
    note = (
        f"Mean ± sample std over {seeds} seed(s). Macro-F1 is the headline metric; "
        "ECE is expected calibration error before and after temperature scaling."
        if seeds
        else "No classifier results found yet: run the Phase 1 and Phase 2 scripts."
    )
    return f"{table}\n\n{note}"


def detection_table(results_dir: str | Path) -> str:
    """Per target country: mAP@0.5 with no target labels, then each recovery method."""
    results_dir = Path(results_dir)
    base = _read(results_dir / "detection_source.json")
    if base is None:
        return "Pending: the detection experiments have not been run (scripts/run_phase3.sh)."
    aug = _read(results_dir / "detection_source_aug.json")
    pseudo = _read(results_dir / "detection_pseudo.json")
    runs = [json.loads(p.read_text()) for p in sorted((results_dir / "fewshot").glob("*.json"))]
    source = base["source_country"]
    src = base["countries"][source]
    targets = [c for c in base["countries"] if c != source]
    zero = {c: base["countries"][c] for c in targets}
    curve = learning_curve(runs, zero, src) if runs else {}

    headers = ["target", f"{source} (source)", "zero-shot", "gap", "+ weather aug"]
    headers += [f"+ {b} labels" for b in BUDGETS] + ["+ pseudo-labels"]
    rows = []
    for country in targets:
        gap = base["gaps"][country]["map50"]["relative_gap"]
        row = [
            country.replace("_", " "),
            f"{src['map50']:.3f}",
            f"{zero[country]['map50']:.3f}",
            _pct(gap),
            f"{aug['countries'][country]['map50']:.3f}" if aug else "pending",
        ]
        for budget in BUDGETS:
            cell = curve.get(country, {}).get(budget)
            row.append(fmt(cell["map50"]) if cell else "pending")
        if pseudo and country in pseudo["countries"]:
            row.append(fmt(pseudo["countries"][country]["map50"]))
        else:
            row.append("pending")
        rows.append(row)
    seeds = sorted({r["seed"] for r in runs})
    note = (
        "mAP@0.5 on each country's own held-out test split. Few-shot and pseudo-label cells are "
        f"mean ± std over {len(seeds)} seed(s). Gap = (source - target) / source."
    )
    return f"{format_table(headers, rows)}\n\n{note}"


def render(results_dir: str | Path, section: str = "all") -> str:
    """The tables as markdown; section is 'all', 'classification' or 'detection'."""
    parts = []
    if section in ("all", "classification"):
        parts += [
            "#### Surface classification (RSCD, six friction classes)",
            classification_table(results_dir),
        ]
    if section in ("all", "detection"):
        parts += [
            "#### Cross-country damage detection (YOLOv8n, RDD2022)",
            detection_table(results_dir),
        ]
    if not parts:
        raise ValueError(f"unknown section {section!r}")
    return "\n\n".join(parts)


def update_readme(readme: str | Path, tables: str) -> bool:
    """Replace the text between the results markers in a file; returns False if they are absent."""
    readme = Path(readme)
    text = readme.read_text()
    if START not in text or END not in text:
        return False
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n\n{tables}\n\n{END}{tail}")
    return True
