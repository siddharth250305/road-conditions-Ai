"""Assemble the cross-country study tables (PRD D1-D5) from the saved result files."""

import json
from pathlib import Path
from typing import Any

from roadcond.detection.few_shot import build_curve, load_runs, save_learning_curve_plot
from roadcond.detection.metrics import (
    fmt,
    format_curve_table,
    gap_recovered,
    relative_gap,
    summarize,
)
from roadcond.detection.train_yolo import Env, results_dir
from roadcond.metrics import format_table


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def build_report(env: Env) -> str:
    """Markdown report; sections for steps that have not been run yet say so."""
    res = results_dir(env)
    base = _load(res / "detection_source.json")
    if base is None:
        raise FileNotFoundError(f"No source results in {res}: run the 'source' step first")
    source = base["source_country"]
    targets = [c for c in base["countries"] if c != source]
    src = base["countries"][source]
    zero = {c: base["countries"][c] for c in targets}
    conf = base["f1_conf"]
    parts = []

    rows = []
    for country, scores in base["countries"].items():
        gap = base["gaps"].get(country)
        rows.append(
            [
                f"{country}{' (source)' if country == source else ''}",
                f"{scores['map50']:.3f}",
                f"{scores['map50_95']:.3f}",
                f"{scores['f1']:.3f}",
                _pct(gap["map50"]["relative_gap"]) if gap else "-",
                _pct(gap["map50_95"]["relative_gap"]) if gap else "-",
            ]
        )
    parts += [
        f"## D1: source detector, scored on each country's held-out test split (F1 at conf {conf})",
        format_table(
            ["country", "mAP@0.5", "mAP@0.5:0.95", "F1", "gap (mAP@0.5)", "gap (mAP@0.5:0.95)"],
            rows,
        ),
        "Relative gap = (source - target) / source. A negative value means the target scored "
        "higher than the source; the gap is then not a loss and nothing is 'recovered'.",
    ]

    aug = _load(res / "detection_source_aug.json")
    parts.append("## D2: weather augmentation, no target images used")
    if aug is None:
        parts.append("Not run yet (`source --augment`).")
    else:
        rows = []
        for country in targets:
            a = aug["countries"][country]["map50"]
            rows.append(
                [
                    country,
                    f"{zero[country]['map50']:.3f}",
                    f"{a:.3f}",
                    f"{a - zero[country]['map50']:+.3f}",
                    _pct(gap_recovered(src["map50"], zero[country]["map50"], a)),
                ]
            )
        parts += [
            format_table(
                ["country", "baseline mAP@0.5", "augmented mAP@0.5", "change", "gap recovered"],
                rows,
            ),
            f"On the source country the augmented detector scores "
            f"{aug['countries'][source]['map50']:.3f} (baseline {src['map50']:.3f}).",
        ]

    curve = build_curve(env)
    parts.append("## D3 and D5: few-shot fine-tuning and the learning curve")
    if curve is None or not load_runs(env):
        parts.append("Not run yet (`fewshot`).")
    else:
        parts += [
            format_curve_table(curve),
            "Mean ± std over seeds; 0 labels is the source detector, run once, "
            "so it has no spread.",
        ]
        save_learning_curve_plot(curve, src["map50"], res / "learning_curve.png")
        parts.append(f"Plot: {res / 'learning_curve.png'}")

    pseudo = _load(res / "detection_pseudo.json")
    parts.append("## D4: pseudo-labelling, no target labels used for training")
    if pseudo is None:
        parts.append("Not run yet (`pseudo`).")
    else:
        rows = []
        for country, entry in pseudo["countries"].items():
            recovered = summarize(
                [
                    gap_recovered(src["map50"], zero[country]["map50"], r["map50"])
                    for r in entry["runs"]
                ]
            )
            rows.append(
                [
                    country,
                    entry["threshold"],
                    fmt(entry["pseudo_label_precision"]),
                    fmt(entry["map50"]),
                    fmt(entry["map50_95"]),
                    fmt(recovered, 1, percent=True),
                ]
            )
        parts += [
            format_table(
                [
                    "country",
                    "threshold",
                    "pseudo-label precision",
                    "mAP@0.5",
                    "mAP@0.5:0.95",
                    "gap recovered",
                ],
                rows,
            ),
            f"Threshold chosen on {pseudo['threshold_selected_on']}; precision is measured "
            "against the pool's hidden labels and is a diagnostic only.",
        ]

    parts.append("## Recovery methods side by side (mAP@0.5, share of gap recovered)")
    rows = []
    budgets = sorted({b for c in (curve or {}).values() for b in c if b > 0})
    for country in targets:
        z = zero[country]["map50"]
        row = [country, f"{z:.3f}"]
        if aug is None:
            row.append("not run")
        else:
            a = aug["countries"][country]["map50"]
            row.append(f"{a:.3f} ({_pct(gap_recovered(src['map50'], z, a))})")
        if pseudo is None or country not in pseudo["countries"]:
            row.append("not run")
        else:
            entry = pseudo["countries"][country]
            rec = summarize([gap_recovered(src["map50"], z, r["map50"]) for r in entry["runs"]])
            row.append(f"{fmt(entry['map50'])} ({fmt(rec, 1, percent=True)})")
        for b in budgets:
            cell = (curve or {}).get(country, {}).get(b)
            row.append(
                "not run"
                if cell is None
                else f"{fmt(cell['map50'])} ({fmt(cell['recovered'], 1, percent=True)})"
            )
        rows.append(row)
    headers = [
        "country",
        "0 labels",
        "+ augmentation",
        "+ pseudo-labels",
        *[f"{b} labels" for b in budgets],
    ]
    parts.append(format_table(headers, rows))
    parts.append(
        f"Source-country reference: mAP@0.5 {src['map50']:.3f}. "
        "Relative gap of the zero-shot detector per target: "
        + ", ".join(f"{c} {_pct(relative_gap(src['map50'], zero[c]['map50']))}" for c in targets)
        + "."
    )
    return "\n\n".join(parts) + "\n"
