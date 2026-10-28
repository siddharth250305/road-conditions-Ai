"""Detection metrics that do not need a model: box matching, F1 at a fixed confidence, gaps.

mAP comes from Ultralytics; this module adds the pieces the PRD asks for on top: F1 at a
fixed confidence threshold, the relative domain gap, the share of the gap a method recovers,
and mean / spread over seeds for the learning curve.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from roadcond.metrics import format_table

EPS = 1e-9


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of corner boxes (x1, y1, x2, y2): shape (len(a), len(b)).

    IoU is unchanged by scaling each axis, so normalised and pixel boxes give the same values.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    iy = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_boxes(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_cls: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    iou_thr: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy one-to-one matching, highest score first, same class only: (tp, fp, fn)."""
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    pred_scores, pred_cls = np.asarray(pred_scores, dtype=float), np.asarray(pred_cls)
    gt_cls = np.asarray(gt_cls)
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt
    iou = box_iou(pred_boxes, gt_boxes) * (pred_cls[:, None] == gt_cls[None, :])
    taken = np.zeros(n_gt, dtype=bool)
    tp = 0
    for i in np.argsort(-pred_scores, kind="stable"):
        candidates = np.where(taken, -1.0, iou[i])
        j = int(np.argmax(candidates))
        if candidates[j] >= iou_thr and candidates[j] > 0:
            taken[j] = True
            tp += 1
    return tp, n_pred - tp, n_gt - tp


def prf1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall and F1 from counts; a zero denominator gives 0.0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def f1_at_conf(
    predictions: Sequence[dict[str, Any]],
    ground_truths: Sequence[dict[str, Any]],
    conf: float,
    iou_thr: float = 0.5,
) -> dict[str, float]:
    """Micro-averaged precision / recall / F1 over images, keeping predictions with score >= conf.

    predictions: per image {"boxes" (N, 4), "scores" (N,), "cls" (N,)};
    ground_truths: per image {"boxes" (M, 4), "cls" (M,)}, in the same order.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"{len(predictions)} predictions but {len(ground_truths)} ground truths")
    tp = fp = fn = 0
    for pred, gt in zip(predictions, ground_truths, strict=True):
        scores = np.asarray(pred["scores"], dtype=float)
        keep = scores >= conf
        boxes = np.asarray(pred["boxes"], dtype=float).reshape(-1, 4)[keep]
        t, f_p, f_n = match_boxes(
            boxes, scores[keep], np.asarray(pred["cls"])[keep], gt["boxes"], gt["cls"], iou_thr
        )
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
    return {**prf1(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn, "conf": conf}


def relative_gap(source: float, target: float) -> float | None:
    """(source - target) / source; None when the source score is not positive."""
    return (source - target) / source if source > EPS else None


def gap_recovered(source: float, zero_shot: float, method: float) -> float | None:
    """Share of the source-to-zero-shot gap that a method wins back (1.0 = all of it).

    None when the target is not worse than the source, since there is no gap to recover.
    """
    gap = source - zero_shot
    return (method - zero_shot) / gap if gap > EPS else None


def summarize(values: Sequence[float | None]) -> dict[str, Any]:
    """Mean and sample standard deviation over seeds, ignoring None; std is None for n < 2."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(clean)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "n": len(arr),
    }


def fmt(summary: dict[str, Any], digits: int = 3, percent: bool = False) -> str:
    """'mean ± std' for a summarize() result; 'n/a' when there is no value."""
    if summary["mean"] is None:
        return "n/a"
    scale = 100.0 if percent else 1.0
    text = f"{summary['mean'] * scale:.{digits}f}"
    if summary["std"] is not None:
        text += f" ± {summary['std'] * scale:.{digits}f}"
    return text + ("%" if percent else "")


def learning_curve(
    runs: Sequence[dict[str, Any]],
    zero_shot: dict[str, dict[str, float]],
    source: dict[str, float],
    metric: str = "map50",
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    """Per country and label budget: mean ± std of metrics, relative gap and gap recovered.

    runs: dicts with country, budget (> 0), seed and metric values (map50, map50_95, f1).
    zero_shot: country -> the source detector's scores on that country (budget 0).
    source: the source detector's scores on the source country (the gap reference).
    """
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        grouped[run["country"]][int(run["budget"])].append(run)
    for country, scores in zero_shot.items():
        grouped[country][0] = [{**scores, "country": country, "budget": 0, "seed": None}]

    curve: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for country, budgets in grouped.items():
        base = zero_shot[country][metric]
        curve[country] = {}
        for budget in sorted(budgets):
            entries = budgets[budget]
            curve[country][budget] = {
                "map50": summarize([e["map50"] for e in entries]),
                "map50_95": summarize([e["map50_95"] for e in entries]),
                "f1": summarize([e["f1"] for e in entries]),
                "relative_gap": summarize(
                    [relative_gap(source[metric], e[metric]) for e in entries]
                ),
                "recovered": summarize(
                    [gap_recovered(source[metric], base, e[metric]) for e in entries]
                ),
            }
    return curve


def format_curve_table(
    curve: dict[str, dict[int, dict[str, dict[str, Any]]]], metric_label: str = "mAP@0.5"
) -> str:
    """Markdown learning-curve table: one row per country and budget."""
    headers = [
        "country",
        "labels",
        metric_label,
        "mAP@0.5:0.95",
        "F1",
        "relative gap",
        "gap recovered",
    ]
    rows = []
    for country, budgets in curve.items():
        for budget, cell in budgets.items():
            rows.append(
                [
                    country,
                    budget,
                    fmt(cell["map50"]),
                    fmt(cell["map50_95"]),
                    fmt(cell["f1"]),
                    fmt(cell["relative_gap"], 1, percent=True),
                    fmt(cell["recovered"], 1, percent=True),
                ]
            )
    return format_table(headers, rows)
