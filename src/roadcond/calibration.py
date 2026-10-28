"""Calibration (PRD C9): temperature scaling, expected calibration error, reliability data.

Fit the temperature on validation logits only. ECE on the data it was fitted on is
optimistic, so judge "error drops after scaling" (H4) on the test split.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import nn


class TemperatureScaler(nn.Module):
    """Divides logits by a single learned temperature T > 0 (stored as log T)."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100):
        """Minimise NLL on held-out logits with L-BFGS; returns self."""
        logits, labels = logits.detach().float(), labels.detach().long()
        if logits.ndim != 2 or labels.shape != logits.shape[:1]:
            raise ValueError(
                f"Need logits (N, C) and labels (N,), got {logits.shape}, {labels.shape}"
            )
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self


def nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.float(), labels.long()))


def reliability_bins(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> dict[str, Any]:
    """Per-bin mean confidence, accuracy and count over equal-width confidence bins (lo, hi]."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels)
    if probs.ndim != 2 or labels.shape != probs.shape[:1]:
        raise ValueError(f"Need probs (N, C) and labels (N,), got {probs.shape}, {labels.shape}")
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    index = np.clip(np.ceil(confidence * n_bins).astype(int) - 1, 0, n_bins - 1)
    count = np.bincount(index, minlength=n_bins)
    safe = np.maximum(count, 1)
    return {
        "edges": np.linspace(0.0, 1.0, n_bins + 1).tolist(),
        "confidence": (np.bincount(index, weights=confidence, minlength=n_bins) / safe).tolist(),
        "accuracy": (np.bincount(index, weights=correct, minlength=n_bins) / safe).tolist(),
        "count": count.tolist(),
    }


def expected_calibration_error(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> float:
    """Sample-weighted mean of |accuracy - confidence| over the confidence bins."""
    bins = reliability_bins(probs, labels, n_bins)
    count = np.asarray(bins["count"])
    gap = np.abs(np.asarray(bins["accuracy"]) - np.asarray(bins["confidence"]))
    return float((count * gap).sum() / count.sum())


def save_reliability_plot(
    before: dict[str, Any],
    after: dict[str, Any],
    ece_before: float,
    ece_after: float,
    path: str | Path,
    title: str = "",
) -> None:
    """Reliability diagrams before and after temperature scaling, side by side."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 4))
    axes = fig.subplots(1, 2, sharey=True)
    panels = ((before, "before scaling", ece_before), (after, "after scaling", ece_after))
    for ax, (bins, name, ece) in zip(axes, panels, strict=True):
        edges = np.asarray(bins["edges"])
        filled = np.asarray(bins["count"]) > 0
        ax.bar(
            ((edges[:-1] + edges[1:]) / 2)[filled],
            np.asarray(bins["accuracy"])[filled],
            width=edges[1] - edges[0],
            edgecolor="black",
            alpha=0.8,
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence")
        ax.set_title(f"{name}: ECE {ece:.3f}")
    axes[0].set_ylabel("accuracy")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
