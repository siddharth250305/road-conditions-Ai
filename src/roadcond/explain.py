"""Grad-CAM heatmaps and correct-vs-wrong diagnostics (PRD C8, H5).

Maps are computed in eval mode on the model's own device; AMP is not used here.
"""

import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from roadcond.data.dataset import RoadDataset

log = logging.getLogger(__name__)


def default_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional stage: SmallCNN.target_layer, ResNet layer4, EfficientNet blocks."""
    if hasattr(model, "target_layer"):
        return model.target_layer
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    raise ValueError(f"No default Grad-CAM layer for {type(model).__name__}; pass target_layer")


def gradcam(
    model: nn.Module,
    inputs: torch.Tensor,
    target_layer: nn.Module | None = None,
    class_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Heatmaps (B, H, W) float32 in [0, 1] at input resolution.

    class_ids picks the class to explain per image; None explains each predicted class.
    """
    model.eval()
    layer = target_layer if target_layer is not None else default_target_layer(model)
    targets = None if class_ids is None else [ClassifierOutputTarget(int(c)) for c in class_ids]
    device = next(model.parameters()).device
    # Gradients must reach the target layer even when every parameter is frozen.
    batch = inputs.detach().to(device).clone().requires_grad_(True)
    with GradCAM(model=model, target_layers=[layer]) as cam:
        return cam(input_tensor=batch, targets=targets)


def overlay(raw_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap (H, W) in [0, 1] onto an RGB uint8 image of the same size."""
    if raw_rgb.shape[:2] != cam.shape:
        raise ValueError(f"Image {raw_rgb.shape[:2]} and heatmap {cam.shape} sizes differ")
    return show_cam_on_image(
        raw_rgb.astype(np.float32) / 255.0, cam, use_rgb=True, image_weight=1.0 - alpha
    )


def select_examples(
    probs: np.ndarray, labels: np.ndarray, n_each: int, seed: int = 0
) -> dict[str, list[int]]:
    """Indices of the most confident mistakes and of randomly chosen correct predictions."""
    preds, confidence = probs.argmax(axis=1), probs.max(axis=1)
    wrong = np.flatnonzero(preds != labels)
    wrong = wrong[np.argsort(-confidence[wrong])][:n_each]
    correct = np.flatnonzero(preds == labels).tolist()
    random.Random(seed).shuffle(correct)
    return {"correct": sorted(correct[:n_each]), "wrong": wrong.tolist()}


def save_gallery(
    model: nn.Module,
    dataset: RoadDataset,
    probs: np.ndarray,
    path: str | Path,
    n_each: int = 4,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Grid of original and Grad-CAM images for correct and wrong predictions.

    The dataset must use the evaluation transform. Returns what was drawn, for the log.
    """
    from matplotlib.figure import Figure

    labels = np.asarray(dataset.targets)
    chosen = select_examples(probs, labels, n_each, seed)
    size = tuple(dataset[0][0].shape[-2:])
    classes = dataset.classes
    columns = [(group, i) for group, idx in chosen.items() for i in idx]
    if not columns:
        log.warning("No examples to draw for %s", path)
        return []

    inputs = torch.stack([dataset[i][0] for _, i in columns])
    preds = probs.argmax(axis=1)
    cams = np.concatenate(
        [
            gradcam(model, inputs[k : k + 16], class_ids=preds[[i for _, i in columns[k : k + 16]]])
            for k in range(0, len(columns), 16)
        ]
    )

    fig = Figure(figsize=(2.6 * len(columns), 4.4))
    axes = fig.subplots(2, len(columns), squeeze=False)
    drawn = []
    for col, ((group, i), cam) in enumerate(zip(columns, cams, strict=True)):
        raw = dataset.load_raw(i, size)
        axes[0][col].imshow(raw)
        axes[1][col].imshow(overlay(raw, cam))
        title = f"{group}\n{classes[labels[i]]} -> {classes[preds[i]]} ({probs[i].max():.2f})"
        axes[0][col].set_title(title, fontsize=8)
        for ax in axes[:, col]:
            ax.axis("off")
        drawn.append(
            {
                "index": int(i),
                "path": dataset.records[i].path,
                "group": group,
                "true": classes[labels[i]],
                "pred": classes[preds[i]],
                "confidence": float(probs[i].max()),
            }
        )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return drawn
