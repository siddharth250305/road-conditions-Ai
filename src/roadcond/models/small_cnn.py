"""Small CNN written from scratch (PRD C3): 4 x (Conv, BatchNorm, ReLU, MaxPool, Dropout)."""

from collections.abc import Sequence

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Four conv stages, global average pooling, one linear layer. About 0.4M parameters."""

    def __init__(
        self,
        num_classes: int = 6,
        widths: Sequence[int] = (32, 64, 128, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        stages = []
        in_channels = 3
        for width in widths:
            stages.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),  # not inplace: Grad-CAM hooks this layer's output
                    nn.MaxPool2d(2),
                    nn.Dropout2d(dropout),
                )
            )
            in_channels = width
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).flatten(1))

    @property
    def target_layer(self) -> nn.Module:
        """Last activation before the final pooling, for Grad-CAM (largest spatial map)."""
        return self.features[-1][2]
