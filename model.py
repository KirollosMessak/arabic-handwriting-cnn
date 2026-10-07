"""A compact CNN for 32x32 single-channel character images."""

import torch
from torch import nn


def conv_block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
    """Two 3x3 convs (each with BatchNorm + ReLU), then 2x2 max-pool and dropout."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class ArabicCNN(nn.Module):
    """32x32 -> 16x16 -> 8x8 -> 4x4 feature maps, then global average pooling.

    Global average pooling (instead of a large fully-connected layer) keeps the
    model small (~290K parameters) and reduces overfitting.
    """

    def __init__(self, num_classes: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, 0.1),
            conv_block(32, 64, 0.2),
            conv_block(64, 128, 0.3),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
