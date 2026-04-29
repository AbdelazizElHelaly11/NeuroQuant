"""
NeuroQuant v2.0 - Example Models

Example model adapters for testing the framework.
These are NOT part of the core framework — they serve
as validation examples only.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2CIFAR(nn.Module):
    """
    MobileNetV2 adapted for CIFAR-10 (32×32 input).

    This is an EXAMPLE model for validating the framework.
    The framework works with ANY nn.Module — this class
    simply provides a convenient test case.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True) -> None:
        super().__init__()
        weights: Optional[models.MobileNet_V2_Weights] = (
            models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = models.mobilenet_v2(weights=weights)
        self.num_classes = num_classes
        self._adapt_for_cifar()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _adapt_for_cifar(self) -> None:
        """
        Adapt MobileNetV2 (designed for 224×224) to work with
        CIFAR-10 (32×32) by reducing the first conv stride and
        swapping the final classifier to match ``num_classes``.
        """
        first_conv = self.backbone.features[0][0]
        if isinstance(first_conv, nn.Conv2d) and first_conv.stride != (1, 1):
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels=first_conv.in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=(1, 1),
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, self.num_classes)
