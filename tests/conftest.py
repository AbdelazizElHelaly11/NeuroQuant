"""Shared pytest fixtures.

Keeps the tiny synthetic model / loader pair in one place so individual
tests stay one screen long. Everything here is CPU-only and finishes in
under a second — fast enough that CI can run on a free GitHub runner.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(scope="session")
def toy_model() -> nn.Module:
    """A 3-layer CNN small enough that PTQ / Fisher / NSGA finish instantly.

    Picked to exercise both Conv2d and Linear (the two quantizable module
    types NeuroQuant supports), plus BatchNorm and ReLU to confirm the
    quantizer leaves non-quantizable modules alone.
    """
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4),
    )


@pytest.fixture(scope="session")
def toy_loader() -> DataLoader:
    """Synthetic (image, label) batches mimicking a 4-class CIFAR-shaped feed."""
    torch.manual_seed(0)
    images = torch.randn(16, 3, 8, 8)
    labels = torch.randint(0, 4, (16,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)
