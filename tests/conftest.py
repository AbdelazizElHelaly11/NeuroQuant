"""
NeuroQuant v2.0 — Shared pytest fixtures (Wave 6 K1).

Concentrates the boilerplate that was duplicated across the wave-1
through wave-5 test files into a single project-wide conftest. Tests
opt in by name (``def test_x(tiny_model, calib_loader): ...``); the
old per-file ``TinyCNN`` / ``calib_loader`` definitions still work,
so existing tests continue to pass without modification.

Provided fixtures:

    tiny_cnn_factory    Callable that builds a fresh ``TinyCNN`` with
                        a fixed manual seed. Use this when a test
                        mutates the model and needs a clean copy.
    tiny_model          Convenience instance of ``TinyCNN`` in eval
                        mode, suitable for read-only tests.
    calib_loader        Tiny synthetic CIFAR-class calibration loader
                        (32 samples, batch 8). Deterministic across
                        runs.
    val_loader          Same shape as ``calib_loader`` but built from
                        a different seed; useful when a test needs
                        two non-overlapping loaders.
    quant_config        Default ``QuantizationConfig`` configured for
                        CPU + seed=0. Tests that need custom values
                        can mutate the returned object — fixtures are
                        function-scoped so mutations don't leak.
    pipeline_skeleton   Minimal ``NeuroQuantPipeline`` instance with
                        the bare attributes set so reporting/summary
                        helpers can be exercised without running the
                        full pipeline.

Module-scoped models are deliberately avoided: pytest's parallel
runners can mutate shared instances unexpectedly. Function scope is
~1ms more per test on the TinyCNN — well worth the isolation.
"""
from __future__ import annotations

from typing import Callable

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reference model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TinyCNN(nn.Module):
    """Tiny CIFAR-class CNN reused throughout the test suite.

    Three Conv2d layers + AdaptiveAvgPool + Linear classifier. Chosen
    so that all six wave-3 method audits (PTQ, QAT, GPTQ,
    SmoothQuant, AWQ, SmoothQuant→GPTQ) execute end-to-end in <2s on
    CPU. Tests that need a different architecture should build their
    own — this is the *common* fixture, not the *only* one.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def tiny_cnn_factory() -> Callable[..., TinyCNN]:
    """Return a builder that produces deterministic TinyCNN instances.

    The factory accepts an optional ``num_classes`` and an optional
    ``seed`` kwarg so tests that need *two* independently-initialised
    models (e.g. teacher / student for KD) can request them with
    different seeds.
    """

    def _build(num_classes: int = 10, seed: int = 0) -> TinyCNN:
        torch.manual_seed(int(seed))
        return TinyCNN(num_classes=num_classes).eval()

    return _build


@pytest.fixture
def tiny_model(tiny_cnn_factory) -> TinyCNN:
    """A single TinyCNN instance, eval mode, deterministic seed."""
    return tiny_cnn_factory()


@pytest.fixture
def calib_loader() -> DataLoader:
    """Synthetic 32-sample CIFAR-class calibration loader (batch 8).

    Same shape used by every wave 1–5 test that exercises the
    quantization pipeline. The seed is pinned so two test runs see
    bit-identical activation distributions.
    """
    torch.manual_seed(0)
    xs = torch.randn(32, 3, 32, 32)
    ys = torch.zeros(32, dtype=torch.long)
    return DataLoader(TensorDataset(xs, ys), batch_size=8, shuffle=False)


@pytest.fixture
def val_loader() -> DataLoader:
    """Second loader with the same shape as ``calib_loader`` but a
    different seed. Use when a test needs two non-overlapping loaders.
    """
    torch.manual_seed(7)
    xs = torch.randn(32, 3, 32, 32)
    ys = torch.zeros(32, dtype=torch.long)
    return DataLoader(TensorDataset(xs, ys), batch_size=8, shuffle=False)


@pytest.fixture
def quant_config():
    """Default ``QuantizationConfig`` pinned to CPU + seed=0.

    Tests can mutate fields (``cfg.hyperparams.qat_epochs = 1``)
    freely because the fixture is function-scoped.
    """
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.seed = 0
    return cfg


@pytest.fixture
def pipeline_skeleton(quant_config):
    """Bare ``NeuroQuantPipeline`` for testing reporting helpers.

    Skips ``run()`` and only sets the attributes the report-printing
    and summary-building methods read. Use this fixture instead of
    constructing a pipeline by hand in every wave-5+ test.
    """
    from main import NeuroQuantPipeline

    p = NeuroQuantPipeline(quant_config)
    p.fp32_acc = 90.0
    p.fp32_size_mb = 8.5
    p.fp32_onnx = {}
    p.method_results = []
    p.pareto_analysis = {}
    p.qat_result = {}
    return p
