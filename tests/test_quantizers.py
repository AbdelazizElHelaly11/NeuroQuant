"""End-to-end smoke tests for the headline quantizers.

These don't aim for correctness numbers — they prove that the
config-less library path advertised in ``docs/library_mode.md`` actually
runs without raising on a real (if tiny) model. If any of these fail,
the README is lying.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuroquant import (
    AWQQuantizer,
    PTQQuantizer,
    QuantizationConfig,
)


def test_ptq_int8_runs_config_less(toy_model, toy_loader) -> None:
    """``PTQQuantizer(model).quantize(loader, bitwidth=8)`` — the
    three-line library example from docs must execute."""
    quantizer = PTQQuantizer(toy_model)
    q_model = quantizer.quantize(toy_loader, bitwidth=8)
    assert isinstance(q_model, nn.Module)
    # Forward pass on a fresh batch — checks that the returned module
    # is actually callable, not just constructed.
    images, _ = next(iter(toy_loader))
    with torch.no_grad():
        out = q_model(images)
    assert out.shape == (images.size(0), 4)


def test_ptq_int4_runs(toy_model, toy_loader) -> None:
    """INT4 path — uses the same wrapper but exercises the lower bitwidth."""
    q_model = PTQQuantizer(toy_model).quantize(toy_loader, bitwidth=4)
    images, _ = next(iter(toy_loader))
    with torch.no_grad():
        q_model(images)


def test_awq_detection_guard_raises_clearly() -> None:
    """AWQ must reject ``task='detection'`` up-front rather than crashing
    deep inside ``torch.cat`` with an unhelpful traceback."""
    cfg = QuantizationConfig(task="detection")
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.Linear(4, 2))
    awq = AWQQuantizer(model, cfg)
    # We don't actually want to drive a detection loader; the guard fires
    # at the top of ``quantize``, before any calibration starts.
    import torch.utils.data as data

    dummy = data.DataLoader(
        data.TensorDataset(torch.randn(2, 3, 4, 4), torch.zeros(2, dtype=torch.long)),
        batch_size=2,
    )
    with pytest.raises(NotImplementedError, match="AWQ does not support detection"):
        awq.quantize(dummy, bitwidth=4)
