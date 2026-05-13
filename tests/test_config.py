"""``QuantizationConfig`` default-construction tests.

The library-mode contract is that ``Quantizer(model)`` without a config
just works because ``QuantizationConfig()`` succeeds with every field
having a sensible default. These tests pin that invariant.
"""
from __future__ import annotations

import pytest

from neuroquant import QuantizationConfig


def test_default_construction() -> None:
    cfg = QuantizationConfig()
    # Tasks the rest of the codebase dispatches on.
    assert cfg.task in {"classification", "detection", "segmentation"}
    # Hyperparams substructure must exist (it's what every phase reads).
    assert cfg.hyperparams is not None
    assert cfg.hyperparams.seed is not None


def test_task_validator_rejects_bogus_values() -> None:
    with pytest.raises(Exception):  # pydantic ValidationError or stdlib ValueError
        QuantizationConfig(task="object_detectoooooon")


@pytest.mark.parametrize("task", ["classification", "detection", "segmentation"])
def test_all_three_tasks_accepted(task: str) -> None:
    cfg = QuantizationConfig(task=task)
    assert cfg.task == task
