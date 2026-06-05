"""Smoke tests for the regression task path.

These don't try to learn anything — they verify the plumbing:
``task='regression'`` is accepted by the config, the loss bridge
returns MSE, and ``compute_regression_metrics`` reports RMSE / MAE / R²
with the ``top1`` "higher is better" invariant satisfied.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuroquant import QuantizationConfig
from neuroquant.utils.metrics import (
    compute_regression_metrics,
    evaluate_primary_metric,
)


@pytest.fixture
def regression_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 1))


@pytest.fixture
def regression_loader() -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(16, 3, 8, 8)
    y = torch.randn(16, 1)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_regression_task_accepted_by_config() -> None:
    cfg = QuantizationConfig(task="regression")
    assert cfg.task == "regression"


def test_compute_regression_metrics_shape(regression_model, regression_loader) -> None:
    out = compute_regression_metrics(
        regression_model, regression_loader, device=torch.device("cpu"),
    )
    for key in ("rmse", "mae", "r2", "top1", "top5"):
        assert key in out
    assert out["rmse"] >= 0.0
    assert out["mae"] >= 0.0
    # ``top1`` MUST be "higher is better" — for regression that means
    # the negated RMSE.
    assert out["top1"] == pytest.approx(-out["rmse"])


def test_evaluate_primary_metric_routes_to_regression(
    regression_model, regression_loader,
) -> None:
    cls_metrics = evaluate_primary_metric(
        regression_model, regression_loader,
        device=torch.device("cpu"), task="classification",
    )
    reg_metrics = evaluate_primary_metric(
        regression_model, regression_loader,
        device=torch.device("cpu"), task="regression",
    )
    # Different routes → different numbers.
    assert "rmse" in reg_metrics
    assert "rmse" not in cls_metrics
