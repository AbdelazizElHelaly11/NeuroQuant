"""
NeuroQuant v2.0 - Metric Utilities

Reusable metric computation for all evaluation paths:
    - Top-k accuracy (top-1 + top-5, safe for num_classes < 5)
    - Inference latency benchmarking (CPU/CUDA aware)
    - Hardware synthesis report parser (JSON/CSV)

No dependencies on NeuroQuant internals — pure torch + numpy.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("neuroquant")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Top-k Accuracy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_topk_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    k: int = 5,
) -> Dict[str, float]:
    """
    Compute top-1 and top-k accuracy on a dataset.

    Handles num_classes < k by clamping k = min(k, num_classes).
    Returns percentages in [0, 100].

    Args:
        model: Model to evaluate (moved to device automatically).
        data_loader: Test/val DataLoader.
        device: Compute device.
        k: Maximum k for top-k (default 5).

    Returns:
        {"top1": float, "top5": float}
        "top5" key is always present but uses k=min(k, C).
    """
    model.eval()
    model.to(device)

    correct_top1 = 0
    correct_topk = 0
    total = 0
    actual_k = k  # Will be clamped on first batch

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)

            # Clamp k to num_classes (safe for CIFAR-10 with k=5, or 3-class with k=5)
            num_classes = outputs.shape[1]
            actual_k = min(k, num_classes)

            # Top-1
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(labels).sum().item()

            # Top-k
            _, pred_topk = outputs.topk(actual_k, dim=1, largest=True, sorted=True)
            correct_topk += pred_topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    top1 = (correct_top1 / max(total, 1)) * 100.0
    topk = (correct_topk / max(total, 1)) * 100.0

    return {"top1": top1, "top5": topk}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Regression Metrics (RMSE / MAE / R²)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_regression_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute RMSE, MAE, and R² on a regression dataset.

    Returns a dict shaped so the NSGA-II search can keep its
    "lower-is-better, scalar primary" contract:

    * ``rmse`` — root mean squared error (primary objective). Used by
      ``evaluate_primary_metric`` to populate the same slot
      classification uses for top-1, so the rest of the pipeline stays
      task-agnostic.
    * ``mae``  — mean absolute error (secondary diagnostic).
    * ``r2``   — coefficient of determination, included for parity with
      sklearn / statsmodels reporting.
    * ``top1`` — set to ``-rmse`` (negated) so any legacy caller that
      reads ``["top1"]`` and expects "higher is better" sees
      monotonically-correct rankings. Documented contract:
      ``higher_better=True`` for ``top1`` in *every* task family.
    """
    model.eval()
    model.to(device)

    sum_sq_err = 0.0
    sum_abs_err = 0.0
    sum_targets = 0.0
    sum_targets_sq = 0.0
    n = 0

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch[0], batch[1]
            if isinstance(x, (list, tuple)):
                x = [t.to(device) for t in x]
            else:
                x = x.to(device)
            y = y.to(device).float()
            out = model(x)
            if isinstance(out, dict):
                out = out.get("out", next(iter(out.values())))
            # Flatten to ``[B*K]`` so single-output and multi-output
            # regressions share the same accumulator path.
            out_flat = out.reshape(-1).float()
            y_flat = y.reshape(-1).float()
            if out_flat.numel() != y_flat.numel():
                # Defensive: if the model has multiple regression heads
                # but the target only covers one, slice to match.
                m = min(out_flat.numel(), y_flat.numel())
                out_flat = out_flat[:m]
                y_flat = y_flat[:m]

            err = out_flat - y_flat
            sum_sq_err += float((err * err).sum().item())
            sum_abs_err += float(err.abs().sum().item())
            sum_targets += float(y_flat.sum().item())
            sum_targets_sq += float((y_flat * y_flat).sum().item())
            n += int(y_flat.numel())

    if n == 0:
        return {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "top1": 0.0, "top5": 0.0}

    mse = sum_sq_err / n
    rmse = float(np.sqrt(mse))
    mae = sum_abs_err / n
    mean_y = sum_targets / n
    ss_tot = sum_targets_sq - n * mean_y * mean_y
    r2 = 1.0 - (sum_sq_err / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        # ``top1`` / ``top5`` slots are populated with ``-rmse`` so any
        # task-agnostic caller that pulls ``["top1"]`` and treats higher
        # as better stays correct on regression too.
        "top1": -rmse,
        "top5": -rmse,
    }


def evaluate_primary_metric(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    task: str = "classification",
) -> Dict[str, float]:
    """Single entry point for "evaluate a model and give me a scalar".

    Routes to :func:`compute_topk_accuracy` for classification (and
    its degenerate-but-safe behaviour on detection/segmentation when
    the upstream caller already extracted a logits view), and to
    :func:`compute_regression_metrics` for regression. The returned
    dict *always* carries a ``top1`` key whose value is "higher is
    better" — that's the invariant NSGA-II's ``_evaluate_accuracy``
    relies on to keep its objective math task-agnostic.
    """
    if task == "regression":
        return compute_regression_metrics(model, data_loader, device)
    # classification / detection / segmentation / nlp all funnel through
    # the topk path. For detection/segmentation the caller is expected
    # to compute task-native metrics separately (mAP, mIoU); the topk
    # number is only used by NSGA's surrogate when explicit labels are
    # present.
    return compute_topk_accuracy(model, data_loader, device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Latency Benchmarking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def benchmark_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    batch_size: int = 1,
    warmup_runs: int = 10,
    measure_runs: int = 50,
) -> Dict[str, float]:
    """
    Benchmark inference latency with warmup + repeated timed runs.

    Handles CPU and CUDA correctly (uses torch.cuda.Event for GPU timing).

    Args:
        model: Model to benchmark.
        input_shape: (C, H, W) — batch dim added automatically.
        device: Compute device.
        batch_size: Batch size for benchmark input.
        warmup_runs: Number of warmup forward passes (not timed).
        measure_runs: Number of timed forward passes.

    Returns:
        {"latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "throughput_fps"}
    """
    model.eval()
    model.to(device)

    dummy = torch.randn(batch_size, *input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(dummy)

    # CUDA sync before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings_ms: List[float] = []

    with torch.no_grad():
        if device.type == "cuda":
            # Use CUDA events for precise GPU timing
            for _ in range(measure_runs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                model(dummy)
                end_event.record()
                torch.cuda.synchronize()
                timings_ms.append(start_event.elapsed_time(end_event))
        else:
            # CPU timing
            for _ in range(measure_runs):
                t0 = time.perf_counter()
                model(dummy)
                t1 = time.perf_counter()
                timings_ms.append((t1 - t0) * 1000.0)

    arr = np.array(timings_ms)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    throughput = (batch_size / mean_ms) * 1000.0 if mean_ms > 0 else 0.0

    return {
        "latency_mean_ms": round(mean_ms, 3),
        "latency_p50_ms": round(p50_ms, 3),
        "latency_p95_ms": round(p95_ms, 3),
        "throughput_fps": round(throughput, 1),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hardware Synthesis Report Parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# Canonical field names we look for (case-insensitive)
_HW_FIELD_MAP = {
    "dsp": "dsp",
    "dsp48e": "dsp",
    "dsp_blocks": "dsp",
    "lut": "lut",
    "luts": "lut",
    "logic_elements": "lut",
    "ff": "ff",
    "flip_flops": "ff",
    "registers": "ff",
    "fmax": "fmax_mhz",
    "fmax_mhz": "fmax_mhz",
    "frequency_mhz": "fmax_mhz",
    "ii": "ii",
    "initiation_interval": "ii",
    "cycle_latency": "cycle_latency",
    "latency_cycles": "cycle_latency",
    "pipeline_depth": "cycle_latency",
}


def parse_hardware_report(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse hardware metrics from an external synthesis report.

    Supports JSON and CSV formats. Maps various vendor field names
    (Vivado HLS, Quartus, etc.) to canonical NeuroQuant names.

    If path is None or file doesn't exist, returns all-null metrics
    with source="not_provided".

    Args:
        path: Path to synthesis report (JSON or CSV).

    Returns:
        {"dsp": int|None, "lut": int|None, "ff": int|None,
         "fmax_mhz": float|None, "ii": int|None,
         "cycle_latency": int|None, "source": str}
    """
    result = {
        "dsp": None,
        "lut": None,
        "ff": None,
        "fmax_mhz": None,
        "ii": None,
        "cycle_latency": None,
        "source": "not_provided",
    }

    if not path:
        return result

    p = Path(path)
    if not p.exists():
        logger.warning("Hardware report not found: %s", path)
        return result

    try:
        if p.suffix == ".json":
            raw = _parse_json_report(p)
        elif p.suffix == ".csv":
            raw = _parse_csv_report(p)
        else:
            # Try JSON first, then CSV
            try:
                raw = _parse_json_report(p)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raw = _parse_csv_report(p)

        # Map vendor field names to canonical names
        for raw_key, raw_val in raw.items():
            canonical = _HW_FIELD_MAP.get(raw_key.lower().strip())
            if canonical and raw_val is not None:
                try:
                    if canonical == "fmax_mhz":
                        result[canonical] = float(raw_val)
                    else:
                        result[canonical] = int(raw_val)
                except (ValueError, TypeError):
                    pass

        # Determine source from file
        result["source"] = p.stem
        if any(v is not None for k, v in result.items() if k != "source"):
            logger.info("Hardware metrics loaded from: %s", path)
        else:
            logger.warning("Hardware report parsed but no known fields found: %s", path)
            result["source"] = "not_provided"

    except Exception as e:
        logger.warning("Failed to parse hardware report '%s': %s", path, e)

    return result


def _parse_json_report(path: Path) -> Dict[str, Any]:
    """Parse a JSON synthesis report (flat or nested)."""
    with open(path, "r") as f:
        data = json.load(f)

    # Flatten one level of nesting (e.g., {"resource": {"dsp": 10}})
    flat: Dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                flat[k2] = v2
        else:
            flat[key] = val
    return flat


def _parse_csv_report(path: Path) -> Dict[str, Any]:
    """Parse a CSV synthesis report (first row = headers, second row = values)."""
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return {}

    headers = [h.strip() for h in rows[0]]
    values = [v.strip() for v in rows[1]]
    return dict(zip(headers, values))
