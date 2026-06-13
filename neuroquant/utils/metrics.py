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
    *,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Top-1 / top-k accuracy, or segmentation **mIOU** when the model emits
    per-pixel logits.

    The function dispatches on the *runtime* output shape, per batch, so
    every accuracy consumer in the pipeline (the FP32 baseline, NSGA
    fitness, method headlines) gets a meaningful "higher is better"
    scalar without any per-call-site task plumbing:

      * **Classification** — ``[B, C]`` logits → standard top-1 / top-k.
        ``k`` is clamped to ``C`` (safe for CIFAR-10 with ``k=5`` etc.).
      * **Segmentation** — ``OrderedDict({"out": [B, C, H, W]})`` (or a
        bare ``[B, C, H, W]`` tensor). We accumulate a per-class
        intersection/union confusion matrix and return **mIOU in the
        ``top1`` slot** (so the same "higher = better" contract holds)
        plus pixel accuracy in ``top5``; the explicit ``miou`` /
        ``pixel_acc`` keys are also provided. ``ignore_index`` (255 by
        convention, e.g. Pascal VOC boundary pixels) is excluded.

    Returns percentages in ``[0, 100]``. The ``top1`` / ``top5`` keys are
    always present so task-agnostic callers keep working.
    """
    model.eval()
    model.to(device)

    # Classification accumulators.
    correct_top1 = 0
    correct_topk = 0
    total = 0
    # Segmentation accumulators (lazily allocated on the first 4-D output).
    seg = False
    inter: Optional[torch.Tensor] = None
    union: Optional[torch.Tensor] = None
    pix_correct = 0
    pix_total = 0

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)

            # Segmentation models return an OrderedDict; unwrap to the
            # main per-pixel logits tensor.
            if isinstance(outputs, dict):
                outputs = outputs.get("out", next(iter(outputs.values())))
            if not isinstance(outputs, torch.Tensor):
                # e.g. a detection model's list-of-dicts — no top-k notion
                # here; skip so the eval degrades gracefully rather than
                # crashing on ``.shape``.
                continue

            if outputs.dim() == 4:
                # ── Segmentation: [B, C, H, W] → per-class IoU + pixel acc ──
                seg = True
                num_classes = outputs.shape[1]
                if inter is None:
                    inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
                    union = torch.zeros(num_classes, dtype=torch.float64, device=device)
                pred = outputs.argmax(dim=1)                       # [B, H, W]
                if labels.dim() == 4 and labels.size(1) == 1:
                    labels = labels.squeeze(1)
                labels = labels.long()
                valid = labels != ignore_index
                for c in range(num_classes):
                    pred_c = (pred == c) & valid
                    label_c = (labels == c) & valid
                    inter[c] += (pred_c & label_c).sum()
                    union[c] += (pred_c | label_c).sum()
                pix_correct += int(((pred == labels) & valid).sum().item())
                pix_total += int(valid.sum().item())
            else:
                # ── Classification: [B, C] → top-1 / top-k ──
                num_classes = outputs.shape[1]
                actual_k = min(k, num_classes)
                _, pred_top1 = outputs.max(1)
                correct_top1 += pred_top1.eq(labels).sum().item()
                _, pred_topk = outputs.topk(actual_k, dim=1, largest=True, sorted=True)
                correct_topk += pred_topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

    if seg and inter is not None and union is not None:
        iou = inter / union.clamp(min=1.0)
        present = union > 0          # ignore classes absent from this split
        miou = float(iou[present].mean().item() * 100.0) if bool(present.any()) else 0.0
        pix_acc = (pix_correct / max(pix_total, 1)) * 100.0
        # mIOU rides in the ``top1`` slot so the pipeline's "higher is
        # better" objective math (fp32 − quant) works unchanged.
        return {"top1": miou, "top5": pix_acc, "miou": miou, "pixel_acc": pix_acc}

    top1 = (correct_top1 / max(total, 1)) * 100.0
    topk = (correct_topk / max(total, 1)) * 100.0

    return {"top1": top1, "top5": topk}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Object-Detection mAP@0.5 (self-contained, VOC-style, no extra deps)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of xyxy boxes → ``[len(a), len(b)]``."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-points (VOC2010+) average precision: area under the P-R curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing (envelope).
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_detection_map(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_images: Optional[int] = None,
) -> Dict[str, float]:
    """Mean Average Precision @ IoU=0.5 for a torchvision detection model.

    Self-contained VOC-style mAP — no ``pycocotools`` / ``torchmetrics``
    dependency. The model is run in eval mode (so it returns a
    ``List[Dict{boxes, labels, scores}]`` per batch); ground-truth boxes
    come from the loader's detection collate (``(images, targets)`` with
    ``targets`` a tuple of ``{"boxes", "labels"}`` dicts).

    Algorithm (per class, standard VOC):
      1. Gather every predicted box across all images with its score.
      2. Sort predictions by descending score.
      3. Greedily match each prediction to an unmatched GT of the same
         class with IoU ≥ ``iou_threshold`` → TP, else FP.
      4. Accumulate precision/recall and integrate (all-points AP).
    AP is averaged over the classes that actually appear in the GT, so
    the COCO 91-id space with absent classes does not dilute the mean.

    The result rides in the ``top1`` slot (× 100) so the pipeline's
    task-agnostic "higher is better" objective math (``fp32 − quant``)
    optimises the mAP drop directly — exactly like segmentation reuses
    that slot for mIOU.
    """
    model.eval()
    model.to(device)

    # Per-class prediction accumulators and GT counts.
    preds_by_class: Dict[int, List[Tuple[float, int, torch.Tensor]]] = {}
    gts_by_class: Dict[int, Dict[int, torch.Tensor]] = {}
    gt_count: Dict[int, int] = {}

    img_idx = 0
    seen = 0
    with torch.no_grad():
        for batch in data_loader:
            if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
                continue
            images, targets = batch
            # Detection collate yields a tuple of per-image tensors.
            if isinstance(images, torch.Tensor):
                images = [images[i] for i in range(images.shape[0])]
            images = [img.to(device) for img in images]

            outputs = model(images)
            if not isinstance(outputs, (list, tuple)):
                # Not a detection forward — nothing to score.
                return {"top1": 0.0, "top5": 0.0, "map": 0.0, "map_50": 0.0}

            for out, tgt in zip(outputs, targets):
                # ── Ground truth for this image ──
                gt_boxes = tgt.get("boxes") if isinstance(tgt, dict) else None
                gt_labels = tgt.get("labels") if isinstance(tgt, dict) else None
                if gt_boxes is None:
                    gt_boxes = torch.zeros((0, 4))
                    gt_labels = torch.zeros((0,), dtype=torch.int64)
                gt_boxes = gt_boxes.to(device).float()
                gt_labels = gt_labels.to(device).long()
                for c in gt_labels.unique().tolist():
                    mask = gt_labels == c
                    gts_by_class.setdefault(c, {})[img_idx] = gt_boxes[mask]
                    gt_count[c] = gt_count.get(c, 0) + int(mask.sum().item())

                # ── Predictions for this image ──
                p_boxes = out.get("boxes", torch.zeros((0, 4), device=device))
                p_scores = out.get("scores", torch.zeros((0,), device=device))
                p_labels = out.get("labels", torch.zeros((0,), dtype=torch.int64, device=device))
                keep = p_scores >= score_threshold
                p_boxes, p_scores, p_labels = p_boxes[keep], p_scores[keep], p_labels[keep]
                for c in p_labels.unique().tolist():
                    cmask = p_labels == c
                    for box, sc in zip(p_boxes[cmask], p_scores[cmask]):
                        preds_by_class.setdefault(c, []).append(
                            (float(sc.item()), img_idx, box)
                        )

                img_idx += 1
                seen += 1
            if max_images is not None and seen >= max_images:
                break

    if not gt_count:
        return {"top1": 0.0, "top5": 0.0, "map": 0.0, "map_50": 0.0}

    aps: List[float] = []
    for c, n_gt in gt_count.items():
        preds = sorted(preds_by_class.get(c, []), key=lambda r: r[0], reverse=True)
        if n_gt == 0:
            continue
        tp = np.zeros(len(preds), dtype=np.float64)
        fp = np.zeros(len(preds), dtype=np.float64)
        matched: Dict[int, set] = {}
        gt_for_c = gts_by_class.get(c, {})
        for i, (_, im_id, box) in enumerate(preds):
            gboxes = gt_for_c.get(im_id)
            if gboxes is None or gboxes.numel() == 0:
                fp[i] = 1.0
                continue
            ious = _box_iou_xyxy(box[None, :].to(gboxes.device), gboxes)[0]
            best = int(torch.argmax(ious).item())
            if float(ious[best].item()) >= iou_threshold and best not in matched.setdefault(im_id, set()):
                tp[i] = 1.0
                matched[im_id].add(best)
            else:
                fp[i] = 1.0
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        aps.append(_voc_ap(recall, precision))

    mean_ap = float(np.mean(aps) * 100.0) if aps else 0.0
    return {"top1": mean_ap, "top5": mean_ap, "map": mean_ap, "map_50": mean_ap}


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
    if task == "detection":
        # Detection batches are ``(List[Tensor], List[Dict])`` — they would
        # crash ``compute_topk_accuracy`` (which assumes a stackable
        # ``batch[0]``). Route to the self-contained VOC mAP@0.5, which
        # also rides its result in the ``top1`` slot so the rest of the
        # pipeline stays task-agnostic.
        return compute_detection_map(model, data_loader, device)
    # classification / segmentation / nlp funnel through
    # ``compute_topk_accuracy``, which dispatches on the output shape:
    # classification yields top-1 and **segmentation yields mIOU** in the
    # ``top1`` slot (so NSGA's ``fp32 − quant`` objective optimises the
    # mIOU drop directly).
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
