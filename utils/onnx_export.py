"""
NeuroQuant v2.0 — ONNX export, static INT8 quantization, and ORT latency.

This module is the production-grade bridge from "INT simulation in FP32
tensors" to "real INT8 ONNX model on disk, benchmarked with ONNX Runtime":

    1. ``export_to_onnx``        — FP32 PyTorch → ONNX graph
    2. ``quantize_onnx_static``  — FP32 ONNX → INT8 ONNX via ORT static
                                    quantization with the framework's own
                                    calibration loader.
    3. ``onnx_disk_size_mb``     — Real on-disk file size of the .onnx
                                    artefact. Replaces the synthetic
                                    ``numel × bitwidth / 8`` model-size
                                    objective with the deployable number.
    4. ``benchmark_onnx_latency``— ORT inference latency (warmup +
                                    timed runs, percentiles).

If ``onnx`` or ``onnxruntime`` are not installed, every public function
raises ``OnnxUnavailable`` — callers must handle that explicitly. We
deliberately do NOT silently fall back to a non-ONNX path: the public
contract is "real INT8 ONNX measurement" and a fake answer is worse
than no answer.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("neuroquant")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optional-dependency guard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OnnxUnavailable(RuntimeError):
    """Raised when onnx / onnxruntime are not importable.

    Catch this in callers that should degrade gracefully (e.g. the
    NSGA latency LUT builder, the per-method export hook). Do *not*
    catch it in tests asserting the ONNX contract — those should xfail
    or skip when the dependency is missing.
    """


def _require_onnx() -> Tuple[Any, Any, Any]:
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime import quantization as ort_quant
    except ImportError as exc:  # pragma: no cover — depends on env
        raise OnnxUnavailable(
            "onnx and onnxruntime must be installed for ONNX export / "
            "static quantization / latency benchmarks. "
            "Install with: pip install onnx onnxruntime"
        ) from exc
    return onnx, ort, ort_quant


def is_onnx_available() -> bool:
    """Return True iff onnx and onnxruntime can be imported.

    Use this to gate optional code paths (e.g. LUT construction at
    pipeline start) without paying the import cost twice.
    """
    try:
        _require_onnx()
        return True
    except OnnxUnavailable:
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. FP32 export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    *,
    batch_size: int = 1,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> str:
    """Export a PyTorch FP32 model to an ONNX file on disk.

    The model is moved to CPU and ``eval`` mode for tracing — both are
    requirements of ``torch.onnx.export``. We use opset 17 because it
    is the floor for ORT's static quantization preprocessing
    (``QuantFormat.QDQ``); older opsets miss ops the quantizer rewrites.

    Args:
        model:        FP32 PyTorch module.
        input_shape:  ``(C, H, W)``. The batch dim is added by this
                      function (default 1).
        output_path:  Destination ``.onnx`` path.
        batch_size:   Trace batch size (default 1 — matches latency
                      benchmark default).
        opset:        ONNX opset version. Bump only if you know what you
                      are doing.
        dynamic_batch:If True, batch dim becomes a dynamic axis so the
                      same .onnx works for any batch at inference time.

    Returns:
        ``output_path`` as a string for easy chaining.
    """
    _require_onnx()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Trace on CPU. Quantized models containing custom modules
    # (SmoothQuant input-scale wrappers, AWQ wrappers) trace fine — the
    # wrapper's forward is pure tensor ops.
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(batch_size, *input_shape)

    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }

    # Use the legacy (TorchScript-based) exporter explicitly. The
    # dynamo path is the future default but still warns on
    # ``dynamic_axes`` and emits opsets that the ORT static-quantization
    # preprocessor cannot version-convert. The legacy path is stable
    # across torch 2.4–2.11 and produces graphs the ORT INT8 quantizer
    # accepts without preprocessing tricks.
    export_kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    try:
        torch.onnx.export(
            model_cpu, dummy, str(out),
            dynamo=False, **export_kwargs,
        )
    except TypeError:
        # torch < 2.5 doesn't accept the ``dynamo`` kwarg.
        torch.onnx.export(model_cpu, dummy, str(out), **export_kwargs)
    logger.info(
        "ONNX export: %s (opset=%d, dynamic_batch=%s, %d bytes)",
        out.name, opset, dynamic_batch, out.stat().st_size,
    )
    return str(out)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Static INT8 quantization (J1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_calibration_reader(
    calibration_loader: DataLoader,
    num_batches: int,
    input_name: str = "input",
):
    """Build an ORT ``CalibrationDataReader`` from a torch ``DataLoader``.

    ORT iterates the reader once per calibration pass; each ``get_next``
    call must return ``{input_name: numpy_array}`` or ``None`` to stop.
    The reader caches all batches up-front so it can be replayed if ORT
    decides to do multi-pass calibration.
    """
    _, _, ort_quant = _require_onnx()

    cached: List[Dict[str, np.ndarray]] = []
    for i, batch in enumerate(calibration_loader):
        if i >= num_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        # Detach + cpu + numpy; ORT requires numpy arrays.
        cached.append({input_name: x.detach().cpu().numpy().astype(np.float32)})

    class _Reader(ort_quant.CalibrationDataReader):
        def __init__(self) -> None:
            self._iter = iter(cached)

        def get_next(self) -> Optional[Dict[str, np.ndarray]]:
            return next(self._iter, None)

        def rewind(self) -> None:
            self._iter = iter(cached)

    return _Reader()


def quantize_onnx_static(
    fp32_onnx_path: str,
    output_path: str,
    calibration_loader: DataLoader,
    *,
    num_batches: int = 10,
    activation_int8: bool = True,
    weight_int8: bool = True,
    per_channel: bool = True,
) -> str:
    """Quantize an FP32 ONNX model to INT8 via ORT static quantization.

    "Static" here means: ORT walks the graph, replays the calibration
    loader through the FP32 model, records activation min/max per
    tensor, and rewrites the graph with QDQ (Quantize-Dequantize) nodes
    so all matmuls/convs run in INT8 at inference time. This is the
    deployment path: the ``.onnx`` written here is what a real ONNX
    Runtime / TensorRT / OpenVINO deployment loads.

    Args:
        fp32_onnx_path:      Source .onnx file (must already be exported).
        output_path:         Destination INT8 .onnx file.
        calibration_loader:  The same loader used for PTQ calibration.
        num_batches:         How many calibration batches to feed ORT.
        activation_int8:     QInt8 activations (deployment default).
        weight_int8:         QInt8 weights (deployment default).
        per_channel:         Per-channel weight quantization (recommended).

    Returns:
        ``output_path`` as string.
    """
    _, _, ort_quant = _require_onnx()

    src = Path(fp32_onnx_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # ORT's preprocess step inserts shape/identity ops the quantizer
    # needs. Skipping it produces "Unable to quantize node X" warnings
    # and a worse INT8 graph; running it costs <1s on small models. If
    # the preprocessor fails (e.g. opset version mismatch), fall back
    # to quantizing the raw FP32 graph — ORT will warn but still
    # produce a working INT8 model.
    preproc = src.with_suffix(".preproc.onnx")
    used_preproc = False
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(src), str(preproc))
        src_for_quant = str(preproc)
        used_preproc = True
    except Exception as exc:
        logger.debug("quant_pre_process unavailable (%s); using raw FP32.", exc)
        src_for_quant = str(src)

    reader = _build_calibration_reader(calibration_loader, num_batches)

    a_type = ort_quant.QuantType.QInt8 if activation_int8 else ort_quant.QuantType.QUInt8
    w_type = ort_quant.QuantType.QInt8 if weight_int8 else ort_quant.QuantType.QUInt8

    ort_quant.quantize_static(
        model_input=src_for_quant,
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=ort_quant.QuantFormat.QDQ,
        per_channel=per_channel,
        activation_type=a_type,
        weight_type=w_type,
    )

    # Best-effort cleanup of the preprocess artefact. Never unlink the
    # original FP32 graph — only the preproc temporary.
    if used_preproc:
        try:
            preproc.unlink(missing_ok=True)
        except Exception:
            pass

    logger.info(
        "ONNX static INT8 quantization: %s → %s (%d → %d bytes)",
        src.name, dst.name, src.stat().st_size, dst.stat().st_size,
    )
    return str(dst)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Real on-disk size (J3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def onnx_disk_size_mb(onnx_path: str) -> float:
    """Return the actual ``.onnx`` file size in MiB (1024 × 1024)."""
    p = Path(onnx_path)
    if not p.exists():
        raise FileNotFoundError(f"No ONNX file at: {onnx_path}")
    return p.stat().st_size / (1024.0 * 1024.0)


def estimate_int4_packed_size_mb(
    model: nn.Module,
    bitwidth_assignment: Dict[str, int],
) -> Dict[str, float]:
    """Estimate the on-disk size if INT4 weights were properly packed.

    ONNX Runtime's ``quantize_static`` only supports QInt8 — it does
    not natively pack INT4 values (two INT4 values per byte). This
    function computes the theoretical packed size so users can compare:

        - ``onnx_size_mb``: actual ``.onnx`` file size (INT8 container)
        - ``packed_size_mb``: what it *would* be with proper INT4 packing
        - ``packing_savings_mb``: the difference (wasted space)

    The estimate is: for each parameter assigned INT4, the on-disk cost
    is ``numel / 2`` bytes (4 bits each, packed in pairs). INT8 params
    cost ``numel`` bytes. Everything else stays at FP32 (4 bytes).

    Returns:
        Dict with ``packed_size_mb``, ``unpacked_size_mb``,
        ``packing_savings_mb``, and ``packing_note``.
    """
    packed_bytes = 0.0
    unpacked_bytes = 0.0

    for name, p in model.named_parameters():
        numel = p.numel()
        bw = bitwidth_assignment.get(name, 32)
        if bw == 4:
            packed_bytes += numel * 0.5     # 4 bits packed
            unpacked_bytes += numel * 1.0   # stored as INT8 in ONNX
        elif bw == 8:
            packed_bytes += numel * 1.0
            unpacked_bytes += numel * 1.0
        else:
            packed_bytes += numel * 4.0     # FP32
            unpacked_bytes += numel * 4.0

    mib = 1024.0 * 1024.0
    packed_mb = packed_bytes / mib
    unpacked_mb = unpacked_bytes / mib
    savings = unpacked_mb - packed_mb

    note = ""
    if savings > 0.01:
        note = (
            f"ORT static quantization stores INT4 as INT8 containers. "
            f"With proper INT4 packing, the file would be "
            f"{packed_mb:.2f} MiB ({savings:.2f} MiB smaller). "
            f"TensorRT and OpenVINO support native INT4 packing."
        )

    return {
        "packed_size_mb": round(packed_mb, 4),
        "unpacked_size_mb": round(unpacked_mb, 4),
        "packing_savings_mb": round(savings, 4),
        "packing_note": note,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ORT latency benchmark (J2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def benchmark_onnx_latency(
    onnx_path: str,
    input_shape: Tuple[int, ...],
    *,
    batch_size: int = 1,
    warmup_runs: int = 10,
    measure_runs: int = 50,
    providers: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Benchmark inference latency of an ONNX model under ONNX Runtime.

    Returns the same dict shape as ``utils.metrics.benchmark_latency``
    so callers can swap the two without restructuring downstream code.

    The session is created with ``intra_op_num_threads`` left at the
    ORT default — the goal is "what does this model do on this
    machine", not a synthetic single-thread number.

    Args:
        onnx_path:    Path to .onnx file (FP32 or quantized — both work).
        input_shape:  ``(C, H, W)``. Batch dim added by this function.
        batch_size:   Inference batch size for the benchmark.
        warmup_runs:  Number of forward passes before timing starts
                      (lets ORT JIT/cache stabilise).
        measure_runs: Number of timed forward passes.
        providers:    ORT execution providers, in priority order. Default
                      ``["CPUExecutionProvider"]`` because that is the
                      one provider always present; pass
                      ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
                      to opt into GPU inference where available.

    Returns:
        ``{"latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
           "throughput_fps"}``.
    """
    _, ort, _ = _require_onnx()

    if providers is None:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    inp_meta = session.get_inputs()[0]
    inp_name = inp_meta.name

    dummy = np.random.randn(batch_size, *input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        session.run(None, {inp_name: dummy})

    timings_ms: List[float] = []
    for _ in range(measure_runs):
        t0 = time.perf_counter()
        session.run(None, {inp_name: dummy})
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    arr = np.array(timings_ms)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    throughput = (batch_size / mean_ms) * 1000.0 if mean_ms > 0 else 0.0

    return {
        "latency_mean_ms": round(mean_ms, 4),
        "latency_p50_ms": round(p50_ms, 4),
        "latency_p95_ms": round(p95_ms, 4),
        "throughput_fps": round(throughput, 1),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Combined helper used by main.py (J1 + J3 + J2 in one call)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def export_quantize_and_benchmark(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_dir: str,
    *,
    name: str,
    calibration_loader: Optional[DataLoader] = None,
    num_batches: int = 10,
    do_int8: bool = True,
    batch_size: int = 1,
    warmup_runs: int = 10,
    measure_runs: int = 50,
) -> Dict[str, Any]:
    """One-stop export → quantize → measure pipeline for a single model.

    Used by ``main.py`` to populate the J3 (real size) and J2 (ORT
    latency) fields on each ``QuantizationResult``. Returns a dict with
    every artefact path and measurement so the caller can stash them
    on the result without rebuilding the staging directory.

    The function is best-effort: any step can be disabled (``do_int8``)
    or fail without crashing the pipeline. On failure, the corresponding
    keys are simply omitted from the returned dict.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The FP32 stem keeps a ".fp32" precision suffix only when an INT8
    # export is being produced alongside it (so the two precisions are
    # distinguishable on disk). For an FP32-only export we drop the
    # suffix — ``foo.onnx`` is clearer than ``foo.fp32.onnx`` when
    # there's nothing else with the same stem.
    if do_int8:
        fp32_path = out_dir / f"{name}.fp32.onnx"
    else:
        fp32_path = out_dir / f"{name}.onnx"
    int8_path = out_dir / f"{name}.int8.onnx"

    info: Dict[str, Any] = {"name": name}

    # 1. FP32 export
    try:
        export_to_onnx(model, input_shape, str(fp32_path),
                       batch_size=batch_size)
        info["fp32_onnx_path"] = str(fp32_path)
        info["fp32_onnx_size_mb"] = onnx_disk_size_mb(str(fp32_path))
    except Exception as exc:
        logger.warning("ONNX FP32 export failed for %s: %s", name, exc)
        return info

    # 2. Static INT8 quantization (only if we have calibration + opt-in)
    if do_int8 and calibration_loader is not None:
        try:
            quantize_onnx_static(
                str(fp32_path), str(int8_path),
                calibration_loader, num_batches=num_batches,
            )
            info["int8_onnx_path"] = str(int8_path)
            info["int8_onnx_size_mb"] = onnx_disk_size_mb(str(int8_path))
        except Exception as exc:
            logger.warning(
                "ONNX static INT8 quantization failed for %s: %s",
                name, exc,
            )

    # 3. ORT latency on whichever artefact we ended up with
    target = info.get("int8_onnx_path") or info.get("fp32_onnx_path")
    if target:
        try:
            info["onnx_latency"] = benchmark_onnx_latency(
                target, input_shape,
                batch_size=batch_size,
                warmup_runs=warmup_runs,
                measure_runs=measure_runs,
            )
        except Exception as exc:
            logger.warning("ORT latency benchmark failed for %s: %s", name, exc)

    return info
