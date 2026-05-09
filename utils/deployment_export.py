"""
NeuroQuant v2.0 — TensorRT & OpenVINO optional export backends.

Best-effort export from a pre-existing FP32 ONNX file to:

  1. **TensorRT** engine (``.trt``) — INT8 calibration via the TRT API.
  2. **OpenVINO** IR (``.xml`` + ``.bin``) — INT8 via NNCF/POT.

Both are **optional**: if the dependency is not installed the function
returns ``None`` and logs a warning. The pipeline never fails because
of a missing deployment backend — ONNX Runtime remains the mandatory
baseline.

Usage::

    from utils.deployment_export import (
        export_tensorrt, export_openvino, available_backends,
    )
    print(available_backends())  # e.g. ['onnxruntime', 'tensorrt']
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("neuroquant")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dependency probes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _has_tensorrt() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _has_openvino() -> bool:
    try:
        from openvino.runtime import Core  # noqa: F401
        return True
    except ImportError:
        return False


def available_backends() -> List[str]:
    """Return the list of deployment backends available on this machine.

    Always includes ``'onnxruntime'`` (it is a hard dependency). The
    others are opt-in.
    """
    backends = ["onnxruntime"]
    if _has_tensorrt():
        backends.append("tensorrt")
    if _has_openvino():
        backends.append("openvino")
    return backends


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TensorRT export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def export_tensorrt(
    onnx_path: str,
    output_path: str,
    input_shape: Tuple[int, ...],
    *,
    batch_size: int = 1,
    precision: str = "int8",
    calibration_data: Optional[np.ndarray] = None,
    workspace_mb: int = 512,
) -> Optional[Dict[str, Any]]:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path:        Source ``.onnx`` file.
        output_path:      Destination ``.trt`` engine file.
        input_shape:      ``(C, H, W)`` — batch dim added automatically.
        batch_size:       Max batch size for the engine.
        precision:        ``'fp32'``, ``'fp16'``, or ``'int8'``.
        calibration_data: Numpy array of calibration inputs for INT8
                          (shape ``(N, C, H, W)``). Required when
                          ``precision='int8'``.
        workspace_mb:     TRT workspace size in MiB.

    Returns:
        Dict with ``engine_path``, ``engine_size_mb``, ``precision``,
        ``build_time_s``, or ``None`` if TensorRT is not installed.
    """
    if not _has_tensorrt():
        logger.warning(
            "TensorRT not installed — skipping engine build. "
            "Install with: pip install tensorrt"
        )
        return None

    import tensorrt as trt

    t0 = time.time()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TRT parse error: %s", parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20),
    )

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.warning("FP16 not supported on this GPU; using FP32.")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibration_data is not None:
                calibrator = _TRTCalibrator(calibration_data)
                config.int8_calibrator = calibrator
            else:
                logger.warning(
                    "INT8 precision requested but no calibration data "
                    "provided; engine may have poor accuracy."
                )
        else:
            logger.warning("INT8 not supported on this GPU; using FP32.")

    # Build the engine
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        logger.error("TensorRT engine build failed.")
        return None

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(serialized)

    elapsed = time.time() - t0
    size_mb = out.stat().st_size / (1024.0 * 1024.0)
    logger.info(
        "TensorRT engine built: %s (%.2f MiB, %s, %.1fs)",
        out.name, size_mb, precision, elapsed,
    )

    return {
        "engine_path": str(out),
        "engine_size_mb": round(size_mb, 4),
        "precision": precision,
        "build_time_s": round(elapsed, 2),
        "backend": "tensorrt",
    }


class _TRTCalibrator:
    """Minimal INT8 calibrator for TensorRT using pre-loaded numpy data."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data.astype(np.float32)
        self.batch_idx = 0
        self.batch_size = 1

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.batch_idx >= len(self.data):
            return None
        batch = self.data[self.batch_idx : self.batch_idx + self.batch_size]
        self.batch_idx += self.batch_size
        import pycuda.driver as cuda  # noqa: WPS433
        d_input = cuda.mem_alloc(batch.nbytes)
        cuda.memcpy_htod(d_input, batch)
        return [int(d_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenVINO export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def export_openvino(
    onnx_path: str,
    output_dir: str,
    *,
    model_name: str = "model",
    compress_to_fp16: bool = True,
) -> Optional[Dict[str, Any]]:
    """Convert an ONNX model to OpenVINO IR format.

    Args:
        onnx_path:       Source ``.onnx`` file.
        output_dir:      Directory for the ``.xml`` + ``.bin`` files.
        model_name:      Base name for the output files.
        compress_to_fp16: Compress constant weights to FP16 (halves
                          model size with negligible accuracy loss).

    Returns:
        Dict with ``xml_path``, ``bin_path``, ``ir_size_mb``,
        ``conversion_time_s``, or ``None`` if OpenVINO is not installed.
    """
    if not _has_openvino():
        logger.warning(
            "OpenVINO not installed — skipping IR conversion. "
            "Install with: pip install openvino"
        )
        return None

    import openvino as ov

    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Read the ONNX model and convert
    ov_model = ov.convert_model(onnx_path)

    xml_path = out / f"{model_name}.xml"
    ov.save_model(
        ov_model, str(xml_path),
        compress_to_fp16=compress_to_fp16,
    )

    bin_path = xml_path.with_suffix(".bin")
    elapsed = time.time() - t0

    # Total IR size = .xml + .bin
    ir_size = 0.0
    if xml_path.exists():
        ir_size += xml_path.stat().st_size
    if bin_path.exists():
        ir_size += bin_path.stat().st_size
    ir_size_mb = ir_size / (1024.0 * 1024.0)

    logger.info(
        "OpenVINO IR: %s (%.2f MiB, FP16=%s, %.1fs)",
        xml_path.name, ir_size_mb, compress_to_fp16, elapsed,
    )

    return {
        "xml_path": str(xml_path),
        "bin_path": str(bin_path),
        "ir_size_mb": round(ir_size_mb, 4),
        "compress_to_fp16": compress_to_fp16,
        "conversion_time_s": round(elapsed, 2),
        "backend": "openvino",
    }


def benchmark_openvino_latency(
    xml_path: str,
    input_shape: Tuple[int, ...],
    *,
    batch_size: int = 1,
    warmup_runs: int = 10,
    measure_runs: int = 50,
) -> Optional[Dict[str, float]]:
    """Benchmark inference latency using the OpenVINO runtime.

    Returns the same dict shape as ``benchmark_onnx_latency`` so callers
    can swap backends without restructuring downstream code.
    """
    if not _has_openvino():
        return None

    import openvino as ov
    import time as _time

    core = ov.Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, "CPU")
    infer_request = compiled.create_infer_request()

    dummy = np.random.randn(batch_size, *input_shape).astype(np.float32)
    input_tensor = ov.Tensor(dummy)

    # Warmup
    for _ in range(warmup_runs):
        infer_request.infer({0: input_tensor})

    timings_ms: List[float] = []
    for _ in range(measure_runs):
        t0 = _time.perf_counter()
        infer_request.infer({0: input_tensor})
        t1 = _time.perf_counter()
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
        "backend": "openvino",
    }
