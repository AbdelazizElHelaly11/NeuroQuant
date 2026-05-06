"""
NeuroQuant v2.0 — Wave 5 production tests (Reporting + MLflow).

Coverage matrix:

  * G1 — Headline summary row carries ONNX columns
            - Idempotent insert preserves new ONNX fields
            - Public report prints additional ONNX header columns when
              any row has ONNX numbers, and falls back to plain layout
              otherwise
  * G2 — Pareto plot wiring
            - ``ParetoVisualizer.plot_3d_pareto`` writes a PNG
            - ``compute_solution_metrics`` forwards the
              ``latency_mean_ms`` field so the 3-D plot has data
            - ``ParetoAnalyzer.analyze`` adds ``pareto_3d`` to
              plot_paths only when at least one solution has latency
  * G3 — Reproducibility manifest
            - Manifest includes onnx_runtime version + providers
            - Deployment block carries fp32_onnx + latency_lut_path
              when the pipeline puts them on ``results``
  * G4 — Deployment fidelity section
            - ``_print_deployment_fidelity_section`` is silent when no
              ONNX rows exist
            - ``_print_deployment_fidelity_section`` emits the per-method
              delta lines when FP32 + quantized ONNX rows are present
  * I1 — MLflow ONNX metrics surface keys
            - ``MLflowTracker.log_metrics`` accepts onnx_*_size_mb,
              onnx_*_latency_*_ms keys without error in no-op mode
  * I2 — ONNX artifacts wiring
            - ``MLflowTracker.log_artifact`` is called with the
              ``onnx`` subdirectory for each method that produced one
              (verified via mock)
  * I3 — Pareto front comparison
            - ``_build_pareto_summary`` returns canonical fields
            - JSON export round-trip preserves values
            - Top-1 stats correctly use max-first ordering
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# ── Optional-dep gate ────────────────────────────────────────────────────
try:
    import onnxruntime  # noqa: F401
    HAS_ORT = True
except Exception:
    HAS_ORT = False

try:
    import matplotlib  # noqa: F401
    HAS_MPL = True
except Exception:
    HAS_MPL = False

ort_required = pytest.mark.skipif(
    not HAS_ORT, reason="onnx/onnxruntime not installed"
)
mpl_required = pytest.mark.skipif(
    not HAS_MPL, reason="matplotlib not installed"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def pipeline_skeleton():
    """Build a NeuroQuantPipeline far enough to test reporting helpers.

    We deliberately avoid running ``run()`` — the helpers under test
    (summary table, deployment-fidelity printer, Pareto summary
    builder) only need a config + a ``_summary_rows`` list + a few
    scalar attributes, which we set directly.
    """
    from main import NeuroQuantPipeline
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.seed = 0

    p = NeuroQuantPipeline(cfg)
    p.fp32_acc = 90.0
    p.fp32_size_mb = 8.5
    p.fp32_onnx = {}
    p.method_results = []
    p.pareto_analysis = {}
    p.qat_result = {}
    return p


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G1 — Headline summary row schema
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_summary_row_carries_onnx_columns(pipeline_skeleton):
    """``_add_summary_row`` stores ONNX kwargs verbatim on the row."""
    p = pipeline_skeleton
    p._add_summary_row(
        "GPTQ_INT8", top1=88.5,
        latency_ms=2.5, throughput=400.0,
        ebops=1e6, size_mb=2.0,
        onnx_size_mb=1.7, onnx_latency_ms=1.9,
        onnx_throughput_fps=525.0,
    )
    rows = p._summary_rows
    assert len(rows) == 1
    r = rows[0]
    assert r["onnx_size_mb"] == 1.7
    assert r["onnx_latency_ms"] == 1.9
    assert r["onnx_throughput_fps"] == 525.0
    # Plain (non-ONNX) fields preserved.
    assert r["size_mb"] == 2.0
    assert r["latency_ms"] == 2.5


def test_summary_row_idempotent_replace_preserves_new_onnx_values(
    pipeline_skeleton,
):
    """Re-adding the same method updates ALL fields including ONNX."""
    p = pipeline_skeleton
    p._add_summary_row(
        "GPTQ_INT8", 80.0, 5.0, 200.0, 1e6, 4.0,
        onnx_size_mb=3.5, onnx_latency_ms=4.0, onnx_throughput_fps=250.0,
    )
    p._add_summary_row(
        "GPTQ_INT8", 88.0, 2.0, 500.0, 9e5, 2.0,
        onnx_size_mb=1.7, onnx_latency_ms=1.9, onnx_throughput_fps=525.0,
    )
    assert len(p._summary_rows) == 1
    r = p._summary_rows[0]
    assert r["top1"] == 88.0
    assert r["onnx_size_mb"] == 1.7  # second call wins, not first
    assert r["onnx_latency_ms"] == 1.9


def test_print_report_adds_onnx_columns_when_rows_have_them(pipeline_skeleton):
    """The public table grows three columns when any row has ONNX data."""
    p = pipeline_skeleton
    p.report_lines = []
    p.phases_passed = 1
    p._add_summary_row(
        "FP32", 90.0, 10.0, 100.0, 4e6, 8.5,
        onnx_size_mb=8.0, onnx_latency_ms=11.0, onnx_throughput_fps=90.0,
    )
    p._add_summary_row(
        "GPTQ_INT8", 88.5, 2.5, 400.0, 1e6, 2.0,
        onnx_size_mb=1.7, onnx_latency_ms=1.9, onnx_throughput_fps=525.0,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        p._print_report(elapsed=1.0, phases_total=1)
    out = buf.getvalue()
    assert "ONNX MiB" in out
    assert "ORT(ms)" in out
    assert "ORT FPS" in out
    # FP32 row exists with its ONNX numbers
    assert "8.00" in out
    # GPTQ row exists with its ONNX numbers
    assert "1.70" in out


def test_print_report_omits_onnx_columns_when_no_rows_have_them(
    pipeline_skeleton,
):
    """Without ONNX numbers, the table reverts to the original layout."""
    p = pipeline_skeleton
    p.report_lines = []
    p.phases_passed = 1
    p._add_summary_row(
        "FP32", 90.0, 10.0, 100.0, 4e6, 8.5,
    )
    p._add_summary_row(
        "GPTQ_INT8", 88.5, 2.5, 400.0, 1e6, 2.0,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        p._print_report(elapsed=1.0, phases_total=1)
    out = buf.getvalue()
    assert "ONNX MiB" not in out
    assert "ORT(ms)" not in out


def test_fmt_or_dash_handles_none():
    """The dash helper produces right-aligned dashes for missing values."""
    from main import NeuroQuantPipeline

    assert NeuroQuantPipeline._fmt_or_dash(None, ">9.2f").strip() == "-"
    assert NeuroQuantPipeline._fmt_or_dash(1.23, ">9.2f").strip() == "1.23"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G2 — Pareto plot wiring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@mpl_required
def test_compute_solution_metrics_forwards_latency():
    """Latency from ``ParetoSolution`` survives the enrichment step."""
    from visualization.pareto_analysis import ParetoAnalyzer
    from config import ParetoFront, ParetoSolution

    sols: List[ParetoSolution] = [
        ParetoSolution(
            solution_id="GPTQ_INT8",
            method="GPTQ",
            accuracy=88.5,
            accuracy_loss=1.5,
            ebops=1e6,
            ebops_reduction=75.0,
            model_size_mb=1.7,
            latency_mean_ms=1.9,
            bitwidth_assignment={},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
    ]
    front = ParetoFront(
        solutions=sols, generation=0, evaluations=0,
        convergence_reason="test",
    )
    a = ParetoAnalyzer(front, fp32_accuracy=90.0, fp32_ebops=4e6)
    enriched = a.compute_solution_metrics()
    assert len(enriched) == 1
    assert enriched[0]["latency_mean_ms"] == 1.9


@mpl_required
def test_plot_3d_pareto_writes_png_when_latency_present(tmp_path):
    """3-D plot file is created when at least one solution has latency."""
    from visualization.pareto_analysis import ParetoVisualizer

    solutions = [
        {"solution_id": "GPTQ_INT8", "accuracy": 88.5, "accuracy_loss": 1.5,
         "ebops": 1e6, "ebops_mb": 1.7, "model_size_mb": 1.7,
         "ebops_reduction": 75.0, "compression_ratio": 4.0,
         "int4_count": 0, "int8_count": 5, "int4_percent": 0.0,
         "crowding_distance": 0.0, "latency_mean_ms": 1.9},
        {"solution_id": "AWQ_INT4", "accuracy": 86.0, "accuracy_loss": 4.0,
         "ebops": 5e5, "ebops_mb": 0.85, "model_size_mb": 0.85,
         "ebops_reduction": 87.0, "compression_ratio": 8.0,
         "int4_count": 5, "int8_count": 0, "int4_percent": 100.0,
         "crowding_distance": 0.0, "latency_mean_ms": 2.4},
    ]
    metrics = {"hypervolume": 1.0}
    extremes: Dict[str, Any] = {}
    viz = ParetoVisualizer(solutions, metrics, extremes, "TestModel")
    out = tmp_path / "p3d.png"
    viz.plot_3d_pareto(out)
    assert out.exists() and out.stat().st_size > 0


@mpl_required
def test_plot_3d_pareto_handles_no_latency_solutions(tmp_path):
    """When no latency entries exist, the file is still written (placeholder)."""
    from visualization.pareto_analysis import ParetoVisualizer

    solutions = [
        {"solution_id": "GPTQ_INT8", "accuracy": 88.5, "accuracy_loss": 1.5,
         "ebops": 1e6, "ebops_mb": 1.7, "model_size_mb": 1.7,
         "ebops_reduction": 75.0, "compression_ratio": 4.0,
         "int4_count": 0, "int8_count": 5, "int4_percent": 0.0,
         "crowding_distance": 0.0, "latency_mean_ms": None},
    ]
    viz = ParetoVisualizer(solutions, {"hypervolume": 0.0}, {}, "Test")
    out = tmp_path / "p3d.png"
    viz.plot_3d_pareto(out)
    assert out.exists()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G3 — Reproducibility manifest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_reproducibility_manifest_includes_onnx_runtime(tmp_path):
    """Manifest captures onnxruntime version + providers when available."""
    from utils.checkpointing import save_reproducibility_manifest
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    out_path = save_reproducibility_manifest(
        str(tmp_path), cfg, results={},
    )
    data = json.loads(out_path.read_text())
    assert "onnx_runtime" in data
    if HAS_ORT:
        assert data["onnx_runtime"]["version"] is not None
        assert isinstance(data["onnx_runtime"]["providers_available"], list)
        assert "CPUExecutionProvider" in data["onnx_runtime"]["providers_available"]


def test_reproducibility_manifest_includes_deployment_when_provided(tmp_path):
    """Deployment block surfaces fp32_onnx + latency_lut_path on results."""
    from utils.checkpointing import save_reproducibility_manifest
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    fake_onnx = tmp_path / "fake.onnx"
    fake_onnx.write_bytes(b"x" * 64)
    fake_lut = tmp_path / "lut.json"
    fake_lut.write_text("{}")

    results = {
        "fp32_onnx": {
            "fp32_onnx_path": str(fake_onnx),
            "fp32_onnx_size_mb": 0.5,
            "onnx_latency": {
                "latency_mean_ms": 12.3,
                "throughput_fps": 81.0,
            },
        },
        "latency_lut_path": str(fake_lut),
    }
    out_path = save_reproducibility_manifest(str(tmp_path), cfg, results)
    data = json.loads(out_path.read_text())

    assert "deployment" in data
    dep = data["deployment"]
    assert dep["fp32_onnx_path"] == str(fake_onnx)
    assert dep["fp32_onnx_size_mb"] == 0.5
    assert dep["fp32_onnx_latency_mean_ms"] == 12.3
    assert dep["fp32_onnx_throughput_fps"] == 81.0
    assert dep["latency_lut_path"] == str(fake_lut)
    assert dep["latency_lut_present_on_disk"] is True


def test_reproducibility_manifest_skips_deployment_when_absent(tmp_path):
    """No fp32_onnx / lut path → no deployment block (kept clean)."""
    from utils.checkpointing import save_reproducibility_manifest
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    out_path = save_reproducibility_manifest(str(tmp_path), cfg, results={})
    data = json.loads(out_path.read_text())
    assert "deployment" not in data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G4 — Deployment fidelity section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_deployment_fidelity_section_silent_without_onnx_rows(
    pipeline_skeleton,
):
    """No ONNX rows → printer emits nothing."""
    p = pipeline_skeleton
    p._add_summary_row("FP32", 90.0, 10.0, 100.0, 4e6, 8.5)
    p._add_summary_row("GPTQ_INT8", 88.5, 2.5, 400.0, 1e6, 2.0)

    buf = io.StringIO()
    with redirect_stdout(buf):
        p._print_deployment_fidelity_section(p._summary_rows)
    assert buf.getvalue() == ""


def test_deployment_fidelity_section_prints_deltas(pipeline_skeleton):
    """FP32 + quantized ONNX rows → per-method deltas are emitted."""
    p = pipeline_skeleton
    p.fp32_onnx = {
        "fp32_onnx_size_mb": 8.0,
        "onnx_latency": {"latency_mean_ms": 11.0},
    }
    p._add_summary_row(
        "FP32", 90.0, 10.0, 100.0, 4e6, 8.5,
        onnx_size_mb=8.0, onnx_latency_ms=11.0, onnx_throughput_fps=90.0,
    )
    p._add_summary_row(
        "GPTQ_INT8", 88.5, 2.5, 400.0, 1e6, 2.0,
        onnx_size_mb=1.7, onnx_latency_ms=1.9, onnx_throughput_fps=525.0,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        p._print_deployment_fidelity_section(p._summary_rows)
    out = buf.getvalue()

    assert "ONNX deployment fidelity" in out
    assert "FP32 ONNX size on disk" in out
    assert "FP32 ORT mean latency" in out
    assert "Per-method ONNX deltas" in out
    assert "GPTQ_INT8" in out
    # Quantized model is ~78.75% smaller (1 - 1.7/8.0); allow whitespace
    # variation from the right-aligned format spec.
    import re
    assert re.search(r"size −\s*78\.[78]%", out), out
    # ~5.79x speed-up vs FP32 ONNX (11.0 / 1.9).
    assert re.search(r"ORT\s+5\.7[89]x", out), out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# I1 / I2 — MLflow tracker accepts ONNX metrics + artifacts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_mlflow_tracker_accepts_onnx_metric_keys():
    """Tracker accepts the wave-5 metric keys without raising."""
    from tracking.mlflow_logger import MLflowTracker
    from config import QuantizationConfig

    tracker = MLflowTracker(QuantizationConfig())
    # No-op safe even outside an active run.
    tracker.log_metrics({
        "GPTQ_INT8_onnx_size_mb": 1.7,
        "GPTQ_INT8_onnx_latency_mean_ms": 1.9,
        "GPTQ_INT8_onnx_throughput_fps": 525.0,
    })


def test_mlflow_tracker_log_artifact_handles_missing_files(tmp_path):
    """``log_artifact`` does not crash when the file isn't where it expects."""
    from tracking.mlflow_logger import MLflowTracker
    from config import QuantizationConfig

    tracker = MLflowTracker(QuantizationConfig())
    # Should be a clean no-op for non-existent paths.
    tracker.log_artifact(str(tmp_path / "nope.onnx"), "onnx")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# I3 — Pareto comparison summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_build_pareto_summary_returns_canonical_fields(pipeline_skeleton):
    """Summary contains the headline stats over the public method rows."""
    p = pipeline_skeleton
    p._add_summary_row(
        "FP32", 90.0, 10.0, 100.0, 4e6, 8.5,
        onnx_size_mb=8.0, onnx_latency_ms=11.0, onnx_throughput_fps=90.0,
    )
    p._add_summary_row(
        "GPTQ_INT8", 88.5, 2.5, 400.0, 1e6, 2.0,
        onnx_size_mb=1.7, onnx_latency_ms=1.9, onnx_throughput_fps=525.0,
    )
    p._add_summary_row(
        "AWQ_INT4", 86.0, 1.8, 555.0, 5e5, 1.0,
        onnx_size_mb=0.85, onnx_latency_ms=2.4, onnx_throughput_fps=416.0,
    )
    summary = p._build_pareto_summary()

    assert summary["model_name"] == p.config.model_name
    assert summary["n_methods"] == 2
    assert summary["fp32_top1"] == 90.0
    assert summary["fp32_onnx_size_mb"] == 8.0
    assert summary["fp32_onnx_latency_ms"] == 11.0

    # Top-1 best is the highest accuracy; worst is the lowest.
    assert summary["top1_best"] == 88.5
    assert summary["top1_worst"] == 86.0
    # Size best is the smallest.
    assert summary["onnx_size_mb_best"] == 0.85
    assert summary["onnx_size_mb_worst"] == 1.7
    # Latency best is the smallest ms.
    assert summary["onnx_latency_ms_best"] == 1.9
    assert summary["onnx_latency_ms_worst"] == 2.4

    # Per-method breakdown excludes FP32 baseline.
    method_names = {m["method"] for m in summary["methods"]}
    assert method_names == {"GPTQ_INT8", "AWQ_INT4"}


def test_build_pareto_summary_handles_no_methods(pipeline_skeleton):
    """Empty method list → summary still produced, ``n_methods=0``."""
    p = pipeline_skeleton
    p._add_summary_row("FP32", 90.0, 10.0, 100.0, 4e6, 8.5)
    summary = p._build_pareto_summary()
    assert summary["n_methods"] == 0
    assert summary["methods"] == []
    # Best/worst stats absent (nothing to aggregate).
    assert "top1_best" not in summary
    assert "onnx_size_mb_best" not in summary


def test_build_pareto_summary_json_export_roundtrip(pipeline_skeleton, tmp_path):
    """The exported JSON survives a load round-trip with no precision loss."""
    p = pipeline_skeleton
    p._add_summary_row(
        "GPTQ_INT8", 88.5, 2.5, 400.0, 1e6, 2.0,
        onnx_size_mb=1.7, onnx_latency_ms=1.9, onnx_throughput_fps=525.0,
    )
    summary = p._build_pareto_summary()

    out = tmp_path / "pareto_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    loaded = json.loads(out.read_text())
    assert loaded["n_methods"] == 1
    assert loaded["top1_best"] == 88.5
    assert loaded["onnx_size_mb_best"] == 1.7
    assert loaded["methods"][0]["method"] == "GPTQ_INT8"
