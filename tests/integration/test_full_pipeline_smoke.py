"""
NeuroQuant v2.0 — Full-pipeline integration smoke (Wave 6 K3).

Runs every phase of ``NeuroQuantPipeline.run()`` end-to-end on a tiny
synthetic dataset + the framework's smallest supported torchvision
model (``mobilenet_v2`` adapted for 32×32). Designed to:

  * Catch *handoff* regressions between phases — e.g. phase 1c
    producing a checkpoint phase 1d can't load.
  * Catch import / wiring regressions that unit tests miss because they
    instantiate one component in isolation.
  * Verify the new wave-4/5 surfaces (ONNX export, deployment-fidelity
    section, MLflow no-op when not installed) survive the full run.

The smoke is opt-in via the ``integration`` pytest marker so the
default ``pytest`` invocation stays under a minute. To run:

    pytest -m integration

Hard rules to keep this fast (<2 minutes on CPU):

  * 4 classes, 16×16 inputs, 64-sample synthetic dataset
  * No training (``training_epochs=0``)
  * NSGA pop=4, gens=2; QAT epochs=1; AdaRound epochs=2
  * Phase 3 (XAI) skipped — pure correctness phase, slow on CPU
  * MLflow uses the no-op fallback (we don't install mlflow in CI)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.integration


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_smoke_config(output_dir: Path):
    """Build a smallest-possible pipeline config.

    Synthetic dataset, 4 classes, 16-channel TinyCNN built via
    torchvision's ``mobilenet_v2`` adapted by ``ModelLoader`` for the
    small input shape. Every NSGA / QAT / AdaRound budget is tuned to
    finish within seconds while still exercising the real code paths.
    """
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    cfg.model_name = "mobilenetv2"
    cfg.num_classes = 4
    cfg.input_shape = (3, 32, 32)
    cfg.dataset_name = "synthetic"
    cfg.batch_size = 8
    cfg.num_workers = 0
    cfg.output_dir = str(output_dir)

    hp = cfg.hyperparams
    hp.device = "cpu"
    hp.seed = 0

    # Calibration / Hessian — tiny.
    hp.calibration_batches = 2
    hp.hessian_batches = 2

    # NSGA — minimum that still runs the search loop.
    hp.nsga_population_size = 4
    hp.nsga_generations = 2

    # AdaRound + QAT — single epoch (smoke).
    hp.adaround_epochs = 2
    hp.qat_epochs = 1
    hp.qat_distill_alpha = 0.0  # no teacher for smoke speed

    # PTQ rerank — single candidate.
    hp.ptq_real_rerank_topk = 1

    # Latency benchmark — minimum runs.
    hp.latency_warmup_runs = 2
    hp.latency_measure_runs = 5

    # Wave 4/5: ONNX export ON; hardware-aware OFF (LUT build is the
    # slow part — covered by its own unit tests).
    hp.onnx_export_enabled = True
    hp.hardware_aware_search = False

    # Skip XAI in the smoke — unit tests cover it; the integration
    # test only needs to prove the phases hand off correctly.
    cfg.run_phases = [
        "phase_0_preparation",
        "phase_1a_hessian_clustering",
        "phase_1b_fitcompress",
        "phase_1c_nsga_search",
        "phase_1d_adaround",
        "phase_1e_qat",
        "phase_1f_gptq_smooth_awq",
        "phase_2_pareto",
        "phase_4_mlflow",
    ]
    return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Smoke
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_full_pipeline_runs_end_to_end(tmp_path):
    """Every wired phase runs to completion; expected artefacts exist."""
    from main import NeuroQuantPipeline

    cfg = _build_smoke_config(tmp_path)
    pipe = NeuroQuantPipeline(cfg, training_epochs=0)
    results = pipe.run()

    # 1. Pipeline returned a results dict with the expected keys.
    assert isinstance(results, dict)
    assert "phases_passed" in results
    assert "phases_total" in results
    # All requested phases must have completed (no early-exit on error).
    assert results["phases_passed"] == results["phases_total"], (
        f"Pipeline incomplete: {results['phases_passed']}/{results['phases_total']}"
    )

    # 2. FP32 baseline accuracy was computed (synthetic data → ~25%
    #    random-chance on 4 classes; just check it's a real number).
    assert "fp32_acc" in results
    assert 0.0 <= float(results["fp32_acc"]) <= 100.0

    # 3. NSGA produced at least one Pareto solution.
    assert results.get("pareto_solutions", 0) >= 1

    # 4. Reproducibility manifest written + valid JSON.
    manifest = tmp_path / "reproducibility_manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["config"]["model_name"] == "mobilenetv2"
    assert data["config"]["num_classes"] == 4
    # ONNX runtime block present (pydantic + onnxruntime are wave 6 deps).
    assert "onnx_runtime" in data

    # 5. Pareto summary written + has the canonical Wave 5 fields.
    summary_path = tmp_path / "pareto_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["model_name"] == "mobilenetv2"
    assert "n_methods" in summary

    # 6. Pipeline report file written.
    report = tmp_path / "pipeline_report.txt"
    assert report.exists()
    assert "NeuroQuant" in report.read_text(encoding="utf-8")


def test_pipeline_produces_at_least_one_onnx_artefact(tmp_path):
    """Wave 4 J1: at least one .onnx file lands under output_dir/onnx/."""
    from main import NeuroQuantPipeline

    cfg = _build_smoke_config(tmp_path)
    pipe = NeuroQuantPipeline(cfg, training_epochs=0)
    pipe.run()

    onnx_dir = tmp_path / "onnx"
    assert onnx_dir.exists(), f"missing {onnx_dir}"
    onnx_files = list(onnx_dir.glob("*.onnx"))
    assert onnx_files, "no .onnx files produced"
    # Every ONNX file is non-trivially sized (≥ 1 KB).
    for p in onnx_files:
        assert p.stat().st_size > 1024, f"{p.name} is suspiciously small"
