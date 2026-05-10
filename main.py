"""
NeuroQuant v2.0 - Main Pipeline Orchestrator

Orchestrates all phases of the quantization pipeline.
This is the single entry point for running the framework.

Usage:
    python main.py                       # default config + MobileNetV2 + CIFAR-10
    python main.py --config config.yaml  # custom config file
    python main.py --phases phase_0_preparation phase_1a_hessian_clustering
    python main.py --epochs 20           # full training run
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import shutil
import sys
import time
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config import (
    ParetoFront,
    ParetoSolution,
    QuantizationConfig,
    QuantizationMethod,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logging Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("neuroquant.main")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model & Data Builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_model(config: QuantizationConfig) -> nn.Module:
    """
    Build or load the model specified in config.

    Delegates entirely to ModelLoader which uses introspection
    to adapt any architecture — no hardcoded layer names.
    """
    from models.model_loader import ModelLoader
    return ModelLoader(config).load()



def build_data_loaders(config: QuantizationConfig):
    """
    Build train / search / val / test / calibration DataLoaders.

    The ``search`` slice is the NSGA-II fitness loader (held-out 10% of
    train, eval-time transforms). It is *separate* from ``val`` (QAT
    early-stop) and ``test`` (the public headline). This separation is
    a correctness fix: the single-loader design that preceded it
    over-fit NSGA fitness to the validation set used for reporting.

    Returns:
        (train_loader, search_loader, val_loader, test_loader,
         calib_loader, class_names)
    """
    from data.data_loader import GenericDatasetLoader

    loader = GenericDatasetLoader(config)
    return (
        loader.get_train_loader(),
        loader.get_search_loader(),
        loader.get_val_loader(),
        loader.get_test_loader(),
        loader.get_calibration_loader(
            num_batches=config.hyperparams.calibration_batches
        ),
        loader.get_class_names(),
    )


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """
    Compute top-1 and top-5 accuracy (%).

    Returns:
        {"top1": float, "top5": float}
    """
    from utils.metrics import compute_topk_accuracy
    return compute_topk_accuracy(model, loader, device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NeuroQuantPipeline — Main Orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class NeuroQuantPipeline:
    """
    Main pipeline orchestrator for NeuroQuant v2.0.

    Coordinates all phases in sequence:
    Phase 0  → Model & data preparation + optional training
    Phase 1a → Hessian computation & layer clustering
    Phase 1b → FITCompress warm-start seed generation
    Phase 1c → NSGA-II cluster-level search
    Phase 1d → Adaround post-calibration optimization
    Phase 1e → QAT fine-tuning from best PTQ
    Phase 1f → GPTQ / SmoothQuant / AWQ (INT8 + INT4 each)
    Phase 2  → Pareto analysis & visualization
    Phase 3  → XAI pipeline (Grad-CAM + SHAP)
    Phase 4  → MLflow tracking (runs throughout)
    """

    # Phase registry: (method_name, display_name)
    PHASES = [
        ("phase_0_preparation",         "Phase 0: Model & Data Preparation"),
        ("phase_1a_hessian_clustering",  "Phase 1a: Hessian + Layer Clustering"),
        ("phase_1b_fitcompress",         "Phase 1b: FITCompress Elite Seed"),
        ("phase_1c_nsga_search",         "Phase 1c: NSGA-II Multi-Objective Search"),
        ("phase_1d_adaround",            "Phase 1d: Adaround Weight Rounding"),
        ("phase_1e_qat",                 "Phase 1e: QAT Warmstart Fine-Tuning"),
        ("phase_1f_gptq_smooth_awq",     "Phase 1f: GPTQ + AWQ + SmoothQuant"),
        ("phase_2_pareto",               "Phase 2: Pareto Analysis & Visualization"),
        ("phase_3_xai",                  "Phase 3: XAI Explainability"),
        ("phase_4_mlflow",               "Phase 4: MLflow Finalisation"),
    ]

    def __init__(
        self,
        config: QuantizationConfig,
        training_epochs: int = 0,
        resume: bool = False,
    ) -> None:
        """
        Initialize the pipeline with configuration.

        Args:
            config: Full framework configuration.
            training_epochs: If > 0, train the model for this many epochs
                           in Phase 0 (set to 0 to skip training).
            resume: If True, skip phases that already have checkpoints.
        """
        self.config = config
        self.training_epochs = training_epochs
        self.resume = resume
        self.device = self._resolve_device(config.hyperparams.device)

        # Reproducibility — strict determinism (cuDNN deterministic,
        # CUBLAS_WORKSPACE_CONFIG, torch.use_deterministic_algorithms).
        # See utils.common.set_seed for the full list of flags. Strict
        # mode adds a small per-op overhead but produces byte-stable
        # outputs across reruns on the same machine, which is the
        # contract a deployable system must offer.
        from utils.common import set_seed
        set_seed(int(config.hyperparams.seed), strict=True)

        # State populated during the pipeline
        self.model: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        # NSGA-II fitness loader. Disjoint from val/test; populated by
        # ``build_data_loaders`` and consumed by phase 1c.
        self.search_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.calib_loader: Optional[DataLoader] = None
        # Dataset class names (optional, used by XAI captions).
        self.class_names: Optional[List[str]] = None

        self.fp32_acc: float = 0.0
        self.fp32_ebops: float = 0.0

        self.hessian_diag: Dict = {}
        self.cluster_result: Dict = {}
        self.cluster_assignments: List = []
        self.fit_seed: Dict = {}
        self.pareto_front: Dict = {}
        self.best_config: Dict[str, int] = {}
        # Separate config used by AdaRound (Phase 1d). May differ from
        # ``best_config`` when the QAT warmstart points at an all-INT8
        # solution — see phase_1c_nsga_search for the rationale.
        self.adaround_config: Dict[str, int] = {}
        self.adaround_result: Dict = {}
        self.qat_result: Dict = {}
        self.method_results: List[Dict] = []
        self.pareto_analysis: Dict = {}

        # Metric state
        self.fp32_top5: float = 0.0
        self.fp32_latency: Dict = {}
        self.fp32_size_mb: float = 0.0
        self.hardware_metrics: Dict = {}
        self._summary_rows: List[Dict] = []

        # Results accumulator
        self.results: Dict[str, Any] = {}
        self.report_lines: List[str] = []
        self.phases_passed: int = 0

        # MLflow tracker (initialized in run())
        self.tracker = None

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint manager
        from utils.checkpointing import CheckpointManager
        self.ckpt = CheckpointManager(str(self.output_dir), resume=self.resume)

    # ==================================================================
    # Main Entry Point
    # ==================================================================

    def run(self) -> Dict[str, Any]:
        """
        Execute the full pipeline end-to-end.

        Returns:
            Dictionary with results from each phase.
        """
        t_start = time.time()

        # Initialize MLflow tracker
        from tracking.mlflow_logger import MLflowTracker
        self.tracker = MLflowTracker(self.config)

        # Determine which phases to run
        active_phases = set(self.config.run_phases)
        phases_total = sum(1 for name, _ in self.PHASES if name in active_phases)

        logger.info("=" * 70)
        logger.info("  NeuroQuant v2.0 Pipeline")
        logger.info("=" * 70)
        logger.info("  Model:   %s", self.config.model_name)
        logger.info("  Dataset: %s", self.config.dataset_name)
        logger.info("  Device:  %s", self.device)
        logger.info("  Phases:  %d/%d active", phases_total, len(self.PHASES))
        logger.info("=" * 70)

        # Execute phases in order
        for phase_name, display_name in self.PHASES:
            if phase_name not in active_phases:
                logger.info("⏭  %s [SKIPPED]", display_name)
                continue

            logger.info("")
            logger.info("=" * 70)
            logger.info(display_name)
            logger.info("=" * 70)

            try:
                if self.ckpt.should_skip(phase_name):
                    # Resume path: restore all in-memory state this phase
                    # normally produces so downstream phases can proceed.
                    self._resume_phase(phase_name, display_name)
                else:
                    method = getattr(self, phase_name)
                    method()
                self.phases_passed += 1
            except Exception as e:
                logger.error(
                    "Phase '%s' FAILED: %s", phase_name, e, exc_info=True
                )
                self.report_lines.append(f"[ERROR] {display_name}: {e}")
                # End any active MLflow run
                if self.tracker:
                    self.tracker.end_run(status="FAILED")
                break  # Stop pipeline on failure

        # Final report
        elapsed = time.time() - t_start
        self._print_report(elapsed, phases_total)

        self.results["elapsed_seconds"] = elapsed
        self.results["phases_passed"] = self.phases_passed
        self.results["phases_total"] = phases_total

        # Save reproducibility manifest
        from utils.checkpointing import save_reproducibility_manifest
        save_reproducibility_manifest(
            str(self.output_dir), self.config, self.results,
        )

        return self.results

    # ==================================================================
    # Phase 0: Model & Data Preparation
    # ==================================================================

    def phase_0_preparation(self) -> None:
        """Phase 0: Load model, dataset, optionally train, and validate setup."""
        self.tracker.start_run("phase_0_preparation", {"phase": "0"})

        # Build model
        self.model = build_model(self.config)
        logger.info("  Model: %s (%s parameters)",
                     self.config.model_name,
                     f"{sum(p.numel() for p in self.model.parameters()):,}")

        # Build data loaders + capture optional class names for XAI captions.
        (self.train_loader, self.search_loader, self.val_loader,
         self.test_loader, self.calib_loader,
         self.class_names) = build_data_loaders(self.config)

        # Optional: Train baseline model
        if self.training_epochs > 0:
            logger.info("  Training FP32 baseline for %d epochs ...",
                         self.training_epochs)
            self.model.to(self.device)
            self.fp32_acc = self._train_model(
                self.model, self.train_loader, self.val_loader,
                epochs=self.training_epochs,
            )
        else:
            # Evaluate existing weights
            self.model.to(self.device)
            acc_dict = evaluate_model(
                self.model, self.val_loader, self.device
            )
            self.fp32_acc = acc_dict["top1"]
            logger.info("  Skipping training (use --epochs N to train)")

        # FP32 top-5
        fp32_acc_full = evaluate_model(self.model, self.val_loader, self.device)
        self.fp32_top5 = fp32_acc_full["top5"]

        # FP32 latency
        from utils.metrics import benchmark_latency, parse_hardware_report
        hp = self.config.hyperparams
        self.fp32_latency = benchmark_latency(
            self.model, self.config.input_shape, self.device,
            batch_size=hp.latency_batch_size,
            warmup_runs=hp.latency_warmup_runs,
            measure_runs=hp.latency_measure_runs,
        )

        # Hardware metrics (optional)
        self.hardware_metrics = parse_hardware_report(
            hp.hardware_report_path or None
        )

        # ── Wave 4 J1+J2+J3: FP32 ONNX export + ORT latency baseline ──
        # The FP32 ONNX export establishes the "deployable" upper bound
        # for both size (.onnx file size on disk) and latency (ORT
        # mean ms). Every quantized method later compares its real
        # numbers against this, instead of against the synthetic
        # ``numel × 32 / 8`` figure that has no disk presence.
        self.fp32_onnx: Dict[str, Any] = {}
        if getattr(hp, "onnx_export_enabled", True):
            try:
                from utils.onnx_export import (
                    export_quantize_and_benchmark, is_onnx_available,
                )
                if is_onnx_available():
                    onnx_dir = self.output_dir / "onnx"
                    # ``name`` becomes the file stem; the helper appends
                    # ``.fp32.onnx`` / ``.int8.onnx`` as the precision
                    # suffix. Pass "fp32_baseline" so the resulting file
                    # is ``fp32_baseline.fp32.onnx`` instead of the
                    # confusing ``fp32.fp32.onnx``.
                    self.fp32_onnx = export_quantize_and_benchmark(
                        self.model,
                        self.config.input_shape,
                        str(onnx_dir),
                        name="fp32_baseline",
                        calibration_loader=None,  # FP32 needs no calibration
                        do_int8=False,
                        batch_size=hp.latency_batch_size,
                        warmup_runs=hp.latency_warmup_runs,
                        measure_runs=hp.latency_measure_runs,
                    )
            except Exception as exc:
                logger.warning("FP32 ONNX export skipped: %s", exc)

        # Save checkpoint
        ckpt_path = self.output_dir / "checkpoint_fp32.pth"
        torch.save(self.model.state_dict(), ckpt_path)

        # Compute FP32 EBops + model size
        self.fp32_ebops = sum(
            p.numel() * 32 for p in self.model.parameters()
        ) / 8.0
        self.fp32_size_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)

        self.tracker.log_params({
            "model": self.config.model_name,
            "training_epochs": self.training_epochs,
            "dataset": self.config.dataset_name,
            "num_params": sum(p.numel() for p in self.model.parameters()),
        })
        # Public MLflow keys: Top-1 only. Top-5 is computed locally
        # for internal diagnostics (kept in the phase-0 checkpoint
        # JSON) but is NOT logged to MLflow as a public metric.
        fp32_metrics: Dict[str, float] = {
            "fp32_top1": self.fp32_acc,
            "fp32_ebops": self.fp32_ebops,
            "fp32_size_mb": self.fp32_size_mb,
            "fp32_latency_mean_ms": self.fp32_latency["latency_mean_ms"],
            "fp32_latency_p50_ms": self.fp32_latency["latency_p50_ms"],
            "fp32_latency_p95_ms": self.fp32_latency["latency_p95_ms"],
            "fp32_throughput_fps": self.fp32_latency["throughput_fps"],
        }
        # Wave 5 I1+I2: FP32 ONNX baseline metrics + artifact.
        fp32_onnx_size = self.fp32_onnx.get("fp32_onnx_size_mb")
        fp32_onnx_lat_full = self.fp32_onnx.get("onnx_latency") or {}
        if fp32_onnx_size is not None:
            fp32_metrics["fp32_onnx_size_mb"] = float(fp32_onnx_size)
        if fp32_onnx_lat_full:
            fp32_metrics["fp32_onnx_latency_mean_ms"] = float(
                fp32_onnx_lat_full.get("latency_mean_ms", 0.0)
            )
            fp32_metrics["fp32_onnx_throughput_fps"] = float(
                fp32_onnx_lat_full.get("throughput_fps", 0.0)
            )
        self.tracker.log_metrics(fp32_metrics)
        if self.fp32_onnx.get("fp32_onnx_path"):
            self.tracker.log_artifact(
                str(self.fp32_onnx["fp32_onnx_path"]), "onnx",
            )
        # Wave 5 G3: surface the ONNX baseline on ``self.results`` so
        # the reproducibility manifest can pick it up at end-of-run
        # without re-walking the file tree.
        if self.fp32_onnx:
            self.results["fp32_onnx"] = self.fp32_onnx
        self.tracker.end_run()

        # Build summary row for final report table — Top-1 only on the
        # public surface (Top-5 stays in internal metrics if computed).
        fp32_onnx_lat = (self.fp32_onnx.get("onnx_latency") or {})
        self._add_summary_row(
            "FP32", self.fp32_acc,
            self.fp32_latency["latency_mean_ms"],
            self.fp32_latency["throughput_fps"],
            self.fp32_ebops, self.fp32_size_mb,
            onnx_size_mb=self.fp32_onnx.get("fp32_onnx_size_mb"),
            onnx_latency_ms=fp32_onnx_lat.get("latency_mean_ms"),
            onnx_throughput_fps=fp32_onnx_lat.get("throughput_fps"),
        )

        self.report_lines.append(
            f"[Phase 0] FP32 baseline: top1={self.fp32_acc:.2f}%, "
            f"latency={self.fp32_latency['latency_mean_ms']:.1f}ms, "
            f"size={self.fp32_size_mb:.2f} MiB, "
            f"checkpoint={ckpt_path.name}"
        )
        self.results["fp32_acc"] = self.fp32_acc

        # Checkpoint
        self.ckpt.save_phase_full("phase_0_preparation", self.model, {
            "fp32_acc": self.fp32_acc,
            "fp32_top5": self.fp32_top5,
            "fp32_ebops": self.fp32_ebops,
            "fp32_size_mb": self.fp32_size_mb,
            "fp32_latency": self.fp32_latency,
            # Wave 5 G3: persist the FP32 ONNX baseline (path + size +
            # ORT latency) so resumes can recreate the deployment-
            # fidelity section without re-exporting.
            "fp32_onnx": self.fp32_onnx,
        })

    # ==================================================================
    # Phase 1a: Hessian + Clustering
    # ==================================================================

    def phase_1a_hessian_clustering(self) -> None:
        """Phase 1a: Compute Hessian diagonal and create layer clusters."""
        self.tracker.start_run("phase_1a_hessian", {"phase": "1a"})

        from quantization.hessian_clustering import HessianComputer, LayerClusterer

        hessian_comp = HessianComputer(self.model, self.config)
        self.hessian_diag = hessian_comp.compute_hessian(
            self.calib_loader, nn.CrossEntropyLoss(),
            num_batches=self.config.hyperparams.hessian_batches,
        )

        clusterer = LayerClusterer(self.model, self.hessian_diag, self.config)
        self.cluster_result = clusterer.create_clusters()
        self.cluster_assignments = self.cluster_result["cluster_assignments"]

        n_clusters = len(self.cluster_assignments)
        self.tracker.log_metrics({
            "num_layers": len(self.hessian_diag),
            "num_clusters": n_clusters,
        })

        # Per-layer sensitivity visualization — BEFORE end_run so
        # MLflow artifact logging works within the active run.
        try:
            from visualization.sensitivity import (
                plot_sensitivity_heatmap, plot_tier_distribution,
            )
            sens_dir = str(self.output_dir)
            sens_path = plot_sensitivity_heatmap(
                self.hessian_diag, self.cluster_result, sens_dir,
                model_name=self.config.model_name,
            )
            tier_path = plot_tier_distribution(
                self.cluster_result, sens_dir,
                model_name=self.config.model_name,
            )
            # Log plots to MLflow so they show in the Artifacts tab.
            if sens_path:
                self.tracker.log_artifact(sens_path, "plots")
            if tier_path:
                self.tracker.log_artifact(tier_path, "plots")
        except Exception as exc:
            logger.warning("  Sensitivity plots skipped: %s", exc)

        self.tracker.end_run()

        self.report_lines.append(
            f"[Phase 1a] Hessian: {len(self.hessian_diag)} layers, "
            f"{n_clusters} clusters"
        )
        self.results["hessian_layers"] = len(self.hessian_diag)
        self.results["num_clusters"] = n_clusters

        # Checkpoint
        self.ckpt.save_phase_json("phase_1a_hessian_clustering", {
            "hessian_diag": self.hessian_diag,
            "cluster_result": self.cluster_result,
        })

    # ==================================================================
    # Phase 1b: FITCompress
    # ==================================================================

    def phase_1b_fitcompress(self) -> None:
        """Phase 1b: Generate elite seed config via FITCompress."""
        self.tracker.start_run("phase_1b_fitcompress", {"phase": "1b"})

        from quantization.fitcompress import FITCompressSeedGenerator

        fit_gen = FITCompressSeedGenerator(
            self.model, self.hessian_diag, self.cluster_result, self.config,
        )
        self.fit_seed = fit_gen.generate_seed()

        self.tracker.log_metrics({
            "fit_compression": self.fit_seed["compression_potential"],
            "fit_int4_count": sum(
                1 for b in self.fit_seed["seed_config"].values() if b == 4
            ),
            "fit_int8_count": sum(
                1 for b in self.fit_seed["seed_config"].values() if b == 8
            ),
        })
        self.tracker.end_run()

        self.report_lines.append(
            f"[Phase 1b] FITCompress: compression={self.fit_seed['compression_potential']:.1f}%, "
            f"status={self.fit_seed['elite_status']}"
        )

        # Checkpoint
        self.ckpt.save_phase_json("phase_1b_fitcompress", self.fit_seed)

    # ==================================================================
    # Phase 1c: NSGA-II Search
    # ==================================================================

    def phase_1c_nsga_search(self) -> None:
        """Phase 1c: Run NSGA-II cluster-level search for PTQ configs."""
        hp = self.config.hyperparams
        self.tracker.start_run("phase_1c_nsga2", {"phase": "1c"})
        self.tracker.log_params({
            "pop_size": hp.nsga_population_size,
            "generations": hp.nsga_generations,
        })

        from quantization.nsga_ii_search import NSGAIIClusterSearch

        # ── Wave 4 J4: closed-loop hardware-aware search ──
        # When ``hardware_aware_search`` is True we build the per-layer
        # ORT latency LUT (C2) once and pass it to NSGA, which switches
        # to 3-objective mode ``[acc_loss, size_mb, latency_ms]``. The
        # LUT is summed over the candidate's bitwidth assignment — fast
        # enough to evaluate every gene without per-eval ONNX exports.
        # The LUT is cached to ``output_dir/latency_lut.json`` so
        # subsequent runs (and resumes) skip the ~minute rebuild cost.
        latency_lut: Optional[Dict[str, Dict[int, float]]] = None
        if getattr(hp, "hardware_aware_search", False):
            try:
                from utils.onnx_export import is_onnx_available
                if is_onnx_available():
                    from quantization.latency_lut import build_latency_lut
                    cache = self.output_dir / "latency_lut.json"
                    bws = tuple(getattr(hp, "latency_lut_bitwidths", [4, 8]))
                    logger.info(
                        "  Phase 1c: hardware-aware mode — building "
                        "per-layer ORT latency LUT (bitwidths=%s)",
                        list(bws),
                    )
                    latency_lut = build_latency_lut(
                        self.model, self.config.input_shape,
                        self.calib_loader,
                        bitwidths=bws,
                        warmup_runs=max(3, hp.latency_warmup_runs // 2),
                        measure_runs=max(10, hp.latency_measure_runs // 2),
                        cache_path=str(cache),
                    )
                    # Wave 5 G3: record the LUT cache path on the
                    # results dict for the reproducibility manifest.
                    self.results["latency_lut_path"] = str(cache)
                else:
                    logger.warning(
                        "  hardware_aware_search=True but ONNX runtime "
                        "is unavailable; falling back to 2-obj NSGA."
                    )
            except Exception as exc:
                logger.warning(
                    "  Latency LUT build failed (%s); falling back to "
                    "2-obj NSGA.", exc,
                )

        nsga = NSGAIIClusterSearch(
            self.model, self.cluster_assignments, self.config,
            latency_lut=latency_lut,
            # Phase 1a output is forwarded so per-layer mode + the
            # sensitivity-weighted mutation operator + the surrogate's
            # Hessian features all share the same source of truth.
            hessian_diag=getattr(self, "hessian_diag", None),
        )
        # NSGA fitness reads from the held-out ``search`` slice (10% of
        # the original train set, eval-time transforms). Falling back to
        # ``val_loader`` only if a legacy resume produced no search
        # split — in fresh runs this branch is never taken.
        nsga_loader = self.search_loader or self.val_loader
        self.pareto_front = nsga.search(
            nsga_loader, self.fp32_acc, self.fit_seed["seed_config"],
        )

        n_pareto = len(self.pareto_front["solutions"])
        self.tracker.log_metrics({
            "pareto_solutions": n_pareto,
            "nsga_evaluations": self.pareto_front["evaluations"],
        })
        self.tracker.end_run()

        # Store the PTQ materialization config for downstream phases.
        # Prefer a mixed (INT4+INT8) solution when available so the
        # resulting PTQ artifact reflects the mixed-precision flow.
        if self.pareto_front["solutions"]:
            ranked = self.pareto_front["solutions"]
            selected = ranked[0]
            # Prefer the most BALANCED mixed-bitwidth solution — i.e. the
            # one whose INT4 fraction is closest to 50%. The previous
            # logic just picked ``mixed_ranked[0]``, which routinely
            # selected solutions with a single INT4 layer (e.g. only
            # the classifier head). That made AdaRound a no-op (it
            # only has weights with bitwidth < 8 to learn-round) and
            # masked the real mixed-precision behaviour in the headline
            # numbers. Balance is the right tie-breaker because it
            # exercises both code paths and gives the downstream
            # AdaRound / QAT phases something meaningful to do.
            mixed_ranked = [
                s for s in ranked
                if self._is_mixed_bitwidth_assignment(
                    s.get("bitwidth_assignment", {}),
                )
            ]
            if mixed_ranked:
                def _balance_score(sol: Dict[str, Any]) -> float:
                    bw = sol.get("bitwidth_assignment", {}) or {}
                    int4 = sum(1 for v in bw.values() if int(v) == 4)
                    total = sum(1 for v in bw.values() if int(v) in (4, 8))
                    if total == 0:
                        return 1.0  # worst
                    return abs(int4 / total - 0.5)
                mixed_ranked.sort(key=_balance_score)
                selected = mixed_ranked[0]
                bw = selected.get("bitwidth_assignment", {})
                int4 = sum(1 for v in bw.values() if int(v) == 4)
                int8 = sum(1 for v in bw.values() if int(v) == 8)
                logger.info(
                    "  Phase 1c: selected mixed PTQ config %s for "
                    "materialization (acc=%.2f%%, size=%.2f MiB, "
                    "INT4/INT8 split=%d/%d).",
                    selected.get("solution_id", "unknown"),
                    float(selected.get("accuracy", 0.0)),
                    float(selected.get("model_size_mb", 0.0)),
                    int4, int8,
                )
            self.best_config = selected.get("bitwidth_assignment", {})

        self.report_lines.append(
            f"[Phase 1c] NSGA-II: {n_pareto} Pareto solutions, "
            f"{self.pareto_front['evaluations']} evals"
        )
        self.results["pareto_solutions"] = n_pareto

        # ── Multi-fidelity PTQ rerank ──
        # NSGA-II searches with fake-quant for speed; the proxy ranking
        # is not the same as real PTQ ranking. We materialise the top-K
        # NSGA candidates through PTQQuantizer + bitwidth-AWARE
        # calibration, then pick:
        #
        #   * ptq_best_acc       — highest real Top-1
        #   * ptq_best_tradeoff  — most compressed candidate within
        #                          hp.ptq_tradeoff_max_acc_drop pp of
        #                          FP32; falls back to the smallest-size
        #                          candidate when none satisfies the cap.
        #
        # Both go into ``self.method_results`` (deduplicated) so the
        # public phase-2 Pareto and the phase-3 XAI matrix can show
        # them as distinct rows when they are not identical.
        methods_enabled = {m.value.lower() for m in self.config.methods}
        ptq_enabled = ("ptq" in methods_enabled or not methods_enabled)

        ptq_best_acc_result: Optional[Dict[str, Any]] = None
        ptq_best_tradeoff_result: Optional[Dict[str, Any]] = None
        ptq_best_acc_model: Optional[nn.Module] = None
        ptq_best_tradeoff_model: Optional[nn.Module] = None

        if self.best_config and ptq_enabled:
            ranked_candidates = self._select_rerank_candidates(
                self.pareto_front.get("solutions", []),
                int(getattr(hp, "ptq_real_rerank_topk", 3)),
            )
            if not ranked_candidates:
                # No NSGA solutions to rerank — fall back to ``best_config``.
                ranked_candidates = [{
                    "solution_id": "ptq_candidate_0",
                    "bitwidth_assignment": self.best_config,
                }]

            (
                ptq_best_acc_model, ptq_best_acc_result,
                ptq_best_tradeoff_model, ptq_best_tradeoff_result,
            ) = self._materialize_and_rerank_ptq(
                ranked_candidates, hp,
            )

            # Surface to results + method_results (avoid duplicates when
            # both pickers landed on the same configuration).
            self.results["ptq_best_acc_result"] = ptq_best_acc_result
            self.results["ptq_best_tradeoff_result"] = ptq_best_tradeoff_result
            if ptq_best_acc_model is not None:
                self.results["ptq_best_acc_model"] = ptq_best_acc_model
                # Backwards compat: legacy keys consumed by phase 3 / phase 2.
                self.results["ptq_model"] = ptq_best_acc_model
                self.results["ptq_best_result"] = ptq_best_acc_result

            distinct = (
                ptq_best_tradeoff_result is not None
                and ptq_best_acc_result is not None
                and ptq_best_tradeoff_result.get("display_name")
                    != ptq_best_acc_result.get("display_name")
            )
            if ptq_best_tradeoff_model is not None and distinct:
                self.results["ptq_best_tradeoff_model"] = ptq_best_tradeoff_model

            # Append to method_results (this is what phase 2 walks).
            for res in (ptq_best_acc_result, ptq_best_tradeoff_result):
                if res is None:
                    continue
                if any(
                    r.get("display_name") == res["display_name"]
                    for r in self.method_results
                ):
                    continue
                self.method_results.append(res)
                lat = res.get("latency") or {}
                onnx_lat = res.get("onnx_latency") or {}
                self._add_summary_row(
                    res["display_name"],
                    res["accuracy"],
                    lat.get("latency_mean_ms", 0.0),
                    lat.get("throughput_fps", 0.0),
                    res["ebops"],
                    res.get("theoretical_size_mb", res["model_size_mb"]),
                    onnx_size_mb=res.get("onnx_size_mb"),
                    onnx_latency_ms=onnx_lat.get("latency_mean_ms"),
                    onnx_throughput_fps=onnx_lat.get("throughput_fps"),
                )
                self.report_lines.append(
                    f"[Phase 1c] {res['display_name']}: "
                    f"acc={res['accuracy']:.2f}%, "
                    f"size={res['model_size_mb']:.2f} MiB"
                )

        # Resolve the QAT/Adaround warmstart source from config and lock
        # in ``self.best_config`` (consumed by phases 1d/1e). This makes
        # the warmstart choice EXPLICIT and persisted.
        warmstart_source = getattr(hp, "qat_warmstart_source", "ptq_best_acc")
        warmstart_source = (warmstart_source or "ptq_best_acc").lower()
        warmstart_pick: Optional[Dict[str, Any]] = None
        if warmstart_source == "ptq_best_tradeoff" and ptq_best_tradeoff_result:
            warmstart_pick = ptq_best_tradeoff_result
        elif warmstart_source == "ptq_best_acc" and ptq_best_acc_result:
            warmstart_pick = ptq_best_acc_result
        elif ptq_best_acc_result is not None:
            warmstart_pick = ptq_best_acc_result  # safe default

        if warmstart_pick is not None:
            self.best_config = dict(warmstart_pick.get("bitwidth_assignment") or {})
            self.results["qat_warmstart_source"] = warmstart_source
            self.results["qat_warmstart_id"] = warmstart_pick.get("display_name")
            logger.info(
                "  Phase 1c: QAT warmstart source = %s → %s",
                warmstart_source, warmstart_pick.get("display_name"),
            )

        # ── Persist a SEPARATE config for AdaRound ──
        # ``self.best_config`` is now whatever the QAT warmstart resolved
        # to — which can be an all-INT8 solution if the warmstart source
        # is ``ptq_best_acc`` and the highest-accuracy NSGA solution is
        # uniform. AdaRound on an all-INT8 config is a no-op (INT8 is
        # too close to FP32 for learned rounding to find headroom).
        # ``adaround_config`` keeps the most-balanced *mixed* solution
        # from the Pareto front so AdaRound always has real INT4 layers
        # to work on. Falls back to ``best_config`` when no mixed
        # solutions exist (e.g. all candidates are uniform INT8).
        mixed_solutions = [
            s for s in self.pareto_front.get("solutions", [])
            if self._is_mixed_bitwidth_assignment(
                s.get("bitwidth_assignment", {})
            )
        ]
        if mixed_solutions:
            def _balance_score(sol: Dict[str, Any]) -> float:
                bw = sol.get("bitwidth_assignment", {}) or {}
                int4 = sum(1 for v in bw.values() if int(v) == 4)
                total = sum(1 for v in bw.values() if int(v) in (4, 8))
                return abs(int4 / total - 0.5) if total > 0 else 1.0
            mixed_solutions.sort(key=_balance_score)
            self.adaround_config = dict(
                mixed_solutions[0].get("bitwidth_assignment") or {}
            )
            int4 = sum(1 for v in self.adaround_config.values() if int(v) == 4)
            int8 = sum(1 for v in self.adaround_config.values() if int(v) == 8)
            logger.info(
                "  Phase 1c: AdaRound config from %s "
                "(INT4/INT8 split=%d/%d).",
                mixed_solutions[0].get("solution_id", "?"), int4, int8,
            )
        else:
            self.adaround_config = dict(self.best_config)
            logger.info(
                "  Phase 1c: no mixed Pareto solution; AdaRound will use "
                "the warmstart config (uniform-bitwidth no-op)."
            )

        # Checkpoint
        self.ckpt.save_phase_json("phase_1c_nsga_search", {
            "pareto_front": self.pareto_front,
            "best_config": self.best_config,
            "adaround_config": self.adaround_config,
            "ptq_best_acc_result": ptq_best_acc_result,
            "ptq_best_tradeoff_result": ptq_best_tradeoff_result,
            "qat_warmstart_source": warmstart_source,
            "qat_warmstart_id": (warmstart_pick or {}).get("display_name"),
        })

    # ==================================================================
    # Phase 1d: Adaround
    # ==================================================================

    def phase_1d_adaround(self) -> None:
        """Phase 1d: Optimize weight rounding with Adaround."""
        self.tracker.start_run("phase_1d_adaround", {"phase": "1d"})

        from quantization.adaround import AdaroundOptimizer

        # Use ``adaround_config`` (most-balanced mixed Pareto solution)
        # rather than ``best_config`` (QAT warmstart pick — often
        # uniform INT8). Falls back to ``best_config`` only when no
        # mixed solution exists in the Pareto front.
        ada_cfg = self.adaround_config or self.best_config
        adaround_model = copy.deepcopy(self.model)
        adaround_opt = AdaroundOptimizer(
            adaround_model, ada_cfg, self.config,
            calib_loader=self.calib_loader,
        )
        self.adaround_result = adaround_opt.run()

        # Public MLflow metrics: weight MSE before/after + reduction,
        # plus the *real* layer-output reconstruction reduction (when
        # the calibration loader was supplied and recon_* were filled).
        ml_metrics = {
            "adaround_mse_before": self.adaround_result["mse_before"],
            "adaround_mse_after": self.adaround_result["mse_after"],
            "adaround_mse_reduction": self.adaround_result["mse_reduction"],
        }
        if self.adaround_result.get("recon_before") is not None:
            ml_metrics["adaround_recon_before"] = float(
                self.adaround_result["recon_before"])
        if self.adaround_result.get("recon_after") is not None:
            ml_metrics["adaround_recon_after"] = float(
                self.adaround_result["recon_after"])
        if self.adaround_result.get("recon_reduction") is not None:
            ml_metrics["adaround_recon_reduction"] = float(
                self.adaround_result["recon_reduction"])
        self.tracker.log_metrics(ml_metrics)
        self.tracker.end_run()

        recon_red = self.adaround_result.get("recon_reduction")
        if recon_red is not None:
            self.report_lines.append(
                f"[Phase 1d] Adaround: weight-MSE redux="
                f"{self.adaround_result['mse_reduction']:.1f}%, "
                f"output-recon redux={recon_red:.1f}%"
            )
        else:
            self.report_lines.append(
                f"[Phase 1d] Adaround: weight-MSE redux="
                f"{self.adaround_result['mse_reduction']:.1f}% "
                f"(weight-only objective; no calib loader)"
            )

        # Checkpoint metadata also captures the reconstruction diagnostics
        # and objective components so they survive resume.
        adaround_meta = {
            "mse_before": self.adaround_result["mse_before"],
            "mse_after": self.adaround_result["mse_after"],
            "mse_reduction": self.adaround_result["mse_reduction"],
            "recon_before": self.adaround_result.get("recon_before"),
            "recon_after": self.adaround_result.get("recon_after"),
            "recon_reduction": self.adaround_result.get("recon_reduction"),
            "objective_components": self.adaround_result.get(
                "objective_components", {},
            ),
            "time_seconds": self.adaround_result.get("time_seconds", 0.0),
        }
        adaround_m = self.adaround_result.get("model")
        if adaround_m and isinstance(adaround_m, nn.Module):
            self.ckpt.save_phase_model(
                "phase_1d_adaround", adaround_m, adaround_meta,
            )
        self.ckpt.save_phase_json("phase_1d_adaround", adaround_meta)

    # ==================================================================
    # Phase 1e: QAT Warmstart
    # ==================================================================

    def phase_1e_qat(self) -> None:
        """Phase 1e: QAT fine-tuning from best PTQ model."""
        self.tracker.start_run("phase_1e_qat", {"phase": "1e"})

        from quantization.qat import QATTrainer

        # Production W+A QAT: hand the trainer the FP32 baseline as a
        # KD teacher and the calibration loader for activation observer
        # initialisation. Wave-2 contract:
        #   - BN is folded into the preceding Conv inside ``prepare_model``.
        #   - Activations are observed once on calib data, then frozen
        #     at the deployment-time INT8 (or ``qat_act_bitwidth``) scale.
        #   - Weights are fake-quantized via an autograd-aware
        #     parametrization so STE clipping actually fires.
        qat_model = copy.deepcopy(self.adaround_result["model"])
        teacher = copy.deepcopy(self.model)  # untouched FP32 baseline
        qat_trainer = QATTrainer(
            qat_model,
            self.best_config,
            self.config,
            teacher=teacher,
            calib_loader=self.calib_loader,
        )
        self.qat_result = qat_trainer.train(self.train_loader, self.val_loader)

        # Headline accuracy for QAT comes from ``test_loader`` — the
        # final QAT model is evaluated against the public test split
        # exactly once, here, after the best-epoch weights have been
        # restored. ``final_val_acc`` (used to drive early stopping)
        # remains in the result for diagnostics.
        qat_test_top1 = self._evaluate_top1(
            self.qat_result.get("model"), self.test_loader,
        )
        if qat_test_top1 is not None:
            self.qat_result["test_top1"] = float(qat_test_top1)
            self.results["qat_test_top1"] = float(qat_test_top1)

        self.tracker.log_metrics({
            "qat_final_acc": self.qat_result["final_val_acc"],
            "qat_test_top1": qat_test_top1 if qat_test_top1 is not None else 0.0,
            "qat_epochs": len(self.qat_result["val_accuracy"]),
        })
        ws_src = self.results.get("qat_warmstart_source") or "ptq_best_acc"
        ws_id = self.results.get("qat_warmstart_id") or "n/a"
        self.tracker.log_params({
            "qat_warmstart_source": str(ws_src),
            "qat_warmstart_id": str(ws_id),
        })
        self.tracker.end_run()

        if qat_test_top1 is not None:
            self.report_lines.append(
                f"[Phase 1e] QAT: test_top1={qat_test_top1:.2f}%, "
                f"val_top1={self.qat_result['final_val_acc']:.2f}% "
                f"(warmstart={ws_src}={ws_id})"
            )
        else:
            self.report_lines.append(
                f"[Phase 1e] QAT: val_top1={self.qat_result['final_val_acc']:.2f}% "
                f"(warmstart={ws_src}={ws_id})"
            )
        self.results["qat_acc"] = self.qat_result["final_val_acc"]

        # Surface QAT into the public summary table so it appears in
        # ``pareto_summary.json`` alongside PTQ/GPTQ/AWQ/SmoothQuant.
        # Previously QAT was logged to MLflow + persisted to the phase
        # checkpoint but never made it into ``_summary_rows``, which is
        # what ``_build_pareto_summary`` consumes — so the headline JSON
        # silently dropped it (n_methods=7 with 8 candidates evaluated).
        # Size + ebops come from ``self.best_config`` (the NSGA-chosen
        # bitwidth assignment QAT was warmstarted on).
        from utils.common import compute_quantized_size_mb
        qat_bw = self.best_config or {}
        # Use the same MIXED-vs-uniform label rule as the PTQ path: the
        # warmstart often hands QAT a mixed-bitwidth config (when
        # ``mixed_ranked`` won the picker), in which case calling the
        # result "QAT_INT8" hides the real bitwidth distribution.
        # Tagging it ``QAT_MIXED`` lets the bitwidth-distribution chart
        # and method tables agree about what's actually quantized.
        if self._is_mixed_bitwidth_assignment(qat_bw):
            qat_tag = "MIXED"
            qat_dom_bw = self._dominant_bitwidth(qat_bw)
        else:
            qat_dom_bw = (
                self._dominant_bitwidth(qat_bw) if qat_bw
                else int(self.config.hyperparams.qat_act_bitwidth)
            )
            qat_tag = f"INT{qat_dom_bw}"
        qat_size_mb = (
            compute_quantized_size_mb(self.model, qat_bw) if qat_bw else 0.0
        )
        qat_ebops = self._ebops_from_bitwidth(qat_bw) if qat_bw else 0.0
        qat_headline_acc = (
            float(qat_test_top1) if qat_test_top1 is not None
            else float(self.qat_result.get("final_val_acc", 0.0) or 0.0)
        )

        # ── ONNX export for QAT ──
        # Previously skipped: QAT_INT8 had ``onnx_size_mb=null`` in the
        # public summary, masking deployment fidelity for the highest-
        # accuracy configuration. The fake-quantized model can be
        # exported through the same J1+J2+J3 path the phase-1f methods
        # use; ORT's static_quantize then materialises real INT8 ops.
        qat_res: Dict[str, Any] = {
            "display_name": f"QAT_{qat_tag}",
            "bitwidth_assignment": qat_bw,
            "model_size_mb": qat_size_mb,
            "theoretical_size_mb": qat_size_mb,
            "ebops": qat_ebops,
        }
        qat_model = self.qat_result.get("model")
        if qat_model is not None and isinstance(qat_model, nn.Module):
            self._export_method_to_onnx(
                qat_res, qat_model, qat_dom_bw, qat_bw,
            )
        # Prefer the ORT-measured latency over the (often missing) PyTorch
        # one; latency is a deployment metric, so the deployable runtime
        # number is the right one to surface in the summary table.
        qat_lat_dict = (
            qat_res.get("onnx_latency")
            or self.qat_result.get("latency")
            or {}
        )

        self._add_summary_row(
            f"QAT_{qat_tag}",
            qat_headline_acc,
            float(qat_lat_dict.get("latency_mean_ms", 0.0) or 0.0),
            float(qat_lat_dict.get("throughput_fps", 0.0) or 0.0),
            qat_ebops,
            qat_res.get("model_size_mb", qat_size_mb),
            onnx_size_mb=qat_res.get("onnx_size_mb"),
            onnx_latency_ms=qat_lat_dict.get("latency_mean_ms"),
            onnx_throughput_fps=qat_lat_dict.get("throughput_fps"),
        )
        # Persist the ONNX deployment fields onto the QAT result so the
        # checkpoint metadata + MLflow run reflect the export.
        if qat_res.get("onnx_path"):
            self.qat_result["onnx_path"] = qat_res["onnx_path"]
            self.qat_result["onnx_size_mb"] = qat_res.get("onnx_size_mb")
            self.qat_result["onnx_latency"] = qat_res.get("onnx_latency")

        # Checkpoint: save the fine-tuned QAT model and the metric history.
        # The warmstart source (ptq_best_acc / ptq_best_tradeoff) and the
        # PTQ ID it selected are persisted alongside so resuming or
        # post-mortem analysis can answer "which PTQ was QAT trained on?".
        qat_meta = {
            "final_val_acc": self.qat_result.get("final_val_acc", 0.0),
            "test_top1": self.qat_result.get("test_top1"),
            "best_epoch": self.qat_result.get("best_epoch", 0),
            "train_accuracy": self.qat_result.get("train_accuracy", []),
            "val_accuracy": self.qat_result.get("val_accuracy", []),
            "train_loss": self.qat_result.get("train_loss", []),
            "time_seconds": self.qat_result.get("time_seconds", 0.0),
            "qat_warmstart_source": self.results.get("qat_warmstart_source"),
            "qat_warmstart_id": self.results.get("qat_warmstart_id"),
        }
        qat_m = self.qat_result.get("model")
        if qat_m and isinstance(qat_m, nn.Module):
            self.ckpt.save_phase_model("phase_1e_qat", qat_m, qat_meta)
        self.ckpt.save_phase_json("phase_1e_qat", qat_meta)

    # ==================================================================
    # Phase 1f: GPTQ + AWQ + SmoothQuant
    # ==================================================================

    def phase_1f_gptq_smooth_awq(self) -> None:
        """Phase 1f: Run GPTQ/AWQ/SmoothQuant at INT8 and low-bit variants.

        Runs up to six configurations (GPTQ INT8+INT4, AWQ mixed+INT8,
        SmoothQuant INT8+INT4) filtered by config.methods. Results are
        appended to ``self.method_results`` so any entries already
        present (e.g. PTQ best from phase 1c, QAT from phase 1e) are
        preserved for the phase‑2 merge.
        """
        self.tracker.start_run("phase_1f_methods", {"phase": "1f"})

        from quantization.gptq import GPTQQuantizer
        from quantization.awq import AWQQuantizer
        from quantization.smoothquant import SmoothQuantQuantizer
        from quantization.smoothquant_gptq import SmoothQuantGPTQQuantizer

        hp = self.config.hyperparams
        methods_enabled = {m.value.lower() for m in self.config.methods}

        # Filterable per-method config matrix. Each entry quantises then
        # evaluates at the given bitwidth; the resulting nn.Module is
        # stashed for phase 3 and persisted by the checkpoint manager.
        plan = [
            ("gptq", "GPTQ", 8, GPTQQuantizer),
            ("gptq", "GPTQ", 4, GPTQQuantizer),
            ("awq",  "AWQ",  4, AWQQuantizer),   # "mixed" variant
            ("awq",  "AWQ",  8, AWQQuantizer),
            ("smoothquant", "SmoothQuant", 8, SmoothQuantQuantizer),
            ("smoothquant", "SmoothQuant", 4, SmoothQuantQuantizer),
            # F4: combined two-stage (SmoothQuant migration → GPTQ).
            # Strict-Pareto improvement over either method alone in
            # almost every configuration we have measured.
            ("smoothquant_gptq", "SmoothQuantGPTQ", 8, SmoothQuantGPTQQuantizer),
            ("smoothquant_gptq", "SmoothQuantGPTQ", 4, SmoothQuantGPTQQuantizer),
        ]

        produced_models: Dict[str, nn.Module] = {}
        produced_models_by_id: Dict[str, nn.Module] = {}
        new_results: List[Dict[str, Any]] = []
        from utils.common import compute_quantized_size_mb

        onnx_enabled = bool(getattr(hp, "onnx_export_enabled", True))
        onnx_dir = self.output_dir / "onnx"
        if onnx_enabled:
            from utils.onnx_export import (
                export_quantize_and_benchmark, is_onnx_available,
            )
            if not is_onnx_available():
                onnx_enabled = False
                logger.warning(
                    "  ONNX export disabled — onnx/onnxruntime not importable. "
                    "Install with `pip install onnx onnxruntime` to enable real "
                    "INT8 disk size + ORT latency (Wave 4 J1/J2/J3). All "
                    "onnx_size_mb / onnx_latency_ms / onnx_throughput_fps "
                    "fields will be null in the final summary."
                )

        # Phase 1a outputs are passed as keyword args to quantizers that
        # accept them (SmoothQuant, AWQ, SmoothQuant→GPTQ). When clusters
        # are absent or the quantizer ignores them, the kwargs are simply
        # not forwarded — keeps GPTQ's signature unchanged.
        cluster_kwargs = {
            "cluster_result": getattr(self, "cluster_result", None),
            "hessian_diag": getattr(self, "hessian_diag", None),
        }
        cluster_aware_keys = {"smoothquant", "awq", "smoothquant_gptq"}

        for i, (key, label, bw, cls) in enumerate(plan, 1):
            if methods_enabled and key not in methods_enabled:
                continue
            logger.info("  [%d/%d] %s INT%d ...", i, len(plan), label, bw)
            try:
                quantizer_kwargs = (
                    cluster_kwargs if key in cluster_aware_keys else {}
                )
                quantizer = cls(self.model, self.config, **quantizer_kwargs)
                q_model = quantizer.quantize(
                    self.calib_loader, bitwidth=bw,
                    num_batches=hp.calibration_batches,
                )
                res = quantizer.evaluate(q_model, self.val_loader, bitwidth=bw)
                res["config_id"] = f"{label}_INT{bw}"
                res["display_name"] = f"{label}_INT{bw}"
                res["bitwidth"] = int(bw)
                # Synthetic (theoretical) size from the bitwidth
                # assignment is preserved as a diagnostic; the public
                # ``model_size_mb`` is overwritten with the on-disk
                # ``.onnx`` file size below when ONNX export succeeds.
                # IMPORTANT: ``bw_assignment`` keys come from the
                # *quantized* model's namespace (see BaseQuantizer.evaluate),
                # which differs from ``self.model`` for methods that wrap
                # layers — e.g. SmoothQuant/AWQ insert input-scale wrappers
                # turning ``features.0.0.weight`` into ``features.0.0.1.weight``.
                # Passing ``self.model`` (FP32 namespace) caused every key to
                # miss in ``compute_ebops`` and silently fall back to 32-bit
                # accounting → AWQ/SmoothQuant reported FP32 sizes.
                # When the quantizer didn't return a bitwidth_assignment
                # (some methods skip this), build one over QUANTIZABLE
                # weights only — the weight tensors of Conv2d / Linear
                # modules. Including BatchNorm γ/β, biases, or the
                # _SmoothInputScale / _AWQInputScale wrapper buffers
                # (which named_parameters() also walks) inflated the
                # apparent layer count from 53 to 158 in the bitwidth
                # distribution chart, falsely showing AWQ/SmoothQuant
                # as quantizing 3× more layers than they actually do.
                from torch.nn import Conv1d, Conv2d, Conv3d, Linear
                _qmod_owners = (Conv1d, Conv2d, Conv3d, Linear)
                _qmod_index = {
                    name: mod for name, mod in q_model.named_modules()
                    if isinstance(mod, _qmod_owners)
                }

                def _is_quantizable_weight(pname: str) -> bool:
                    if not pname.endswith(".weight"):
                        return False
                    return pname.rsplit(".", 1)[0] in _qmod_index

                bw_assignment = res.get("bitwidth_assignment", {}) or {
                    n: bw for n, _ in q_model.named_parameters()
                    if _is_quantizable_weight(n)
                }
                res["bitwidth_assignment"] = bw_assignment
                theoretical_mb = compute_quantized_size_mb(
                    q_model, bw_assignment,
                )
                res["theoretical_size_mb"] = theoretical_mb
                res["model_size_mb"] = theoretical_mb

                # ── J1 + J2 + J3: real ONNX export, on-disk size, ORT latency ──
                if onnx_enabled:
                    try:
                        info = export_quantize_and_benchmark(
                            q_model,
                            self.config.input_shape,
                            str(onnx_dir),
                            name=f"{res['display_name']}",
                            calibration_loader=self.calib_loader,
                            num_batches=min(8, hp.calibration_batches),
                            do_int8=(bw <= 8),
                            batch_size=hp.latency_batch_size,
                            warmup_runs=hp.latency_warmup_runs,
                            measure_runs=hp.latency_measure_runs,
                        )
                        if info.get("int8_onnx_size_mb") is not None:
                            res["onnx_size_mb"] = info["int8_onnx_size_mb"]
                            res["onnx_path"] = info["int8_onnx_path"]
                            res["model_size_mb"] = info["int8_onnx_size_mb"]
                        elif info.get("fp32_onnx_size_mb") is not None:
                            res["onnx_size_mb"] = info["fp32_onnx_size_mb"]
                            res["onnx_path"] = info["fp32_onnx_path"]
                            res["model_size_mb"] = info["fp32_onnx_size_mb"]
                        if info.get("onnx_latency"):
                            res["onnx_latency"] = info["onnx_latency"]
                            res["latency_ms"] = info["onnx_latency"]["latency_mean_ms"]
                            res["latency"] = info["onnx_latency"]
                    except Exception as exc:
                        logger.warning(
                            "    %s ONNX export failed: %s — falling back to "
                            "theoretical_size_mb. Real on-disk size + ORT "
                            "latency unavailable for this method.",
                            res["display_name"], exc,
                        )

                # INT4 packing estimate: show the gap between ONNX's
                # INT8-container storage and true INT4 packed size.
                if bw == 4 or (bw_assignment and 4 in bw_assignment.values()):
                    try:
                        from utils.onnx_export import estimate_int4_packed_size_mb
                        packing = estimate_int4_packed_size_mb(
                            q_model, bw_assignment or {},
                        )
                        res["int4_packing"] = packing
                        if packing.get("packing_note"):
                            logger.info(
                                "    %s: %s",
                                res["display_name"], packing["packing_note"],
                            )
                    except Exception:
                        pass  # non-critical

                self._attach_split_metrics(res, q_model)
                self.method_results.append(res)
                new_results.append(res)
                produced_models.setdefault(key, q_model)
                produced_models_by_id[res["display_name"]] = q_model
                logger.info(
                    "    %s INT%d: test_top1=%.2f%%, val_top1=%.2f%%",
                    label, bw,
                    float(res.get("test_top1") or res.get("accuracy", 0.0)),
                    float(res.get("val_top1") or 0.0),
                )
            except Exception as e:
                logger.warning("    %s INT%d FAILED: %s", label, bw, e)

        # Log results to MLflow with bitwidth-tagged keys; Top-5 is NOT
        # surfaced (kept internal only). Wave 5 I1+I2: when an ONNX
        # artefact exists for the method, log the on-disk size, ORT
        # latency stats, and ORT throughput as public metrics, and
        # attach the .onnx file to the run as an artifact so MLflow's
        # Compare Runs view shows the actual deployable model alongside
        # its accuracy/size numbers.
        for res in new_results:
            tag = res["display_name"]
            metrics: Dict[str, float] = {
                f"{tag}_top1": res["accuracy"],
                f"{tag}_ebops": res["ebops"],
                f"{tag}_size_mb": res["model_size_mb"],
                f"{tag}_latency_ms": res.get("latency_ms", 0) or 0,
            }
            theo = res.get("theoretical_size_mb")
            if theo is not None:
                metrics[f"{tag}_theoretical_size_mb"] = float(theo)
            onnx_lat_full = res.get("onnx_latency") or {}
            if res.get("onnx_size_mb") is not None:
                metrics[f"{tag}_onnx_size_mb"] = float(res["onnx_size_mb"])
            if onnx_lat_full:
                metrics[f"{tag}_onnx_latency_mean_ms"] = float(
                    onnx_lat_full.get("latency_mean_ms", 0.0)
                )
                metrics[f"{tag}_onnx_latency_p50_ms"] = float(
                    onnx_lat_full.get("latency_p50_ms", 0.0)
                )
                metrics[f"{tag}_onnx_latency_p95_ms"] = float(
                    onnx_lat_full.get("latency_p95_ms", 0.0)
                )
                metrics[f"{tag}_onnx_throughput_fps"] = float(
                    onnx_lat_full.get("throughput_fps", 0.0)
                )
            self.tracker.log_metrics(metrics)

            # Attach the deployable ONNX file to the MLflow run.
            onnx_path = res.get("onnx_path")
            if onnx_path:
                self.tracker.log_artifact(str(onnx_path), "onnx")

            lat = res.get("latency") or {}
            onnx_lat = res.get("onnx_latency") or {}
            self._add_summary_row(
                res["display_name"],
                res["accuracy"],
                lat.get("latency_mean_ms", 0) or 0.0,
                lat.get("throughput_fps", 0) or 0.0,
                res["ebops"],
                res.get("theoretical_size_mb", res["model_size_mb"]),
                onnx_size_mb=res.get("onnx_size_mb"),
                onnx_latency_ms=onnx_lat.get("latency_mean_ms"),
                onnx_throughput_fps=onnx_lat.get("throughput_fps"),
            )

        # ── Quantization Error Attribution ──
        # Compute per-layer activation errors between FP32 and each
        # quantized model, generating visual diagnostics. The PNGs land
        # in their own ``error_attribution/`` subdirectory so the
        # top-level ``artifacts/`` folder doesn't fan out across N
        # methods × M plots — the report.py glob still picks them up.
        # Runs BEFORE end_run() so MLflow artifact logging works.
        try:
            from visualization.error_attribution import (
                compute_layer_errors, plot_error_attribution,
                plot_error_comparison,
            )
            ea_subdir = self.output_dir / "error_attribution"
            ea_subdir.mkdir(parents=True, exist_ok=True)
            ea_dir = str(ea_subdir)
            all_errors: Dict[str, list] = {}
            for label, q_model in produced_models_by_id.items():
                if not isinstance(q_model, nn.Module):
                    continue
                try:
                    errors = compute_layer_errors(
                        self.model, q_model, self.calib_loader,
                        self.device, num_batches=3,
                    )
                    if errors:
                        all_errors[label] = errors
                        ea_path = plot_error_attribution(
                            errors, ea_dir,
                            method_name=label,
                            model_name=self.config.model_name,
                        )
                        if ea_path:
                            self.tracker.log_artifact(ea_path, "plots")
                except Exception as exc:
                    logger.debug("  Error attribution for %s: %s", label, exc)
            if len(all_errors) > 1:
                cmp_path = plot_error_comparison(
                    all_errors, ea_dir,
                    model_name=self.config.model_name,
                )
                if cmp_path:
                    self.tracker.log_artifact(cmp_path, "plots")
        except Exception as exc:
            logger.warning("  Error attribution skipped: %s", exc)

        self.tracker.end_run()

        # Short per-method report line (best of each family).
        def _best(label: str) -> str:
            hits = [r for r in new_results if r["method"].lower().startswith(label.lower())]
            if not hits:
                return f"{label}=skip"
            best = max(hits, key=lambda r: r["accuracy"])
            return f"{label}={best['accuracy']:.2f}%@{best['config_id']}"

        self.report_lines.append(
            f"[Phase 1f] {_best('GPTQ')}, {_best('AWQ')}, {_best('SmoothQuant')}"
        )

        # Store every bitwidth-tagged model for Phase 3 (XAI shows all
        # of them as separate rows). Family-level keys remain as the
        # default "first INT8 if present, else first INT4" so downstream
        # consumers that only want one rep keep working.
        self.results["phase_1f_models"] = dict(produced_models_by_id)
        if "gptq" in produced_models:
            self.results["gptq_model"] = produced_models["gptq"]
        if "awq" in produced_models:
            self.results["awq_model"] = produced_models["awq"]
        if "smoothquant" in produced_models:
            self.results["sq_model"] = produced_models["smoothquant"]
        if "smoothquant_gptq" in produced_models:
            self.results["sq_gptq_model"] = produced_models["smoothquant_gptq"]

        # Checkpoint: persist each family's representative so phase 3 can
        # resume without re-running quantization, and the method results
        # so phase 2 can merge solutions on resume. AWQ + SmoothQuant
        # + SmoothQuant→GPTQ all carry input-scale wrappers, so the
        # safe-module path (state_dict + JSON manifest) is used; loads
        # run under ``weights_only=True`` and never execute pickle.
        if "gptq" in produced_models:
            self.ckpt.save_named_model("phase_1f_gptq_model.pth",
                                       produced_models["gptq"])
        if "awq" in produced_models:
            from quantization.awq import serialize_awq_metadata
            awq_meta = serialize_awq_metadata(produced_models["awq"])
            self.ckpt.save_safe_module(
                "phase_1f_awq_model.pt",
                produced_models["awq"], metadata=awq_meta,
            )
        if "smoothquant" in produced_models:
            from quantization.smoothquant import (
                serialize_smoothquant_metadata,
            )
            sq_meta = serialize_smoothquant_metadata(
                produced_models["smoothquant"],
            )
            self.ckpt.save_safe_module(
                "phase_1f_sq_model.pt",
                produced_models["smoothquant"],
                metadata=sq_meta,
            )
        if "smoothquant_gptq" in produced_models:
            from quantization.smoothquant import (
                serialize_smoothquant_metadata,
            )
            sg_meta = serialize_smoothquant_metadata(
                produced_models["smoothquant_gptq"],
            )
            self.ckpt.save_safe_module(
                "phase_1f_sq_gptq_model.pt",
                produced_models["smoothquant_gptq"], metadata=sg_meta,
            )
        self.ckpt.save_phase_json("phase_1f_gptq_smooth_awq", {
            "method_results": self.method_results,
        })

    # ==================================================================
    # Phase 2: Pareto Analysis
    # ==================================================================

    def phase_2_pareto(self) -> None:
        """Phase 2: Compute unified Pareto front and generate plots."""
        self.tracker.start_run("phase_2_pareto", {"phase": "2"})

        from visualization.pareto_analysis import ParetoAnalyzer
        from utils.common import compute_quantized_size_mb

        # Public Pareto pool is built ONLY from real, evaluated methods:
        # PTQ_best (phase 1c), QAT (phase 1e), and the phase 1f
        # variants. NSGA-II solutions remain available in
        # self.pareto_front (and the phase 1c checkpoint) for
        # reproducibility/debug, but they are deliberately excluded from
        # the public final Pareto so users see only methods that were
        # actually instantiated and evaluated end-to-end.
        all_solutions: List[ParetoSolution] = []

        # QAT entry — synthesise from the in-memory result.
        if self.qat_result:
            qat_acc = float(self.qat_result.get("final_val_acc", 0.0) or 0.0)
            qat_bw = self.best_config or {}
            qat_ebops = self._ebops_from_bitwidth(qat_bw)
            qat_red = (
                (self.fp32_ebops - qat_ebops) / max(self.fp32_ebops, 1) * 100
            )
            qat_lat = self.qat_result.get("latency_mean_ms")
            if qat_lat is None:
                qat_lat_dict = self.qat_result.get("latency") or {}
                qat_lat = qat_lat_dict.get("latency_mean_ms")
            qat_dom_bw = self._dominant_bitwidth(qat_bw) if qat_bw else 8
            qat_size_mb = (
                compute_quantized_size_mb(self.model, qat_bw)
                if qat_bw else 0.0
            )
            all_solutions.append(ParetoSolution(
                solution_id=f"QAT_INT{qat_dom_bw}",
                method="QAT",
                accuracy=qat_acc,
                accuracy_loss=self.fp32_acc - qat_acc,
                ebops=qat_ebops,
                ebops_reduction=qat_red,
                model_size_mb=qat_size_mb,
                latency_mean_ms=qat_lat,
                bitwidth_assignment=qat_bw,
                rank=1,
                crowding_distance=0.0,
                is_dominated=False,
            ))

        for res in self.method_results:
            ebops_red = (
                (self.fp32_ebops - res["ebops"])
                / max(self.fp32_ebops, 1) * 100
            )
            res_lat = None
            lat_dict = res.get("latency") or {}
            if "latency_mean_ms" in lat_dict:
                res_lat = lat_dict.get("latency_mean_ms")
            elif res.get("latency_ms") is not None:
                res_lat = res.get("latency_ms")
            display = res.get("display_name") or res.get("config_id", res["method"])
            sol = ParetoSolution(
                solution_id=display,
                method=res["method"],
                accuracy=res["accuracy"],
                accuracy_loss=self.fp32_acc - res["accuracy"],
                ebops=res["ebops"],
                ebops_reduction=ebops_red,
                model_size_mb=res.get("model_size_mb")
                    or compute_quantized_size_mb(
                        self.model, res.get("bitwidth_assignment", {}),
                    ),
                latency_mean_ms=res_lat,
                bitwidth_assignment=res.get("bitwidth_assignment", {}),
                rank=1,
                crowding_distance=0.0,
                is_dominated=False,
            )
            all_solutions.append(sol)

        # Hard guard: drop any leaked NSGA-internal entries
        # (e.g. ``nsga_gen12_r1``) before they reach plots/tables.
        public_candidates = [
            s for s in all_solutions
            if not str(s.get("solution_id", "")).startswith("nsga_")
        ]
        non_dominated = self._filter_non_dominated_solutions(public_candidates)
        non_dominated_ids = {
            str(s.get("solution_id", "")) for s in non_dominated
        }
        if len(non_dominated) < len(public_candidates):
            logger.info(
                "  Public Pareto filter: %d non-dominated of %d candidates",
                len(non_dominated), len(public_candidates),
            )

        # Mark every candidate as dominated or non-dominated so the
        # visualiser can show all solutions with highlighting.
        for s in public_candidates:
            sid = str(s.get("solution_id", ""))
            s["is_dominated"] = sid not in non_dominated_ids

        # Sort: non-dominated first (by accuracy desc), then dominated
        # (by accuracy desc).
        public_candidates.sort(
            key=lambda s: (s.get("is_dominated", False),
                           s.get("accuracy_loss", 0.0)),
        )

        merged_front = ParetoFront(
            solutions=public_candidates,
            generation=self.pareto_front.get("generation", 0),
            evaluations=self.pareto_front.get("evaluations", 0),
            convergence_reason="public_methods_only",
        )

        pareto_dir = self.output_dir / "pareto"
        pareto_dir.mkdir(parents=True, exist_ok=True)

        analyzer = ParetoAnalyzer(
            merged_front, self.fp32_acc, self.fp32_ebops,
            self.config.model_name,
        )
        self.pareto_analysis = analyzer.analyze(str(pareto_dir))

        hv = self.pareto_analysis["metrics"].get("hypervolume", 0)
        metrics_to_log = {
            "total_solutions": len(non_dominated),
            "public_candidates": len(public_candidates),
            "hypervolume": hv,
        }
        # Optional third Pareto axis: mean latency across solutions. Kept
        # as an auxiliary metric so existing 2D analysis is unchanged.
        if self.config.hyperparams.use_latency_in_pareto:
            latencies = [
                s.get("latency_mean_ms") for s in non_dominated
                if s.get("latency_mean_ms")
            ]
            if latencies:
                metrics_to_log["pareto_latency_mean_ms"] = (
                    sum(latencies) / len(latencies)
                )
        self.tracker.log_metrics(metrics_to_log)
        # Log plot artifacts
        plot_path = pareto_dir / "pareto_scatter.png"
        if plot_path.exists():
            self.tracker.log_artifact(str(plot_path), "plots")
        self.tracker.end_run()

        self.report_lines.append(
            f"[Phase 2] Pareto: {len(non_dominated)} non-dominated / "
            f"{len(public_candidates)} candidates, "
            f"HV={hv:.2f}"
        )
        self.results["hypervolume"] = hv

        # Checkpoint: pareto_analysis + hypervolume so resume can skip this.
        self.ckpt.save_phase_json("phase_2_pareto", {
            "pareto_analysis": self.pareto_analysis,
            "hypervolume": hv,
            "total_solutions": len(non_dominated),
            "public_candidates": len(public_candidates),
        })

    # ==================================================================
    # Phase 3: XAI Explainability
    # ==================================================================

    def phase_3_xai(self) -> None:
        """Phase 3: Generate Grad-CAM heatmaps + SHAP analysis."""
        self.tracker.start_run("phase_3_xai", {"phase": "3"})

        from xai.explainability import XAIGenerator

        # Get test images
        test_images, test_labels = [], []
        for batch in self.val_loader:
            test_images.append(batch[0])
            test_labels.append(batch[1])
            if len(test_images) * batch[0].shape[0] >= self.config.hyperparams.xai_num_images:
                break

        test_images = torch.cat(test_images, dim=0)[:self.config.hyperparams.xai_num_images]
        test_labels = torch.cat(test_labels, dim=0)[:self.config.hyperparams.xai_num_images]

        xai_dir = self.output_dir / "xai"
        xai_gen = XAIGenerator(self.config, device=self.device)

        # Build the public technique × sample matrix. Each row uses a
        # canonical, bitwidth-tagged label (e.g. ``GPTQ_INT8``,
        # ``AWQ_INT4``, ``SmoothQuant_INT8``) — never the raw family
        # name and never NSGA internal ``nsga_*`` IDs.
        quant_models: Dict[str, nn.Module] = {}

        # PTQ best-accuracy (display_name set in phase 1c, e.g. PTQ_INT8)
        ptq_acc_res = (
            self.results.get("ptq_best_acc_result")
            or self.results.get("ptq_best_result")
            or {}
        )
        ptq_acc_model = (
            self.results.get("ptq_best_acc_model")
            or self.results.get("ptq_model")
        )
        if isinstance(ptq_acc_model, nn.Module):
            ptq_label = ptq_acc_res.get("display_name") or "PTQ_INT8"
            quant_models[ptq_label] = ptq_acc_model

        # PTQ best-tradeoff — only if it differs from best-acc
        ptq_to_res = self.results.get("ptq_best_tradeoff_result") or {}
        ptq_to_model = self.results.get("ptq_best_tradeoff_model")
        if isinstance(ptq_to_model, nn.Module):
            to_label = ptq_to_res.get("display_name") or "PTQ_MIXED"
            if to_label not in quant_models:
                quant_models[to_label] = ptq_to_model

        # QAT fine-tuned (label includes the dominant bitwidth from
        # best_config so it's never just "QAT_warmstart").
        qat_m = self.qat_result.get("model") if self.qat_result else None
        if isinstance(qat_m, nn.Module):
            qat_dom = self._dominant_bitwidth(self.best_config or {}) \
                if self.best_config else 8
            quant_models[f"QAT_INT{qat_dom}"] = qat_m

        # Every bitwidth-tagged model produced in phase 1f. Falls back
        # to the legacy family-level slots when the new map is missing
        # (e.g. on a pre-upgrade resume).
        phase1f_models = self.results.get("phase_1f_models") or {}
        for label, m in phase1f_models.items():
            if isinstance(m, nn.Module):
                quant_models[label] = m
        if not phase1f_models:
            if "gptq_model" in self.results:
                quant_models["GPTQ_INT8"] = self.results["gptq_model"]
            if "awq_model" in self.results:
                quant_models["AWQ_INT4"] = self.results["awq_model"]
            if "sq_model" in self.results:
                quant_models["SmoothQuant_INT8"] = self.results["sq_model"]

        xai_result = xai_gen.run(
            fp32_model=self.model,
            quantized_models=quant_models,
            test_images=test_images,
            test_labels=test_labels,
            output_dir=str(xai_dir),
            class_names=self.class_names,
        )

        n_heatmaps = sum(
            len(p) for p in xai_result["grad_cam_paths"].values()
        )
        self.tracker.log_metrics({"xai_heatmaps": n_heatmaps})
        for mid, score in xai_result["consistency_scores"].items():
            self.tracker.log_metrics({f"xai_consistency_{mid}": score})

        if xai_dir.exists():
            self.tracker.log_artifact(str(xai_dir), "xai")

        self.tracker.end_run()

        self.report_lines.append(f"[Phase 3] XAI: {n_heatmaps} heatmaps")

        # Checkpoint: preserve artefact references and consistency scores.
        self.ckpt.save_phase_json("phase_3_xai", {
            "grad_cam_paths": xai_result.get("grad_cam_paths", {}),
            "shap_paths": xai_result.get("shap_paths", {}),
            "comparison_grid": xai_result.get("comparison_grid", ""),
            "consistency_scores": xai_result.get("consistency_scores", {}),
            "predictions": xai_result.get("predictions", {}),
            "n_heatmaps": n_heatmaps,
            "xai_dir": str(xai_dir),
        })

    # ==================================================================
    # Phase 4: MLflow Finalisation
    # ==================================================================

    def phase_4_mlflow(self) -> None:
        """Phase 4: Finalize MLflow tracking and log summary artifacts."""
        self.tracker.start_run("phase_4_summary", {"phase": "4"})

        # Log overall config
        self.tracker.log_params({
            "model_name": self.config.model_name,
            "dataset_name": self.config.dataset_name,
            "num_classes": self.config.num_classes,
            "bitwidths": str(self.config.supported_bitwidths),
        })

        # Log final summary metrics
        summary_metrics = {
            "fp32_accuracy": self.fp32_acc,
            "phases_completed": self.phases_passed,
        }
        if self.qat_result:
            summary_metrics["best_qat_acc"] = self.qat_result.get("final_val_acc", 0)
        if self.pareto_analysis:
            summary_metrics["hypervolume"] = self.pareto_analysis.get(
                "metrics", {}
            ).get("hypervolume", 0)

        # ── Wave 5 I3: Pareto front comparison summary ──
        # Aggregate the public method rows into the headline stats
        # MLflow's "Compare Runs" view consumes — best/median/worst on
        # every public objective. The same dict goes to disk as
        # ``pareto_summary.json`` so a downstream consumer (paper plot
        # generator, dashboard) does not have to re-walk the per-method
        # results to reconstruct it.
        pareto_summary = self._build_pareto_summary()
        for k, v in pareto_summary.items():
            if isinstance(v, (int, float)):
                summary_metrics[f"pareto_{k}"] = float(v)

        self.tracker.log_metrics(summary_metrics)

        # Persist the full summary (including non-scalar fields) as a
        # JSON artefact so reviewers can read it without spinning up
        # MLflow.
        summary_path = self.output_dir / "pareto_summary.json"
        try:
            with open(summary_path, "w") as f:
                json.dump(pareto_summary, f, indent=2, default=str)
            self.tracker.log_artifact(str(summary_path), "reports")
        except Exception as exc:
            logger.warning("Pareto summary export failed: %s", exc)

        # Save config as artifact
        config_path = self.output_dir / "config_used.json"
        self.config.to_json(config_path)
        self.tracker.log_artifact(str(config_path), "config")

        # Save summary report
        report_path = self.output_dir / "pipeline_report.txt"
        with open(report_path, "w") as f:
            f.write("NeuroQuant v2.0 Pipeline Report\n")
            f.write("=" * 50 + "\n\n")
            for line in self.report_lines:
                f.write(f"  {line}\n")
        self.tracker.log_artifact(str(report_path), "reports")

        self.tracker.end_run()

        self.report_lines.append("[Phase 4] MLflow: summary logged")

        # ── Deployment backends info ──
        try:
            from utils.deployment_export import available_backends
            backends = available_backends()
            self.results["deployment_backends"] = backends
            logger.info("  Deployment backends: %s", backends)
        except Exception:
            pass

        # ── Detection-only: TensorRT / OpenVINO export ──
        # For detection workloads, edge deployment typically targets
        # TensorRT (NVIDIA Jetson) or OpenVINO (Intel CPU/iGPU). Auto-
        # invoke both backends when available, using each method's
        # already-exported FP32/INT8 ONNX file as the source. Best
        # method per family is chosen by accuracy. Failures are logged
        # but never abort the pipeline — ONNX Runtime stays the
        # mandatory baseline. Classification runs skip this entirely
        # to keep the existing behaviour unchanged.
        task = getattr(self.config, "task", "classification")
        if task == "detection":
            self._export_detection_backends()

        # ── HTML Report Generation ──
        # Compile all pipeline artifacts into a self-contained HTML file.
        try:
            from visualization.report import generate_html_report
            report_html = generate_html_report(
                str(self.output_dir), self.config, self.results,
            )
            if report_html:
                self.tracker.log_artifact(report_html, "reports")
                self.report_lines.append("[Phase 4] HTML report generated")
        except Exception as exc:
            logger.warning("  HTML report generation skipped: %s", exc)

        # Checkpoint: completion marker so a second --resume run short-circuits
        # this phase instead of re-logging a duplicate MLflow summary.
        self.ckpt.save_phase_json("phase_4_mlflow", {
            "completed": True,
            "timestamp": time.time(),
            "phases_completed": self.phases_passed + 1,
        })

    # ==================================================================
    # Resume: per-phase state restoration from checkpoints
    # ==================================================================

    def _resume_phase(self, phase_name: str, display_name: str) -> None:
        """Restore the in-memory state that `phase_name` normally produces,
        so downstream phases can continue without re-executing this one."""
        restorer = getattr(self, f"_resume_{phase_name}", None)
        if restorer is None:
            logger.warning(
                "  [RESUME] No restorer for %s; skipping without state load.",
                phase_name,
            )
            return
        restorer()
        self.report_lines.append(f"[{display_name}] Restored from checkpoint")

    def _resume_phase_0_preparation(self) -> None:
        # Architecture is built from config, then weights are loaded from
        # the phase_0 .pth checkpoint — this preserves the trained baseline
        # even when --epochs defaults to 0 on the resume invocation.
        self.model = build_model(self.config)
        self.ckpt.load_phase_model("phase_0_preparation", self.model)
        self.model.to(self.device)

        # Data loaders are not serialised; rebuild from config and grab
        # the dataset class-name list for downstream XAI captions.
        (self.train_loader, self.search_loader, self.val_loader,
         self.test_loader, self.calib_loader,
         self.class_names) = build_data_loaders(self.config)

        data = self.ckpt.load_phase_json("phase_0_preparation")
        self.fp32_acc = float(data.get("fp32_acc", 0.0))
        self.fp32_top5 = float(data.get("fp32_top5", 0.0))
        self.fp32_ebops = float(data.get("fp32_ebops", 0.0))
        self.fp32_size_mb = float(data.get("fp32_size_mb", 0.0))
        self.fp32_latency = data.get("fp32_latency", {}) or {}
        self.fp32_onnx = data.get("fp32_onnx", {}) or {}

        self.results["fp32_acc"] = self.fp32_acc
        self.results["fp32_top5"] = self.fp32_top5

        # Recreate the FP32 row in the summary table for the final report.
        # Top-1 only on the public surface.
        fp32_onnx_lat = (self.fp32_onnx.get("onnx_latency") or {})
        self._add_summary_row(
            "FP32", self.fp32_acc,
            self.fp32_latency.get("latency_mean_ms", 0.0),
            self.fp32_latency.get("throughput_fps", 0.0),
            self.fp32_ebops, self.fp32_size_mb,
            onnx_size_mb=self.fp32_onnx.get("fp32_onnx_size_mb"),
            onnx_latency_ms=fp32_onnx_lat.get("latency_mean_ms"),
            onnx_throughput_fps=fp32_onnx_lat.get("throughput_fps"),
        )

    def _resume_phase_1a_hessian_clustering(self) -> None:
        data = self.ckpt.load_phase_json("phase_1a_hessian_clustering")
        self.hessian_diag = data.get("hessian_diag", {})
        self.cluster_result = data.get("cluster_result", {})
        self.cluster_assignments = self.cluster_result.get(
            "cluster_assignments", []
        )
        self.results["hessian_layers"] = len(self.hessian_diag)
        self.results["num_clusters"] = len(self.cluster_assignments)

    def _resume_phase_1b_fitcompress(self) -> None:
        self.fit_seed = self.ckpt.load_phase_json("phase_1b_fitcompress")

    def _resume_phase_1c_nsga_search(self) -> None:
        data = self.ckpt.load_phase_json("phase_1c_nsga_search")
        self.pareto_front = data.get("pareto_front", {}) or {}
        self.best_config = data.get("best_config", {}) or {}
        if not self.best_config and self.pareto_front.get("solutions"):
            self.best_config = self.pareto_front["solutions"][0].get(
                "bitwidth_assignment", {}
            )
        # Restore the AdaRound-specific mixed config; older checkpoints
        # didn't store this field, so fall back to best_config.
        self.adaround_config = data.get("adaround_config") or dict(
            self.best_config
        )
        self.results["pareto_solutions"] = len(
            self.pareto_front.get("solutions", [])
        )
        # Reranked PTQ picks + warmstart source so phases 1d/1e/2/3 see
        # the same selections that the original run used.
        ptq_acc = data.get("ptq_best_acc_result")
        ptq_to = data.get("ptq_best_tradeoff_result")
        if ptq_acc:
            self.results["ptq_best_acc_result"] = ptq_acc
            self.results["ptq_best_result"] = ptq_acc
        if ptq_to:
            self.results["ptq_best_tradeoff_result"] = ptq_to
        if data.get("qat_warmstart_source"):
            self.results["qat_warmstart_source"] = data["qat_warmstart_source"]
        if data.get("qat_warmstart_id"):
            self.results["qat_warmstart_id"] = data["qat_warmstart_id"]
        # Re-add the PTQ rows to the public summary table on resume so
        # the final report mirrors the original run.
        for res in (ptq_acc, ptq_to):
            if not res:
                continue
            lat = res.get("latency") or {}
            onnx_lat = res.get("onnx_latency") or {}
            self._add_summary_row(
                res.get("display_name") or res.get("config_id", "PTQ"),
                float(res.get("accuracy", 0.0)),
                lat.get("latency_mean_ms", 0.0),
                lat.get("throughput_fps", 0.0),
                float(res.get("ebops", 0.0)),
                float(res.get("theoretical_size_mb",
                              res.get("model_size_mb", 0.0))),
                onnx_size_mb=res.get("onnx_size_mb"),
                onnx_latency_ms=onnx_lat.get("latency_mean_ms"),
                onnx_throughput_fps=onnx_lat.get("throughput_fps"),
            )

    def _resume_phase_1d_adaround(self) -> None:
        # Build a fresh architecture and load the adaround weights into it so
        # phase 1e (QAT warmstart) has a concrete nn.Module to start from.
        adaround_model = build_model(self.config)
        metadata = self.ckpt.load_phase_model(
            "phase_1d_adaround", adaround_model,
        )
        adaround_model.to(self.device)

        # Fall back to companion JSON if the .pth metadata envelope is empty.
        if not metadata and self.ckpt.phase_exists("phase_1d_adaround"):
            try:
                metadata = self.ckpt.load_phase_json("phase_1d_adaround")
            except FileNotFoundError:
                metadata = {}

        self.adaround_result = {
            "model": adaround_model,
            "mse_before": float(metadata.get("mse_before", 0.0)),
            "mse_after": float(metadata.get("mse_after", 0.0)),
            "mse_reduction": float(metadata.get("mse_reduction", 0.0)),
            "time_seconds": float(metadata.get("time_seconds", 0.0)),
            "alpha_stats": metadata.get("alpha_stats", {}),
        }

    def _resume_phase_1e_qat(self) -> None:
        qat_model = build_model(self.config)
        metadata: Dict[str, Any] = {}
        if self.ckpt.phase_exists("phase_1e_qat"):
            # Prefer .pth if present (contains the fine-tuned weights).
            try:
                metadata = self.ckpt.load_phase_model(
                    "phase_1e_qat", qat_model,
                )
                qat_model.to(self.device)
            except FileNotFoundError:
                qat_model = None
            if not metadata:
                try:
                    metadata = self.ckpt.load_phase_json("phase_1e_qat")
                except FileNotFoundError:
                    metadata = {}

        self.qat_result = {
            "model": qat_model,
            "final_val_acc": float(metadata.get("final_val_acc", 0.0)),
            "test_top1": metadata.get("test_top1"),
            "best_epoch": int(metadata.get("best_epoch", 0)),
            "train_accuracy": metadata.get("train_accuracy", []),
            "val_accuracy": metadata.get("val_accuracy", []),
            "train_loss": metadata.get("train_loss", []),
            "time_seconds": float(metadata.get("time_seconds", 0.0)),
        }
        self.results["qat_acc"] = self.qat_result["final_val_acc"]
        if self.qat_result.get("test_top1") is not None:
            self.results["qat_test_top1"] = float(self.qat_result["test_top1"])
        # Restore warmstart provenance so post-mortem reads from
        # checkpoint match the original run.
        if metadata.get("qat_warmstart_source"):
            self.results["qat_warmstart_source"] = metadata["qat_warmstart_source"]
        if metadata.get("qat_warmstart_id"):
            self.results["qat_warmstart_id"] = metadata["qat_warmstart_id"]

        # Re-add QAT to the public summary table on resume so the
        # rebuilt ``pareto_summary.json`` matches the original run.
        from utils.common import compute_quantized_size_mb
        qat_bw = self.best_config or {}
        qat_dom_bw = (
            self._dominant_bitwidth(qat_bw) if qat_bw
            else int(self.config.hyperparams.qat_act_bitwidth)
        )
        qat_size_mb = (
            compute_quantized_size_mb(self.model, qat_bw) if qat_bw else 0.0
        )
        qat_ebops = self._ebops_from_bitwidth(qat_bw) if qat_bw else 0.0
        qat_headline_acc = (
            float(self.qat_result["test_top1"])
            if self.qat_result.get("test_top1") is not None
            else float(self.qat_result.get("final_val_acc", 0.0) or 0.0)
        )
        self._add_summary_row(
            f"QAT_INT{qat_dom_bw}",
            qat_headline_acc,
            0.0,  # latency not persisted in qat_meta
            0.0,
            qat_ebops,
            qat_size_mb,
        )

    def _resume_phase_1f_gptq_smooth_awq(self) -> None:
        data = self.ckpt.load_phase_json("phase_1f_gptq_smooth_awq")
        self.method_results = data.get("method_results", []) or []

        # Rematerialise quantized models used by phase 3 (XAI). Each is the
        # same architecture as the baseline with a modified state_dict.
        def _restore(name: str) -> Optional[nn.Module]:
            fname = f"phase_1f_{name}_model.pth"
            if not self.ckpt.file_exists(fname):
                logger.warning(
                    "  [RESUME] %s missing; phase 3 will skip %s.",
                    fname, name.upper(),
                )
                return None
            m = build_model(self.config)
            self.ckpt.load_named_model(fname, m)
            m.to(self.device)
            return m

        gptq_model = _restore("gptq")
        if gptq_model is not None:
            self.results["gptq_model"] = gptq_model

        # AWQ: production AWQ now ships with ``_AWQInputScale`` wrappers
        # so it follows the same safe-module + manifest pattern as
        # SmoothQuant. Fall back to the legacy state_dict path only for
        # checkpoints predating the F2 audit.
        if self.ckpt.file_exists("phase_1f_awq_model.pt"):
            from quantization.awq import restore_awq_wrappers
            awq_model = build_model(self.config)
            self.ckpt.load_safe_module(
                "phase_1f_awq_model.pt",
                awq_model, rebuild=restore_awq_wrappers,
            )
            self.results["awq_model"] = awq_model.to(self.device)
        else:
            awq_model = _restore("awq")
            if awq_model is not None:
                self.results["awq_model"] = awq_model

        # SmoothQuant: rebuild the FP32 architecture, replay the
        # ``_SmoothInputScale`` wrappers from the JSON manifest captured at
        # save time, then load the state_dict in safe (``weights_only=True``)
        # mode. No pickle code path is exercised. Fall back to the legacy
        # state_dict path only for checkpoints predating the security fix.
        if self.ckpt.file_exists("phase_1f_sq_model.pt"):
            from quantization.smoothquant import restore_smoothquant_wrappers
            sq_model = build_model(self.config)
            self.ckpt.load_safe_module(
                "phase_1f_sq_model.pt",
                sq_model,
                rebuild=restore_smoothquant_wrappers,
            )
            self.results["sq_model"] = sq_model.to(self.device)
        else:
            sq_model = _restore("sq")
            if sq_model is not None:
                self.results["sq_model"] = sq_model

        # Combined SmoothQuant→GPTQ (F4): same wrapper shape as
        # SmoothQuant — restore via the same helper, no special-casing
        # for the GPTQ rounding step (it's already in the state_dict).
        if self.ckpt.file_exists("phase_1f_sq_gptq_model.pt"):
            from quantization.smoothquant import restore_smoothquant_wrappers
            sg_model = build_model(self.config)
            self.ckpt.load_safe_module(
                "phase_1f_sq_gptq_model.pt",
                sg_model, rebuild=restore_smoothquant_wrappers,
            )
            self.results["sq_gptq_model"] = sg_model.to(self.device)

        # Rebuild the summary-table rows produced on the original run.
        # Use the canonical display_name (no METHOD_METHOD duplication)
        # and Top-1 only.
        for res in self.method_results:
            lat = res.get("latency") or {}
            display = (
                res.get("display_name")
                or res.get("config_id", res.get("method", "?"))
            )
            onnx_lat = res.get("onnx_latency") or {}
            self._add_summary_row(
                display,
                res.get("accuracy", 0.0),
                lat.get("latency_mean_ms", 0.0),
                lat.get("throughput_fps", 0.0),
                res.get("ebops", 0.0),
                res.get("theoretical_size_mb", res.get("model_size_mb", 0.0)),
                onnx_size_mb=res.get("onnx_size_mb"),
                onnx_latency_ms=onnx_lat.get("latency_mean_ms"),
                onnx_throughput_fps=onnx_lat.get("throughput_fps"),
            )

    def _resume_phase_2_pareto(self) -> None:
        data = self.ckpt.load_phase_json("phase_2_pareto")
        self.pareto_analysis = data.get("pareto_analysis", {}) or {}
        hv = float(data.get("hypervolume", 0.0))
        self.results["hypervolume"] = hv

    def _resume_phase_3_xai(self) -> None:
        # Phase 3 writes artefacts to disk; restoring its JSON summary is
        # enough for the final report and for phase 4 to include it.
        data = self.ckpt.load_phase_json("phase_3_xai")
        self.results["xai_summary"] = {
            "consistency_scores": data.get("consistency_scores", {}),
            "n_heatmaps": data.get("n_heatmaps", 0),
            "xai_dir": data.get("xai_dir", ""),
        }

    def _resume_phase_4_mlflow(self) -> None:
        # Nothing to reinstate in memory — MLflow already has the logged run.
        self.ckpt.load_phase_json("phase_4_mlflow")

    # ==================================================================
    # Training Helper
    # ==================================================================

    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> float:
        """Train the model and return best validation top-1 accuracy."""
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_acc = correct / max(total, 1) * 100
            train_loss = running_loss / max(total, 1)

            val_dict = evaluate_model(model, val_loader, self.device)
            val_acc = val_dict["top1"]
            best_acc = max(best_acc, val_acc)
            scheduler.step()

            logger.info(
                "  Epoch %d/%d: loss=%.4f, train_acc=%.2f%%, val_acc=%.2f%%",
                epoch, epochs, train_loss, train_acc, val_acc,
            )

        return best_acc

    # ==================================================================
    # Report & Utilities
    # ==================================================================

    def _evaluate_top1(
        self,
        model: nn.Module,
        loader: Optional[DataLoader],
    ) -> Optional[float]:
        """Compute Top-1 accuracy on ``loader`` if available, else None.

        Cheap wrapper used by ``_attach_split_metrics`` to populate the
        ``val_top1`` / ``test_top1`` / ``search_top1`` fields on every
        method result. Keeping the call sites explicit makes it obvious
        which loader produced which number, which is the whole point of
        the A1/A2 split-isolation contract.
        """
        if loader is None or model is None:
            return None
        try:
            return float(evaluate_model(model, loader, self.device)["top1"])
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("  [eval] top-1 on loader failed: %s", exc)
            return None

    def _attach_split_metrics(
        self,
        result: Dict[str, Any],
        model: nn.Module,
    ) -> Dict[str, Any]:
        """Populate ``val_top1`` + ``test_top1`` on a method result and
        promote ``test_top1`` to the public ``accuracy`` field.

        Always recomputes ``val_top1`` and ``test_top1`` from the model
        directly so the headline is independent of whatever loader the
        upstream evaluator happened to use (rerank uses ``search_loader``
        for selection; phase 1f's ``BaseQuantizer.evaluate`` uses
        ``val_loader``). After this call:

        * ``result["search_top1"]`` — kept as set by the caller (or
          ``None``); used internally for selection only.
        * ``result["val_top1"]``    — diagnostic; QAT early-stop signal.
        * ``result["test_top1"]``   — public headline.
        * ``result["accuracy"]``    — alias of ``test_top1``.

        If ``test_loader`` is unavailable (legacy resume), the function
        leaves the prior ``accuracy`` value untouched and writes only
        the val number — strictly better than nothing on those paths.
        """
        val_top1 = self._evaluate_top1(model, self.val_loader)
        if val_top1 is not None:
            result["val_top1"] = val_top1

        test_top1 = self._evaluate_top1(model, self.test_loader)
        if test_top1 is not None:
            result["test_top1"] = test_top1
            # Public contract: the headline accuracy is the test-set
            # Top-1. Never overwrite with val/search.
            result["accuracy"] = test_top1
        elif val_top1 is not None and "accuracy" not in result:
            # Last-resort fallback: no test loader → val becomes headline.
            result["accuracy"] = val_top1
        return result

    @staticmethod
    def _select_rerank_candidates(
        nsga_solutions: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Pick up to ``top_k`` NSGA candidates for real-PTQ rerank.

        Sort by ``accuracy_loss`` first (best proxy accuracy first), then
        by ``model_size_mb`` (smallest first) as a tiebreaker. This gives
        the rerank pool a mix of high-accuracy and high-compression
        candidates so the downstream tradeoff pick has options to
        choose from.
        """
        if not nsga_solutions or top_k < 1:
            return []
        seen_keys: set = set()
        deduped: List[Dict[str, Any]] = []
        for s in nsga_solutions:
            ba = s.get("bitwidth_assignment") or {}
            # Hashable key from the assignment so identical configs (e.g.
            # the same gene reached from different parents) collapse.
            key = tuple(sorted((str(k), int(v)) for k, v in ba.items()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(s)
        ranked = sorted(
            deduped,
            key=lambda s: (
                float(s.get("accuracy_loss", float("inf"))),
                float(s.get("model_size_mb", float("inf"))),
            ),
        )
        return ranked[:max(1, int(top_k))]

    def _materialize_and_rerank_ptq(
        self,
        candidates: List[Dict[str, Any]],
        hp: Any,
    ) -> tuple:
        """Run real PTQ + bitwidth-aware calibration on each candidate
        and return ``(best_acc_model, best_acc_res, best_tradeoff_model,
        best_tradeoff_res)``.

        ``best_acc_*`` is the candidate with the highest real Top-1.
        ``best_tradeoff_*`` is the smallest-size candidate within
        ``hp.ptq_tradeoff_max_acc_drop`` of the best Top-1 (knee-like
        fallback if none satisfies the cap).
        """
        from quantization.ptq import PTQQuantizer
        from utils.common import compute_quantized_size_mb

        # Selection loader for the rerank decision (NSGA-fitness slice).
        # The public ``test_loader`` is *not* read here so it stays
        # untouched until the headline-evaluation pass below.
        select_loader = self.search_loader or self.val_loader

        materialized: List[Dict[str, Any]] = []
        models: List[nn.Module] = []
        for cand in candidates:
            assignment = dict(cand.get("bitwidth_assignment") or {})
            if not assignment:
                continue
            ptq = PTQQuantizer(self.model, self.config)
            # Bitwidth-aware calibration: each layer gets a threshold
            # computed at its OWN target bitwidth (INT4 vs INT8).
            ptq.calibrate_with_assignment(
                self.calib_loader,
                assignment,
                num_batches=hp.calibration_batches,
            )
            q_model = ptq.quantize_with_config(assignment)
            dom_bw = self._dominant_bitwidth(assignment)
            # Evaluate on the SEARCH slice to drive the selection. The
            # value lands in ``search_top1``; the public ``accuracy``
            # field is overwritten with the test number after a winner
            # is picked.
            res = ptq.evaluate(q_model, select_loader, bitwidth=dom_bw)
            res["search_top1"] = float(res.get("accuracy", 0.0))

            # Mixed assignments get an explicit "MIXED" tag so they can
            # be distinguished from the uniform-INT8/INT4 picks in the
            # public report and the XAI matrix.
            tag = "MIXED" if self._is_mixed_bitwidth_assignment(assignment) \
                else f"INT{dom_bw}"
            display = f"PTQ_{tag}"
            res["config_id"] = display
            res["display_name"] = display
            res["bitwidth"] = int(dom_bw)
            res["bitwidth_assignment"] = assignment
            res["model_size_mb"] = compute_quantized_size_mb(self.model, assignment)
            materialized.append(res)
            models.append(q_model)
            logger.info(
                "  PTQ rerank candidate %s: search_top1=%.2f%%, size=%.2f MiB",
                display, float(res["search_top1"]),
                float(res["model_size_mb"]),
            )

        if not materialized:
            return None, None, None, None

        # Best by SEARCH Top-1 (held-out, not val/test).
        best_acc_idx = max(
            range(len(materialized)),
            key=lambda i: float(materialized[i]["search_top1"]),
        )

        # Tradeoff: most compressed among those within the accuracy cap;
        # fallback to smallest size overall when none satisfies the cap.
        cap = float(getattr(hp, "ptq_tradeoff_max_acc_drop", 1.0))
        ref_acc = float(materialized[best_acc_idx]["search_top1"])
        within_cap = [
            i for i, r in enumerate(materialized)
            if (ref_acc - float(r["search_top1"])) <= cap
        ]
        if within_cap:
            best_tradeoff_idx = min(
                within_cap,
                key=lambda i: float(materialized[i]["model_size_mb"]),
            )
        else:
            best_tradeoff_idx = min(
                range(len(materialized)),
                key=lambda i: float(materialized[i]["model_size_mb"]),
            )

        # Disambiguate display names when best_acc and best_tradeoff
        # land on the same config (the mixed/INT8 tag would collide).
        same_pick = (best_acc_idx == best_tradeoff_idx)

        if not same_pick and (
            materialized[best_acc_idx]["display_name"]
            == materialized[best_tradeoff_idx]["display_name"]
        ):
            # When tags collide but configs differ, append a discriminator.
            tradeoff_dom = self._dominant_bitwidth(
                materialized[best_tradeoff_idx]["bitwidth_assignment"]
            )
            tradeoff_tag = (
                "MIXED" if self._is_mixed_bitwidth_assignment(
                    materialized[best_tradeoff_idx]["bitwidth_assignment"]
                ) else f"INT{tradeoff_dom}"
            )
            new_name = f"PTQ_{tradeoff_tag}_tradeoff"
            materialized[best_tradeoff_idx]["display_name"] = new_name
            materialized[best_tradeoff_idx]["config_id"] = new_name

        # Promote the winners' headline accuracy to the test-set Top-1
        # via ``_attach_split_metrics``. ``search_top1`` is preserved on
        # the result (used internally) and ``val_top1`` is filled from
        # the val_loader. The public ``accuracy`` field becomes the
        # test number — that is the deployment headline.
        self._attach_split_metrics(
            materialized[best_acc_idx], models[best_acc_idx],
        )
        if not same_pick:
            self._attach_split_metrics(
                materialized[best_tradeoff_idx], models[best_tradeoff_idx],
            )

        # ── ONNX export for the PTQ rerank winners ──
        # Previously skipped, leaving ``onnx_size_mb`` / ``onnx_latency``
        # null in pareto_summary.json for the two highest-accuracy
        # configurations. Re-using the centralised helper so PTQ /
        # phase-1f / QAT all share the same J1+J2+J3 contract.
        best_bw_acc = self._dominant_bitwidth(
            materialized[best_acc_idx].get("bitwidth_assignment", {})
        )
        self._export_method_to_onnx(
            materialized[best_acc_idx], models[best_acc_idx], best_bw_acc,
            materialized[best_acc_idx].get("bitwidth_assignment"),
        )
        if not same_pick:
            best_bw_to = self._dominant_bitwidth(
                materialized[best_tradeoff_idx].get("bitwidth_assignment", {})
            )
            self._export_method_to_onnx(
                materialized[best_tradeoff_idx],
                models[best_tradeoff_idx],
                best_bw_to,
                materialized[best_tradeoff_idx].get("bitwidth_assignment"),
            )

        return (
            models[best_acc_idx], materialized[best_acc_idx],
            models[best_tradeoff_idx], materialized[best_tradeoff_idx],
        )

    def _export_method_to_onnx(
        self,
        res: Dict[str, Any],
        q_model: nn.Module,
        bitwidth: int,
        bw_assignment: Optional[Dict[str, int]] = None,
    ) -> None:
        """Run the J1+J2+J3 ONNX export contract on ``q_model`` and
        update ``res`` with the on-disk size + ORT latency.

        Centralised so PTQ rerank (phase 1c), QAT (phase 1e), and the
        per-method block in phase 1f all produce the same deployment
        fields. Without this, the headline ``pareto_summary.json`` had
        ``onnx_size_mb=null`` for the two highest-accuracy methods
        (PTQ_MIXED and QAT_INT8), which broke the "deployment-ready"
        contract.

        Failures are logged as warnings and never abort the caller —
        the synthetic ``theoretical_size_mb`` remains the public size
        when the export can't run (e.g. onnxruntime missing).
        """
        hp = self.config.hyperparams
        if not bool(getattr(hp, "onnx_export_enabled", True)):
            return
        try:
            from utils.onnx_export import (
                export_quantize_and_benchmark, is_onnx_available,
            )
        except Exception:
            return
        if not is_onnx_available():
            return

        onnx_dir = self.output_dir / "onnx"
        try:
            info = export_quantize_and_benchmark(
                q_model,
                self.config.input_shape,
                str(onnx_dir),
                name=f"{res['display_name']}",
                calibration_loader=self.calib_loader,
                num_batches=min(8, hp.calibration_batches),
                do_int8=(int(bitwidth) <= 8),
                batch_size=hp.latency_batch_size,
                warmup_runs=hp.latency_warmup_runs,
                measure_runs=hp.latency_measure_runs,
            )
        except Exception as exc:
            logger.warning(
                "    %s ONNX export failed: %s — falling back to "
                "theoretical_size_mb. Real on-disk size + ORT "
                "latency unavailable for this method.",
                res["display_name"], exc,
            )
            return

        if info.get("int8_onnx_size_mb") is not None:
            res["onnx_size_mb"] = info["int8_onnx_size_mb"]
            res["onnx_path"] = info["int8_onnx_path"]
            res["model_size_mb"] = info["int8_onnx_size_mb"]
        elif info.get("fp32_onnx_size_mb") is not None:
            res["onnx_size_mb"] = info["fp32_onnx_size_mb"]
            res["onnx_path"] = info["fp32_onnx_path"]
            res["model_size_mb"] = info["fp32_onnx_size_mb"]
        if info.get("onnx_latency"):
            res["onnx_latency"] = info["onnx_latency"]
            res["latency_ms"] = info["onnx_latency"]["latency_mean_ms"]
            res["latency"] = info["onnx_latency"]

        # INT4 packing diagnostic when any weight is below INT8.
        try:
            assignment = bw_assignment or res.get("bitwidth_assignment") or {}
            if int(bitwidth) <= 4 or any(
                int(v) == 4 for v in assignment.values()
            ):
                from utils.onnx_export import estimate_int4_packed_size_mb
                packing = estimate_int4_packed_size_mb(q_model, assignment)
                res["int4_packing"] = packing
        except Exception:
            pass

    @staticmethod
    def _filter_non_dominated_solutions(
        solutions: List[ParetoSolution],
    ) -> List[ParetoSolution]:
        """Return only non-dominated solutions.

        Uses the same set of objectives the search consumed:
          * 2-obj fallback: ``(accuracy_loss, ebops)``
          * 3-obj when latency is available: ``(accuracy_loss, ebops,
            latency_mean_ms)`` — kicks in when *every* candidate
            carries an ORT latency. Falls back to 2-obj otherwise so
            we don't accidentally tag everything as non-dominated when
            a single solution lacks a latency reading.

        Standard Pareto definition (minimisation): X dominates Y when
        ``X[k] ≤ Y[k]`` on all objectives and ``X[k] < Y[k]`` on at
        least one.
        """
        if not solutions:
            return []

        # Decide on the objective set once, up-front.
        all_have_latency = all(
            s.get("latency_mean_ms") is not None for s in solutions
        )
        if all_have_latency:
            obj_keys: Tuple[str, ...] = ("accuracy_loss", "ebops",
                                         "latency_mean_ms")
        else:
            obj_keys = ("accuracy_loss", "ebops")

        def _objs(sol: ParetoSolution) -> Tuple[float, ...]:
            return tuple(
                float(sol.get(k, float("inf"))) for k in obj_keys
            )

        kept: List[ParetoSolution] = []
        for i, sol_i in enumerate(solutions):
            oi = _objs(sol_i)
            dominated = False
            for j, sol_j in enumerate(solutions):
                if i == j:
                    continue
                oj = _objs(sol_j)
                # j dominates i iff j ≤ i on every axis AND strictly
                # better on at least one.
                if all(b <= a for a, b in zip(oi, oj)) and any(
                    b < a for a, b in zip(oi, oj)
                ):
                    dominated = True
                    break
            if not dominated:
                kept.append(sol_i)

        # Stable user-facing order: best accuracy first.
        return sorted(kept, key=lambda s: float(s.get("accuracy_loss", 0.0)))

    def _ebops_from_bitwidth(self, bitwidth_assignment: Dict[str, int]) -> float:
        """Compute EBops ≈ sum(numel * bitwidth) / 8 from an assignment.

        Parameters not present in the assignment fall back to FP32 (32 bits).
        Used for synthesising Pareto points for QAT/PTQ on phase 2 merge.
        """
        if self.model is None:
            return 0.0
        total_bits = 0.0
        for name, p in self.model.named_parameters():
            bw = int(bitwidth_assignment.get(name, 32))
            total_bits += p.numel() * bw
        return total_bits / 8.0

    @staticmethod
    def _dominant_bitwidth(bitwidth_assignment: Dict[str, int]) -> int:
        """Return the mode bitwidth of an assignment (for reporting/ebops)."""
        if not bitwidth_assignment:
            return 32
        counts: Dict[int, int] = {}
        for bw in bitwidth_assignment.values():
            counts[int(bw)] = counts.get(int(bw), 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _is_mixed_bitwidth_assignment(bitwidth_assignment: Dict[str, int]) -> bool:
        """True when both INT4 and INT8 are present in the assignment."""
        if not bitwidth_assignment:
            return False
        bws = {int(bw) for bw in bitwidth_assignment.values() if int(bw) < 32}
        return 4 in bws and 8 in bws

    def _add_summary_row(
        self, method: str, top1: float,
        latency_ms: float, throughput: float,
        ebops: float, size_mb: float,
        *,
        onnx_size_mb: Optional[float] = None,
        onnx_latency_ms: Optional[float] = None,
        onnx_throughput_fps: Optional[float] = None,
    ) -> None:
        """Add or update a row in the public summary table.

        Idempotent by ``method`` key: if a row with the same method
        label already exists it is replaced in place rather than
        appended. This keeps resume runs deterministic — when both
        ``_resume_phase_1c_nsga_search`` and
        ``_resume_phase_1f_gptq_smooth_awq`` reach the table with the
        same PTQ entry, the row appears exactly once.

        Top-5 is intentionally NOT a column on the public report — only
        Top-1 is surfaced. Top-5 may still be computed internally but is
        excluded from the user-facing result contract.

        Wave 5 G1 additions: when ``onnx_*`` fields are supplied (typical
        for any quantized method after Wave 4 wired ONNX export into
        phase 1f), they are stored on the row so the public table can
        show real on-disk size and ORT latency next to the synthetic
        ``size_mb`` and PyTorch ``latency_ms`` numbers. Rows without
        ONNX numbers (e.g. FP32 baseline before ONNX export ran)
        leave those fields ``None``; the printer falls back to "-".
        """
        if not hasattr(self, '_summary_rows'):
            self._summary_rows = []
        row = {
            "method": method, "top1": top1,
            "latency_ms": latency_ms, "throughput": throughput,
            "ebops": ebops, "size_mb": size_mb,
            "onnx_size_mb": onnx_size_mb,
            "onnx_latency_ms": onnx_latency_ms,
            "onnx_throughput_fps": onnx_throughput_fps,
        }
        for i, existing in enumerate(self._summary_rows):
            if existing.get("method") == method:
                self._summary_rows[i] = row
                return
        self._summary_rows.append(row)

    def _print_report(self, elapsed: float, phases_total: int) -> None:
        """Print the final pipeline report with metrics summary table."""
        print("\n")
        print("=" * 90)
        print("  NeuroQuant v2.0 Pipeline Report")
        print("=" * 90)
        print(f"  Model:          {self.config.model_name}")
        print(f"  Dataset:        {self.config.dataset_name}")
        print(f"  Device:         {self.device}")
        print(f"  Total Runtime:  {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        print(f"  Phases Passed:  {self.phases_passed}/{phases_total}")
        # Public contract: top-1 is the only surfaced accuracy metric.
        print("  Primary Acc:    top1")
        print()
        for line in self.report_lines:
            marker = "[OK]" if not line.startswith("[ERROR]") else "[FAIL]"
            print(f"  {marker} {line}")

        # Public summary table — Top-1 only, Size(MiB) is a first-class
        # column so the accuracy/size/EBops trade-off is visible.
        # Wave 5 G1: when any row has ONNX numbers, three deployment
        # columns ``ONNX MiB``, ``ORT(ms)``, ``ORT FPS`` are appended
        # to the table. Rows without ONNX numbers print "-" so the
        # table stays rectangular even when ONNX is disabled.
        rows = getattr(self, '_summary_rows', [])
        if rows:
            has_onnx = any(
                r.get("onnx_size_mb") is not None
                or r.get("onnx_latency_ms") is not None
                for r in rows
            )
            print()
            base_hdr = (
                f"  {'Method':<22} {'Top-1':>7} {'Lat(ms)':>9} "
                f"{'FPS':>9} {'EBops':>12} {'Size(MiB)':>10}"
            )
            if has_onnx:
                hdr = (
                    base_hdr
                    + f" {'ONNX MiB':>9} {'ORT(ms)':>8} {'ORT FPS':>9}"
                )
            else:
                hdr = base_hdr
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for r in rows:
                base_line = (
                    f"  {r['method']:<22} {r['top1']:>6.2f}% "
                    f"{r['latency_ms']:>8.2f} {r['throughput']:>8.1f} "
                    f"{r['ebops']:>12.0f} {r['size_mb']:>9.2f}"
                )
                if has_onnx:
                    onnx_size = r.get("onnx_size_mb")
                    onnx_lat = r.get("onnx_latency_ms")
                    onnx_fps = r.get("onnx_throughput_fps")
                    base_line += (
                        f" {self._fmt_or_dash(onnx_size, '>9.2f')}"
                        f" {self._fmt_or_dash(onnx_lat, '>8.2f')}"
                        f" {self._fmt_or_dash(onnx_fps, '>9.1f')}"
                    )
                print(base_line)

            # ── G4: deployment fidelity section ──
            # Spell out theoretical vs on-disk size and PyTorch vs ORT
            # latency so the report makes deployment-equivalence claims
            # explicit. Only printed when ONNX export actually produced
            # numbers; degraded gracefully otherwise.
            self._print_deployment_fidelity_section(rows)

        # Hardware metrics
        hw = getattr(self, 'hardware_metrics', {})
        if hw and hw.get('source', 'not_provided') != 'not_provided':
            print()
            print("  Hardware Synthesis Metrics:")
            for key in ('dsp', 'lut', 'ff', 'fmax_mhz', 'ii', 'cycle_latency'):
                val = hw.get(key)
                if val is not None:
                    print(f"    {key:<16} {val}")
            print(f"    {'source':<16} {hw.get('source', 'n/a')}")

        print()
        if self.phases_passed == phases_total:
            status = "ALL PHASES COMPLETE"
        else:
            status = f"INCOMPLETE ({self.phases_passed}/{phases_total})"
        print(f"  Status: {status}")
        print("=" * 90)

    def _export_detection_backends(self) -> None:
        """Export the best per-method ONNX file to TensorRT and OpenVINO.

        Detection workloads commonly target edge accelerators that need
        a vendor-specific runtime (TRT for Jetson, OpenVINO for Intel
        CPU/iGPU). For each quantization method that produced a usable
        ``.onnx`` file in Phase 1f, attempt a TRT engine build (INT8
        when calibration data is loadable, otherwise FP32) and an
        OpenVINO IR conversion. Results are stashed under
        ``self.results['deployment_exports']`` and logged as MLflow
        artifacts. Missing backends or per-method failures are warnings,
        never errors — the public ONNX path is the mandatory contract.
        """
        try:
            from utils.deployment_export import (
                available_backends, export_tensorrt, export_openvino,
            )
        except Exception as exc:
            logger.warning("  deployment_export import failed: %s", exc)
            return

        backends = set(available_backends())
        if not (backends & {"tensorrt", "openvino"}):
            logger.info(
                "  Detection deployment skipped — neither TensorRT nor "
                "OpenVINO is installed."
            )
            return

        # Pick the candidate ONNX file per method family by accuracy.
        # Each method already exposes ``onnx_path`` (set in Phase 1f).
        best_per_family: Dict[str, Dict[str, Any]] = {}
        for res in self.method_results:
            onnx_path = res.get("onnx_path")
            if not onnx_path or not Path(onnx_path).exists():
                continue
            family = str(res.get("display_name", "")).split("_")[0] or "method"
            cur = best_per_family.get(family)
            if cur is None or float(res.get("accuracy", 0.0)) > float(
                cur.get("accuracy", 0.0)
            ):
                best_per_family[family] = res

        if not best_per_family:
            logger.info(
                "  Detection deployment skipped — no ONNX artefacts "
                "available to convert."
            )
            return

        export_dir = self.output_dir / "deployment"
        export_dir.mkdir(parents=True, exist_ok=True)

        exports: List[Dict[str, Any]] = []
        for family, res in best_per_family.items():
            onnx_path = res["onnx_path"]
            tag = res.get("display_name", family)

            if "tensorrt" in backends:
                try:
                    trt_info = export_tensorrt(
                        onnx_path,
                        str(export_dir / f"{tag}.trt"),
                        tuple(self.config.input_shape),
                        batch_size=self.config.hyperparams.latency_batch_size,
                        precision="fp32",  # INT8 needs a calibration array
                    )
                    if trt_info:
                        trt_info["method"] = tag
                        exports.append(trt_info)
                        self.tracker.log_artifact(
                            trt_info["engine_path"], "deployment",
                        )
                except Exception as exc:
                    logger.warning(
                        "  TensorRT export failed for %s: %s", tag, exc,
                    )

            if "openvino" in backends:
                try:
                    ov_info = export_openvino(
                        onnx_path, str(export_dir),
                        model_name=tag,
                    )
                    if ov_info:
                        ov_info["method"] = tag
                        exports.append(ov_info)
                        self.tracker.log_artifact(
                            ov_info["xml_path"], "deployment",
                        )
                        if Path(ov_info["bin_path"]).exists():
                            self.tracker.log_artifact(
                                ov_info["bin_path"], "deployment",
                            )
                except Exception as exc:
                    logger.warning(
                        "  OpenVINO export failed for %s: %s", tag, exc,
                    )

        if exports:
            self.results["deployment_exports"] = exports
            self.report_lines.append(
                f"[Phase 4] Detection deployment: "
                f"{len(exports)} engine(s) exported to {export_dir}"
            )
            logger.info(
                "  Detection deployment: %d artefact(s) under %s",
                len(exports), export_dir,
            )

    def _build_pareto_summary(self) -> Dict[str, Any]:
        """Build the public Pareto comparison dict for I3.

        Aggregates the headline metrics across every public method row
        into best / median / worst statistics. Numeric fields end up
        prefixed with ``pareto_`` in MLflow; the full dict (including
        the per-method breakdown) is also written to
        ``pareto_summary.json`` for downstream consumers.

        The summary is built from ``self._summary_rows`` rather than
        ``self.method_results`` so it always matches the public report
        table — same row inclusion rules, same numbers.
        """
        rows = list(getattr(self, "_summary_rows", []))
        # Drop the FP32 baseline row from the per-method aggregation —
        # it's the comparator, not a competitor.
        method_rows = [r for r in rows if r.get("method") != "FP32"]
        fp32_row = next(
            (r for r in rows if r.get("method") == "FP32"), {},
        )

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            sorted_v = sorted(values)
            return {
                "best": min(sorted_v),
                "median": sorted_v[len(sorted_v) // 2],
                "worst": max(sorted_v),
            }

        # Top-1 stats are best=max not min; handle separately.
        top1_vals = [float(r["top1"]) for r in method_rows if r.get("top1") is not None]
        top1_stats: Dict[str, float] = {}
        if top1_vals:
            sorted_top1 = sorted(top1_vals, reverse=True)
            top1_stats = {
                "best": sorted_top1[0],
                "median": sorted_top1[len(sorted_top1) // 2],
                "worst": sorted_top1[-1],
            }

        size_stats = _stats(
            [float(r["size_mb"]) for r in method_rows if r.get("size_mb") is not None]
        )
        onnx_size_stats = _stats(
            [float(r["onnx_size_mb"]) for r in method_rows
             if r.get("onnx_size_mb") is not None]
        )
        onnx_lat_stats = _stats(
            [float(r["onnx_latency_ms"]) for r in method_rows
             if r.get("onnx_latency_ms") is not None]
        )

        summary: Dict[str, Any] = {
            "model_name": self.config.model_name,
            "fp32_top1": float(self.fp32_acc),
            "fp32_size_mb": float(self.fp32_size_mb),
            "n_methods": len(method_rows),
        }
        if fp32_row.get("onnx_size_mb") is not None:
            summary["fp32_onnx_size_mb"] = float(fp32_row["onnx_size_mb"])
        if fp32_row.get("onnx_latency_ms") is not None:
            summary["fp32_onnx_latency_ms"] = float(fp32_row["onnx_latency_ms"])

        for prefix, stats in [
            ("top1", top1_stats),
            ("size_mb", size_stats),
            ("onnx_size_mb", onnx_size_stats),
            ("onnx_latency_ms", onnx_lat_stats),
        ]:
            for key, val in stats.items():
                summary[f"{prefix}_{key}"] = val

        # Full breakdown so the JSON is self-contained — not flattened
        # into MLflow metrics, but available for downstream tooling.
        summary["methods"] = [
            {
                "method": r["method"],
                "top1": r.get("top1"),
                "size_mb": r.get("size_mb"),
                "onnx_size_mb": r.get("onnx_size_mb"),
                "onnx_latency_ms": r.get("onnx_latency_ms"),
                "onnx_throughput_fps": r.get("onnx_throughput_fps"),
                "ebops": r.get("ebops"),
            }
            for r in method_rows
        ]
        if self.pareto_analysis:
            summary["hypervolume"] = float(
                self.pareto_analysis.get("metrics", {}).get("hypervolume", 0.0)
            )
            summary["spacing"] = float(
                self.pareto_analysis.get("metrics", {}).get("spacing", 0.0)
            )
            summary["plot_paths"] = self.pareto_analysis.get("plot_paths", {})

        # ── Latency-backend caveat ──
        # ORT's QInt8 ops on CPU don't always beat FP32 — especially on
        # depthwise convs (MobileNet family) and unfused activation/conv
        # pairs. When that happens here, surface it explicitly in the
        # summary so consumers don't misread "INT8 = slower" as a
        # framework regression. The check is data-driven: any quantized
        # method whose ORT latency exceeds the FP32 baseline triggers
        # the note.
        fp32_ort = summary.get("fp32_onnx_latency_ms")
        if fp32_ort is not None:
            slower = [
                r["method"] for r in method_rows
                if r.get("onnx_latency_ms") is not None
                and float(r["onnx_latency_ms"]) > float(fp32_ort)
            ]
            try:
                import onnxruntime as _ort
                providers = list(_ort.get_available_providers())
            except Exception:
                providers = []
            backend_note = {
                "fp32_baseline_ms": float(fp32_ort),
                "ort_providers_available": providers,
                "methods_slower_than_fp32": slower,
                "note": (
                    "ORT QInt8 on CPU is not faster than FP32 for every "
                    "model. Depthwise convolutions in particular lack a "
                    "fast INT8 kernel, so methods listed in "
                    "'methods_slower_than_fp32' are slower in this "
                    "deployment-runtime measurement. Quote the backend "
                    "(provider) and op family when comparing latencies."
                ),
            }
            summary["latency_backend_note"] = backend_note

        return summary

    @staticmethod
    def _fmt_or_dash(value: Optional[float], fmt: str) -> str:
        """Format ``value`` with ``fmt`` or return a right-aligned ``"-"``.

        Used by the public report so missing ONNX columns don't break
        the rectangular layout. ``fmt`` is the trailing portion of an
        f-string spec (e.g. ``">9.2f"``); when ``value`` is ``None`` we
        render a dash padded to the same width derived from the spec.
        """
        if value is None:
            try:
                width = int(fmt.lstrip("<>=").split(".")[0])
            except Exception:
                width = 8
            return f"{'-':>{width}}"
        return f"{value:{fmt}}"

    def _print_deployment_fidelity_section(
        self,
        rows: List[Dict[str, Any]],
    ) -> None:
        """Public ONNX deployment-fidelity summary (Wave 5 G4).

        Surfaces three things the user always asks first:

          * "How big is the model when I actually deploy it?" — average
            ratio of on-disk INT8 ``.onnx`` size to the synthetic
            ``size_mb`` (closer to 1.0 means the synthetic estimate
            tracked reality; ratios below 1.0 indicate compiler-level
            packing the synthetic count missed).
          * "How fast is it under ONNX Runtime?" — median ORT latency
            across quantized methods, plus speedup vs the FP32 ONNX
            baseline.
          * "Where does each method land?" — a per-method delta line so
            the table-skim user can see immediately which methods
            actually improved deployment metrics.

        Only printed when ONNX numbers are present on at least one
        method row; otherwise this section is silent so reports without
        ONNX (e.g. ``onnx_export_enabled=false``) stay clean.
        """
        quantized = [
            r for r in rows
            if r.get("method", "") != "FP32"
            and r.get("onnx_size_mb") is not None
        ]
        if not quantized:
            return

        fp32_onnx = getattr(self, "fp32_onnx", {}) or {}
        fp32_onnx_size = fp32_onnx.get("fp32_onnx_size_mb")
        fp32_onnx_lat = (fp32_onnx.get("onnx_latency") or {}).get(
            "latency_mean_ms"
        )

        size_ratios = [
            float(r["onnx_size_mb"]) / float(r["size_mb"])
            for r in quantized
            if r.get("size_mb")
        ]
        latencies = [
            float(r["onnx_latency_ms"]) for r in quantized
            if r.get("onnx_latency_ms") is not None
        ]

        print()
        print("  ONNX deployment fidelity:")
        if fp32_onnx_size is not None:
            print(f"    FP32 ONNX size on disk:    {fp32_onnx_size:>8.2f} MiB")
        if fp32_onnx_lat is not None:
            print(f"    FP32 ORT mean latency:     {fp32_onnx_lat:>8.2f} ms")
        if size_ratios:
            mean_ratio = sum(size_ratios) / len(size_ratios)
            print(
                f"    Mean (on-disk / theoretical) size ratio: "
                f"{mean_ratio:>5.2f}x  "
                f"({len(size_ratios)} method"
                f"{'' if len(size_ratios) == 1 else 's'})"
            )
        if latencies:
            sorted_lat = sorted(latencies)
            median = sorted_lat[len(sorted_lat) // 2]
            line = f"    Median quantized ORT latency: {median:>5.2f} ms"
            if fp32_onnx_lat:
                speedup = fp32_onnx_lat / max(median, 1e-9)
                line += f"  ({speedup:.2f}x vs FP32 ONNX)"
            print(line)

        # Per-method deltas — only if FP32 baseline ONNX numbers exist.
        if fp32_onnx_size is not None or fp32_onnx_lat is not None:
            print()
            print("    Per-method ONNX deltas vs FP32 baseline:")
            for r in quantized:
                bits: List[str] = []
                if fp32_onnx_size and r.get("onnx_size_mb") is not None:
                    pct = (
                        (1.0 - float(r["onnx_size_mb"]) / fp32_onnx_size)
                        * 100.0
                    )
                    bits.append(f"size −{pct:>5.1f}%")
                if fp32_onnx_lat and r.get("onnx_latency_ms") is not None:
                    speed = fp32_onnx_lat / max(float(r["onnx_latency_ms"]), 1e-9)
                    bits.append(f"ORT {speed:>4.2f}x")
                if bits:
                    print(
                        f"      {r['method']:<22} " + ", ".join(bits)
                    )

        # ── INT4 packing caveat ──
        # ORT's static_quantize emits QInt8 weights only; INT4 is stored
        # one-value-per-byte in INT8 containers, so an "INT4" .onnx file
        # has the same on-disk size as its INT8 sibling. The synthetic
        # ``size_mb`` (numel × bw / 8) IS halved for INT4, which is why
        # the ``Size(MiB)`` column in the table above shows 1.07 vs 2.13
        # while ``ONNX MiB`` is identical. Surface this explicitly so
        # reviewers don't read the matching ONNX numbers as a bug.
        int4_methods_with_onnx = [
            r for r in quantized
            if r.get("onnx_size_mb") is not None
            and "INT4" in str(r.get("method", ""))
        ]
        if int4_methods_with_onnx:
            print()
            print(
                "    Note: ORT stores INT4 weights in INT8 containers "
                "(no native INT4 packing),"
            )
            print(
                "          so INT4 / INT8 ``ONNX MiB`` columns match "
                "for the same method family."
            )
            print(
                "          The ``Size(MiB)`` column reflects the "
                "true packed cost (numel × bw / 8)."
            )
            print(
                "          TensorRT and OpenVINO support native INT4 "
                "packing — see ``utils.deployment_export``."
            )

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device_str)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NeuroQuant v2.0 - Neural Network Quantization Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                               # default config, no training
  python main.py --config config.yaml          # load config from YAML
  python main.py --epochs 20                   # train 20 epochs first
  python main.py --epochs 3 --device cpu       # smoke test on CPU
  python main.py --phases phase_0_preparation phase_1a_hessian_clustering
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        type=str,
        default=None,
        help="Specific phases to run. Default: all phases.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for all artifacts (default: ./artifacts).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for reproducibility (default: from config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device: auto, cuda, cpu, mps (default: from config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="FP32 baseline training epochs. 0=skip training (default: 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: from config).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from checkpoints: skip phases that already completed.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        default=False,
        help="Write the bundled default config.yaml to the current directory and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Allow --init to overwrite an existing ./config.yaml.",
    )
    return parser.parse_args()


def _run_init_command(force: bool) -> int:
    """Copy the bundled default config to ``./config.yaml`` and return an exit code.

    Reads the template from ``quantization/_default_config.yaml`` via
    ``importlib.resources`` so it works both from a source checkout and from
    a wheel installed under ``site-packages``.
    """

    target = Path.cwd() / "config.yaml"
    pre_existing = target.exists()
    if pre_existing and not force:
        print(
            f"Refusing to overwrite existing {target}. "
            "Re-run with `--init --force` to replace it.",
            file=sys.stderr,
        )
        return 1

    try:
        source_traversable = importlib_resources.files("quantization") / "_default_config.yaml"
        with importlib_resources.as_file(source_traversable) as source_path:
            shutil.copy(source_path, target)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        print(
            f"Could not locate the bundled default config: {exc}. "
            "Reinstall NeuroQuant or report this as a packaging bug.",
            file=sys.stderr,
        )
        return 1

    action = "Overwrote" if pre_existing else "Created"
    print(f"{action} config.yaml in the current directory ({target}).")
    print("Edit it to point at your model + dataset, then run `neuroquant --config config.yaml`.")
    return 0


def main() -> None:
    """Entry point for the NeuroQuant pipeline."""
    args = parse_args()

    # ── --init: write the bundled default config and exit. ──────────────
    if args.init:
        sys.exit(_run_init_command(force=args.force))
    if args.force:
        print("`--force` has no effect without `--init`; ignoring.", file=sys.stderr)

    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in (".yml", ".yaml"):
            config = QuantizationConfig.from_yaml(config_path)
        elif config_path.suffix == ".json":
            config = QuantizationConfig.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        logger.info("Loaded config from: %s", config_path)
    else:
        config = QuantizationConfig()  # Use defaults
        logger.info("Using default configuration")

    # Override with CLI args
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.hyperparams.seed = args.seed
    if args.device is not None:
        config.hyperparams.device = args.device
    if args.phases:
        config.run_phases = args.phases
    if args.batch_size:
        config.batch_size = args.batch_size

    # Windows compatibility: avoid multiprocessing issues
    if sys.platform == "win32":
        config.num_workers = 0

    # Validate
    config.validate()

    # Run pipeline
    pipeline = NeuroQuantPipeline(
        config, training_epochs=args.epochs, resume=args.resume,
    )
    results = pipeline.run()

    logger.info("NeuroQuant pipeline finished.")
    sys.exit(0 if results.get("phases_passed") == results.get("phases_total") else 1)


if __name__ == "__main__":
    main()
