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
import sys
import time
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
    Build train/val/test/calibration DataLoaders from config.

    Returns:
        (train_loader, val_loader, test_loader, calib_loader, class_names)
        ``class_names`` is the dataset's ``.classes`` list when available,
        otherwise ``None``. Older callers that unpack 4 values continue
        to work via slicing.
    """
    from data.data_loader import GenericDatasetLoader

    loader = GenericDatasetLoader(config)
    return (
        loader.get_train_loader(),
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

        # Reproducibility
        seed = config.hyperparams.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # State populated during the pipeline
        self.model: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
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
        (self.train_loader, self.val_loader, self.test_loader,
         self.calib_loader, self.class_names) = build_data_loaders(self.config)

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
        self.tracker.log_metrics({
            "fp32_top1": self.fp32_acc,
            "fp32_ebops": self.fp32_ebops,
            "fp32_size_mb": self.fp32_size_mb,
            "fp32_latency_mean_ms": self.fp32_latency["latency_mean_ms"],
            "fp32_latency_p50_ms": self.fp32_latency["latency_p50_ms"],
            "fp32_latency_p95_ms": self.fp32_latency["latency_p95_ms"],
            "fp32_throughput_fps": self.fp32_latency["throughput_fps"],
        })
        self.tracker.end_run()

        # Build summary row for final report table — Top-1 only on the
        # public surface (Top-5 stays in internal metrics if computed).
        self._add_summary_row(
            "FP32", self.fp32_acc,
            self.fp32_latency["latency_mean_ms"],
            self.fp32_latency["throughput_fps"],
            self.fp32_ebops, self.fp32_size_mb,
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

        nsga = NSGAIIClusterSearch(
            self.model, self.cluster_assignments, self.config
        )
        self.pareto_front = nsga.search(
            self.val_loader, self.fp32_acc, self.fit_seed["seed_config"],
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
            mixed_ranked = [
                s for s in ranked
                if self._is_mixed_bitwidth_assignment(
                    s.get("bitwidth_assignment", {}),
                )
            ]
            if mixed_ranked:
                selected = mixed_ranked[0]
                logger.info(
                    "  Phase 1c: selected mixed PTQ config %s for "
                    "materialization (acc=%.2f%%, size=%.2f MiB).",
                    selected.get("solution_id", "unknown"),
                    float(selected.get("accuracy", 0.0)),
                    float(selected.get("model_size_mb", 0.0)),
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
                self._add_summary_row(
                    res["display_name"],
                    res["accuracy"],
                    lat.get("latency_mean_ms", 0.0),
                    lat.get("throughput_fps", 0.0),
                    res["ebops"],
                    res["model_size_mb"],
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

        # Checkpoint
        self.ckpt.save_phase_json("phase_1c_nsga_search", {
            "pareto_front": self.pareto_front,
            "best_config": self.best_config,
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

        adaround_model = copy.deepcopy(self.model)
        adaround_opt = AdaroundOptimizer(
            adaround_model, self.best_config, self.config,
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

        qat_model = copy.deepcopy(self.adaround_result["model"])
        qat_trainer = QATTrainer(qat_model, self.best_config, self.config)
        self.qat_result = qat_trainer.train(self.train_loader, self.val_loader)

        self.tracker.log_metrics({
            "qat_final_acc": self.qat_result["final_val_acc"],
            "qat_epochs": len(self.qat_result["val_accuracy"]),
        })
        ws_src = self.results.get("qat_warmstart_source") or "ptq_best_acc"
        ws_id = self.results.get("qat_warmstart_id") or "n/a"
        self.tracker.log_params({
            "qat_warmstart_source": str(ws_src),
            "qat_warmstart_id": str(ws_id),
        })
        self.tracker.end_run()

        self.report_lines.append(
            f"[Phase 1e] QAT: final_acc={self.qat_result['final_val_acc']:.2f}% "
            f"(warmstart={ws_src}={ws_id})"
        )
        self.results["qat_acc"] = self.qat_result["final_val_acc"]

        # Checkpoint: save the fine-tuned QAT model and the metric history.
        # The warmstart source (ptq_best_acc / ptq_best_tradeoff) and the
        # PTQ ID it selected are persisted alongside so resuming or
        # post-mortem analysis can answer "which PTQ was QAT trained on?".
        qat_meta = {
            "final_val_acc": self.qat_result.get("final_val_acc", 0.0),
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
        ]

        produced_models: Dict[str, nn.Module] = {}
        produced_models_by_id: Dict[str, nn.Module] = {}
        new_results: List[Dict[str, Any]] = []
        from utils.common import compute_quantized_size_mb

        for i, (key, label, bw, cls) in enumerate(plan, 1):
            if methods_enabled and key not in methods_enabled:
                continue
            logger.info("  [%d/%d] %s INT%d ...", i, len(plan), label, bw)
            try:
                quantizer = cls(self.model, self.config)
                q_model = quantizer.quantize(
                    self.calib_loader, bitwidth=bw,
                    num_batches=hp.calibration_batches,
                )
                res = quantizer.evaluate(q_model, self.val_loader, bitwidth=bw)
                # Canonical, bitwidth-tagged identity. ``display_name``
                # is the public label used everywhere (summary table,
                # Pareto plot, XAI rows). It deliberately does NOT
                # double-prefix with the method family.
                res["config_id"] = f"{label}_INT{bw}"
                res["display_name"] = f"{label}_INT{bw}"
                res["bitwidth"] = int(bw)
                # Real model size from the bitwidth assignment; the
                # quantizer may not have populated this consistently.
                bw_assignment = res.get("bitwidth_assignment", {}) or {
                    n: bw for n, _ in self.model.named_parameters()
                    if "weight" in n
                }
                res["bitwidth_assignment"] = bw_assignment
                res["model_size_mb"] = compute_quantized_size_mb(
                    self.model, bw_assignment,
                )
                self.method_results.append(res)
                new_results.append(res)
                produced_models.setdefault(key, q_model)
                produced_models_by_id[res["display_name"]] = q_model
                logger.info("    %s INT%d: acc=%.2f%%", label, bw, res["accuracy"])
            except Exception as e:
                logger.warning("    %s INT%d FAILED: %s", label, bw, e)

        # Log results to MLflow with bitwidth-tagged keys; Top-5 is NOT
        # surfaced (kept internal only).
        for res in new_results:
            tag = res["display_name"]
            self.tracker.log_metrics({
                f"{tag}_top1": res["accuracy"],
                f"{tag}_ebops": res["ebops"],
                f"{tag}_size_mb": res["model_size_mb"],
                f"{tag}_latency_ms": res.get("latency_ms", 0) or 0,
            })
            lat = res.get("latency") or {}
            self._add_summary_row(
                res["display_name"],
                res["accuracy"],
                lat.get("latency_mean_ms", 0) or 0.0,
                lat.get("throughput_fps", 0) or 0.0,
                res["ebops"],
                res["model_size_mb"],
            )
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

        # Checkpoint: persist each family's representative so phase 3 can
        # resume without re-running quantization, and the method results
        # so phase 2 can merge solutions on resume.
        if "gptq" in produced_models:
            self.ckpt.save_named_model("phase_1f_gptq_model.pth",
                                       produced_models["gptq"])
        if "awq" in produced_models:
            self.ckpt.save_named_model("phase_1f_awq_model.pth",
                                       produced_models["awq"])
        if "smoothquant" in produced_models:
            # SmoothQuant inserts architectural wrappers (_SmoothInputScale)
            # per layer; a state-dict-only save would drop them on reload,
            # so pickle the full module instead.
            self.ckpt.save_full_module("phase_1f_sq_model.pt",
                                       produced_models["smoothquant"])
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
        all_solutions = self._filter_non_dominated_solutions(public_candidates)
        if len(all_solutions) < len(public_candidates):
            logger.info(
                "  Public Pareto filter: kept %d non-dominated of %d candidates",
                len(all_solutions), len(public_candidates),
            )

        merged_front = ParetoFront(
            solutions=all_solutions,
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
            "total_solutions": len(all_solutions),
            "public_candidates": len(public_candidates),
            "hypervolume": hv,
        }
        # Optional third Pareto axis: mean latency across solutions. Kept
        # as an auxiliary metric so existing 2D analysis is unchanged.
        if self.config.hyperparams.use_latency_in_pareto:
            latencies = [
                s.get("latency_mean_ms") for s in all_solutions
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
            f"[Phase 2] Pareto: {len(all_solutions)} non-dominated / "
            f"{len(public_candidates)} candidates, "
            f"HV={hv:.2f}"
        )
        self.results["hypervolume"] = hv

        # Checkpoint: pareto_analysis + hypervolume so resume can skip this.
        self.ckpt.save_phase_json("phase_2_pareto", {
            "pareto_analysis": self.pareto_analysis,
            "hypervolume": hv,
            "total_solutions": len(all_solutions),
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

        self.tracker.log_metrics(summary_metrics)

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
        (self.train_loader, self.val_loader,
         self.test_loader, self.calib_loader,
         self.class_names) = build_data_loaders(self.config)

        data = self.ckpt.load_phase_json("phase_0_preparation")
        self.fp32_acc = float(data.get("fp32_acc", 0.0))
        self.fp32_top5 = float(data.get("fp32_top5", 0.0))
        self.fp32_ebops = float(data.get("fp32_ebops", 0.0))
        self.fp32_size_mb = float(data.get("fp32_size_mb", 0.0))
        self.fp32_latency = data.get("fp32_latency", {}) or {}

        self.results["fp32_acc"] = self.fp32_acc
        self.results["fp32_top5"] = self.fp32_top5

        # Recreate the FP32 row in the summary table for the final report.
        # Top-1 only on the public surface.
        self._add_summary_row(
            "FP32", self.fp32_acc,
            self.fp32_latency.get("latency_mean_ms", 0.0),
            self.fp32_latency.get("throughput_fps", 0.0),
            self.fp32_ebops, self.fp32_size_mb,
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
            self._add_summary_row(
                res.get("display_name") or res.get("config_id", "PTQ"),
                float(res.get("accuracy", 0.0)),
                lat.get("latency_mean_ms", 0.0),
                lat.get("throughput_fps", 0.0),
                float(res.get("ebops", 0.0)),
                float(res.get("model_size_mb", 0.0)),
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
            "best_epoch": int(metadata.get("best_epoch", 0)),
            "train_accuracy": metadata.get("train_accuracy", []),
            "val_accuracy": metadata.get("val_accuracy", []),
            "train_loss": metadata.get("train_loss", []),
            "time_seconds": float(metadata.get("time_seconds", 0.0)),
        }
        self.results["qat_acc"] = self.qat_result["final_val_acc"]
        # Restore warmstart provenance so post-mortem reads from
        # checkpoint match the original run.
        if metadata.get("qat_warmstart_source"):
            self.results["qat_warmstart_source"] = metadata["qat_warmstart_source"]
        if metadata.get("qat_warmstart_id"):
            self.results["qat_warmstart_id"] = metadata["qat_warmstart_id"]

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
        awq_model = _restore("awq")
        if gptq_model is not None:
            self.results["gptq_model"] = gptq_model
        if awq_model is not None:
            self.results["awq_model"] = awq_model

        # SmoothQuant is pickled as a full module so its input-scaling
        # wrappers survive resume. Fall back to the legacy state_dict path
        # for checkpoints produced before that change.
        if self.ckpt.file_exists("phase_1f_sq_model.pt"):
            self.results["sq_model"] = self.ckpt.load_full_module(
                "phase_1f_sq_model.pt"
            ).to(self.device)
        else:
            sq_model = _restore("sq")
            if sq_model is not None:
                self.results["sq_model"] = sq_model

        # Rebuild the summary-table rows produced on the original run.
        # Use the canonical display_name (no METHOD_METHOD duplication)
        # and Top-1 only.
        for res in self.method_results:
            lat = res.get("latency") or {}
            display = (
                res.get("display_name")
                or res.get("config_id", res.get("method", "?"))
            )
            self._add_summary_row(
                display,
                res.get("accuracy", 0.0),
                lat.get("latency_mean_ms", 0.0),
                lat.get("throughput_fps", 0.0),
                res.get("ebops", 0.0),
                res.get("model_size_mb", 0.0),
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
            res = ptq.evaluate(q_model, self.val_loader, bitwidth=dom_bw)

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
                "  PTQ rerank candidate %s: acc=%.2f%%, size=%.2f MiB",
                display, float(res["accuracy"]), float(res["model_size_mb"]),
            )

        if not materialized:
            return None, None, None, None

        # Best by REAL Top-1.
        best_acc_idx = max(
            range(len(materialized)),
            key=lambda i: float(materialized[i]["accuracy"]),
        )

        # Tradeoff: most compressed among those within the accuracy cap;
        # fallback to smallest size overall when none satisfies the cap.
        cap = float(getattr(hp, "ptq_tradeoff_max_acc_drop", 1.0))
        ref_acc = float(materialized[best_acc_idx]["accuracy"])
        within_cap = [
            i for i, r in enumerate(materialized)
            if (ref_acc - float(r["accuracy"])) <= cap
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
        if best_acc_idx == best_tradeoff_idx:
            return (
                models[best_acc_idx], materialized[best_acc_idx],
                models[best_tradeoff_idx], materialized[best_tradeoff_idx],
            )

        # When tags collide but configs differ, append a discriminator.
        if (
            materialized[best_acc_idx]["display_name"]
            == materialized[best_tradeoff_idx]["display_name"]
        ):
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

        return (
            models[best_acc_idx], materialized[best_acc_idx],
            models[best_tradeoff_idx], materialized[best_tradeoff_idx],
        )

    @staticmethod
    def _filter_non_dominated_solutions(
        solutions: List[ParetoSolution],
    ) -> List[ParetoSolution]:
        """Return only non-dominated solutions (minimize loss + ebops)."""
        if not solutions:
            return []

        kept: List[ParetoSolution] = []
        for i, sol_i in enumerate(solutions):
            dominated = False
            li = float(sol_i.get("accuracy_loss", float("inf")))
            ei = float(sol_i.get("ebops", float("inf")))
            for j, sol_j in enumerate(solutions):
                if i == j:
                    continue
                lj = float(sol_j.get("accuracy_loss", float("inf")))
                ej = float(sol_j.get("ebops", float("inf")))
                if (lj <= li and ej <= ei) and (lj < li or ej < ei):
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
        """
        if not hasattr(self, '_summary_rows'):
            self._summary_rows = []
        row = {
            "method": method, "top1": top1,
            "latency_ms": latency_ms, "throughput": throughput,
            "ebops": ebops, "size_mb": size_mb,
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
        rows = getattr(self, '_summary_rows', [])
        if rows:
            print()
            hdr = (
                f"  {'Method':<22} {'Top-1':>7} {'Lat(ms)':>9} "
                f"{'FPS':>9} {'EBops':>12} {'Size(MiB)':>10}"
            )
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for r in rows:
                print(
                    f"  {r['method']:<22} {r['top1']:>6.2f}% "
                    f"{r['latency_ms']:>8.2f} {r['throughput']:>8.1f} "
                    f"{r['ebops']:>12.0f} {r['size_mb']:>9.2f}"
                )

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
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, cpu (default: auto).",
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
    return parser.parse_args()


def main() -> None:
    """Entry point for the NeuroQuant pipeline."""
    args = parse_args()

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
    if args.seed:
        config.hyperparams.seed = args.seed
    if args.device:
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
