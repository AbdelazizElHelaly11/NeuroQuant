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
        (train_loader, val_loader, test_loader, calib_loader)
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

        # Build data loaders
        self.train_loader, self.val_loader, self.test_loader, self.calib_loader = \
            build_data_loaders(self.config)

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
        self.tracker.log_metrics({
            "fp32_top1": self.fp32_acc,
            "fp32_top5": self.fp32_top5,
            "fp32_ebops": self.fp32_ebops,
            "fp32_latency_mean_ms": self.fp32_latency["latency_mean_ms"],
            "fp32_latency_p50_ms": self.fp32_latency["latency_p50_ms"],
            "fp32_latency_p95_ms": self.fp32_latency["latency_p95_ms"],
            "fp32_throughput_fps": self.fp32_latency["throughput_fps"],
        })
        self.tracker.end_run()

        # Build summary row for final report table
        self._add_summary_row(
            "FP32", self.fp32_acc, self.fp32_top5,
            self.fp32_latency["latency_mean_ms"],
            self.fp32_latency["throughput_fps"],
            self.fp32_ebops, self.fp32_size_mb,
        )

        self.report_lines.append(
            f"[Phase 0] FP32 baseline: top1={self.fp32_acc:.2f}%, "
            f"top5={self.fp32_top5:.2f}%, "
            f"latency={self.fp32_latency['latency_mean_ms']:.1f}ms, "
            f"checkpoint={ckpt_path.name}"
        )
        self.results["fp32_acc"] = self.fp32_acc
        self.results["fp32_top5"] = self.fp32_top5

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

        # Store the best solution's config for Adaround/QAT
        if self.pareto_front["solutions"]:
            self.best_config = self.pareto_front["solutions"][0]["bitwidth_assignment"]

        self.report_lines.append(
            f"[Phase 1c] NSGA-II: {n_pareto} Pareto solutions, "
            f"{self.pareto_front['evaluations']} evals"
        )
        self.results["pareto_solutions"] = n_pareto

        # ── Real PTQ on the NSGA winner ──
        # NSGA-II searches with fake-quant for speed; here we materialise
        # the best configuration through PTQQuantizer + calibration so the
        # downstream phases (XAI, reporting) have an actual quantized model
        # rather than only search-time surrogates.
        from quantization.ptq import PTQQuantizer

        methods_enabled = {m.value.lower() for m in self.config.methods}
        if self.best_config and ("ptq" in methods_enabled or not methods_enabled):
            ptq = PTQQuantizer(self.model, self.config)
            ptq.calibrate(
                self.calib_loader,
                num_batches=self.config.hyperparams.calibration_batches,
            )
            ptq_model = ptq.quantize_with_config(self.best_config)
            ptq_res = ptq.evaluate(
                ptq_model, self.val_loader,
                bitwidth=self._dominant_bitwidth(self.best_config),
            )
            ptq_res["config_id"] = "PTQ_best"
            ptq_res["bitwidth_assignment"] = self.best_config

            # Expose to downstream phases + report/summary.
            self.results["ptq_model"] = ptq_model
            self.results["ptq_best_result"] = ptq_res
            self.method_results.append(ptq_res)
            lat = ptq_res.get("latency") or {}
            self._add_summary_row(
                f"PTQ_{ptq_res['config_id']}",
                ptq_res["accuracy"],
                ptq_res.get("top5_accuracy", 0) or 0.0,
                lat.get("latency_mean_ms", 0.0),
                lat.get("throughput_fps", 0.0),
                ptq_res["ebops"],
                ptq_res["model_size_mb"],
            )
            self.report_lines.append(
                f"[Phase 1c] PTQ best: acc={ptq_res['accuracy']:.2f}%, "
                f"ebops={ptq_res['ebops']:.0f}"
            )

        # Checkpoint
        self.ckpt.save_phase_json("phase_1c_nsga_search", {
            "pareto_front": self.pareto_front,
            "best_config": self.best_config,
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
            adaround_model, self.best_config, self.config
        )
        self.adaround_result = adaround_opt.run()

        self.tracker.log_metrics({
            "adaround_mse_before": self.adaround_result["mse_before"],
            "adaround_mse_after": self.adaround_result["mse_after"],
            "adaround_mse_reduction": self.adaround_result["mse_reduction"],
        })
        self.tracker.end_run()

        self.report_lines.append(
            f"[Phase 1d] Adaround: MSE reduction={self.adaround_result['mse_reduction']:.1f}%"
        )

        # Checkpoint: save the adaround model (.pth with metadata envelope)
        # and a companion JSON so resume can restore the full result dict
        # even when only metadata fields are needed.
        adaround_meta = {
            "mse_before": self.adaround_result["mse_before"],
            "mse_after": self.adaround_result["mse_after"],
            "mse_reduction": self.adaround_result["mse_reduction"],
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
        self.tracker.end_run()

        self.report_lines.append(
            f"[Phase 1e] QAT: final_acc={self.qat_result['final_val_acc']:.2f}%"
        )
        self.results["qat_acc"] = self.qat_result["final_val_acc"]

        # Checkpoint: save the fine-tuned QAT model and the metric history.
        qat_meta = {
            "final_val_acc": self.qat_result.get("final_val_acc", 0.0),
            "best_epoch": self.qat_result.get("best_epoch", 0),
            "train_accuracy": self.qat_result.get("train_accuracy", []),
            "val_accuracy": self.qat_result.get("val_accuracy", []),
            "train_loss": self.qat_result.get("train_loss", []),
            "time_seconds": self.qat_result.get("time_seconds", 0.0),
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
        new_results: List[Dict[str, Any]] = []

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
                res["config_id"] = f"{label}_INT{bw}"
                self.method_results.append(res)
                new_results.append(res)
                # Keep the first model per method family for phase 3 comparison.
                produced_models.setdefault(key, q_model)
                # Additionally expose a bitwidth-tagged slot so phase 3 can
                # pick the preferred variant without re-quantization.
                produced_models[f"{key}_int{bw}"] = q_model
                logger.info("    %s INT%d: acc=%.2f%%", label, bw, res["accuracy"])
            except Exception as e:
                logger.warning("    %s INT%d FAILED: %s", label, bw, e)

        # Log all new results plus add their rows to the summary table.
        for res in new_results:
            method = res['method']
            # config_id is set above as "METHOD_INT<bw>" — embed it as a tag
            # for the MLflow metric keys so multiple bitwidths per method
            # don't collide.
            tag = res.get("config_id", method)
            self.tracker.log_metrics({
                f"{tag}_top1": res["accuracy"],
                f"{tag}_top5": res.get("top5_accuracy", 0) or 0,
                f"{tag}_ebops": res["ebops"],
                f"{tag}_latency_ms": res.get("latency_ms", 0) or 0,
            })
            lat = res.get("latency") or {}
            self._add_summary_row(
                f"{method}_{res['config_id']}",
                res["accuracy"],
                res.get("top5_accuracy", 0) or 0.0,
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

        # Store representative models for Phase 3.
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

        # Merge NSGA-II solutions + Phase 1f results + PTQ + QAT into
        # a single solution pool. NSGA-II solutions already live in
        # self.pareto_front["solutions"]; PTQ/1f results live in
        # self.method_results (appended by phases 1c and 1f); QAT is
        # synthesised here from self.qat_result if present.
        all_solutions = list(self.pareto_front.get("solutions", []))

        # QAT entry — treated as a point solution for the merged Pareto.
        if self.qat_result:
            qat_acc = float(self.qat_result.get("final_val_acc", 0.0) or 0.0)
            # Re-use NSGA best_config as QAT's effective bitwidth map and
            # derive ebops from it; fall back to FP32 if not available.
            qat_bw = self.best_config or {}
            qat_ebops = self._ebops_from_bitwidth(qat_bw)
            qat_red = (
                (self.fp32_ebops - qat_ebops) / max(self.fp32_ebops, 1) * 100
            )
            # Prefer a real measured latency if the QAT result carries one;
            # otherwise leave None so downstream aggregation ignores it.
            qat_lat = self.qat_result.get("latency_mean_ms")
            if qat_lat is None:
                qat_lat_dict = self.qat_result.get("latency") or {}
                qat_lat = qat_lat_dict.get("latency_mean_ms")
            all_solutions.append(ParetoSolution(
                solution_id="QAT_warmstart",
                method="QAT",
                accuracy=qat_acc,
                accuracy_loss=self.fp32_acc - qat_acc,
                ebops=qat_ebops,
                ebops_reduction=qat_red,
                model_size_mb=qat_ebops / 1e6,
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
            # Prefer the structured latency dict; fall back to the flat
            # latency_ms field. Leave None if neither is populated so the
            # downstream aggregation can safely skip missing values.
            res_lat = None
            lat_dict = res.get("latency") or {}
            if "latency_mean_ms" in lat_dict:
                res_lat = lat_dict.get("latency_mean_ms")
            elif res.get("latency_ms") is not None:
                res_lat = res.get("latency_ms")
            sol = ParetoSolution(
                solution_id=f"{res['method']}_{res['config_id']}",
                method=res["method"],
                accuracy=res["accuracy"],
                accuracy_loss=self.fp32_acc - res["accuracy"],
                ebops=res["ebops"],
                ebops_reduction=ebops_red,
                model_size_mb=res["ebops"] / 1e6,
                latency_mean_ms=res_lat,
                bitwidth_assignment=res.get("bitwidth_assignment", {}),
                rank=1,
                crowding_distance=0.0,
                is_dominated=False,
            )
            all_solutions.append(sol)

        # Build merged ParetoFront
        merged_front = ParetoFront(
            solutions=all_solutions,
            generation=self.pareto_front.get("generation", 0),
            evaluations=self.pareto_front.get("evaluations", 0),
            convergence_reason="merged",
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
            f"[Phase 2] Pareto: {len(all_solutions)} total solutions, "
            f"HV={hv:.2f}"
        )
        self.results["hypervolume"] = hv

        # Checkpoint: pareto_analysis + hypervolume so resume can skip this.
        self.ckpt.save_phase_json("phase_2_pareto", {
            "pareto_analysis": self.pareto_analysis,
            "hypervolume": hv,
            "total_solutions": len(all_solutions),
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

        # Build quantized models dict: include PTQ best, QAT fine-tuned,
        # and each phase‑1f family representative that was produced.
        quant_models: Dict[str, nn.Module] = {}
        if "ptq_model" in self.results:
            quant_models["PTQ_best"] = self.results["ptq_model"]
        qat_m = self.qat_result.get("model") if self.qat_result else None
        if isinstance(qat_m, nn.Module):
            quant_models["QAT_warmstart"] = qat_m
        if "gptq_model" in self.results:
            quant_models["GPTQ"] = self.results["gptq_model"]
        if "awq_model" in self.results:
            quant_models["AWQ"] = self.results["awq_model"]
        if "sq_model" in self.results:
            quant_models["SmoothQuant"] = self.results["sq_model"]

        xai_result = xai_gen.run(
            fp32_model=self.model,
            quantized_models=quant_models,
            test_images=test_images,
            test_labels=test_labels,
            output_dir=str(xai_dir),
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

        # Data loaders are not serialised; rebuild from config.
        (self.train_loader, self.val_loader,
         self.test_loader, self.calib_loader) = build_data_loaders(self.config)

        data = self.ckpt.load_phase_json("phase_0_preparation")
        self.fp32_acc = float(data.get("fp32_acc", 0.0))
        self.fp32_top5 = float(data.get("fp32_top5", 0.0))
        self.fp32_ebops = float(data.get("fp32_ebops", 0.0))
        self.fp32_size_mb = float(data.get("fp32_size_mb", 0.0))
        self.fp32_latency = data.get("fp32_latency", {}) or {}

        self.results["fp32_acc"] = self.fp32_acc
        self.results["fp32_top5"] = self.fp32_top5

        # Recreate the FP32 row in the summary table for the final report.
        self._add_summary_row(
            "FP32", self.fp32_acc, self.fp32_top5,
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
        for res in self.method_results:
            lat = res.get("latency") or {}
            self._add_summary_row(
                f"{res.get('method', '?')}_{res.get('config_id', '?')}",
                res.get("accuracy", 0.0),
                res.get("top5_accuracy", 0) or 0.0,
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

    def _add_summary_row(
        self, method: str, top1: float, top5: float,
        latency_ms: float, throughput: float,
        ebops: float, size_mb: float,
    ) -> None:
        """Add a row to the summary table for the final report."""
        if not hasattr(self, '_summary_rows'):
            self._summary_rows = []
        self._summary_rows.append({
            "method": method, "top1": top1, "top5": top5,
            "latency_ms": latency_ms, "throughput": throughput,
            "ebops": ebops, "size_mb": size_mb,
        })

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
        primary = getattr(
            self.config.hyperparams, "eval_primary_accuracy", "top1"
        )
        print(f"  Primary Acc:    {primary}")
        print()
        for line in self.report_lines:
            marker = "[OK]" if not line.startswith("[ERROR]") else "[FAIL]"
            print(f"  {marker} {line}")

        # Summary table
        rows = getattr(self, '_summary_rows', [])
        if rows:
            print()
            hdr = f"  {'Method':<22} {'Top-1':>7} {'Top-5':>7} {'Lat(ms)':>9} {'FPS':>9} {'EBops':>12} {'Size(MB)':>10}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for r in rows:
                print(
                    f"  {r['method']:<22} {r['top1']:>6.2f}% {r['top5']:>6.2f}% "
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
