"""
NeuroQuant v2.0 - MLflow Experiment Tracker (Phase 4)

Provides a unified logging interface for all pipeline phases.
Uses local file-based MLflow tracking for 100% reproducibility.

Graceful degradation: if mlflow is not installed, all operations
become no-ops (the pipeline still runs without tracking).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from config import QuantizationConfig, QuantizationResult, ParetoSolution

logger = logging.getLogger("neuroquant")

# Optional MLflow
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class MLflowTracker:
    """
    MLflow experiment tracker for the NeuroQuant pipeline.

    Logs parameters, metrics, and artifacts for each phase.
    Degrades gracefully if MLflow is not installed.
    """

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config
        self._active = HAS_MLFLOW
        self._run_active = False

        if self._active:
            tracking_uri = getattr(config, "mlflow_tracking_uri", "./mlruns")
            experiment_name = getattr(config, "experiment_name", "neuroquant_v2")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow tracking: uri=%s, experiment=%s",
                        tracking_uri, experiment_name)
        else:
            logger.info("MLflow not installed. Tracking disabled (no-op mode).")

    def start_run(
        self, run_name: str, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Start a new MLflow run."""
        if not self._active:
            logger.info("[MLflow:no-op] start_run('%s')", run_name)
            return
        try:
            # End any active run first
            if self._run_active:
                self.end_run()
            mlflow.start_run(run_name=run_name, tags=tags or {})
            self._run_active = True
            logger.info("[MLflow] Started run: %s", run_name)
        except Exception as e:
            logger.warning("[MLflow] Failed to start run: %s", e)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        if not self._active:
            return
        try:
            if self._run_active:
                mlflow.end_run(status=status)
                self._run_active = False
        except Exception as e:
            logger.warning("[MLflow] Failed to end run: %s", e)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dictionary of parameters."""
        if not self._active or not self._run_active:
            return
        try:
            # MLflow params must be strings
            safe_params = {k: str(v)[:250] for k, v in params.items()}
            mlflow.log_params(safe_params)
        except Exception as e:
            logger.warning("[MLflow] Failed to log params: %s", e)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log a dictionary of metrics."""
        if not self._active or not self._run_active:
            return
        try:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not (
                    v != v  # NaN check
                ):
                    mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            logger.warning("[MLflow] Failed to log metrics: %s", e)

    def log_artifact(self, local_path: str, artifact_path: str = "") -> None:
        """Log a file or directory as an artifact."""
        if not self._active or not self._run_active:
            return
        try:
            p = Path(local_path)
            if p.is_dir():
                mlflow.log_artifacts(str(p), artifact_path)
            elif p.is_file():
                mlflow.log_artifact(str(p), artifact_path)
        except Exception as e:
            logger.warning("[MLflow] Failed to log artifact '%s': %s",
                          local_path, e)

    def log_model(self, model: nn.Module, artifact_name: str) -> None:
        """Log a PyTorch model as an artifact."""
        if not self._active or not self._run_active:
            return
        try:
            # Save to temp file, then log
            tmp_path = Path("./artifacts") / f"{artifact_name}.pth"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), tmp_path)
            mlflow.log_artifact(str(tmp_path), "models")
        except Exception as e:
            logger.warning("[MLflow] Failed to log model: %s", e)

    def log_config(self, config: QuantizationConfig) -> None:
        """Log the full configuration as a JSON artifact."""
        if not self._active or not self._run_active:
            return
        try:
            config_dict = {
                "dataset_name": config.dataset_name,
                "batch_size": config.batch_size,
                "num_classes": config.num_classes,
            }
            # Add hyperparams
            hp = config.hyperparams
            for field in hp.__dataclass_fields__:
                val = getattr(hp, field)
                if hasattr(val, "value"):
                    val = val.value
                config_dict[f"hp_{field}"] = val

            tmp = Path("./artifacts/config_logged.json")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
            mlflow.log_artifact(str(tmp))
        except Exception as e:
            logger.warning("[MLflow] Failed to log config: %s", e)

    # ------------------------------------------------------------------
    # Phase-specific convenience methods
    # ------------------------------------------------------------------

    def log_hessian_computation(
        self,
        hessian_diag: Dict[str, float],
        cluster_metadata: Dict[str, Any],
    ) -> None:
        """Log Phase 1a results."""
        self.log_metrics({
            "hessian_num_layers": len(hessian_diag),
            "hessian_mean": sum(hessian_diag.values()) / max(len(hessian_diag), 1),
            "hessian_max": max(hessian_diag.values()) if hessian_diag else 0,
        })
        if "statistics" in cluster_metadata:
            stats = cluster_metadata["statistics"]
            self.log_metrics({
                "cluster_high": stats.get("HIGH", 0),
                "cluster_medium": stats.get("MEDIUM", 0),
                "cluster_low": stats.get("LOW", 0),
            })

    def log_ptq_configs(
        self,
        configs: List[Dict[str, int]],
        results: List[QuantizationResult],
    ) -> None:
        """Log Phase 1c results."""
        self.log_metrics({"nsga_num_configs": len(configs)})
        for i, res in enumerate(results):
            self.log_metrics({
                f"nsga_config_{i}_acc": res["accuracy"],
                f"nsga_config_{i}_ebops": res["ebops"],
            })

    def log_qat_results(
        self,
        train_losses: List[float],
        val_accuracies: List[float],
        final_result: QuantizationResult,
    ) -> None:
        """Log Phase 1e results."""
        for i, (loss, acc) in enumerate(zip(train_losses, val_accuracies)):
            self.log_metrics({"qat_train_loss": loss, "qat_val_acc": acc}, step=i)
        self.log_metrics({
            "qat_final_acc": final_result.get("accuracy", 0),
        })

    def log_method_results(
        self, method_name: str, results: List[QuantizationResult]
    ) -> None:
        """Log Phase 1f results."""
        for res in results:
            self.log_metrics({
                f"{method_name}_acc": res["accuracy"],
                f"{method_name}_ebops": res["ebops"],
            })

    def log_adaround_optimization(
        self,
        before_mse: Dict[str, float],
        after_mse: Dict[str, float],
        model_path: str,
    ) -> None:
        """Log Phase 1d results."""
        avg_before = sum(before_mse.values()) / max(len(before_mse), 1)
        avg_after = sum(after_mse.values()) / max(len(after_mse), 1)
        self.log_metrics({
            "adaround_mse_before": avg_before,
            "adaround_mse_after": avg_after,
            "adaround_mse_reduction": (avg_before - avg_after) / max(avg_before, 1e-10) * 100,
        })
        self.log_artifact(model_path, "models")

    def log_final_pareto_front(
        self,
        pareto_results: List[ParetoSolution],
        plot_path: str,
    ) -> None:
        """Log Phase 2 results."""
        self.log_metrics({
            "pareto_num_solutions": len(pareto_results),
        })
        if pareto_results:
            best = max(pareto_results, key=lambda s: s.get("accuracy", 0))
            self.log_metrics({
                "pareto_best_acc": best.get("accuracy", 0),
                "pareto_best_ebops": best.get("ebops", 0),
            })
        if Path(plot_path).exists():
            self.log_artifact(plot_path, "plots")

    def log_xai_artifacts(self, xai_folder_path: str) -> None:
        """Log Phase 3 artifacts."""
        self.log_artifact(xai_folder_path, "xai")
