"""
NeuroQuant v2.0 - Configuration Module

Defines all configuration dataclasses and type definitions
for the quantization framework. Supports YAML/JSON config loading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom Type Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from typing import TypedDict


class LayerSensitivity(TypedDict):
    """Hessian sensitivity info for a single layer."""
    layer_name: str
    hessian_diag: float
    layer_type: str  # 'Conv2d', 'Linear', 'BatchNorm2d', etc.
    num_parameters: int


class ClusterAssignment(TypedDict):
    """Cluster assignment for a group of layers."""
    cluster_id: int
    tier: str  # 'HIGH', 'MEDIUM', 'LOW'
    layer_names: List[str]
    allowed_bitwidths: List[int]  # e.g., [8] for HIGH, [4, 8] for MEDIUM/LOW
    mean_sensitivity: float


class LatencyResult(TypedDict):
    """Inference latency benchmark results."""
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_fps: float


class HardwareMetrics(TypedDict, total=False):
    """Optional hardware synthesis metrics from external report."""
    dsp: Optional[int]
    lut: Optional[int]
    ff: Optional[int]
    fmax_mhz: Optional[float]
    ii: Optional[int]              # initiation interval
    cycle_latency: Optional[int]
    source: str                    # 'vivado_hls' | 'quartus' | 'not_provided'


class QuantizationResult(TypedDict):
    """Result from evaluating a single quantized model."""
    config_id: str
    method: str  # 'PTQ', 'QAT', 'GPTQ', 'SmoothQuant', 'AWQ'
    bitwidth_assignment: Dict[str, int]  # layer_name -> bitwidth
    accuracy: float
    top5_accuracy: Optional[float]
    model_size_mb: float
    ebops: float
    latency_ms: Optional[float]
    latency: Optional[LatencyResult]
    hardware: Optional[HardwareMetrics]
    model_path: Optional[str]


class ParetoSolution(TypedDict, total=False):
    """A non-dominated solution on the Pareto front."""
    solution_id: str
    method: str
    accuracy: float
    top5_accuracy: Optional[float]
    accuracy_loss: float              # FP32_acc - quantized_acc (%)
    ebops: float
    ebops_reduction: float            # (FP32 - quantized) / FP32 (%)
    model_size_mb: float
    latency_mean_ms: Optional[float]
    bitwidth_assignment: Dict[str, int]
    rank: int                         # 1 = Pareto front, >1 = dominated
    crowding_distance: float          # Diversity metric [0, inf]
    is_dominated: bool


class FITCompressSeed(TypedDict):
    """Output of FITCompress warm-start seed generation."""
    seed_config: Dict[str, int]       # param_name -> bitwidth (4 or 8)
    fit_scores: Dict[str, float]      # param_name -> normalized FIT score [0, 1]
    compression_potential: float      # % memory saved vs FP32
    elite_status: str                 # 'HIGH_POTENTIAL' or 'BALANCED'


class ParetoFront(TypedDict):
    """Output of NSGA-II search: the full Pareto front."""
    solutions: List[ParetoSolution]   # Non-dominated solutions (rank 1)
    generation: int                   # Final generation number
    evaluations: int                  # Total configurations evaluated
    convergence_reason: str           # 'max_gen' or 'pareto_stable'


class AdaroundResult(TypedDict):
    """Output of Adaround weight rounding optimisation."""
    model: Any                        # nn.Module with optimised weights
    alpha_stats: Dict[str, Dict]      # {param_name: {mean, min, max, n_learned, n_total}}
    mse_before: float                 # MSE before Adaround
    mse_after: float                  # MSE after Adaround
    mse_reduction: float              # Percentage improvement
    time_seconds: float               # Wall-clock time


class QATResult(TypedDict):
    """Output of QAT warmstart fine-tuning."""
    model: Any                        # nn.Module with fine-tuned weights
    train_accuracy: List[float]       # Per-epoch training accuracy
    val_accuracy: List[float]         # Per-epoch validation accuracy
    train_loss: List[float]           # Per-epoch training loss
    best_epoch: int                   # Epoch with best val_acc
    final_val_acc: float              # Best validation accuracy (%)
    time_seconds: float               # Wall-clock time


class ParetoAnalysisResult(TypedDict):
    """Output of Phase 2 Pareto front analysis."""
    solutions_ranked: List[Dict]      # Solutions sorted by accuracy (best first)
    metrics: Dict[str, float]         # HV, spacing, coverage, etc.
    extreme_solutions: Dict           # {best_acc, best_ebops, balanced}
    compression_ratios: List[float]   # Per-solution
    plot_paths: Dict[str, str]        # {plot_name: file_path}
    summary_report: str               # Markdown format


class XAIResult(TypedDict, total=False):
    """Output of Phase 3 XAI explainability analysis."""
    grad_cam_paths: Dict[str, List[str]]  # {model_id: [img_paths]}
    shap_paths: Dict[str, List[str]]      # {model_id: [html/png paths]}
    comparison_grid: str                  # Path to combined grid image
    consistency_scores: Dict[str, float]  # {model_id: Pearson correlation vs FP32}
    report: str                           # Markdown summary
    # Per-(technique, sample) predictions surfaced by the comparison
    # matrix and per-image captions. Each entry is a dict with
    # pred_idx, pred_name, confidence, gt_idx, gt_name, correct.
    predictions: Dict[str, List[Dict[str, Any]]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QuantizationMethod(Enum):
    """Supported quantization methods."""
    PTQ = "ptq"
    QAT = "qat"
    GPTQ = "gptq"
    SMOOTHQUANT = "smoothquant"
    AWQ = "awq"


class SensitivityTier(Enum):
    """Layer sensitivity tiers based on Hessian analysis."""
    HIGH = "high"       # >66th percentile → force INT8
    MEDIUM = "medium"   # 33-66th percentile → NSGA-II decides
    LOW = "low"         # <33rd percentile → allow INT4


class CalibrationStrategy(Enum):
    """Observer calibration strategy per layer position."""
    KL_DIVERGENCE = "kl_divergence"   # For input/output layers
    MSE = "mse"                        # For intermediate layers


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration Dataclasses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class HyperparameterSet:
    """Hyperparameters for all quantization and optimization phases."""

    # General
    seed: int = 42
    device: str = "auto"  # 'auto', 'cuda', 'cpu', 'mps'

    # Calibration
    calibration_batches: int = 20
    calibration_strategy_io: CalibrationStrategy = CalibrationStrategy.KL_DIVERGENCE
    calibration_strategy_intermediate: CalibrationStrategy = CalibrationStrategy.MSE

    # Hessian / Clustering
    hessian_batches: int = 20
    cluster_high_percentile: float = 0.66
    cluster_low_percentile: float = 0.33

    # NSGA-II
    nsga_population_size: int = 8
    nsga_generations: int = 20
    nsga_crossover_prob: float = 0.9
    nsga_mutation_prob: float = 0.1

    # FITCompress
    fit_low_percentile: float = 0.25   # Below this → INT4 (compress)
    fit_high_percentile: float = 0.75  # Above this → INT8 (preserve)

    # Adaround
    adaround_epochs: int = 100
    adaround_lr: float = 0.0001
    adaround_reg_param: float = 0.01  # lambda for entropy regularizer

    # QAT
    qat_epochs: int = 5
    qat_lr: float = 0.001
    qat_momentum: float = 0.9
    qat_weight_decay: float = 1e-4
    qat_early_stop_patience: int = 3  # Stop if no val improvement for N epochs

    # GPTQ
    gptq_block_size: int = 128
    gptq_percdamp: float = 0.01

    # SmoothQuant
    smoothquant_alpha: float = 0.5

    # AWQ
    awq_group_size: int = 128

    # XAI / Explainability
    xai_num_images: int = 5               # Number of test images to explain
    xai_grad_cam_alpha: float = 0.4       # Heatmap overlay transparency
    xai_shap_n_samples: int = 50          # SHAP background samples
    xai_plot_dpi: int = 150               # Plot resolution

    # Evaluation / Metrics
    eval_primary_accuracy: str = "top1"   # Public reporting stays top-1 only
    nsga_accuracy_objective: str = "top1" # 'top1' or 'top5' — NSGA-II objective
    latency_warmup_runs: int = 10         # Warmup passes before timing
    latency_measure_runs: int = 50        # Timed passes for statistics
    latency_batch_size: int = 1           # Batch size for latency benchmark
    hardware_report_path: str = ""        # Path to synthesis report (JSON/CSV)
    use_latency_in_pareto: bool = False   # Add latency as 3rd Pareto objective


@dataclass
class QuantizationConfig:
    """Master configuration for the NeuroQuant framework."""

    # ── Model ──
    model_name: str = "mobilenetv2"
    model_path: Optional[str] = None  # Path to saved model weights
    model_class: Optional[str] = None  # Fully qualified class name
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (3, 32, 32)

    # ── Dataset ──
    dataset_name: str = "cifar10"
    dataset_path: str = "./data"
    batch_size: int = 128
    num_workers: int = 4

    # ── Output ──
    output_dir: str = "./artifacts"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "neuroquant_v2"

    # ── Methods to run ──
    methods: List[QuantizationMethod] = field(
        default_factory=lambda: [
            QuantizationMethod.PTQ,
            QuantizationMethod.QAT,
            QuantizationMethod.GPTQ,
            QuantizationMethod.SMOOTHQUANT,
            QuantizationMethod.AWQ,
        ]
    )

    # ── Bitwidths ──
    supported_bitwidths: List[int] = field(default_factory=lambda: [4, 8])
    io_layer_bitwidth: int = 8  # Enforce INT8 for input/output layers

    # ── Hyperparameters ──
    hyperparams: HyperparameterSet = field(default_factory=HyperparameterSet)

    # ── Phase control ──
    run_phases: List[str] = field(
        default_factory=lambda: [
            "phase_0_preparation",
            "phase_1a_hessian_clustering",
            "phase_1b_fitcompress",
            "phase_1c_nsga_search",
            "phase_1d_adaround",
            "phase_1e_qat",
            "phase_1f_gptq_smooth_awq",
            "phase_2_pareto",
            "phase_3_xai",
            "phase_4_mlflow",
        ]
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "QuantizationConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "QuantizationConfig":
        """Load configuration from a JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "QuantizationConfig":
        """Build a QuantizationConfig from a nested dict (YAML/JSON)."""
        config = cls()

        # Model
        model = data.get("model", {})
        if "name" in model:
            config.model_name = model["name"]
        if "path" in model and model["path"]:
            config.model_path = model["path"]
        if "class" in model and model["class"]:
            config.model_class = model["class"]
        if "num_classes" in model:
            config.num_classes = model["num_classes"]
        if "input_shape" in model:
            config.input_shape = tuple(model["input_shape"])

        # Dataset
        ds = data.get("dataset", {})
        if "name" in ds:
            config.dataset_name = ds["name"]
        if "path" in ds:
            config.dataset_path = ds["path"]
        if "batch_size" in ds:
            config.batch_size = ds["batch_size"]
        if "num_workers" in ds:
            config.num_workers = ds["num_workers"]

        # Output
        out = data.get("output", {})
        if "dir" in out:
            config.output_dir = out["dir"]
        if "mlflow_tracking_uri" in out:
            config.mlflow_tracking_uri = out["mlflow_tracking_uri"]
        if "experiment_name" in out:
            config.experiment_name = out["experiment_name"]

        # Methods
        methods_raw = data.get("methods", [])
        if methods_raw:
            config.methods = [QuantizationMethod(m) for m in methods_raw]

        # Bitwidths
        bw = data.get("bitwidths", {})
        if "supported" in bw:
            config.supported_bitwidths = bw["supported"]
        if "io_layer" in bw:
            config.io_layer_bitwidth = bw["io_layer"]

        # Hyperparameters
        hp_data = data.get("hyperparams", {})
        hp = config.hyperparams
        for key, value in hp_data.items():
            if hasattr(hp, key) and value is not None:
                setattr(hp, key, value)

        # Phases
        phases_raw = data.get("phases", [])
        if phases_raw:
            config.run_phases = phases_raw

        return config

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._to_dict()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _to_dict(self) -> Dict[str, Any]:
        """Serialize to a nested dict matching the YAML structure."""
        return {
            "model": {
                "name": self.model_name,
                "path": self.model_path,
                "class": self.model_class,
                "num_classes": self.num_classes,
                "input_shape": list(self.input_shape),
            },
            "dataset": {
                "name": self.dataset_name,
                "path": self.dataset_path,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
            },
            "output": {
                "dir": self.output_dir,
                "mlflow_tracking_uri": self.mlflow_tracking_uri,
                "experiment_name": self.experiment_name,
            },
            "methods": [m.value for m in self.methods],
            "bitwidths": {
                "supported": self.supported_bitwidths,
                "io_layer": self.io_layer_bitwidth,
            },
            "hyperparams": {
                k: v for k, v in self.hyperparams.__dict__.items()
                if not k.startswith("_")
            },
            "phases": self.run_phases,
        }

    def validate(self) -> None:
        """Validate configuration consistency and raise on errors."""
        errors = []

        # ── Model ──
        if not self.model_name and not self.model_class:
            errors.append(
                "Either model_name or model_class must be set."
            )

        # ── Dataset ──
        if not self.dataset_name:
            errors.append("dataset_name must not be empty.")

        # ── Input shape ──
        if len(self.input_shape) != 3:
            errors.append(
                f"input_shape must be (C, H, W) with 3 dims, "
                f"got {self.input_shape} ({len(self.input_shape)} dims)."
            )
        else:
            c, h, w = self.input_shape
            if c not in (1, 3, 4):
                errors.append(
                    f"input_shape channels={c} unexpected "
                    f"(expected 1, 3, or 4)."
                )
            if h < 8 or w < 8:
                errors.append(
                    f"input_shape spatial dims ({h}x{w}) too small "
                    f"(minimum 8x8)."
                )

        # ── Num classes ──
        if self.num_classes < 2:
            errors.append("num_classes must be >= 2.")

        # ── Batch size ──
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1.")

        # ── Bitwidths ──
        if not self.supported_bitwidths:
            errors.append("supported_bitwidths must not be empty.")
        if self.io_layer_bitwidth not in [4, 8, 16, 32]:
            errors.append(f"io_layer_bitwidth={self.io_layer_bitwidth} is invalid.")

        # ── Device ──
        hp = self.hyperparams
        if hp.device not in ("auto", "cpu", "cuda", "mps"):
            errors.append(
                f"device='{hp.device}' invalid. "
                f"Use 'auto', 'cuda', 'cpu', or 'mps'."
            )

        # ── NSGA-II ──
        if hp.nsga_population_size < 4:
            errors.append("nsga_population_size must be >= 4.")
        if hp.nsga_generations < 1:
            errors.append("nsga_generations must be >= 1.")

        # ── Clustering ──
        if not (0 < hp.cluster_low_percentile < hp.cluster_high_percentile < 1):
            errors.append(
                "Cluster percentiles must satisfy 0 < low < high < 1."
            )

        # ── Learning rates ──
        if hp.adaround_lr <= 0:
            errors.append("adaround_lr must be > 0.")
        if hp.qat_lr <= 0:
            errors.append("qat_lr must be > 0.")

        # ── Phases ──
        valid_phases = {name for name, _ in [
            ("phase_0_preparation", ""),
            ("phase_1a_hessian_clustering", ""),
            ("phase_1b_fitcompress", ""),
            ("phase_1c_nsga_search", ""),
            ("phase_1d_adaround", ""),
            ("phase_1e_qat", ""),
            ("phase_1f_gptq_smooth_awq", ""),
            ("phase_2_pareto", ""),
            ("phase_3_xai", ""),
            ("phase_4_mlflow", ""),
        ]}
        for phase in self.run_phases:
            if phase not in valid_phases:
                errors.append(
                    f"Unknown phase '{phase}'. "
                    f"Valid: {sorted(valid_phases)}"
                )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  " + "\n  ".join(errors)
            )
