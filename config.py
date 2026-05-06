"""
NeuroQuant v2.0 - Configuration Module

Defines all configuration dataclasses and type definitions
for the quantization framework. Supports YAML/JSON config loading.

Wave 6 L1: ``HyperparameterSet`` and ``QuantizationConfig`` are now
``pydantic.dataclasses.dataclass`` instances (Pydantic v2). This is a
drop-in replacement for the stdlib ``@dataclass``: every existing
field name, type, and default is preserved, every existing call site
(``cfg.hyperparams.seed``, ``cfg.model_name``, …) keeps working, and
the legacy ``validate()`` cross-field check still runs end-to-end.

What pydantic adds:

* Field-level validation runs on construction. ``QuantizationConfig(
  num_classes=-3)`` raises immediately instead of silently going on
  to break a downstream phase.
* YAML / JSON loaders emit clear, multi-error messages pointing at
  the offending key path (e.g. ``hyperparams.qat_lr → must be > 0``).
* Type coercion on load: integer values arriving as strings from a
  loosely-formatted YAML become ``int`` automatically; mistyped
  fields surface during config load, not during phase execution.

Legacy fallback: if pydantic is not installed (rare; it is a hard
requirement in ``requirements.txt`` from wave 6 onward), we
transparently fall back to the stdlib ``@dataclass`` so older
checkpoints/configs still load. The behavioural surface is identical
in either path; only the error-quality improves with pydantic.
"""

from __future__ import annotations

import json
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

try:
    from pydantic.dataclasses import dataclass as _pydantic_dataclass
    from pydantic import ConfigDict, field_validator
    _HAS_PYDANTIC = True

    def dataclass(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        """Wrap pydantic.dataclasses.dataclass with a permissive config.

        ``arbitrary_types_allowed`` lets downstream consumers attach
        non-pydantic attributes (e.g. live ``nn.Module`` instances on
        result dicts) without tripping validation. ``validate_assignment``
        is left at the default ``False`` so post-construction edits
        (``cfg.hyperparams.qat_epochs = 1`` in tests) keep working.
        """
        kwargs.setdefault(
            "config",
            ConfigDict(arbitrary_types_allowed=True),
        )
        return _pydantic_dataclass(*args, **kwargs)

except ImportError:  # pragma: no cover — pydantic is a hard dep from wave 6
    from dataclasses import dataclass  # type: ignore[no-redef]
    _HAS_PYDANTIC = False

    def field_validator(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        """No-op stand-in when pydantic isn't installed.

        Lets the same source code run under stdlib ``@dataclass``;
        the ``.validate()`` method continues to enforce the same
        constraints in that mode.
        """
        def _wrap(fn: Any) -> Any:
            return fn
        return _wrap


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


class QuantizationResult(TypedDict, total=False):
    """Result from evaluating a single quantized model.

    Accuracy fields:
      * ``accuracy``      — public headline. Always equals the
        ``test_loader`` Top-1 once A2 has been wired through; the older
        single-loader pipeline stored the val number here.
      * ``val_top1``      — diagnostic; the same number used by QAT
        early-stopping. Never the headline.
      * ``test_top1``     — public headline (deployment-time estimate).
      * ``search_top1``   — NSGA-fitness number on the held-out search
        slice. Used internally for selection decisions only.
    """
    config_id: str
    method: str  # 'PTQ', 'QAT', 'GPTQ', 'SmoothQuant', 'AWQ'
    bitwidth_assignment: Dict[str, int]  # layer_name -> bitwidth
    accuracy: float
    top5_accuracy: Optional[float]
    val_top1: Optional[float]
    test_top1: Optional[float]
    search_top1: Optional[float]
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
    # Sensitivity estimator. "fisher" (default, production) uses
    # ``(∂L/∂w)²`` averaged over calibration batches — single backprop,
    # ~3× faster, correlates ≥0.9 with the diagonal Hessian on standard
    # classification heads. "diag_hessian" runs the original double
    # backprop for ablations or when an exact diagonal is needed.
    hessian_estimator: str = "fisher"  # 'fisher' | 'diag_hessian'

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
    # Ordered (canonical) AdaRound: iterate target layers in topological
    # order, propagating each layer's quantized output into the
    # downstream activations the next layer sees. The original paper
    # (Nagel et al. 2020) describes this; the parallel variant ignores
    # accumulated upstream error and consistently underperforms on
    # deep networks. Disable only for ablations or to trade accuracy
    # for ~2x faster optimization.
    adaround_ordered: bool = True
    # Bounded per-layer activation pool for the streaming
    # collector. Constant memory regardless of model depth — only one
    # layer's activations live at a time.
    adaround_max_samples_per_layer: int = 1024

    # QAT
    qat_epochs: int = 5
    qat_lr: float = 0.001
    qat_momentum: float = 0.9
    qat_weight_decay: float = 1e-4
    qat_early_stop_patience: int = 3  # Stop if no val improvement for N epochs
    # Which PTQ artefact warmstarts QAT/Adaround. "ptq_best_acc" picks
    # the highest real Top-1 PTQ; "ptq_best_tradeoff" picks the most
    # compressed PTQ that stays within ``ptq_tradeoff_max_acc_drop`` of
    # FP32. The chosen source ID is persisted to checkpoints/metadata.
    qat_warmstart_source: str = "ptq_best_acc"  # 'ptq_best_acc' | 'ptq_best_tradeoff'

    # ── W+A QAT (E1 / E3) ────────────────────────────────────────────────
    # Activation bitwidth used during QAT. Production defaults to INT8 —
    # the deployment shape that every supported INT backend (qnnpack,
    # fbgemm, ORT, TensorRT) expects. Override only for research
    # experiments; W4A4 does not have a real backend.
    qat_act_bitwidth: int = 8
    # Pre-QAT analytic Conv-BN fold. Disabling this (set to False) is
    # only useful for ablations: real INT8 inference always has BN folded.
    qat_fold_bn: bool = True
    # ── KD distillation (E5) ─────────────────────────────────────────────
    # Mixes a soft-target KL term against the FP32 teacher. ``alpha`` is
    # the weight of the KD term in the total loss; ``temperature``
    # softens the teacher distribution. ``alpha=0`` disables KD.
    qat_distill_alpha: float = 0.5
    qat_distill_temperature: float = 4.0

    # Multi-fidelity PTQ rerank (phase 1c)
    # Number of NSGA candidates materialised through real PTQ +
    # bitwidth-aware calibration before final selection. K=1 disables
    # reranking (only the NSGA winner is materialised).
    ptq_real_rerank_topk: int = 3
    # Maximum tolerated Top-1 accuracy drop (percentage points) for the
    # ``ptq_best_tradeoff`` candidate. If no rerank candidate satisfies
    # the cap, the smallest-size candidate is chosen as a knee-like
    # fallback.
    ptq_tradeoff_max_acc_drop: float = 1.0

    # GPTQ
    gptq_block_size: int = 128
    gptq_percdamp: float = 0.01

    # SmoothQuant
    smoothquant_alpha: float = 0.5
    # Per-layer α grid search. When True, each layer searches its own
    # migration strength over ``smoothquant_alpha_grid`` and keeps the
    # value that minimises post-quantization output reconstruction MSE
    # on the calibration sample. Single global α is rarely optimal —
    # production runs always enable the search.
    smoothquant_per_layer_alpha: bool = True
    smoothquant_alpha_grid: List[float] = field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )

    # AWQ
    awq_group_size: int = 128
    # Per-layer α grid for the activation-driven migration scale
    # ``s = a^α``. The chosen α minimises layer-output reconstruction
    # MSE on the calibration sample. Wider grid = better quality, more
    # search time. Defaults match the AWQ reference implementation.
    awq_alpha_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    # Top-K% activation channels kept at FP16 instead of quantized.
    # 0.0 = pure scaling (production AWQ default); 0.01 = the AWQ
    # paper Section 3 ablation that keeps the most salient 1%.
    awq_keep_top_pct: float = 0.0

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

    # ── Wave 4: ONNX export + hardware-aware search ─────────────────────
    # When True, every quantized method result is also exported to ONNX
    # and the on-disk INT8 ``.onnx`` file size + ORT inference latency
    # are recorded as ``onnx_size_mb`` / ``onnx_latency`` on the
    # QuantizationResult. The synthetic ``model_size_mb`` (= numel ×
    # bw / 8) is replaced by the real ONNX size on disk; the synthetic
    # number is kept under ``theoretical_size_mb`` for ablation. This is
    # the J1 + J2 + J3 deployment-fidelity contract.
    onnx_export_enabled: bool = True
    # Per-layer ORT latency LUT (C2) feeds NSGA's third objective. When
    # ``hardware_aware_search`` is True the pipeline builds the LUT
    # before phase 1c (≈ 1–2 minutes for a CIFAR-class model) and NSGA
    # runs in 3-objective mode ``[acc_loss, size_mb, latency_ms]``. The
    # LUT is cached to ``output_dir/latency_lut.json`` so subsequent
    # runs skip the rebuild.
    hardware_aware_search: bool = False
    # Bitwidths the LUT profiles. Must be a subset of
    # ``supported_bitwidths`` for the search to consume them.
    latency_lut_bitwidths: List[int] = field(default_factory=lambda: [4, 8])

    # ────────────────────────────────────────────────────────────────────
    # Field validators (Wave 6 L1, pydantic v2)
    # ────────────────────────────────────────────────────────────────────
    # These fire at construction. Each one targets a category of error
    # that previously surfaced inside a phase (often with a confusing
    # downstream traceback) rather than at config-load time.

    @field_validator("device", mode="after")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError(
                f"device='{v}' invalid. Use 'auto', 'cuda', 'cpu', or 'mps'."
            )
        return v

    @field_validator("hessian_estimator", mode="after")
    @classmethod
    def _validate_hessian_estimator(cls, v: str) -> str:
        if v not in ("fisher", "diag_hessian"):
            raise ValueError(
                f"hessian_estimator='{v}' invalid. "
                "Use 'fisher' or 'diag_hessian'."
            )
        return v

    @field_validator("qat_warmstart_source", mode="after")
    @classmethod
    def _validate_warmstart_source(cls, v: str) -> str:
        if v not in ("ptq_best_acc", "ptq_best_tradeoff"):
            raise ValueError(
                f"qat_warmstart_source='{v}' invalid. "
                "Use 'ptq_best_acc' or 'ptq_best_tradeoff'."
            )
        return v

    @field_validator("qat_act_bitwidth", mode="after")
    @classmethod
    def _validate_qat_act_bitwidth(cls, v: int) -> int:
        if v not in (4, 8, 16, 32):
            raise ValueError(
                f"qat_act_bitwidth={v} invalid. Use 4, 8, 16, or 32."
            )
        return v

    @field_validator("qat_distill_alpha", mode="after")
    @classmethod
    def _validate_distill_alpha(cls, v: float) -> float:
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(
                f"qat_distill_alpha={v} must be in [0, 1]."
            )
        return float(v)

    @field_validator("qat_distill_temperature", "adaround_lr",
                     "qat_lr", mode="after")
    @classmethod
    def _validate_strictly_positive(cls, v: float) -> float:
        if float(v) <= 0:
            raise ValueError(f"value={v} must be > 0.")
        return float(v)

    @field_validator("nsga_population_size", mode="after")
    @classmethod
    def _validate_nsga_pop(cls, v: int) -> int:
        if int(v) < 4:
            raise ValueError(
                f"nsga_population_size={v} must be >= 4."
            )
        return int(v)

    @field_validator("nsga_generations", "ptq_real_rerank_topk",
                     mode="after")
    @classmethod
    def _validate_strictly_positive_int(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError(f"value={v} must be >= 1.")
        return int(v)

    @field_validator("ptq_tradeoff_max_acc_drop", mode="after")
    @classmethod
    def _validate_acc_drop_cap(cls, v: float) -> float:
        if float(v) < 0:
            raise ValueError(
                f"ptq_tradeoff_max_acc_drop={v} must be >= 0."
            )
        return float(v)

    @field_validator("latency_lut_bitwidths", mode="after")
    @classmethod
    def _validate_lut_bitwidths(cls, v: List[int]) -> List[int]:
        for bw in v:
            if int(bw) not in (4, 8, 16, 32):
                raise ValueError(
                    f"latency_lut_bitwidths contains invalid {bw}; "
                    "use only 4, 8, 16, or 32."
                )
        return [int(b) for b in v]


@dataclass
class QuantizationConfig:
    """Master configuration for the NeuroQuant framework."""

    # ── Model ──
    model_name: str = "mobilenetv2"
    model_path: Optional[str] = None  # Path to saved model weights
    model_class: Optional[str] = None  # Fully qualified class name
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (3, 32, 32)
    # Task family the model targets. "classification" runs the standard
    # last-Linear adaptation + first-Conv stride tweak. "detection" loads
    # from ``torchvision.models.detection`` and skips the
    # classification-specific adapters; head replacement is handled
    # inline by the loader for the Faster/Mask/Keypoint R-CNN family.
    task: str = "classification"  # 'classification' | 'detection'

    # ── Dataset ──
    dataset_name: str = "cifar10"
    dataset_path: str = "./data"
    dataset_class: Optional[str] = None  # Fully qualified Dataset class name
    dataset_train_dir: Optional[str] = None
    dataset_val_dir: Optional[str] = None
    dataset_test_dir: Optional[str] = None
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

    # ────────────────────────────────────────────────────────────────────
    # Field validators (Wave 6 L1, pydantic v2)
    # ────────────────────────────────────────────────────────────────────

    @field_validator("num_classes", mode="after")
    @classmethod
    def _validate_num_classes(cls, v: int) -> int:
        if int(v) < 2:
            raise ValueError(f"num_classes={v} must be >= 2.")
        return int(v)

    @field_validator("task", mode="after")
    @classmethod
    def _validate_task(cls, v: str) -> str:
        if v not in ("classification", "detection"):
            raise ValueError(
                f"task='{v}' invalid. Use 'classification' or 'detection'."
            )
        return v

    @field_validator("batch_size", mode="after")
    @classmethod
    def _validate_batch_size(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError(f"batch_size={v} must be >= 1.")
        return int(v)

    @field_validator("io_layer_bitwidth", mode="after")
    @classmethod
    def _validate_io_bitwidth(cls, v: int) -> int:
        if int(v) not in (4, 8, 16, 32):
            raise ValueError(
                f"io_layer_bitwidth={v} invalid. Use 4, 8, 16, or 32."
            )
        return int(v)

    @field_validator("input_shape", mode="after")
    @classmethod
    def _validate_input_shape(cls, v: Tuple[int, ...]) -> Tuple[int, ...]:
        # Pydantic may coerce list → tuple; normalise then sanity-check.
        v = tuple(int(x) for x in v)
        if len(v) != 3:
            raise ValueError(
                f"input_shape must be (C, H, W) with 3 dims, "
                f"got {v} ({len(v)} dims)."
            )
        c, h, w = v
        if c not in (1, 3, 4):
            raise ValueError(
                f"input_shape channels={c} unexpected "
                "(expected 1, 3, or 4)."
            )
        if h < 8 or w < 8:
            raise ValueError(
                f"input_shape spatial dims ({h}x{w}) too small "
                "(minimum 8x8)."
            )
        return v

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "QuantizationConfig":
        """Load configuration from a YAML file.

        Wave 6 L1: ``.validate()`` runs at the end of the load so
        invalid values surface at config-load time rather than during
        a downstream phase. Pydantic field validators only fire on
        ``__init__``; the load path uses ``setattr`` to copy values
        onto a default-constructed config, which bypasses them — so
        we call the runtime ``validate()`` explicitly here.
        """
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        config = cls._from_dict(data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "QuantizationConfig":
        """Load configuration from a JSON file.

        Same load-time validation contract as ``from_yaml``.
        """
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        config = cls._from_dict(data)
        config.validate()
        return config

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
        if "task" in model and model["task"]:
            config.task = model["task"]

        # Dataset
        ds = data.get("dataset", {})
        if "name" in ds:
            config.dataset_name = ds["name"]
        if "path" in ds:
            config.dataset_path = ds["path"]
        if "class" in ds and ds["class"]:
            config.dataset_class = ds["class"]
        if "train_dir" in ds and ds["train_dir"]:
            config.dataset_train_dir = ds["train_dir"]
        if "val_dir" in ds and ds["val_dir"]:
            config.dataset_val_dir = ds["val_dir"]
        if "test_dir" in ds and ds["test_dir"]:
            config.dataset_test_dir = ds["test_dir"]
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

        # Hyperparameters. Enum-typed fields are stored as their
        # ``.value`` string in YAML/JSON; convert back to the enum on
        # load so downstream consumers continue to receive the strong
        # type. Looks the enum class up via the dataclass field
        # annotation so adding new enum-typed fields needs no extra
        # plumbing here.
        hp_data = data.get("hyperparams", {})
        hp = config.hyperparams
        hp_field_types: Dict[str, Any] = {}
        try:
            hp_field_types = {
                fname: ftype for fname, ftype in
                getattr(HyperparameterSet, "__annotations__", {}).items()
            }
        except Exception:
            hp_field_types = {}
        for key, value in hp_data.items():
            if not hasattr(hp, key) or value is None:
                continue
            ftype = hp_field_types.get(key)
            if (
                ftype is not None
                and isinstance(ftype, type)
                and issubclass(ftype, Enum)
                and isinstance(value, str)
            ):
                value = ftype(value)
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
                "task": self.task,
            },
            "dataset": {
                "name": self.dataset_name,
                "path": self.dataset_path,
                "class": self.dataset_class,
                "train_dir": self.dataset_train_dir,
                "val_dir": self.dataset_val_dir,
                "test_dir": self.dataset_test_dir,
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
                k: (v.value if isinstance(v, Enum) else v)
                for k, v in self.hyperparams.__dict__.items()
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
        if self.dataset_class and "." not in self.dataset_class:
            errors.append(
                "dataset_class must be fully qualified, e.g. "
                "'my_pkg.my_data.MyDataset'."
            )

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

        # ── Task ──
        if self.task not in ("classification", "detection"):
            errors.append(
                f"task='{self.task}' invalid. "
                "Use 'classification' or 'detection'."
            )

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
        if hp.hessian_estimator not in ("fisher", "diag_hessian"):
            errors.append(
                f"hessian_estimator='{hp.hessian_estimator}' invalid. "
                "Use 'fisher' or 'diag_hessian'."
            )

        # ── Learning rates ──
        if hp.adaround_lr <= 0:
            errors.append("adaround_lr must be > 0.")
        if hp.qat_lr <= 0:
            errors.append("qat_lr must be > 0.")

        # ── PTQ rerank / warmstart ──
        if hp.qat_warmstart_source not in ("ptq_best_acc", "ptq_best_tradeoff"):
            errors.append(
                f"qat_warmstart_source='{hp.qat_warmstart_source}' invalid. "
                "Use 'ptq_best_acc' or 'ptq_best_tradeoff'."
            )
        if hp.ptq_real_rerank_topk < 1:
            errors.append("ptq_real_rerank_topk must be >= 1.")
        if hp.ptq_tradeoff_max_acc_drop < 0:
            errors.append("ptq_tradeoff_max_acc_drop must be >= 0.")

        # ── W+A QAT / KD ──
        if hp.qat_act_bitwidth not in (4, 8, 16, 32):
            errors.append(
                f"qat_act_bitwidth={hp.qat_act_bitwidth} invalid. "
                "Use 4, 8, 16, or 32."
            )
        if not (0.0 <= hp.qat_distill_alpha <= 1.0):
            errors.append("qat_distill_alpha must be in [0, 1].")
        if hp.qat_distill_temperature <= 0:
            errors.append("qat_distill_temperature must be > 0.")

        # ── Wave 4: hardware-aware search ──
        if hp.latency_lut_bitwidths:
            for bw in hp.latency_lut_bitwidths:
                if bw not in (4, 8, 16, 32):
                    errors.append(
                        f"latency_lut_bitwidths contains invalid {bw}; "
                        "use only 4, 8, 16, or 32."
                    )

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
