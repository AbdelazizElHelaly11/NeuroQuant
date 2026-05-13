# API Reference

This page is generated automatically from the docstrings inside
`neuroquant/` by [`mkdocstrings`](https://mkdocstrings.github.io/). Edit
a class or method in the source tree and the change shows up here the
next time `mkdocs build` runs.

!!! tip "Two valid import paths"

    Every symbol below is reachable from the flat package root, e.g.

    ```python
    from neuroquant import PTQQuantizer
    ```

    The deep path (`neuroquant.quantization.ptq.PTQQuantizer`) is what
    mkdocstrings uses to find the source — both forms point at the
    same object.

---

## Configuration

### `QuantizationConfig`

The single configuration dataclass used by the whole framework. Pass a
hand-built `QuantizationConfig()` to a quantizer constructor, or pass
`None` to use defaults.

::: neuroquant.config.QuantizationConfig
    options:
      show_bases: false
      members_order: source

---

## Quantizers

### `BaseQuantizer`

Abstract base of every quantizer; defines the shared
`evaluate / save_model / quantize_tensor / collect_layer_inputs` API.

::: neuroquant.quantization.base.BaseQuantizer
    options:
      show_bases: true

### `PTQQuantizer`

Post-training quantization with KL-divergence calibration on I/O
layers and MSE on intermediate layers. Supports both the pipeline
form (`quantize(bitwidth_assignment_dict)`) and the standalone form
(`quantize(calibration_loader, bitwidth=…)`).

::: neuroquant.quantization.ptq.PTQQuantizer
    options:
      show_bases: false

### `AWQQuantizer`

Activation-aware Weight Quantization with cluster-amortized α-search
and the F2 production-corrected input-scale wrappers.

::: neuroquant.quantization.awq.AWQQuantizer
    options:
      show_bases: false

### `GPTQQuantizer`

Layer-wise quantization using approximate Hessian inverse; optimal
column-wise rounding.

::: neuroquant.quantization.gptq.GPTQQuantizer
    options:
      show_bases: false

### `SmoothQuantQuantizer`

Per-channel migration that balances activation outliers into the
weights before quantization.

::: neuroquant.quantization.smoothquant.SmoothQuantQuantizer
    options:
      show_bases: false

### `SmoothQuantGPTQQuantizer`

The combined F4 method: SmoothQuant migration then GPTQ on the
smoothed weights.

::: neuroquant.quantization.smoothquant_gptq.SmoothQuantGPTQQuantizer
    options:
      show_bases: false

### `AdaroundOptimizer`

Learned weight rounding via the stretched-sigmoid α parameterisation
from Nagel et al. (2020).

::: neuroquant.quantization.adaround.AdaroundOptimizer
    options:
      show_bases: false

### `QATTrainer`

Quantization-aware training with knowledge distillation against the
FP32 teacher.

::: neuroquant.quantization.qat.QATTrainer
    options:
      show_bases: false

---

## Multi-objective Search

### `LayerClusterer`

Hessian / Fisher per-layer sensitivity estimator + 3-tier
(HIGH / MEDIUM / LOW) bucketing used to shape the NSGA-II search
space.

::: neuroquant.quantization.hessian_clustering.LayerClusterer
    options:
      show_bases: false

### `NSGAIIClusterSearch`

Surrogate-Assisted NSGA-II (BRP-NAS / OFA-style). 2-objective
(`acc_loss`, `size`) or 3-objective (`acc_loss`, `size`, `latency`)
mode depending on whether a per-layer ORT latency LUT is supplied.

::: neuroquant.quantization.nsga_ii_search.NSGAIIClusterSearch
    options:
      show_bases: false

### `AccuracySurrogate`

The XGBoost / GradientBoosting model that ranks candidates inside
NSGA-II so a single population can scan thousands of configs per
generation instead of dozens.

::: neuroquant.quantization.surrogate.AccuracySurrogate
    options:
      show_bases: false

---

## Explainability (XAI)

### `XAIGenerator`

Orchestrates Grad-CAM + SHAP across FP32 baseline and quantized
models, computes consistency scores against FP32 attention, and
produces a comparison matrix that shows each technique's prediction
per sample. Task-aware — dispatches on classification logits,
detection score lists, *and* segmentation `OrderedDict({"out": ...})`
outputs without any extra glue.

::: neuroquant.xai.explainability.XAIGenerator
    options:
      show_bases: false

### `GradCAMExplainer`

The standalone Grad-CAM implementation. Useful when you want one
heatmap from one model without spinning up the full XAI pipeline.

::: neuroquant.xai.explainability.GradCAMExplainer
    options:
      show_bases: false

### `SHAPExplainer`

SHAP-based feature importance, with a gradient×input fallback for
detection / segmentation models (where `shap.GradientExplainer`
doesn't apply).

::: neuroquant.xai.explainability.SHAPExplainer
    options:
      show_bases: false

---

## Pareto Analysis & Visualisation

### `ParetoAnalyzer`

Computes hypervolume, knee point, extreme solutions, compression
ratios. Consumes the `ParetoFront` produced by Phase 1c.

::: neuroquant.visualization.pareto_analysis.ParetoAnalyzer
    options:
      show_bases: false

### `ParetoVisualizer`

Renders the scatter / 3D / bitwidth-distribution / metrics-table
plots used in the HTML report and the `pareto_summary.json`.

::: neuroquant.visualization.pareto_analysis.ParetoVisualizer
    options:
      show_bases: false

### Error attribution helpers

::: neuroquant.visualization.error_attribution.compute_layer_errors
    options:
      heading_level: 4

::: neuroquant.visualization.error_attribution.plot_error_attribution
    options:
      heading_level: 4

::: neuroquant.visualization.error_attribution.plot_error_comparison
    options:
      heading_level: 4

### Sensitivity plots

::: neuroquant.visualization.sensitivity.plot_sensitivity_heatmap
    options:
      heading_level: 4

::: neuroquant.visualization.sensitivity.plot_tier_distribution
    options:
      heading_level: 4

### HTML report generator

::: neuroquant.visualization.report.generate_html_report
    options:
      heading_level: 4

---

## Output Type Dispatch (XAI internals)

These three helpers are what makes Grad-CAM and SHAP task-agnostic.
They are exposed so advanced users can build their own attribution
flows against the same dispatch contract.

::: neuroquant.xai.explainability.infer_task_kind
    options:
      heading_level: 4

::: neuroquant.xai.explainability.compute_backward_target
    options:
      heading_level: 4

::: neuroquant.xai.explainability.predict_from_output
    options:
      heading_level: 4
