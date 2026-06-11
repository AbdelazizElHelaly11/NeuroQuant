# NeuroQuant v2.1

[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()
[![docs](https://img.shields.io/badge/docs-mkdocs--material-526CFE)](https://AbdelazizElHelaly11.github.io/NeuroQuant/)

**Production-grade neural-network quantization framework — multi-objective NSGA search, ONNX deployment fidelity, and built-in explainability across classification, detection, segmentation, regression, and HuggingFace NLP.**

NeuroQuant takes a pre-trained PyTorch model and produces deployable INT8 / mixed-precision artefacts that have been **measured, not estimated**, on the same runtime that ships in production. Every public number is the result of running a real quantized graph through ONNX Runtime — no synthetic shortcuts.

It has **two front doors**:

- a **CLI pipeline** that runs the whole flow from a single YAML, and
- a **flat Python library** (`from neuroquant import PTQQuantizer, …`) where every quantizer works config-free in three lines.

---

## Tasks supported

| Task             | Models                                                                   | Primary metric   | XAI                                       |
| ---------------- | ------------------------------------------------------------------------ | ---------------- | ----------------------------------------- |
| `classification` | any torchvision / timm / custom CNN **or** ViT                           | Top-1            | Grad-CAM (CNN) · Attention Rollout (ViT)  |
| `detection`      | `torchvision.models.detection` (Faster/Mask R-CNN, SSD, RetinaNet, FCOS) | Top-1 surrogate  | task-aware Grad-CAM                        |
| `segmentation`   | `torchvision.models.segmentation` (FCN, DeepLabV3, LRASPP)               | mIOU             | task-aware Grad-CAM                        |
| `regression`     | any `[B, K]`-output model                                                | RMSE / MAE / R²  | Grad-CAM / Attention Rollout              |
| `nlp`            | any HuggingFace model (`pip install neuroquant[nlp]`)                     | Top-1            | —                                         |

Vision Transformers are auto-detected and routed through **Attention Rollout** (Abnar & Zuidema, 2020) in Phase 3 — no Conv2d feature maps required.

---

## What it does

```
   FP32 PyTorch model  ─────►  9-phase pipeline  ─────►  INT8 .onnx + metrics

   P0   Prepare model + dataset, FP32 baseline (scored on the test split)
   P1a  Hessian / Fisher per-layer sensitivity + 3-tier clustering
   P1c  Surrogate-assisted NSGA multi-objective search (2- or 3-obj)
   P1d  AdaRound canonical input→output weight rounding
   P1e  Real W+A QAT with FP32-teacher knowledge distillation
   P1f  GPTQ + SmoothQuant + AWQ + SmoothQuant→GPTQ (INT4 + INT8 each)
   P2   Pareto analysis + plots
   P3   Grad-CAM / Attention Rollout + SHAP explainability
   P4   MLflow finalisation + HTML report + reproducibility manifest
```

The pipeline runs to completion in roughly a minute on CPU for a CIFAR-class model.

> Phase IDs are intentionally non-contiguous: the old `phase_1b` FITCompress
> seed was removed (Hessian-tier clustering + the in-loop surrogate cover
> the same role more cheaply), so legacy checkpoints still resolve.

---

## Why it is production-grade

This framework was built deliberately to avoid the "research prototype" failure modes that disqualify most academic quantization tooling from real deployment:

| Concern                       | What NeuroQuant does                                                                                                                            |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Real INT inference**        | Emits true static-INT8 ONNX graphs via `onnxruntime.quantization.quantize_static`, not FP32 simulation.                                         |
| **Real on-disk size**         | `model_size_mb` is the literal `.onnx` filesystem size, not `numel × bw / 8`. The synthetic estimate is kept as `theoretical_size_mb`.          |
| **Real latency**              | `latency_ms` is measured under ONNX Runtime on the same machine that will deploy the artefact.                                                  |
| **Hardware-aware search**     | The optional NSGA third objective sums a per-layer ORT latency LUT — every gene's latency cost is a real timing, not a FLOP estimate.           |
| **No leakage between splits** | Train / search / val / test are 80/10/10/test-set; NSGA fitness reads search, QAT early-stop reads val, and **the FP32 baseline _and_ every method headline read test** (so the comparison is apples-to-apples). |
| **Honest calibration**        | PTQ/GPTQ/AWQ/SmoothQuant, Fisher sensitivity, and AdaRound calibrate on an **eval-transform** view of the data — never randomly-augmented images. |
| **Strict determinism**        | `set_seed(strict=True)` enforces `CUBLAS_WORKSPACE_CONFIG`, `use_deterministic_algorithms`, `cudnn.deterministic`, plus a seeded loader + worker RNG. |
| **Safe checkpoints**          | All `torch.load(weights_only=True)`; pickle path is closed. Architectural wrappers persist as JSON manifests, resumable phase-by-phase.         |
| **Real W+A QAT**              | INT8 activations always; weight parametrisation via `torch.nn.utils.parametrize` (autograd-aware STE) with FP32-teacher KD.                     |
| **Validated config**          | Pydantic v2 dataclasses with field validators — bad values (including search-mode / α-strategy / bitwidth choices) fail at load, not deep in a phase. |

---

## Install

NeuroQuant is published on PyPI as **`neuroquant`** and supports Python 3.10+.

```bash
pip install neuroquant
neuroquant --help
```

GPU users install PyTorch from the CUDA wheel index first, then NeuroQuant on top:

```bash
# CUDA 12.1 example — check pytorch.org for your driver/CUDA combo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install neuroquant
```

From a source checkout (for development):

```bash
git clone https://github.com/AbdelazizElHelaly11/NeuroQuant
cd NeuroQuant
pip install -e ".[dev]"        # editable + dev extras
```

### Optional extras

The core install stays small; heavier dependencies are opt-in:

| Extra                | Adds                                       | Needed for                                                     |
| -------------------- | ------------------------------------------ | -------------------------------------------------------------- |
| `neuroquant[xai]`    | `shap`                                      | Phase 3 SHAP attribution (Grad-CAM works without it).          |
| `neuroquant[nlp]`    | `transformers`, `datasets`, `tokenizers`    | `task: nlp` and `dataset_name: hf:<name>` HuggingFace support. |
| `neuroquant[dev]`    | `ruff`, `build`, `pytest`, `pytest-cov`     | Linting, wheel builds, and the test suite.                     |
| `neuroquant[docs]`   | `mkdocs-material`, `mkdocstrings[python]`    | Building / serving the documentation site.                     |

Combine with comma syntax: `pip install neuroquant[xai,nlp]`.

---

## Quickstart — CLI

The `neuroquant` console script is installed on PATH (it is `neuroquant.cli:main`; `python -m neuroquant.cli` works too).

```bash
# Scaffold a fully-commented config.yaml in the current directory
neuroquant --init

# Full pipeline on the bundled config (CIFAR-10 + MobileNetV2)
neuroquant --config config.yaml --epochs 20

# Fast smoke (CPU, no training, first three phases)
neuroquant --config config.yaml --epochs 0 --device cpu \
  --phases phase_0_preparation phase_1a_hessian_clustering phase_1c_nsga_search

# Resume after interruption — skips phases that already have checkpoints
neuroquant --config config.yaml --epochs 20 --resume
```

Everything is written to `output_dir` (default `./artifacts/`):

```
artifacts/
├── checkpoints/                 # per-phase resume points (.json + .pth)
├── onnx/                        # FP32 + per-method INT8 .onnx files
├── pareto/                      # Pareto scatter / 3-D / bitwidth / table + JSON
├── error_attribution/          # per-method per-layer error PNGs
├── xai/                         # Grad-CAM / rollout heatmaps + comparison matrix
├── sensitivity_heatmap.png      # Phase 1a sensitivity + tier distribution
├── tier_distribution.png
├── pareto_summary.json
├── pipeline_report.txt
├── neuroquant_report.html       # self-contained, shareable HTML report
├── reproducibility_manifest.json
└── latency_lut.json             # only when hardware_aware_search=true
```

Open `artifacts/neuroquant_report.html` in any browser to read the run end-to-end — method table, Pareto plots, sensitivity heatmap, XAI grid, per-method error attribution, and deployment-fidelity caveats.

---

## Quickstart — library

Every quantizer accepts `config=None` and falls back to a fully-defaulted `QuantizationConfig()`, so you can drive it from a notebook without any YAML:

```python
from neuroquant import PTQQuantizer

ptq = PTQQuantizer(my_model)                       # config-free
q_model = ptq.quantize(calib_loader, bitwidth=4)   # your original model is untouched
```

The whole public surface imports flat:

```python
from neuroquant import (
    QuantizationConfig,
    PTQQuantizer, AWQQuantizer, GPTQQuantizer,
    SmoothQuantQuantizer, SmoothQuantGPTQQuantizer,
    QATTrainer, AdaroundOptimizer,
    NSGAIIClusterSearch, LayerClusterer, AccuracySurrogate,
    XAIGenerator, ParetoAnalyzer, ParetoVisualizer,
)
```

See the [library guide](docs/library_mode.md) for detection, segmentation, regression, NLP, and ViT examples.

---

## Configuration

All knobs live in [`config.yaml`](config.yaml) (regenerate it any time with `neuroquant --init`). Common overrides:

```yaml
model:
  name: resnet18              # any torchvision name (CNN or ViT)
  num_classes: 10
  input_shape: [3, 32, 32]
  task: classification        # classification | detection | segmentation | regression | nlp

dataset:
  name: cifar10               # cifar10 | cifar100 | imagefolder | synthetic | hf:<name>
  class: null                 # optional "pkg.module.MyDataset"
  train_dir: null             # optional ImageFolder split dirs
  val_dir: null
  test_dir: null
  batch_size: 128

methods: [ptq, qat, gptq, smoothquant, awq, smoothquant_gptq]
bitwidths:
  supported: [4, 8]
  io_layer: 8                 # force first/last layers to INT8

hyperparams:
  hardware_aware_search: true        # 3-objective NSGA [acc, size, ORT latency]
  onnx_export_enabled: true          # real INT8 ONNX size + ORT latency
  nsga_use_surrogate: true           # BRP-NAS / OFA-style accuracy surrogate
  qat_distill_alpha: 0.5             # KD with the FP32 teacher
  smoothquant_per_layer_alpha: true  # per-layer migration strength
  hessian_estimator: fisher          # ~3× faster than the diagonal Hessian
```

Pydantic field validators run at load time — invalid values surface immediately with the offending field path:

```text
ValueError: Configuration validation failed:
  num_classes must be >= 2.
```

---

## Quantization methods

| Method                | When to use                                                            | Module                                                                          |
| --------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **PTQ**               | Fast baseline; per-output-channel INT8/INT4 weights.                   | [`neuroquant/quantization/ptq.py`](neuroquant/quantization/ptq.py)               |
| **QAT**               | Best accuracy at INT8; real W+A training with FP32-teacher KD.         | [`neuroquant/quantization/qat.py`](neuroquant/quantization/qat.py)               |
| **GPTQ**              | Best accuracy at INT4 weights; Hessian-inverse optimal rounding.       | [`neuroquant/quantization/gptq.py`](neuroquant/quantization/gptq.py)             |
| **SmoothQuant**       | Activation-friendly INT8; per-layer α (closed-form or grid).          | [`neuroquant/quantization/smoothquant.py`](neuroquant/quantization/smoothquant.py) |
| **AWQ**               | INT4 with salient-channel preservation; per-layer α + FP16 carve-out.  | [`neuroquant/quantization/awq.py`](neuroquant/quantization/awq.py)               |
| **SmoothQuant→GPTQ**  | Production recipe — strict-Pareto improvement over either method alone. | [`neuroquant/quantization/smoothquant_gptq.py`](neuroquant/quantization/smoothquant_gptq.py) |
| **AdaRound**          | Post-PTQ refinement; canonical input→output traversal.                 | [`neuroquant/quantization/adaround.py`](neuroquant/quantization/adaround.py)     |

> **AWQ does not support `task=detection`** — its per-layer α search needs static activation shapes, which the RPN/RoI heads of torchvision detectors don't provide. It raises a clear `NotImplementedError` pointing you to PTQ / QAT.

---

## Documentation

Full docs (MkDocs Material) are published at **<https://AbdelazizElHelaly11.github.io/NeuroQuant/>** and live in [`docs/`](docs/):

| Page                                             | For                                                        |
| ------------------------------------------------ | --------------------------------------------------------- |
| [Getting Started](docs/getting_started.md)       | Install matrix, optional extras, verifying the install.   |
| [Using the CLI Pipeline](docs/pipeline_mode.md)  | Researchers running a full, reproducible run from YAML.    |
| [Using the Python Library](docs/library_mode.md) | Developers integrating quantizers into their own scripts. |
| [API Reference](docs/api_reference.md)           | Auto-generated from docstrings via `mkdocstrings`.        |

Build the site locally with `pip install -e ".[docs]" && mkdocs serve`.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md). The latest release, **v2.1.1**, is a
pipeline-audit pass: the FP32 baseline is now scored on the test split
(matching every method's headline), calibration runs on inference-time
preprocessing, mixed-precision size/EBops report real savings, ONNX
deployment fields survive `--resume`, the Pareto hypervolume is
normalized, and config load-time validation covers every choice field.

---

## License

MIT. See [LICENSE](LICENSE) for the full text.
