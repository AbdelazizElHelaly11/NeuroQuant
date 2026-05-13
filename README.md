# NeuroQuant v2.0

[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()

**Production-grade neural-network quantization framework with multi-objective NSGA search, ONNX deployment, and hardware-aware optimisation.**

NeuroQuant takes a pre-trained PyTorch model and produces deployable INT8 / mixed-precision artefacts that have been measured (not estimated) on the same runtime that ships in production. Every public number is the result of running a real quantized graph through ONNX Runtime — no synthetic shortcuts.

---

## What it does

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                                                                        │
   │  FP32 PyTorch model  ─────►  10-phase pipeline  ─────►  INT8 .onnx     │
   │                                                          + metrics     │
   │  ┌──────────────────────────────────────────────────────────────┐     │
   │  │  P0  Prepare model + dataset, FP32 baseline                  │     │
   │  │  P1a Hessian / Fisher per-layer sensitivity                  │     │
   │  │  P1b FITCompress warm-start seed                             │     │
   │  │  P1c NSGA multi-objective search (2- or 3-obj)               │     │
   │  │  P1d AdaRound canonical-order weight rounding                │     │
   │  │  P1e Real W+A QAT with FP32 teacher distillation             │     │
   │  │  P1f GPTQ + SmoothQuant + AWQ + SmoothQuant→GPTQ             │     │
   │  │  P2  Pareto analysis + plots                                 │     │
   │  │  P3  Grad-CAM + SHAP explainability                          │     │
   │  │  P4  MLflow finalisation + reproducibility manifest          │     │
   │  └──────────────────────────────────────────────────────────────┘     │
   │                                                                        │
   └────────────────────────────────────────────────────────────────────────┘
```

The pipeline runs to completion in **~60 seconds** on CPU for a CIFAR-class model.

---

## Why it is production-grade

This framework was built deliberately to avoid the "research prototype" failure modes that disqualify most academic quantization tooling from real deployment:

| Concern                         | What NeuroQuant does                                                                                              |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Real INT inference**          | Wave 4 emits true static-INT8 ONNX graphs via `onnxruntime.quantization.quantize_static`, not FP32 simulation.    |
| **Real on-disk size**           | `model_size_mb` is the literal `.onnx` filesystem size, not `numel × bw / 8`. The synthetic estimate is kept as `theoretical_size_mb` for ablation. |
| **Real latency**                | `latency_ms` is measured under ONNX Runtime on the same machine that will deploy the artefact.                    |
| **Hardware-aware search**       | The NSGA third objective sums a per-layer ORT latency LUT (Wave 4 C2). Every gene's latency cost is a real timing.|
| **No leakage between splits**   | Train / search / val / test are 80/10/10/test-set; NSGA fitness reads search, QAT early-stop reads val, headline reads test. |
| **Strict determinism**          | `set_seed(strict=True)` enforces `CUBLAS_WORKSPACE_CONFIG`, `use_deterministic_algorithms`, `cudnn.deterministic`. |
| **Safe checkpoints**            | All `torch.load(weights_only=True)`; pickle path is closed. Architectural wrappers persist as JSON manifests.     |
| **Real W+A QAT**                | INT8 activations always; weight parametrisation via `torch.nn.utils.parametrize` (autograd-aware STE).            |
| **Validated config**            | Pydantic v2 dataclasses with field validators — bad values fail at load, not deep in a phase.                      |

---

## Install

### From the wheel

```bash
pip install neuroquant-2.0.0-py3-none-any.whl
neuroquant --help
```

### From source

```bash
git clone https://github.com/AbdelazizElHelaly11/NeuroQuant
cd NeuroQuant
pip install -e ".[dev]"        # editable + dev extras
```

GPU users:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

---

## Run

The console-script `neuroquant` is installed by the wheel; it accepts the same flags as `python main.py`.

```bash
# Full pipeline on the bundled config (CIFAR-10 + MobileNetV2)
neuroquant --config config.yaml --epochs 20

# Fast smoke (CPU, no training, first three phases)
neuroquant --config config.yaml --epochs 0 --device cpu \
  --phases phase_0_preparation phase_1a_hessian_clustering phase_1b_fitcompress

# Resume after interruption
neuroquant --config config.yaml --epochs 20 --resume

# Hardware-aware mode (3-objective NSGA + ORT latency LUT)
# Set hardware_aware_search: true in config.yaml, then:
neuroquant --config config.yaml --epochs 20
```

The pipeline writes everything to `output_dir` (default `./artifacts/`):

```
artifacts/
├── checkpoints/          # per-phase resume points
├── onnx/                 # FP32 + per-method INT8 .onnx files
├── pareto/               # Pareto plots + JSON
├── reports/              # pipeline_report.txt, pareto_summary.json
├── reproducibility_manifest.json
├── latency_lut.json      # only when hardware_aware_search=true
└── pipeline_report.txt
```

---

## Configuration

All knobs live in [`config.yaml`](config.yaml). Common overrides:

```yaml
model:
  name: resnet18              # any torchvision name
  num_classes: 10
  input_shape: [3, 32, 32]

dataset:
  name: cifar10               # cifar10 | cifar100 | imagefolder | synthetic | custom
  class: null                 # optional "pkg.module.MyDataset"
  train_dir: null             # optional ImageFolder split dirs
  val_dir: null
  test_dir: null
  batch_size: 128

methods: [ptq, qat, gptq, smoothquant, awq]
bitwidths:
  supported: [4, 8]
  io_layer: 8                 # force first/last layers to INT8

hyperparams:
  hardware_aware_search: true     # Wave 4 J4: 3-obj NSGA
  onnx_export_enabled: true       # Wave 4 J1/J2/J3
  qat_distill_alpha: 0.5          # Wave 2 E5: KD with FP32 teacher
  smoothquant_per_layer_alpha: true  # Wave 3 F3
  hessian_estimator: fisher       # Wave 3 B2: 3× faster than diag
```

Pydantic field validators run at load time — invalid values surface immediately with the offending field path:

```text
ValueError: Configuration validation failed:
  num_classes must be >= 2.
```

---

## Architecture

The framework was built in seven waves, each ending with a strict-format report. Per-wave architecture notes live in [`docs/architecture/`](docs/architecture/):

| Wave | Theme                          | Notes                                      |
| ---- | ------------------------------ | ------------------------------------------ |
| 1    | Foundation (security + leakage) | [wave1.md](docs/architecture/wave1.md)     |
| 2    | Real W+A QAT pipeline           | [wave2.md](docs/architecture/wave2.md)     |
| 3    | Method audits + Fisher          | [wave3.md](docs/architecture/wave3.md)     |
| 4    | ONNX + hardware-aware search    | [wave4.md](docs/architecture/wave4.md)     |
| 5    | Reporting + MLflow              | [wave5.md](docs/architecture/wave5.md)     |
| 6    | Config validation (Pydantic)    | [wave6.md](docs/architecture/wave6.md)     |
| 7    | Packaging + docs                | [wave7.md](docs/architecture/wave7.md)     |

---

## Quantization methods

| Method                | When to use                                                           | Module                                                |
| --------------------- | --------------------------------------------------------------------- | ----------------------------------------------------- |
| **PTQ**               | Fast baseline; INT8 with bitwidth-aware calibration.                  | [`quantization/ptq.py`](quantization/ptq.py)           |
| **QAT**               | Best accuracy at INT8; requires fine-tuning data.                     | [`quantization/qat.py`](quantization/qat.py)           |
| **GPTQ**              | Best accuracy at INT4 weights; data-aware optimal rounding.           | [`quantization/gptq.py`](quantization/gptq.py)         |
| **SmoothQuant**       | Activation-friendly INT8; per-layer α grid search.                    | [`quantization/smoothquant.py`](quantization/smoothquant.py) |
| **AWQ**               | INT4 with salient-channel preservation; per-layer α + FP16 carve-out. | [`quantization/awq.py`](quantization/awq.py)           |
| **SmoothQuant→GPTQ**  | Production recipe — strict-Pareto improvement over either method alone. | [`quantization/smoothquant_gptq.py`](quantization/smoothquant_gptq.py) |
| **AdaRound**          | Post-PTQ refinement; canonical input→output traversal.                 | [`quantization/adaround.py`](quantization/adaround.py) |

---

## License

MIT. See [LICENSE](LICENSE) for the full text.

---


