# NeuroQuant — Usage Guide

This document is the practical guide for **using** NeuroQuant: how to install, how to configure, what's supported out of the box, and what isn't. For the technical "why" of every design decision, see [`PROJECT_REPORT.md`](PROJECT_REPORT.md). For the hardening history, see [`architecture/`](architecture/).

---

## 1. Quick start

### 1.1 Install

```bash
git clone <repo-url>
cd NeuroQuant
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For an editable install with test extras:

```bash
pip install -e ".[dev]"
```

For a CUDA install of PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 1.2 Run the default pipeline

```bash
python main.py --config config.yaml --epochs 3
```

This will:

1. Download CIFAR-10 to `./data/` on first run.
2. Train MobileNetV2 for 3 epochs on the CIFAR-10 training split.
3. Run all 10 quantization phases (sensitivity → search → AdaRound → QAT → GPTQ/SmoothQuant/AWQ → Pareto → XAI → MLflow).
4. Write everything to `./artifacts/`.

If you've installed the wheel:

```bash
neuroquant --config config.yaml --epochs 3
```

### 1.3 Faster runs while you're getting started

```bash
# Skip training (assumes you already have a checkpoint, or just want to test wiring)
python main.py --config config.yaml --epochs 0

# Override the device
python main.py --config config.yaml --epochs 0 --device cpu

# Run only specific phases
python main.py --config config.yaml --epochs 0 \
  --phases phase_0_preparation phase_1a_hessian_clustering phase_1b_fitcompress

# Resume after a crash (reuses every previously-completed phase)
python main.py --config config.yaml --epochs 3 --resume
```

---

## 2. The `config.yaml` file

Every option lives in a single YAML file. The default `config.yaml` at the project root is annotated; this section explains the categories.

### 2.1 Model section

```yaml
model:
  name: "mobilenetv2"          # torchvision name, OR null if using `class`
  class: null                  # OR fully-qualified Python path: "my_pkg.my_module.MyModel"
  path: null                   # optional: load weights from this checkpoint after building
  num_classes: 10
  input_shape: [3, 32, 32]     # (channels, height, width)
```

Three loading routes, in priority order:

1. **By torchvision name.** Set `model.name` to any class in `torchvision.models` (`mobilenetv2`, `resnet18`, `vgg16`, `efficientnet_b0`, etc.). The classifier head is automatically resized to `num_classes`; the first conv is automatically adapted for small inputs (32×32, 64×64).

2. **By fully-qualified class name.** Set `model.class` to e.g. `"my_pkg.my_module.MyCustomCNN"`. The class is imported with `importlib`, instantiated with no arguments, and used directly. Useful for entirely custom architectures.

3. **By saved checkpoint.** Set `model.path` to a `.pth` file. The architecture is built first (from `name` or `class`), then weights are loaded with `weights_only=True` (safe — no pickle execution).

### 2.2 Dataset section

```yaml
dataset:
  name: "cifar10"              # cifar10 | cifar100 | imagefolder | synthetic | <torchvision name>
  class: null                  # OR fully-qualified Dataset class: "my_pkg.my_data.MyDataset"
  path: "./data"               # where to download / look for data
  train_dir: null              # for imagefolder: optional explicit split paths
  val_dir: null
  test_dir: null
  batch_size: 128
  num_workers: 4               # auto-forced to 0 on Windows
```

Five loading routes:

1. **`cifar10` or `cifar100`.** Auto-downloads to `dataset.path` on first run. Native CIFAR transforms applied (random crop + horizontal flip for training, eval transforms for everything else).

2. **`imagefolder`.** Reads `dataset.path` (or the explicit `train_dir`/`val_dir`/`test_dir`) using `torchvision.datasets.ImageFolder`. Class folders inside become labels. If only one folder is given, the framework splits it into train/search/val/test automatically (80/10/10/test-via-val if no test set exists).

3. **`synthetic`.** Generates random tensors of shape `model.input_shape` with random labels. No I/O, fastest possible runs — use for testing wiring only, not for real accuracy numbers.

4. **Other torchvision name.** If `dataset.name` doesn't match any of the above, the framework treats it as another `torchvision.datasets.<Name>` class and tries to load it. This is best-effort — most torchvision datasets work, but the transforms list is generic (resize-to-input + normalize-to-(0.5, 0.5, 0.5)).

5. **Custom Dataset class.** Set `dataset.class` to a fully-qualified path. The framework introspects the constructor and routes through one of three patterns:
   - **`split=` argument:** instantiates with `split="train"` and `split="test"` (HuggingFace-style).
   - **`train=` argument:** instantiates with `train=True` and `train=False` (torchvision-style).
   - **Neither:** instantiates once and splits 70/10/10/10 randomly.

### 2.3 Output section

```yaml
output:
  dir: "./artifacts"           # where every artefact lands
  mlflow_tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "neuroquant_v2"
```

The pipeline creates:

```
artifacts/
├── checkpoints/                   # one .pth or .pt per phase (resume points)
├── onnx/                          # FP32 + per-method INT8 .onnx files
├── pareto/                        # 2-D + 3-D Pareto plots, JSON export
├── xai/                           # Grad-CAM / SHAP outputs
├── reproducibility_manifest.json  # versions + ONNX runtime + LUT path
├── pareto_summary.json            # Wave 5 I3 — best/median/worst across methods
├── pipeline_report.txt            # human-readable summary
└── latency_lut.json               # only when hardware_aware_search: true
```

### 2.4 Methods + bitwidths

```yaml
methods: [ptq, qat, gptq, smoothquant, awq]
bitwidths:
  supported: [4, 8]
  io_layer: 8                  # input + output layers forced to INT8 always
```

`methods` is a filter — only the listed methods run in Phase 1f. To skip QAT and AWQ, just remove them from the list.

`bitwidths.supported` is the search space. NSGA picks per-layer bitwidths from this list. `[4, 8]` is the standard mixed-precision setup; `[8]` runs INT8-only (NSGA degenerates to a single search-space point); `[2, 4, 8]` enables ternary-ish exploration (experimental).

### 2.5 Hyperparameters — the most-used knobs

```yaml
hyperparams:
  seed: 42
  device: "auto"               # auto | cuda | cpu | mps

  # Calibration (PTQ activation observers)
  calibration_batches: 20
  calibration_strategy_io: "kl_divergence"
  calibration_strategy_intermediate: "mse"

  # Sensitivity (Phase 1a)
  hessian_estimator: "fisher"  # fisher (3× faster) | diag_hessian

  # NSGA (Phase 1c)
  nsga_population_size: 8
  nsga_generations: 20

  # AdaRound (Phase 1d)
  adaround_epochs: 100
  adaround_ordered: true       # canonical traversal — keep on

  # QAT (Phase 1e)
  qat_epochs: 5
  qat_lr: 0.001
  qat_act_bitwidth: 8          # always INT8 for deployment
  qat_distill_alpha: 0.5       # 0 disables FP32 teacher distillation

  # SmoothQuant
  smoothquant_per_layer_alpha: true

  # AWQ
  awq_alpha_grid: [0.0, 0.25, 0.5, 0.75, 1.0]
  awq_keep_top_pct: 0.0        # 0.01 keeps the salient 1% at FP16

  # Wave 4: ONNX + hardware-aware
  onnx_export_enabled: true    # measure real on-disk size + ORT latency
  hardware_aware_search: false # opt-in: 3-objective NSGA with latency LUT
```

For the full list of knobs (~50 in total), see [`config.yaml`](../config.yaml) — every field has a comment explaining its purpose.

### 2.6 Phase selection

```yaml
phases:
  - phase_0_preparation
  - phase_1a_hessian_clustering
  - phase_1b_fitcompress
  - phase_1c_nsga_search
  - phase_1d_adaround
  - phase_1e_qat
  - phase_1f_gptq_smooth_awq
  - phase_2_pareto
  - phase_3_xai
  - phase_4_mlflow
```

Comment out any phase to skip it. The pipeline always runs in fixed order — the YAML order is just for readability.

---

## 3. What's supported

### 3.1 Models

| Type                                   | Status            | Notes                                                       |
| -------------------------------------- | ----------------- | ----------------------------------------------------------- |
| Any `torchvision.models` classifier     | ✅ Supported      | mobilenetv2, resnet, vgg, efficientnet, mnasnet, regnet, ...|
| Custom `nn.Module` via `model.class`    | ✅ Supported      | Must be importable; classifier head adapted via introspection |
| Loading from `.pth` checkpoint          | ✅ Supported      | Always `weights_only=True` (safe pickle path closed)        |
| Single classifier head                  | ✅ Supported      | One trailing `nn.Linear`                                    |
| Multi-head (object detection)           | ❌ Not supported  | Detection / segmentation needs a different head adaptor     |
| Sequence models (transformers)          | ⚠️ Partial        | Linear layers quantize fine; attention isn't specialised    |
| Recurrent models (LSTM/GRU)             | ⚠️ Partial        | Linear layers quantize; recurrent ops aren't specialised    |

### 3.2 Datasets

| Source                                  | Status            | How to use                                                  |
| --------------------------------------- | ----------------- | ----------------------------------------------------------- |
| CIFAR-10                                | ✅ Supported      | `dataset.name: cifar10`                                     |
| CIFAR-100                               | ✅ Supported      | `dataset.name: cifar100`                                    |
| ImageFolder (single dir)                | ✅ Supported      | `dataset.name: imagefolder` + `dataset.path`                |
| ImageFolder (explicit splits)           | ✅ Supported      | `dataset.train_dir / val_dir / test_dir`                    |
| Synthetic (random)                      | ✅ Supported      | `dataset.name: synthetic` (testing wiring only)             |
| Other torchvision names                 | ⚠️ Best-effort    | `dataset.name: mnist` etc. — generic transforms applied     |
| Custom `torch.utils.data.Dataset`       | ✅ Supported      | `dataset.class: my_pkg.my_module.MyDataset`                 |
| HuggingFace `datasets`                  | ❌ Not direct     | Wrap in a custom `Dataset` class first                      |

### 3.3 Quantization methods

| Method                  | INT8 | INT4 | Mixed | Status          |
| ----------------------- | :--: | :--: | :---: | --------------- |
| **PTQ**                 |  ✅  |  ✅  |   ✅   | Bitwidth-aware calibration |
| **QAT**                 |  ✅  |  ⚠️  |   ✅   | INT8-A always; INT4-W experimental |
| **GPTQ**                |  ✅  |  ✅  |   ✅   | Conv2d + Linear; block-wise |
| **SmoothQuant**         |  ✅  |  ⚠️  |   ✅   | Per-layer α grid search; INT8 production target |
| **AWQ**                 |  ⚠️  |  ✅  |   ✅   | INT4 production target; FP16 carve-out optional |
| **SmoothQuant→GPTQ**    |  ✅  |  ✅  |   ✅   | Strict-Pareto improvement over either alone |
| **AdaRound** (refinement) |  ✅  |  ✅  |   ✅   | Canonical input→output traversal |

### 3.4 Inference targets

| Target                                  | Status            | Notes                                                       |
| --------------------------------------- | ----------------- | ----------------------------------------------------------- |
| ONNX Runtime CPU                        | ✅ Measured       | Default — every method reports real ORT-CPU latency         |
| ONNX Runtime CUDA                       | ✅ Supported      | Pass `providers=["CUDAExecutionProvider"]` to the latency benchmark |
| TensorRT                                | ⚠️ Compatible     | The .onnx files load in TensorRT, but TensorRT-specific calibration isn't run |
| OpenVINO                                | ⚠️ Compatible     | Same as TensorRT — files load, vendor-specific tooling not bundled |
| qnnpack / fbgemm (PyTorch native)       | ❌ Not supported  | The framework targets ONNX, not PyTorch's native INT backends |
| FPGA / hardware synthesis               | ⚠️ Reports only   | `hardware_report_path` parses Vivado/Quartus outputs into the public report; no synthesis is done |

### 3.5 Reporting + tracking

| Output                                  | Status            |
| --------------------------------------- | ----------------- |
| Public summary table (terminal)         | ✅ With ONNX columns when available |
| Reproducibility manifest (JSON)         | ✅ ORT version, LUT path, ONNX baseline, package versions |
| Pareto plots (2-D)                      | ✅ Accuracy vs ONNX size, per-method palette |
| Pareto plots (3-D)                      | ✅ Accuracy vs size vs ORT latency (when `hardware_aware_search: true`) |
| Pareto JSON export                      | ✅ For paper plots / dashboards |
| MLflow per-method runs                  | ✅ Metrics + .onnx artefact upload |
| MLflow Pareto comparison summary        | ✅ best/median/worst across methods |
| Hardware synthesis report parsing       | ✅ JSON / CSV (Vivado HLS, Quartus); optional |
| TensorBoard                             | ❌ MLflow only |

### 3.6 Reproducibility

| Property                                | Status            |
| --------------------------------------- | ----------------- |
| Same seed → same results                | ✅ `set_seed(strict=True)` |
| Bit-identical across reruns (CPU)       | ✅ |
| Bit-identical across reruns (CUDA)      | ✅ With `CUBLAS_WORKSPACE_CONFIG` set (auto by `set_seed`) |
| Bit-identical across machines           | ⚠️ Same machine class only (CPU instruction set, GPU model) |
| Resume after crash (per-phase)          | ✅ All 10 phases checkpointed |
| Resume preserves metric numbers         | ✅ Verified by Wave 1 + Wave 5 tests |

---

## 4. What's not supported

These are scope decisions, not bugs.

### 4.1 Distributed training

QAT runs on a single device. There's no `torch.distributed` wrapping. For models that require multi-GPU training, train the FP32 baseline elsewhere, save the checkpoint, and load it via `model.path`.

### 4.2 Native INT4 inference kernels

No deployment backend ships with native INT4 conv/linear kernels. The framework's INT4 results are weights-only quantization that runs on INT8 kernels at inference time after weight unpacking. The latency LUT records this honestly — the INT4 row equals the INT8 row.

### 4.3 Dynamic input shapes

ONNX export sets a dynamic batch axis but assumes a fixed `(C, H, W)`. Sequence models with variable input lengths or images of varying resolution would need additional dynamic-axis configuration that isn't currently exposed.

### 4.4 Object detection / segmentation

The classifier-head adaptation in `_find_last_linear` assumes a single trailing `nn.Linear`. Detection heads (multiple branches, anchor heads, FPN) would need a different adaptor. The framework correctly *quantizes* such models if you supply them as-is — it just won't auto-resize the final layer for `num_classes`.

### 4.5 KV-cache quantization for transformers

`nn.Linear` layers quantize correctly, but transformer-specific optimizations (paged KV cache, sliding-window attention) aren't specialised.

### 4.6 Quantization to non-power-of-2 bitwidths

Supported bitwidths are `{4, 8, 16, 32}`. Custom values like INT6 or INT3 require new calibration logic (and have no deployment backend anyway).

### 4.7 Online / streaming quantization

The framework is offline. It calibrates once on a calibration dataset, then ships a fixed quantized model. Online recalibration on production inputs isn't supported.

### 4.8 Model surgery beyond head + first-conv

The framework adapts the classifier head and (when needed) the first convolution stem. It does not modify intermediate layers (e.g. swap a conv for a depthwise+pointwise pair, replace BatchNorm with GroupNorm). Bring a model that already has the architecture you want.

---

## 5. Common workflows

### 5.1 "I have a custom model — how do I plug it in?"

1. Save your model class in a file `my_models/my_cnn.py`:
   ```python
   import torch.nn as nn

   class MyCNN(nn.Module):
       def __init__(self):
           super().__init__()
           # ... your architecture, no special hooks needed ...
       def forward(self, x):
           return ...
   ```

2. Make the file importable from the project root (add it to `sys.path`, install as a package, or just place it where Python can find it).

3. Edit `config.yaml`:
   ```yaml
   model:
     name: null
     class: "my_models.my_cnn.MyCNN"
     num_classes: 10            # only used if your model has a final Linear
     input_shape: [3, 32, 32]
   ```

4. Run as usual. The framework finds the trailing `nn.Linear` and the leading `nn.Conv2d` automatically.

### 5.2 "I have a custom dataset — how do I plug it in?"

1. Subclass `torch.utils.data.Dataset` in `my_data/my_dataset.py`:
   ```python
   from torch.utils.data import Dataset

   class MyDataset(Dataset):
       def __init__(self, split="train", transform=None):
           # split is one of "train" / "val" / "test"
           # transform is the eval-time transform from the framework
           ...
       def __len__(self):
           return ...
       def __getitem__(self, idx):
           return image_tensor, label_int
   ```

2. Edit `config.yaml`:
   ```yaml
   dataset:
     name: "custom"
     class: "my_data.my_dataset.MyDataset"
     batch_size: 64
   ```

3. Run as usual. The framework instantiates `MyDataset(split="train", transform=...)` and `MyDataset(split="test", transform=...)`.

### 5.3 "I want to disable QAT (it's slow)"

```yaml
methods: [ptq, gptq, smoothquant, awq]   # qat removed
phases:
  # Comment out phase_1e_qat:
  - phase_0_preparation
  - phase_1a_hessian_clustering
  - phase_1b_fitcompress
  - phase_1c_nsga_search
  - phase_1d_adaround
  # - phase_1e_qat
  - phase_1f_gptq_smooth_awq
  - phase_2_pareto
  - phase_3_xai
  - phase_4_mlflow
```

### 5.4 "I want hardware-aware search"

```yaml
hyperparams:
  hardware_aware_search: true
  latency_lut_bitwidths: [4, 8]
```

The first run builds the latency LUT (~1–2 minutes for a CIFAR-class model). Subsequent runs reuse the cache at `./artifacts/latency_lut.json`. NSGA switches to 3-objective mode and the Pareto plots gain a 3-D version.

### 5.5 "I want to quantize an already-trained checkpoint"

```yaml
model:
  name: "resnet18"
  path: "/path/to/my_resnet18.pth"
  num_classes: 1000
```

```bash
python main.py --config config.yaml --epochs 0
```

`--epochs 0` skips training and goes straight into quantization on the loaded weights.

### 5.6 "I want to resume after a crash"

```bash
python main.py --config config.yaml --epochs 3 --resume
```

The pipeline reads `./artifacts/checkpoints/` and skips every phase that already has a checkpoint. To force a fresh run, just delete the checkpoints directory.

### 5.7 "I want to compare runs in MLflow"

After several runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000`. Each phase appears as its own run; the Phase 4 summary run has the cross-method `pareto_*_best/median/worst` metrics for direct comparison.

---

## 6. Troubleshooting

### "Coverage failed under 80%"

The coverage gate fires on partial test runs. Append `--no-cov` when running a subset:

```bash
pytest test_wave4_production.py --no-cov
```

### "ConstructorError: could not determine a constructor for the tag ..."

Old YAML config from a pre-Wave-6 run. Regenerate with `python -c "from config import QuantizationConfig; QuantizationConfig().to_yaml('config.yaml')"`.

### "ONNX export failed for ..."

Most common cause: a custom module with control flow (`if`/`for` inside `forward`). The framework uses the legacy TorchScript exporter. Either rewrite the control flow to use tensor ops, or set `onnx_export_enabled: false` to skip ONNX export and fall back to synthetic size + PyTorch latency.

### "RuntimeError: operator torchvision::nms does not exist"

Mismatched torch / torchvision versions. Reinstall both pinned to compatible majors:

```bash
pip install torch==2.7 torchvision==0.22
```

### "Pipeline crashes with `validate()` error mentioning a field I didn't change"

The default `config.yaml` is out of sync with `config.py`. Regenerate it:

```bash
python -c "from config import QuantizationConfig; QuantizationConfig().to_yaml('config.yaml')"
```

### "I get different numbers on the same seed"

Make sure you're running on the same machine and same library versions. Across-machine bit-identity isn't guaranteed (different CPU instruction sets, GPU models). Within the same machine, check the `reproducibility_manifest.json` of the original run and confirm package versions match.

---

## 7. Going further

- **Production deployment:** The `.onnx` files in `./artifacts/onnx/` are the deployment artefacts. Load them with `onnxruntime.InferenceSession` in your serving code.
- **Custom Pareto metrics:** Subclass `ParetoVisualizer` in `visualization/pareto_analysis.py` to add new plots.
- **Custom quantization method:** Subclass `BaseQuantizer` in `quantization/base.py`, implement `quantize()` and `_get_method_name()`, register the class in Phase 1f's plan list in `main.py`.
- **External hardware report:** Set `hyperparams.hardware_report_path` to a Vivado/Quartus JSON or CSV. The metrics are parsed and added to the public report under "Hardware Synthesis Metrics".

For the design rationale behind every choice in this guide, see [`PROJECT_REPORT.md`](PROJECT_REPORT.md) §4 (Genericity contract) and the per-wave architecture notes under [`architecture/`](architecture/).
