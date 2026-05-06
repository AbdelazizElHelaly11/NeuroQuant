# NeuroQuant v2.0 — External Review Report

> **Audience:** independent reviewer (e.g. Codex acting as an auditor) with **zero prior knowledge** of this project.
> **Goal:** verify that NeuroQuant is a generic, production-grade neural-network quantization framework — that the implementation works, that the genericity claims hold (no hardcoded model / dataset / architecture), and that the test suite actually exercises what the documentation says it does.
>
> Every numerical claim in this report can be verified by running the commands in [§9 — Verification protocol](#9-verification-protocol). Every architectural claim cites the file and (where useful) the line number.

---

## 0. TL;DR for the reviewer

| Property                             | Claim                                                                            | Where to verify                                                |
| ------------------------------------ | -------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Generic on model**                 | Any torchvision name OR any user-supplied class via `model.class` config field   | [`models/model_loader.py`](#42-model-genericity)               |
| **Generic on dataset**               | `cifar10`, `cifar100`, `imagefolder`, `synthetic`, or user-supplied `dataset.class`; other torchvision names are best-effort | [`data/data_loader.py`](#43-dataset-genericity)                |
| **No hardcoded layer names**         | Architecture-specific operations use `named_modules()` introspection             | [`models/model_loader.py:45-95`](#44-no-hardcoded-layer-names) |
| **Real INT inference**               | `.onnx` files measured with ONNX Runtime, not FP32 simulation                    | [`utils/onnx_export.py`](#71-real-int-inference)               |
| **Strict determinism**               | `set_seed(strict=True)` enforces deterministic kernels                           | [`utils/common.py:28-78`](#72-determinism)                     |
| **No data leakage**                  | Train / search / val / test are 4 disjoint loaders                               | [`data/data_loader.py`](#6-data-flow--split-isolation)         |
| **Tests pass, gate enforced**        | 167 unit tests + 2 integration smokes pass, ≥80% line coverage gated             | [§9](#9-verification-protocol)                                 |

---

## 1. What is this project?

NeuroQuant takes a pre-trained PyTorch classifier and produces deployable INT8 / mixed-precision quantized variants together with measured (not estimated) accuracy, on-disk size, and inference-latency numbers under ONNX Runtime.

The framework is built around a **10-phase pipeline** that runs end-to-end on any compatible (model, dataset) pair the user names in `config.yaml`:

```
P0  Prepare model + dataset, FP32 baseline + ONNX export
P1a Per-layer sensitivity (Fisher / diag-Hessian)
P1b FITCompress warm-start seed
P1c NSGA multi-objective search (2- or 3-objective)
P1d AdaRound canonical-order weight rounding
P1e W+A QAT with FP32 teacher distillation
P1f GPTQ + SmoothQuant + AWQ + SmoothQuant→GPTQ (INT8 + INT4 each)
P2  Pareto analysis + plots (2-D and 3-D)
P3  Grad-CAM + SHAP explainability
P4  MLflow finalisation + reproducibility manifest
```

Each phase has its own checkpoint and resume path; a crash in P1f does not lose P0–P1e.

The framework was hardened over **seven waves**, each ending with a strict-format report and a bundled test suite. Per-wave architecture notes (decision matrix, what shipped, tests) live in [`docs/architecture/wave{1..7}.md`](architecture/).

---

## 2. Project layout

```
NeuroQuant/
├── README.md                       Public entry point (install / run / methods)
├── LICENSE                         MIT
├── pyproject.toml                  Build + pytest + coverage config
├── requirements.txt                Runtime dependencies
├── config.py                       Pydantic-backed config (917 lines)
├── config.yaml                     Default config — every knob exposed
├── main.py                         NeuroQuantPipeline orchestrator (2629 lines)
├── conftest.py                     Shared pytest fixtures
│
├── data/
│   └── data_loader.py              GenericDatasetLoader — pluggable on dataset
├── models/
│   └── model_loader.py             ModelLoader — pluggable on model class
├── quantization/
│   ├── base.py                     BaseQuantizer (abstract API)
│   ├── ptq.py                      Post-training quantization
│   ├── qat.py                      Quantization-aware training (W+A INT8)
│   ├── gptq.py                     GPTQ optimal-rounding
│   ├── smoothquant.py              SmoothQuant migration + per-layer α
│   ├── awq.py                      AWQ with input-scale wrapper + FP16 carve-out
│   ├── smoothquant_gptq.py         Combined SmoothQuant→GPTQ
│   ├── adaround.py                 AdaRound canonical-order traversal
│   ├── bn_folding.py               Conv-BN analytic fusion
│   ├── fitcompress.py              FITCompress warm-start seed
│   ├── hessian_clustering.py       Fisher / diag-Hessian + cluster tiers
│   ├── latency_lut.py              Per-layer ORT latency LUT (Wave 4 C2)
│   └── nsga_ii_search.py           N-objective NSGA (961 lines)
├── utils/
│   ├── common.py                   set_seed, get_device, model size
│   ├── numerics.py                 Centralised epsilon constants
│   ├── metrics.py                  topk accuracy, latency, hardware report
│   ├── checkpointing.py            Safe state_dict + manifest, repro manifest
│   └── onnx_export.py              FP32 export, static INT8 quantization, ORT latency
├── tracking/
│   └── mlflow_logger.py            MLflowTracker with no-op fallback
├── visualization/
│   ├── pareto_analysis.py          ParetoAnalyzer + ParetoVisualizer (933 lines)
│   └── style.py                    Per-method colour/marker palette
├── xai/
│   └── explainability.py           Grad-CAM + SHAP + comparison matrix
│
├── tests/
│   └── integration/
│       └── test_full_pipeline_smoke.py   End-to-end smoke (Wave 6 K3)
├── test_wave1_production.py through test_wave7_production.py    Wave-by-wave tests
├── test_metrics.py                 hw-report parser, top-k edge case
├── test_xai_outputs.py             Phase 3 output quality
│
├── docs/
│   ├── architecture/wave1.md … wave7.md   Per-wave decision matrices
│   └── PROJECT_REPORT.md           ← this file
└── .github/workflows/tests.yml     CI: lint + unit tests + coverage gate + integration
```

**Total project size:** ~11.5 K lines of Python source. No vendored binaries; everything reads through `git ls-files`.

---

## 3. Setup & quick verification

The reviewer should be able to clone and run on Linux / macOS / Windows with Python 3.10–3.12.

### 3.1 Install

```bash
git clone <repo-url>
cd NeuroQuant
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-cov hypothesis    # test extras
```

For a CUDA-only install, replace the `torch` line with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3.2 First run — fast smoke (~60 s on CPU)

```bash
pytest -m integration --no-cov
```

Expected: **2 tests pass** in ~60–90 seconds. This proves the entire pipeline runs end-to-end on a synthetic 4-class 32×32 dataset with no downloads.

### 3.3 Full unit-test suite (~100 s on CPU)

```bash
pytest -m "not integration"
```

Expected: **167 tests pass**, **coverage 81.3%** (gate is ≥80%).

If the coverage check fails on a partial run (e.g. the reviewer runs a subset), append `--no-cov`:

```bash
pytest test_wave4_production.py --no-cov
```

### 3.4 Real pipeline run

```bash
neuroquant --config config.yaml --epochs 3
```

(or `python main.py --config config.yaml --epochs 3` if the wheel isn't installed)

This downloads CIFAR-10 to `./data/`, trains MobileNetV2 for 3 epochs, runs every phase, and writes the artefacts to `./artifacts/`.

For the reviewer's purposes a 0-epoch run on the synthetic dataset is faster:

```bash
python main.py --epochs 0 --device cpu
# Edit config.yaml first: set dataset.name to "synthetic"
```

---

## 4. Genericity contract — the reviewer's primary concern

This section exists because the project's claim is now scoped more precisely: *"NeuroQuant is a configurable image-classification quantization framework for torchvision-style classifiers, custom PyTorch classifier classes, CIFAR/ImageFolder/synthetic datasets, and user-supplied `Dataset` classes."* The reviewer should verify each row of the table by inspecting the cited files.

### 4.1 What is configurable vs hardcoded

| Concern                       | Configurable via                                     | Hardcoded? | Verification                                      |
| ----------------------------- | ---------------------------------------------------- | ---------- | ------------------------------------------------- |
| Model architecture            | `model.name` (torchvision) OR `model.class` (FQ name) | No         | [`models/model_loader.py:251-310`](../models/model_loader.py)               |
| Number of classes             | `model.num_classes`                                  | No         | [`models/model_loader.py:104-140`](../models/model_loader.py) — classifier head adapted via introspection |
| Input shape                   | `model.input_shape: [C, H, W]`                       | No         | Validated by pydantic; first conv adapted on-the-fly                      |
| Dataset                       | `dataset.name` OR `dataset.class`                    | No         | [`data/data_loader.py`](../data/data_loader.py)                              |
| Batch size                    | `dataset.batch_size`                                 | No         | Used everywhere as `cfg.batch_size`                                          |
| Data split seed               | `hyperparams.seed`                                   | No         | Train/search/val splits use the configured seed                              |
| ImageFolder split dirs        | `dataset.train_dir`, `dataset.val_dir`, `dataset.test_dir` | No  | Relative to `dataset.path` or absolute                                       |
| Quantization methods to run   | `methods` list                                       | No         | Phase 1f filters its plan against this list                                  |
| Bitwidths                     | `bitwidths.supported`, `bitwidths.io_layer`          | Partial    | PTQ/QAT respect the config; NSGA/FIT/method presets are still INT4/INT8-oriented |
| Hessian estimator             | `hessian_estimator: fisher \| diag_hessian`          | No         | [`quantization/hessian_clustering.py`](../quantization/hessian_clustering.py) |
| QAT activation bitwidth       | `qat_act_bitwidth`                                   | No         | Pydantic validator restricts to `{4, 8, 16, 32}`                             |
| KD distillation               | `qat_distill_alpha`, `qat_distill_temperature`       | No         | `α=0` disables KD entirely                                                   |
| Hardware-aware mode           | `hardware_aware_search`                              | No         | Toggles 3-objective NSGA + LUT build                                         |

Every config field is exposed in [`config.yaml`](../config.yaml) and validated by `QuantizationConfig.validate()` plus pydantic field validators ([`config.py`](../config.py)).

### 4.2 Model genericity

The framework supports three loading modes, all routed through [`ModelLoader.load()`](../models/model_loader.py#L251):

1. **By torchvision name:**
   ```yaml
   model:
     name: "mobilenetv2"   # or resnet18, vgg16, efficientnet_b0, etc.
     num_classes: 10
   ```
   `ModelLoader` calls `getattr(torchvision.models, normalised_name)` and adapts the final classifier to `num_classes` automatically.

2. **By fully-qualified class name:**
   ```yaml
   model:
     class: "my_pkg.my_module.MyCustomCNN"
     num_classes: 10
   ```
   `ModelLoader` does `importlib.import_module(...)` and instantiates the class.

3. **By saved checkpoint:**
   ```yaml
   model:
     name: "mobilenetv2"   # or model.class — needed for the architecture
     path: "./artifacts/checkpoint_fp32.pth"
   ```
   Architecture is built fresh; `state_dict` is loaded with `weights_only=True`.

The classifier head is adapted **without hardcoded layer names** — see [`_find_last_linear`](../models/model_loader.py#L45). The first conv is adapted for small-input datasets the same way ([`_find_first_conv`](../models/model_loader.py#L60)).

### 4.3 Dataset genericity

`GenericDatasetLoader` ([`data/data_loader.py`](../data/data_loader.py)) routes on `cfg.dataset_name`:

| Config value           | Behaviour                                                          |
| ---------------------- | ------------------------------------------------------------------ |
| `dataset.name: cifar10` / `cifar100` | Explicit torchvision adapters, downloads to `cfg.dataset_path` |
| `dataset.name: imagefolder` | `torchvision.datasets.ImageFolder` with configurable train/val/test dirs |
| `dataset.name: synthetic` | Random tensors of shape `cfg.input_shape` (no I/O, fastest tests) |
| `dataset.class: "pkg.mod.MyDataset"` | Dynamically imports a user-supplied `torch.utils.data.Dataset` class |
| other `dataset.name` values | Best-effort torchvision lookup for datasets with compatible constructors |

Every loader returns the same five DataLoaders: train, search, val, test, calibration. The split contract is identical across modes ([§6](#6-data-flow--split-isolation)).

For ImageFolder, `dataset.train_dir`, `dataset.val_dir`, and `dataset.test_dir` can be absolute paths or paths relative to `dataset.path`. If `val_dir` is omitted, validation/search are split from the train folder. If `test_dir` is omitted or missing, the loader falls back to the validation split and logs a warning.

For `dataset.class`, the class should be fully qualified and return a `torch.utils.data.Dataset`. The loader supports common constructor styles such as `root=...`, `data_dir=...`, `path=...`, `split=...`, `train=...`, and `transform=...`. If the class exposes `split` or `train`, the loader uses those split controls; otherwise it splits the full dataset into train/search/val/test with `hyperparams.seed`.

This is intentionally **not** a claim that every dataset in torchvision or every possible data source works automatically. Datasets with unusual constructor arguments should use `dataset.class` or a small adapter class.

### 4.4 No hardcoded layer names

The framework never references a specific architecture's named modules. Architecture-specific operations all go through introspection:

| Operation                        | Implementation                                                  | File:Line                                   |
| -------------------------------- | --------------------------------------------------------------- | ------------------------------------------- |
| Find classifier head             | `_find_last_linear` — last `nn.Linear` in `named_modules()`     | `models/model_loader.py:45`                 |
| Find input stem                  | `_find_first_conv` — first `nn.Conv2d` in `named_modules()`     | `models/model_loader.py:60`                 |
| Replace a submodule              | `_set_module_by_name` — getattr/setattr along dotted path       | `models/model_loader.py:72`                 |
| Enumerate quantizable params     | `_build_quantizable_weight_set` — Conv2d/Linear `.weight` only  | `quantization/nsga_ii_search.py:889`        |
| Conv-BN pair detection           | `list_conv_bn_pairs` — walks named modules looking for the pattern | `quantization/bn_folding.py`                |
| Per-layer activation collection  | Forward hooks on every Conv2d/Linear                            | `quantization/adaround.py`, `latency_lut.py`|

The only place a layer name appears literally is in the **per-run** bitwidth assignment dict (e.g. `{"features.0.weight": 8, ...}`) — and that dict is *generated* from `model.named_parameters()`, never hard-typed.

**To verify:** swap `model.name: mobilenetv2` for `model.name: resnet18` in `config.yaml` and run the integration smoke. The pipeline must complete with no edits to source code.

---

## 5. Quantization methods — production decisions

For each method the report explains: (a) what the algorithm is, (b) **what would have been the naive implementation and why it would fail in production**, (c) what's actually implemented. The reviewer can use this section to spot-check that the code matches the claim.

### 5.1 PTQ — [`quantization/ptq.py`](../quantization/ptq.py)

Per-tensor symmetric INT8 calibration with two strategies (`kl_divergence` for I/O layers, `mse` for intermediate layers, controlled by `calibration_strategy_*` config fields). The naive failure mode would be a single-strategy calibration; both are needed because input/output activations have different distribution shapes than intermediate features.

Bitwidth-aware calibration: when an NSGA candidate assigns INT4 to a layer, that layer's threshold is computed at INT4 (different `qmax`), not at INT8 — see `calibrate_with_assignment` ([`ptq.py`](../quantization/ptq.py)). Without this, the INT4 thresholds inherited from INT8 calibration produce the wrong scale.

### 5.2 QAT — [`quantization/qat.py`](../quantization/qat.py)

**Real W+A quantization-aware training**, not the simulation common in research code:

- Activations always INT8 (`qat_act_bitwidth`); deployment shape every supported backend (qnnpack, fbgemm, ORT, TensorRT) expects.
- Conv-BN folded analytically *before* QAT ([`bn_folding.py`](../quantization/bn_folding.py)); deployment INT8 always has BN folded.
- Weight quantization via `torch.nn.utils.parametrize` with a custom `_FakeQuantizeSTE` autograd `Function` — the gradient flows through the round operation correctly (the naive `mod.weight.data = quantize(mod.weight.data)` bypass would lose autograd connection).
- Activation observer is a 3-phase state machine: `passthrough` (collect ranges) → `calibrating` (set scale from KL/MSE) → `quantizing` (fake-quantize forward).
- Optional FP32 teacher KD: `loss = α·T²·KL(student/T || teacher/T) + (1-α)·CE`. Teacher is `copy.deepcopy(self.model)` frozen at QAT start.

### 5.3 GPTQ — [`quantization/gptq.py`](../quantization/gptq.py)

Optimal-rounding via second-order weight error compensation. Block size and damping are config knobs (`gptq_block_size`, `gptq_percdamp`). Implemented over `nn.Linear` and `nn.Conv2d` — the conv path reshapes the kernel into a 2-D matrix so the same algorithm applies.

### 5.4 SmoothQuant — [`quantization/smoothquant.py`](../quantization/smoothquant.py)

Migrates per-channel activation difficulty into the weights via `W' = s · W` plus a compensating `X' = X / s` wrapper before each layer (`_SmoothInputScale`). The forward is mathematically equivalent to the original; the *quantization* is now easier because the per-channel weight magnitudes are proportional to the per-channel activation magnitudes.

**Production decision:** per-layer α grid search (`smoothquant_per_layer_alpha`). The naive global α from the SmoothQuant paper is rarely optimal across a heterogeneous network; per-layer search picks the α minimising layer-output reconstruction MSE on the calibration sample.

The wrapper survives a save/load round-trip via JSON metadata + state_dict (`serialize_smoothquant_metadata` / `restore_smoothquant_wrappers`) — pickling the wrapper directly was rejected for security (RCE on `weights_only=False`).

### 5.5 AWQ — [`quantization/awq.py`](../quantization/awq.py)

`Y = (X / s) · quantize(s · W)` with `s = max(a, ε)^α` mean-1 normalised. **The previous version was mathematically broken** — applied weight scaling without input compensation; this rewrite fixed that.

Per-layer α search over `awq_alpha_grid`. Optional top-K% salient channels kept at FP16 (`awq_keep_top_pct`) — the AWQ paper's Section 3 ablation, off by default.

### 5.6 AdaRound — [`quantization/adaround.py`](../quantization/adaround.py)

Per-layer rounding optimisation with an entropy regulariser. Two modes:

- `adaround_ordered: true` (default) — canonical input→output traversal: each layer's quantized output propagates into the next layer's input, so downstream layers see the *quantized* activations they will see at deployment.
- `adaround_ordered: false` — parallel mode (research baseline). Consistently underperforms on deep networks because it ignores accumulated upstream error.

Streaming activation collection (`_collect_activations_for_one_layer`, `adaround_max_samples_per_layer` cap) keeps memory constant regardless of model depth.

**Caught bug:** the target-parameter list previously included BatchNorm weights (`bn1.weight`, etc.). The wave-2 test `test_adaround_topological_order_matches_module_order` exposed this; fix in `_is_quantizable_weight` filters to Conv/Linear weights only.

### 5.7 SmoothQuant→GPTQ — [`quantization/smoothquant_gptq.py`](../quantization/smoothquant_gptq.py)

Two-stage method that ships in production CNN/LLM stacks in 2024+. Stage 1: SmoothQuant migration only (`apply_smoothing_only`). Stage 2: GPTQ on the smoothed weights, with calibration activations captured *after* the input-scaling wrapper so GPTQ sees the post-divide activations the deployment graph actually uses.

Strict-Pareto improvement over either method alone in almost every measured configuration.

---

## 6. Data flow — split isolation

The single most important correctness property: **no leakage between the loader used to drive search, the loader used for early-stopping, and the loader used as the public headline.**

```
┌──────────── original train set ────────────┐
│  90% (train_loader)   │  10% (search_loader) │           ┌─── val (100%) ──┐
│  augmentation enabled │  eval-time, no aug   │           │  no aug         │
└───────────────────────┴──────────────────────┘           └─────────────────┘
                                                                    │
              ┌─── test (held out) ───┐                              │
              │  no aug, no peeking   │ ◄── public headline accuracy ┘
              └───────────────────────┘
```

| Loader               | Purpose                                                 | Read by                                          |
| -------------------- | ------------------------------------------------------- | ------------------------------------------------ |
| `train_loader`       | Optimisation only (training, QAT)                       | `phase_0_preparation` (training), `phase_1e_qat` |
| `search_loader`      | NSGA fitness + PTQ rerank selection                     | `phase_1c_nsga_search`                           |
| `val_loader`         | QAT early-stopping, internal diagnostics                | `phase_1e_qat`                                   |
| `test_loader`        | **Public headline accuracy only**                       | `_attach_split_metrics` in `main.py`             |
| `calibration_loader` | PTQ activation observers, AWQ/SmoothQuant α search, LUT | `phase_1a..1f`                                   |

`_attach_split_metrics` in [`main.py`](../main.py) recomputes `val_top1` and `test_top1` from the model directly so the public headline is always the test number, never the val number that drove early-stopping.

**To verify:** see `test_search_loader_disjoint_from_val_and_test` in [`test_wave1_production.py`](../test_wave1_production.py).

---

## 7. Production-grade features

### 7.1 Real INT inference

`model_size_mb` and `latency_ms` on every quantized result are **measured**, not estimated:

- **Size:** literal `.onnx` file size after `onnxruntime.quantization.quantize_static` (QDQ format, per-channel weights, QInt8 activations). The synthetic `numel × bw / 8` figure is preserved as `theoretical_size_mb` for ablation.
- **Latency:** `onnxruntime.InferenceSession` warmup + timed runs.

See [`utils/onnx_export.py`](../utils/onnx_export.py).

### 7.2 Determinism

`utils/common.py:set_seed(seed, strict=True)` enforces:

- `PYTHONHASHSEED` (stable dict iteration)
- `CUBLAS_WORKSPACE_CONFIG=":4096:8"` (deterministic cuBLAS GEMM)
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `torch.backends.cudnn.deterministic = True`, `benchmark = False`

Called once in `NeuroQuantPipeline.__init__` before any DataLoader fork or CUDA context.

### 7.3 Safe checkpoints

Every `torch.load` in the project passes `weights_only=True` (verifiable by `grep -rn "torch.load(" .`). Architectural wrappers that can't be expressed in a pure `state_dict` (SmoothQuant input-scale, AWQ input-scale) persist as JSON metadata + state_dict via `save_safe_module` / `load_safe_module` ([`utils/checkpointing.py`](../utils/checkpointing.py)).

### 7.4 Pydantic-backed config

`HyperparameterSet` and `QuantizationConfig` use `pydantic.dataclasses.dataclass` (Pydantic v2) with field validators. Bad values fail at construction with the offending field path:

```text
ValueError: device='tpu' invalid. Use 'auto', 'cuda', 'cpu', or 'mps'.
```

`from_yaml` / `from_json` call `.validate()` after building, so YAML-loose values (e.g. `num_classes: "10"` as a string) are coerced and cross-field constraints (low < high percentile, valid phase names) are checked.

### 7.5 Hardware-aware search

Opt-in via `hardware_aware_search: true`. Pipeline:

1. Build per-layer ORT latency LUT once ([`quantization/latency_lut.py`](../quantization/latency_lut.py)) — for each Conv/Linear, build a single-op micro-graph, statically quantize to INT8, benchmark under ORT.
2. Cache to `output_dir/latency_lut.json`.
3. NSGA runs in 3-objective mode `[acc_loss, model_size_mb, latency_ms]`. Each candidate's latency is a sum over the LUT — no per-eval ONNX export.

The `_dominates` predicate and `_non_dominated_sort` in [`nsga_ii_search.py`](../quantization/nsga_ii_search.py) are generalised over arbitrary N, so the same routine drives both 2-obj and 3-obj searches.

---

## 8. Test surface

| File                                | Tests | Wave | What it covers                                                                |
| ----------------------------------- | ----- | ---- | ----------------------------------------------------------------------------- |
| `test_wave1_production.py`          | 10    | 1    | Safe pickle, deterministic seed, split disjointness, headline integrity       |
| `test_wave2_production.py`          | 13    | 2    | Conv-BN folding equivalence, weight-parametrization gradient, observer state, KD loss, AdaRound topological order |
| `test_wave3_production.py`          | 12    | 3    | Fisher correlation, per-layer α coverage, AWQ forward equivalence, SmoothQuant→GPTQ wrapper roundtrip |
| `test_wave4_production.py`          | 20    | 4    | ONNX round-trip, INT8 .onnx smaller than FP32, per-layer LUT covers all Conv/Linear, 3-obj non-dominated sort |
| `test_wave5_production.py`          | 18    | 5    | Headline schema, deployment fidelity printer, manifest blocks, pareto summary |
| `test_wave6_production.py`          | 40    | 6    | Shared fixtures, pydantic validators (24 parametrised), 4 hypothesis property tests, coverage gate file |
| `test_wave7_production.py`          | 16    | 7    | Pyproject metadata, console-script entry point, wheel structure, README + per-wave docs exist |
| `test_metrics.py`                   | 5     | —    | Hardware-report parser (JSON/CSV), top-k edge case (`num_classes < 5`), MLflow keys |
| `test_xai_outputs.py`               | 3     | —    | Phase 3 style helpers, prediction matrix, classname fallback                  |
| `tests/integration/test_full_pipeline_smoke.py` | 2 | 6 | Full pipeline runs end-to-end (synthetic dataset, ~60s)                       |
| **Total**                           | **139 + 2 integration** |     | (Some `parametrize` tests count multiple cases — pytest reports 167 unit + 2 integration) |

Property-based tests use `hypothesis` to generate inputs:

```python
@given(elems=st.lists(st.floats(min_value=-100.0, max_value=100.0), min_size=4, max_size=64),
       bitwidth=st.sampled_from([4, 8, 16]))
def test_quantize_tensor_stays_in_symmetric_range(elems, bitwidth):
    ...
```

These exercise the quantization invariants (round-trip in symmetric range, per-channel scale positivity, MSE monotonic in bitwidth, latency-LUT sum is associative) on 50–200 generated inputs each.

### 8.1 Coverage

```
quantization/awq.py                    547 lines    87%
quantization/qat.py                    739 lines    91%
quantization/nsga_ii_search.py         961 lines    65%   (main loop only fires when pop_size > search_space)
quantization/ptq.py                    613 lines    94%
quantization/smoothquant.py            617 lines    90%
quantization/latency_lut.py            368 lines    90%
utils/onnx_export.py                   454 lines    87%
visualization/pareto_analysis.py       933 lines    90%
config.py                              917 lines   100%
TOTAL                                                81.3%   (gate ≥80%)
```

The 65% on `nsga_ii_search.py` reflects the exhaustive vs evolutionary code paths: the integration smoke uses a 4-cluster search that fits in one population, so the evolutionary loop is exercised by the unit tests directly (not the smoke).

---

## 9. Verification protocol

The reviewer should run each of these commands and confirm the expected outcome.

### 9.1 Setup is reproducible

```bash
pip install -r requirements.txt
pip install pytest pytest-cov hypothesis
```

**Expected:** all packages install without conflicts.

### 9.2 Default config validates cleanly

```bash
python -c "from config import QuantizationConfig; cfg = QuantizationConfig(); cfg.validate(); print('OK', cfg.model_name)"
```

**Expected:** `OK mobilenetv2`.

### 9.3 Pydantic catches bad config at construction

```bash
python -c "from config import QuantizationConfig; QuantizationConfig(num_classes=-3)"
```

**Expected:** `pydantic.ValidationError` or `ValueError` mentioning `num_classes`.

### 9.4 Full unit-test suite passes with coverage gate

```bash
pytest -m "not integration"
```

**Expected:** `167 passed, 2 deselected, ... Required test coverage of 80% reached. Total coverage: 81.26%`.

### 9.5 Integration smoke runs end-to-end

```bash
pytest -m integration --no-cov
```

**Expected:** `2 passed in 60–90s`. This proves all 9 default phases run, ONNX exports happen, the manifest + summary JSON are written.

### 9.6 Genericity claim — swap the model

```bash
python -c "
from config import QuantizationConfig
from models.model_loader import ModelLoader
cfg = QuantizationConfig(num_classes=10, input_shape=(3,32,32))
for name in ['mobilenetv2', 'resnet18', 'vgg11', 'efficientnet_b0']:
    cfg.model_name = name
    m = ModelLoader(cfg).load()
    n = sum(p.numel() for p in m.parameters())
    print(f'{name:20s}  {n:>10,} params')
"
```

**Expected:** four different models load with different parameter counts, **no source-code edits**.

### 9.7 Genericity claim — swap the dataset

```bash
python -c "
from config import QuantizationConfig
from data.data_loader import GenericDatasetLoader
for ds in ['synthetic', 'cifar10']:
    cfg = QuantizationConfig(dataset_name=ds, batch_size=8)
    cfg.input_shape = (3, 32, 32)
    cfg.num_classes = 10
    loader = GenericDatasetLoader(cfg)
    print(f'{ds:12s}  train={len(loader.get_train_loader())} batches')
"
```

**Expected:** both datasets produce non-zero train batches. (CIFAR-10 will download to `./data/` on first run.)

For custom datasets, use the `dataset.class` field with a fully-qualified `torch.utils.data.Dataset` class. The focused no-download regression test is:

```bash
pytest test_genericity_config.py --no-cov
```

**Expected:** the custom dataset class, config-backed split seed, platform-aware `num_workers`, and CLI override behavior all pass.

### 9.8 Real INT8 ONNX shrinks the model

```bash
python -c "
import torch, torch.nn as nn, tempfile, os
from torch.utils.data import DataLoader, TensorDataset
from utils.onnx_export import export_to_onnx, quantize_onnx_static, onnx_disk_size_mb

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 32, 3, padding=1)
        self.f = nn.Linear(32, 10)
    def forward(self, x):
        return self.f(x.mean([2,3]) @ self.c.weight.flatten(1).T)

m = M().eval()
calib = DataLoader(TensorDataset(torch.randn(32,3,32,32), torch.zeros(32,dtype=torch.long)), batch_size=8)
with tempfile.TemporaryDirectory() as d:
    export_to_onnx(m, (3,32,32), os.path.join(d,'fp32.onnx'))
    quantize_onnx_static(os.path.join(d,'fp32.onnx'), os.path.join(d,'int8.onnx'), calib, num_batches=2)
    print(f'FP32: {onnx_disk_size_mb(os.path.join(d,\"fp32.onnx\")):.4f} MiB')
    print(f'INT8: {onnx_disk_size_mb(os.path.join(d,\"int8.onnx\")):.4f} MiB')
"
```

**Expected:** INT8 file is **smaller** than FP32 file. (The exact ratio depends on the model — for tiny graphs the QDQ overhead can shrink the gap, but INT8 must never be larger.)

### 9.9 Determinism

```bash
python -c "
import torch
from utils.common import set_seed
set_seed(42, strict=True)
a = torch.randn(3, 4)
set_seed(42, strict=True)
b = torch.randn(3, 4)
assert torch.equal(a, b), 'set_seed is not deterministic'
print('OK — bit-identical tensors across reruns')
"
```

**Expected:** `OK` printed.

### 9.10 No hardcoded layer names

```bash
grep -rn --include="*.py" "features\.0\|classifier\.1\|layer1\." \
    quantization/ utils/ models/ main.py
```

**Expected:** at most one match — the docstring example for `_set_module_by_name` in `models/model_loader.py:76` (`Example: _set_module_by_name(model, "classifier.1", nn.Linear(1280, 10))`). No matches in executable code paths. Verifiable by inspecting [`models/model_loader.py:_find_last_linear`](../models/model_loader.py#L45) and the per-method quantization modules — the framework refers to layer names exclusively through `model.named_modules()` / `named_parameters()`.

### 9.11 Wheel builds and installs

```bash
python -m pip install build
python -m build --wheel
pip install --no-deps dist/neuroquant-2.0.0-py3-none-any.whl
neuroquant --help
```

**Expected:** wheel builds, install succeeds, `neuroquant --help` prints argparse usage.

---

## 10. What the reviewer should look for as red flags

| Red flag                                                   | Why it would matter                                                  | Where it would appear                          |
| ---------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------- |
| `torch.load(..., weights_only=False)` in **executable code** | Pickle RCE risk (the project banned this)                            | `grep -rn "weights_only=False" --include="*.py"` should match only docstrings/comments documenting what was removed (test the line by checking it isn't followed by `:` or appearing inside a function body executable line) |
| References to specific layer names (`features.0.weight`)   | Architecture-specific code that won't generalise                     | quantization/ modules                          |
| `top1` from `val_loader` printed as headline                | Leakage between early-stop and reporting                             | `_attach_split_metrics` should override        |
| Hardcoded number-of-classes assumption                     | Breaks on non-CIFAR datasets                                         | `_find_last_linear` adapts; `_set_module_by_name` replaces |
| Magic epsilon constants scattered in code                  | Inconsistent numerics; hard to ablate                                | Should all live in `utils/numerics.py`         |
| `model_size_mb = numel × bw / 8` as the public headline    | Synthetic estimate, not deployment-faithful                          | Wave 4 J3 replaced with on-disk ONNX size      |
| `validate()` cross-field checks moved into pydantic        | Loses the ability to enforce e.g. `low < high` percentile relations | Both must coexist; some checks are field-level, some are cross-field |
| Tests that always pass regardless of whether the code works | False security                                                      | Spot-check by mutating a wave-1/2/3 module's logic and re-running its tests |

---

## 11. Honest known limitations

The project does not claim to do these:

1. **Distributed / multi-GPU training** — the QAT path is single-device. `torch.distributed` would be a future feature.
2. **Dynamic-shape inference** — the ONNX export sets a dynamic batch axis but assumes a fixed `(C, H, W)`. Sequence models (transformers with variable lengths) would need extra work.
3. **INT4 native kernels** — no deployment backend ships with native INT4 conv kernels. The framework's INT4 results are weights-only quantization that runs on INT8 kernels at inference time. The latency LUT records this faithfully (INT4 row equals INT8 row) — see [`quantization/latency_lut.py`](../quantization/latency_lut.py).
4. **Object detection / segmentation heads** — the framework's classifier-head adaptation in `_find_last_linear` assumes a single trailing `nn.Linear`. A detection head with multiple branches would need a different adaptor.
5. **Quantization of attention modules** — covered for `nn.Linear` but not specialised for KV-cache quantization in transformers.

These are scope decisions for a graduation project, not bugs.

---

## 12. The 7-wave hardening process

The framework was hardened over seven sequential waves, each with a strict-format report and a bundled test suite. Per-wave decision matrices live under [`docs/architecture/`](architecture/):

| Wave | Theme                            | Key outputs                                                | Tests |
| ---- | -------------------------------- | ---------------------------------------------------------- | ----- |
| 1    | Foundation: security + determinism + split isolation | Safe checkpoints, `set_seed`, train/search/val/test split | 10    |
| 2    | Real W+A QAT pipeline             | Conv-BN folding, weight parametrization, KD distillation   | 13    |
| 3    | Method audits + Fisher estimator  | AWQ rewrite, SmoothQuant per-layer α, SmoothQuant→GPTQ     | 12    |
| 4    | ONNX + hardware-aware search      | Static INT8 export, ORT latency, per-layer LUT, 3-obj NSGA | 20    |
| 5    | Reporting + MLflow                | Deployment fidelity section, ONNX in MLflow, Pareto summary | 18    |
| 6    | Testing + CI + Pydantic           | Shared fixtures, coverage gate, integration smoke, pydantic validators | 40    |
| 7    | Packaging + docs                  | `pip install neuroquant`, console script, README, this report | 16    |

Each wave document follows the same format:
1. **Decision matrix** — every item considered, with the production decision.
2. **What shipped** — file:section references for every change.
3. **Tests** — what test file backs the contract.
4. **Outcomes** — what the wave proves.

---

## 13. Reviewer checklist (one-page summary)

```
[ ] Clone + install (§3.1)
[ ] Default config validates (§9.2)
[ ] Bad config raises at construction (§9.3)
[ ] All unit tests pass (§9.4) — expect 167 passed, 81.3% coverage
[ ] Integration smoke passes (§9.5) — expect 2 passed in ~60s
[ ] Multiple model architectures load (§9.6)
[ ] Multiple datasets load (§9.7)
[ ] Real INT8 ONNX shrinks the model (§9.8)
[ ] Deterministic seeding works (§9.9)
[ ] No hardcoded layer names (§9.10)
[ ] Wheel builds + installs (§9.11)
[ ] Inspect quantization/ modules — match the production decisions in §5
[ ] Inspect tests/ — coverage matches the table in §8
[ ] Read at least 2 wave architecture docs (suggest wave4.md, wave6.md)
```

If every box is checked and every command produced the expected output, the framework's claim of being a generic, production-grade quantization framework is verified. Anything that fails should be reported back with the exact command, expected output, and observed output.
