# Using the Python Library

> Audience: **developers** who want to integrate quantization into an
> existing training loop or notebook — without touching YAML or the CLI.

Every NeuroQuant quantizer accepts `config=None` and falls back to a
fully-defaulted `QuantizationConfig()`. That makes the library usable
out-of-the-box in a Jupyter notebook the same way you'd use
`librosa.load(...)` or `transformers.AutoModel.from_pretrained(...)`.

## 1 · The flat public API

`import neuroquant` exposes everything the framework provides:

```python
from neuroquant import (
    # Configuration (optional — every quantizer accepts None)
    QuantizationConfig,

    # Quantizers
    PTQQuantizer, AWQQuantizer, GPTQQuantizer,
    SmoothQuantQuantizer, SmoothQuantGPTQQuantizer,
    QATTrainer, AdaroundOptimizer,

    # Multi-objective search + clustering + surrogate
    NSGAIIClusterSearch, LayerClusterer, AccuracySurrogate,

    # Explainability + Pareto visualization
    XAIGenerator, ParetoAnalyzer, ParetoVisualizer,
    plot_error_attribution, plot_sensitivity_heatmap,
)
```

No deep imports needed; nothing is hidden behind `neuroquant.subpkg.module`.

## 2 · Standalone PTQ — three lines

The minimum-viable usage. Bring your own model + calibration loader,
take a quantized model:

```python
import torch
from neuroquant import PTQQuantizer

model = torch.load("checkpoint.pth", map_location="cpu")
quantizer = PTQQuantizer(model)                       # config defaults
q_model = quantizer.quantize(calib_loader, bitwidth=4)
```

That's it. `PTQQuantizer` internally:

1.  Builds a default `QuantizationConfig()` (`device="auto"`,
    `calibration_batches=20`, KL/MSE strategy for I/O / intermediate
    layers, etc.).
2.  Detects every quantizable Conv2d / Linear weight.
3.  Runs KL-divergence calibration on the I/O layers, MSE on the rest,
    via forward hooks against your `calib_loader`.
4.  Applies symmetric quantize→dequantize per layer with per-channel
    scales for Conv2d and per-tensor for Linear.
5.  Returns a deep-copied model — your original `model` is untouched.

## 3 · Detection example — Faster R-CNN, end-to-end

Notebook-style quantization of a torchvision detection model, no YAML:

```python
import torch
import torchvision
from neuroquant import PTQQuantizer

# 1. Build a detection model. NeuroQuant has zero hardcoded
#    architecture assumptions — anything that's an ``nn.Module``
#    works, including torchvision detectors.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None,
    num_classes=21,           # Pascal VOC
)

# 2. Give the model a small calibration set.
#    Detection datasets yield (image_tensor, target_dict) tuples; the
#    PTQ calibration only reads ``batch[0]`` so any DataLoader that
#    produces images works here.
from torch.utils.data import DataLoader
calib_loader = DataLoader(my_voc_calib_subset, batch_size=2)

# 3. Quantize. Every quantizer accepts a bare model — config-less.
quantizer = PTQQuantizer(model)
q_model = quantizer.quantize(calib_loader, bitwidth=8)

# 4. Use the quantized model exactly like the FP32 one — torchvision
#    detection contract is preserved (List[Dict[str, Tensor]] output).
q_model.eval()
with torch.no_grad():
    predictions = q_model([sample_image_tensor])
print(predictions[0]["boxes"].shape, predictions[0]["scores"].max())
```

!!! info "Why this works without configuration"

    The detection contract is enforced by `torchvision`, not by
    NeuroQuant. The quantizer just replaces weight tensors — it
    never touches the head / heads. So the model still emits the
    canonical `List[Dict[str, Tensor]]` and downstream eval / NMS /
    Grad-CAM all work unchanged.

!!! warning "AWQ is not supported on detection models"

    Use `PTQQuantizer`, `GPTQQuantizer`, or `QATTrainer` for detection.
    AWQ's per-layer α search concatenates calibration activations
    along the batch axis, which assumes a **static activation shape**
    across batches. Detection models (Faster R-CNN, RetinaNet, …) emit
    variable-size tensors from the RPN / RoI heads — the number of
    proposals depends on the image, so `torch.cat` along the batch
    dimension fails. This is a property of the AWQ algorithm itself
    (designed for static-shape LLM / vision-backbone graphs), not a
    bug in NeuroQuant. Calling `AWQQuantizer(...).quantize(...)` with
    `task="detection"` raises a clear `NotImplementedError` pointing
    you back to PTQ / QAT.

    Segmentation is fine — `OrderedDict({"out": ...})` has a static
    spatial shape per batch, so AWQ applies normally.

## 4 · Segmentation example — DeepLabV3 + Grad-CAM

```python
import torch
import torchvision
from neuroquant import GPTQQuantizer, XAIGenerator

model = torchvision.models.segmentation.deeplabv3_resnet101(
    weights=None,
    weights_backbone=None,
    num_classes=21,
)

# GPTQ uses a small calibration set to build the inverse Hessian and
# round columns optimally. Same API shape as PTQ.
quantizer = GPTQQuantizer(model)
q_model = quantizer.quantize(calib_loader, bitwidth=4, num_batches=8)

# Grad-CAM on a segmentation model — the XAI module auto-dispatches
# on the output shape (OrderedDict({"out": ...})) and computes the
# backward against the sum of the per-pixel mask for the target class.
xai = XAIGenerator(config=None)                 # also config-optional
result = xai.run(
    fp32_model=model,
    quantized_models={"GPTQ_INT4": q_model},
    test_images=sample_batch,                   # [N, C, H, W]
    test_labels=sample_labels,
    output_dir="./xai_segmentation",
)
print(result["consistency_scores"])
# {'GPTQ_INT4': 0.91}  ← Pearson correlation vs FP32 attention
```

### Bring your own segmentation model + dataset

> Added in v2.2.0 — segmentation now runs **end-to-end** (FP32 baseline,
> NSGA-II, AdaRound, GPTQ/SmoothQuant/AWQ, Pareto) and is scored by **mIOU**.

Three contracts make *any* segmentation model work:

1. **Model output.** `forward` must return per-pixel logits as a bare
   `[B, C, H, W]` tensor **or** a dict carrying them under `"out"`
   (`OrderedDict({"out": [B, C, H, W]})`, the torchvision shape). The loss
   bridge and the mIOU metric both unwrap `output.get("out", first_value)`.
   Load it via `model.name` (a torchvision seg model — optionally
   `model.pretrained: true`) or `model.class: "pkg.mod.MyNet"`, with
   `model.num_classes` = the output channel count.

2. **Dataset output.** Each item is `(image[3,H,W] float, mask[H,W] long)`,
   where the mask holds class indices `0..C-1` and `255` for ignore/void
   pixels (`[B,1,H,W]` masks are auto-squeezed). Wire it with
   `dataset.class: "pkg.mod.MyDataset"`. Segmentation needs the mask
   resized **in lockstep** with the image (image→bilinear, mask→nearest),
   so your dataset must apply its **own** synchronized transform — the
   generic loader only transforms the image. The bundled
   `neuroquant.data.voc_segmentation.VOCSegmentationDataset` is a copyable
   template (and works out of the box for Pascal VOC).

3. **Config.** Just set `task: segmentation`, an `input_shape` that matches
   what your dataset returns, and your model + dataset — **you do not need
   to touch `phases`.** The pipeline is task-aware: it auto-skips QAT
   (Phase 1e, which is classification-only) for non-classification tasks
   and reports **AdaRound** in its place. So the stock 9-phase default
   "just works".

```yaml
model:
  class: "mypkg.MySegNet"        # or  name: deeplabv3_resnet50  + pretrained: true
  task: segmentation             # ← the only task-specific switch you need
  num_classes: 21
  input_shape: [3, 512, 512]
dataset:
  class: "mypkg.MySegDataset"    # yields (image[3,512,512], mask[512,512] long)
  path: "./data"
  batch_size: 8
methods: [ptq, gptq, smoothquant, awq]
hyperparams:
  onnx_export_enabled: false     # optional — dict-output ONNX export is heavy for seg
  hardware_aware_search: false
```

That's it — no `phases:` block required. QAT is dropped automatically (you'll
see `Phase 1e … [SKIPPED — QAT is classification-only]` in the log).

!!! note "What you get / what to know"
    - **mIOU is reported in the `Top-1` column** (and the `top1` field
      everywhere); pixel accuracy lands in `Top-5`. NSGA-II optimises the
      mIOU drop directly.
    - The ignore label is **255** (the VOC / Cityscapes / ADE convention).
    - Reported numbers are **weight-only** quantization (activations stay
      FP32 in the PyTorch eval).
    - AWQ works on segmentation (fixed input → static activation shapes)
      but is memory-hungry at high resolution; it is skipped gracefully if
      it runs out of memory.

## 5 · Mix and match — surrogate-NSGA + your own training loop

The search and the QAT trainer are also library objects. You can
script a custom flow that runs NSGA-II to pick a per-layer bitwidth
assignment, then QAT-finetunes the resulting config inside your own
training script:

```python
from neuroquant import (
    QuantizationConfig, LayerClusterer, NSGAIIClusterSearch, QATTrainer,
)
from neuroquant.quantization.hessian_clustering import HessianComputer

# The search, clusterer, and QAT trainer take a real config (unlike the
# quantizers, they don't default ``config=None`` to ``QuantizationConfig()``).
cfg = QuantizationConfig()

# 1. Hessian / Fisher sensitivity, then cluster layers into
#    HIGH / MEDIUM / LOW tiers — tells NSGA which layers are too
#    sensitive to push below INT8.
hessian = HessianComputer(model, cfg).compute_hessian(calib_loader)
cluster_result = LayerClusterer(model, hessian, cfg).create_clusters()

# 2. Surrogate-Assisted NSGA-II. Defaults to per-layer mode with
#    sensitivity-weighted mutation. Returns a Pareto front of
#    mixed-precision configs. Score candidates on a held-out loader
#    (the pipeline uses the dedicated ``search`` split — never test).
nsga = NSGAIIClusterSearch(
    model,
    cluster_result["cluster_assignments"],
    cfg,
    hessian_diag=hessian,
)
pareto = nsga.search(search_loader, fp32_accuracy=92.5)
best_config = pareto["solutions"][0]["bitwidth_assignment"]

# 3. QAT fine-tune the winning config, with the FP32 model as the
#    knowledge-distillation teacher. The bitwidth assignment + teacher
#    go to the constructor; ``train`` runs the loop and returns the
#    best-val model.
qat = QATTrainer(
    model, best_config, cfg,
    teacher=model,                 # FP32 teacher for KD
    calib_loader=calib_loader,     # initialises the activation observers
)
qat_result = qat.train(train_loader, val_loader)
final_model = qat_result["model"]
```

## 6 · Regression — continuous outputs

> Added in v2.1.

NeuroQuant's task router treats regression as a first-class task. The
loss bridge swaps `CrossEntropyLoss` for `MSELoss`, the metrics layer
emits `RMSE` / `MAE` / `R²` instead of Top-1, and NSGA-II keeps its
`fp32 − quant` objective math unchanged because the metric helper puts
`-RMSE` in the canonical `top1` slot (so "higher is better" still
holds and quantization-induced RMSE growth becomes the equivalent of
accuracy loss).

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from neuroquant import PTQQuantizer, QuantizationConfig
from neuroquant.utils.metrics import compute_regression_metrics

# 1. A regression head — anything that outputs ``[B, K]`` continuous values.
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, 64),
    nn.ReLU(),
    nn.Linear(64, 1),                       # single continuous output
)

# 2. Bind ``task='regression'`` so the loss bridge and metric router
#    pick MSE + RMSE instead of CE + Top-1.
cfg = QuantizationConfig(task="regression")

# 3. Calibration + eval loaders — the targets are floats, not class ids.
calib_loader = DataLoader(
    TensorDataset(torch.randn(256, 3, 32, 32), torch.randn(256, 1)),
    batch_size=32,
)
eval_loader = DataLoader(
    TensorDataset(torch.randn(64, 3, 32, 32), torch.randn(64, 1)),
    batch_size=32,
)

# 4. Quantize. Same library API, same three-line shape.
q_model = PTQQuantizer(model, cfg).quantize(calib_loader, bitwidth=8)

# 5. Headline metrics include RMSE / MAE / R²; ``top1`` is ``-RMSE``
#    so any task-agnostic caller still sees "higher = better".
metrics = compute_regression_metrics(q_model, eval_loader, torch.device("cpu"))
print(f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.3f}")
```

!!! note "Why the `-RMSE` trick"

    NSGA-II's accuracy-loss objective is `fp32_top1 − quant_top1`.
    Storing `-RMSE` in `top1` makes that expression evaluate to
    `(-fp32_rmse) − (-quant_rmse) = quant_rmse − fp32_rmse` — exactly
    "extra error introduced by quantization", with the same
    "lower-is-better" semantics. Zero NSGA code had to change.

## 7 · NLP / HuggingFace transformers

> Added in v2.1. Requires the optional `[nlp]` extras.

```bash
pip install neuroquant[nlp]
```

NLP support hooks into the same task-aware bridge: set `task='nlp'`
and the loss function becomes `model(**inputs, labels=labels).loss`
(the canonical HuggingFace contract). The data layer ships a tiny
HuggingFace dataset loader that activates whenever `dataset_name`
starts with `hf:` or `task == 'nlp'`.

```python
from neuroquant import PTQQuantizer, QuantizationConfig
from neuroquant.data import GenericDatasetLoader      # uses [nlp] extras when needed
from transformers import AutoModelForSequenceClassification

# 1. Load any HuggingFace classifier — BERT, RoBERTa, DistilBERT, …
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2,
)

# 2. Configure the NLP task. ``dataset_name="hf:imdb"`` tells the loader
#    to pull IMDB via ``datasets.load_dataset``; the tokenizer is
#    inferred from ``model_name`` (defaults to bert-base-uncased
#    if not set).
cfg = QuantizationConfig(
    task="nlp",
    model_name="distilbert-base-uncased",
    dataset_name="hf:imdb",
    num_classes=2,
    batch_size=16,
)
cfg.hyperparams.nlp_max_seq_len = 128         # default; bump for long-form text

# 3. Build the calibration loader. Each batch is a single dict
#    ``{"input_ids": ..., "attention_mask": ..., "labels": ...}``
#    — exactly what HF models consume natively.
loader = GenericDatasetLoader(cfg).get_calibration_loader(num_batches=20)

# 4. Quantize. The loss bridge auto-splats the dict into
#    ``model(**x, labels=y)`` and reads ``.loss`` off the output.
q_model = PTQQuantizer(model, cfg).quantize(loader, bitwidth=8)
```

!!! tip "Custom HuggingFace pipelines"

    You don't have to use the bundled HF loader — any iterable that
    yields `({"input_ids": …, "attention_mask": …}, labels)` tuples
    or a single dict with `labels` inside it will work. The Hessian
    estimator and Phase 1c NSGA evaluator both call the same
    task-aware bridge, so swap in a custom DataLoader and everything
    downstream just works.

!!! warning "AWQ on transformers"

    AWQ works on transformer **encoders** with static sequence
    lengths (BERT, DistilBERT, …) but fails on decoder LLMs whose
    KV-cache changes the activation shape across calibration
    batches — same root cause as the detection guard. Prefer
    `PTQQuantizer` or `GPTQQuantizer` for causal LM quantization
    if you hit `torch.cat` errors.

## 8 · Vision Transformers (ViT) and Attention Rollout

> Added in v2.1.

`XAIGenerator` auto-detects when the FP32 model is a Vision Transformer
(presence of `nn.MultiheadAttention` plus absence of `nn.Conv2d`) and
swaps Grad-CAM for **Attention Rollout** (Abnar & Zuidema, 2020). One
forward pass, no extra dependencies, no Conv2d required — and the
output is the same `np.ndarray[H, W] ∈ [0, 1]` heatmap that the rest
of the XAI pipeline already consumes.

```python
import torch
from neuroquant import PTQQuantizer, QuantizationConfig, XAIGenerator
from torchvision.models import vit_b_16

# 1. Any ViT — torchvision, timm, or a custom one.
model = vit_b_16(weights=None, num_classes=10)

# 2. Quantize as usual.
cfg = QuantizationConfig(task="classification", num_classes=10)
q_model = PTQQuantizer(model, cfg).quantize(calib_loader, bitwidth=8)

# 3. Run XAI. The generator detects the ViT and routes to Attention
#    Rollout automatically — Grad-CAM would crash because there's no
#    last Conv2d to hook.
xai = XAIGenerator(cfg)
result = xai.run(
    fp32_model=model,
    quantized_models={"PTQ_INT8": q_model},
    test_images=sample_batch,           # [N, 3, 224, 224]
    test_labels=sample_labels,
    output_dir="./xai_vit",
)
print(result["consistency_scores"])
# {'PTQ_INT8': 0.93}  ← Pearson correlation vs FP32 rollout
```

You can also drive the explainer directly when you want one heatmap
without spinning up the full pipeline:

```python
from neuroquant.xai.explainability import (
    AttentionRolloutExplainer, is_vision_transformer,
)

assert is_vision_transformer(model)             # sanity check
rollout = AttentionRolloutExplainer(model, device=torch.device("cuda"))
heatmap = rollout.compute(image_tensor)         # [H, W] in [0, 1]
```

!!! info "Hybrid models (Conv-stem + transformer body)"

    Models that mix Conv2d with attention (ConvNeXt-style hybrids,
    Swin's patch-embed Conv layer) intentionally fall through to the
    Grad-CAM path — the convolutional stem produces a more useful
    spatial signal than rollout on a partially-attended forward.
    Override via `XAIGenerator.run(target_layer_name="...")` if you
    want to force Grad-CAM on a specific module.

## 9 · Mixing library + pipeline

Library and pipeline are *complementary*, not exclusive. A common
pattern: drive the heavy phases (clustering + NSGA + QAT) from the CLI
for reproducibility, then load the resulting Pareto front into a
notebook for analysis:

```python
import json

with open("artifacts/pareto_summary.json") as f:
    summary = json.load(f)

print(f"FP32 baseline: {summary['fp32_top1']:.2f}% top-1, "
      f"{summary['fp32_size_mb']:.2f} MiB · "
      f"hypervolume {summary.get('hypervolume', 0):.3f}")

# Pick the most compressed method within 1 pp of the best top-1 — the
# same "knee" idea the CLI's ParetoAnalyzer applies, straight off the
# summary the pipeline already wrote.
methods = summary["methods"]
best_top1 = max(m["top1"] for m in methods)
within = [m for m in methods if best_top1 - m["top1"] <= 1.0]
knee = min(within, key=lambda m: m.get("onnx_size_mb") or m["size_mb"])
print(f"Knee method: {knee['method']} "
      f"({knee['top1']:.2f}% top-1, {knee['size_mb']:.2f} MiB)")
```

Or the inverse direction: do a quick standalone notebook experiment to
pick a baseline, then commit those choices to `config.yaml` and run the
full pipeline.

## 10 · Where each class lives

Even though the flat import works, knowing the underlying layout helps
when you read the API reference:

```text
neuroquant/
├── quantization/
│   ├── ptq.py             → PTQQuantizer
│   ├── awq.py             → AWQQuantizer
│   ├── gptq.py            → GPTQQuantizer
│   ├── smoothquant.py     → SmoothQuantQuantizer
│   ├── smoothquant_gptq.py→ SmoothQuantGPTQQuantizer
│   ├── adaround.py        → AdaroundOptimizer
│   ├── qat.py             → QATTrainer
│   ├── nsga_ii_search.py  → NSGAIIClusterSearch
│   ├── hessian_clustering.py → LayerClusterer
│   └── surrogate.py       → AccuracySurrogate
├── xai/explainability.py  → XAIGenerator
├── visualization/         → ParetoAnalyzer, plot_* helpers
└── config.py              → QuantizationConfig
```

[:octicons-arrow-right-24: Continue to the auto-generated API reference](api_reference.md)
