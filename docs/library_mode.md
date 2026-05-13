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

## 5 · Mix and match — surrogate-NSGA + your own training loop

The search and the QAT trainer are also library objects. You can
script a custom flow that runs NSGA-II to pick a per-layer bitwidth
assignment, then QAT-finetunes the resulting config inside your own
training script:

```python
from neuroquant import (
    LayerClusterer, NSGAIIClusterSearch, QATTrainer,
)

# 1. Hessian clustering — tells NSGA which layers are too sensitive
#    to push below INT8.
clusterer = LayerClusterer(config=None)
hessian = clusterer.compute_hessian(model, calib_loader)
cluster_result = clusterer.create_clusters(hessian)

# 2. Surrogate-Assisted NSGA-II. Defaults to per-layer mode with
#    sensitivity-weighted mutation. Returns a Pareto front of
#    mixed-precision configs.
nsga = NSGAIIClusterSearch(
    model,
    cluster_result["cluster_assignments"],
    config=None,
    hessian_diag=hessian,
)
pareto = nsga.search(val_loader, fp32_accuracy=92.5)
best_config = pareto["solutions"][0]["bitwidth_assignment"]

# 3. QAT fine-tune the winning config, with a knowledge-distillation
#    teacher pointing at the FP32 baseline.
qat = QATTrainer(model, config=None)
qat_result = qat.run(
    bitwidth_assignment=best_config,
    train_loader=train_loader,
    val_loader=val_loader,
    teacher_model=model,                # FP32 teacher
)
final_model = qat_result["model"]
```

## 6 · Mixing library + pipeline

Library and pipeline are *complementary*, not exclusive. A common
pattern: drive the heavy phases (clustering + NSGA + QAT) from the CLI
for reproducibility, then load the resulting Pareto front into a
notebook for analysis:

```python
import json
from neuroquant import ParetoAnalyzer

with open("artifacts/pareto_summary.json") as f:
    pareto = json.load(f)

analyzer = ParetoAnalyzer(pareto["solutions"], baseline_accuracy=92.5)
knee = analyzer.find_knee_point()
print(f"Knee solution: {knee['solution_id']} "
      f"({knee['accuracy']:.2f}%, {knee['model_size_mb']:.2f} MiB)")
```

Or the inverse direction: do a quick standalone notebook experiment to
pick a baseline, then commit those choices to `config.yaml` and run the
full pipeline.

## 7 · Where each class lives

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
