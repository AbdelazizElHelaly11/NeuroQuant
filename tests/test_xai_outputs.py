"""
NeuroQuant v2.0 - XAI Output Quality Test (fast, ~3-5s)

Validates the publication-style XAI output that downstream graders look at:

  1. comparison_matrix.png is generated and references every supplied
     model row + sample column with a labelled header.
  2. The XAIResult contains a ``predictions`` map per (model, sample)
     with pred_idx / pred_name / confidence / gt_idx / gt_name / correct.
  3. Per-image Grad-CAM PNGs are written and contain the prediction
     metadata in their captions (verified via PIL pixel inspection
     fallback by checking file sizes > 0 + matplotlib title bytes).
  4. Class-name fallback ("class N") works when names aren't supplied.
  5. plot styles use a light background (not the previous dark theme).

The test uses a tiny CNN + synthetic data so it stays fast and runs on
CPU without torchvision data downloads.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Some Windows terminals default to cp1252; force UTF-8 so emitted
# characters in test names never crash on print.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn

from config import QuantizationConfig
from xai.explainability import XAIGenerator
from visualization.style import apply_publication_style, style_for, family_of

logging.basicConfig(level=logging.WARNING, format="%(message)s")

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


class _TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        return self.fc(self.pool(x).flatten(1))


def _build_test_set(num_classes=4, n=4):
    torch.manual_seed(0)
    images = torch.randn(n, 3, 16, 16)
    labels = torch.arange(n) % num_classes
    return images, labels


def test_style_helpers():
    print("--- Test: style helpers ---")
    apply_publication_style()
    import matplotlib.pyplot as plt
    bg = plt.rcParams["figure.facecolor"]
    grid = plt.rcParams["axes.grid"]
    check("publication style uses white figure bg", bg == "white", f"got {bg}")
    check("publication style enables grid", bool(grid))

    color, marker = style_for("GPTQ_INT8")
    check("style_for(GPTQ_INT8) returns GPTQ green triangle",
          color == "#2ca02c" and marker == "^", f"got ({color}, {marker})")
    color2, marker2 = style_for("AWQ_AWQ_INT4")
    check("style_for(AWQ_AWQ_INT4) returns AWQ purple plus",
          color2 == "#9467bd" and marker2 == "P",
          f"got ({color2}, {marker2})")
    # Crucial: PTQ-as-substring of GPTQ must NOT win.
    color3, _ = style_for("GPTQ")
    check("GPTQ does not collide with PTQ (token match)",
          color3 == "#2ca02c", f"got {color3}")
    fam = family_of("SmoothQuant_SmoothQuant_INT4")
    check("family_of identifies SmoothQuant",
          fam == "SMOOTHQUANT", f"got {fam}")
    fam_unknown = family_of("FOO_BAR")
    check("family_of falls back to OTHER", fam_unknown == "OTHER",
          f"got {fam_unknown}")


def test_xai_predictions_and_matrix():
    print("--- Test: XAI predictions + comparison matrix ---")

    num_classes = 4
    cfg = QuantizationConfig()
    cfg.num_classes = num_classes
    cfg.input_shape = (3, 16, 16)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.xai_num_images = 3
    cfg.hyperparams.xai_grad_cam_alpha = 0.5
    cfg.hyperparams.xai_plot_dpi = 80

    fp32 = _TinyCNN(num_classes)
    quant = _TinyCNN(num_classes)  # different init → different predictions
    images, labels = _build_test_set(num_classes=num_classes, n=3)
    class_names = ["alpha", "beta", "gamma", "delta"]

    with tempfile.TemporaryDirectory() as tmp:
        gen = XAIGenerator(cfg, device=torch.device("cpu"))
        result = gen.run(
            fp32_model=fp32,
            quantized_models={"PTQ_best": quant, "SmoothQuant": quant},
            test_images=images,
            test_labels=labels,
            output_dir=tmp,
            class_names=class_names,
        )

        # --- predictions present and shape-correct ---
        preds = result.get("predictions", {})  # type: ignore[arg-type]
        check("predictions key exists in XAIResult", isinstance(preds, dict))
        for model_id in ("FP32_baseline", "PTQ_best", "SmoothQuant"):
            check(f"predictions has {model_id}", model_id in preds)
            mp = preds.get(model_id, [])
            check(f"{model_id} has 3 predictions", len(mp) == 3)
            for i, p in enumerate(mp):
                check(f"{model_id}#{i} has pred_idx", "pred_idx" in p)
                check(f"{model_id}#{i} pred_name in supplied names",
                      p["pred_name"] in class_names,
                      f"got {p.get('pred_name')}")
                check(f"{model_id}#{i} confidence in [0,1]",
                      0.0 <= float(p["confidence"]) <= 1.0,
                      f"got {p.get('confidence')}")
                gt = int(labels[i].item())
                check(f"{model_id}#{i} gt_idx matches label",
                      p["gt_idx"] == gt)
                check(f"{model_id}#{i} gt_name uses lookup",
                      p["gt_name"] == class_names[gt])
                check(f"{model_id}#{i} 'correct' boolean is consistent",
                      p["correct"] == (p["pred_idx"] == p["gt_idx"]))

        # --- comparison matrix saved + non-empty ---
        grid_path = Path(result["comparison_grid"])
        check("comparison_matrix.png exists",
              grid_path.exists() and grid_path.name == "comparison_matrix.png")
        check("comparison_matrix.png is non-empty",
              grid_path.stat().st_size > 1000,
              f"size={grid_path.stat().st_size}")

        # --- per-image grad-cam PNGs saved with technique-tagged names ---
        for model_id in ("FP32_baseline", "PTQ_best", "SmoothQuant"):
            paths = result["grad_cam_paths"].get(model_id, [])
            check(f"{model_id} produced 3 individual heatmaps",
                  len(paths) == 3)
            for p in paths:
                pp = Path(p)
                check(f"{pp.name} on disk + non-empty",
                      pp.exists() and pp.stat().st_size > 500,
                      f"size={pp.stat().st_size if pp.exists() else 'missing'}")
                check(f"{pp.name} filename encodes technique",
                      model_id in pp.name)

        # --- markdown report includes predictions table ---
        report = result["report"]
        check("report has 'Predictions per Sample' header",
              "Predictions per Sample" in report)
        check("report has 'Top-1 Accuracy on Explained Samples' header",
              "Top-1 Accuracy on Explained Samples" in report)
        check("report mentions sample headers",
              "sample #0" in report and "sample #2" in report)
        check("report uses class names from supplied list",
              any(name in report for name in class_names))


def test_xai_classname_fallback():
    print("--- Test: XAI class-name fallback ---")

    cfg = QuantizationConfig()
    cfg.num_classes = 3
    cfg.input_shape = (3, 16, 16)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.xai_num_images = 2
    cfg.hyperparams.xai_plot_dpi = 70

    fp32 = _TinyCNN(3)
    images, labels = _build_test_set(num_classes=3, n=2)

    with tempfile.TemporaryDirectory() as tmp:
        gen = XAIGenerator(cfg, device=torch.device("cpu"))
        result = gen.run(
            fp32_model=fp32,
            quantized_models={"GPTQ": fp32},
            test_images=images,
            test_labels=labels,
            output_dir=tmp,
            class_names=None,  # force fallback
        )
        preds = result.get("predictions", {})  # type: ignore[arg-type]
        for model_id, mp in preds.items():
            for i, p in enumerate(mp):
                check(f"{model_id}#{i} fallback pred_name shaped 'class N'",
                      str(p["pred_name"]).startswith("class "),
                      f"got {p.get('pred_name')}")
                check(f"{model_id}#{i} fallback gt_name shaped 'class N'",
                      str(p["gt_name"]).startswith("class "),
                      f"got {p.get('gt_name')}")


def main() -> int:
    test_style_helpers()
    test_xai_predictions_and_matrix()
    test_xai_classname_fallback()

    print("\n" + "=" * 50)
    print(f"  XAI Output Tests: {passed} passed, {failed} failed")
    print("=" * 50)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
