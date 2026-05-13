"""
NeuroQuant v2.0 - XAI Explainability Pipeline (Phase 3)

Generates Grad-CAM heatmaps and SHAP explanations to visualise
how quantization affects model attention and feature importance.

Key components:
    1. GradCAMExplainer: per-layer attention heatmaps with STE-safe hooks
    2. SHAPExplainer: gradient-based feature importance (optional, graceful)
    3. XAIGenerator: orchestrates multi-model comparison + consistency scoring

Output enrichments (added in v2.0):
    - Every per-image figure carries a caption with model name, predicted
      class (name + confidence) and ground-truth class. ✓ / ✗ correctness
      indicator if a label is supplied.
    - The comparison_matrix.png is a fully-labelled grid: rows are models,
      columns are sample images with their GT label, and each cell shows
      the technique's prediction inline.
    - Light publication theme via ``visualization.style``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroquant.config import QuantizationConfig, XAIResult

logger = logging.getLogger("neuroquant")

# Optional dependency checks
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output-type dispatch (classification / detection / segmentation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Torchvision returns three different shapes depending on the task:
#
#   * classification — ``Tensor[B, num_classes]`` logits.
#   * detection      — ``List[Dict[str, Tensor]]`` where each dict has
#                      ``boxes``, ``labels``, ``scores`` (eval mode).
#   * segmentation   — ``OrderedDict({"out": Tensor[B, C, H, W], ...})``.
#
# The helpers below collapse those into a single contract the
# Grad-CAM and SHAP code paths can consume: ``infer_task_kind`` tags
# the output, ``compute_backward_target`` returns a scalar tensor to
# ``.backward()`` on (which triggers the activation/gradient hooks
# uniformly), and ``predict_from_output`` extracts a representative
# ``(class_idx, confidence)`` pair for the caption/report.


def infer_task_kind(output: Any) -> str:
    """Return ``"classification" | "detection" | "segmentation" | "unknown"``.

    Dispatches purely on the *runtime* shape of the forward output, so
    it works whether or not the caller knows the configured task — the
    same helper drives both Grad-CAM and SHAP and stays correct after
    quantization wrappers (which may not preserve the original module
    identity).
    """
    # Detection: a non-empty list of dicts with the canonical keys.
    if isinstance(output, list) and output and isinstance(output[0], dict):
        keys = output[0].keys()
        if "scores" in keys or "labels" in keys or "boxes" in keys:
            return "detection"
    # Segmentation: dict / OrderedDict carrying an ``out`` tensor.
    if isinstance(output, dict) and "out" in output:
        if isinstance(output["out"], torch.Tensor):
            return "segmentation"
    # Classification: a plain 2-D logits tensor.
    if isinstance(output, torch.Tensor):
        return "classification"
    return "unknown"


def compute_backward_target(
    output: Any,
    target_class: Optional[int],
    *,
    kind: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Reduce a task-specific forward output to a scalar for ``.backward()``.

    Returning a scalar (rather than calling ``.backward()`` ourselves)
    keeps the contract minimal — the caller decides whether to retain
    the graph, zero grads, etc. The scalar is selected so that the
    activation/gradient hooks the Grad-CAM machinery already installed
    fire correctly:

      * ``classification`` → ``output[0, target_class]``
      * ``detection``      → score of the highest-scoring box whose
                              ``labels[i] == target_class``; falls back
                              to the overall top-scoring detection when
                              the class is absent or ``target_class is None``.
      * ``segmentation``   → ``output['out'][0, target_class, :, :].sum()``

    Returns ``None`` when no usable scalar can be extracted (e.g.
    detection model returned an empty box list, or the output type is
    unrecognised); callers should treat this as a "skip" and log.
    """
    if kind is None:
        kind = infer_task_kind(output)

    if kind == "classification":
        logits = output if isinstance(output, torch.Tensor) else None
        if logits is None or logits.dim() < 2:
            return None
        cls = int(target_class) if target_class is not None else int(
            logits.argmax(dim=1).item()
        )
        cls = max(0, min(cls, logits.size(1) - 1))
        return logits[0, cls]

    if kind == "detection":
        if not isinstance(output, list) or not output:
            return None
        det = output[0]
        scores = det.get("scores")
        labels = det.get("labels")
        if not isinstance(scores, torch.Tensor) or scores.numel() == 0:
            return None
        # Prefer the highest-scoring box for the requested class. If
        # the class isn't present (or no class requested) fall back to
        # the overall top-scoring box — guarantees a usable gradient
        # signal even for sparse detections.
        if (
            target_class is not None
            and isinstance(labels, torch.Tensor)
            and labels.numel() == scores.numel()
        ):
            mask = labels == int(target_class)
            if mask.any():
                masked_scores = scores.clone()
                masked_scores[~mask] = float("-inf")
                best_idx = int(masked_scores.argmax().item())
                return scores[best_idx]
        best_idx = int(scores.argmax().item())
        return scores[best_idx]

    if kind == "segmentation":
        if not isinstance(output, dict) or "out" not in output:
            return None
        seg = output["out"]
        if not isinstance(seg, torch.Tensor) or seg.dim() != 4:
            return None
        num_classes = seg.size(1)
        if target_class is None:
            # Pick the class with the largest mean spatial score.
            target_class = int(seg[0].mean(dim=(1, 2)).argmax().item())
        cls = max(0, min(int(target_class), num_classes - 1))
        return seg[0, cls, :, :].sum()

    return None


def predict_from_output(
    output: Any,
    *,
    kind: Optional[str] = None,
) -> Tuple[int, float]:
    """Best-effort ``(pred_idx, confidence)`` extraction across tasks.

    ``confidence`` is normalised into ``[0, 1]``:
      * classification → softmax probability of the argmax class.
      * detection      → top detection's raw score (already in [0, 1]
                          for torchvision detectors).
      * segmentation   → mean softmax probability of the dominant class
                          across spatial dims (i.e. how confident the
                          model is about its majority prediction).
    """
    if kind is None:
        kind = infer_task_kind(output)

    if kind == "classification" and isinstance(output, torch.Tensor):
        probs = F.softmax(output, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item())
        return pred_idx, confidence

    if kind == "detection" and isinstance(output, list) and output:
        det = output[0]
        scores = det.get("scores")
        labels = det.get("labels")
        if isinstance(scores, torch.Tensor) and scores.numel() > 0:
            best = int(scores.argmax().item())
            label = (
                int(labels[best].item())
                if isinstance(labels, torch.Tensor) and labels.numel() > best
                else -1
            )
            return label, float(scores[best].item())
        return -1, 0.0

    if kind == "segmentation" and isinstance(output, dict) and "out" in output:
        seg = output["out"]
        if isinstance(seg, torch.Tensor) and seg.dim() == 4:
            probs = F.softmax(seg, dim=1)
            mean_probs = probs[0].mean(dim=(1, 2))
            pred_idx = int(mean_probs.argmax().item())
            return pred_idx, float(mean_probs[pred_idx].item())

    return -1, 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Grad-CAM Implementation (from scratch, no external deps)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GradCAMExplainer:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.

    Computes which spatial regions of feature maps most influenced
    the model's prediction for a given class. Works on any Conv2d layer.

    Algorithm (Selvaraju et al., 2017):
        1. Forward pass: capture activations at target layer
        2. Backward pass: capture gradients w.r.t. those activations
        3. Weights = global-avg-pool of gradients (per channel)
        4. Heatmap = ReLU(sum(weight_k * activation_k))
        5. Normalise to [0, 1], resize to input size
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def compute(
        self,
        image: torch.Tensor,
        target_layer: nn.Module,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Works across all three computer-vision paradigms — the backward
        pass uses the task-aware ``compute_backward_target`` so the
        installed activation/gradient hooks fire regardless of whether
        the model returns logits, a torchvision detection list, or a
        segmentation OrderedDict.

        Args:
            image: Input tensor [1, C, H, W] (normalised).
            target_layer: Conv2d module to visualise.
            target_class: Class index for gradient computation.
                * classification → index into the logits dim.
                * detection      → highest-scoring box with this label
                                   (falls back to overall top score when
                                   the class is absent).
                * segmentation   → channel index into ``output['out']``;
                                   gradient is the sum over the whole
                                   spatial mask for that class.
                ``None`` = use the model's own argmax (per task).

        Returns:
            Heatmap as numpy array [H, W] in [0, 1].
        """
        self._activations = None
        self._gradients = None

        # Capture activation + gradient via a single forward hook that
        # attaches a tensor-level ``register_hook`` to the output. We
        # deliberately avoid ``register_full_backward_hook`` because it
        # wraps the module output in a ``BackwardHookFunction`` view,
        # which conflicts with any downstream in-place op (e.g.
        # ``ReLU6(inplace=True)`` in MobileNetV2 — especially after QAT
        # folds Conv-BN to an Identity, leaving the inplace activation
        # directly consuming the captured tensor).
        fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._hooks = [fwd_hook]

        try:
            self.model.eval()
            image = image.to(self.device)

            # Forward pass. The output shape depends on the task family:
            #   classification → Tensor[B, C]
            #   detection      → List[Dict[str, Tensor]]
            #   segmentation   → OrderedDict({"out": Tensor[B, C, H, W]})
            # ``compute_backward_target`` collapses any of these to a
            # single scalar whose backward triggers the activation and
            # gradient hooks installed above, so the rest of this method
            # stays task-agnostic.
            output = self.model(image)
            kind = infer_task_kind(output)

            scalar = compute_backward_target(
                output, target_class, kind=kind,
            )
            if scalar is None:
                logger.warning(
                    "Grad-CAM: could not extract a backward target from "
                    "%s output (target_class=%s). Returning zero heatmap.",
                    kind, target_class,
                )
                return np.zeros((image.shape[2], image.shape[3]))

            # Backward pass on the task-specific scalar.
            self.model.zero_grad()
            scalar.backward(retain_graph=True)

            if self._activations is None or self._gradients is None:
                logger.warning("Grad-CAM hooks did not fire. Returning zero heatmap.")
                return np.zeros((image.shape[2], image.shape[3]))

            # Grad-CAM weights: global average pooling of gradients
            # gradients shape: [1, K, H_feat, W_feat]
            weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, K, 1, 1]

            # Weighted sum of activations
            cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

            # ReLU (only positive contributions)
            cam = F.relu(cam)

            # Normalise to [0, 1]
            cam = cam.squeeze().detach().cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)

            # Resize to input spatial dimensions
            h_in, w_in = image.shape[2], image.shape[3]
            cam_resized = np.array(
                _resize_2d(cam, (h_in, w_in))
            )

            return cam_resized

        finally:
            # Always clean up hooks
            for h in self._hooks:
                h.remove()
            self._hooks.clear()

    def _save_activation(self, module, input, output):
        """Forward hook: cache activations and register tensor-level grad hook."""
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(self._save_gradient)
        self._activations = (
            output.detach() if isinstance(output, torch.Tensor) else output
        )

    def _save_gradient(self, grad: torch.Tensor) -> None:
        """Tensor-level backward hook: cache gradient w.r.t. captured activation."""
        self._gradients = grad.detach()


def _resize_2d(arr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a 2D numpy array via bilinear interpolation using PyTorch.
    Avoids scipy dependency.
    """
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_size, mode="bilinear",
                            align_corners=False)
    return resized.squeeze().numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto the original image.

    Args:
        image: Original image [H, W, 3] in [0, 1].
        heatmap: Grad-CAM heatmap [H, W] in [0, 1].
        alpha: Heatmap transparency (0=invisible, 1=opaque).
        colormap: Matplotlib colormap name.

    Returns:
        Blended image [H, W, 3] in [0, 1].
    """
    if not HAS_MATPLOTLIB:
        # Fallback: just return the image with heatmap as red channel
        overlay = image.copy()
        overlay[:, :, 0] = np.clip(image[:, :, 0] + heatmap * alpha, 0, 1)
        return overlay

    cmap = cm.get_cmap(colormap)
    heatmap_coloured = cmap(heatmap)[:, :, :3]  # [H, W, 3] RGB
    blended = (1 - alpha) * image + alpha * heatmap_coloured
    return np.clip(blended, 0, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHAP Wrapper (optional dependency)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SHAPExplainer:
    """
    SHAP-based feature importance using GradientExplainer.

    Optional: gracefully degrades to gradient-based attribution
    if SHAP library is not installed.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def compute(
        self,
        image: torch.Tensor,
        background: torch.Tensor,
        n_samples: int = 50,
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for a single image.

        Args:
            image: Input [1, C, H, W].
            background: Background dataset [N, C, H, W] (typically 50 samples).
            n_samples: Number of background samples to use.

        Returns:
            SHAP values [C, H, W] or None if SHAP unavailable.
        """
        # SHAP's ``GradientExplainer`` is built around the classification
        # contract (tensor logits per class). For detection and
        # segmentation we go straight to the gradient*input fallback —
        # it dispatches on output type and always works.
        if not HAS_SHAP:
            return self._fallback_gradient_attribution(image)

        with torch.no_grad():
            probe = self.model(image.to(self.device))
        if infer_task_kind(probe) != "classification":
            return self._fallback_gradient_attribution(image)

        try:
            self.model.eval()
            bg = background[:n_samples].to(self.device)
            img = image.to(self.device)

            explainer = shap.GradientExplainer(self.model, bg)
            shap_values = explainer.shap_values(img)

            # shap_values is a list (one per class); take predicted class
            with torch.no_grad():
                pred = self.model(img).argmax(dim=1).item()

            if isinstance(shap_values, list):
                sv = shap_values[pred]
            else:
                sv = shap_values

            # sv shape: [1, C, H, W] -> [C, H, W]
            return np.array(sv[0])

        except Exception as e:
            logger.warning("SHAP computation failed: %s. Using fallback.", e)
            return self._fallback_gradient_attribution(image)

    def _fallback_gradient_attribution(
        self, image: torch.Tensor
    ) -> np.ndarray:
        """
        Fallback: simple gradient * input attribution.

        Faster than SHAP, always works, and dispatches on the model's
        output type so the same code path handles classification,
        detection, and segmentation. When the dispatch can't find a
        backward target (e.g. detection model that produced zero
        boxes), returns a zero attribution rather than crashing.
        """
        self.model.eval()
        img = image.to(self.device).requires_grad_(True)

        output = self.model(img)
        scalar = compute_backward_target(output, target_class=None)
        if scalar is None:
            logger.warning(
                "SHAP fallback: no backward target available (likely empty "
                "detections); returning zero attribution."
            )
            return np.zeros_like(img.detach().cpu().numpy()[0])

        self.model.zero_grad()
        scalar.backward()

        if img.grad is None:
            logger.warning(
                "SHAP fallback: input gradient was None (graph detached?); "
                "returning zero attribution."
            )
            return np.zeros_like(img.detach().cpu().numpy()[0])

        # gradient * input
        attr = (img.grad * img).detach().cpu().numpy()[0]  # [C, H, W]
        return attr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer Detection Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def auto_detect_target_layer(model: nn.Module) -> Tuple[str, nn.Module]:
    """
    Auto-detect the best layer for Grad-CAM (last Conv2d).

    For generic CNNs, we find the last Conv2d layer before any
    pooling or flatten operation. This captures the highest-level
    spatial features before they are collapsed.

    Args:
        model: Any nn.Module.

    Returns:
        (layer_name, layer_module) — the target Conv2d layer.

    Raises:
        ValueError: If no Conv2d layer found in model.
    """
    last_conv_name = None
    last_conv_module = None

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            last_conv_name = name
            last_conv_module = module

    if last_conv_module is None:
        raise ValueError(
            "No Conv2d layer found in model. Cannot compute Grad-CAM."
        )

    return last_conv_name, last_conv_module


def find_layer_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Find a module by its dotted name path."""
    for n, m in model.named_modules():
        if n == name:
            return m
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Consistency Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_heatmap_consistency(
    heatmap_fp32: np.ndarray,
    heatmap_quant: np.ndarray,
) -> float:
    """
    Compute Pearson correlation between two heatmaps.

    Measures how similarly two models attend to spatial regions.
    Range: [-1, 1] where 1 = identical attention patterns.

    Args:
        heatmap_fp32: FP32 baseline heatmap [H, W].
        heatmap_quant: Quantized model heatmap [H, W].

    Returns:
        Pearson correlation coefficient.
    """
    a = heatmap_fp32.flatten().astype(np.float64)
    b = heatmap_quant.flatten().astype(np.float64)

    if a.std() < 1e-10 or b.std() < 1e-10:
        return 0.0  # Flat heatmap = no signal

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    numerator = np.sum(a_centered * b_centered)
    denominator = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# XAIGenerator — Orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class XAIGenerator:
    """
    Generates explainability visualisations for quantization analysis.

    Orchestrates Grad-CAM + SHAP across FP32 baseline and quantized
    models, computing consistency scores and generating a comparison
    matrix that surfaces each technique's prediction per sample image.

    Usage:
        gen = XAIGenerator(config)
        result = gen.run(
            fp32_model=model,
            quantized_models={"PTQ_best": ptq, "GPTQ": gptq},
            test_images=images,            # [N, C, H, W]
            test_labels=labels,            # [N]
            class_names=["airplane", ...], # optional; falls back to indices
            output_dir="artifacts/xai",
        )
    """

    # Light publication theme — concrete colours kept here for non-axes
    # decorations (correct/incorrect badges, header strips). Global
    # rcParams come from visualization.style.apply_publication_style.
    BADGE_OK = "#2e7d32"      # green (✓)
    BADGE_BAD = "#c62828"     # red (✗)
    HEADER_BG = "#eceff1"
    HEADER_FG = "#1f3a93"
    CAPTION_BG = "#ffffff"
    CAPTION_EDGE = "#bdbdbd"

    def __init__(
        self,
        config: QuantizationConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        if device is not None:
            self.device = device
        else:
            self.device = self._resolve_device(config.hyperparams.device)

    def run(
        self,
        fp32_model: nn.Module,
        quantized_models: Dict[str, nn.Module],
        test_images: torch.Tensor,
        test_labels: torch.Tensor,
        output_dir: str = "./artifacts/xai",
        target_layer_name: Optional[str] = None,
        background_data: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
    ) -> XAIResult:
        """
        Run the full XAI pipeline.

        Args:
            fp32_model: FP32 baseline model.
            quantized_models: {model_id: quantized_model}.
            test_images: [N, C, H, W] test images.
            test_labels: [N] ground truth labels.
            output_dir: Directory for output artefacts.
            target_layer_name: Specific layer name (auto-detect if None).
            background_data: Background for SHAP (uses test_images[:50] if None).

        Returns:
            XAIResult with all paths, scores, and report.
        """
        logger.info("=" * 70)
        logger.info("Phase 3: XAI Explainability Analysis")
        logger.info("=" * 70)

        out_dir = Path(output_dir)
        grad_cam_dir = out_dir / "grad_cam"
        shap_dir = out_dir / "shap"
        grad_cam_dir.mkdir(parents=True, exist_ok=True)
        shap_dir.mkdir(parents=True, exist_ok=True)

        hp = self.config.hyperparams
        num_images = min(hp.xai_num_images, len(test_images))
        alpha = hp.xai_grad_cam_alpha
        dpi = hp.xai_plot_dpi

        # All models: FP32 + quantized
        all_models = {"FP32_baseline": fp32_model}
        all_models.update(quantized_models)

        # Resolve a class-name lookup. Falls back to "class N" when the
        # caller hasn't supplied names (e.g. synthetic datasets).
        class_lookup = self._build_class_name_lookup(
            class_names, num_classes=int(self.config.num_classes),
        )

        logger.info("  Models: %d (%s)", len(all_models),
                     ", ".join(all_models.keys()))
        logger.info("  Images: %d", num_images)
        logger.info("  Class names: %s",
                    "supplied" if class_names else "fallback to indices")

        # Auto-detect target layer from FP32 model
        if target_layer_name is None:
            layer_name, _ = auto_detect_target_layer(fp32_model)
            logger.info("  Auto-detected target layer: '%s'", layer_name)
        else:
            layer_name = target_layer_name
            logger.info("  Target layer: '%s'", layer_name)

        # Background data for SHAP
        if background_data is None:
            background_data = test_images[:min(50, len(test_images))]

        # ── Generate Grad-CAM heatmaps + record predictions ──
        logger.info("  Generating Grad-CAM heatmaps + predictions ...")
        all_heatmaps: Dict[str, List[np.ndarray]] = {}
        grad_cam_paths: Dict[str, List[str]] = {}
        # predictions[model_id][i] = {pred_idx, pred_name, confidence,
        #                              gt_idx, gt_name, correct}
        predictions: Dict[str, List[Dict[str, Any]]] = {}

        for model_id, model in all_models.items():
            model.to(self.device)
            model.eval()

            target_module = find_layer_by_name(model, layer_name)
            if target_module is None:
                try:
                    _, target_module = auto_detect_target_layer(model)
                except ValueError:
                    logger.warning("    No conv layer in '%s', skipping", model_id)
                    continue

            grad_cam = GradCAMExplainer(model, self.device)
            heatmaps: List[np.ndarray] = []
            paths: List[str] = []
            preds: List[Dict[str, Any]] = []

            for i in range(num_images):
                img = test_images[i:i+1]
                gt_idx = int(test_labels[i].item())

                # Capture model prediction + confidence BEFORE the Grad-CAM
                # backward pass so the recorded probabilities reflect the
                # untouched forward output.
                pred_idx, confidence = self._predict(model, img)
                pred_meta = {
                    "pred_idx": pred_idx,
                    "pred_name": class_lookup(pred_idx),
                    "confidence": confidence,
                    "gt_idx": gt_idx,
                    "gt_name": class_lookup(gt_idx),
                    "correct": pred_idx == gt_idx,
                }
                preds.append(pred_meta)

                heatmap = grad_cam.compute(
                    img, target_module, target_class=gt_idx,
                )
                heatmaps.append(heatmap)

                if HAS_MATPLOTLIB:
                    img_np = self._tensor_to_image(img)
                    overlay = overlay_heatmap(img_np, heatmap, alpha=alpha)
                    path = grad_cam_dir / f"{model_id}_img{i}.png"
                    self._save_image_with_prediction(
                        overlay, path, dpi=dpi,
                        model_id=model_id, image_idx=i, meta=pred_meta,
                    )
                    paths.append(str(path))

            all_heatmaps[model_id] = heatmaps
            grad_cam_paths[model_id] = paths
            predictions[model_id] = preds
            logger.info(
                "    %s: %d heatmaps · %d/%d correct",
                model_id, len(heatmaps),
                sum(1 for p in preds if p["correct"]), len(preds),
            )

        # ── Generate SHAP attributions ──
        logger.info("  Generating SHAP/gradient attributions ...")
        shap_paths: Dict[str, List[str]] = {}

        for model_id, model in all_models.items():
            model.to(self.device)
            model.eval()

            shap_exp = SHAPExplainer(model, self.device)
            paths: List[str] = []

            preds_for_model = predictions.get(model_id, [])
            for i in range(num_images):
                img = test_images[i:i+1]
                attr = shap_exp.compute(img, background_data,
                                        n_samples=hp.xai_shap_n_samples)

                if attr is not None and HAS_MATPLOTLIB:
                    path = shap_dir / f"{model_id}_img{i}_attr.png"
                    if i < len(preds_for_model):
                        p = preds_for_model[i]
                        title = (
                            f"{model_id} · sample #{i}\n"
                            f"pred: {p['pred_name']} ({p['confidence']*100:.1f}%) "
                            f"· GT: {p['gt_name']}"
                        )
                    else:
                        title = f"{model_id} · sample #{i}"
                    self._save_attribution_plot(attr, path, dpi=dpi,
                                                title=title)
                    paths.append(str(path))

            shap_paths[model_id] = paths
            logger.info("    %s: %d attribution maps generated",
                         model_id, len(paths))

        # ── Compute consistency scores ──
        logger.info("  Computing consistency scores (vs FP32) ...")
        consistency_scores: Dict[str, float] = {}

        fp32_heatmaps = all_heatmaps.get("FP32_baseline", [])
        for model_id, heatmaps in all_heatmaps.items():
            if model_id == "FP32_baseline":
                continue
            if not fp32_heatmaps or not heatmaps:
                continue

            scores = []
            for hm_fp32, hm_q in zip(fp32_heatmaps, heatmaps):
                s = compute_heatmap_consistency(hm_fp32, hm_q)
                scores.append(s)

            avg_score = sum(scores) / max(len(scores), 1)
            consistency_scores[model_id] = avg_score
            logger.info("    %s: correlation = %.4f", model_id, avg_score)

        # ── Generate comparison grid ──
        grid_path = ""
        if HAS_MATPLOTLIB and all_heatmaps:
            grid_path = str(out_dir / "comparison_matrix.png")
            self._generate_comparison_matrix(
                all_heatmaps=all_heatmaps,
                predictions=predictions,
                test_images=test_images,
                test_labels=test_labels,
                class_lookup=class_lookup,
                num_images=num_images,
                output_path=grid_path,
                alpha=alpha,
                dpi=dpi,
            )
            logger.info("  Comparison matrix saved: comparison_matrix.png")

        # ── Generate report ──
        report = self._generate_report(
            all_models, num_images, consistency_scores, grad_cam_paths,
            shap_paths, predictions,
        )

        logger.info("=" * 70)

        # XAIResult includes per-(technique, sample) predictions so
        # callers (and the resume restorer) can surface model outputs
        # without re-running inference.
        result: XAIResult = {
            "grad_cam_paths": grad_cam_paths,
            "shap_paths": shap_paths,
            "comparison_grid": grid_path,
            "consistency_scores": consistency_scores,
            "report": report,
            "predictions": predictions,
        }
        return result

    # ------------------------------------------------------------------
    # Visualisation Helpers
    # ------------------------------------------------------------------

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a [1, C, H, W] or [C, H, W] tensor to [H, W, 3] numpy.
        Handles normalised images by rescaling to [0, 1].
        """
        if tensor.dim() == 4:
            tensor = tensor[0]
        img = tensor.detach().cpu().numpy()
        if img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)  # Grayscale -> RGB
        # Normalise to [0, 1]
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)
        return np.clip(img, 0, 1)

    def _save_image_with_prediction(
        self,
        image: np.ndarray,
        path: Path,
        dpi: int,
        model_id: str,
        image_idx: int,
        meta: Dict[str, Any],
    ) -> None:
        """Save a Grad-CAM overlay with a fully-labelled caption underneath.

        The caption shows the technique name, the predicted class and its
        confidence, and the ground-truth label. A green ✓ / red ✗ badge
        encodes correctness at a glance.
        """
        from neuroquant.visualization.style import apply_publication_style
        apply_publication_style()

        fig, ax = plt.subplots(figsize=(4.4, 4.7))
        ax.imshow(image)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ok = bool(meta["correct"])
        badge = "✓" if ok else "✗"
        badge_color = self.BADGE_OK if ok else self.BADGE_BAD
        title = (
            f"{model_id}  ·  sample #{image_idx}\n"
            f"pred: {meta['pred_name']}  ({meta['confidence'] * 100:.1f}%)  "
            f"{badge}\n"
            f"ground truth: {meta['gt_name']}"
        )
        ax.set_title(title, fontsize=10, pad=6, loc="center")
        # Coloured strip below the image makes correctness scannable.
        ax.text(
            0.5, -0.06, badge,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=20, color=badge_color, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    def _save_attribution_plot(
        self,
        attr: np.ndarray,
        path: Path,
        dpi: int = 150,
        title: str = "",
    ) -> None:
        """Save a SHAP/gradient attribution as a diverging heatmap."""
        from neuroquant.visualization.style import apply_publication_style
        apply_publication_style()

        if attr.ndim == 3:
            attr_map = attr.mean(axis=0)  # [H, W]
        else:
            attr_map = attr

        fig, ax = plt.subplots(figsize=(4, 4.3))
        vmax = max(abs(attr_map.min()), abs(attr_map.max()), 1e-8)
        im = ax.imshow(attr_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10, pad=6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    def _generate_comparison_matrix(
        self,
        all_heatmaps: Dict[str, List[np.ndarray]],
        predictions: Dict[str, List[Dict[str, Any]]],
        test_images: torch.Tensor,
        test_labels: torch.Tensor,
        class_lookup,
        num_images: int,
        output_path: str,
        alpha: float = 0.4,
        dpi: int = 150,
    ) -> None:
        """Render the technique × sample comparison matrix.

        Layout
        ------
        Row 0       : sample image previews labelled "img_i / GT: <class>"
        Rows 1..M   : one row per technique (FP32, PTQ_best, …)
        Each cell   : Grad-CAM overlay for that (technique, sample)
                       with a caption "pred: <class> (conf%) ✓/✗"
        Leftmost    : technique row labels rendered in a header strip on
                      a tinted background so they are unambiguous.
        """
        from neuroquant.visualization.style import apply_publication_style
        apply_publication_style()

        model_ids = list(all_heatmaps.keys())
        n_models = len(model_ids)
        n_imgs = min(num_images, max(len(hm) for hm in all_heatmaps.values()))
        if n_models == 0 or n_imgs == 0:
            return

        # Total grid: 1 header row (samples) + n_models rows
        # Total cols: 1 header col (model names) + n_imgs cols
        n_rows = n_models + 1
        n_cols = n_imgs + 1

        # Each cell ~3.0in wide, 3.4in tall (extra height for captions).
        fig_w = 1.6 + 2.8 * n_imgs
        fig_h = 1.4 + 3.0 * n_models
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            n_rows, n_cols,
            width_ratios=[0.9] + [1.0] * n_imgs,
            height_ratios=[0.85] + [1.0] * n_models,
            wspace=0.18, hspace=0.30,
        )

        # ── (0, 0): empty corner ─────────────────────────────────────
        corner = fig.add_subplot(gs[0, 0])
        corner.axis("off")
        corner.text(
            0.5, 0.5,
            "Technique  ↓\nSample  →",
            transform=corner.transAxes, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#444444",
        )

        # ── Top header row: sample previews + GT ─────────────────────
        for j in range(n_imgs):
            ax = fig.add_subplot(gs[0, j + 1])
            preview = self._tensor_to_image(test_images[j:j + 1])
            ax.imshow(preview)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            gt_idx = int(test_labels[j].item())
            ax.set_title(
                f"sample #{j}\nGT: {class_lookup(gt_idx)}",
                fontsize=10, fontweight="bold", pad=4,
                color=self.HEADER_FG,
            )
            # subtle background stripe to mark this as a header
            ax.set_facecolor(self.HEADER_BG)

        # ── Left column header strip: technique names ────────────────
        for i, model_id in enumerate(model_ids):
            ax = fig.add_subplot(gs[i + 1, 0])
            ax.axis("off")
            # tinted rectangle so the header is visually distinct
            ax.add_patch(
                plt.Rectangle(
                    (0.02, 0.05), 0.96, 0.90,
                    transform=ax.transAxes,
                    facecolor=self.HEADER_BG, edgecolor="#cfd8dc",
                    linewidth=0.8,
                )
            )
            ax.text(
                0.5, 0.55, model_id,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=self.HEADER_FG, wrap=True,
            )
            preds = predictions.get(model_id, [])
            n_correct = sum(1 for p in preds if p.get("correct"))
            ax.text(
                0.5, 0.30,
                f"{n_correct}/{len(preds)} correct",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#555555",
            )

        # ── Inner cells: heatmap + per-cell prediction ──────────────
        for i, model_id in enumerate(model_ids):
            heatmaps = all_heatmaps[model_id]
            preds = predictions.get(model_id, [])
            for j in range(n_imgs):
                ax = fig.add_subplot(gs[i + 1, j + 1])
                if j < len(heatmaps):
                    img_np = self._tensor_to_image(test_images[j:j + 1])
                    overlay = overlay_heatmap(img_np, heatmaps[j], alpha=alpha)
                    ax.imshow(overlay)
                else:
                    ax.set_facecolor("#f5f5f5")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if j < len(preds):
                    p = preds[j]
                    ok = bool(p["correct"])
                    badge = "✓" if ok else "✗"
                    badge_color = self.BADGE_OK if ok else self.BADGE_BAD
                    caption = (
                        f"pred: {p['pred_name']}  "
                        f"({p['confidence'] * 100:.1f}%)  {badge}"
                    )
                    ax.text(
                        0.5, -0.08, caption,
                        transform=ax.transAxes,
                        ha="center", va="top",
                        fontsize=9, color=badge_color,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            fc=self.CAPTION_BG,
                            ec=self.CAPTION_EDGE,
                            alpha=0.95,
                        ),
                    )

        fig.suptitle(
            "Grad-CAM comparison: technique × sample\n"
            "rows = quantization technique · columns = sample image · "
            "cell caption = prediction (confidence)",
            fontsize=12, fontweight="bold", y=0.995,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # Keep the old name as an alias so external callers don't break.
    _generate_comparison_grid = _generate_comparison_matrix

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        models: Dict[str, nn.Module],
        num_images: int,
        consistency_scores: Dict[str, float],
        grad_cam_paths: Dict[str, List[str]],
        shap_paths: Dict[str, List[str]],
        predictions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> str:
        """Generate markdown XAI analysis report."""
        lines = []
        lines.append("# XAI Analysis: Explainability Across Quantization")
        lines.append("")
        lines.append("## Models Analysed")
        for i, model_id in enumerate(models.keys()):
            lines.append(f"{i+1}. **{model_id}**")
        lines.append("")
        lines.append(f"## Test Images: {num_images}")
        lines.append("")

        total_gc = sum(len(p) for p in grad_cam_paths.values())
        total_shap = sum(len(p) for p in shap_paths.values())
        lines.append("## Outputs Generated")
        lines.append(f"- Grad-CAM heatmaps: {total_gc}")
        lines.append(f"- SHAP/gradient attributions: {total_shap}")
        lines.append("")

        if predictions:
            lines.append("## Predictions per Sample")
            lines.append("")
            header = "| Technique | " + " | ".join(
                f"sample #{i}" for i in range(num_images)
            ) + " |"
            sep = "|" + "---|" * (num_images + 1)
            lines.append(header)
            lines.append(sep)
            for model_id, preds in predictions.items():
                cells = []
                for i in range(num_images):
                    if i >= len(preds):
                        cells.append("—")
                        continue
                    p = preds[i]
                    badge = "✓" if p.get("correct") else "✗"
                    cells.append(
                        f"{p['pred_name']} ({p['confidence']*100:.0f}%) {badge}"
                    )
                lines.append(f"| {model_id} | " + " | ".join(cells) + " |")
            lines.append("")

            # Per-technique accuracy summary
            lines.append("## Top-1 Accuracy on Explained Samples")
            for model_id, preds in predictions.items():
                if not preds:
                    continue
                correct = sum(1 for p in preds if p.get("correct"))
                lines.append(
                    f"- **{model_id}**: {correct}/{len(preds)} correct "
                    f"({correct / max(len(preds), 1) * 100:.0f}%)"
                )
            lines.append("")

        if consistency_scores:
            lines.append("## Consistency Analysis (vs FP32 baseline)")
            lines.append("")
            lines.append("| Model | Pearson Correlation | Interpretation |")
            lines.append("|-------|--------------------:|----------------|")
            for model_id, score in consistency_scores.items():
                if score >= 0.8:
                    interp = "Very similar (quantization invisible)"
                elif score >= 0.6:
                    interp = "Moderate shift (acceptable)"
                elif score >= 0.4:
                    interp = "Significant change (review needed)"
                else:
                    interp = "Major divergence (concerning)"
                lines.append(f"| {model_id} | {score:.4f} | {interp} |")
            lines.append("")

            lines.append("## Insights")
            sorted_scores = sorted(consistency_scores.items(),
                                    key=lambda x: x[1], reverse=True)
            for model_id, score in sorted_scores:
                if score >= 0.8:
                    lines.append(f"- **{model_id}:** Quantization is invisible "
                                  f"to model attention (r={score:.3f})")
                elif score >= 0.6:
                    lines.append(f"- **{model_id}:** Some attention shift but "
                                  f"interpretability maintained (r={score:.3f})")
                else:
                    lines.append(f"- **{model_id}:** Significant attention change "
                                  f"-- review for safety-critical use (r={score:.3f})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict(
        self, model: nn.Module, image: torch.Tensor,
    ) -> Tuple[int, float]:
        """Return ``(pred_idx, confidence)`` for a single image.

        Task-aware so the caption shows a meaningful prediction for
        classification, detection, and segmentation alike:
          * classification → argmax class, softmax probability.
          * detection      → label of the top-scoring detection (or
                              ``-1`` and confidence ``0`` when the
                              detector returned no boxes).
          * segmentation   → most-likely-on-average class across the
                              spatial mask, mean softmax confidence.

        Forward pass runs under ``torch.no_grad`` so the model state is
        untouched before Grad-CAM's gradient pass.
        """
        model.eval()
        img = image.to(self.device)
        with torch.no_grad():
            output = model(img)
        return predict_from_output(output)

    @staticmethod
    def _build_class_name_lookup(
        class_names: Optional[List[str]],
        num_classes: int,
    ):
        """Return a callable ``idx -> name`` with safe fallback.

        Falls back to ``"class N"`` whenever a name is missing or out of
        range. Always returns a string so captions never break.
        """
        names = list(class_names) if class_names else []

        def _lookup(idx: int) -> str:
            if 0 <= idx < len(names):
                return str(names[idx])
            if 0 <= idx < num_classes:
                return f"class {idx}"
            return f"class ?{idx}"

        return _lookup

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)
