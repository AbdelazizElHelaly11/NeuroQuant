"""
NeuroQuant v2.0 - XAI Explainability Pipeline (Phase 3)

Generates Grad-CAM heatmaps and SHAP explanations to visualise
how quantization affects model attention and feature importance.

Key components:
    1. GradCAMExplainer: per-layer attention heatmaps with STE-safe hooks
    2. SHAPExplainer: gradient-based feature importance (optional, graceful)
    3. XAIGenerator: orchestrates multi-model comparison + consistency scoring

Enhancements over the spec:
    - Auto-detection of target layer (finds last Conv2d before pooling)
    - Graceful degradation: SHAP is optional; Grad-CAM always works
    - Consistency scoring via Pearson correlation of flattened heatmaps
    - Dark-themed comparison grid for publication quality
    - Handles nn.Sequential models (no hardcoded layer names)
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

from config import QuantizationConfig, XAIResult

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

        Args:
            image: Input tensor [1, C, H, W] (normalised).
            target_layer: Conv2d module to visualise.
            target_class: Class index for gradient computation.
                         None = use predicted class.

        Returns:
            Heatmap as numpy array [H, W] in [0, 1].
        """
        self._activations = None
        self._gradients = None

        # Register hooks on target layer
        fwd_hook = target_layer.register_forward_hook(self._save_activation)
        bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)
        self._hooks = [fwd_hook, bwd_hook]

        try:
            self.model.eval()
            image = image.to(self.device)

            # Forward pass
            output = self.model(image)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Backward pass for target class
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot, retain_graph=True)

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
        """Forward hook: save activations."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save gradients."""
        self._gradients = grad_output[0].detach()


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
        if not HAS_SHAP:
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
        Faster than SHAP, always works, gives similar spatial info.
        """
        self.model.eval()
        img = image.to(self.device).requires_grad_(True)

        output = self.model(img)
        pred_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1.0
        output.backward(gradient=one_hot)

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
    models, computing consistency scores and generating a comparison grid.

    Usage:
        gen = XAIGenerator(config)
        result = gen.run(
            fp32_model=model,
            quantized_models={"INT8": model_int8, "INT4": model_int4},
            test_images=images,    # [N, C, H, W]
            test_labels=labels,    # [N]
            output_dir="artifacts/xai",
        )
    """

    # Dark theme colour palette
    COLORS = {
        "bg": "#1e1e2e",
        "panel": "#2d2d3d",
        "text": "#e0e0e0",
        "grid": "#424242",
    }

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

        logger.info("  Models: %d (%s)", len(all_models),
                     ", ".join(all_models.keys()))
        logger.info("  Images: %d", num_images)

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

        # ── Generate Grad-CAM heatmaps ──
        logger.info("  Generating Grad-CAM heatmaps ...")
        all_heatmaps: Dict[str, List[np.ndarray]] = {}
        grad_cam_paths: Dict[str, List[str]] = {}

        for model_id, model in all_models.items():
            model.to(self.device)
            model.eval()

            # Find target layer in this model
            target_module = find_layer_by_name(model, layer_name)
            if target_module is None:
                # Try auto-detect for this model
                try:
                    _, target_module = auto_detect_target_layer(model)
                except ValueError:
                    logger.warning("    No conv layer in '%s', skipping", model_id)
                    continue

            grad_cam = GradCAMExplainer(model, self.device)
            heatmaps: List[np.ndarray] = []
            paths: List[str] = []

            for i in range(num_images):
                img = test_images[i:i+1]
                heatmap = grad_cam.compute(img, target_module,
                                            target_class=test_labels[i].item())
                heatmaps.append(heatmap)

                # Save individual heatmap
                if HAS_MATPLOTLIB:
                    img_np = self._tensor_to_image(img)
                    overlay = overlay_heatmap(img_np, heatmap, alpha=alpha)
                    path = grad_cam_dir / f"{model_id}_img{i}.png"
                    self._save_image(overlay, path, dpi=dpi,
                                     title=f"{model_id} - Image {i}")
                    paths.append(str(path))

            all_heatmaps[model_id] = heatmaps
            grad_cam_paths[model_id] = paths
            logger.info("    %s: %d heatmaps generated", model_id, len(heatmaps))

        # ── Generate SHAP attributions ──
        logger.info("  Generating SHAP/gradient attributions ...")
        shap_paths: Dict[str, List[str]] = {}

        for model_id, model in all_models.items():
            model.to(self.device)
            model.eval()

            shap_exp = SHAPExplainer(model, self.device)
            paths: List[str] = []

            for i in range(num_images):
                img = test_images[i:i+1]
                attr = shap_exp.compute(img, background_data,
                                        n_samples=hp.xai_shap_n_samples)

                if attr is not None and HAS_MATPLOTLIB:
                    path = shap_dir / f"{model_id}_img{i}_attr.png"
                    self._save_attribution_plot(attr, path, dpi=dpi,
                                                 title=f"{model_id} - Image {i}")
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
            self._generate_comparison_grid(
                all_heatmaps, test_images, num_images, grid_path,
                alpha=alpha, dpi=dpi,
            )
            logger.info("  Comparison grid saved: comparison_matrix.png")

        # ── Generate report ──
        report = self._generate_report(
            all_models, num_images, consistency_scores, grad_cam_paths,
            shap_paths,
        )

        logger.info("=" * 70)

        return XAIResult(
            grad_cam_paths=grad_cam_paths,
            shap_paths=shap_paths,
            comparison_grid=grid_path,
            consistency_scores=consistency_scores,
            report=report,
        )

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

    def _save_image(
        self,
        image: np.ndarray,
        path: Path,
        dpi: int = 150,
        title: str = "",
    ) -> None:
        """Save an RGB image as PNG."""
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor(self.COLORS["bg"])
        ax.imshow(image)
        ax.set_title(title, fontsize=9, color=self.COLORS["text"], pad=4)
        ax.axis("off")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=self.COLORS["bg"])
        plt.close(fig)

    def _save_attribution_plot(
        self,
        attr: np.ndarray,
        path: Path,
        dpi: int = 150,
        title: str = "",
    ) -> None:
        """Save a SHAP/gradient attribution as a diverging heatmap."""
        # attr shape: [C, H, W] -> mean across channels
        if attr.ndim == 3:
            attr_map = attr.mean(axis=0)  # [H, W]
        else:
            attr_map = attr

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor(self.COLORS["bg"])
        vmax = max(abs(attr_map.min()), abs(attr_map.max()), 1e-8)
        ax.imshow(attr_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=9, color=self.COLORS["text"], pad=4)
        ax.axis("off")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=self.COLORS["bg"])
        plt.close(fig)

    def _generate_comparison_grid(
        self,
        all_heatmaps: Dict[str, List[np.ndarray]],
        test_images: torch.Tensor,
        num_images: int,
        output_path: str,
        alpha: float = 0.4,
        dpi: int = 150,
    ) -> None:
        """
        Generate NxM grid: rows=models, cols=images.
        Each cell shows Grad-CAM overlay.
        """
        model_ids = list(all_heatmaps.keys())
        n_models = len(model_ids)
        n_imgs = min(num_images, max(len(hm) for hm in all_heatmaps.values()))

        if n_models == 0 or n_imgs == 0:
            return

        fig, axes = plt.subplots(
            n_models, n_imgs,
            figsize=(3 * n_imgs, 3 * n_models),
        )
        fig.patch.set_facecolor(self.COLORS["bg"])

        # Handle single row/col
        if n_models == 1:
            axes = [axes]
        if n_imgs == 1:
            axes = [[ax] for ax in axes]

        for row, model_id in enumerate(model_ids):
            heatmaps = all_heatmaps[model_id]
            for col in range(n_imgs):
                ax = axes[row][col] if isinstance(axes[row], (list, np.ndarray)) else axes[row]

                if col < len(heatmaps):
                    img_np = self._tensor_to_image(test_images[col:col+1])
                    overlay_img = overlay_heatmap(img_np, heatmaps[col], alpha=alpha)
                    ax.imshow(overlay_img)
                else:
                    ax.set_facecolor(self.COLORS["panel"])

                ax.axis("off")

                # Row labels (model names)
                if col == 0:
                    ax.set_ylabel(model_id, fontsize=9, color=self.COLORS["text"],
                                  rotation=0, labelpad=60, ha="right", va="center")

                # Column labels (image indices)
                if row == 0:
                    ax.set_title(f"Image {col}", fontsize=9,
                                 color=self.COLORS["text"], pad=8)

        fig.suptitle(
            "Grad-CAM Comparison Across Quantization Strategies",
            fontsize=13, fontweight="bold", color=self.COLORS["text"],
            y=1.02,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor=self.COLORS["bg"])
        plt.close(fig)

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
