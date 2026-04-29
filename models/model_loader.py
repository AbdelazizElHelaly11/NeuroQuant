"""
NeuroQuant v2.0 - Generic Model Loader

Provides a factory for loading ANY pre-trained PyTorch model
with zero hardcoded architecture assumptions.

Supports:
    - TorchVision models (by name: mobilenetv2, resnet18, vgg16, ...)
    - Custom model classes (via fully-qualified Python path)
    - Saved checkpoint files (.pt / .pth)

Key introspection features:
    - Auto-detects and adapts the final classifier head for num_classes
    - Auto-adapts first conv layer for small-input datasets
    - Uses dummy forward passes — never hardcodes layer names
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import QuantizationConfig

logger = logging.getLogger("neuroquant")

# Optional torchvision
try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Introspection Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _find_last_linear(model: nn.Module) -> Optional[Tuple[str, nn.Linear]]:
    """
    Find the last nn.Linear layer in the model (the classifier head).

    Walks all named modules and returns the last one that is Linear.
    Works for any architecture — no hardcoded attribute names.
    """
    last_name, last_module = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_module = module
    return (last_name, last_module) if last_module is not None else None


def _find_first_conv(model: nn.Module) -> Optional[Tuple[str, nn.Conv2d]]:
    """
    Find the first nn.Conv2d layer in the model (the input stem).

    Returns (dotted_name, module). Works for any architecture.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module
    return None


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """
    Replace a submodule by its dotted name path.

    Example: _set_module_by_name(model, "classifier.1", nn.Linear(1280, 10))
    This handles arbitrary nesting via getattr/setattr.
    """
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _infer_classifier_in_features(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> int:
    """
    Infer the in_features of the classifier head via a dummy forward pass.

    Temporarily replaces the last Linear with an identity hook that
    captures the input tensor shape, then restores the original.
    """
    result = _find_last_linear(model)
    if result is None:
        raise ValueError("No Linear layer found in model; cannot infer in_features.")

    last_name, last_linear = result
    captured_shape = [None]

    class _CaptureHook(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original

        def forward(self, x):
            captured_shape[0] = x.shape
            return self.original(x)

    # Install capture hook
    hook_module = _CaptureHook(last_linear)
    _set_module_by_name(model, last_name, hook_module)

    try:
        model.eval()
        model.to(device)
        dummy = torch.randn(1, *input_shape, device=device)
        with torch.no_grad():
            model(dummy)
    except Exception:
        pass  # Some models may fail on wrong input size; we'll handle below
    finally:
        # Restore original
        _set_module_by_name(model, last_name, last_linear)

    if captured_shape[0] is not None:
        return captured_shape[0][-1]  # Last dim = in_features

    # Fallback: use the existing in_features
    return last_linear.in_features


def adapt_classifier(
    model: nn.Module,
    num_classes: int,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> nn.Module:
    """
    Adapt the model's final classifier head for `num_classes`.

    Uses introspection to find the last Linear layer, infers
    its in_features via a dummy forward pass, and replaces it.
    No hardcoded layer names.
    """
    result = _find_last_linear(model)
    if result is None:
        logger.warning("No Linear layer found — cannot adapt classifier.")
        return model

    last_name, last_linear = result

    # If already correct, skip
    if last_linear.out_features == num_classes:
        logger.info("  Classifier already has %d classes, no adaptation needed.", num_classes)
        return model

    # Infer in_features
    in_features = _infer_classifier_in_features(model, input_shape, device)

    # Replace
    new_linear = nn.Linear(in_features, num_classes)
    _set_module_by_name(model, last_name, new_linear)
    logger.info(
        "  Adapted classifier '%s': Linear(%d, %d) → Linear(%d, %d)",
        last_name, last_linear.in_features, last_linear.out_features,
        in_features, num_classes,
    )
    return model


def adapt_input_conv(
    model: nn.Module,
    input_shape: Tuple[int, ...],
) -> nn.Module:
    """
    Adapt the first Conv2d for small-input datasets.

    If the model's first conv has stride > 1 and the input spatial size
    is small (≤ 64), reduce stride to 1 to avoid collapsing spatial dims
    too aggressively. This is a standard technique when using ImageNet
    architectures on CIFAR-like datasets.

    No hardcoded layer names — finds the first Conv2d by introspection.
    """
    spatial_size = input_shape[-1] if len(input_shape) >= 2 else 224
    if spatial_size > 64:
        return model  # No adaptation needed for large inputs

    result = _find_first_conv(model)
    if result is None:
        return model

    name, conv = result

    # Only adapt if stride > 1 (typical for ImageNet models)
    if conv.stride == (1, 1) or conv.stride == 1:
        return model

    # Replace with stride=1, same kernel/channels, adjusted padding
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=1,
        padding=conv.kernel_size[0] // 2 if isinstance(conv.kernel_size, tuple) else conv.kernel_size // 2,
        bias=conv.bias is not None,
    )
    _set_module_by_name(model, name, new_conv)
    logger.info(
        "  Adapted first conv '%s': stride %s → (1, 1) for %dx%d input",
        name, conv.stride, spatial_size, spatial_size,
    )

    # Also remove aggressive spatial-reduction layers (e.g., maxpool)
    # after the first conv, if the model has one at the top level
    _remove_early_maxpool(model)

    return model


def _remove_early_maxpool(model: nn.Module) -> None:
    """
    Replace early MaxPool2d/AvgPool2d with Identity if found
    in the top-level attributes. Prevents over-downsampling.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            setattr(model, name, nn.Identity())
            logger.info("  Removed early pooling '%s' → Identity", name)
            return  # Only remove the first one


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ModelLoader — Main Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ModelLoader:
    """
    Generic model factory that can load any PyTorch nn.Module.

    Resolution order:
        1. model_path → load checkpoint (requires model_name or model_class)
        2. model_class → dynamically import and instantiate
        3. model_name → load from torchvision by name

    After loading, automatically adapts:
        - Classifier head for config.num_classes
        - First conv stride for small-input datasets
    """

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.hyperparams.device)

    def load(self) -> nn.Module:
        """
        Load, adapt, and return the model in FP32.

        Returns:
            nn.Module ready for the quantization pipeline.
        """
        logger.info("=" * 70)
        logger.info("Model Loading")
        logger.info("=" * 70)

        # Step 1: Build the base model
        model = self._build_base_model()

        # Step 2: Adapt for target task
        model = adapt_input_conv(model, self.config.input_shape)
        model = adapt_classifier(
            model, self.config.num_classes,
            self.config.input_shape, self.device,
        )

        # Step 3: Load checkpoint weights (if provided)
        if self.config.model_path and Path(self.config.model_path).exists():
            self._load_checkpoint(model, self.config.model_path)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("  Model ready: %s, %s params", self.config.model_name, f"{n_params:,}")
        logger.info("=" * 70)

        return model

    # ------------------------------------------------------------------
    # Private: Build base model
    # ------------------------------------------------------------------

    def _build_base_model(self) -> nn.Module:
        """Build a base model from config (before adaptation)."""

        # Priority 1: Custom class
        if self.config.model_class:
            return self._load_from_class(self.config.model_class)

        # Priority 2: TorchVision model by name
        return self._load_torchvision_model(self.config.model_name)

    def _load_from_class(self, class_path: str) -> nn.Module:
        """
        Dynamically import and instantiate a model class.

        Args:
            class_path: Fully qualified class name, e.g.,
                       "examples.mobilenet_cifar.MobileNetV2CIFAR"
        """
        parts = class_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"model_class must be fully qualified "
                f"(e.g., 'module.ClassName'), got: '{class_path}'"
            )

        module_name, class_name = parts
        logger.info("  Loading custom model: %s from %s", class_name, module_name)

        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)

        # Try common constructor signatures
        try:
            return cls(num_classes=self.config.num_classes)
        except TypeError:
            pass
        try:
            return cls(self.config.num_classes)
        except TypeError:
            pass
        return cls()

    def _load_torchvision_model(self, name: str) -> nn.Module:
        """
        Load a model from torchvision.models by name.

        Uses getattr(torchvision.models, name) — no hardcoded if/elif chain.
        Works for: mobilenet_v2, resnet18, resnet50, vgg16, efficientnet_b0, etc.
        """
        if not HAS_TORCHVISION:
            raise ImportError(
                "torchvision is required to load models by name. "
                "Install it or set config.model_class to a custom model."
            )

        # Normalise name: "mobilenetv2" → "mobilenet_v2"
        name_normalised = self._normalise_model_name(name)

        if not hasattr(tv_models, name_normalised):
            available = [
                n for n in dir(tv_models)
                if not n.startswith("_") and callable(getattr(tv_models, n))
            ]
            raise ValueError(
                f"Unknown model '{name}' (normalised: '{name_normalised}'). "
                f"Available torchvision models include: "
                f"{', '.join(sorted(available)[:20])}..."
            )

        logger.info("  Loading torchvision model: %s", name_normalised)
        model_fn = getattr(tv_models, name_normalised)
        model = model_fn(weights=None)
        return model

    def _load_checkpoint(self, model: nn.Module, path: str) -> None:
        """Load weights from a checkpoint file."""
        logger.info("  Loading checkpoint: %s", path)
        state = torch.load(path, map_location="cpu", weights_only=True)

        # Handle both bare state_dict and wrapped checkpoint
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state, strict=False)
        logger.info("  Checkpoint loaded successfully.")

    # ------------------------------------------------------------------
    # Name normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_model_name(name: str) -> str:
        """
        Normalise user-friendly model names to torchvision function names.

        Examples:
            "mobilenetv2"   → "mobilenet_v2"
            "resnet18"      → "resnet18"
            "efficientnetb0"→ "efficientnet_b0"
            "vgg16"         → "vgg16"
        """
        name = name.lower().strip()
        # Common aliases
        aliases = {
            "mobilenetv2": "mobilenet_v2",
            "mobilenet_v2": "mobilenet_v2",
            "mobilenetv3small": "mobilenet_v3_small",
            "mobilenetv3large": "mobilenet_v3_large",
            "efficientnetb0": "efficientnet_b0",
            "efficientnetb1": "efficientnet_b1",
            "efficientnetb2": "efficientnet_b2",
            "inception_v3": "inception_v3",
            "inceptionv3": "inception_v3",
            "googlenet": "googlenet",
        }
        return aliases.get(name, name)

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device_str)
