"""
NeuroQuant v2.0 - SmoothQuant Quantizer (Phase 1f)

Migrates quantization difficulty from activations to weights
by applying a mathematically equivalent per-channel scaling.
Supports INT8 and INT4.

Reference: Xiao et al., "SmoothQuant: Accurate and Efficient
Post-Training Quantization for Large Language Models" (2023).

Algorithm:
    For each layer:
    1. Collect per-channel activation max: a_j = max(|X[:, j, :, :]|)
       (over batch + spatial dims for CNNs)
    2. Collect per-channel weight max: w_j = max(|W[:, j, :, :]|)
       (over output channels + kernel dims)
    3. Compute smooth scale:
       s_j = (a_j^alpha) / (w_j^(1-alpha))
       where alpha in [0, 1] controls migration strength
       (alpha=0.5 is balanced default)
    4. Apply smoothing: W_smooth[:, j] = W[:, j] * s_j
       (absorbs activation spikiness into weights)
    5. Quantize smoothed weights to target bitwidth

CNN Adaptation:
    - Activation max: dims (0, 2, 3) for [batch, ch, H, W]
    - Weight max: over dims (0, 2, 3) for Conv2d [out, in, kH, kW]
    - Scale applied along input channel dimension (dim=1)
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import QuantizationConfig, QuantizationResult
from quantization.base import BaseQuantizer

logger = logging.getLogger("neuroquant")


class _SmoothInputScale(nn.Module):
    """Pre-forward module that divides a layer's input by the per-channel
    smoothing scale ``s``. Paired with ``W' = W * s`` on the wrapped layer,
    this preserves ``Y = (X/s) @ (s*W) = X @ W`` — the algebraic equivalence
    that makes SmoothQuant lossless *before* quantization.
    """

    def __init__(self, inv_scale: torch.Tensor) -> None:
        super().__init__()
        # Stored as a buffer so it participates in state_dict, deepcopy,
        # and .to(device) moves alongside the wrapped layer.
        self.register_buffer("inv_scale", inv_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.inv_scale


class SmoothQuantQuantizer(BaseQuantizer):
    """
    SmoothQuant: Smooth activation outliers before quantization.

    Applies per-channel scaling: Y = (X / s) @ (s * W)
    where s balances the quantization difficulty between
    activations and weights using a migration strength alpha.

    Alpha controls the trade-off:
        alpha = 0.0: All difficulty stays in activations
        alpha = 0.5: Balanced (default, recommended)
        alpha = 1.0: All difficulty migrated to weights
    """

    def __init__(self, model: nn.Module, config: QuantizationConfig) -> None:
        super().__init__(model, config)

    def quantize(
        self,
        calibration_loader: DataLoader,
        bitwidth: int = 8,
        num_batches: int = 10,
    ) -> nn.Module:
        """
        Apply SmoothQuant quantization.

        Steps:
        1. Profile activation ranges during calibration
        2. Compute per-channel smoothing scales
        3. Absorb scales into weights (mathematically equivalent)
        4. Quantize smoothed weights

        Args:
            calibration_loader: Calibration data for scale computation.
            bitwidth: Target bitwidth (4 or 8).
            num_batches: Number of calibration batches.

        Returns:
            SmoothQuant-quantized model (deep copy).
        """
        start = time.time()
        q_model = copy.deepcopy(self.model)
        q_model.to(self.device)
        q_model.eval()

        alpha = self.config.hyperparams.smoothquant_alpha
        target_layers = self._find_quantizable_layers(q_model)
        logger.info("SmoothQuant INT%d: %d layers (alpha=%.2f)",
                     bitwidth, len(target_layers), alpha)

        # Step 1: Collect activation statistics
        act_max, weight_max = self._collect_stats(
            q_model, target_layers, calibration_loader, num_batches,
        )

        # Step 2 & 3: Compute scales and smooth weights
        for layer_name, module in target_layers.items():
            # Skip depthwise convolutions: scale broadcast corrupts weight shape
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                with torch.no_grad():
                    module.weight.data = self.quantize_tensor(
                        module.weight.data, bitwidth, per_channel=True
                    )
                logger.info("    %s: standard INT%d (depthwise conv)",
                             layer_name, bitwidth)
                continue

            if layer_name not in act_max:
                # No activation data; standard quantization
                with torch.no_grad():
                    module.weight.data = self.quantize_tensor(
                        module.weight.data, bitwidth, per_channel=True
                    )
                logger.info("    %s: standard INT%d (no calibration data)",
                             layer_name, bitwidth)
                continue

            a_max = act_max[layer_name].to(self.device)
            w_max = weight_max[layer_name].to(self.device)

            # Compute smooth scales
            scales = self._compute_smooth_scales(a_max, w_max, alpha)

            # Insert the compensating input divide (X' = X / s) before the
            # layer, then absorb the inverse scaling into its weights
            # (W' = W * s). Together these give Y = X @ W unchanged
            # mathematically, which is what makes the transform lossless
            # before quantization kicks in.
            self._wrap_input_scale(q_model, layer_name, module, scales)
            self._apply_smoothing(module, scales)

            # Quantize smoothed weights
            with torch.no_grad():
                module.weight.data = self.quantize_tensor(
                    module.weight.data, bitwidth, per_channel=True
                )

            logger.info("    %s: smoothed + INT%d (scale range=[%.4f, %.4f])",
                         layer_name, bitwidth,
                         scales.min().item(), scales.max().item())

        elapsed = time.time() - start
        logger.info("SmoothQuant INT%d complete in %.1fs", bitwidth, elapsed)
        return q_model

    def _collect_stats(
        self,
        model: nn.Module,
        target_layers: Dict[str, nn.Module],
        data_loader: DataLoader,
        num_batches: int,
    ) -> tuple:
        """
        Collect per-channel activation and weight statistics.

        Returns:
            (act_max, weight_max): Dicts mapping layer_name to
            per-input-channel maximum magnitudes.
        """
        # Activation max via hooks
        act_max: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str, is_conv: bool):
            def hook_fn(module, inp, out):
                x = inp[0].detach()
                if is_conv and x.dim() == 4:
                    # [batch, in_ch, H, W] -> max over (0, 2, 3)
                    ch_max = x.abs().amax(dim=(0, 2, 3))
                elif x.dim() >= 2:
                    dims = [d for d in range(x.dim()) if d != 1]
                    ch_max = x.abs().amax(dim=dims) if dims else x.abs().squeeze()
                else:
                    ch_max = x.abs()

                ch_max = ch_max.cpu()
                if name in act_max:
                    act_max[name] = torch.max(act_max[name], ch_max)
                else:
                    act_max[name] = ch_max
            return hook_fn

        for name, module in target_layers.items():
            is_conv = isinstance(module, nn.Conv2d)
            h = module.register_forward_hook(make_hook(name, is_conv))
            hooks.append(h)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                images = batch[0].to(self.device)
                model(images)

        for h in hooks:
            h.remove()

        # Weight max per input channel
        weight_max: Dict[str, torch.Tensor] = {}
        for name, module in target_layers.items():
            w = module.weight.data
            if isinstance(module, nn.Conv2d):
                # [out, in, kH, kW] -> max over (0, 2, 3) -> [in]
                w_ch_max = w.abs().amax(dim=(0, 2, 3))
            else:
                # [out, in] -> max over 0 -> [in]
                w_ch_max = w.abs().amax(dim=0)
            weight_max[name] = w_ch_max.cpu()

        return act_max, weight_max

    def _compute_smooth_scales(
        self,
        act_max: torch.Tensor,
        weight_max: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Compute per-channel smoothing scales.

        s_j = act_max_j^alpha / weight_max_j^(1-alpha)

        This migrates quantization difficulty from activations
        (which may have outlier spikes) to weights (which are
        more uniformly distributed).
        """
        # Clamp to avoid division by zero
        a = act_max.clamp(min=1e-8)
        w = weight_max.clamp(min=1e-8)

        # SmoothQuant formula
        scales = a.pow(alpha) / w.pow(1.0 - alpha)

        # Clamp scales to reasonable range to avoid numerical issues
        scales = scales.clamp(min=1e-4, max=1e4)

        return scales

    def _wrap_input_scale(
        self,
        root: nn.Module,
        full_name: str,
        module: nn.Module,
        scales: torch.Tensor,
    ) -> None:
        """Replace ``root.<full_name>`` with ``Sequential(_SmoothInputScale, module)``
        so the layer's input is multiplied by ``1/s`` at forward time.

        The inverse scale is reshaped to broadcast along the input-channel
        axis of the wrapped layer (Conv2d: [1, C_in, 1, 1]; Linear: [1, C_in]).
        """
        if isinstance(module, nn.Conv2d):
            inv = (1.0 / scales).view(1, -1, 1, 1).detach().clone()
        else:
            inv = (1.0 / scales).view(1, -1).detach().clone()

        wrapper = nn.Sequential(_SmoothInputScale(inv), module)
        wrapper.to(module.weight.device)

        parent_name, _, attr = full_name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, attr, wrapper)

    def _apply_smoothing(
        self, module: nn.Module, scales: torch.Tensor
    ) -> None:
        """
        Absorb smoothing scales into weights.

        W_smooth[:, j, ...] = W[:, j, ...] * s_j

        This is mathematically equivalent to dividing activations
        by s and multiplying weights by s (maintaining Y = X @ W).
        """
        w = module.weight.data

        if isinstance(module, nn.Conv2d):
            # scales: [in_channels] -> [1, in, 1, 1]
            s = scales.view(1, -1, 1, 1)
        else:
            # scales: [in_features] -> [1, in]
            s = scales.view(1, -1)

        module.weight.data = w * s

    def _find_quantizable_layers(
        self, model: nn.Module
    ) -> Dict[str, nn.Module]:
        """Find all Conv2d and Linear layers."""
        layers = {}
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                layers[name] = m
        return layers

    def _get_method_name(self) -> str:
        return "SmoothQuant"
