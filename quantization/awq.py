"""
NeuroQuant v2.0 - AWQ Quantizer (Phase 1f)

Activation-Aware Weight Quantization that protects salient
weight channels based on activation magnitude.
Supports mixed INT4/INT8 per-channel quantization.

Reference: Lin et al., "AWQ: Activation-aware Weight Quantization
for LLM Compression and Acceleration" (2024).

Algorithm:
    For each layer:
    1. Collect activation magnitudes during calibration:
       a_i = max(|A[:, i, :, :]|)  (over batch+spatial for CNNs)
    2. Normalize ranges to [0, 1]: a_norm_i = a_i / max(a)
    3. Assign per-channel bitwidths:
       - High activation channels (a_norm > threshold) -> INT8 (preserve)
       - Low activation channels (a_norm <= threshold) -> INT4 (compress)
    4. Compute per-channel scales that protect salient channels:
       - Scale up salient channels before quantization
       - Scale down after quantization to preserve original magnitude
    5. Quantize weights per-channel with assigned bitwidths

CNN Adaptation:
    - Activation max over dims (0, 2, 3) for [batch, ch, H, W]
      (LLMs use dims (0, 1) for [batch, seq, hidden])
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import QuantizationConfig, QuantizationResult
from quantization.base import BaseQuantizer

logger = logging.getLogger("neuroquant")


class AWQQuantizer(BaseQuantizer):
    """
    AWQ: Protect salient weights identified by activation patterns.

    Key insight: A small fraction of input channels carry
    disproportionately large activations. Protecting corresponding
    weight channels (via scaling) before quantization preserves
    accuracy with minimal overhead.
    """

    def __init__(self, model: nn.Module, config: QuantizationConfig) -> None:
        super().__init__(model, config)

    def quantize(
        self,
        calibration_loader: DataLoader,
        bitwidth: int = 4,
        num_batches: int = 10,
        activation_threshold: float = 0.5,
    ) -> nn.Module:
        """
        Apply AWQ quantization with per-channel bitwidth selection.

        Args:
            calibration_loader: Calibration data for activation profiling.
            bitwidth: Base bitwidth (4 or 8). High-saliency channels
                     may be promoted to 8 regardless.
            num_batches: Number of calibration batches.
            activation_threshold: Percentile threshold for assigning
                                 INT8 (above) vs INT4 (below).

        Returns:
            AWQ-quantized model (deep copy).
        """
        start = time.time()
        q_model = copy.deepcopy(self.model)
        q_model.to(self.device)
        q_model.eval()

        target_layers = self._find_quantizable_layers(q_model)
        logger.info("AWQ INT%d: %d quantizable layers (threshold=%.2f)",
                     bitwidth, len(target_layers), activation_threshold)

        # Step 1: Collect activation statistics
        act_stats = self._collect_activation_stats(
            q_model, target_layers, calibration_loader, num_batches,
        )

        # Step 2: Compute per-channel saliency and assign bitwidths
        # Step 3: Compute optimal scales and quantize
        total_int4 = 0
        total_int8 = 0

        for layer_name, module in target_layers.items():
            # Skip depthwise convolutions: scale broadcast corrupts weight shape
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                with torch.no_grad():
                    module.weight.data = self.quantize_tensor(
                        module.weight.data, bitwidth, per_channel=True
                    )
                logger.info("    %s: standard INT%d (depthwise conv)", layer_name, bitwidth)
                continue

            if layer_name not in act_stats:
                # No stats; quantize uniformly
                with torch.no_grad():
                    module.weight.data = self.quantize_tensor(
                        module.weight.data, bitwidth, per_channel=True
                    )
                continue

            act_range = act_stats[layer_name].to(self.device)  # [in_channels]

            # Determine per-input-channel bitwidths
            channel_bitwidths = self._assign_channel_bitwidths(
                act_range, bitwidth, activation_threshold,
            )

            # Compute protection scales for salient channels
            scales = self._compute_protection_scales(
                module, act_range, channel_bitwidths,
            )

            # Apply scales, quantize, undo scales
            self._scale_quantize_unscale(module, scales, channel_bitwidths)

            n4 = (channel_bitwidths == 4).sum().item()
            n8 = (channel_bitwidths == 8).sum().item()
            total_int4 += n4
            total_int8 += n8
            logger.info("    %s: INT4=%d, INT8=%d channels",
                         layer_name, n4, n8)

        elapsed = time.time() - start
        logger.info("AWQ complete: %d INT4 + %d INT8 channels in %.1fs",
                     total_int4, total_int8, elapsed)
        return q_model

    def _collect_activation_stats(
        self,
        model: nn.Module,
        target_layers: Dict[str, nn.Module],
        data_loader: DataLoader,
        num_batches: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect per-input-channel activation magnitudes.

        For Conv2d: max(|A|) over (batch, H, W) -> [in_channels]
        For Linear: max(|A|) over (batch,) -> [in_features]
        """
        # Accumulate max activations
        max_acts: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str, is_conv: bool):
            def hook_fn(module, inp, out):
                x = inp[0].detach()
                if is_conv and x.dim() == 4:
                    # [batch, in_ch, H, W] -> max over (0, 2, 3) -> [in_ch]
                    channel_max = x.abs().amax(dim=(0, 2, 3))
                elif x.dim() >= 2:
                    # [batch, in_features, ...] -> max over all except channel dim
                    dims = [d for d in range(x.dim()) if d != 1]
                    if dims:
                        channel_max = x.abs().amax(dim=dims)
                    else:
                        channel_max = x.abs().squeeze()
                else:
                    channel_max = x.abs()

                channel_max = channel_max.cpu()
                if name in max_acts:
                    max_acts[name] = torch.max(max_acts[name], channel_max)
                else:
                    max_acts[name] = channel_max
            return hook_fn

        for name, module in target_layers.items():
            is_conv = isinstance(module, nn.Conv2d)
            h = module.register_forward_hook(make_hook(name, is_conv))
            hooks.append(h)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                images = batch[0].to(self.device)
                model(images)

        for h in hooks:
            h.remove()

        return max_acts

    def _assign_channel_bitwidths(
        self,
        act_range: torch.Tensor,
        base_bitwidth: int,
        threshold: float,
    ) -> torch.Tensor:
        """
        Assign per-channel bitwidths based on activation ranges.

        High-activation channels get INT8, low get base_bitwidth.
        """
        # Normalize to [0, 1]
        amax = act_range.max().clamp(min=1e-8)
        a_norm = act_range / amax

        # Assign bitwidths
        bitwidths = torch.full_like(a_norm, base_bitwidth, dtype=torch.int32)
        if base_bitwidth == 4:
            # Promote high-activation channels to INT8
            bitwidths[a_norm > threshold] = 8
        else:
            # Already INT8 base, keep everything INT8
            bitwidths[:] = 8

        return bitwidths

    def _compute_protection_scales(
        self,
        module: nn.Module,
        act_range: torch.Tensor,
        channel_bitwidths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-channel protection scales.

        Salient channels (INT8) are scaled up before quantization
        to reduce relative quantization error, then scaled back down.

        scale_i = act_range_i^0.5 (heuristic from AWQ paper)
        """
        # Only scale channels being promoted from INT4 to INT8
        scales = torch.ones(act_range.shape[0], device=self.device)

        # For salient channels: scale = sqrt(activation range)
        # This ensures high-activation weight channels get finer quantization
        salient_mask = channel_bitwidths == 8
        if salient_mask.any():
            sal_range = act_range[salient_mask].clamp(min=1e-8)
            scales[salient_mask] = sal_range.sqrt()

        return scales

    def _scale_quantize_unscale(
        self,
        module: nn.Module,
        scales: torch.Tensor,
        channel_bitwidths: torch.Tensor,
    ) -> None:
        """
        Apply AWQ: scale weights, quantize per-channel, unscale.

        For Conv2d: weight shape [out, in, kH, kW], scale along dim=1
        For Linear: weight shape [out, in], scale along dim=1
        """
        w = module.weight.data

        if isinstance(module, nn.Conv2d):
            # scales shape: [in_channels] -> broadcast to [1, in, 1, 1]
            s = scales.view(1, -1, 1, 1).to(w.device)
        else:
            # scales shape: [in_features] -> broadcast to [1, in]
            s = scales.view(1, -1).to(w.device)

        # Scale weights
        w_scaled = w * s

        # Quantize per output channel with mixed bitwidths
        # Since per-channel bitwidths apply to input channels,
        # we quantize each output row with the appropriate scale
        qmin_4 = -(2 ** 3)
        qmax_4 = 2 ** 3 - 1
        qmin_8 = -(2 ** 7)
        qmax_8 = 2 ** 7 - 1

        w_flat = w_scaled.reshape(w_scaled.shape[0], w_scaled.shape[1], -1)
        w_q = w_flat.clone()

        for ch_idx in range(w_flat.shape[1]):
            bw = int(channel_bitwidths[ch_idx].item())
            qmin = -(2 ** (bw - 1))
            qmax = 2 ** (bw - 1) - 1

            col = w_flat[:, ch_idx, :]
            amax = col.abs().max().clamp(min=1e-8)
            scale = amax / qmax
            w_q[:, ch_idx, :] = (col / scale).round().clamp(qmin, qmax) * scale

        # Unscale
        w_q = w_q.reshape_as(w_scaled)
        w_result = w_q / s

        module.weight.data = w_result

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
        return "AWQ"
