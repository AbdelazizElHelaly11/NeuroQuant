"""
NeuroQuant v2.0 - Base Quantizer (Abstract)

Defines the common interface that ALL quantization methods must implement.
Ensures uniform evaluation, saving, and metric computation across methods.
"""

from __future__ import annotations

import abc
import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import QuantizationConfig, QuantizationResult
from utils.numerics import MIN_SCALE

logger = logging.getLogger("neuroquant")


class BaseQuantizer(abc.ABC):
    """
    Abstract base class for all quantization methods.

    Every quantizer must be able to:
    1. Accept any nn.Module + config
    2. Quantize the model
    3. Evaluate accuracy / size / EBops
    4. Save the quantized model
    """

    def __init__(self, model: nn.Module, config: QuantizationConfig) -> None:
        """
        Initialize quantizer with a model and configuration.

        Args:
            model: Pre-trained FP32 PyTorch model.
            config: Framework configuration.
        """
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.hyperparams.device)

    @abc.abstractmethod
    def quantize(self, *args: Any, **kwargs: Any) -> nn.Module:
        """
        Apply quantization to the model.

        Returns:
            Quantized nn.Module.
        """
        pass

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        bitwidth: int = 8,
    ) -> QuantizationResult:
        """
        Evaluate a quantized model on all metrics.

        Computes top-1, top-5 accuracy and latency benchmarks.

        Args:
            model: Quantized model to evaluate.
            test_loader: Test DataLoader.
            bitwidth: Quantization bitwidth used.

        Returns:
            QuantizationResult with accuracy, top5, latency, ebops.
        """
        from utils.metrics import compute_topk_accuracy, benchmark_latency

        # Top-k accuracy
        acc = compute_topk_accuracy(model, test_loader, self.device)
        top1 = acc["top1"]
        top5 = acc["top5"]

        # EBops and model size
        ebops = self._compute_ebops(model, bitwidth)
        model_size_mb = ebops / 1e6

        # Latency
        hp = self.config.hyperparams
        latency = benchmark_latency(
            model, self.config.input_shape, self.device,
            batch_size=hp.latency_batch_size,
            warmup_runs=hp.latency_warmup_runs,
            measure_runs=hp.latency_measure_runs,
        )

        return QuantizationResult(
            config_id=f"{self._get_method_name()}_INT{bitwidth}",
            method=self._get_method_name(),
            bitwidth_assignment={
                name: bitwidth for name, _ in model.named_parameters()
                if _.requires_grad or True
            },
            accuracy=top1,
            top5_accuracy=top5,
            model_size_mb=model_size_mb,
            ebops=ebops,
            latency_ms=latency["latency_mean_ms"],
            latency=latency,
            hardware=None,
            model_path=None,
        )

    def save_model(self, model: nn.Module, path: str) -> None:
        """Save quantized model to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p)
        logger.info("  Model saved to: %s", path)

    def _compute_accuracy(
        self, model: nn.Module, data_loader: DataLoader
    ) -> float:
        """Compute top-1 accuracy on a dataset."""
        model.eval()
        model.to(self.device)
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return (correct / max(total, 1)) * 100.0

    def _compute_ebops(self, model: nn.Module, bitwidth: int) -> float:
        """Compute effective bits of precision (memory footprint in bytes)."""
        total = 0.0
        for p in model.parameters():
            total += p.numel() * bitwidth / 8.0
        return total

    @abc.abstractmethod
    def _get_method_name(self) -> str:
        """Return the name of this quantization method."""
        pass

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device_str)

    @staticmethod
    def quantize_tensor(
        tensor: torch.Tensor,
        bitwidth: int,
        per_channel: bool = False,
        channel_dim: int = 0,
    ) -> torch.Tensor:
        """
        Symmetric quantize/dequantize a tensor.

        Args:
            tensor: Float tensor to quantize.
            bitwidth: Target bitwidth (4 or 8).
            per_channel: If True, compute scale per output channel.
            channel_dim: Which dimension is the channel dimension.

        Returns:
            Quantized-then-dequantized tensor (still float, but discretised).
        """
        qmin = -(2 ** (bitwidth - 1))
        qmax = 2 ** (bitwidth - 1) - 1

        if per_channel:
            # Compute scale per channel
            shape = [1] * tensor.dim()
            shape[channel_dim] = tensor.size(channel_dim)
            amax = tensor.abs().flatten(1).max(dim=1)[0]
            amax = amax.clamp(min=MIN_SCALE)
            scale = amax / qmax
            scale = scale.view(shape)
        else:
            amax = tensor.abs().max().clamp(min=MIN_SCALE)
            scale = amax / qmax

        # Quantize then dequantize
        q = (tensor / scale).round().clamp(qmin, qmax)
        return q * scale

    @staticmethod
    def collect_layer_inputs(
        model: nn.Module,
        target_layers: List[str],
        data_loader: DataLoader,
        device: torch.device,
        num_batches: int = 10,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Collect input activations to named layers during calibration.

        Uses forward hooks to capture the inputs flowing into
        each target layer. Works for Conv2d, Linear, etc.

        Args:
            model: The model to instrument.
            target_layers: List of layer names to capture.
            data_loader: Calibration data.
            device: Compute device.
            num_batches: Number of batches to collect.

        Returns:
            Dict mapping layer_name -> list of input tensors.
        """
        collected: Dict[str, List[torch.Tensor]] = {n: [] for n in target_layers}
        hooks = []

        def make_hook(name: str):
            def hook_fn(module, inp, out):
                # inp is a tuple; take first element
                collected[name].append(inp[0].detach().cpu())
            return hook_fn

        for name, module in model.named_modules():
            if name in target_layers:
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)

        model.eval()
        model.to(device)
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                images = batch[0].to(device)
                model(images)

        for h in hooks:
            h.remove()

        return collected
