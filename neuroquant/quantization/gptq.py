"""
NeuroQuant v2.0 - GPTQ Quantizer (Phase 1f)

Generalized Post-Training Quantization using approximate
second-order information (Hessian inverse) for optimal
weight quantization. Supports INT8 and INT4.

Reference: Frantar et al., "GPTQ: Accurate Post-Training
Quantization for Generative Pre-Trained Transformers" (2023).

Algorithm:
    For each layer with weight matrix W (reshaped to 2D):
    1. Compute H = X^T X (Hessian approximation from calibration inputs)
    2. Add dampening: H += lambda * I  (lambda = damp_percent * mean(diag(H)))
    3. Invert: H_inv = (H + lambda*I)^-1
    4. For each column i (sequentially):
       a. Quantize: w_q_i = round(w_i / scale) * scale
       b. Error: e_i = (w_i - w_q_i) / H_inv[i,i]
       c. Propagate: w_{j>i} -= e_i * H_inv[i,j] / H_inv[i,i]
    Result: Weights minimally disturbed by quantization error

CNN Adaptation:
    - Conv2d weights [out, in, kH, kW] reshaped to [out, in*kH*kW]
    - Conv2d inputs unfolded via im2col to [batch*H_out*W_out, in*kH*kW]
    - Linear inputs already [batch, in_features]
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuroquant.config import QuantizationConfig, QuantizationResult
from neuroquant.quantization.base import BaseQuantizer
from neuroquant.utils.numerics import MIN_DAMP, MIN_SCALE

logger = logging.getLogger("neuroquant")


class GPTQQuantizer(BaseQuantizer):
    """
    GPTQ: Layer-wise quantization using approximate Hessian inverse.

    Quantizes weights one column at a time, using the inverse
    Hessian to optimally distribute quantization error across
    remaining unquantized weights.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None,
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model, config, device=device)

    def quantize(
        self,
        calibration_loader: DataLoader,
        bitwidth: int = 8,
        num_batches: int = 10,
    ) -> nn.Module:
        """
        Apply GPTQ quantization at the specified bitwidth.

        Args:
            calibration_loader: Calibration data for Hessian estimation.
            bitwidth: Target bitwidth (4 or 8).
            num_batches: Number of calibration batches.

        Returns:
            GPTQ-quantized model (deep copy, original unchanged).
        """
        start = time.time()
        q_model = copy.deepcopy(self.model)
        q_model.to(self.device)
        q_model.eval()

        hp = self.config.hyperparams
        damp_pct = hp.gptq_percdamp

        # Find quantizable layers
        target_layers = self._find_quantizable_layers(q_model)
        logger.info("GPTQ INT%d: %d quantizable layers", bitwidth, len(target_layers))

        # Collect layer inputs via calibration
        layer_inputs = self.collect_layer_inputs(
            q_model, list(target_layers.keys()),
            calibration_loader, self.device, num_batches,
        )

        # Quantize each layer. The per-layer calibration inputs are kept
        # on the CPU by ``collect_layer_inputs`` and streamed onto the GPU
        # one chunk at a time inside ``_quantize_layer`` — we deliberately
        # do NOT ``torch.cat(...).to(device)`` the whole stack, because the
        # im2col expansion of that concatenation is what OOMs a small GPU
        # (e.g. a 20 GB A100 MIG slice) on large convs.
        for layer_name, module in target_layers.items():
            inputs = layer_inputs.get(layer_name) or []
            if not inputs:
                logger.warning("  No calibration data for '%s', skipping", layer_name)
                continue

            self._quantize_layer(module, layer_name, inputs, bitwidth, damp_pct)

            # Release this layer's calibration activations as we advance so
            # peak CPU RAM doesn't grow with model depth, and reclaim any
            # freed GPU blocks before the next (possibly larger) layer.
            layer_inputs[layer_name] = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.time() - start
        logger.info("GPTQ INT%d complete in %.1fs", bitwidth, elapsed)
        return q_model

    def _quantize_layer(
        self,
        layer: nn.Module,
        layer_name: str,
        inputs: List[torch.Tensor],
        bitwidth: int,
        damp_pct: float,
    ) -> None:
        """
        Quantize a single layer using the GPTQ OBS algorithm.

        Reshapes Conv2d weights to 2D and unfolds Conv2d inputs via im2col
        for Hessian computation. ``inputs`` is the list of per-batch
        calibration activations captured by ``collect_layer_inputs`` (kept
        on the CPU); they are streamed onto the GPU in row-chunks so the
        full im2col matrix is never materialised at once.

        Falls back to standard per-channel quantization when:
          * the layer is depthwise (groups > 1) — Hessian dim mismatch,
          * the unfolded input width doesn't match the weight columns, or
          * the full GPTQ step OOMs and ``gptq_oom_fallback`` is enabled.
        """
        weight = layer.weight.data
        is_conv = isinstance(layer, nn.Conv2d)

        def _standard_fallback(reason: str) -> None:
            with torch.no_grad():
                layer.weight.data = self.quantize_tensor(
                    weight, bitwidth, per_channel=True,
                )
            logger.info("    %s: standard INT%d (%s)",
                        layer_name, bitwidth, reason)

        # Skip depthwise convolutions (groups > 1): Hessian dim mismatch
        if is_conv and layer.groups > 1:
            _standard_fallback(f"depthwise conv, groups={layer.groups}")
            return

        # Reshape weight to 2D: [out_features, in_features]
        if is_conv:
            out_ch = weight.shape[0]
            w_2d = weight.reshape(out_ch, -1)  # [out, in*kH*kW]
        else:
            w_2d = weight.clone()  # [out, in]
        in_features = w_2d.shape[1]

        hp = self.config.hyperparams
        chunk_rows = int(getattr(hp, "gptq_hessian_chunk_rows", 8192) or 0)
        oom_fallback = bool(getattr(hp, "gptq_oom_fallback", True))

        try:
            # ── Memory-bounded Hessian accumulation: H = (1/N) Σ xᵀx ──
            # The full im2col matrix is the OOM culprit on large convs, so
            # we stream each calibration batch through im2col, then add its
            # rows into H in sub-chunks of ``chunk_rows``. Peak activation
            # memory is one sub-chunk, not the whole calibration set.
            H = torch.zeros(in_features, in_features, device=self.device)
            n_samples = 0
            for chunk in inputs:
                xc = chunk.to(self.device, non_blocking=True)
                if is_conv:
                    x_2d = self._unfold_conv_input(xc, layer)
                elif xc.dim() > 2:
                    x_2d = xc.reshape(-1, xc.shape[-1])
                else:
                    x_2d = xc

                # Verify dims once on the first chunk; bail to fallback.
                if x_2d.shape[1] != in_features:
                    mismatch = x_2d.shape[1]
                    del xc, x_2d, H
                    _standard_fallback(
                        f"dim mismatch: H={mismatch} vs W={in_features}")
                    return

                rows = x_2d.shape[0]
                step = chunk_rows if chunk_rows > 0 else rows
                for r0 in range(0, rows, step):
                    xr = x_2d[r0:r0 + step]
                    H += xr.t() @ xr
                    n_samples += xr.shape[0]
                del xc, x_2d

            if n_samples == 0:
                del H
                _standard_fallback("no calibration rows")
                return
            H /= n_samples

            # Dampening — add to the diagonal in place (no dense identity).
            diag_mean = H.diag().mean().item()
            damp = max(damp_pct * diag_mean, MIN_DAMP)
            H.diagonal().add_(damp)

            # Invert Hessian (Cholesky for numerical stability).
            try:
                L = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(L)
                del L
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise
                # Non-PD matrix: add more dampening and use a dense inverse.
                H.diagonal().add_(0.1)
                H_inv = torch.inverse(H)
            del H

            # GPTQ: quantize columns sequentially with error propagation.
            n_cols = in_features
            qmin = -(2 ** (bitwidth - 1))
            qmax = 2 ** (bitwidth - 1) - 1

            for col in range(n_cols):
                w_col = w_2d[:, col].clone()

                # Compute scale for this column.
                amax = w_col.abs().max().clamp(min=MIN_SCALE)
                scale = amax / qmax

                # Quantize.
                w_q = (w_col / scale).round().clamp(qmin, qmax) * scale

                # Compute error.
                error = w_col - w_q

                # Apply quantized value.
                w_2d[:, col] = w_q

                # Propagate error to subsequent columns via Hessian inverse.
                if col < n_cols - 1:
                    h_ii = H_inv[col, col].clamp(min=MIN_SCALE)
                    w_2d[:, col + 1:] -= (
                        error.unsqueeze(1) * H_inv[col, col + 1:].unsqueeze(0) / h_ii
                    )

            del H_inv

            # Write back.
            if is_conv:
                layer.weight.data = w_2d.reshape_as(weight)
            else:
                layer.weight.data = w_2d

            logger.info("    %s: GPTQ INT%d applied (%d cols)",
                         layer_name, bitwidth, n_cols)

        except RuntimeError as exc:
            # CUDA OOM (or any runtime failure) on this single layer: keep
            # the method alive by degrading to per-channel rounding for just
            # this layer instead of aborting all of GPTQ.
            if "out of memory" not in str(exc).lower() or not oom_fallback:
                raise
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            _standard_fallback("OOM on full GPTQ → per-channel fallback")

    def _unfold_conv_input(
        self, inp: torch.Tensor, layer: nn.Conv2d
    ) -> torch.Tensor:
        """
        Unfold conv2d input via im2col for Hessian computation.

        Converts [batch, in_ch, H, W] -> [batch*H_out*W_out, in_ch*kH*kW]
        """
        # Use F.unfold to extract sliding windows
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        dilation = layer.dilation

        # unfold: [batch, in_ch*kH*kW, L] where L = H_out * W_out
        unfolded = F.unfold(
            inp, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )

        # Transpose and reshape: [batch*L, in_ch*kH*kW]
        batch, features, L = unfolded.shape
        x_2d = unfolded.permute(0, 2, 1).reshape(batch * L, features)
        return x_2d

    def _find_quantizable_layers(
        self, model: nn.Module
    ) -> Dict[str, nn.Module]:
        """Find all Conv2d and Linear layers to quantize."""
        layers = {}
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                layers[name] = m
        return layers

    def _get_method_name(self) -> str:
        return "GPTQ"
