"""
NeuroQuant v2.0 - Combined SmoothQuant → GPTQ (Phase 1f, F4).

Production-grade two-stage quantizer:

  1. ``SmoothQuantQuantizer.apply_smoothing_only`` migrates per-channel
     activation difficulty into the weights via ``W' = s · W`` and
     inserts a compensating ``X' = X / s`` wrapper before each layer.
     The result is an FP32 model whose forward is mathematically
     identical to the original, but whose weights are *easier to
     quantize* because the per-channel weight magnitudes are now
     proportional to the per-channel activation magnitudes.

  2. ``GPTQQuantizer.quantize`` then runs the optimal-rounding GPTQ
     algorithm on the smoothed weights, with calibration activations
     captured *after* the input-scaling wrapper — so GPTQ sees the
     post-divide activations that the deployment graph actually uses.

Either method alone leaves accuracy on the table:

  * SmoothQuant alone uses round-to-nearest per-output-channel and
    cannot exploit second-order weight interactions.
  * GPTQ alone has to handle the raw activation outliers in its
    Hessian, which inflates per-channel scales and wastes INT8 codes
    on values that almost never occur.

Combined, the migration pre-conditions GPTQ for a much better result
— this is the standard production recipe for both LLM and CNN
quantization shipping in 2024+.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import QuantizationConfig, QuantizationResult
from quantization.base import BaseQuantizer
from quantization.gptq import GPTQQuantizer
from quantization.smoothquant import (
    SmoothQuantQuantizer,
    _SmoothInputScale,
    serialize_smoothquant_metadata,
)

logger = logging.getLogger("neuroquant")


class SmoothQuantGPTQQuantizer(BaseQuantizer):
    """SmoothQuant migration followed by GPTQ on the smoothed weights.

    Compared to either method alone:
      * The input-scale wrappers preserve activation-side FP16 — GPTQ
        sees post-divide activations and computes a cleaner Hessian.
      * GPTQ replaces SmoothQuant's per-channel round-to-nearest,
        recovering the optimal-rounding accuracy GPTQ is known for.

    Compared to running them sequentially in user code, this class:
      * Reuses SmoothQuant's per-layer α grid search (E3 production
        contract) so the migration scales are already tuned by the time
        GPTQ takes over.
      * Persists the SmoothQuant wrapper manifest under the same JSON
        contract used by ``serialize_smoothquant_metadata`` — the
        combined model resumes safely under ``weights_only=True``.
    """

    def __init__(self, model: nn.Module, config: QuantizationConfig) -> None:
        super().__init__(model, config)

    def quantize(
        self,
        calibration_loader: DataLoader,
        bitwidth: int = 8,
        num_batches: int = 10,
    ) -> nn.Module:
        start = time.time()
        logger.info(
            "SmoothQuant→GPTQ INT%d: starting two-stage quantization.",
            bitwidth,
        )

        # ── Stage 1: SmoothQuant migration only ──
        sq = SmoothQuantQuantizer(self.model, self.config)
        smoothed = sq.apply_smoothing_only(
            calibration_loader, num_batches=num_batches,
        )
        # Move to device (apply_smoothing_only already does, but be
        # explicit so the GPTQ stage runs on the same device).
        smoothed.to(self.device)

        # ── Stage 2: GPTQ on the smoothed model ──
        # GPTQ deep-copies internally, so ``smoothed`` stays untouched
        # and we receive a fresh quantized model. GPTQ collects layer
        # inputs by walking ``named_modules()``; the ``_SmoothInputScale``
        # wrappers are between the parent and the inner Conv/Linear, so
        # GPTQ correctly captures the post-divide activations.
        gptq = GPTQQuantizer(smoothed, self.config)
        q_model = gptq.quantize(
            calibration_loader, bitwidth=bitwidth,
            num_batches=num_batches,
        )

        # Carry forward the SmoothQuant per-layer α dict (for diagnostics)
        # and the wrapper manifest (for safe resume).
        if hasattr(smoothed, "_smoothquant_alpha"):
            q_model._smoothquant_alpha = (  # type: ignore[attr-defined]
                smoothed._smoothquant_alpha
            )
        q_model._smoothquant_metadata = (  # type: ignore[attr-defined]
            serialize_smoothquant_metadata(q_model)
        )

        elapsed = time.time() - start
        logger.info(
            "SmoothQuant→GPTQ INT%d complete in %.1fs.",
            bitwidth, elapsed,
        )
        return q_model

    def _get_method_name(self) -> str:
        return "SmoothQuantGPTQ"
