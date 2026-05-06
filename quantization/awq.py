"""
NeuroQuant v2.0 - AWQ Quantizer (Phase 1f) — production-corrected.

Activation-Aware Weight Quantization following the canonical AWQ
algorithm (Lin et al., MLSys 2024). Crucially, the deployment-equivalent
forward pass is

    Y = (X / s) · quantize(s · W)        (i)

with ``s`` chosen per *input channel* to migrate quantization difficulty
from activations to weights. The previous version of this file
implemented ``Y = X · (quantize(s · W) / s)`` and skipped the
input-side compensation entirely — that does NOT preserve the AWQ
equivalence and is, in effect, a per-channel weight quantizer with
extra buggy bookkeeping. This rewrite:

  * Inserts an ``_AWQInputScale`` wrapper before each quantizable layer
    so equation (i) holds at deployment time. The wrapper is a tiny
    ``nn.Module`` whose buffer is the inverse-scale tensor; mirrors
    SmoothQuant's ``_SmoothInputScale`` for code reuse.
  * Searches the migration exponent α per-layer over a small grid; the
    chosen α minimises the layer-output reconstruction MSE
    ``||Y_q − Y_fp32||²`` on a calibration sample. Single global α
    is rarely optimal, exactly as the AWQ paper observes.
  * Optionally keeps the most-salient top-K% activation channels at
    FP16 (``awq_keep_top_pct``). Default ``0.0`` matches the
    production AWQ recipe; the paper's Section 3 ablation ships this
    knob for completeness.
  * Quantizes the smoothed weight per *output channel* with the
    standard symmetric quantizer — what every INT8 backend actually
    expects. No more per-input-channel mixed-bitwidth
    pseudo-quantization.

The output is mathematically equivalent to the FP32 forward up to
weight quantization error, so the model can be exported to ONNX /
TFLite / TensorRT without separate engineering for the AWQ path.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import QuantizationConfig, QuantizationResult
from quantization.base import BaseQuantizer
from utils.numerics import MIN_SCALE

logger = logging.getLogger("neuroquant")


_QUANTIZABLE_TYPES = (nn.Conv2d, nn.Linear)


class _AWQInputScale(nn.Module):
    """Pre-forward divider used to compensate the AWQ weight smoothing.

    Stored ``inv_scale = 1 / s`` is broadcast against the input channel
    axis: ``[1, C_in, 1, 1]`` for Conv2d, ``[1, C_in]`` for Linear.
    Together with ``W' = W · s`` this preserves the FP32 forward up to
    weight quantization error.
    """

    def __init__(self, inv_scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("inv_scale", inv_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.inv_scale


class AWQQuantizer(BaseQuantizer):
    """AWQ with input-side compensation, per-layer α search, and optional
    salient-channel FP16 carve-out.

    Production contract:
      * Forward equivalence: ``Y_q = (X / s) · quantize_per_out_ch(s · W)``.
      * Per-layer α searched over ``hyperparams.awq_alpha_grid`` (default
        ``[0.0, 0.25, 0.5, 0.75, 1.0]``); the value minimising
        layer-output MSE on calibration data wins.
      * Salient-channel keep at FP16 controlled by
        ``hyperparams.awq_keep_top_pct`` (default ``0.0``). The carved
        channels are *not* fake-quantized — their column of ``W`` is
        copied back over the quantized result before division by ``s``.
    """

    def __init__(self, model: nn.Module, config: QuantizationConfig) -> None:
        super().__init__(model, config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(
        self,
        calibration_loader: DataLoader,
        bitwidth: int = 4,
        num_batches: int = 10,
    ) -> nn.Module:
        start = time.time()
        q_model = copy.deepcopy(self.model)
        q_model.to(self.device)
        q_model.eval()

        hp = self.config.hyperparams
        alpha_grid = list(getattr(
            hp, "awq_alpha_grid", [0.0, 0.25, 0.5, 0.75, 1.0],
        ))
        keep_top_pct = float(getattr(hp, "awq_keep_top_pct", 0.0))
        target_layers = self._find_quantizable_layers(q_model)
        logger.info(
            "AWQ INT%d: %d layers, α-grid=%s, keep_top_pct=%.2f%%",
            bitwidth, len(target_layers), alpha_grid, keep_top_pct * 100.0,
        )

        # Step 1: Collect per-input-channel activation statistics + a
        # small input pool used to score α candidates per layer.
        act_max = self._collect_activation_stats(
            q_model, target_layers, calibration_loader, num_batches,
        )
        layer_x_pool = self._collect_layer_input_pool(
            q_model, target_layers, calibration_loader, num_batches,
            max_samples=64,
        )

        chosen_alpha: Dict[str, float] = {}
        salient_kept: Dict[str, int] = {}

        # Step 2 & 3: per-layer α search → smooth + quantize → optional
        # salient-channel restore → wrap input scale.
        for layer_name, module in target_layers.items():
            # Skip depthwise convolutions: per-input-channel scaling
            # would corrupt the weight shape because groups != 1.
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                with torch.no_grad():
                    module.weight.data = self.quantize_tensor(
                        module.weight.data, bitwidth, per_channel=True,
                    )
                logger.info(
                    "    %s: standard INT%d (depthwise conv)",
                    layer_name, bitwidth,
                )
                continue

            if layer_name not in act_max or layer_name not in layer_x_pool:
                with torch.no_grad():
                    module.weight.data = self.quantize_tensor(
                        module.weight.data, bitwidth, per_channel=True,
                    )
                logger.info(
                    "    %s: standard INT%d (no calibration data)",
                    layer_name, bitwidth,
                )
                continue

            a_max = act_max[layer_name].to(self.device)
            x_sample = layer_x_pool[layer_name].to(self.device)

            # ── Per-layer α search ──
            best_alpha = self._search_layer_alpha(
                module, x_sample, a_max, alpha_grid, bitwidth,
                keep_top_pct=keep_top_pct,
            )
            chosen_alpha[layer_name] = float(best_alpha)

            # Compute s = a_max^α (per input channel). The AWQ
            # formulation does not divide by w^(1-α) — that is
            # SmoothQuant. Pure activation-driven scaling is the AWQ
            # signature.
            s = self._compute_awq_scales(a_max, best_alpha)

            # Identify salient-channel mask (top-K% by activation),
            # used both to skip quantization on those columns and to
            # log the count.
            salient_mask = self._top_k_mask(a_max, keep_top_pct)
            salient_kept[layer_name] = int(salient_mask.sum().item())

            # Apply: insert input divider, smooth weights, quantize,
            # restore salient columns.
            self._wrap_input_scale(q_model, layer_name, module, s)
            self._apply_smoothing_and_quantize(
                module, s, bitwidth, salient_mask,
            )

            logger.info(
                "    %s: α=%.2f, INT%d, scale=[%.3f, %.3f], salient=%d",
                layer_name, best_alpha, bitwidth,
                float(s.min()), float(s.max()),
                int(salient_mask.sum().item()),
            )

        # Stash diagnostics on the model so the artifact JSON / debug
        # path can read them back without re-running the search.
        q_model._awq_alpha = chosen_alpha            # type: ignore[attr-defined]
        q_model._awq_salient_kept = salient_kept     # type: ignore[attr-defined]

        elapsed = time.time() - start
        total_salient = sum(salient_kept.values())
        logger.info(
            "AWQ INT%d complete: %d salient channels kept FP16, %.1fs",
            bitwidth, total_salient, elapsed,
        )
        return q_model

    # ------------------------------------------------------------------
    # Calibration: per-channel max + input pool for α search
    # ------------------------------------------------------------------

    def _collect_activation_stats(
        self,
        model: nn.Module,
        target_layers: Dict[str, nn.Module],
        data_loader: DataLoader,
        num_batches: int,
    ) -> Dict[str, torch.Tensor]:
        max_acts: Dict[str, torch.Tensor] = {}
        hooks = []

        def _make_hook(name: str, is_conv: bool):
            def _hook(_mod, inp, _out):
                x = inp[0].detach()
                if is_conv and x.dim() == 4:
                    ch_max = x.abs().amax(dim=(0, 2, 3))
                elif x.dim() >= 2:
                    dims = [d for d in range(x.dim()) if d != 1]
                    ch_max = x.abs().amax(dim=dims) if dims else x.abs().squeeze()
                else:
                    ch_max = x.abs()
                ch_max = ch_max.cpu()
                if name in max_acts:
                    max_acts[name] = torch.max(max_acts[name], ch_max)
                else:
                    max_acts[name] = ch_max
            return _hook

        for name, module in target_layers.items():
            hooks.append(module.register_forward_hook(
                _make_hook(name, isinstance(module, nn.Conv2d)),
            ))

        try:
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= num_batches:
                        break
                    images = batch[0].to(self.device)
                    model(images)
        finally:
            for h in hooks:
                h.remove()
        return max_acts

    def _collect_layer_input_pool(
        self,
        model: nn.Module,
        target_layers: Dict[str, nn.Module],
        data_loader: DataLoader,
        num_batches: int,
        max_samples: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """Capture a bounded per-layer input sample for the α search.

        Pooled once on the FP32 baseline so every α candidate scores
        against the same X. Bounded to ``max_samples`` per layer to
        keep the grid search cheap.
        """
        pools: Dict[str, List[torch.Tensor]] = {n: [] for n in target_layers}
        taken: Dict[str, int] = {n: 0 for n in target_layers}
        hooks = []

        def _make_hook(name: str):
            def _hook(_mod, inp, _out):
                if not inp or taken[name] >= max_samples:
                    return
                x = inp[0].detach()
                n = min(int(x.shape[0]), max_samples - taken[name])
                if n > 0:
                    pools[name].append(x[:n].cpu())
                    taken[name] += n
            return _hook

        for name, module in target_layers.items():
            hooks.append(module.register_forward_hook(_make_hook(name)))

        try:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= num_batches:
                        break
                    images = batch[0].to(self.device)
                    model(images)
        finally:
            for h in hooks:
                h.remove()
        return {n: torch.cat(chunks, dim=0)
                for n, chunks in pools.items() if chunks}

    # ------------------------------------------------------------------
    # AWQ math
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_awq_scales(a_max: torch.Tensor, alpha: float) -> torch.Tensor:
        """``s_j = max(a_j, ε)^α`` — pure activation-driven scaling.

        Floored at MIN_SCALE because dead channels (a_j ≈ 0) would
        yield ``s_j ≈ 0`` and an infinite division by ``s`` in the
        compensating wrapper.
        """
        s = a_max.clamp(min=MIN_SCALE).pow(float(alpha))
        # Rescale so the *mean* scale is 1, so the network's effective
        # input magnitude is preserved (otherwise α=1 dilates X by
        # roughly 1/||a||). This is the same normalisation the AWQ
        # reference implementation uses.
        s = s / s.mean().clamp(min=MIN_SCALE)
        return s.clamp(min=MIN_SCALE)

    @staticmethod
    def _top_k_mask(a_max: torch.Tensor, keep_top_pct: float) -> torch.Tensor:
        """Boolean mask of the top-K% channels by activation magnitude.

        Returns an all-False mask when ``keep_top_pct == 0``. Channels
        in the True positions skip weight quantization and are kept at
        FP16 in the final model.
        """
        n = int(a_max.numel())
        if keep_top_pct <= 0.0 or n == 0:
            return torch.zeros(n, dtype=torch.bool, device=a_max.device)
        k = max(1, int(round(n * keep_top_pct)))
        thresh = a_max.kthvalue(n - k + 1).values
        return a_max >= thresh

    def _search_layer_alpha(
        self,
        module: nn.Module,
        x: torch.Tensor,
        a_max: torch.Tensor,
        alpha_grid: List[float],
        bitwidth: int,
        keep_top_pct: float,
    ) -> float:
        """Pick α minimising the AWQ-equivalent layer output MSE."""
        original_w = module.weight.detach()

        if isinstance(module, nn.Conv2d):
            forward = lambda m, xi, w: F.conv2d(
                xi, w, bias=m.bias,
                stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups,
            )
            shape_w = (1, -1, 1, 1)
            shape_x = (1, -1, 1, 1)
        elif isinstance(module, nn.Linear):
            forward = lambda m, xi, w: F.linear(xi, w, m.bias)
            shape_w = (1, -1)
            shape_x = (1, -1)
        else:
            return float(alpha_grid[len(alpha_grid) // 2])

        with torch.no_grad():
            y_ref = forward(module, x, original_w)

        salient_mask = self._top_k_mask(a_max, keep_top_pct)

        best_alpha = alpha_grid[0]
        best_mse = float("inf")
        with torch.no_grad():
            for alpha in alpha_grid:
                s = self._compute_awq_scales(a_max, float(alpha))
                # Smooth and quantize per output channel.
                w_smoothed = original_w * s.view(*shape_w)
                w_q = self.quantize_tensor(
                    w_smoothed, bitwidth, per_channel=True,
                )
                # Salient carve-out: copy back FP16 columns over the
                # quantized result before the final divide.
                if salient_mask.any():
                    w_q = self._restore_salient_columns(
                        w_q, w_smoothed, salient_mask,
                    )
                # Forward with input-side compensation.
                x_compensated = x / s.view(*shape_x)
                y_q = forward(module, x_compensated, w_q)
                mse = float((y_q - y_ref).pow(2).mean().item())
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = float(alpha)
        return best_alpha

    @staticmethod
    def _restore_salient_columns(
        w_q: torch.Tensor,
        w_smoothed: torch.Tensor,
        salient_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace quantized salient columns with their FP16 originals.

        Operates on the input-channel axis (dim=1) of weight tensors —
        the same axis the smoothing scale was applied on, so the AWQ
        equivalence ``Y = (X/s) · quantize_per_out_ch(s·W)`` keeps
        holding for non-salient channels while salient channels stay
        in FP16 untouched (other than the smoothing factor s).
        """
        idx = torch.where(salient_mask)[0]
        if idx.numel() == 0:
            return w_q
        out = w_q.clone()
        if w_q.dim() == 4:  # Conv2d: [out, in, kH, kW]
            out[:, idx, :, :] = w_smoothed[:, idx, :, :]
        elif w_q.dim() == 2:  # Linear: [out, in]
            out[:, idx] = w_smoothed[:, idx]
        return out

    # ------------------------------------------------------------------
    # Apply the smoothing + quantization in place
    # ------------------------------------------------------------------

    def _wrap_input_scale(
        self,
        root: nn.Module,
        full_name: str,
        module: nn.Module,
        scales: torch.Tensor,
    ) -> None:
        """Replace ``root.<full_name>`` with ``Sequential(_AWQInputScale, module)``
        so the layer's input is divided by ``s`` at forward time."""
        if isinstance(module, nn.Conv2d):
            inv = (1.0 / scales).view(1, -1, 1, 1).detach().clone()
        else:
            inv = (1.0 / scales).view(1, -1).detach().clone()

        wrapper = nn.Sequential(_AWQInputScale(inv), module)
        wrapper.to(module.weight.device)

        parent_name, _, attr = full_name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, attr, wrapper)

    def _apply_smoothing_and_quantize(
        self,
        module: nn.Module,
        scales: torch.Tensor,
        bitwidth: int,
        salient_mask: torch.Tensor,
    ) -> None:
        """In-place: smooth weights, quantize per-output-channel,
        restore salient columns to FP16."""
        w = module.weight.data
        if isinstance(module, nn.Conv2d):
            s = scales.view(1, -1, 1, 1)
        else:
            s = scales.view(1, -1)

        w_smoothed = w * s
        w_q = self.quantize_tensor(
            w_smoothed, bitwidth, per_channel=True, channel_dim=0,
        )
        if salient_mask.any():
            w_q = self._restore_salient_columns(w_q, w_smoothed, salient_mask)
        module.weight.data = w_q

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_quantizable_layers(
        self, model: nn.Module,
    ) -> Dict[str, nn.Module]:
        return {
            name: m for name, m in model.named_modules()
            if isinstance(m, _QUANTIZABLE_TYPES)
        }

    def _get_method_name(self) -> str:
        return "AWQ"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Safe (no-pickle) persistence — symmetric with SmoothQuant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Like SmoothQuant, AWQ inserts an architectural wrapper
# (_AWQInputScale) per quantized layer. The same JSON-safe metadata
# pattern lets the resume path rebuild wrappers under
# ``torch.load(weights_only=True)`` instead of pickling the whole
# module.


def serialize_awq_metadata(model: nn.Module) -> Dict[str, Any]:
    """Return a JSON-safe description of every AWQ wrapper in ``model``."""
    entries: List[Dict[str, Any]] = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Sequential) or len(m) != 2:
            continue
        head = m[0]
        if not isinstance(head, _AWQInputScale):
            continue
        inv = head.inv_scale
        entries.append({
            "name": name,
            "inv_scale_shape": list(inv.shape),
            "inv_scale_dtype": str(inv.dtype),
        })
    return {"wrappers": entries}


def restore_awq_wrappers(
    model: nn.Module,
    metadata: Dict[str, Any],
) -> nn.Module:
    """Rebuild AWQ wrappers on ``model`` according to ``metadata``."""
    wrappers = (metadata or {}).get("wrappers", []) or []
    for entry in wrappers:
        name = entry["name"]
        shape = tuple(int(d) for d in entry["inv_scale_shape"])
        try:
            dtype = getattr(torch, str(entry["inv_scale_dtype"]).split(".")[-1])
        except AttributeError:
            dtype = torch.float32
        try:
            inner = model.get_submodule(name)
        except AttributeError:
            logger.warning(
                "  [AWQ restore] submodule '%s' missing; skipping.", name,
            )
            continue
        if isinstance(inner, nn.Sequential) and len(inner) == 2 \
                and isinstance(inner[0], _AWQInputScale):
            continue
        placeholder = torch.zeros(shape, dtype=dtype)
        wrapper = nn.Sequential(_AWQInputScale(placeholder), inner)
        try:
            wrapper.to(next(inner.parameters()).device)
        except StopIteration:
            pass
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, wrapper)
    return model
