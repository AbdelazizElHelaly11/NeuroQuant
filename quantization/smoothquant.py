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
from utils.numerics import MAX_MIGRATION, MIN_MIGRATION, MIN_SCALE

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

    def apply_smoothing_only(
        self,
        calibration_loader: DataLoader,
        num_batches: int = 10,
    ) -> nn.Module:
        """Run SmoothQuant migration WITHOUT the final weight quantization.

        Returns a deep-copied model with:
          * ``_SmoothInputScale`` wrappers in front of every quantizable
            layer (per-channel input divide ``X' = X / s``);
          * smoothed FP32 weights ``W' = W · s`` on those layers;
          * per-layer α stashed under ``model._smoothquant_alpha``.

        Used by ``SmoothQuantGPTQQuantizer`` (F4) — the combined method
        feeds this output into GPTQ instead of the per-channel symmetric
        quantizer SmoothQuant would otherwise apply, recovering both
        the input-scale equivalence AND GPTQ's optimal-rounding effect.
        """
        return self._quantize_or_smooth(
            calibration_loader, bitwidth=None, num_batches=num_batches,
        )

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
        2. Compute per-channel smoothing scales (per-layer α grid search
           when ``smoothquant_per_layer_alpha`` is enabled)
        3. Absorb scales into weights (mathematically equivalent)
        4. Quantize smoothed weights per-channel symmetrically

        Args:
            calibration_loader: Calibration data for scale computation.
            bitwidth: Target bitwidth (4 or 8).
            num_batches: Number of calibration batches.

        Returns:
            SmoothQuant-quantized model (deep copy).
        """
        return self._quantize_or_smooth(
            calibration_loader, bitwidth=bitwidth, num_batches=num_batches,
        )

    def _quantize_or_smooth(
        self,
        calibration_loader: DataLoader,
        bitwidth: Optional[int],
        num_batches: int,
    ) -> nn.Module:
        """Internal: shared implementation of ``quantize`` and
        ``apply_smoothing_only``. ``bitwidth=None`` skips the final
        per-channel weight quantization step so a downstream quantizer
        (e.g. GPTQ in F4) can take over."""
        start = time.time()
        q_model = copy.deepcopy(self.model)
        q_model.to(self.device)
        q_model.eval()

        hp = self.config.hyperparams
        global_alpha = float(hp.smoothquant_alpha)
        per_layer = bool(getattr(hp, "smoothquant_per_layer_alpha", False))
        alpha_grid = list(getattr(
            hp, "smoothquant_alpha_grid", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ))

        target_layers = self._find_quantizable_layers(q_model)
        smooth_only = bitwidth is None
        bw_label = "smooth-only" if smooth_only else f"INT{bitwidth}"
        logger.info(
            "SmoothQuant %s: %d layers, %s α (%s)",
            bw_label, len(target_layers),
            "per-layer" if per_layer else "global",
            f"grid={alpha_grid}" if per_layer else f"α={global_alpha:.2f}",
        )

        # Step 1: Collect activation statistics + (when grid-searching)
        # one calibration sample per layer for the α scoring step.
        act_max, weight_max = self._collect_stats(
            q_model, target_layers, calibration_loader, num_batches,
        )
        layer_x_samples: Dict[str, torch.Tensor] = {}
        if per_layer:
            layer_x_samples = self._collect_layer_inputs_for_alpha_search(
                q_model, target_layers, calibration_loader, num_batches,
            )

        chosen_alpha: Dict[str, float] = {}

        # Step 2 & 3: Compute scales and smooth weights
        for layer_name, module in target_layers.items():
            # Skip depthwise convolutions: scale broadcast corrupts weight shape
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                if not smooth_only:
                    with torch.no_grad():
                        module.weight.data = self.quantize_tensor(
                            module.weight.data, bitwidth, per_channel=True,
                        )
                logger.info("    %s: standard %s (depthwise conv)",
                             layer_name, bw_label)
                continue

            if layer_name not in act_max:
                # No activation data; standard quantization (skip when
                # smooth_only — leave the layer FP32 for the downstream
                # method to handle).
                if not smooth_only:
                    with torch.no_grad():
                        module.weight.data = self.quantize_tensor(
                            module.weight.data, bitwidth, per_channel=True,
                        )
                logger.info("    %s: standard %s (no calibration data)",
                             layer_name, bw_label)
                continue

            a_max = act_max[layer_name].to(self.device)
            w_max = weight_max[layer_name].to(self.device)

            # Choose α: either per-layer grid search or the global value.
            # The α search needs a quantizer; when smooth_only is set we
            # score against the bitwidth that the downstream method
            # (e.g. GPTQ) will use — assume INT8 as a conservative default.
            search_bw = bitwidth if bitwidth is not None else 8
            if per_layer and layer_name in layer_x_samples:
                alpha = self._search_layer_alpha(
                    module, layer_x_samples[layer_name].to(self.device),
                    a_max, w_max, alpha_grid, search_bw,
                )
            else:
                alpha = global_alpha
            chosen_alpha[layer_name] = float(alpha)

            # Compute smooth scales using the chosen α.
            scales = self._compute_smooth_scales(a_max, w_max, alpha)

            # Insert the compensating input divide (X' = X / s) before the
            # layer, then absorb the inverse scaling into its weights
            # (W' = W * s). Together these give Y = X @ W unchanged
            # mathematically, which is what makes the transform lossless
            # before quantization kicks in.
            self._wrap_input_scale(q_model, layer_name, module, scales)
            self._apply_smoothing(module, scales)

            # Quantize smoothed weights — skipped when smooth_only so the
            # downstream method receives FP32 smoothed weights.
            if not smooth_only:
                with torch.no_grad():
                    module.weight.data = self.quantize_tensor(
                        module.weight.data, bitwidth, per_channel=True,
                    )

            logger.info(
                "    %s: smoothed (α=%.2f) + %s (scale range=[%.4f, %.4f])",
                layer_name, alpha, bw_label,
                scales.min().item(), scales.max().item(),
            )

        # Stash the per-layer α for the artifact JSON / debugging.
        q_model._smoothquant_alpha = chosen_alpha  # type: ignore[attr-defined]

        elapsed = time.time() - start
        logger.info("SmoothQuant %s complete in %.1fs", bw_label, elapsed)
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
        # Clamp to avoid division by zero (per-tensor scale floor).
        a = act_max.clamp(min=MIN_SCALE)
        w = weight_max.clamp(min=MIN_SCALE)

        # SmoothQuant formula
        scales = a.pow(alpha) / w.pow(1.0 - alpha)

        # Clamp migration scales to a wider band — typical SmoothQuant
        # values span 0.01–100 and the clamp must not neuter migration.
        scales = scales.clamp(min=MIN_MIGRATION, max=MAX_MIGRATION)

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

    def _collect_layer_inputs_for_alpha_search(
        self,
        model: nn.Module,
        target_layers: Dict[str, nn.Module],
        data_loader: DataLoader,
        num_batches: int,
        max_samples_per_layer: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """Capture a small per-layer input pool used by the α search.

        ``num_batches`` × ``batch_size`` samples flow through the model;
        we keep at most ``max_samples_per_layer`` per layer to bound
        the cost of the grid search. The pool is collected ONCE on the
        FP32 baseline so all α candidates score against the same X.
        """
        pools: Dict[str, List[torch.Tensor]] = {n: [] for n in target_layers}
        taken: Dict[str, int] = {n: 0 for n in target_layers}
        hooks = []

        def _make_hook(name: str):
            def _hook(_mod, inputs, _output):
                if not inputs or taken[name] >= max_samples_per_layer:
                    return
                x = inputs[0].detach()
                n = min(int(x.shape[0]), max_samples_per_layer - taken[name])
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

    def _search_layer_alpha(
        self,
        module: nn.Module,
        x: torch.Tensor,
        a_max: torch.Tensor,
        w_max: torch.Tensor,
        alpha_grid: List[float],
        bitwidth: int,
    ) -> float:
        """Pick α ∈ ``alpha_grid`` that minimises layer-output recon MSE.

        For each candidate α we simulate the full SmoothQuant transform —
        per-channel ``s = a^α / w^(1-α)``, weight smoothing ``W' = W·s``,
        per-channel symmetric quantization, and the compensating input
        divide ``X' = X / s`` — then compare the resulting layer output
        against the FP32 reference ``layer(X; W)``. The mathematics is
        the same one the deployment path executes; minimising this MSE
        is the correctness signal a global α cannot deliver.
        """
        original_w = module.weight.detach()
        if isinstance(module, nn.Conv2d):
            forward = lambda mod, xi, w: torch.nn.functional.conv2d(
                xi, w, bias=mod.bias,
                stride=mod.stride, padding=mod.padding,
                dilation=mod.dilation, groups=mod.groups,
            )
            scale_view = lambda s: s.view(1, -1, 1, 1)
            weight_scale_view = lambda s: s.view(1, -1, 1, 1)
        elif isinstance(module, nn.Linear):
            forward = lambda mod, xi, w: torch.nn.functional.linear(
                xi, w, mod.bias,
            )
            scale_view = lambda s: s.view(1, -1)
            weight_scale_view = lambda s: s.view(1, -1)
        else:
            return float(alpha_grid[len(alpha_grid) // 2])

        # FP32 reference output once.
        with torch.no_grad():
            y_ref = forward(module, x, original_w)

        best_alpha = alpha_grid[0]
        best_mse = float("inf")

        with torch.no_grad():
            for alpha in alpha_grid:
                s = self._compute_smooth_scales(a_max, w_max, float(alpha))
                w_smoothed = original_w * weight_scale_view(s)
                w_q = self.quantize_tensor(
                    w_smoothed, bitwidth, per_channel=True,
                )
                # Compensating input divide → X' = X / s.
                x_compensated = x / scale_view(s)
                y_q = forward(module, x_compensated, w_q)
                mse = float((y_q - y_ref).pow(2).mean().item())
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = float(alpha)

        return best_alpha

    def _get_method_name(self) -> str:
        return "SmoothQuant"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Safe (no-pickle) persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# SmoothQuant inserts ``Sequential(_SmoothInputScale, original_layer)`` per
# wrapped layer, which means a state_dict-only save would lose the wrapper
# structure on reload. The previous implementation pickled the full module
# via ``torch.save(module)`` + ``torch.load(weights_only=False)`` — that
# path executes arbitrary code on a malicious checkpoint and is unsuitable
# for a deployable system.
#
# The safe replacement records, in a small JSON-serialisable metadata blob,
# (a) which submodules were wrapped, and (b) the shape + dtype of each
# wrapper's ``inv_scale`` buffer. On reload, we rebuild the wrappers with
# placeholder buffers of the right shape, then load the ``state_dict``
# (which fills in the real ``inv_scale`` values) under
# ``weights_only=True``.


def serialize_smoothquant_metadata(model: nn.Module) -> Dict[str, Any]:
    """Return a JSON-safe description of every SmoothQuant wrapper in
    ``model``.

    The returned dict has shape::

        {"wrappers": [{"name": "...", "inv_scale_shape": [...],
                       "inv_scale_dtype": "torch.float32"}, ...]}

    Every entry corresponds to a ``Sequential(_SmoothInputScale, inner)``
    whose ``inner`` lives at ``model.get_submodule(name + ".1")`` after
    wrapping. The metadata is sufficient to rebuild the wrapper layout
    on a fresh FP32 model with no Python code execution required at load
    time.
    """
    entries: List[Dict[str, Any]] = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Sequential) or len(m) != 2:
            continue
        head = m[0]
        if not isinstance(head, _SmoothInputScale):
            continue
        inv = head.inv_scale
        entries.append({
            "name": name,
            "inv_scale_shape": list(inv.shape),
            "inv_scale_dtype": str(inv.dtype),
        })
    return {"wrappers": entries}


def restore_smoothquant_wrappers(
    model: nn.Module,
    metadata: Dict[str, Any],
) -> nn.Module:
    """Rebuild SmoothQuant wrappers on ``model`` according to ``metadata``.

    Walks ``metadata["wrappers"]`` and, for each entry, replaces
    ``model.get_submodule(name)`` with a ``Sequential(_SmoothInputScale,
    original_inner)``. ``inv_scale`` is initialised with zeros — the real
    values are filled in by a subsequent
    ``model.load_state_dict(state_dict, strict=False)``.
    """
    wrappers = (metadata or {}).get("wrappers", []) or []
    for entry in wrappers:
        name = entry["name"]
        shape = tuple(int(d) for d in entry["inv_scale_shape"])
        try:
            dtype = getattr(torch, str(entry["inv_scale_dtype"]).split(".")[-1])
        except AttributeError:
            dtype = torch.float32

        # Locate inner module + its parent so we can swap it for the wrapper.
        try:
            inner = model.get_submodule(name)
        except AttributeError:
            logger.warning(
                "  [SmoothQuant restore] submodule '%s' missing from "
                "blank model; skipping wrapper.", name,
            )
            continue
        # If it's already wrapped (e.g. resume-of-resume), don't double-wrap.
        if isinstance(inner, nn.Sequential) and len(inner) == 2 \
                and isinstance(inner[0], _SmoothInputScale):
            continue

        placeholder = torch.zeros(shape, dtype=dtype)
        wrapper = nn.Sequential(_SmoothInputScale(placeholder), inner)
        # Move wrapper to the inner module's device.
        try:
            param_iter = inner.parameters()
            first = next(param_iter)
            wrapper.to(first.device)
        except StopIteration:
            pass

        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, wrapper)

    return model
