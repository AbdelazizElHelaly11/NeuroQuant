"""
NeuroQuant v2.0 - Adaround: Learned Weight Rounding (Phase 1d)

Instead of rounding each weight to the nearest quantization level,
Adaround learns whether to round each weight UP or DOWN to minimise
the per-layer **output** reconstruction error on calibration data.

Correct formulation (Nagel et al., ICLR 2021):
    For each weight w_i, compute floor(w_i/scale) = z_floor.
    The quantized value is either z_floor or z_floor + 1.
    Learn a continuous variable V_i ∈ R that maps to h(V) ∈ [0, 1]
    via a stretched sigmoid: h(V) = clamp(sigmoid(V) * 1.2 - 0.1, 0, 1).
    The soft quantized weight is:
        w_q = (z_floor + h(V)) * scale

    For a quantizable layer (Conv2d / Linear) with collected calibration
    input X:
        Loss_recon = || layer(X; w_q) - layer(X; w) ||²
        Loss_round = h * (1 - h)            # push h toward {0, 1}
        Loss = Loss_recon + λ · Loss_round

    The reconstruction loss is what makes Adaround non-trivial: a
    pure weight-MSE objective is uniquely minimised by ``h = frac``,
    which collapses to round-to-nearest after hard rounding. Driving
    the *layer output* error instead lets h trade weight-MSE against
    activation-correlated rounding directions.

Key insight: h(V) replaces the hard round() with a differentiable
function, so gradients always flow correctly. No STE needed.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import AdaroundResult, QuantizationConfig

logger = logging.getLogger("neuroquant")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Differentiable rounding via stretched sigmoid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _stretched_sigmoid(v: torch.Tensor, stretch: float = 6.0) -> torch.Tensor:
    """
    Stretched sigmoid mapping V → h(V) ∈ [0, 1].

    h(V) = clamp(sigmoid(V / stretch) * (1 + 2δ) - δ, 0, 1)

    where δ = 0.1 is a small margin that allows h to reach exactly 0 and 1.
    The stretch parameter controls how "sharp" the sigmoid is.

    Args:
        v: Learnable rounding variable (unconstrained).
        stretch: Temperature parameter (larger = smoother).

    Returns:
        h(V) in [0, 1], differentiable w.r.t. v.
    """
    delta = 0.1
    sig = torch.sigmoid(v / stretch)
    return torch.clamp(sig * (1.0 + 2.0 * delta) - delta, 0.0, 1.0)


def _rounding_regularizer(h: torch.Tensor) -> torch.Tensor:
    """
    Regularizer that pushes h(V) toward 0 or 1 (binary decisions).

    R(h) = sum( h * (1 - h) )

    Zeros at h=0 and h=1, maximum at h=0.5 (ambiguous).
    Minimising this prevents h from sitting at intermediate values.

    Args:
        h: Rounding probabilities in [0, 1].

    Returns:
        Scalar regularization loss.
    """
    return (h * (1.0 - h)).mean()


def _compute_quant_params(
    weight: torch.Tensor, bitwidth: int
) -> Tuple[torch.Tensor, int, int]:
    """
    Compute symmetric quantization parameters for a weight tensor.

    Args:
        weight: FP32 weight tensor.
        bitwidth: Target bitwidth.

    Returns:
        (scale, qmin, qmax)
    """
    qmax = 2 ** (bitwidth - 1) - 1
    qmin = -(qmax + 1)
    abs_max = weight.abs().max()
    scale = torch.clamp(abs_max / qmax, min=1e-8)
    return scale, qmin, qmax


def _fake_quantize(tensor: torch.Tensor, bitwidth: int) -> torch.Tensor:
    """
    Standard symmetric fake-quantization (no gradient tricks).
    Used for MSE measurement, NOT for Adaround training.
    """
    if bitwidth >= 32:
        return tensor
    scale, qmin, qmax = _compute_quant_params(tensor, bitwidth)
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return quantized * scale


def _compute_mse(
    model: nn.Module,
    original_weights: Dict[str, torch.Tensor],
    bitwidth_config: Dict[str, int],
) -> float:
    """Compute total MSE between fake-quantized weights and originals."""
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for name in bitwidth_config:
            if name not in original_weights:
                continue
            bw = bitwidth_config[name]
            orig = original_weights[name]
            w_q = _fake_quantize(orig, bw)
            total_mse += (w_q - orig).pow(2).mean().item()
            count += 1
    return total_mse / max(count, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AdaroundOptimizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AdaroundOptimizer:
    """
    Learns optimal rounding direction for each weight element.

    For each weight w_i, instead of round(w_i/scale), we learn whether
    to use floor(w_i/scale) or ceil(w_i/scale). The decision variable
    V_i maps to h(V_i) ∈ [0, 1] via a stretched sigmoid, giving:

        w_q_i = (floor(w_i/scale) + h(V_i)) * scale

    Trained against per-layer **output reconstruction**:
        Loss_recon(layer) = || layer(X; w_q) - layer(X; w) ||²
    where X is calibration input collected via forward hooks. A small
    rounding regulariser pushes h toward {0, 1} so the final hard
    rounding incurs no extra distortion.
    """

    def __init__(
        self,
        model: nn.Module,
        bitwidth_config: Dict[str, int],
        config: QuantizationConfig,
        calib_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Args:
            model: Model with FP32 weights (will be modified in-place).
            bitwidth_config: {param_name -> bitwidth (4 or 8)}.
            config: Framework configuration (uses adaround_* hyperparameters).
            calib_loader: DataLoader used to collect per-layer activations
                for the layer-output reconstruction objective. Required
                for the strong objective; if ``None`` Adaround degrades
                to the legacy weight-MSE objective with a clear warning.
        """
        self.model = model
        self.bitwidth_config = bitwidth_config
        self.config = config
        self.calib_loader = calib_loader
        self.device = self._resolve_device(config.hyperparams.device)

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Learnable variables V (unconstrained), keyed by param name
        self._v_params: Dict[str, nn.Parameter] = {}

        # Pre-computed quantization parameters (frozen)
        self._quant_scales: Dict[str, torch.Tensor] = {}
        self._quant_floors: Dict[str, torch.Tensor] = {}

        # Original FP32 weights
        self._original_weights: Dict[str, torch.Tensor] = {}

        # Target param names (only quantized weight params) and their
        # owning module. Prioritise low-bit tensors (INT4) where learned
        # rounding matters most; if none exist, fall back to all quantized
        # weights (<32-bit).
        quantized_weights = [
            name for name in bitwidth_config
            if "weight" in name and int(bitwidth_config[name]) < 32
        ]
        low_bit_weights = [
            name for name in quantized_weights
            if int(bitwidth_config[name]) < 8
        ]
        self._target_params: List[str] = (
            low_bit_weights if low_bit_weights else quantized_weights
        )
        self._owner_modules: Dict[str, nn.Module] = {}
        # Cached calibration inputs per layer (filled by collect_activations).
        self._layer_inputs: Dict[str, torch.Tensor] = {}
        self._objective_components: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Step 1: Initialize
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Create learnable V parameters and pre-compute quantisation floors.

        For each weight w_i:
            scale = max(|w|) / qmax
            z_floor = floor(w / scale)
            fractional_part = w/scale - z_floor  (in [0, 1])
            V_init such that sigmoid(V/stretch) ≈ fractional_part
                → initialisation near round-to-nearest
        """
        logger.info("Initializing %d rounding variables ...", len(self._target_params))

        for name, param in self.model.named_parameters():
            if name not in self._target_params:
                continue

            w = param.data.clone()
            bw = self.bitwidth_config[name]

            # Quantization parameters
            scale, qmin, qmax = _compute_quant_params(w, bw)
            self._quant_scales[name] = scale

            # Floor of w/scale
            w_div_scale = w / scale
            z_floor = w_div_scale.floor()
            z_floor = torch.clamp(z_floor, qmin, qmax - 1)  # leave room for +1
            self._quant_floors[name] = z_floor

            # Store originals
            self._original_weights[name] = w

            # Initialize V so h(V) ≈ frac(w/scale)
            # frac = w/scale - floor(w/scale), in [0, 1]
            frac = torch.clamp(w_div_scale - z_floor, 0.0, 1.0)
            # If frac ≈ 0.5, round-to-nearest would pick 1 (>=0.5) or 0 (<0.5).
            # We initialise V so h(V) = frac, then let optimisation improve it.
            # inverse of stretched sigmoid: V = stretch * logit((frac + delta) / (1 + 2*delta))
            delta = 0.1
            stretch = 6.0
            p_init = torch.clamp(frac, 0.01, 0.99)  # avoid log(0)
            adjusted = (p_init + delta) / (1.0 + 2.0 * delta)
            adjusted = torch.clamp(adjusted, 0.01, 0.99)
            v_init = stretch * torch.log(adjusted / (1.0 - adjusted))

            v_param = nn.Parameter(v_init.to(self.device), requires_grad=True)
            self._v_params[name] = v_param

        logger.info(
            "  Created %d V tensors (total elements: %d)",
            len(self._v_params),
            sum(v.numel() for v in self._v_params.values()),
        )

    # ------------------------------------------------------------------
    # Step 1.5: Collect calibration activations per layer
    # ------------------------------------------------------------------

    def _resolve_owner_modules(self) -> None:
        """Map each target weight parameter to its owning Conv2d/Linear.

        The owner name is the parameter name with the trailing ``.weight``
        stripped — generic across architectures because we only rely on
        the standard PyTorch parameter naming convention.
        """
        name_to_module = dict(self.model.named_modules())
        for pname in self._target_params:
            owner_name = pname.rsplit(".", 1)[0]
            module = name_to_module.get(owner_name)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._owner_modules[pname] = module

    def collect_activations(
        self,
        max_batches: Optional[int] = None,
        max_samples_per_layer: int = 1024,
    ) -> None:
        """Collect a bounded pool of calibration inputs for each target layer.

        Inputs are gathered via forward hooks on the owner module of each
        target weight. The pool is capped per-layer so memory stays
        constant regardless of dataset size.
        """
        if self.calib_loader is None:
            return
        if not self._owner_modules:
            self._resolve_owner_modules()
        if not self._owner_modules:
            logger.warning(
                "Adaround: no Conv2d/Linear owners found for the bitwidth "
                "config; activation reconstruction will be skipped."
            )
            return

        hp = self.config.hyperparams
        n_batches = int(max_batches if max_batches is not None
                        else hp.calibration_batches)

        # Buffers keyed by parameter name (so multiple weights inside the
        # same module are still keyed independently downstream).
        buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in self._owner_modules}
        taken: Dict[str, int] = {n: 0 for n in self._owner_modules}

        # Reverse index: module → list of param names that share this module.
        module_to_params: Dict[int, List[str]] = {}
        for pname, m in self._owner_modules.items():
            module_to_params.setdefault(id(m), []).append(pname)

        hooks = []
        for pname, module in self._owner_modules.items():
            param_names = module_to_params[id(module)]

            def _hook_factory(names: List[str]):
                def _hook(_mod, inputs, _output):
                    if not inputs:
                        return
                    x = inputs[0].detach()
                    for n in names:
                        if taken[n] >= max_samples_per_layer:
                            continue
                        flat = x[: max_samples_per_layer - taken[n]]
                        buffers[n].append(flat.cpu())
                        taken[n] += flat.shape[0]
                return _hook

            hooks.append(module.register_forward_hook(_hook_factory(param_names)))

        try:
            self.model.eval()
            self.model.to(self.device)
            with torch.no_grad():
                for i, batch in enumerate(self.calib_loader):
                    if i >= n_batches:
                        break
                    images = batch[0].to(self.device)
                    self.model(images)
        finally:
            for h in hooks:
                h.remove()

        for pname, chunks in buffers.items():
            if chunks:
                self._layer_inputs[pname] = torch.cat(chunks, dim=0)

        logger.info(
            "Adaround: collected calibration inputs for %d/%d target layers",
            len(self._layer_inputs), len(self._target_params),
        )

    # ------------------------------------------------------------------
    # Step 2: Optimize
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_forward(
        module: nn.Module, x: torch.Tensor, weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single Conv2d/Linear forward with an externally-supplied
        weight tensor (so gradients flow into V via w_q_soft = f(V))."""
        if isinstance(module, nn.Conv2d):
            return torch.nn.functional.conv2d(
                x, weight, bias=module.bias,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
            )
        if isinstance(module, nn.Linear):
            return torch.nn.functional.linear(x, weight, module.bias)
        raise TypeError(f"Unsupported module type for Adaround: {type(module)!r}")

    def optimize(
        self,
        num_epochs: Optional[int] = None,
        lr: Optional[float] = None,
        lambda_reg: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """
        Train V parameters to minimise per-layer output reconstruction
        plus the rounding regulariser.

        Loss(layer) = || layer(X; w_q) - layer(X; w) ||² + λ · R(h(V))

        The reconstruction term is what makes Adaround non-trivial:
        a pure weight-MSE term degenerates to round-to-nearest because
        ``h = frac`` is the unique optimum.

        Returns:
            Training history dict with epoch-aggregated components.
        """
        epochs = num_epochs or self.config.hyperparams.adaround_epochs
        learning_rate = lr or self.config.hyperparams.adaround_lr
        lam = lambda_reg or self.config.hyperparams.adaround_reg_param

        if not self._v_params:
            logger.warning("No V parameters initialized. Call initialize() first.")
            return {"epoch_losses": [], "recon_losses": [],
                    "weight_mse_losses": [], "reg_losses": []}

        # Make sure activations are available; if calib_loader was supplied
        # but collect_activations() has not been called yet, do it now.
        if self.calib_loader is not None and not self._layer_inputs:
            self.collect_activations()

        use_recon = bool(self._layer_inputs)
        if not use_recon:
            logger.warning(
                "Adaround: no calibration activations available; falling "
                "back to weight-MSE objective. Pass a calib_loader to "
                "AdaroundOptimizer for the proper output-reconstruction "
                "objective."
            )

        optimizer = torch.optim.Adam(
            list(self._v_params.values()), lr=learning_rate,
        )

        history: Dict[str, List[float]] = {
            "epoch_losses": [],
            "recon_losses": [],
            "weight_mse_losses": [],
            "reg_losses": [],
        }

        logger.info(
            "Training V (%d epochs, lr=%.6f, lambda=%.4f, objective=%s) ...",
            epochs, learning_rate, lam,
            "layer_output_reconstruction" if use_recon else "weight_mse",
        )

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_w_mse = 0.0
            epoch_reg = 0.0

            optimizer.zero_grad()

            for name in self._target_params:
                if name not in self._v_params:
                    continue

                v = self._v_params[name]
                z_floor = self._quant_floors[name]
                scale = self._quant_scales[name]
                original_w = self._original_weights[name]
                bw = self.bitwidth_config[name]
                qmax = 2 ** (bw - 1) - 1
                qmin = -(qmax + 1)

                # Differentiable rounding: h(V) ∈ [0, 1]
                h = _stretched_sigmoid(v)
                z_soft = z_floor + h
                z_clamped = torch.clamp(z_soft, qmin, qmax)
                w_q_soft = z_clamped * scale

                # Always track weight MSE for diagnostics — but only use
                # it as the objective when no activations are available.
                weight_mse = (w_q_soft - original_w).pow(2).mean()

                if use_recon and name in self._layer_inputs:
                    module = self._owner_modules[name]
                    x = self._layer_inputs[name].to(self.device)
                    with torch.no_grad():
                        y_ref = self._layer_forward(module, x, original_w)
                    y_q = self._layer_forward(module, x, w_q_soft)
                    loss_recon = (y_q - y_ref).pow(2).mean()
                    main_loss = loss_recon
                else:
                    loss_recon = weight_mse.detach()
                    main_loss = weight_mse

                loss_reg = _rounding_regularizer(h)
                loss = main_loss + lam * loss_reg

                epoch_loss += loss.item()
                epoch_recon += float(loss_recon.item())
                epoch_w_mse += float(weight_mse.item())
                epoch_reg += float(loss_reg.item())

                loss.backward()

            optimizer.step()

            n_params = max(len(self._target_params), 1)
            history["epoch_losses"].append(epoch_loss / n_params)
            history["recon_losses"].append(epoch_recon / n_params)
            history["weight_mse_losses"].append(epoch_w_mse / n_params)
            history["reg_losses"].append(epoch_reg / n_params)

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                logger.info(
                    "  Epoch %d/%d: total=%.6f recon=%.6f w_mse=%.6f reg=%.6f",
                    epoch, epochs,
                    history["epoch_losses"][-1],
                    history["recon_losses"][-1],
                    history["weight_mse_losses"][-1],
                    history["reg_losses"][-1],
                )

        # Stash the final epoch components for the AdaroundResult report.
        if history["epoch_losses"]:
            self._objective_components = {
                "final_total": history["epoch_losses"][-1],
                "final_recon": history["recon_losses"][-1],
                "final_weight_mse": history["weight_mse_losses"][-1],
                "final_reg": history["reg_losses"][-1],
                "objective": (
                    "layer_output_reconstruction" if use_recon
                    else "weight_mse_fallback"
                ),
                "epochs": int(epochs),
                "lambda_reg": float(lam),
            }

        logger.info(
            "  Training complete. Final loss: %.6f",
            history["epoch_losses"][-1] if history["epoch_losses"] else 0,
        )

        return history

    # ------------------------------------------------------------------
    # Step 3: Apply
    # ------------------------------------------------------------------

    def apply(self) -> nn.Module:
        """
        Apply learned rounding decisions to model weights.

        h(V) is rounded to 0 or 1 (hard decision), and the final
        quantized weight is: w_q = (z_floor + round(h(V))) * scale.

        Updates model weights in-place.
        """
        logger.info("Applying learned rounding to %d parameters ...", len(self._v_params))

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self._v_params:
                    continue

                v = self._v_params[name]
                z_floor = self._quant_floors[name]
                scale = self._quant_scales[name]
                bw = self.bitwidth_config[name]
                qmax = 2 ** (bw - 1) - 1
                qmin = -(qmax + 1)

                # Hard decision: round h(V) to 0 or 1
                h = _stretched_sigmoid(v)
                h_hard = h.round()  # 0 = floor, 1 = ceil

                z_final = torch.clamp(z_floor + h_hard, qmin, qmax)
                param.data.copy_(z_final * scale)

        logger.info("  Weights updated with learned rounding.")
        return self.model

    # ------------------------------------------------------------------
    # Step 4: Statistics
    # ------------------------------------------------------------------

    def compute_alpha_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-parameter statistics about the learned rounding.

        Returns h(V) statistics showing how many weights round up vs down.
        """
        stats: Dict[str, Dict[str, float]] = {}

        for name, v in self._v_params.items():
            with torch.no_grad():
                h = _stretched_sigmoid(v)
            n = h.numel()

            stats[name] = {
                "mean": h.mean().item(),
                "min": h.min().item(),
                "max": h.max().item(),
                "std": h.std().item(),
                "n_near_zero": int((h < 0.1).sum().item()),   # round down
                "n_near_half": int(((h > 0.4) & (h < 0.6)).sum().item()),  # undecided
                "n_near_one": int((h > 0.9).sum().item()),    # round up
                "n_total": n,
            }

        return stats

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run(self) -> AdaroundResult:
        """Execute the full Adaround pipeline."""
        logger.info("=" * 70)
        logger.info("Phase 1d: Adaround - Learned Weight Rounding Optimisation")
        logger.info("=" * 70)

        t_start = time.time()

        # Resolve owner modules first so we can compute the layer-output
        # reconstruction baseline before optimisation modifies any weight.
        self._resolve_owner_modules()

        # Snapshot the FP32 weights NOW so the recon-before call below
        # can read them. ``initialize()`` will populate the same dict
        # again later — that's fine, the values are identical.
        self._original_weights = self._get_original_weights_from_model()

        # MSE before (baseline: round-to-nearest, weight-space)
        mse_before = _compute_mse(
            self.model, self._original_weights, self.bitwidth_config,
        )
        logger.info("  MSE before Adaround (round-to-nearest): %.8f", mse_before)

        # Collect activations once and reuse them for both training and
        # the output-reconstruction sanity metrics.
        self.collect_activations()

        recon_before = self._compute_output_reconstruction(
            use_round_to_nearest=True,
        )
        if recon_before is not None:
            logger.info(
                "  Layer-output recon error before (round-to-nearest): %.8f",
                recon_before,
            )

        # Step 1-3
        self.initialize()
        self.optimize()
        self.apply()

        # MSE after (with learned rounding) — weight-space
        mse_after_weights: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if name in self.bitwidth_config and "weight" in name:
                mse_after_weights[name] = self._original_weights[name]
        mse_after = 0.0
        count = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in mse_after_weights:
                    continue
                orig = mse_after_weights[name]
                # Model weights are already quantized via apply()
                mse_after += (param.data - orig).pow(2).mean().item()
                count += 1
        mse_after = mse_after / max(count, 1)

        logger.info("  MSE after Adaround (learned rounding): %.8f", mse_after)

        # Output-reconstruction error after — using the *current* model
        # weights (which apply() already wrote in place).
        recon_after = self._compute_output_reconstruction(
            use_round_to_nearest=False,
        )
        if recon_after is not None:
            logger.info(
                "  Layer-output recon error after (Adaround):           %.8f",
                recon_after,
            )

        mse_reduction = (
            (mse_before - mse_after) / mse_before * 100.0 if mse_before > 0 else 0.0
        )
        recon_reduction = None
        if recon_before and recon_after is not None and recon_before > 0:
            recon_reduction = (recon_before - recon_after) / recon_before * 100.0

        alpha_stats = self.compute_alpha_stats()
        t_elapsed = time.time() - t_start

        logger.info("-" * 70)
        logger.info("Adaround Results:")
        logger.info("  Weight-MSE reduction: %.1f%% (%.8f -> %.8f)",
                     mse_reduction, mse_before, mse_after)
        if recon_reduction is not None:
            logger.info(
                "  Output-recon reduction: %.1f%% (%.8f -> %.8f)",
                recon_reduction, recon_before, recon_after,
            )
        logger.info("  Time: %.1f seconds", t_elapsed)
        logger.info("  Rounding stats (sample):")
        for name, s in list(alpha_stats.items())[:3]:
            logger.info(
                "    %s: h_mean=%.3f, round_down=%d, round_up=%d, "
                "undecided=%d, total=%d",
                name, s["mean"], s["n_near_zero"], s.get("n_near_one", 0),
                s["n_near_half"], s["n_total"],
            )
        logger.info("=" * 70)

        result = AdaroundResult(
            model=self.model,
            alpha_stats=alpha_stats,
            mse_before=mse_before,
            mse_after=mse_after,
            mse_reduction=mse_reduction,
            time_seconds=t_elapsed,
        )
        # Attach the activation-reconstruction diagnostics + objective
        # components. AdaroundResult is a TypedDict — extra keys are
        # allowed at runtime and surfaced in the phase-1d checkpoint.
        result["objective_components"] = self._objective_components  # type: ignore[typeddict-item]
        result["recon_before"] = recon_before  # type: ignore[typeddict-item]
        result["recon_after"] = recon_after  # type: ignore[typeddict-item]
        result["recon_reduction"] = recon_reduction  # type: ignore[typeddict-item]
        return result

    def _compute_output_reconstruction(
        self, use_round_to_nearest: bool,
    ) -> Optional[float]:
        """Mean per-layer ``||layer(X; w_q) - layer(X; w)||²`` across targets.

        Returns ``None`` if no calibration inputs were captured (which
        happens when the caller did not supply a ``calib_loader``). When
        ``use_round_to_nearest`` is True, ``w_q`` is the round-to-nearest
        fake-quantized weight; otherwise the *current* (post-Adaround)
        param tensor in the model is used.
        """
        if not self._layer_inputs or not self._owner_modules:
            return None
        total = 0.0
        count = 0
        with torch.no_grad():
            for pname in self._target_params:
                if pname not in self._layer_inputs:
                    continue
                module = self._owner_modules[pname]
                x = self._layer_inputs[pname].to(self.device)
                w_orig = self._original_weights[pname]
                if use_round_to_nearest:
                    bw = self.bitwidth_config[pname]
                    w_q = _fake_quantize(w_orig, bw)
                else:
                    # Pull the current weight from the model (post-apply).
                    w_q = dict(module.named_parameters())["weight"].data
                y_ref = self._layer_forward(module, x, w_orig)
                y_q = self._layer_forward(module, x, w_q)
                total += float((y_q - y_ref).pow(2).mean().item())
                count += 1
        return total / max(count, 1) if count else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_original_weights_from_model(self) -> Dict[str, torch.Tensor]:
        """Snapshot current model weights as the pre-Adaround baseline."""
        result: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if name in self.bitwidth_config and "weight" in name:
                result[name] = param.data.clone()
        return result

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)
