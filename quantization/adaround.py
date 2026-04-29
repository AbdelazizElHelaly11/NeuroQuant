"""
NeuroQuant v2.0 - Adaround: Learned Weight Rounding (Phase 1d)

Instead of rounding each weight to the nearest quantization level,
Adaround learns whether to round each weight UP or DOWN, minimising
per-layer quantization MSE.

Correct formulation (Nagel et al., ICLR 2021):
    For each weight w_i, compute floor(w_i/scale) = z_floor.
    The quantized value is either z_floor or z_floor + 1.
    Learn a continuous variable V_i ∈ R that maps to h(V) ∈ [0, 1]
    via a stretched sigmoid: h(V) = clamp(sigmoid(V) * 1.2 - 0.1, 0, 1).
    The soft quantized weight is:
        w_q = (z_floor + h(V)) * scale
    MSE loss: ||w_q - w||^2
    Regularizer: pushes h(V) to 0 or 1 (binary decision).

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

    At convergence, h(V) ≈ 0 (round down) or h(V) ≈ 1 (round up).
    The loss MSE(w_q, w) + λR(h) is fully differentiable.
    """

    def __init__(
        self,
        model: nn.Module,
        bitwidth_config: Dict[str, int],
        config: QuantizationConfig,
    ) -> None:
        """
        Args:
            model: Model with FP32 weights (will be modified in-place).
            bitwidth_config: {param_name -> bitwidth (4 or 8)}.
            config: Framework configuration (uses adaround_* hyperparameters).
        """
        self.model = model
        self.bitwidth_config = bitwidth_config
        self.config = config
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

        # Target param names (only weight params)
        self._target_params: List[str] = [
            name for name in bitwidth_config if "weight" in name
        ]

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
    # Step 2: Optimize
    # ------------------------------------------------------------------

    def optimize(
        self,
        num_epochs: Optional[int] = None,
        lr: Optional[float] = None,
        lambda_reg: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """
        Train V parameters to minimise MSE + regularization.

        Loss = MSE(w_q_soft, w_original) + λ * R(h(V))

        where w_q_soft = (z_floor + h(V)) * scale.

        Returns:
            Training history dict.
        """
        epochs = num_epochs or self.config.hyperparams.adaround_epochs
        learning_rate = lr or self.config.hyperparams.adaround_lr
        lam = lambda_reg or self.config.hyperparams.adaround_reg_param

        if not self._v_params:
            logger.warning("No V parameters initialized. Call initialize() first.")
            return {"epoch_losses": [], "mse_losses": [], "reg_losses": []}

        optimizer = torch.optim.Adam(list(self._v_params.values()), lr=learning_rate)

        history: Dict[str, List[float]] = {
            "epoch_losses": [],
            "mse_losses": [],
            "reg_losses": [],
        }

        logger.info(
            "Training V (%d epochs, lr=%.6f, lambda=%.4f) ...",
            epochs, learning_rate, lam,
        )

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_mse = 0.0
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

                # Soft quantized weight
                z_soft = z_floor + h
                z_clamped = torch.clamp(z_soft, qmin, qmax)
                w_q_soft = z_clamped * scale

                # MSE loss
                loss_mse = (w_q_soft - original_w).pow(2).mean()

                # Regularizer: push h toward 0 or 1
                loss_reg = _rounding_regularizer(h)

                loss = loss_mse + lam * loss_reg

                epoch_loss += loss.item()
                epoch_mse += loss_mse.item()
                epoch_reg += loss_reg.item()

                loss.backward()

            optimizer.step()

            n_params = max(len(self._target_params), 1)
            history["epoch_losses"].append(epoch_loss / n_params)
            history["mse_losses"].append(epoch_mse / n_params)
            history["reg_losses"].append(epoch_reg / n_params)

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                logger.info(
                    "  Epoch %d/%d: loss=%.6f (mse=%.6f, reg=%.6f)",
                    epoch, epochs,
                    history["epoch_losses"][-1],
                    history["mse_losses"][-1],
                    history["reg_losses"][-1],
                )

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

        # MSE before (baseline: round-to-nearest)
        mse_before = _compute_mse(
            self.model, self._get_original_weights_from_model(),
            self.bitwidth_config,
        )
        logger.info("  MSE before Adaround (round-to-nearest): %.8f", mse_before)

        # Step 1-3
        self.initialize()
        self.optimize()
        self.apply()

        # MSE after (with learned rounding)
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

        mse_reduction = (
            (mse_before - mse_after) / mse_before * 100.0 if mse_before > 0 else 0.0
        )

        alpha_stats = self.compute_alpha_stats()
        t_elapsed = time.time() - t_start

        logger.info("-" * 70)
        logger.info("Adaround Results:")
        logger.info("  MSE reduction: %.1f%% (%.8f -> %.8f)",
                     mse_reduction, mse_before, mse_after)
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

        return AdaroundResult(
            model=self.model,
            alpha_stats=alpha_stats,
            mse_before=mse_before,
            mse_after=mse_after,
            mse_reduction=mse_reduction,
            time_seconds=t_elapsed,
        )

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
