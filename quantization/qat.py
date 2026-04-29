"""
NeuroQuant v2.0 - QAT Warmstart: Fine-Tuning from Adaround (Phase 1e)

Quantization-Aware Training with warmstart from Adaround-optimised
weights. Instead of training from scratch (20-50 epochs), we fine-tune
for only 5-10 epochs since the weights are already well-rounded.

Key design decisions:
    1. Train on train_loader, evaluate on val_loader (NOT train on val_loader
       as the spec originally suggested — that would be data leakage).
    2. Fake-quantization applied to weights during forward pass via STE:
       forward = round-to-nearest, backward = straight-through (identity).
    3. Batch norm frozen: with only 5 epochs, BN stats would be noisy.
    4. Cosine annealing LR scheduler added for smooth convergence.
    5. Gradient clipping to prevent exploding gradients from quantized
       weight distributions.
    6. Early stopping based on validation accuracy plateau.

STE (Straight-Through Estimator):
    Forward: w_q = clamp(round(w/s), qmin, qmax) * s
    Backward: dL/dw = dL/dw_q  if qmin <= w/s <= qmax, else 0
    This clips gradients for out-of-range weights (prevents runaway).
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import QATResult, QuantizationConfig

logger = logging.getLogger("neuroquant")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STE Fake-Quantization (differentiable for QAT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _FakeQuantizeSTE(torch.autograd.Function):
    """
    Fake-quantization with Straight-Through Estimator.

    Forward:  w_q = clamp(round(w / scale), qmin, qmax) * scale
    Backward: gradient passes through where w/scale is in [qmin, qmax],
              zeroed where it's out of range (gradient clamping).

    This is the standard STE used in all QAT literature.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        # Save for backward gradient clamping
        w_div_scale = weight / scale
        ctx.save_for_backward(w_div_scale)
        ctx.qmin = qmin
        ctx.qmax = qmax

        # Quantize → dequantize
        w_int = torch.clamp(torch.round(w_div_scale), qmin, qmax)
        return w_int * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w_div_scale,) = ctx.saved_tensors

        # STE: pass gradient through where w/scale is in range,
        # zero gradient where it's out of range.
        mask = (w_div_scale >= ctx.qmin) & (w_div_scale <= ctx.qmax)
        grad_input = grad_output * mask.float()

        # No gradient for scale, qmin, qmax
        return grad_input, None, None, None


def _compute_scale(weight: torch.Tensor, bitwidth: int) -> torch.Tensor:
    """Compute per-tensor symmetric quantization scale."""
    qmax = 2 ** (bitwidth - 1) - 1
    abs_max = weight.detach().abs().max()
    return torch.clamp(abs_max / max(qmax, 1), min=1e-8)


def fake_quantize_weight(weight: torch.Tensor, bitwidth: int) -> torch.Tensor:
    """
    Apply STE fake-quantization to a weight tensor.

    Args:
        weight: FP32 weight tensor (requires_grad=True during QAT).
        bitwidth: Target bitwidth (4 or 8).

    Returns:
        Fake-quantized weight with STE gradients.
    """
    if bitwidth >= 32:
        return weight

    qmax = 2 ** (bitwidth - 1) - 1
    qmin = -(qmax + 1)
    scale = _compute_scale(weight, bitwidth)

    return _FakeQuantizeSTE.apply(weight, scale, qmin, qmax)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fake-Quantization Hooks (insert/remove during training)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _FakeQuantHookManager:
    """
    Manages forward pre-hooks that apply fake-quantization to weights
    during the forward pass. Hooks are registered per layer based on
    the bitwidth config and can be removed cleanly.
    """

    def __init__(self) -> None:
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._bitwidth_map: Dict[str, int] = {}

    def register(
        self,
        model: nn.Module,
        bitwidth_config: Dict[str, int],
    ) -> None:
        """
        Register forward pre-hooks on modules whose weight params
        are in the bitwidth config.

        Args:
            model: Model to attach hooks to.
            bitwidth_config: {param_name -> bitwidth}.
        """
        # Build module_name -> bitwidth mapping
        # param "3.weight" → module "3" with bitwidth
        module_bitwidths: Dict[str, int] = {}
        for pname, bw in bitwidth_config.items():
            if "weight" in pname:
                # "features.3.conv.weight" → module = "features.3.conv"
                parts = pname.rsplit(".", 1)
                module_name = parts[0] if len(parts) > 1 else pname
                module_bitwidths[module_name] = bw

        for name, module in model.named_modules():
            if name not in module_bitwidths:
                continue
            if not hasattr(module, "weight"):
                continue

            bw = module_bitwidths[name]

            # Forward pre-hook: fake-quantize weight before forward pass
            def _make_hook(bitwidth: int):
                def hook(mod, inputs):
                    mod.weight.data = fake_quantize_weight(
                        mod.weight, bitwidth
                    ).data
                return hook

            h = module.register_forward_pre_hook(_make_hook(bw))
            self._hooks.append(h)

        logger.info(
            "  Registered %d fake-quantization hooks", len(self._hooks)
        )

    def remove(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QATTrainer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class QATTrainer:
    """
    Quantization-Aware Training with warmstart from Adaround.

    Fine-tunes model weights while fake-quantization hooks simulate
    the quantization error during forward pass. STE allows gradients
    to flow through the non-differentiable rounding operations.

    Enhancements over basic QAT:
        - Cosine annealing LR scheduler for smooth convergence
        - Gradient clipping (max_norm=1.0) for stability
        - Early stopping on validation accuracy plateau
        - Best model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        bitwidth_config: Dict[str, int],
        config: QuantizationConfig,
    ) -> None:
        """
        Args:
            model: Model from Phase 1d Adaround (FP32 with optimised weights).
            bitwidth_config: {param_name -> bitwidth (4 or 8)}.
            config: Framework configuration (uses qat_* hyperparameters).
        """
        self.model = model
        self.bitwidth_config = bitwidth_config
        self.config = config
        self.device = self._resolve_device(config.hyperparams.device)

        self.model.to(self.device)

        # Hook manager for fake-quantization
        self._hook_manager = _FakeQuantHookManager()

        # Best model state
        self._best_state: Optional[Dict[str, Any]] = None
        self._best_val_acc: float = -1.0
        self._best_epoch: int = 0

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_model(self) -> None:
        """
        Prepare model for QAT:
            1. Freeze batch norm (keep statistics from pre-training)
            2. Enable gradients for weight parameters
            3. Register fake-quantization hooks
        """
        logger.info("Preparing model for QAT ...")

        # Freeze BatchNorm: set to eval mode (uses running stats)
        # and disable gradient on BN parameters
        bn_frozen = 0
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()  # Use running mean/var
                for p in module.parameters():
                    p.requires_grad_(False)
                bn_frozen += 1
        logger.info("  Frozen %d BatchNorm layers", bn_frozen)

        # Enable gradients for all non-BN parameters
        trainable = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                continue  # already trainable
            # Skip BN params (already frozen above)
            is_bn = False
            for mod_name, mod in self.model.named_modules():
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    for pn, _ in mod.named_parameters():
                        if name == f"{mod_name}.{pn}":
                            is_bn = True
                            break
                if is_bn:
                    break
            if not is_bn:
                param.requires_grad_(True)
                trainable += 1
        logger.info("  Enabled gradients for %d parameters", trainable)

        # Register fake-quantization hooks
        self._hook_manager.register(self.model, self.bitwidth_config)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> QATResult:
        """
        Run the full QAT warmstart training loop.

        Args:
            train_loader: Training data (NOT validation data — avoid leakage).
            val_loader: Validation data for accuracy measurement.
            criterion: Loss function (defaults to CrossEntropyLoss).

        Returns:
            QATResult with trained model and training curves.
        """
        hp = self.config.hyperparams
        epochs = hp.qat_epochs
        lr = hp.qat_lr
        momentum = hp.qat_momentum
        weight_decay = hp.qat_weight_decay
        patience = hp.qat_early_stop_patience

        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        logger.info("=" * 70)
        logger.info("Phase 1e: QAT Warmstart - Fine-Tuning from Adaround")
        logger.info("=" * 70)
        logger.info(
            "  Epochs: %d, LR: %.4f, Momentum: %.1f, WD: %.1e, Patience: %d",
            epochs, lr, momentum, weight_decay, patience,
        )

        t_start = time.time()

        # Prepare
        self.prepare_model()

        # Get trainable parameters (non-frozen)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info("  Trainable parameters: %d tensors (%d elements)",
                     len(trainable_params),
                     sum(p.numel() for p in trainable_params))

        # Optimizer: SGD with momentum (stable for warmstart fine-tuning)
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Cosine annealing scheduler (smooth LR decay over training)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        # Reproducibility
        torch.manual_seed(hp.seed)

        # Training history
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        no_improve_count = 0

        # ── Training loop ──
        for epoch in range(1, epochs + 1):
            # ---- Train ----
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            # ---- Validate ----
            val_acc = self._validate(val_loader)
            history["val_accuracy"].append(val_acc)

            # ---- LR step ----
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            # ---- Best model checkpoint ----
            is_best = val_acc > self._best_val_acc
            if is_best:
                self._best_val_acc = val_acc
                self._best_epoch = epoch
                self._best_state = copy.deepcopy(self.model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            # ---- Log ----
            best_marker = " [*best*]" if is_best else ""
            logger.info(
                "  Epoch %d/%d: train_loss=%.4f, train_acc=%.2f%%, "
                "val_acc=%.2f%%, lr=%.6f%s",
                epoch, epochs, train_loss, train_acc, val_acc,
                current_lr, best_marker,
            )

            # ---- Early stopping ----
            if no_improve_count >= patience and epoch >= 3:
                logger.info(
                    "  Early stopping: no improvement for %d epochs", patience
                )
                break

        # ── Restore best model ──
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            logger.info("  Restored best model from epoch %d", self._best_epoch)

        # ── Clean up hooks ──
        self._hook_manager.remove()

        t_elapsed = time.time() - t_start

        # ── Log summary ──
        logger.info("-" * 70)
        logger.info("QAT Results:")
        logger.info("  Best epoch: %d (val_acc=%.2f%%)", self._best_epoch,
                     self._best_val_acc)
        logger.info("  Final train loss: %.4f",
                     history["train_loss"][-1] if history["train_loss"] else 0)
        logger.info("  Time: %.1f seconds", t_elapsed)
        logger.info("=" * 70)

        return QATResult(
            model=self.model,
            train_accuracy=history["train_accuracy"],
            val_accuracy=history["val_accuracy"],
            train_loss=history["train_loss"],
            best_epoch=self._best_epoch,
            final_val_acc=self._best_val_acc,
            time_seconds=t_elapsed,
        )

    # ------------------------------------------------------------------
    # Single-epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            (avg_loss, accuracy_percent)
        """
        self.model.train()

        # Re-freeze BatchNorm (model.train() would un-eval them)
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward (fake-quant hooks active)
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents explosion from quantized weights)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )

            optimizer.step()

            # Track stats
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / max(total, 1)
        accuracy = (correct / max(total, 1)) * 100.0
        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> float:
        """
        Evaluate model accuracy on validation set.

        Returns:
            accuracy_percent
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return (correct / max(total, 1)) * 100.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
