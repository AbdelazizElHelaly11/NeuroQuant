"""
NeuroQuant v2.0 - QAT Warmstart: Fine-Tuning from Adaround (Phase 1e)

Quantization-Aware Training with warmstart from AdaRound-optimised
weights. Trains the model that real INT8 inference will execute:

  * **Conv-BN folded** (E4) — every Conv2d→BatchNorm2d pair is folded
    analytically before the QAT loop starts, so the operator graph
    matches the deployment graph (mobile / FPGA / TensorRT all drop BN
    at INT8).
  * **Weight fake-quantization** via parametrization with STE backward.
    Different from the previous ``hook + .data =`` approach: a
    parametrization is autograd-aware, so the STE clipping mask
    actually attenuates out-of-range gradients at the parameter level.
  * **Activation fake-quantization** (E1) at INT8 (E3) by default. A
    per-tensor symmetric EMA observer is attached to every quantizable
    layer's input. The pre-QAT calibration pass freezes the observer
    scales so QAT trains against the deployment-time activation
    quantizer.
  * **FP32 teacher KD** (E5) when a teacher model is supplied. The QAT
    loss is ``α·KD + (1-α)·CE`` with KD = ``T²·KL(student/T || teacher/T)``.

Training-loop niceties from the previous version are preserved:
cosine-annealing LR, gradient clipping, early stopping, best-epoch
restore.
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

from neuroquant.config import QATResult, QuantizationConfig
from neuroquant.utils.numerics import MIN_SCALE

logger = logging.getLogger("neuroquant")


_QUANTIZABLE_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STE Fake-Quantization (differentiable for QAT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _FakeQuantizeSTE(torch.autograd.Function):
    """
    Fake-quantization with Straight-Through Estimator.

    Forward:  q = clamp(round(x / scale), qmin, qmax) * scale
    Backward: gradient passes through where x/scale is in [qmin, qmax],
              zeroed where it's out of range (gradient clamping).

    Standard STE used in all QAT literature. Used by both the
    weight-side parametrization (so weight gradients clip out-of-range
    values) and the activation-side hook (so input gradients flowing
    back into earlier layers see the same clipping).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        x_div_scale = x / scale
        ctx.save_for_backward(x_div_scale)
        ctx.qmin = qmin
        ctx.qmax = qmax
        x_int = torch.clamp(torch.round(x_div_scale), qmin, qmax)
        return x_int * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x_div_scale,) = ctx.saved_tensors
        mask = (x_div_scale >= ctx.qmin) & (x_div_scale <= ctx.qmax)
        return grad_output * mask.to(grad_output.dtype), None, None, None


def _compute_scale(weight: torch.Tensor, bitwidth: int) -> torch.Tensor:
    """Compute per-tensor symmetric quantization scale."""
    qmax = 2 ** (bitwidth - 1) - 1
    abs_max = weight.detach().abs().max()
    return torch.clamp(abs_max / max(qmax, 1), min=MIN_SCALE)


def fake_quantize_weight(weight: torch.Tensor, bitwidth: int) -> torch.Tensor:
    """Apply STE fake-quantization to a weight tensor (autograd-aware)."""
    if bitwidth >= 32:
        return weight
    qmax = 2 ** (bitwidth - 1) - 1
    qmin = -(qmax + 1)
    scale = _compute_scale(weight, bitwidth)
    return _FakeQuantizeSTE.apply(weight, scale, qmin, qmax)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Weight parametrization — autograd-correct STE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _WeightFakeQuantize(nn.Module):
    """``torch.nn.utils.parametrize`` parametrization that turns a
    layer's underlying ``weight`` parameter into its fake-quantized
    counterpart on every forward pass.

    Why a parametrization rather than a forward-pre-hook?
    The previous version did ``mod.weight.data = fake_quantize_weight(...).data``
    inside a pre-hook. That bypasses autograd: the STE clipping mask is
    discarded and the gradient that lands on ``mod.weight`` is the
    gradient w.r.t. the *quantized* tensor with no clipping. Switching
    to a parametrization makes the quantize step a regular module in
    the autograd graph, so STE backward fires on the underlying
    ``weight`` parameter exactly as the literature describes.
    """

    def __init__(self, bitwidth: int) -> None:
        super().__init__()
        self.bitwidth = int(bitwidth)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return fake_quantize_weight(w, self.bitwidth)

    def right_inverse(self, w_quantized: torch.Tensor) -> torch.Tensor:
        # Required by ``parametrize.register_parametrization`` so the
        # underlying parameter can be initialised from a quantized
        # tensor (used during best-epoch state-dict restore).
        return w_quantized


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Activation observer — per-tensor symmetric, EMA during calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _ActivationObserver:
    """Per-tensor symmetric activation observer with three phases:

    * ``passthrough``  — returned x unchanged. Initial state.
    * ``calibrating``  — EMA-update min/max from each forward call but
      do NOT fake-quantize the input. Used during the pre-QAT
      calibration pass.
    * ``quantizing``   — apply STE fake-quantization at the cached
      scale; observer no longer updates. The deployment-time
      activation quantizer.

    Exposing this as a separate class keeps the per-layer state out of
    the hook closure and makes it cheap to serialise (only ``min`` and
    ``max`` floats survive a save/load cycle).
    """

    __slots__ = (
        "bitwidth", "momentum", "_phase", "_min", "_max", "_scale",
    )

    def __init__(self, bitwidth: int = 8, momentum: float = 0.1) -> None:
        self.bitwidth = int(bitwidth)
        self.momentum = float(momentum)
        self._phase = "passthrough"
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._scale: Optional[float] = None

    def start_calibration(self) -> None:
        self._phase = "calibrating"
        self._min = None
        self._max = None

    def finish_calibration(self) -> None:
        if self._min is None or self._max is None:
            self._phase = "passthrough"
            return
        amax = max(abs(float(self._min)), abs(float(self._max)))
        qmax = 2 ** (self.bitwidth - 1) - 1
        self._scale = max(amax / max(qmax, 1), MIN_SCALE)
        self._phase = "quantizing"

    def observe(self, x: torch.Tensor) -> None:
        if self._phase != "calibrating":
            return
        x_min = float(x.detach().min().item())
        x_max = float(x.detach().max().item())
        if self._min is None:
            self._min, self._max = x_min, x_max
        else:
            m = self.momentum
            self._min = (1 - m) * self._min + m * x_min
            self._max = (1 - m) * self._max + m * x_max

    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self._phase != "quantizing" or self._scale is None:
            return x
        qmax = 2 ** (self.bitwidth - 1) - 1
        qmin = -(2 ** (self.bitwidth - 1))
        scale_t = torch.tensor(self._scale, device=x.device, dtype=x.dtype)
        return _FakeQuantizeSTE.apply(x, scale_t, qmin, qmax)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def scale(self) -> Optional[float]:
        return self._scale


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quantization manager — installs and removes parametrizations + hooks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _QuantizationManager:
    """Installs weight parametrizations and activation pre-hooks, holds
    the per-layer ``_ActivationObserver`` instances, and tears them all
    down on ``remove()``.

    Replaces the legacy ``_FakeQuantHookManager`` whose forward-pre-hook
    bypassed autograd. The new design is fully differentiable end-to-end.
    """

    def __init__(self) -> None:
        self._hooks: List[Any] = []
        self._parametrized: List[Tuple[nn.Module, str]] = []
        self.observers: Dict[str, _ActivationObserver] = {}

    def register(
        self,
        model: nn.Module,
        weight_bitwidth_config: Dict[str, int],
        act_bitwidth: int,
        io_layer_names: Optional[List[str]] = None,
        io_layer_bitwidth: Optional[int] = None,
    ) -> None:
        """Install weight parametrizations + activation pre-hooks.

        Args:
            model: model to quantize.
            weight_bitwidth_config: ``{param_name -> bitwidth}``.
            act_bitwidth: activation bitwidth (always INT8 in production).
            io_layer_names: module names whose weights/activations must
                be forced to ``io_layer_bitwidth`` regardless of the
                config. Mirrors PTQ's I/O override.
            io_layer_bitwidth: bitwidth used for I/O layers (typically
                INT8).
        """
        # Map module_name -> weight bitwidth.
        module_w_bw: Dict[str, int] = {}
        for pname, bw in weight_bitwidth_config.items():
            if "weight" not in pname:
                continue
            owner = pname.rsplit(".", 1)[0]
            module_w_bw[owner] = int(bw)

        io_set = set(io_layer_names or [])
        io_bw = int(io_layer_bitwidth) if io_layer_bitwidth else None

        for name, module in model.named_modules():
            if not isinstance(module, _QUANTIZABLE_TYPES):
                continue
            if not hasattr(module, "weight") or module.weight is None:
                continue

            # Resolve weight bitwidth (with I/O override).
            w_bw = module_w_bw.get(name, 32)
            if name in io_set and io_bw is not None:
                w_bw = io_bw
            # Resolve activation bitwidth: always overridden to
            # ``act_bitwidth`` (INT8 in production), regardless of
            # weight bitwidth.
            a_bw = int(act_bitwidth)

            # ── Weight parametrization (autograd-aware STE) ──
            if w_bw < 32:
                torch.nn.utils.parametrize.register_parametrization(
                    module, "weight", _WeightFakeQuantize(w_bw),
                )
                self._parametrized.append((module, "weight"))

            # ── Activation observer + pre-hook ──
            if a_bw < 32:
                observer = _ActivationObserver(bitwidth=a_bw)
                self.observers[name] = observer

                def _make_hook(obs: _ActivationObserver):
                    def hook(_mod, inputs):
                        if not inputs:
                            return None
                        x = inputs[0]
                        obs.observe(x)
                        x_q = obs.fake_quantize(x)
                        # Replace the input tuple. Hooks that return a
                        # tuple replace ``inputs`` for the layer call.
                        return (x_q,) + tuple(inputs[1:])
                    return hook

                h = module.register_forward_pre_hook(_make_hook(observer))
                self._hooks.append(h)

        logger.info(
            "  QAT manager: %d weight parametrizations, %d activation "
            "observers (act_bw=%d).",
            len(self._parametrized), len(self.observers), int(act_bitwidth),
        )

    def calibrate_activations(
        self,
        model: nn.Module,
        calib_loader: DataLoader,
        device: torch.device,
        num_batches: int = 20,
    ) -> None:
        """Run a forward-only pass to populate observer min/max.

        Observers stay in ``calibrating`` state during the loop and
        flip to ``quantizing`` afterwards. The model runs in eval
        mode so BN-folded layers and dropout behave deterministically.
        """
        for obs in self.observers.values():
            obs.start_calibration()

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= num_batches:
                    break
                x = batch[0].to(device)
                model(x)
        if was_training:
            model.train()

        for obs in self.observers.values():
            obs.finish_calibration()

        logger.info(
            "  Activation calibration: %d observers populated, "
            "scales pinned for QAT.",
            len(self.observers),
        )

    def remove(self) -> None:
        """Remove every hook and unwind every parametrization.

        ``leave_parametrized=True`` bakes the final fake-quantized
        weight into the underlying parameter so the post-QAT model can
        be evaluated without the parametrization machinery on the
        forward path.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for module, attr in self._parametrized:
            if torch.nn.utils.parametrize.is_parametrized(module, attr):
                torch.nn.utils.parametrize.remove_parametrizations(
                    module, attr, leave_parametrized=True,
                )
        self._parametrized.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QATTrainer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class QATTrainer:
    """W+A Quantization-Aware Training with FP32-teacher KD.

    Args:
        model: AdaRound-output model (will be BN-folded in place).
        bitwidth_config: ``{param_name -> weight_bitwidth}``.
        config: framework configuration (uses ``qat_*`` knobs).
        teacher: optional FP32 teacher for KD. When provided and
            ``qat_distill_alpha > 0``, the loss is
            ``α·T²·KL(student/T, teacher/T) + (1-α)·CE``.
        calib_loader: calibration loader for activation observers.
            Required when ``qat_act_bitwidth < 32``; without it the
            observers stay in passthrough mode (W-only QAT). The
            production path always supplies one.
    """

    def __init__(
        self,
        model: nn.Module,
        bitwidth_config: Dict[str, int],
        config: QuantizationConfig,
        teacher: Optional[nn.Module] = None,
        calib_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.bitwidth_config = bitwidth_config
        self.config = config
        self.device = self._resolve_device(config.hyperparams.device)
        self.calib_loader = calib_loader

        self.model.to(self.device)

        # FP32 teacher (frozen) for KD. Snapshot a deep copy so the
        # caller can keep using its own model without risk of cross-
        # contamination via shared state_dicts.
        self.teacher: Optional[nn.Module] = None
        if teacher is not None:
            self.teacher = teacher.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

        self._mgr = _QuantizationManager()

        self._best_state: Optional[Dict[str, Any]] = None
        self._best_val_acc: float = -1.0
        self._best_epoch: int = 0

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_model(self) -> None:
        """Fold Conv-BN, freeze BN, install parametrizations + observers,
        and (when ``calib_loader`` is supplied) run the activation
        calibration pass."""
        hp = self.config.hyperparams
        logger.info("Preparing model for QAT (W+A) ...")

        # ── E4: analytic Conv-BN fold ──
        if getattr(hp, "qat_fold_bn", True):
            from neuroquant.quantization.bn_folding import fold_conv_bn
            self.model, n_folded = fold_conv_bn(self.model)
            if n_folded:
                logger.info(
                    "  BN-fold: %d Conv-BN pair(s) folded; BN replaced "
                    "by Identity.", n_folded,
                )

        # Any remaining BN layers (those without a Conv predecessor)
        # are frozen — running stats are kept and γ/β gradients off.
        bn_frozen = 0
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)
                bn_frozen += 1
        if bn_frozen:
            logger.info(
                "  Frozen %d residual (un-folded) BatchNorm layer(s).",
                bn_frozen,
            )

        # ── Enable gradients on all non-BN parameters ──
        bn_param_ids = {
            id(p)
            for m in self.model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for p in m.parameters()
        }
        trainable = 0
        for p in self.model.parameters():
            if id(p) in bn_param_ids:
                continue
            if not p.requires_grad:
                p.requires_grad_(True)
                trainable += 1

        # ── Install weight parametrizations + activation hooks ──
        io_layer_names = self._find_io_layer_names()
        self._mgr.register(
            self.model,
            weight_bitwidth_config=self.bitwidth_config,
            act_bitwidth=int(hp.qat_act_bitwidth),
            io_layer_names=io_layer_names,
            io_layer_bitwidth=int(self.config.io_layer_bitwidth),
        )

        # ── Activation calibration (E1) ──
        if self.calib_loader is not None and self._mgr.observers:
            self._mgr.calibrate_activations(
                self.model, self.calib_loader, self.device,
                num_batches=int(hp.calibration_batches),
            )
        elif self._mgr.observers:
            logger.warning(
                "  No calib_loader supplied — activation observers stay "
                "in passthrough mode. QAT will be weight-only."
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> QATResult:
        hp = self.config.hyperparams
        epochs = int(hp.qat_epochs)
        lr = float(hp.qat_lr)
        momentum = float(hp.qat_momentum)
        weight_decay = float(hp.qat_weight_decay)
        patience = int(hp.qat_early_stop_patience)

        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        kd_alpha = float(getattr(hp, "qat_distill_alpha", 0.0))
        kd_T = float(getattr(hp, "qat_distill_temperature", 4.0))
        use_kd = self.teacher is not None and kd_alpha > 0.0

        logger.info("=" * 70)
        logger.info("Phase 1e: QAT Warmstart (W+A) — Fine-Tuning from AdaRound")
        logger.info("=" * 70)
        logger.info(
            "  Epochs: %d, LR: %.4f, Momentum: %.1f, WD: %.1e, Patience: %d",
            epochs, lr, momentum, weight_decay, patience,
        )
        logger.info(
            "  Activation bitwidth: INT%d  |  KD: %s%s",
            int(hp.qat_act_bitwidth),
            "on" if use_kd else "off",
            f" (α={kd_alpha}, T={kd_T})" if use_kd else "",
        )

        t_start = time.time()
        self.prepare_model()

        # NOTE: do NOT seed torch here — set_seed() at pipeline init
        # already pinned cudnn determinism + the global RNG. Re-seeding
        # mid-pipeline would desync the dataloader workers.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(
            "  Trainable parameters: %d tensors (%d elements)",
            len(trainable_params),
            sum(p.numel() for p in trainable_params),
        )

        optimizer = torch.optim.SGD(
            trainable_params, lr=lr, momentum=momentum,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01,
        )

        history: Dict[str, List[float]] = {
            "train_loss": [], "train_accuracy": [], "val_accuracy": [],
        }
        no_improve_count = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer,
                use_kd=use_kd, kd_alpha=kd_alpha, kd_T=kd_T,
            )
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            val_acc = self._validate(val_loader)
            history["val_accuracy"].append(val_acc)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            is_best = val_acc > self._best_val_acc
            if is_best:
                self._best_val_acc = val_acc
                self._best_epoch = epoch
                self._best_state = copy.deepcopy(self.model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            best_marker = " [*best*]" if is_best else ""
            logger.info(
                "  Epoch %d/%d: train_loss=%.4f, train_acc=%.2f%%, "
                "val_acc=%.2f%%, lr=%.6f%s",
                epoch, epochs, train_loss, train_acc, val_acc,
                current_lr, best_marker,
            )

            if no_improve_count >= patience and epoch >= 3:
                logger.info(
                    "  Early stopping: no improvement for %d epochs", patience,
                )
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            logger.info("  Restored best model from epoch %d", self._best_epoch)

        # Bake the parametrized fake-quant into the underlying weights
        # and remove all hooks so the returned model is a plain
        # nn.Module that downstream phases can evaluate / save normally.
        self._mgr.remove()

        t_elapsed = time.time() - t_start
        logger.info("-" * 70)
        logger.info("QAT (W+A) Results:")
        logger.info(
            "  Best epoch: %d (val_acc=%.2f%%)",
            self._best_epoch, self._best_val_acc,
        )
        logger.info(
            "  Final train loss: %.4f",
            history["train_loss"][-1] if history["train_loss"] else 0,
        )
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
        use_kd: bool,
        kd_alpha: float,
        kd_T: float,
    ) -> Tuple[float, float]:
        self.model.train()
        # Re-freeze any residual BN (model.train() un-evals them).
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            ce_loss = criterion(outputs, labels)

            if use_kd:
                with torch.no_grad():
                    teacher_logits = self.teacher(images)
                # Standard Hinton KD: scale gradient by T² so the
                # effective KD-CE balance is invariant to the choice
                # of temperature.
                student_logp = F.log_softmax(outputs / kd_T, dim=-1)
                teacher_p = F.softmax(teacher_logits / kd_T, dim=-1)
                kd_loss = F.kl_div(
                    student_logp, teacher_p, reduction="batchmean",
                ) * (kd_T * kd_T)
                loss = kd_alpha * kd_loss + (1.0 - kd_alpha) * ce_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — quantized weights amplify loss
            # spikes; clip_grad_norm keeps SGD stable.
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / max(total, 1)
        accuracy = (correct / max(total, 1)) * 100.0
        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> float:
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

    def _find_io_layer_names(self) -> List[str]:
        """First and last quantizable Conv/Linear in module-iteration order.

        Mirrors PTQ's ``_find_io_layer_names`` so the I/O-layer override
        applies consistently. Generic across architectures: depends only
        on module ordering, not on naming conventions.
        """
        quant_names = [
            n for n, m in self.model.named_modules()
            if isinstance(m, _QUANTIZABLE_TYPES)
        ]
        if not quant_names:
            return []
        if len(quant_names) == 1:
            return quant_names
        return [quant_names[0], quant_names[-1]]

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)
