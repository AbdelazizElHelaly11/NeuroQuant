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
  * **FP32 teacher KD** (E5) when a teacher model is supplied — for
    classification only. Non-classification tasks disable KD
    automatically because their output formats are incompatible with
    the KL-div objective (detection returns dicts; segmentation maps
    need pixel-wise alignment that is out of scope for Phase 1e).

Task support:
  * ``classification`` — CrossEntropyLoss, Top-1 accuracy early-stop.
  * ``segmentation``   — CrossEntropyLoss(ignore_index=255) on the
    ``"out"`` head, negative-val-loss early-stop (mIoU is too
    expensive per epoch).
  * ``detection``      — model returns its own loss dict in train mode;
    sum of dict values is the scalar loss. Negative-val-loss early-stop.
  * ``nlp``            — HuggingFace contract: ``model(**x, labels=y).loss``.
    Negative-val-loss early-stop.
  * ``regression``     — MSELoss, negative-val-loss early-stop.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        batch_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    ) -> None:
        """Run a forward-only pass to populate observer min/max.

        Observers stay in ``calibrating`` state during the loop and
        flip to ``quantizing`` afterwards. The model runs in eval
        mode so BN-folded layers and dropout behave deterministically.

        Args:
            batch_fn: optional callable that extracts the image/input
                tensor from a batch. Defaults to ``batch[0]`` (the
                standard CV convention). Pass a custom extractor for
                NLP dict batches, detection list batches, etc.
        """
        for obs in self.observers.values():
            obs.start_calibration()

        # Default batch extractor: CV tuple (images, labels) → images.
        _extract = batch_fn if batch_fn is not None else (lambda b: b[0])

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= num_batches:
                    break
                try:
                    x = _extract(batch)
                except Exception as exc:
                    logger.warning(
                        "  QAT calibration: batch_fn failed on batch %d "
                        "(%s); skipping.", i, exc,
                    )
                    continue
                if isinstance(x, dict):
                    x = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in x.items()
                    }
                    model(**x)
                elif isinstance(x, (list, tuple)):
                    x = [t.to(device) if isinstance(t, torch.Tensor) else t
                         for t in x]
                    model(x)
                else:
                    model(x.to(device))
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
# Task-aware helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_batch_extractor(task: str) -> Callable[[Any], Any]:
    """Return a callable that unpacks the image/input tensor from a
    batch tuple/dict/list, handling every task's loader convention.

    Used for activation calibration so observers see the right input
    format regardless of task.
    """
    if task == "nlp":
        # HF loaders often pack everything into a dict; return it as-is
        # so the model receives ``model(**batch_dict)``.
        def _extract_nlp(batch):
            if isinstance(batch, dict):
                # Drop labels key so we only feed inputs to the model.
                return {k: v for k, v in batch.items() if k != "labels"}
            # Fallback: (input_dict, labels) tuple from some loaders.
            x = batch[0]
            return x
        return _extract_nlp

    if task == "detection":
        # torchvision detection: list of image tensors in [0, 1].
        # Calibration in eval mode — just feed the image list.
        def _extract_det(batch):
            images, _ = batch[0], batch[1]
            # images may already be a list[Tensor] or a stacked Tensor.
            if isinstance(images, torch.Tensor):
                return [img for img in images]
            return list(images)
        return _extract_det

    # Default (classification, segmentation, regression):
    # loader returns (images_tensor, labels_tensor).
    return lambda batch: batch[0]


def _make_task_forward(
    task: str,
    model: nn.Module,
    x: Any,
    y: Any,
    criterion: Optional[nn.Module],
    seg_aux_weight: float = 0.4,
) -> torch.Tensor:
    """Run one task-aware forward pass and return a scalar loss.

    Centralises all task dispatch so ``_train_epoch`` stays clean.

    Args:
        task: one of classification / segmentation / detection / nlp /
            regression.
        model: the (fake-quantized) student model.
        x: model input — tensor for CV tasks, dict for NLP, list for
            detection.
        y: target — label tensor, mask tensor, list of dicts, etc.
        criterion: loss module (used by classification / segmentation /
            regression). ``None`` for detection and NLP where the model
            produces its own loss.
        seg_aux_weight: weight on the auxiliary segmentation head loss
            (applied only when the output contains an ``"aux"`` key).
    """
    if task == "segmentation":
        output = model(x)
        if isinstance(output, dict):
            logits = output.get("out", next(iter(output.values())))
        else:
            logits = output
        # Masks can be [B, 1, H, W] from some loaders.
        target = y.squeeze(1) if (y.dim() == 4 and y.size(1) == 1) else y
        loss = criterion(logits, target.long())
        # Auxiliary deep-supervision head (present in DeepLabV3+, FCN).
        if isinstance(output, dict) and "aux" in output and seg_aux_weight > 0:
            aux_logits = output["aux"]
            loss = loss + seg_aux_weight * criterion(aux_logits, target.long())
        return loss

    if task == "detection":
        # torchvision detection in train() mode returns a dict of named
        # scalar losses; sum them into one scalar.
        loss_dict = model(x, y)
        if isinstance(loss_dict, dict):
            return sum(loss_dict.values())
        # Custom detector that already returns a scalar.
        return loss_dict

    if task == "nlp":
        if isinstance(x, dict):
            out = model(**x, labels=y) if y is not None else model(**x)
        else:
            out = model(x)
        if hasattr(out, "loss") and out.loss is not None:
            return out.loss
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            return out[0]
        raise RuntimeError(
            "NLP model did not expose ``.loss``; ensure labels are passed "
            "or included in the input dict."
        )

    if task == "regression":
        output = model(x)
        if isinstance(output, dict):
            output = output.get("out", next(iter(output.values())))
        pred = output.reshape(-1).float()
        target = y.reshape(-1).float()
        if pred.numel() != target.numel():
            m = min(pred.numel(), target.numel())
            pred, target = pred[:m], target[:m]
        return criterion(pred, target)

    # Default: classification.
    return criterion(model(x), y)


def _unpack_batch(
    batch: Any,
    task: str,
    device: torch.device,
) -> Tuple[Any, Any]:
    """Unpack a loader batch into ``(x, y)`` and move both to ``device``.

    Handles every batch format used by the framework:
      * ``(Tensor, Tensor)``          — CV classification / segmentation
      * ``(List[Tensor], List[dict])``— torchvision detection
      * ``dict``                      — HuggingFace NLP (labels inside)
      * ``(dict, Tensor)``            — some custom NLP loaders

    Returns ``(x, y)`` where ``x`` is already on ``device`` and ``y``
    is on ``device`` when it is a Tensor (or a list of device dicts for
    detection).
    """
    if isinstance(batch, dict):
        # Full HF dict: labels may be embedded.
        y = batch.pop("labels", None)
        x = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if isinstance(y, torch.Tensor):
            y = y.to(device)
        return x, y

    x_raw, y_raw = batch[0], batch[1]

    # x: list/tuple → detection image list
    if isinstance(x_raw, (list, tuple)):
        x = [t.to(device) if isinstance(t, torch.Tensor) else t
             for t in x_raw]
    elif isinstance(x_raw, dict):
        x = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in x_raw.items()
        }
    else:
        x = x_raw.to(device)

    # y: list of dicts → detection targets
    if (
        isinstance(y_raw, (list, tuple))
        and y_raw
        and isinstance(y_raw[0], dict)
    ):
        y = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in y_raw
        ]
    elif isinstance(y_raw, torch.Tensor):
        y = y_raw.to(device)
    else:
        y = y_raw

    return x, y


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QATTrainer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class QATTrainer:
    """W+A Quantization-Aware Training with FP32-teacher KD (classification)
    or loss-driven fine-tuning (all other tasks).

    Args:
        model: AdaRound-output model (will be BN-folded in place).
        bitwidth_config: ``{param_name -> weight_bitwidth}``.
        config: framework configuration (uses ``qat_*`` knobs).
        task: one of ``"classification"``, ``"segmentation"``,
            ``"detection"``, ``"nlp"``, ``"regression"``.
            Determines the loss function, batch unpacking, and the
            early-stopping metric.
        teacher: optional FP32 teacher for knowledge distillation.
            **Only used when ``task == "classification"``**; ignored
            silently for other tasks because output-space KD is not
            straightforward for detection/segmentation/NLP.
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
        task: str = "classification",
        teacher: Optional[nn.Module] = None,
        calib_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.bitwidth_config = bitwidth_config
        self.config = config
        self.task = str(task).lower()
        self.device = self._resolve_device(config.hyperparams.device)
        self.calib_loader = calib_loader

        self.model.to(self.device)

        # FP32 teacher (frozen) — classification only.
        self.teacher: Optional[nn.Module] = None
        if teacher is not None:
            if self.task == "classification":
                self.teacher = teacher.to(self.device)
                self.teacher.eval()
                for p in self.teacher.parameters():
                    p.requires_grad_(False)
            else:
                logger.info(
                    "  QATTrainer: teacher supplied but task='%s'. "
                    "Knowledge distillation is disabled for non-"
                    "classification tasks — output-space KL requires "
                    "compatible logit tensors. The teacher is ignored.",
                    self.task,
                )

        self._mgr = _QuantizationManager()

        # ``_best_val_metric`` stores the highest observed validation
        # metric so far. For classification this is Top-1 accuracy
        # (higher = better). For all other tasks it is the negative of
        # the mean val loss (higher = less loss = better), so the same
        # ``is_best = metric > _best_val_metric`` condition works
        # universally.
        self._best_val_metric: float = -float("inf")
        self._best_state: Optional[Dict[str, Any]] = None
        self._best_epoch: int = 0

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _build_criterion(self) -> Optional[nn.Module]:
        """Return the appropriate loss module for the configured task.

        Returns ``None`` for tasks where the model produces its own
        loss (detection in train mode, NLP via HuggingFace contract).
        In those cases ``_make_task_forward`` bypasses the criterion
        argument entirely.
        """
        hp = self.config.hyperparams
        if self.task == "classification":
            return nn.CrossEntropyLoss()
        if self.task == "segmentation":
            return nn.CrossEntropyLoss(ignore_index=255)
        if self.task == "regression":
            return nn.MSELoss()
        # detection and nlp: model owns its loss.
        return None

    def prepare_model(self) -> None:
        """Fold Conv-BN, freeze BN, install parametrizations + observers,
        and (when ``calib_loader`` is supplied) run the activation
        calibration pass."""
        hp = self.config.hyperparams
        logger.info(
            "Preparing model for QAT (W+A, task=%s) ...", self.task,
        )

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
        for p in self.model.parameters():
            if id(p) in bn_param_ids:
                continue
            if not p.requires_grad:
                p.requires_grad_(True)

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
            batch_fn = _make_batch_extractor(self.task)
            self._mgr.calibrate_activations(
                self.model, self.calib_loader, self.device,
                num_batches=int(hp.calibration_batches),
                batch_fn=batch_fn,
            )
        elif self._mgr.observers:
            logger.warning(
                "  No calib_loader supplied — activation observers stay "
                "in passthrough mode. QAT will be weight-only."
            )

        # ── Detection backbone freeze warmup ──
        # Freeze backbone weights for the first N epochs to stabilise
        # training before fine-tuning the full network.
        det_freeze = int(getattr(hp, "qat_det_freeze_backbone_epochs", 0))
        self._det_freeze_epochs = det_freeze if self.task == "detection" else 0

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
        seg_aux_weight = float(getattr(hp, "qat_seg_aux_weight", 0.4))

        if criterion is None:
            criterion = self._build_criterion()
        if criterion is not None:
            criterion = criterion.to(self.device)

        kd_alpha = float(getattr(hp, "qat_distill_alpha", 0.0))
        kd_T = float(getattr(hp, "qat_distill_temperature", 4.0))
        use_kd = (
            self.teacher is not None
            and kd_alpha > 0.0
            and self.task == "classification"
        )

        logger.info("=" * 70)
        logger.info(
            "Phase 1e: QAT Warmstart (W+A) — task=%s", self.task,
        )
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
        early_stop_desc = (
            "Top-1 accuracy" if self.task == "classification"
            else "negative val loss"
        )
        logger.info(
            "  Early-stop metric: %s", early_stop_desc,
        )

        t_start = time.time()
        self.prepare_model()

        # NOTE: do NOT seed torch here — set_seed() at pipeline init
        # already pinned cudnn determinism + the global RNG.
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
            # Detection backbone freeze: unfreeze after warmup epochs.
            if self._det_freeze_epochs > 0:
                if epoch == 1:
                    self._freeze_backbone()
                elif epoch == self._det_freeze_epochs + 1:
                    self._unfreeze_backbone()
                    logger.info(
                        "  [Epoch %d] Detection backbone unfrozen.", epoch,
                    )

            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer,
                use_kd=use_kd, kd_alpha=kd_alpha, kd_T=kd_T,
                seg_aux_weight=seg_aux_weight,
            )
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc if train_acc is not None else 0.0)

            val_metric, val_display = self._validate(val_loader)
            history["val_accuracy"].append(val_metric)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            is_best = val_metric > self._best_val_metric
            if is_best:
                self._best_val_metric = val_metric
                self._best_epoch = epoch
                self._best_state = copy.deepcopy(self.model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            best_marker = " [*best*]" if is_best else ""
            logger.info(
                "  Epoch %d/%d: train_loss=%.4f, train_acc=%s, "
                "val_%s=%.4f, lr=%.6f%s",
                epoch, epochs, train_loss,
                f"{train_acc:.2f}%" if train_acc is not None else "n/a",
                "top1" if self.task == "classification" else "metric",
                val_display,
                current_lr, best_marker,
            )

            if no_improve_count >= patience and epoch >= 3:
                logger.info(
                    "  Early stopping: no improvement for %d epochs", patience,
                )
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            logger.info(
                "  Restored best model from epoch %d", self._best_epoch,
            )

        # Bake the parametrized fake-quant into the underlying weights
        # and remove all hooks so the returned model is a plain
        # nn.Module that downstream phases can evaluate / save normally.
        self._mgr.remove()

        t_elapsed = time.time() - t_start
        logger.info("-" * 70)
        logger.info("QAT (W+A) Results (task=%s):", self.task)
        logger.info(
            "  Best epoch: %d (val_metric=%.4f)",
            self._best_epoch, self._best_val_metric,
        )
        logger.info(
            "  Final train loss: %.4f",
            history["train_loss"][-1] if history["train_loss"] else 0,
        )
        logger.info("  Time: %.1f seconds", t_elapsed)
        logger.info("=" * 70)

        # ``final_val_acc`` is kept for backward compatibility:
        # classification→ top-1%; other tasks → negative val loss
        # (a larger negative-loss is still "higher" in the best-epoch
        # tracker; callers that need the raw loss can negate it).
        return QATResult(
            model=self.model,
            train_accuracy=history["train_accuracy"],
            val_accuracy=history["val_accuracy"],
            train_loss=history["train_loss"],
            best_epoch=self._best_epoch,
            final_val_acc=self._best_val_metric,
            time_seconds=t_elapsed,
        )

    # ------------------------------------------------------------------
    # Single-epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        use_kd: bool,
        kd_alpha: float,
        kd_T: float,
        seg_aux_weight: float = 0.4,
    ) -> Tuple[float, Optional[float]]:
        """Run one training epoch.

        Returns ``(avg_loss, accuracy_or_None)``. ``accuracy`` is
        Top-1 % for classification; ``None`` for all other tasks (they
        have no per-batch scalar accuracy notion at training time).
        """
        self.model.train()
        # Detection models manage their own train/eval state per module;
        # for segmentation / classification we re-freeze BN after
        # model.train() resets it.
        if self.task != "detection":
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        samples_seen = 0

        for batch in train_loader:
            x, y = _unpack_batch(batch, self.task, self.device)

            # Batch size — task-aware.
            if isinstance(x, (list, tuple)):
                batch_size = len(x)
            elif isinstance(x, dict):
                sentinel = x.get("input_ids", next(iter(x.values()), None))
                batch_size = (
                    sentinel.size(0) if isinstance(sentinel, torch.Tensor) else 1
                )
            else:
                batch_size = x.size(0)

            # Task-aware forward + loss.
            ce_loss = _make_task_forward(
                self.task, self.model, x, y, criterion,
                seg_aux_weight=seg_aux_weight,
            )

            if use_kd:
                # KD is classification-only (enforced in __init__).
                # We need the raw logits from the student — NOT the scalar
                # ce_loss — to compute KL-div. The ce_loss computed above
                # already captures the student loss; run a second forward
                # to get logits for the soft-target KL term.
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                student_logits = self.model(x)  # raw logits [B, C]
                student_logp = F.log_softmax(student_logits / kd_T, dim=-1)
                teacher_p = F.softmax(teacher_logits / kd_T, dim=-1)
                kd_loss = F.kl_div(
                    student_logp, teacher_p, reduction="batchmean",
                ) * (kd_T * kd_T)
                loss = kd_alpha * kd_loss + (1.0 - kd_alpha) * ce_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            running_loss += loss.item() * batch_size
            samples_seen += batch_size

            # Top-1 accumulator — classification only.
            if self.task == "classification" and isinstance(y, torch.Tensor):
                with torch.no_grad():
                    outputs = self.model(x)
                    _, predicted = outputs.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

        avg_loss = running_loss / max(samples_seen, 1)
        accuracy = (
            (correct / max(total, 1)) * 100.0
            if self.task == "classification" and total > 0
            else None
        )
        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Compute the validation metric for early stopping.

        Returns ``(metric, display_value)`` where:
          * classification → ``(top1_accuracy, top1_accuracy)``
          * other tasks    → ``(-mean_val_loss, mean_val_loss)``

        The first element is used for ``is_best`` comparisons (higher
        is always better). The second is logged for readability.

        Detection note: the model is set to eval() for inference — in
        eval mode, torchvision detectors return prediction dicts, not
        loss dicts. We compute val loss by temporarily switching to
        train mode per batch, then restoring eval.
        """
        self.model.eval()

        if self.task == "classification":
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = _unpack_batch(batch, self.task, self.device)
                    outputs = self.model(x)
                    _, predicted = outputs.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()
            top1 = (correct / max(total, 1)) * 100.0
            return top1, top1

        # Non-classification: compute mean val loss as the proxy.
        criterion = self._build_criterion()
        if criterion is not None:
            criterion = criterion.to(self.device)
        total_loss = 0.0
        count = 0

        for batch in val_loader:
            x, y = _unpack_batch(batch, self.task, self.device)
            try:
                if self.task == "detection":
                    # Detection: must be in train() to produce a loss dict.
                    # Switch per-batch and restore immediately after.
                    self.model.train()
                    with torch.enable_grad():
                        loss_dict = self.model(x, y)
                    self.model.eval()
                    if isinstance(loss_dict, dict):
                        loss_val = float(sum(v.item() for v in loss_dict.values()))
                    else:
                        loss_val = float(loss_dict.item())
                else:
                    with torch.no_grad():
                        loss = _make_task_forward(
                            self.task, self.model, x, y, criterion,
                        )
                        loss_val = float(loss.item())
                total_loss += loss_val
                count += 1
            except Exception as exc:
                logger.warning(
                    "  QAT val loss computation failed on batch: %s", exc,
                )
        mean_loss = total_loss / max(count, 1)
        # Return negative so "higher = better" holds universally.
        return -mean_loss, mean_loss

    # ------------------------------------------------------------------
    # Detection backbone freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze the backbone parameters for detection warm-up.

        Heuristic: parameters whose name starts with ``"backbone"``
        are considered backbone. Generalises to torchvision Faster
        RCNN, RetinaNet, SSD, and similar architectures without
        hard-coding module names beyond the conventional prefix.
        """
        frozen = 0
        for name, p in self.model.named_parameters():
            if name.startswith("backbone"):
                p.requires_grad_(False)
                frozen += 1
        logger.info(
            "  [Detection QAT] Backbone frozen: %d parameters "
            "(warmup for %d epoch(s)).",
            frozen, self._det_freeze_epochs,
        )

    def _unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters after the warm-up phase."""
        unfrozen = 0
        for name, p in self.model.named_parameters():
            if name.startswith("backbone"):
                p.requires_grad_(True)
                unfrozen += 1
        logger.info(
            "  [Detection QAT] Backbone unfrozen: %d parameters.",
            unfrozen,
        )

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
