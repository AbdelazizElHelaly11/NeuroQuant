"""
NeuroQuant v2.0 — Conv-BN analytic folding (Wave-2 production prep).

Real INT8 inference always ships with BatchNorm folded into the
preceding Conv2d / Linear, because mobile / FPGA / TensorRT backends do
not implement a separate BN op at INT8. If we run QAT with BN as a
distinct module, we are training the *wrong* operator: the deployment
graph drops BN, and the rounding error of the folded conv is what
actually determines accuracy.

This module performs the algebraic fold ahead of QAT. After folding,
BN is replaced by ``nn.Identity()`` and the conv's weight + bias absorb
``(γ, β, μ, σ²)``:

    σ        = sqrt(running_var + eps)
    factor   = γ / σ                                # [out_channels]
    weight'  = weight * factor[:, None, None, None]
    bias'    = (bias_or_zero - μ) * factor + β

The fold is mathematically lossless when the BN is in eval mode (which
is exactly what real INT8 inference uses). After the fold:

    folded(x) ≡ BN(Conv(x))   for all x

Generic across architectures: the only assumption is that a Conv layer
is immediately followed by a BN layer in the parent module's
``_modules`` order. Sequential blocks, ResNet-style attribute blocks,
and MobileNet inverted residuals all qualify.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("neuroquant")


_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _fold_conv_bn_pair(conv: nn.Module, bn: nn.Module) -> None:
    """Analytically fold ``bn`` into ``conv`` in-place.

    Requires the BN to have ``track_running_stats=True`` and finite
    ``running_mean`` / ``running_var`` (the FP32 baseline calibrated
    them during training; the standard pipeline already exercises
    this). Asserts shape compatibility so a misordered pair does not
    silently corrupt the conv.
    """
    if not getattr(bn, "track_running_stats", True):
        raise ValueError(
            "Cannot fold BatchNorm with track_running_stats=False — its "
            "running stats are undefined."
        )

    out_channels = conv.weight.shape[0]
    if bn.num_features != out_channels:
        raise ValueError(
            f"Conv out_channels={out_channels} does not match "
            f"BN num_features={bn.num_features}; refusing to fold."
        )

    device = conv.weight.device
    dtype = conv.weight.dtype

    gamma = (
        bn.weight.detach().to(device=device, dtype=dtype)
        if bn.affine and bn.weight is not None
        else torch.ones(out_channels, device=device, dtype=dtype)
    )
    beta = (
        bn.bias.detach().to(device=device, dtype=dtype)
        if bn.affine and bn.bias is not None
        else torch.zeros(out_channels, device=device, dtype=dtype)
    )
    mu = bn.running_mean.detach().to(device=device, dtype=dtype)
    var = bn.running_var.detach().to(device=device, dtype=dtype)
    eps = float(bn.eps)

    sigma = torch.sqrt(var + eps)
    factor = gamma / sigma  # [out_channels]

    # Reshape factor to broadcast across all spatial dims of the conv
    # weight: [out, in, *kernel] for Conv1d/2d/3d.
    weight_factor_shape = [out_channels] + [1] * (conv.weight.dim() - 1)
    new_weight = conv.weight.detach() * factor.view(*weight_factor_shape)

    if conv.bias is None:
        old_bias = torch.zeros(out_channels, device=device, dtype=dtype)
    else:
        old_bias = conv.bias.detach()
    new_bias = (old_bias - mu) * factor + beta

    # Commit the folded values. Conv2d created with ``bias=False`` has
    # ``conv.bias is None``; in that case we register a fresh bias
    # parameter rather than mutating a non-existent tensor.
    with torch.no_grad():
        conv.weight.copy_(new_weight)
        if conv.bias is None:
            conv.bias = nn.Parameter(new_bias)
        else:
            conv.bias.copy_(new_bias)


def fold_conv_bn(model: nn.Module) -> Tuple[nn.Module, int]:
    """Fold every Conv→BN pair in ``model`` in place.

    A pair is detected when the parent module's ``_modules``
    OrderedDict contains a Conv layer directly followed by a BN layer
    of matching width. ``nn.Sequential`` is the canonical case; it
    also works for any custom block whose ``__init__`` defines the BN
    immediately after the Conv (the Python class attribute order is
    preserved by ``OrderedDict`` insertion).

    Limitations / non-folds:
      * If a BN module is referenced from multiple parents (weight
        sharing), only the *first* fold proceeds; subsequent
        occurrences are skipped to avoid corrupting other paths.
      * Pairs that do not match in width are left untouched.
      * Standalone BN layers (no preceding Conv) are left untouched
        — they may legitimately exist (e.g. as an input normaliser).

    Returns ``(model, n_folded)`` so callers can assert the expected
    number of pairs were collapsed.
    """
    n_folded = 0
    folded_bn_ids: set = set()

    for parent in model.modules():
        # Snapshot child names; we will mutate _modules inside the loop.
        names = list(parent._modules.keys())
        for i in range(len(names) - 1):
            n_conv, n_bn = names[i], names[i + 1]
            conv = parent._modules.get(n_conv)
            bn = parent._modules.get(n_bn)
            if conv is None or bn is None:
                continue
            if not isinstance(conv, _CONV_TYPES):
                continue
            if not isinstance(bn, _BN_TYPES):
                continue
            if id(bn) in folded_bn_ids:
                logger.debug(
                    "  [BN-fold] skipping %s.%s — BN module already folded "
                    "from another parent (weight sharing).",
                    type(parent).__name__, n_bn,
                )
                continue
            try:
                _fold_conv_bn_pair(conv, bn)
            except ValueError as exc:
                logger.debug("  [BN-fold] skip %s/%s: %s", n_conv, n_bn, exc)
                continue
            folded_bn_ids.add(id(bn))
            # Replace BN with Identity so downstream code (QAT hooks,
            # the forward pass itself) sees the folded operator only.
            parent._modules[n_bn] = nn.Identity()
            n_folded += 1

    if n_folded:
        logger.info(
            "  [BN-fold] folded %d Conv-BN pair(s); BN replaced by Identity.",
            n_folded,
        )
    return model, n_folded


def list_conv_bn_pairs(model: nn.Module) -> List[Tuple[str, str]]:
    """Return the (conv_name, bn_name) pairs that ``fold_conv_bn`` would
    target. Useful for tests and for logging the fold plan before
    mutating the model.
    """
    pairs: List[Tuple[str, str]] = []
    name_lookup = {id(m): n for n, m in model.named_modules()}

    for parent in model.modules():
        names = list(parent._modules.keys())
        for i in range(len(names) - 1):
            conv = parent._modules.get(names[i])
            bn = parent._modules.get(names[i + 1])
            if (
                isinstance(conv, _CONV_TYPES)
                and isinstance(bn, _BN_TYPES)
                and conv.weight.shape[0] == bn.num_features
            ):
                pairs.append((
                    name_lookup.get(id(conv), names[i]),
                    name_lookup.get(id(bn), names[i + 1]),
                ))
    return pairs
