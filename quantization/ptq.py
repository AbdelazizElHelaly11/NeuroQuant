"""
NeuroQuant v2.0 - PTQ Quantizer (Phase 1c core)

Post-Training Quantization with cluster-aware mixed-precision.

Calibration is strategy-aware:
    * kl_divergence  — TensorRT-style histogram search: the per-layer
      clipping threshold is the one that minimises the KL divergence
      between the original activation distribution and its quantized
      reconstruction.
    * mse — clip-range grid search: the threshold is the one that
      minimises the mean squared error between the calibration samples
      and their fake-quantized version.

By default, I/O layers use KL and intermediate layers use MSE, exactly
as declared in ``HyperparameterSet``. The two searches run on the same
collected samples and can produce materially different thresholds for
the same distribution (see the unit tests in ``test_ptq_wiring.py``).
"""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ClusterAssignment, QuantizationConfig, QuantizationResult
from quantization.base import BaseQuantizer

logger = logging.getLogger("neuroquant")


# Quantizable module types considered by the generic PTQ traversal.
_QUANTIZABLE_TYPES = (nn.Conv2d, nn.Linear)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy-specific threshold search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _kl_threshold(
    data: torch.Tensor,
    bitwidth: int,
    num_bins: int = 256,
    num_candidates: int = 32,
) -> float:
    """Pick a symmetric clipping threshold that minimises KL(P || Q).

    ``P`` is the normalised histogram of the absolute activations, truncated
    at the candidate threshold. ``Q`` is the same histogram after it has
    been coarse-grained into ``2^(bitwidth-1) - 1`` quantization levels and
    expanded back to the original resolution — i.e. the reconstruction the
    quantizer would produce at that clipping.
    """
    if data.numel() == 0:
        return 0.0
    abs_data = data.abs().to(torch.float32)
    amax = float(abs_data.max().item())
    if amax < 1e-12:
        return amax

    hist = torch.histc(abs_data, bins=num_bins, min=0.0, max=amax)
    levels = max(2 ** (bitwidth - 1) - 1, 1)

    # Candidate thresholds cover the upper half of the range — below 50%
    # the clipping distortion is almost always worse than the rounding
    # distortion, so narrowing the search keeps calibration cheap.
    fractions = torch.linspace(0.5, 1.0, num_candidates)

    best_t = amax
    best_kl = float("inf")
    for frac in fractions:
        t_val = float((amax * frac).item())
        cutoff = max(1, min(num_bins, int(round((t_val / amax) * num_bins))))

        # Build P: hist up to cutoff, with overflow folded into the last bin
        # — this mirrors what a real clipping quantizer would see.
        P = hist[:cutoff].clone()
        overflow = hist[cutoff:].sum()
        P[-1] = P[-1] + overflow

        # Build Q: coarse-grain P into ``levels`` bins, then expand back.
        if cutoff <= levels:
            Q = P.clone()
        else:
            bucket = cutoff / float(levels)
            Q_small = torch.zeros(levels)
            counts = torch.zeros(levels)
            for i in range(cutoff):
                j = min(int(i / bucket), levels - 1)
                Q_small[j] += P[i]
                counts[j] += 1
            Q = torch.zeros(cutoff)
            for i in range(cutoff):
                j = min(int(i / bucket), levels - 1)
                mean = Q_small[j] / counts[j].clamp(min=1.0)
                Q[i] = mean

        P_sum = P.sum().clamp(min=1e-12)
        Q_sum = Q.sum().clamp(min=1e-12)
        P_norm = P / P_sum + 1e-12
        Q_norm = Q / Q_sum + 1e-12
        kl = float((P_norm * (P_norm.log() - Q_norm.log())).sum().item())

        if kl < best_kl:
            best_kl = kl
            best_t = t_val
    return best_t


def _mse_threshold(
    data: torch.Tensor,
    bitwidth: int,
    num_candidates: int = 50,
) -> float:
    """Pick a symmetric clipping threshold that minimises reconstruction MSE.

    Grid-searches the clipping fraction over the full observed range and
    fake-quantizes the sample at each candidate, keeping the fraction that
    yields the lowest L2 error against the original sample.
    """
    if data.numel() == 0:
        return 0.0
    d = data.to(torch.float32)
    amax = float(d.abs().max().item())
    if amax < 1e-12:
        return amax

    qmax = max(2 ** (bitwidth - 1) - 1, 1)
    qmin = -(2 ** (bitwidth - 1))

    best_t = amax
    best_mse = float("inf")
    for frac in torch.linspace(0.3, 1.0, num_candidates):
        t_val = float((amax * frac).item())
        scale = max(t_val / qmax, 1e-12)
        q = (d / scale).round().clamp(qmin, qmax) * scale
        mse = float(((d - q) ** 2).mean().item())
        if mse < best_mse:
            best_mse = mse
            best_t = t_val
    return best_t


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Activation Observer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _ActivationObserver:
    """Records activation statistics for a single module during calibration.

    Keeps a bounded pool of activation samples plus running min/max so the
    strategy-specific threshold searches can run once at the end of
    calibration. ``threshold`` is ``None`` until :meth:`compute` is called.
    """

    def __init__(
        self,
        strategy: str = "mse",
        max_samples: int = 4096,
        num_bins: int = 256,
    ) -> None:
        self.strategy = (strategy or "mse").lower()
        self.max_samples = int(max_samples)
        self.num_bins = int(num_bins)

        self._samples: List[torch.Tensor] = []
        self._taken = 0
        self.total_seen = 0
        self.num_batches = 0

        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self.threshold: Optional[float] = None
        self.amax: Optional[float] = None

    def observe(self, tensor: torch.Tensor) -> None:
        """Record a batch of activations.

        Uses reservoir-style bounded sampling so memory stays constant
        regardless of how many batches are fed in.
        """
        t = tensor.detach()
        flat = t.reshape(-1).to(torch.float32)
        n = flat.numel()
        self.num_batches += 1
        if n == 0:
            return

        self.total_seen += n
        tmin = float(flat.min().item())
        tmax = float(flat.max().item())
        if self.min_val is None:
            self.min_val, self.max_val = tmin, tmax
        else:
            self.min_val = min(self.min_val, tmin)
            self.max_val = max(self.max_val, tmax)

        if self._taken < self.max_samples:
            remaining = self.max_samples - self._taken
            if n <= remaining:
                self._samples.append(flat.detach().cpu())
                self._taken += n
            else:
                idx = torch.randperm(n)[:remaining]
                self._samples.append(flat[idx].detach().cpu())
                self._taken += remaining

    def collected(self) -> torch.Tensor:
        """Return all collected samples as a single 1-D tensor."""
        if not self._samples:
            return torch.zeros(0)
        return torch.cat(self._samples, dim=0)

    def compute(self, bitwidth: int) -> Optional[float]:
        """Run the strategy-specific search and cache the chosen threshold.

        Returns the threshold (or ``None`` if no samples were captured).
        """
        data = self.collected()
        if data.numel() == 0:
            self.threshold = None
            return None
        self.amax = float(data.abs().max().item())
        if self.strategy.startswith("kl"):
            self.threshold = _kl_threshold(data, bitwidth, self.num_bins)
        else:
            self.threshold = _mse_threshold(data, bitwidth)
        return self.threshold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PTQQuantizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PTQQuantizer(BaseQuantizer):
    """
    Post-Training Quantization with:
    - Per-layer calibration (KL divergence for I/O, MSE for intermediate)
    - Cluster-aware mixed-precision bitwidth assignment
    - INT8 enforcement for input/output layers
    """

    def __init__(self, model: nn.Module, config: QuantizationConfig) -> None:
        super().__init__(model, config)
        # Observer state is keyed by module name; populated by calibrate().
        self._observers: Dict[str, _ActivationObserver] = {}
        self._io_layer_names: List[str] = self._find_io_layer_names(model)
        # Remember the last bitwidth used during calibration so that
        # quantize() can reuse the cached thresholds consistently.
        self._calibrated_bitwidth: Optional[int] = None
        # Per-layer bitwidth used during calibration when the
        # bitwidth-assignment-aware path was taken. Empty when the
        # legacy single-bitwidth ``calibrate()`` path was used.
        self._per_layer_bitwidth: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, bitwidth_assignment: Dict[str, int]) -> nn.Module:
        """Quantize the model with a specific parameter-wise bitwidth map.

        Uses the calibration-derived clipping threshold for each layer
        whenever available: the symmetric scale becomes ``threshold / qmax``
        rather than ``|w|.max() / qmax``. This is where the KL vs MSE
        choice becomes observable — different strategies produce different
        thresholds and therefore different quantized weights.

        Falls back to per-channel (Conv2d) / per-tensor (Linear) max-based
        quantization when a layer has no observer or an observer without a
        computed threshold, logging a single warning per uncalibrated layer.
        """
        q_model = copy.deepcopy(self.model)
        q_model.to(self.device)
        q_model.eval()

        io_bw = int(self.config.io_layer_bitwidth)
        supported = set(self.config.supported_bitwidths)
        name_to_module = dict(q_model.named_modules())

        with torch.no_grad():
            for pname, param in q_model.named_parameters():
                if "weight" not in pname:
                    continue
                owner = pname.rsplit(".", 1)[0]
                module = name_to_module.get(owner, None)
                if not isinstance(module, _QUANTIZABLE_TYPES):
                    continue

                # I/O layers stay at io_layer_bitwidth regardless of the
                # search result, matching config.io_layer_bitwidth.
                if owner in self._io_layer_names:
                    bw = io_bw
                else:
                    bw = int(bitwidth_assignment.get(pname, 32))
                    if supported and bw != 32 and bw not in supported:
                        bw = max(supported)

                if bw >= 32:
                    continue  # no-op

                observer = self._observers.get(owner)
                threshold = observer.threshold if observer is not None else None

                if threshold is not None and threshold > 0.0:
                    # Use the calibration-derived symmetric threshold as
                    # the quantization range. Same bitwidth on both strategies,
                    # but different threshold → different scale → different q.
                    qmax = 2 ** (bw - 1) - 1
                    qmin = -(2 ** (bw - 1))
                    scale = max(float(threshold) / max(qmax, 1), 1e-12)
                    q = (param.data / scale).round().clamp(qmin, qmax)
                    param.data = q * scale
                else:
                    # Deterministic fallback: per-channel for Conv2d,
                    # per-tensor for Linear. Warn once per layer.
                    if observer is None:
                        logger.warning(
                            "  [PTQ] No calibration for %s — falling back to "
                            "max-scale %s quantization.",
                            owner,
                            "per-channel" if isinstance(module, nn.Conv2d)
                            else "per-tensor",
                        )
                    per_channel = isinstance(module, nn.Conv2d)
                    param.data = self.quantize_tensor(
                        param.data, bitwidth=bw,
                        per_channel=per_channel, channel_dim=0,
                    )
        return q_model

    def quantize_with_config(
        self, bitwidth_assignment: Dict[str, int]
    ) -> nn.Module:
        """Alias of :meth:`quantize` provided for API symmetry with clusters."""
        return self.quantize(bitwidth_assignment)

    def calibrate(
        self,
        calibration_loader: DataLoader,
        num_batches: int = 20,
        bitwidth: int = 8,
    ) -> None:
        """Run calibration to populate per-layer activation observers.

        Observers collect bounded activation samples and min/max of the
        first input to each quantizable module. The strategy tag chosen
        per-layer (KL for I/O, MSE for intermediate — from
        ``hyperparams.calibration_strategy_*``) drives the post-collection
        threshold search.

        The resulting thresholds are cached on the quantizer and consumed
        by :meth:`quantize`.
        """
        self._observers.clear()
        self._calibrated_bitwidth = int(bitwidth)
        hp = self.config.hyperparams
        io_names = set(self._io_layer_names)

        strategy_io = getattr(
            hp.calibration_strategy_io, "value", str(hp.calibration_strategy_io)
        )
        strategy_intermediate = getattr(
            hp.calibration_strategy_intermediate,
            "value", str(hp.calibration_strategy_intermediate),
        )

        for name, module in self.model.named_modules():
            if not isinstance(module, _QUANTIZABLE_TYPES):
                continue
            strategy = strategy_io if name in io_names else strategy_intermediate
            self._apply_observer(module, strategy)
            self._observers[name] = module._nq_observer  # type: ignore[attr-defined]

        hooks = []
        for name, module in self.model.named_modules():
            obs = self._observers.get(name)
            if obs is None:
                continue

            def make_hook(observer: _ActivationObserver):
                def _hook(_mod, inputs, _output):
                    if inputs:
                        observer.observe(inputs[0])
                return _hook

            hooks.append(module.register_forward_hook(make_hook(obs)))

        try:
            self.model.eval()
            self.model.to(self.device)
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if i >= num_batches:
                        break
                    images = batch[0].to(self.device)
                    self.model(images)
        finally:
            for h in hooks:
                h.remove()

        # Final per-layer threshold search using the calibration strategy.
        for name, obs in self._observers.items():
            obs.compute(bitwidth=self._calibrated_bitwidth)

        logger.info(
            "  PTQ calibration: %d observers populated, strategies={io:%s, "
            "intermediate:%s}, bitwidth=%d",
            len(self._observers), strategy_io, strategy_intermediate,
            self._calibrated_bitwidth,
        )

    def calibrate_with_assignment(
        self,
        calibration_loader: DataLoader,
        bitwidth_assignment: Dict[str, int],
        num_batches: int = 20,
    ) -> None:
        """Run calibration where each layer's threshold is computed at its
        own target bitwidth from ``bitwidth_assignment``.

        This is the mixed-precision-aware variant of :meth:`calibrate`.
        For an INT4/INT8 mixed assignment, INT4 layers get a threshold
        chosen for ``2^3 - 1 = 7`` quantization levels and INT8 layers
        get one chosen for ``2^7 - 1 = 127`` levels — both via the same
        per-layer KL/MSE strategy. ``calibrate()`` would instead apply a
        single global bitwidth to every layer's threshold search, which
        is wrong on mixed configurations.

        I/O layer enforcement still applies: the first and last
        quantizable modules use ``config.io_layer_bitwidth`` regardless
        of what's in ``bitwidth_assignment``.
        """
        self._observers.clear()
        self._calibrated_bitwidth = None  # mixed/per-layer; see _per_layer_bitwidth
        self._per_layer_bitwidth = {}

        hp = self.config.hyperparams
        io_names = set(self._io_layer_names)
        io_bw = int(self.config.io_layer_bitwidth)
        supported = list(self.config.supported_bitwidths) or [4, 8]
        default_bw = max(supported)

        strategy_io = getattr(
            hp.calibration_strategy_io, "value", str(hp.calibration_strategy_io)
        )
        strategy_intermediate = getattr(
            hp.calibration_strategy_intermediate,
            "value", str(hp.calibration_strategy_intermediate),
        )

        # Resolve a per-module target bitwidth from the parameter-keyed
        # assignment. We look up ``<module>.weight`` first, then fall back
        # to any other key whose owner is this module.
        def _resolve_bw(module_name: str) -> int:
            if module_name in io_names:
                return io_bw
            weight_key = f"{module_name}.weight"
            if weight_key in bitwidth_assignment:
                return int(bitwidth_assignment[weight_key])
            for pname, bw in bitwidth_assignment.items():
                if pname.rsplit(".", 1)[0] == module_name and "weight" in pname:
                    return int(bw)
            return default_bw

        for name, module in self.model.named_modules():
            if not isinstance(module, _QUANTIZABLE_TYPES):
                continue
            bw_target = _resolve_bw(name)
            if supported and bw_target != 32 and bw_target not in supported:
                bw_target = max(supported)
            self._per_layer_bitwidth[name] = bw_target
            strategy = strategy_io if name in io_names else strategy_intermediate
            self._apply_observer(module, strategy)
            self._observers[name] = module._nq_observer  # type: ignore[attr-defined]

        hooks = []
        for name, module in self.model.named_modules():
            obs = self._observers.get(name)
            if obs is None:
                continue

            def make_hook(observer: _ActivationObserver):
                def _hook(_mod, inputs, _output):
                    if inputs:
                        observer.observe(inputs[0])
                return _hook

            hooks.append(module.register_forward_hook(make_hook(obs)))

        try:
            self.model.eval()
            self.model.to(self.device)
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if i >= num_batches:
                        break
                    images = batch[0].to(self.device)
                    self.model(images)
        finally:
            for h in hooks:
                h.remove()

        # Per-layer threshold search at each layer's target bitwidth.
        bw_distribution: Dict[int, int] = {}
        for name, obs in self._observers.items():
            bw = self._per_layer_bitwidth[name]
            obs.compute(bitwidth=bw)
            bw_distribution[bw] = bw_distribution.get(bw, 0) + 1

        logger.info(
            "  PTQ mixed-precision calibration: %d observers, "
            "strategies={io:%s, intermediate:%s}, bitwidths=%s",
            len(self._observers), strategy_io, strategy_intermediate,
            sorted(bw_distribution.items()),
        )

    def generate_cluster_configs(
        self, cluster_assignments: List[ClusterAssignment]
    ) -> List[Dict[str, int]]:
        """Enumerate bitwidth configurations from cluster constraints.

        HIGH tier clusters are fixed at INT8. MEDIUM/LOW clusters expand
        across their ``allowed_bitwidths``. The Cartesian product can get
        large, so it is capped to keep the grid search tractable on small
        configurations; this does not affect NSGA-driven runs.
        """
        fixed: Dict[str, int] = {}
        searchable: List[ClusterAssignment] = []

        for ca in cluster_assignments:
            tier = ca.get("tier", "LOW").upper()
            if tier == "HIGH":
                for pname in ca["layer_names"]:
                    fixed[pname] = 8
            else:
                searchable.append(ca)

        if not searchable:
            return [fixed]

        choice_lists = [list(ca["allowed_bitwidths"]) for ca in searchable]
        configs: List[Dict[str, int]] = []
        max_configs = 64  # defensive cap for exhaustive enumeration
        for combo in itertools.product(*choice_lists):
            cfg = dict(fixed)
            for ca, bw in zip(searchable, combo):
                for pname in ca["layer_names"]:
                    cfg[pname] = int(bw)
            configs.append(cfg)
            if len(configs) >= max_configs:
                break
        return configs

    def evaluate_all_configs(
        self,
        configs: List[Dict[str, int]],
        test_loader: DataLoader,
    ) -> List[QuantizationResult]:
        """Evaluate every supplied bitwidth configuration."""
        results: List[QuantizationResult] = []
        for idx, cfg in enumerate(configs):
            q_model = self.quantize(cfg)
            # Derive a representative bitwidth for ebops accounting:
            # use the weighted mean across quantized weights.
            weights_bw = [v for v in cfg.values() if v < 32]
            bw_repr = int(round(sum(weights_bw) / len(weights_bw))) if weights_bw else 32
            res = self.evaluate(q_model, test_loader, bitwidth=bw_repr)
            # Overwrite the config_id with a search-aware label and reattach
            # the exact bitwidth assignment used.
            res["config_id"] = f"PTQ_cfg{idx}"
            res["bitwidth_assignment"] = cfg
            results.append(res)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_observer(self, layer: nn.Module, strategy: str) -> None:
        """Attach a fresh activation observer to a layer in-place."""
        layer._nq_observer = _ActivationObserver(strategy=strategy)  # type: ignore[attr-defined]

    @staticmethod
    def _find_io_layer_names(model: nn.Module) -> List[str]:
        """Locate the first and last quantizable module names.

        Used to enforce ``config.io_layer_bitwidth`` on input/output layers
        regardless of the cluster assignment. Generic across architectures:
        relies only on module type order, not naming conventions.
        """
        quant_names = [
            name for name, m in model.named_modules()
            if isinstance(m, _QUANTIZABLE_TYPES)
        ]
        if not quant_names:
            return []
        if len(quant_names) == 1:
            return quant_names
        return [quant_names[0], quant_names[-1]]

    def _get_method_name(self) -> str:
        return "PTQ"
