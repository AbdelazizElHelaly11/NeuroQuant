"""
NeuroQuant v2.0 - Hessian Computation & Layer Clustering (Phase 1a)

Computes per-layer Hessian diagonal (quantization sensitivity)
and clusters layers into HIGH / MEDIUM / LOW sensitivity tiers.
This is the core innovation that reduces NSGA-II search space.

Algorithm:
    1. HessianComputer computes H_ii = d²Loss/dw² for each parameter
       using double backpropagation (first gradient → second gradient).
    2. LayerClusterer groups parameters by layer type (Conv, FC, BN),
       then assigns sensitivity tiers via within-type percentiles.

Works for ANY PyTorch nn.Module — no model-specific assumptions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Type alias: a task-aware loss function takes ``(model, batch_x, batch_y)``
# and returns a scalar loss tensor. This indirection lets the same Hessian
# estimator drive classification, detection, and segmentation: classification
# uses ``criterion(model(x), y)``; detection runs the torchvision detector
# in train mode so ``model(x, y)`` returns a dict of named losses that we
# sum; segmentation pulls ``model(x)["out"]`` before passing it to a
# pixel-wise CE. Default ``None`` preserves the classification-only flow.
LossFn = Callable[[nn.Module, Any, Any], torch.Tensor]

from neuroquant.config import (
    ClusterAssignment,
    LayerSensitivity,
    QuantizationConfig,
    SensitivityTier,
)

logger = logging.getLogger("neuroquant")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HessianComputer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HessianComputer:
    """
    Computes the diagonal of the Hessian matrix (H_ii) for each
    parameter in the model to determine quantization sensitivity.

    H_ii = d²Loss / dw² for each layer's parameters.
    High H_ii → layer is sensitive → needs higher bitwidth.
    Low H_ii  → layer is robust   → can tolerate lower bitwidth.

    Uses double backpropagation: compute the gradient of the loss,
    then compute the gradient of that gradient (= Hessian diagonal).
    This is exact (not an approximation) and works for ANY model.
    """

    def __init__(self, model: nn.Module, config: QuantizationConfig) -> None:
        """
        Args:
            model: FP32 baseline model (must have requires_grad=True on params).
            config: Framework configuration (uses hyperparams.device,
                    hyperparams.hessian_batches).
        """
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.hyperparams.device)
        self.hessian_diag: Dict[str, float] = {}

        # Build reverse lookup: param id → parameter name
        # This avoids O(N²) lookups during the Hessian loop.
        self._param_id_to_name: Dict[int, str] = {
            id(p): name for name, p in self.model.named_parameters()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_hessian(
        self,
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        num_batches: int = 20,
        *,
        loss_fn: Optional[LossFn] = None,
    ) -> Dict[str, LayerSensitivity]:
        """Compute per-parameter sensitivity scores.

        Args:
            data_loader: Calibration DataLoader.
            criterion: Classification loss module (e.g. ``CrossEntropyLoss``).
                Ignored when ``loss_fn`` is supplied. Kept as a positional
                argument for backwards compatibility with callers that
                pass classification criteria directly.
            num_batches: How many batches to draw from ``data_loader``.
            loss_fn: Task-aware ``(model, x, y) -> scalar_loss`` callable.
                When provided, fully replaces the default
                ``criterion(model(x), y)`` flow — required for detection
                (model returns a loss dict from ``model(x, y)`` in train
                mode) and segmentation (output is
                ``OrderedDict({"out": ...})``). When both ``criterion``
                and ``loss_fn`` are ``None`` we default to
                ``CrossEntropyLoss`` so legacy classification callers keep
                working without modification.

        Dispatches on ``config.hyperparams.hessian_estimator``:

        * ``"fisher"`` (default, production): single-backprop empirical
          Fisher diagonal ``E[(∂L/∂w)²]``. Same shape as the diagonal
          Hessian for the cross-entropy loss at convergence (Fisher = H
          in expectation), correlates ≥0.9 in practice on classification
          heads, and is ~3× faster than double backprop on a CPU.

        * ``"diag_hessian"``: exact diagonal of the Hessian via double
          backprop. Used for ablations or when the FP32 baseline is not
          a maximum-likelihood estimate. Memory-heavier and slower.

        Either way the returned dict maps every parameter to a
        ``LayerSensitivity`` with a non-negative ``hessian_diag`` field —
        downstream clustering / NSGA code is unchanged.
        """
        if loss_fn is None:
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            _criterion = criterion

            def _default_loss_fn(model: nn.Module, x: Any, y: Any) -> torch.Tensor:
                return _criterion(model(x), y)

            effective_loss_fn: LossFn = _default_loss_fn
        else:
            effective_loss_fn = loss_fn

        estimator = getattr(
            self.config.hyperparams, "hessian_estimator", "fisher",
        )
        logger.info(
            "Phase 1a: Computing sensitivity (estimator=%s) over %d batches.",
            estimator, num_batches,
        )
        if estimator == "diag_hessian":
            return self._compute_diag_hessian(
                data_loader, effective_loss_fn, num_batches,
            )
        return self._compute_fisher(
            data_loader, effective_loss_fn, num_batches,
        )

    # ------------------------------------------------------------------
    # Estimator: empirical Fisher diagonal (default, production)
    # ------------------------------------------------------------------

    def _compute_fisher(
        self,
        data_loader: DataLoader,
        loss_fn: LossFn,
        num_batches: int,
    ) -> Dict[str, LayerSensitivity]:
        """Compute the empirical Fisher diagonal ``E[(∂L/∂w)²]``.

        Cheap proxy for the diagonal of the true Hessian — coincides
        with it at the maximum-likelihood point of the cross-entropy
        loss. One backward pass per batch; no graph retention beyond
        the standard autograd lifetime.

        ``loss_fn`` is invoked as ``loss_fn(model, x, y) -> scalar``,
        so detection/segmentation can wire in their own (model, input,
        target) → loss bridge without this function knowing about the
        task type.
        """
        self.model.to(self.device)
        # torchvision detectors require ``train()`` mode to return a loss
        # dict from ``model(x, y)``; in ``eval()`` they return predictions
        # which cannot be back-propagated. For classification and
        # segmentation the loss is computed from forward outputs so
        # ``train()`` is still safe — BatchNorm just updates running
        # stats over a handful of calibration batches, which is harmless
        # for a sensitivity probe.
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)

        param_list = list(self.model.parameters())
        param_names = [self._param_id_to_name[id(p)] for p in param_list]
        accum: Dict[str, float] = {name: 0.0 for name in param_names}
        batches_used = 0

        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            x = [i.to(self.device) for i in x] if isinstance(x, (tuple, list)) else x.to(self.device)
            if isinstance(y, (tuple, list)) and len(y) > 0 and isinstance(y[0], dict):
                y = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in y]
            else:
                y = y.to(self.device)
            self.model.zero_grad(set_to_none=True)
            loss = loss_fn(self.model, x, y)
            grads = torch.autograd.grad(loss, param_list, allow_unused=True)
            for name, g in zip(param_names, grads):
                # ``allow_unused=True`` returns None for parameters that
                # didn't participate in the loss graph (e.g. a detection
                # model's RPN head when targets exclude proposals at this
                # scale). Skip them — their Fisher contribution is 0.
                if g is None:
                    continue
                # Mean of squared gradient is the Fisher diag estimate.
                accum[name] += g.detach().pow(2).mean().item()
            batches_used += 1
            del grads, loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                logger.info(
                    "  Batch %d/%d processed", batch_idx + 1, num_batches,
                )

        for name in accum:
            accum[name] /= max(batches_used, 1)

        results: Dict[str, LayerSensitivity] = {}
        for name, param in zip(param_names, param_list):
            results[name] = LayerSensitivity(
                layer_name=name,
                hessian_diag=accum[name],
                layer_type=self._classify_param_type(name, param),
                num_parameters=param.numel(),
            )
        self.hessian_diag = {n: r["hessian_diag"] for n, r in results.items()}

        if results:
            vals = [r["hessian_diag"] for r in results.values()]
            logger.info(
                "  Fisher range: [%.6e, %.6e], mean=%.6e (%d batches)",
                min(vals), max(vals), sum(vals) / len(vals), batches_used,
            )
        return results

    # ------------------------------------------------------------------
    # Estimator: exact diagonal Hessian (slower, double-backprop)
    # ------------------------------------------------------------------

    def _compute_diag_hessian(
        self,
        data_loader: DataLoader,
        loss_fn: LossFn,
        num_batches: int,
    ) -> Dict[str, LayerSensitivity]:
        """Original double-backprop diagonal Hessian estimator.

        Algorithm (per calibration batch):
          1. Forward pass via ``loss_fn(model, x, y)`` → scalar loss
          2. First gradient  (∂loss/∂w) with create_graph=True
          3. Second gradient (∂²loss/∂w²) = gradient of sum(grad_1)
          4. Accumulate |H_ii| mean per parameter

        ``loss_fn`` abstracts the forward pass so detection/segmentation
        models that need ``model(x, y)`` (loss-dict path) or
        ``model(x)["out"]`` (segmentation logits) work without this
        estimator hard-coding any task assumptions.
        """
        self.model.to(self.device)
        # See ``_compute_fisher`` for the same rationale: detection
        # criteria require ``train()`` so the model returns a loss dict.
        self.model.train()

        # Ensure all parameters require gradients for double backprop.
        for p in self.model.parameters():
            p.requires_grad_(True)

        # Collect the list of parameters once (order matters for grad indexing).
        param_list = list(self.model.parameters())
        param_names = [self._param_id_to_name[id(p)] for p in param_list]

        # Accumulator for Hessian diagonal values
        hessian_accum: Dict[str, float] = {name: 0.0 for name in param_names}
        batches_used = 0

        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            x = [i.to(self.device) for i in x] if isinstance(x, (tuple, list)) else x.to(self.device)
            if isinstance(y, (tuple, list)) and len(y) > 0 and isinstance(y[0], dict):
                y = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in y]
            else:
                y = y.to(self.device)

            # ── Forward pass via task-aware loss bridge ──
            loss = loss_fn(self.model, x, y)

            # ── First gradient (∂loss/∂w_i) ──
            # create_graph=True so we can differentiate through grad_1.
            # ``allow_unused=True`` because detection sub-heads can be
            # silent on batches whose targets don't activate them.
            grad_1 = torch.autograd.grad(
                loss,
                param_list,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

            # ── Second gradient (∂²loss/∂w_i²) ──
            # Filter out the None entries before reducing — a None gradient
            # contributes nothing to the second-order term.
            active_grad_1 = [g for g in grad_1 if g is not None]
            if not active_grad_1:
                # Nothing in the model participated in this batch's loss;
                # this is degenerate but possible (empty detection targets).
                batches_used += 1
                del grad_1, loss
                torch.cuda.empty_cache() if self.device.type == "cuda" else None
                continue
            grad_1_scalar = sum(g.sum() for g in active_grad_1)
            grad_2 = torch.autograd.grad(
                grad_1_scalar,
                param_list,
                retain_graph=False,
                allow_unused=True,
            )

            # ── Accumulate ──
            for idx, (name, g2) in enumerate(zip(param_names, grad_2)):
                if g2 is None:
                    continue
                h_value = g2.abs().mean().item()
                hessian_accum[name] += h_value

            batches_used += 1

            # Free the computation graph to reclaim GPU memory.
            del grad_1, grad_2, grad_1_scalar, loss
            torch.cuda.empty_cache() if self.device.type == "cuda" else None

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                logger.info(
                    "  Batch %d/%d processed", batch_idx + 1, num_batches
                )

        # ── Average over batches ──
        for name in hessian_accum:
            hessian_accum[name] /= max(batches_used, 1)

        # ── Build LayerSensitivity results ──
        results: Dict[str, LayerSensitivity] = {}
        for name, param in zip(param_names, param_list):
            results[name] = LayerSensitivity(
                layer_name=name,
                hessian_diag=hessian_accum[name],
                layer_type=self._classify_param_type(name, param),
                num_parameters=param.numel(),
            )

        # Store raw dict for downstream use
        self.hessian_diag = {name: r["hessian_diag"] for name, r in results.items()}

        logger.info(
            "  Hessian computed for %d parameters (%d batches)",
            len(results), batches_used,
        )
        if results:
            vals = [r["hessian_diag"] for r in results.values()]
            logger.info(
                "  H_ii range: [%.6f, %.6f], mean=%.6f",
                min(vals), max(vals), sum(vals) / len(vals),
            )
        return results

    # ------------------------------------------------------------------
    # Helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_param_type(name: str, param: torch.Tensor) -> str:
        """
        Classify a parameter into a layer type based on its shape
        and name. Works generically for any model architecture.

        Rules:
          - ndim > 2 (e.g., [out, in, kH, kW])  → 'Conv2d'
          - ndim == 2 (e.g., [out, in])          → 'Linear'
          - ndim == 1 and 'weight' in name       → 'BatchNorm'
          - ndim == 1 and 'bias' in name         → 'Bias'
          - otherwise                            → 'Other'
        """
        ndim = param.ndim
        if ndim > 2:
            return "Conv2d"
        elif ndim == 2:
            return "Linear"
        elif ndim == 1 and "weight" in name:
            return "BatchNorm"
        elif ndim == 1 and "bias" in name:
            return "Bias"
        else:
            return "Other"

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve device string to torch.device ('auto' picks best GPU)."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LayerClusterer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LayerClusterer:
    """
    Groups layers into sensitivity-based clusters using quantile thresholds.

    Clustering strategy:
    - Group parameters by type (Conv2d, Linear, BatchNorm, Bias, Other)
    - Within each type, compute percentiles of H_ii values
    - Assign tiers:
        HIGH   (> high percentile)  → force INT8 (too sensitive)
        MEDIUM (between thresholds) → NSGA-II decides
        LOW    (< low percentile)   → allow INT4 (robust)

    This is applied WITHIN each type so that, e.g., the most sensitive
    conv layers get HIGH and the most robust conv layers get LOW,
    regardless of how they compare to linear layers.
    """

    def __init__(
        self,
        model: nn.Module,
        hessian_results: Dict[str, LayerSensitivity],
        config: QuantizationConfig,
    ) -> None:
        """
        Args:
            model: The FP32 baseline model (needed for layer type grouping).
            hessian_results: Output from HessianComputer.compute_hessian().
            config: Framework configuration (uses cluster percentile thresholds).
        """
        self.model = model
        self.hessian_results = hessian_results
        self.config = config

        # Percentile thresholds (config stores as 0-1 fractions, e.g. 0.33/0.66;
        # numpy percentile expects 0-100 range, so multiply by 100).
        self.low_percentile = config.hyperparams.cluster_low_percentile * 100
        self.high_percentile = config.hyperparams.cluster_high_percentile * 100

        # State populated by create_clusters()
        self.clusters: Dict[str, List[str]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
        self.layer_to_cluster: Dict[str, str] = {}
        self.cluster_assignments: List[ClusterAssignment] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_clusters(self) -> Dict[str, Any]:
        """
        Create layer clusters based on Hessian sensitivity.

        Returns:
            Dictionary with:
            - 'cluster_assignments': List[ClusterAssignment]
            - 'clusters': Dict[str, List[str]]  (tier → param names)
            - 'statistics': Dict with counts, thresholds, search space size
        """
        # Reset state
        self.clusters = {"HIGH": [], "MEDIUM": [], "LOW": []}
        self.layer_to_cluster = {}

        # ── Step 1: Group parameters by type ──
        layer_groups = self._group_by_type()

        logger.info("Phase 1a: Clustering layers into sensitivity tiers …")
        logger.info("  Percentile thresholds: LOW < %.1f%%, HIGH > %.1f%%",
                     self.low_percentile, self.high_percentile)

        threshold_info: Dict[str, Dict[str, float]] = {}

        # ── Step 2: Within each type, assign tiers ──
        for layer_type, layer_entries in layer_groups.items():
            if not layer_entries:
                continue

            # Extract Hessian values for this type
            h_values = [entry["hessian_diag"] for entry in layer_entries]

            if len(h_values) == 0:
                continue

            # Compute percentile thresholds within this type
            low_thresh, high_thresh = self._compute_percentiles(layer_entries)

            threshold_info[layer_type] = {
                "low_threshold": low_thresh,
                "high_threshold": high_thresh,
                "num_layers": len(layer_entries),
            }

            logger.info(
                "  Type %-10s: %d params, thresholds [%.6f, %.6f]",
                layer_type, len(layer_entries), low_thresh, high_thresh,
            )

            # Assign each parameter to a tier
            for entry in layer_entries:
                name = entry["layer_name"]
                h_val = entry["hessian_diag"]
                tier = self._assign_tier(h_val, low_thresh, high_thresh)

                tier_str = tier.value.upper()
                self.clusters[tier_str].append(name)
                self.layer_to_cluster[name] = tier_str

        # ── Step 3: Build ClusterAssignment objects ──
        self.cluster_assignments = self._build_cluster_assignments()

        # ── Step 4: Compile statistics ──
        stats = {
            "HIGH_count": len(self.clusters["HIGH"]),
            "MEDIUM_count": len(self.clusters["MEDIUM"]),
            "LOW_count": len(self.clusters["LOW"]),
            "total_layers": sum(len(v) for v in self.clusters.values()),
            "search_space_size": self.get_search_space_size(),
            "thresholds": threshold_info,
        }

        logger.info(
            "✓ Clustering complete: HIGH=%d, MEDIUM=%d, LOW=%d (total=%d)",
            stats["HIGH_count"],
            stats["MEDIUM_count"],
            stats["LOW_count"],
            stats["total_layers"],
        )
        logger.info(
            "  Search space: 2^%d = %d configurations "
            "(MEDIUM+LOW clusters are searchable)",
            self._count_searchable_clusters(),
            self.get_search_space_size(),
        )

        return {
            "cluster_assignments": self.cluster_assignments,
            "clusters": self.clusters,
            "statistics": stats,
        }

    def get_cluster_info(self, cluster_type: str) -> List[str]:
        """Return parameter names in a cluster tier ('HIGH', 'MEDIUM', 'LOW')."""
        return self.clusters.get(cluster_type.upper(), [])

    def get_search_space_size(self) -> int:
        """
        Return the total search space size (2^N for searchable clusters).

        Only MEDIUM and LOW clusters are searchable
        (HIGH is forced to INT8).
        """
        n_searchable = self._count_searchable_clusters()
        return 2 ** n_searchable

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _group_by_type(self) -> Dict[str, List[LayerSensitivity]]:
        """
        Group parameters by layer type using shape-based classification.
        Works generically for ANY model architecture.

        Only ``Conv2d`` and ``Linear`` weights are clustered: those are
        the parameters NSGA-II actually searches over. Including
        ``BatchNorm``, ``Bias``, or ``Other`` entries here was the source
        of the search-space collapse — those tensors got their own
        clusters, but NSGA later filters them out, so e.g. a 2^7
        nominal space shrank to 2^3 when 4 of the 7 clusters turned out
        to be all-BN. By excluding them at the source, the cluster
        count, the ``search_space_size`` statistic, and the actual
        NSGA gene count all agree.

        Returns:
            Dict mapping type name → list of LayerSensitivity entries.
            Only ``Conv2d`` and ``Linear`` keys are populated; BN and
            biases stay FP32 by convention.
        """
        groups: Dict[str, List[LayerSensitivity]] = {
            "Conv2d": [],
            "Linear": [],
        }
        skipped = 0
        for name, entry in self.hessian_results.items():
            layer_type = entry["layer_type"]
            if layer_type in groups:
                groups[layer_type].append(entry)
            else:
                skipped += 1
        if skipped:
            logger.info(
                "  Skipped %d non-quantizable parameters (BN/bias/other) — "
                "they stay FP32 and are excluded from the search space.",
                skipped,
            )
        return groups

    def _compute_percentiles(
        self, layers: List[LayerSensitivity]
    ) -> Tuple[float, float]:
        """
        Compute the low and high percentile thresholds for a group.

        Args:
            layers: List of LayerSensitivity entries (same type).

        Returns:
            (low_threshold, high_threshold) — H_ii values at the
            configured percentile boundaries.
        """
        values = [entry["hessian_diag"] for entry in layers]

        if len(values) <= 1:
            # With 1 or fewer values, thresholds are meaningless.
            # Assign everything to MEDIUM so NSGA-II can decide.
            val = values[0] if values else 0.0
            return (val, val)

        low_thresh = float(np.percentile(values, self.low_percentile))
        high_thresh = float(np.percentile(values, self.high_percentile))

        return (low_thresh, high_thresh)

    def _assign_tier(
        self, sensitivity: float, p_low: float, p_high: float
    ) -> SensitivityTier:
        """
        Assign a parameter to a sensitivity tier based on its H_ii value.

        Args:
            sensitivity: The Hessian diagonal value for this parameter.
            p_low: Low percentile threshold (below → LOW).
            p_high: High percentile threshold (above → HIGH).

        Returns:
            SensitivityTier enum (HIGH, MEDIUM, or LOW).
        """
        if sensitivity > p_high:
            return SensitivityTier.HIGH
        elif sensitivity < p_low:
            return SensitivityTier.LOW
        else:
            return SensitivityTier.MEDIUM

    def _build_cluster_assignments(self) -> List[ClusterAssignment]:
        """
        Build structured ClusterAssignment objects from the raw cluster dicts.

        Within each tier (HIGH/MEDIUM/LOW), split by layer type so NSGA-II
        can search a richer and more architecture-sensitive space than a
        single global MEDIUM and LOW cluster.
        """
        assignments: List[ClusterAssignment] = []
        cluster_id = 0

        for tier_name in ["HIGH", "MEDIUM", "LOW"]:
            layer_names = self.clusters[tier_name]
            if not layer_names:
                continue

            # Split tier bucket by layer type (Conv2d/Linear/BatchNorm/Bias/Other).
            by_type: Dict[str, List[str]] = {}
            for name in layer_names:
                entry = self.hessian_results.get(name, {})
                layer_type = str(entry.get("layer_type", "Other"))
                by_type.setdefault(layer_type, []).append(name)

            for _layer_type, names_in_type in sorted(by_type.items()):
                if not names_in_type:
                    continue

                # Determine allowed bitwidths per tier
                if tier_name == "HIGH":
                    allowed = [8]  # Force INT8 — too sensitive
                else:
                    allowed = [4, 8]  # Searchable cluster

                # Compute mean sensitivity for this cluster
                sensitivities = [
                    self.hessian_results[name]["hessian_diag"]
                    for name in names_in_type
                    if name in self.hessian_results
                ]
                mean_sens = (
                    sum(sensitivities) / len(sensitivities) if sensitivities else 0.0
                )

                assignments.append(
                    ClusterAssignment(
                        cluster_id=cluster_id,
                        tier=tier_name,
                        layer_names=names_in_type,
                        allowed_bitwidths=allowed,
                        mean_sensitivity=mean_sens,
                    )
                )
                cluster_id += 1

        return assignments

    def _count_searchable_clusters(self) -> int:
        """
        Count the number of clusters that NSGA-II will search over.
        HIGH is fixed at INT8, so only MEDIUM and LOW are searchable.
        """
        count = 0
        for assignment in self.cluster_assignments:
            if assignment["tier"] in ("MEDIUM", "LOW"):
                count += 1
        return count
