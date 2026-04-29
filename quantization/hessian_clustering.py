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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
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
        criterion: nn.Module,
        num_batches: int = 20,
    ) -> Dict[str, LayerSensitivity]:
        """
        Compute Hessian diagonal for all parameters.

        Algorithm (per calibration batch):
          1. Forward pass  → loss
          2. First gradient  (∂loss/∂w) with create_graph=True
          3. Second gradient (∂²loss/∂w²) = gradient of sum(grad_1)
          4. Accumulate |H_ii| mean per parameter

        Args:
            data_loader: Calibration data loader.
            criterion: Loss function (e.g., CrossEntropyLoss).
            num_batches: Number of calibration batches to use.

        Returns:
            Dictionary mapping parameter_name → LayerSensitivity.
        """
        self.model.to(self.device)
        self.model.eval()  # BN in eval mode, but grads still flow

        # Ensure all parameters require gradients for double backprop.
        for p in self.model.parameters():
            p.requires_grad_(True)

        # Collect the list of parameters once (order matters for grad indexing).
        param_list = list(self.model.parameters())
        param_names = [self._param_id_to_name[id(p)] for p in param_list]

        # Accumulator for Hessian diagonal values
        hessian_accum: Dict[str, float] = {name: 0.0 for name in param_names}
        batches_used = 0

        logger.info(
            "Phase 1a: Computing Hessian diagonal for %d parameters "
            "using %d calibration batches …",
            len(param_list),
            num_batches,
        )

        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            # ── Forward pass ──
            output = self.model(x)
            loss = criterion(output, y)

            # ── First gradient (∂loss/∂w_i) ──
            # create_graph=True so we can differentiate through grad_1.
            grad_1 = torch.autograd.grad(
                loss,
                param_list,
                create_graph=True,
                retain_graph=True,
            )

            # ── Second gradient (∂²loss/∂w_i²) ──
            # sum(g.sum()) aggregates all first-order gradients into a scalar;
            # differentiating that scalar w.r.t. each param gives H_ii.
            grad_1_scalar = sum(g.sum() for g in grad_1)
            grad_2 = torch.autograd.grad(
                grad_1_scalar,
                param_list,
                retain_graph=False,
            )

            # ── Accumulate ──
            for idx, (name, g2) in enumerate(zip(param_names, grad_2)):
                h_value = g2.abs().mean().item()
                hessian_accum[name] += h_value

            batches_used += 1

            # Free the computation graph to reclaim GPU memory.
            del grad_1, grad_2, grad_1_scalar, loss, output
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
            "✓ Hessian computed for %d parameters (%d batches)",
            len(results),
            batches_used,
        )

        # Log a brief summary of sensitivity range
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

        Returns:
            Dict mapping type name → list of LayerSensitivity entries.
            Types: 'Conv2d', 'Linear', 'BatchNorm', 'Bias', 'Other'
        """
        groups: Dict[str, List[LayerSensitivity]] = {
            "Conv2d": [],
            "Linear": [],
            "BatchNorm": [],
            "Bias": [],
            "Other": [],
        }

        for name, entry in self.hessian_results.items():
            layer_type = entry["layer_type"]
            if layer_type in groups:
                groups[layer_type].append(entry)
            else:
                groups["Other"].append(entry)

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
        Groups layers within each tier and computes mean sensitivity.
        """
        assignments: List[ClusterAssignment] = []
        cluster_id = 0

        for tier_name in ["HIGH", "MEDIUM", "LOW"]:
            layer_names = self.clusters[tier_name]
            if not layer_names:
                continue

            # Determine allowed bitwidths per tier
            if tier_name == "HIGH":
                allowed = [8]  # Force INT8 — too sensitive
            elif tier_name == "MEDIUM":
                allowed = [4, 8]  # NSGA-II searches
            else:  # LOW
                allowed = [4, 8]  # NSGA-II searches (INT4 preferred)

            # Compute mean sensitivity for this cluster
            sensitivities = [
                self.hessian_results[name]["hessian_diag"]
                for name in layer_names
                if name in self.hessian_results
            ]
            mean_sens = (
                sum(sensitivities) / len(sensitivities) if sensitivities else 0.0
            )

            assignments.append(
                ClusterAssignment(
                    cluster_id=cluster_id,
                    tier=tier_name,
                    layer_names=layer_names,
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
