"""
NeuroQuant v2.0 - FITCompress Warm-Start Seed Generation (Phase 1b)

Generates an elite seed configuration for NSGA-II using the
Fisher Information Trace (FIT) metric from the PQuantML paper.

Algorithm:
    1. Compute FIT per layer:  FIT_i = H_ii × ||W_i||²_F
       (Hessian sensitivity × weight magnitude = compressibility score)
    2. Normalize FIT scores to [0, 1]
    3. Quantile-based assignment:
       - HIGH FIT (>75th percentile)  → INT8 (preserve accuracy)
       - LOW FIT  (<25th percentile)  → INT4 (compress aggressively)
       - MEDIUM   (25-75th)           → INT8 (safe default)
    4. Validate against Phase 1a cluster tier constraints
    5. Return 1 elite seed to warm-start NSGA-II population

Design note:
    Our Phase 1a HessianComputer stores H_ii as a scalar (mean |∂²L/∂w²|)
    per parameter, not a per-element tensor. So the FIT formula is adapted:
        FIT_i = H_ii_scalar × Σ(W_i²)
    This preserves the same ranking: layers with high sensitivity AND large
    weights get high FIT scores (= hard to compress).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import (
    ClusterAssignment,
    FITCompressSeed,
    LayerSensitivity,
    QuantizationConfig,
)

logger = logging.getLogger("neuroquant")


class FITCompressSeedGenerator:
    """
    Generates an elite seed configuration using Fisher Information Trace.

    FIT captures both loss-sensitivity (Hessian) and weight magnitude,
    giving a more informed compressibility score than either alone.

    Pipeline:
        1. compute_fit_scores()      → raw FIT per weight parameter
        2. normalize_fit_scores()    → FIT in [0, 1]
        3. assign_seed_bitwidths()   → percentile-based INT4/INT8
        4. validate_against_clusters()→ enforce Phase 1a tier constraints
        5. generate_seed()           → full pipeline, returns FITCompressSeed
    """

    def __init__(
        self,
        model: nn.Module,
        hessian_results: Dict[str, LayerSensitivity],
        cluster_result: Dict[str, Any],
        config: QuantizationConfig,
    ) -> None:
        """
        Args:
            model: FP32 baseline model.
            hessian_results: Output from HessianComputer.compute_hessian().
                Dict mapping param_name → LayerSensitivity.
            cluster_result: Output from LayerClusterer.create_clusters().
                Dict with 'clusters', 'cluster_assignments', 'statistics'.
            config: Framework configuration (uses fit_low/high_percentile).
        """
        self.model = model
        self.hessian_results = hessian_results
        self.cluster_result = cluster_result
        self.config = config

        # Build mapping: param_name → tier string ('HIGH', 'MEDIUM', 'LOW')
        # from the cluster result's clusters dict.
        self._param_to_tier: Dict[str, str] = {}
        for tier, param_names in cluster_result["clusters"].items():
            for pname in param_names:
                self._param_to_tier[pname] = tier

        # Build mapping: module_name → list of param names in that module,
        # so we can link module.weight to hessian_results.
        self._module_to_params: Dict[str, List[str]] = {}
        for pname in self.hessian_results:
            # "features.3.conv.0.weight" → module = "features.3.conv.0"
            parts = pname.rsplit(".", 1)
            module_name = parts[0] if len(parts) > 1 else pname
            self._module_to_params.setdefault(module_name, []).append(pname)

    # ------------------------------------------------------------------
    # Step 1: Compute FIT scores
    # ------------------------------------------------------------------

    def compute_fit_scores(self) -> Dict[str, float]:
        """
        Compute Fisher Information Trace per weight parameter.

        FIT_i = H_ii × sum(W_i²)

        Only computes FIT for weight parameters of Conv2d and Linear layers
        (these are the quantizable parameters).

        Returns:
            fit_scores: {param_name → raw FIT score (float)}
        """
        fit_scores: Dict[str, float] = {}

        for name, param in self.model.named_parameters():
            # Only process weight parameters (not biases, BN params)
            if name not in self.hessian_results:
                continue

            layer_info = self.hessian_results[name]

            # Only quantize Conv2d and Linear weights
            if layer_info["layer_type"] not in ("Conv2d", "Linear"):
                continue

            # H_ii = scalar mean Hessian diagonal for this parameter
            h_ii = layer_info["hessian_diag"]

            # sum(W²) = Frobenius norm squared
            w_squared_sum = (param.data ** 2).sum().item()

            # FIT = H_ii × ||W||²_F
            fit = h_ii * w_squared_sum
            fit_scores[name] = max(fit, 1e-10)  # prevent zero

        logger.info(
            "  FIT computed for %d quantizable weight parameters", len(fit_scores)
        )

        return fit_scores

    # ------------------------------------------------------------------
    # Step 2: Normalize
    # ------------------------------------------------------------------

    def normalize_fit_scores(
        self, fit_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize FIT scores to [0, 1] range.

        normalized_i = fit_i / max(fit_all)

        Args:
            fit_scores: Raw FIT scores from compute_fit_scores().

        Returns:
            Normalized FIT scores: {param_name → score in [0, 1]}
        """
        if not fit_scores:
            return {}

        max_fit = max(fit_scores.values())
        if max_fit <= 0:
            max_fit = 1.0

        normalized = {name: score / max_fit for name, score in fit_scores.items()}

        logger.info(
            "  FIT normalized: min=%.4f, max=%.4f, mean=%.4f",
            min(normalized.values()),
            max(normalized.values()),
            sum(normalized.values()) / len(normalized),
        )

        return normalized

    # ------------------------------------------------------------------
    # Step 3: Assign bitwidths
    # ------------------------------------------------------------------

    def assign_seed_bitwidths(
        self, fit_normalized: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Assign bitwidths based on FIT percentile thresholds.

        Strategy:
            - HIGH FIT (> high_percentile)  → INT8 (preserve, hard to compress)
            - LOW FIT  (< low_percentile)   → INT4 (compress, easy to compress)
            - MEDIUM                        → INT8 (safe default)

        Args:
            fit_normalized: Normalized FIT scores {param_name → [0, 1]}.

        Returns:
            seed_config: {param_name → bitwidth (4 or 8)}
        """
        if not fit_normalized:
            return {}

        low_pctl = self.config.hyperparams.fit_low_percentile
        high_pctl = self.config.hyperparams.fit_high_percentile

        # Compute percentile thresholds from the actual FIT distribution
        values = sorted(fit_normalized.values())
        n = len(values)

        low_idx = max(0, min(int(n * low_pctl), n - 1))
        high_idx = max(0, min(int(n * high_pctl), n - 1))

        low_threshold = values[low_idx]
        high_threshold = values[high_idx]

        logger.info(
            "  FIT percentile thresholds: LOW < %.4f (p%.0f), HIGH > %.4f (p%.0f)",
            low_threshold, low_pctl * 100, high_threshold, high_pctl * 100,
        )

        seed_config: Dict[str, int] = {}
        for name, fit_score in fit_normalized.items():
            if fit_score > high_threshold:
                seed_config[name] = 8   # HIGH FIT → preserve
            elif fit_score < low_threshold:
                seed_config[name] = 4   # LOW FIT → compress
            else:
                seed_config[name] = 8   # MEDIUM → safe default

        n_int4 = sum(1 for bw in seed_config.values() if bw == 4)
        n_int8 = sum(1 for bw in seed_config.values() if bw == 8)
        logger.info("  Seed assignment: INT4=%d, INT8=%d layers", n_int4, n_int8)

        return seed_config

    # ------------------------------------------------------------------
    # Step 4: Validate against cluster constraints
    # ------------------------------------------------------------------

    def validate_against_clusters(
        self, seed_config: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Validate seed bitwidths against Phase 1a cluster tier constraints.

        Rules:
            - HIGH tier → MUST be INT8 (override any INT4 assignment)
            - MEDIUM/LOW tier → flexible (INT4 or INT8 allowed)

        Args:
            seed_config: Raw seed from assign_seed_bitwidths().

        Returns:
            Validated seed config with tier constraints enforced.
        """
        validated = seed_config.copy()
        overrides = 0

        for name, bitwidth in validated.items():
            tier = self._param_to_tier.get(name)
            if tier == "HIGH" and bitwidth != 8:
                logger.info(
                    "  [Validation] %s: %d-bit -> 8-bit (HIGH tier constraint)",
                    name, bitwidth,
                )
                validated[name] = 8
                overrides += 1

        if overrides > 0:
            logger.info("  %d bitwidth(s) overridden by cluster constraints", overrides)
        else:
            logger.info("  No overrides needed - seed is cluster-compatible")

        return validated

    # ------------------------------------------------------------------
    # Step 5: Compute compression potential
    # ------------------------------------------------------------------

    def compute_compression_potential(
        self, seed_config: Dict[str, int]
    ) -> float:
        """
        Estimate memory savings of the seed config vs FP32 baseline.

        Uses the EBops formula: sum(params × bitwidth) / 8
        Compares quantized EBops to FP32 EBops.

        Args:
            seed_config: {param_name → bitwidth}

        Returns:
            Compression potential as percentage [0, 100].
        """
        ebops_fp32 = 0.0
        ebops_quantized = 0.0

        for name, param in self.model.named_parameters():
            numel = param.numel()
            ebops_fp32 += numel * 32  # FP32 = 32 bits

            # Use seed bitwidth if available, else FP32 (unquantized)
            bitwidth = seed_config.get(name, 32)
            ebops_quantized += numel * bitwidth

        if ebops_fp32 == 0:
            return 0.0

        compression = (1.0 - ebops_quantized / ebops_fp32) * 100.0
        return max(compression, 0.0)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def generate_seed(self) -> FITCompressSeed:
        """
        Run the full FITCompress pipeline.

        Returns:
            FITCompressSeed with seed_config, fit_scores,
            compression_potential, and elite_status.
        """
        logger.info("=" * 70)
        logger.info("Phase 1b: FITCompress - Generating Elite Seed Configuration")
        logger.info("=" * 70)

        # Step 1: Compute FIT
        logger.info("Step 1/5: Computing Fisher Information Trace per layer ...")
        fit_scores = self.compute_fit_scores()

        # Step 2: Normalize
        logger.info("Step 2/5: Normalizing FIT scores ...")
        fit_normalized = self.normalize_fit_scores(fit_scores)

        # Step 3: Assign bitwidths
        logger.info("Step 3/5: Assigning seed bitwidths via FIT percentiles ...")
        seed_config = self.assign_seed_bitwidths(fit_normalized)

        # Step 4: Validate against clusters
        logger.info("Step 4/5: Validating against cluster tier constraints ...")
        validated_config = self.validate_against_clusters(seed_config)

        # Step 5: Compression potential
        logger.info("Step 5/5: Computing compression potential ...")
        compression = self.compute_compression_potential(validated_config)
        elite_status = "HIGH_POTENTIAL" if compression > 30 else "BALANCED"
        logger.info(
            "  Compression: %.1f%% vs FP32, status: %s",
            compression, elite_status,
        )

        # Build result
        seed = FITCompressSeed(
            seed_config=validated_config,
            fit_scores=fit_normalized,
            compression_potential=compression,
            elite_status=elite_status,
        )

        # Log summary
        n_int4 = sum(1 for bw in validated_config.values() if bw == 4)
        n_int8 = sum(1 for bw in validated_config.values() if bw == 8)
        total = len(validated_config)

        logger.info("-" * 70)
        logger.info("FITCompress Seed Summary:")
        logger.info("  Total quantizable layers: %d", total)
        logger.info("  INT4: %d (%.1f%%)", n_int4, 100 * n_int4 / max(total, 1))
        logger.info("  INT8: %d (%.1f%%)", n_int8, 100 * n_int8 / max(total, 1))
        logger.info("  Compression potential: %.1f%%", compression)
        logger.info("  Elite status: %s", elite_status)
        logger.info("=" * 70)

        return seed
