"""
NeuroQuant v2.0 - NSGA-II Cluster-Level Search (Phase 1c)

Multi-objective evolutionary optimization at the CLUSTER level
instead of the layer level. This is the key speedup innovation:
search space is 2^N (searchable clusters) vs 2^L (all layers).

Key design decisions:
    - Only MEDIUM and LOW clusters are searchable (HIGH = fixed INT8)
    - Individuals encode bitwidths as binary genes (0=INT4, 1=INT8)
    - Fake quantization used for fast evaluation during search
    - Proper NSGA-II: non-dominated sorting + crowding distance,
      generalized to N objectives so the same code handles the 2-obj
      ``[acc_loss, size]`` and 3-obj ``[acc_loss, size, latency]``
      cases. The 3-objective mode is what wave J4 calls
      "hardware-aware search" — pass a per-layer latency LUT and the
      search will pick configurations that are also fast on the
      deployment runtime, not just compressed.
    - Warm-started with FITCompress elite seed from Phase 1b

Objectives (all minimised):
    1. Accuracy loss = FP32_accuracy - quantized_accuracy
    2. Model size (MiB) = sum(params x bitwidth) / 8 / (1024²)
    3. Latency (ms) = Σ over all layers of LUT[param][bitwidth].
       Present only when ``latency_lut`` is supplied at construction;
       otherwise the search runs in 2-objective mode.
"""

from __future__ import annotations

import copy
import itertools
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    ClusterAssignment,
    ParetoFront,
    ParetoSolution,
    QuantizationConfig,
)
from utils.common import model_size_mb_from_bytes
from utils.numerics import MIN_SCALE

logger = logging.getLogger("neuroquant")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lightweight fake-quantization for fast NSGA-II evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _fake_quantize_tensor(tensor: torch.Tensor, bitwidth: int) -> torch.Tensor:
    """
    Fake-quantize a tensor to the specified bitwidth (symmetric).

    Simulates quantization by: quantize → dequantize (stored as FP32).
    This introduces realistic quantization noise without needing
    actual INT kernels.

    Args:
        tensor: FP32 weight tensor.
        bitwidth: Target bitwidth (4 or 8).

    Returns:
        Fake-quantized tensor (same shape, FP32 dtype).
    """
    if bitwidth >= 32:
        return tensor  # No quantization

    qmax = 2 ** (bitwidth - 1) - 1
    scale = tensor.abs().max() / max(qmax, 1)
    scale = max(scale.item(), MIN_SCALE)

    quantized = (tensor / scale).round().clamp(-qmax - 1, qmax)
    return quantized * scale


def _apply_fake_quantization(
    model: nn.Module,
    bitwidth_config: Dict[str, int],
) -> nn.Module:
    """
    Apply fake quantization to all weight parameters in the model.

    Args:
        model: Model to fake-quantize (modified in-place).
        bitwidth_config: {param_name -> bitwidth (4, 8, or 32)}.

    Returns:
        The model with fake-quantized weights.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            bitwidth = bitwidth_config.get(name, 32)
            if bitwidth < 32 and "weight" in name:
                param.data = _fake_quantize_tensor(param.data, bitwidth)
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NSGAIIClusterSearch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class NSGAIIClusterSearch:
    """
    NSGA-II multi-objective optimizer operating on cluster-level bitwidths.

    Instead of searching over every layer independently (2^L configs),
    we search over SEARCHABLE clusters only (2^N configs, N << L).
    HIGH-tier clusters are fixed at INT8 — only MEDIUM and LOW are
    encoded in the GA individual.

    Warm-started with FITCompress elite seed from Phase 1b.
    """

    def __init__(
        self,
        model: nn.Module,
        cluster_assignments: List[ClusterAssignment],
        config: QuantizationConfig,
        *,
        latency_lut: Optional[Dict[str, Dict[int, float]]] = None,
        hessian_diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model: FP32 baseline model (will be cloned for each evaluation).
            cluster_assignments: From Phase 1a LayerClusterer.create_clusters().
            config: Framework configuration (uses nsga_* hyperparameters).
            latency_lut: Optional per-layer latency table built by
                ``quantization.latency_lut.build_latency_lut`` — when
                provided, the search adds a third objective
                ``latency_ms`` so candidates that are equally accurate
                and equally compressed are discriminated by their
                deployment-runtime latency. If None, the search keeps
                its original 2-objective behaviour and remains
                backwards-compatible.
        """
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.hyperparams.device)
        self._latency_lut: Optional[Dict[str, Dict[int, float]]] = latency_lut
        self._num_objectives: int = 3 if latency_lut else 2
        # Cache the FP32 latency from the LUT (for ``latency_reduction``
        # diagnostics on each Pareto solution).
        self._fp32_latency_ms: float = 0.0
        if latency_lut:
            from quantization.latency_lut import latency_for_assignment
            self._fp32_latency_ms = latency_for_assignment(
                {pname: 32 for pname in latency_lut}, latency_lut,
            )

        # Build the canonical set of TRULY quantizable parameters (the
        # weight tensors of Conv2d / Linear modules). Anything else
        # — BatchNorm γ/β, biases, embeddings — must never appear in a
        # searchable cluster, must never get a gene, and must never be
        # accounted for in the size objective. Doing so was the source
        # of pollution in the prior implementation.
        self._quantizable_weights: set = self._build_quantizable_weight_set(model)

        # Filter every incoming cluster down to the quantizable set, then
        # split into fixed (HIGH) and searchable (MEDIUM/LOW). Empty
        # clusters after filtering are dropped — they can't contribute to
        # the search.
        self.fixed_clusters: List[ClusterAssignment] = []
        self.searchable_clusters: List[ClusterAssignment] = []

        dropped_layers: List[str] = []
        for ca in cluster_assignments:
            keep = [n for n in ca.get("layer_names", [])
                    if n in self._quantizable_weights]
            dropped_layers.extend(
                n for n in ca.get("layer_names", []) if n not in self._quantizable_weights
            )
            if not keep:
                continue
            filtered_ca: ClusterAssignment = {
                **ca,
                "layer_names": keep,
            }
            if ca["tier"] == "HIGH":
                self.fixed_clusters.append(filtered_ca)
            else:
                self.searchable_clusters.append(filtered_ca)

        if dropped_layers:
            logger.info(
                "NSGA-II: dropped %d non-quantizable params from clusters "
                "(BN/bias/embeddings stay FP32).",
                len(dropped_layers),
            )

        # ── Search-mode dispatch ──
        # ``per_layer`` (HAWQ-V3 / HAQ): one binary gene per Conv/Linear
        # weight. ``cluster``: legacy gene-per-tier encoding kept for
        # paper-baseline ablations. The decision happens here so the
        # rest of the class can be agnostic — the only contract the
        # encoder/decoder enforces is ``num_genes == len(self._gene_layers)``.
        hp = config.hyperparams
        self._search_mode = str(getattr(hp, "nsga_search_mode", "per_layer")).lower()

        if self._search_mode == "per_layer":
            # Every quantizable weight gets its own gene. Order is
            # stable: parameters appear in the order given by
            # ``model.named_parameters()`` so resume/checkpoint paths
            # stay deterministic.
            self._gene_layers: List[str] = [
                pname for pname, _ in model.named_parameters()
                if pname in self._quantizable_weights
            ]
        else:
            self._gene_layers = []  # cluster mode keeps using searchable_clusters

        self.num_genes = (
            len(self._gene_layers)
            if self._search_mode == "per_layer"
            else len(self.searchable_clusters)
        )

        # Build the fixed part of the bitwidth config (HIGH → INT8). Only
        # quantizable weights end up here by construction.
        self._fixed_config: Dict[str, int] = {}
        for ca in self.fixed_clusters:
            for pname in ca["layer_names"]:
                self._fixed_config[pname] = 8

        # ── Hessian sensitivity (per-layer, optional) ──
        # When supplied, drives both the surrogate features and the
        # sensitivity-weighted mutation operator. Phase 1a stores the
        # values as nested dicts (``{"hessian_diag": float, ...}``);
        # we normalise here so downstream code can consume scalars.
        self._hessian_per_layer: Dict[str, float] = {}
        for name, value in (hessian_diag or {}).items():
            if isinstance(value, dict):
                v = value.get("hessian_diag", value.get("score", 0.0))
            else:
                v = value
            try:
                self._hessian_per_layer[name] = float(v)
            except (TypeError, ValueError):
                continue

        # Per-gene mutation rate. When sensitivity-weighted mutation
        # is enabled, each gene's flip probability is scaled inversely
        # with that gene's normalised Hessian sensitivity:
        # high-sensitivity genes flip rarely (need to keep INT8),
        # low-sensitivity genes flip aggressively (free to be INT4).
        # The base mutation rate from config is the *mean* across
        # genes, so the global expected number of flips per individual
        # is unchanged — only their distribution shifts.
        self._gene_mutation_rates: List[float] = self._build_gene_mutation_rates()

        # Precompute FP32 EBops (bytes) for reduction calculation, and
        # the equivalent public objective in MiB.
        self._fp32_ebops = self._compute_ebops_for_config({})  # empty = FP32
        self._fp32_size_mb = model_size_mb_from_bytes(self._fp32_ebops)

        # Track evolution history
        self._pareto_size_history: List[int] = []

        # Last search result (populated by search()) — consulted by
        # get_pareto_front() so callers can retrieve the Pareto set
        # without keeping the ParetoFront object themselves.
        self._last_pareto: List[ParetoSolution] = []

        logger.info(
            "NSGA-II initialized: mode=%s, %d genes, search space=2^%d=%s, "
            "fixed clusters=%d (INT8), objectives=%d %s",
            self._search_mode,
            self.num_genes,
            self.num_genes,
            f"{2 ** self.num_genes:,}" if self.num_genes < 64 else "huge",
            len(self.fixed_clusters),
            self._num_objectives,
            "(acc_loss, size, latency)" if self._num_objectives == 3
            else "(acc_loss, size)",
        )

    # ------------------------------------------------------------------
    # Encoding: Individual ↔ Bitwidth Config
    # ------------------------------------------------------------------

    def individual_to_config(self, individual: List[int]) -> Dict[str, int]:
        """
        Convert a GA individual (binary genes) to a full bitwidth config.

        Per-layer mode: gene[i] selects the bitwidth of
        ``self._gene_layers[i]`` directly.
        Cluster mode: gene[i] selects the bitwidth of every layer in
        ``self.searchable_clusters[i]``.

        ``0 = INT4``, ``1 = INT8``. The fixed clusters (HIGH tier) are
        always INT8 regardless of mode.
        """
        config = self._fixed_config.copy()

        if self._search_mode == "per_layer":
            for gene_idx, pname in enumerate(self._gene_layers):
                config[pname] = 8 if individual[gene_idx] == 1 else 4
        else:
            for gene_idx, ca in enumerate(self.searchable_clusters):
                bitwidth = 8 if individual[gene_idx] == 1 else 4
                for pname in ca["layer_names"]:
                    config[pname] = bitwidth

        return config

    def config_to_individual(self, config: Dict[str, int]) -> List[int]:
        """
        Convert a bitwidth config to a GA individual.

        Layers absent from ``config`` default to INT8 (the safe choice
        when the seed config from FITCompress doesn't enumerate every
        searchable parameter).
        """
        if self._search_mode == "per_layer":
            return [
                1 if int(config.get(pname, 8)) == 8 else 0
                for pname in self._gene_layers
            ]
        # Cluster mode (legacy)
        individual: List[int] = []
        for ca in self.searchable_clusters:
            first_layer = ca["layer_names"][0] if ca["layer_names"] else None
            if first_layer and first_layer in config:
                individual.append(1 if int(config[first_layer]) == 8 else 0)
            else:
                individual.append(1)  # Default INT8
        return individual

    # ------------------------------------------------------------------
    # Sensitivity-weighted mutation
    # ------------------------------------------------------------------

    def _maybe_build_surrogate(self) -> Optional[Any]:
        """Construct an ``AccuracySurrogate`` when sklearn is available.

        Returns None when:
          * ``nsga_use_surrogate`` is False (caller already filtered),
          * sklearn is missing,
          * ``num_genes`` is too small for the surrogate to add value
            (≤ 4 — exhaustive mode handles those exactly),
          * the import raises for any other reason.

        The Hessian feature vector is built per-gene (per-layer in
        per_layer mode, per-cluster mean in cluster mode) so the
        surrogate generalises across both encodings.
        """
        try:
            from quantization.surrogate import AccuracySurrogate
        except Exception as exc:
            logger.debug("Surrogate import failed: %s", exc)
            return None

        if self.num_genes <= 4:
            logger.info(
                "  Surrogate disabled — search space 2^%d is too small "
                "(falls back to plain NSGA / exhaustive).",
                self.num_genes,
            )
            return None

        # Hessian features aligned with the gene order.
        if self._search_mode == "per_layer":
            hessian_features = [
                self._hessian_per_layer.get(pn, 0.0) for pn in self._gene_layers
            ]
        else:
            hessian_features = []
            for ca in self.searchable_clusters:
                vals = [
                    self._hessian_per_layer.get(pn, 0.0)
                    for pn in ca["layer_names"]
                ]
                hessian_features.append(
                    float(np.mean(vals)) if vals else 0.0
                )

        try:
            return AccuracySurrogate(
                n_layers=self.num_genes,
                hessian_features=hessian_features if any(hessian_features) else None,
                min_train_samples=int(getattr(
                    self.config.hyperparams, "nsga_surrogate_warmup_evals", 30,
                )),
                random_state=int(self.config.hyperparams.seed),
            )
        except Exception as exc:
            logger.debug("Surrogate build failed: %s", exc)
            return None

    def _build_gene_mutation_rates(self) -> List[float]:
        """Compute per-gene mutation probabilities from Hessian sensitivity.

        High-sensitivity genes get a *lower* flip probability — flipping
        a sensitive layer to INT4 is the canonical accuracy killer, so
        the operator should bias the search away from those moves.

        The vector is rescaled so the *mean* matches
        ``hp.nsga_mutation_prob``, keeping the expected total number
        of flips per individual stable. Falls back to uniform when:
          * sensitivity-weighted mutation is disabled, OR
          * no Hessian sensitivity is available, OR
          * the per-gene sensitivity values are all equal.
        """
        hp = self.config.hyperparams
        base = float(hp.nsga_mutation_prob)
        n = self.num_genes
        if n <= 0:
            return []

        enabled = bool(getattr(hp, "nsga_sensitivity_weighted_mutation", True))
        if not enabled or not self._hessian_per_layer:
            return [base] * n

        # Gather per-gene sensitivity (per-layer mode → direct lookup;
        # cluster mode → mean sensitivity within the cluster).
        if self._search_mode == "per_layer":
            sens = np.asarray(
                [self._hessian_per_layer.get(pn, 0.0) for pn in self._gene_layers],
                dtype=float,
            )
        else:
            sens_list: List[float] = []
            for ca in self.searchable_clusters:
                vals = [
                    self._hessian_per_layer.get(pn, 0.0)
                    for pn in ca["layer_names"]
                ]
                sens_list.append(float(np.mean(vals)) if vals else 0.0)
            sens = np.asarray(sens_list, dtype=float)

        if sens.size == 0 or float(sens.std()) < 1e-12:
            return [base] * n

        # Inverse-rank weighting: high sensitivity → small weight.
        # Use rank rather than raw values so heavy-tailed Hessians
        # don't push the rates to extremes.
        order = sens.argsort()
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n)
        # Map ranks into [0.5, 1.5] so the mean is 1.0; invert so
        # high sensitivity (high rank) maps to a low multiplier.
        normalised = ranks / max(n - 1, 1)
        multipliers = 1.5 - normalised  # [0.5, 1.5], descending with sensitivity
        rates = base * multipliers
        # Re-centre so the mean is exactly ``base``.
        rates *= base / float(rates.mean())
        # Clamp away from 0 so even the most sensitive gene can flip
        # occasionally (search needs to verify HIGH tier choices).
        rates = np.clip(rates, base * 0.1, min(0.95, base * 3.0))
        return [float(x) for x in rates]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_individual(
        self,
        individual: List[int],
        val_loader: DataLoader,
        fp32_accuracy: float,
    ) -> Tuple[float, ...]:
        """
        Evaluate a single individual: fake-quantize model, measure metrics.

        Args:
            individual: Binary gene list.
            val_loader: Validation DataLoader.
            fp32_accuracy: Baseline FP32 accuracy (%).

        Returns:
            * 2-objective mode (no LUT): ``(accuracy_loss, model_size_mb)``
            * 3-objective mode (with LUT):
              ``(accuracy_loss, model_size_mb, latency_ms)``

            All objectives are minimised. The number of values matches
            ``self._num_objectives`` exactly so callers can write
            objective-agnostic loops over the tuple.
        """
        config = self.individual_to_config(individual)
        infinity = tuple([float("inf")] * self._num_objectives)

        try:
            # Clone and fake-quantize
            model_copy = copy.deepcopy(self.model)
            model_copy.to(self.device)
            _apply_fake_quantization(model_copy, config)

            # Evaluate accuracy
            accuracy = self._evaluate_accuracy(model_copy, val_loader)
            accuracy_loss = fp32_accuracy - accuracy

            # Compute objective #2: model size in MiB
            ebops = self._compute_ebops_for_config(config)
            model_size_mb = model_size_mb_from_bytes(ebops)

            # Clean up before the (potentially expensive) LUT lookup so
            # we don't hold the cloned model in memory longer than
            # necessary.
            del model_copy
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if self._num_objectives == 3:
                from quantization.latency_lut import latency_for_assignment
                latency_ms = latency_for_assignment(
                    config, self._latency_lut or {},
                )
                return accuracy_loss, model_size_mb, latency_ms

            return accuracy_loss, model_size_mb

        except Exception as e:
            logger.warning("Evaluation failed for individual: %s", e)
            return infinity

    def _evaluate_accuracy(
        self, model: nn.Module, data_loader: DataLoader
    ) -> float:
        """
        Compute accuracy on a dataset for NSGA-II fitness.

        Uses config.hyperparams.nsga_accuracy_objective ('top1' or 'top5')
        to select which accuracy metric drives the search.
        Default: top-1 (better discrimination on small datasets like CIFAR-10).
        """
        from utils.metrics import compute_topk_accuracy
        acc = compute_topk_accuracy(model, data_loader, self.device)
        objective = getattr(self.config.hyperparams, 'nsga_accuracy_objective', 'top1')
        return acc.get(objective, acc["top1"])

    def _compute_ebops_for_config(self, config: Dict[str, int]) -> float:
        """
        Compute EBops = sum(params x bitwidth) / 8 for a config.

        Params not in config default to FP32 (32 bits).
        """
        total_bits = 0.0
        for name, param in self.model.named_parameters():
            bitwidth = config.get(name, 32)
            total_bits += param.numel() * bitwidth
        return total_bits / 8.0

    # ------------------------------------------------------------------
    # NSGA-II Core: Non-dominated Sorting
    # ------------------------------------------------------------------

    @staticmethod
    def _non_dominated_sort(
        objectives: List[Tuple[float, ...]],
    ) -> List[List[int]]:
        """
        Perform fast non-dominated sorting (Deb et al. 2002).

        All objectives are minimised. This function is dimension-agnostic:
        each entry of ``objectives`` is an N-tuple, with the same N for
        every entry (enforced by the caller — ``self._num_objectives``).
        That lets the same routine drive both the 2-objective
        ``(acc_loss, size)`` search and the 3-objective
        ``(acc_loss, size, latency)`` hardware-aware search.

        Args:
            objectives: List of N-tuples, one per individual. Smaller is
                better in every dimension.

        Returns:
            List of fronts, where fronts[0] is the Pareto front (rank 1),
            fronts[1] is rank 2, etc. Each front is a list of indices.
        """
        n = len(objectives)
        if n == 0:
            return []
        domination_count = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]

        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if NSGAIIClusterSearch._dominates(objectives[i], objectives[j]):
                    dominated_set[i].append(j)
                elif NSGAIIClusterSearch._dominates(objectives[j], objectives[i]):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front: List[int] = []
            for i in fronts[current_front]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        if not fronts[-1]:
            fronts.pop()

        return fronts

    @staticmethod
    def _dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        """Return True iff ``a`` Pareto-dominates ``b`` for minimisation.

        ``a`` dominates ``b`` ⇔ ``a[k] ≤ b[k]`` for every k AND
        ``a[k] < b[k]`` for at least one k. Generalised to arbitrary N
        so the same predicate works in 2- and 3-objective modes. Tuples
        of different length are never compared — the caller guarantees
        homogeneity.
        """
        leq_all = True
        lt_any = False
        for x, y in zip(a, b):
            if x > y:
                leq_all = False
                break
            if x < y:
                lt_any = True
        return leq_all and lt_any

    @staticmethod
    def _crowding_distance(
        objectives: List[Tuple[float, ...]],
        front: List[int],
    ) -> Dict[int, float]:
        """
        Compute crowding distance for individuals in a front.

        Boundary individuals get infinity. Interior individuals get
        the sum of normalised distances to their neighbours in each
        objective dimension. Generalised over N objectives — the only
        change vs. the 2-objective version is that we iterate
        ``range(num_obj)`` instead of hard-coding ``range(2)``.

        Args:
            objectives: Full list of N-tuples.
            front: Indices of individuals in this front.

        Returns:
            {index -> crowding_distance} for each index in front.
        """
        distances: Dict[int, float] = {i: 0.0 for i in front}

        if len(front) <= 2:
            for i in front:
                distances[i] = float("inf")
            return distances

        if not objectives:
            return distances
        num_obj = len(objectives[front[0]])

        for obj_dim in range(num_obj):
            sorted_front = sorted(front, key=lambda i: objectives[i][obj_dim])

            distances[sorted_front[0]] = float("inf")
            distances[sorted_front[-1]] = float("inf")

            obj_range = (
                objectives[sorted_front[-1]][obj_dim]
                - objectives[sorted_front[0]][obj_dim]
            )
            if obj_range <= 0:
                continue

            for k in range(1, len(sorted_front) - 1):
                idx = sorted_front[k]
                prev_val = objectives[sorted_front[k - 1]][obj_dim]
                next_val = objectives[sorted_front[k + 1]][obj_dim]
                distances[idx] += (next_val - prev_val) / obj_range

        return distances

    # ------------------------------------------------------------------
    # Genetic Operators
    # ------------------------------------------------------------------

    @staticmethod
    def _tournament_select(
        ranks: List[int],
        crowding: List[float],
        tournament_size: int = 2,
    ) -> int:
        """
        Binary tournament selection: prefer lower rank, then higher
        crowding distance (for diversity).

        Returns:
            Index of the selected individual.
        """
        candidates = random.sample(range(len(ranks)), min(tournament_size, len(ranks)))
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and crowding[c] > crowding[best]:
                best = c
        return best

    @staticmethod
    def _crossover(
        parent1: List[int],
        parent2: List[int],
        probability: float,
    ) -> Tuple[List[int], List[int]]:
        """Uniform crossover: swap genes independently with 50% chance."""
        if random.random() > probability:
            return parent1[:], parent2[:]

        child1, child2 = parent1[:], parent2[:]
        for i in range(len(child1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def _mutate(
        self, individual: List[int], probability: float,
    ) -> List[int]:
        """Bit-flip mutation with optional per-gene rates.

        When ``self._gene_mutation_rates`` is populated (sensitivity-
        weighted mutation), each gene uses its own rate so high-
        sensitivity layers flip less often than low-sensitivity ones.
        Otherwise falls back to the uniform ``probability`` argument
        — preserves the legacy contract of the previous static method.
        """
        mutant = individual[:]
        rates = self._gene_mutation_rates
        for i in range(len(mutant)):
            p = rates[i] if rates and i < len(rates) else probability
            if random.random() < p:
                mutant[i] = 1 - mutant[i]
        return mutant

    # ------------------------------------------------------------------
    # Main Search Loop
    # ------------------------------------------------------------------

    def search(
        self,
        val_loader: DataLoader,
        fp32_accuracy: float,
        seed_config: Dict[str, int],
    ) -> ParetoFront:
        """
        Run the full NSGA-II search.

        Args:
            val_loader: Validation DataLoader for accuracy evaluation.
            fp32_accuracy: Baseline FP32 accuracy (%).
            seed_config: Elite seed from Phase 1b FITCompress.

        Returns:
            ParetoFront with non-dominated solutions.
        """
        pop_size = self.config.hyperparams.nsga_population_size
        max_gen = self.config.hyperparams.nsga_generations
        cx_prob = self.config.hyperparams.nsga_crossover_prob
        mut_prob = self.config.hyperparams.nsga_mutation_prob
        seed = self.config.hyperparams.seed

        # ── Surrogate-assisted search (BRP-NAS / OFA style) ──
        # After ``warmup`` real evaluations a GradientBoosting model
        # learns the (bitwidth_vector → accuracy_loss) mapping. Each
        # generation we then *propose* up to ``proposed_per_gen``
        # candidates via crossover/mutation, score them with the
        # surrogate (microseconds each), and only real-evaluate the
        # top ``pop_size`` predicted picks. Falls back to plain NSGA
        # when the surrogate is disabled or sklearn is missing.
        hp = self.config.hyperparams
        use_surrogate = bool(getattr(hp, "nsga_use_surrogate", True))
        warmup = int(getattr(hp, "nsga_surrogate_warmup_evals", 30))
        proposed_per_gen = int(getattr(hp, "nsga_surrogate_proposed_per_gen", 256))
        surrogate = self._maybe_build_surrogate() if use_surrogate else None

        # Reproducibility
        random.seed(seed)
        np.random.seed(seed)

        logger.info("=" * 70)
        logger.info(
            "Phase 1c: NSGA-II Multi-Objective Search (mode=%s%s)",
            self._search_mode,
            ", surrogate-assisted" if surrogate else "",
        )
        logger.info("=" * 70)
        search_space_size = max(1, 2 ** self.num_genes)
        if search_space_size < pop_size:
            logger.info(
                "  Population capped by finite search space: %d -> %d",
                pop_size, search_space_size,
            )
            pop_size = search_space_size
        logger.info(
            "  Search space: 2^%d = %d configs  |  Pop: %d  |  Gens: %d",
            self.num_genes, search_space_size, pop_size, max_gen,
        )
        logger.info("  FP32 baseline: %.2f%% accuracy, %.2f MiB model size",
                     fp32_accuracy, self._fp32_size_mb)
        logger.info("=" * 70)

        # When the full configuration space fits in one population, evaluate
        # it exactly instead of running a stochastic evolutionary loop.
        if search_space_size <= pop_size:
            return self._search_exhaustive(val_loader, fp32_accuracy)

        # ── Generation 0: Initialize population ──
        logger.info("Generation 0: Initializing population ...")

        # Population entries are ``(individual, objectives_tuple)`` where
        # ``objectives_tuple`` has length ``self._num_objectives`` (2 or
        # 3 depending on whether a latency LUT was supplied). Storing
        # the tuple verbatim keeps the loop objective-agnostic.
        population: List[Tuple[List[int], Tuple[float, ...]]] = []

        # Helper that real-evaluates AND records the result for the
        # surrogate's training set in one call. Centralised so every
        # fresh evaluation feeds the surrogate without scattering
        # ``add_observation`` calls across the loop.
        def _eval_and_record(ind: List[int]) -> Tuple[float, ...]:
            obj = self.evaluate_individual(ind, val_loader, fp32_accuracy)
            if surrogate is not None and obj and obj[0] != float("inf"):
                surrogate.add_observation(ind, float(obj[0]))
            return obj

        # Add elite seed
        elite_individual = self.config_to_individual(seed_config)
        elite_obj = _eval_and_record(elite_individual)
        population.append((elite_individual, elite_obj))
        logger.info(
            "  Elite seed: %s",
            self._format_objectives(elite_obj),
        )

        # Fill with random individuals (avoid duplicates)
        seen = {tuple(elite_individual)}
        attempts = 0
        while len(population) < pop_size and attempts < pop_size * 10:
            ind = [random.randint(0, 1) for _ in range(self.num_genes)]
            key = tuple(ind)
            if key not in seen:
                seen.add(key)
                obj = _eval_and_record(ind)
                population.append((ind, obj))
            attempts += 1

        logger.info("  Population initialised: %d individuals", len(population))
        total_evals = len(population)

        # ── Evolution loop ──
        final_gen = 0
        convergence_reason = "max_gen"
        stability_window = 10

        for gen in range(1, max_gen + 1):
            final_gen = gen

            objectives = [ind[1] for ind in population]

            # Non-dominated sort
            fronts = self._non_dominated_sort(objectives)

            # Assign ranks and crowding distances
            ranks = [0] * len(population)
            crowding = [0.0] * len(population)
            for rank, front in enumerate(fronts, start=1):
                cd = self._crowding_distance(objectives, front)
                for idx in front:
                    ranks[idx] = rank
                    crowding[idx] = cd[idx]

            # Track Pareto front size
            pareto_size = len(fronts[0]) if fronts else 0
            self._pareto_size_history.append(pareto_size)

            if gen % 5 == 0 or gen == 1:
                if fronts and fronts[0]:
                    best_idx = min(fronts[0], key=lambda i: objectives[i][0])
                    logger.info(
                        "  Gen %d/%d: Pareto=%d, best=%s",
                        gen, max_gen, pareto_size,
                        self._format_objectives(objectives[best_idx]),
                    )

            # ── Surrogate refit (best-effort each generation) ──
            # Cheap (milliseconds), so retrain whenever
            # ``retrain_every`` real evaluations have accumulated.
            if surrogate is not None and surrogate.needs_refit():
                surrogate.fit()

            # ── Create offspring ──
            # Two paths share the same crossover/mutation operators:
            #
            # 1. Plain NSGA-II (default until surrogate.is_ready()):
            #    Generate exactly ``pop_size`` children, real-evaluate
            #    every one. Identical behaviour to pre-surrogate NSGA.
            #
            # 2. Surrogate-assisted: generate ``proposed_per_gen``
            #    *unique* children, surrogate-rank them, real-evaluate
            #    only the top ``pop_size`` (smallest predicted
            #    accuracy_loss). This raises effective per-generation
            #    throughput by 10–50× without burning real-eval budget
            #    on candidates the surrogate already deems weak.
            offspring: List[Tuple[List[int], Tuple[float, ...]]] = []

            if surrogate is not None and surrogate.is_ready():
                proposals: List[List[int]] = []
                proposal_keys: set = set()
                # Generate up to ``proposed_per_gen`` unique candidates.
                max_attempts = proposed_per_gen * 4
                for _ in range(max_attempts):
                    if len(proposals) >= proposed_per_gen:
                        break
                    p1_idx = self._tournament_select(ranks, crowding)
                    p2_idx = self._tournament_select(ranks, crowding)
                    c1, c2 = self._crossover(
                        population[p1_idx][0], population[p2_idx][0], cx_prob,
                    )
                    for child in (
                        self._mutate(c1, mut_prob),
                        self._mutate(c2, mut_prob),
                    ):
                        key = tuple(child)
                        if key not in proposal_keys:
                            proposal_keys.add(key)
                            proposals.append(child)
                            if len(proposals) >= proposed_per_gen:
                                break

                # Surrogate-score every proposal.
                scores = surrogate.score_batch(proposals)
                if scores is None:
                    # Fall back to plain NSGA for this generation.
                    for child in proposals[:pop_size]:
                        offspring.append((child, _eval_and_record(child)))
                        total_evals += 1
                else:
                    # Pick the top-K by predicted accuracy_loss.
                    order = np.argsort(scores)[:pop_size]
                    for idx in order:
                        child = proposals[int(idx)]
                        offspring.append((child, _eval_and_record(child)))
                        total_evals += 1
            else:
                # Plain NSGA path — used during warmup and when sklearn
                # is unavailable.
                while len(offspring) < pop_size:
                    p1_idx = self._tournament_select(ranks, crowding)
                    p2_idx = self._tournament_select(ranks, crowding)
                    c1, c2 = self._crossover(
                        population[p1_idx][0], population[p2_idx][0], cx_prob,
                    )
                    c1 = self._mutate(c1, mut_prob)
                    c2 = self._mutate(c2, mut_prob)
                    offspring.append((c1, _eval_and_record(c1)))
                    offspring.append((c2, _eval_and_record(c2)))
                    total_evals += 2

                # Cold-start the surrogate as soon as warmup is met so
                # the next generation can use it.
                if surrogate is not None and surrogate.num_observations() >= warmup:
                    surrogate.fit()

            # ── Survivor selection ──
            combined = population + offspring
            combined_obj = [x[1] for x in combined]
            combined_fronts = self._non_dominated_sort(combined_obj)

            # Fill next population front-by-front
            next_pop: List[Tuple[List[int], Tuple[float, ...]]] = []
            for front in combined_fronts:
                if len(next_pop) + len(front) <= pop_size:
                    next_pop.extend(combined[i] for i in front)
                else:
                    cd = self._crowding_distance(combined_obj, front)
                    remaining = pop_size - len(next_pop)
                    sorted_by_cd = sorted(front, key=lambda i: -cd[i])
                    next_pop.extend(combined[i] for i in sorted_by_cd[:remaining])
                    break

            population = next_pop

            # ── Convergence check ──
            if len(self._pareto_size_history) >= stability_window:
                recent = self._pareto_size_history[-stability_window:]
                if all(s == recent[0] for s in recent):
                    logger.info(
                        "  Gen %d: Pareto front stable for %d generations, "
                        "terminating early.",
                        gen, stability_window,
                    )
                    convergence_reason = "pareto_stable"
                    break

        # ── Extract final Pareto front ──
        final_objectives = [x[1] for x in population]
        final_fronts = self._non_dominated_sort(final_objectives)
        final_cd = (
            self._crowding_distance(final_objectives, final_fronts[0])
            if final_fronts
            else {}
        )

        pareto_solutions: List[ParetoSolution] = []
        pareto_indices = final_fronts[0] if final_fronts else []

        for rank_idx, pop_idx in enumerate(pareto_indices):
            individual, obj = population[pop_idx]
            pareto_solutions.append(
                self._build_pareto_solution(
                    individual, obj,
                    rank_id=f"nsga_gen{final_gen}_r{rank_idx + 1}",
                    crowding_distance=final_cd.get(pop_idx, 0.0),
                    fp32_accuracy=fp32_accuracy,
                )
            )

        # Sort by accuracy (best first, lowest loss)
        pareto_solutions.sort(key=lambda s: s["accuracy_loss"])
        self._last_pareto = pareto_solutions

        result = ParetoFront(
            solutions=pareto_solutions,
            generation=final_gen,
            evaluations=total_evals,
            convergence_reason=convergence_reason,
        )

        # ── Log summary ──
        logger.info("=" * 70)
        logger.info(
            "NSGA-II Complete: %d Pareto solutions found in %d generations "
            "(%d evaluations)",
            len(pareto_solutions), final_gen, total_evals,
        )
        logger.info("-" * 70)
        logger.info("Pareto Front (sorted by accuracy):")
        for i, sol in enumerate(pareto_solutions):
            logger.info(
                "  [%d] Acc: %.2f%% (loss: %.2f%%), EBops reduction: %.1f%%, "
                "ID: %s",
                i + 1,
                sol["accuracy"],
                sol["accuracy_loss"],
                sol["ebops_reduction"],
                sol["solution_id"],
            )
        logger.info("=" * 70)

        return result

    def _search_exhaustive(
        self, val_loader: DataLoader, fp32_accuracy: float,
    ) -> ParetoFront:
        """Evaluate all possible individuals exactly (small search spaces)."""
        logger.info(
            "Exhaustive mode: evaluating all 2^%d = %d configurations",
            self.num_genes, max(1, 2 ** self.num_genes),
        )

        if self.num_genes <= 0:
            individuals: List[List[int]] = [[]]
        else:
            individuals = [list(bits) for bits in itertools.product([0, 1], repeat=self.num_genes)]

        population: List[Tuple[List[int], Tuple[float, ...]]] = []
        for ind in individuals:
            obj = self.evaluate_individual(ind, val_loader, fp32_accuracy)
            population.append((ind, obj))

        objectives = [x[1] for x in population]
        fronts = self._non_dominated_sort(objectives)
        pareto_indices = fronts[0] if fronts else []
        cd = self._crowding_distance(objectives, pareto_indices) if pareto_indices else {}

        pareto_solutions: List[ParetoSolution] = []
        for rank_idx, idx in enumerate(pareto_indices):
            individual, obj = population[idx]
            pareto_solutions.append(
                self._build_pareto_solution(
                    individual, obj,
                    rank_id=f"nsga_exhaustive_r{rank_idx + 1}",
                    crowding_distance=cd.get(idx, 0.0),
                    fp32_accuracy=fp32_accuracy,
                )
            )

        pareto_solutions.sort(key=lambda s: s["accuracy_loss"])
        self._last_pareto = pareto_solutions

        result = ParetoFront(
            solutions=pareto_solutions,
            generation=1,
            evaluations=len(population),
            convergence_reason="exhaustive",
        )

        logger.info(
            "Exhaustive NSGA-II complete: %d Pareto solutions from %d evaluations",
            len(pareto_solutions), len(population),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_pareto_front(self) -> List[ParetoSolution]:
        """Return the Pareto solutions found by the most recent search().

        Returns an empty list when search() has not been called yet.
        """
        return list(self._last_pareto)

    def get_quantizable_param_names(self) -> List[str]:
        """Return the parameter names that NSGA-II treats as quantizable.

        These are the only parameters that can appear in a gene-derived
        bitwidth assignment. BatchNorm/bias/etc. are excluded by
        construction.
        """
        return sorted(self._quantizable_weights)

    def get_searchable_param_names(self) -> List[str]:
        """Return the parameter names that genes actually search over.

        Equivalent to the union of ``layer_names`` across all
        searchable (MEDIUM/LOW) clusters, after the non-quantizable
        filter has been applied.
        """
        names: set = set()
        for ca in self.searchable_clusters:
            for n in ca.get("layer_names", []):
                names.add(n)
        return sorted(names)

    def _build_pareto_solution(
        self,
        individual: List[int],
        objectives_tuple: Tuple[float, ...],
        *,
        rank_id: str,
        crowding_distance: float,
        fp32_accuracy: float,
    ) -> ParetoSolution:
        """Construct a ``ParetoSolution`` from an objectives tuple.

        Centralises the (acc_loss, size, [latency]) → ParetoSolution
        conversion so both the evolutionary loop and the exhaustive
        search produce solutions with identical fields. The 3-objective
        path additionally populates ``latency_mean_ms`` (the LUT-summed
        deployment-runtime number used during the search).
        """
        config = self.individual_to_config(individual)
        ebops = self._compute_ebops_for_config(config)
        ebops_reduction = (
            (self._fp32_ebops - ebops) / self._fp32_ebops * 100.0
            if self._fp32_ebops > 0
            else 0.0
        )

        acc_loss = float(objectives_tuple[0])
        model_size_mb = float(objectives_tuple[1])
        latency_mean_ms = (
            float(objectives_tuple[2]) if len(objectives_tuple) >= 3 else None
        )

        return ParetoSolution(
            solution_id=rank_id,
            method="PTQ",
            accuracy=fp32_accuracy - acc_loss,
            accuracy_loss=acc_loss,
            ebops=ebops,
            ebops_reduction=ebops_reduction,
            model_size_mb=model_size_mb,
            latency_mean_ms=latency_mean_ms,
            bitwidth_assignment=config,
            rank=1,
            crowding_distance=crowding_distance,
            is_dominated=False,
        )

    def _format_objectives(self, obj: Tuple[float, ...]) -> str:
        """Pretty-printer for an N-objective tuple, used in log messages."""
        if len(obj) >= 3:
            return (
                f"acc_loss={obj[0]:.2f}%%, size={obj[1]:.2f} MiB, "
                f"latency={obj[2]:.2f} ms"
            )
        return f"acc_loss={obj[0]:.2f}%%, size={obj[1]:.2f} MiB"

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

    @staticmethod
    def _build_quantizable_weight_set(model: nn.Module) -> set:
        """Return the set of parameter names that NSGA-II is allowed to
        assign a quantization bitwidth to.

        Only the ``.weight`` tensor of ``nn.Conv2d`` / ``nn.Linear``
        modules is considered quantizable. BatchNorm γ/β, biases, and
        any other module's parameters stay FP32 — both during evaluation
        (no fake-quantization) and during real PTQ materialization.
        """
        quantizable: set = set()
        for module_name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if not hasattr(module, "weight") or module.weight is None:
                continue
            pname = f"{module_name}.weight" if module_name else "weight"
            quantizable.add(pname)
        return quantizable
