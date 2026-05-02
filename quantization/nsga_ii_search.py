"""
NeuroQuant v2.0 - NSGA-II Cluster-Level Search (Phase 1c)

Multi-objective evolutionary optimization at the CLUSTER level
instead of the layer level. This is the key speedup innovation:
search space is 2^N (searchable clusters) vs 2^L (all layers).

Key design decisions:
    - Only MEDIUM and LOW clusters are searchable (HIGH = fixed INT8)
    - Individuals encode bitwidths as binary genes (0=INT4, 1=INT8)
    - Fake quantization used for fast evaluation during search
    - Proper NSGA-II: non-dominated sorting + crowding distance
    - Warm-started with FITCompress elite seed from Phase 1b

Objectives (both minimised):
    1. Accuracy loss = FP32_accuracy - quantized_accuracy
    2. Model size (MiB) = sum(params x bitwidth) / 8 / (1024²)
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
    scale = max(scale.item(), 1e-8)

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
    ) -> None:
        """
        Args:
            model: FP32 baseline model (will be cloned for each evaluation).
            cluster_assignments: From Phase 1a LayerClusterer.create_clusters().
            config: Framework configuration (uses nsga_* hyperparameters).
        """
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.hyperparams.device)

        # Separate clusters into fixed (HIGH) and searchable (MEDIUM/LOW)
        self.fixed_clusters: List[ClusterAssignment] = []
        self.searchable_clusters: List[ClusterAssignment] = []

        for ca in cluster_assignments:
            if ca["tier"] == "HIGH":
                self.fixed_clusters.append(ca)
            else:
                self.searchable_clusters.append(ca)

        self.num_genes = len(self.searchable_clusters)

        # Build the fixed part of the bitwidth config (HIGH → INT8)
        self._fixed_config: Dict[str, int] = {}
        for ca in self.fixed_clusters:
            for pname in ca["layer_names"]:
                self._fixed_config[pname] = 8

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
            "NSGA-II initialized: %d searchable clusters (%d genes), "
            "%d fixed clusters (INT8), search space = 2^%d = %d",
            self.num_genes,
            self.num_genes,
            len(self.fixed_clusters),
            self.num_genes,
            2 ** self.num_genes,
        )

    # ------------------------------------------------------------------
    # Encoding: Individual ↔ Bitwidth Config
    # ------------------------------------------------------------------

    def individual_to_config(self, individual: List[int]) -> Dict[str, int]:
        """
        Convert a GA individual (binary genes) to a full bitwidth config.

        Individual is a list of length num_genes where:
            0 = INT4, 1 = INT8

        The fixed clusters (HIGH tier) are always INT8.

        Args:
            individual: Binary gene list.

        Returns:
            Full bitwidth config: {param_name -> bitwidth (4 or 8)}.
        """
        config = self._fixed_config.copy()

        for gene_idx, ca in enumerate(self.searchable_clusters):
            bitwidth = 8 if individual[gene_idx] == 1 else 4
            for pname in ca["layer_names"]:
                config[pname] = bitwidth

        return config

    def config_to_individual(self, config: Dict[str, int]) -> List[int]:
        """
        Convert a bitwidth config to a GA individual.

        Args:
            config: {param_name -> bitwidth}.

        Returns:
            Binary gene list (only searchable clusters encoded).
        """
        individual = []
        for ca in self.searchable_clusters:
            # Use the first layer in the cluster to determine bitwidth
            first_layer = ca["layer_names"][0] if ca["layer_names"] else None
            if first_layer and first_layer in config:
                individual.append(1 if config[first_layer] == 8 else 0)
            else:
                individual.append(1)  # Default INT8
        return individual

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_individual(
        self,
        individual: List[int],
        val_loader: DataLoader,
        fp32_accuracy: float,
    ) -> Tuple[float, float]:
        """
        Evaluate a single individual: fake-quantize model, measure metrics.

        Args:
            individual: Binary gene list.
            val_loader: Validation DataLoader.
            fp32_accuracy: Baseline FP32 accuracy (%).

        Returns:
            (accuracy_loss, model_size_mb) — both to be minimised.
        """
        config = self.individual_to_config(individual)

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

            # Clean up
            del model_copy
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return accuracy_loss, model_size_mb

        except Exception as e:
            logger.warning("Evaluation failed for individual: %s", e)
            return float("inf"), float("inf")

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
        objectives: List[Tuple[float, float]],
    ) -> List[List[int]]:
        """
        Perform fast non-dominated sorting (Deb et al. 2002).

        Both objectives are minimised: (accuracy_loss, model_size_mb).

        Args:
            objectives: List of (obj1, obj2) tuples.

        Returns:
            List of fronts, where fronts[0] is the Pareto front (rank 1),
            fronts[1] is rank 2, etc. Each front is a list of indices.
        """
        n = len(objectives)
        domination_count = [0] * n       # How many solutions dominate i
        dominated_set: List[List[int]] = [[] for _ in range(n)]  # Solutions i dominates

        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominates j if i is <= in all objectives and < in at least one
                i_leq_all = (
                    objectives[i][0] <= objectives[j][0]
                    and objectives[i][1] <= objectives[j][1]
                )
                i_lt_any = (
                    objectives[i][0] < objectives[j][0]
                    or objectives[i][1] < objectives[j][1]
                )
                if i_leq_all and i_lt_any:
                    dominated_set[i].append(j)
                elif (
                    objectives[j][0] <= objectives[i][0]
                    and objectives[j][1] <= objectives[i][1]
                    and (
                        objectives[j][0] < objectives[i][0]
                        or objectives[j][1] < objectives[i][1]
                    )
                ):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
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

        # Remove the trailing empty front
        if not fronts[-1]:
            fronts.pop()

        return fronts

    @staticmethod
    def _crowding_distance(
        objectives: List[Tuple[float, float]],
        front: List[int],
    ) -> Dict[int, float]:
        """
        Compute crowding distance for individuals in a front.

        Boundary individuals get infinity. Interior individuals get
        the sum of normalised distances to their neighbours in each
        objective dimension.

        Args:
            objectives: Full list of (obj1, obj2) tuples.
            front: Indices of individuals in this front.

        Returns:
            {index -> crowding_distance} for each index in front.
        """
        distances: Dict[int, float] = {i: 0.0 for i in front}

        if len(front) <= 2:
            for i in front:
                distances[i] = float("inf")
            return distances

        # For each objective dimension
        for obj_dim in range(2):
            # Sort front by this objective
            sorted_front = sorted(front, key=lambda i: objectives[i][obj_dim])

            # Boundary points get infinity
            distances[sorted_front[0]] = float("inf")
            distances[sorted_front[-1]] = float("inf")

            # Range of this objective
            obj_range = (
                objectives[sorted_front[-1]][obj_dim]
                - objectives[sorted_front[0]][obj_dim]
            )
            if obj_range <= 0:
                continue

            # Interior points
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

    @staticmethod
    def _mutate(individual: List[int], probability: float) -> List[int]:
        """Bit-flip mutation: each gene flips with given probability."""
        mutant = individual[:]
        for i in range(len(mutant)):
            if random.random() < probability:
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

        # Reproducibility
        random.seed(seed)
        np.random.seed(seed)

        logger.info("=" * 70)
        logger.info("Phase 1c: NSGA-II Multi-Objective Cluster-Level Search")
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

        # Population: list of (individual, accuracy_loss, model_size_mb)
        population: List[Tuple[List[int], float, float]] = []

        # Add elite seed
        elite_individual = self.config_to_individual(seed_config)
        loss, size_mb = self.evaluate_individual(
            elite_individual, val_loader, fp32_accuracy
        )
        population.append((elite_individual, loss, size_mb))
        logger.info(
            "  Elite seed: acc_loss=%.2f%%, size=%.2f MiB",
            loss, size_mb,
        )

        # Fill with random individuals (avoid duplicates)
        seen = {tuple(elite_individual)}
        attempts = 0
        while len(population) < pop_size and attempts < pop_size * 10:
            ind = [random.randint(0, 1) for _ in range(self.num_genes)]
            key = tuple(ind)
            if key not in seen:
                seen.add(key)
                l, e = self.evaluate_individual(ind, val_loader, fp32_accuracy)
                population.append((ind, l, e))
            attempts += 1

        logger.info("  Population initialised: %d individuals", len(population))
        total_evals = len(population)

        # ── Evolution loop ──
        final_gen = 0
        convergence_reason = "max_gen"
        stability_window = 10

        for gen in range(1, max_gen + 1):
            final_gen = gen

            # Extract objectives for sorting
            objectives = [(ind[1], ind[2]) for ind in population]

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
                # Log best solution on Pareto front
                if fronts and fronts[0]:
                    best_idx = min(fronts[0], key=lambda i: objectives[i][0])
                    logger.info(
                        "  Gen %d/%d: Pareto=%d, best_loss=%.2f%%, best_size=%.2f MiB",
                        gen, max_gen, pareto_size,
                        objectives[best_idx][0], objectives[best_idx][1],
                    )

            # ── Create offspring ──
            offspring: List[Tuple[List[int], float, float]] = []
            while len(offspring) < pop_size:
                # Tournament selection
                p1_idx = self._tournament_select(ranks, crowding)
                p2_idx = self._tournament_select(ranks, crowding)

                # Crossover
                c1, c2 = self._crossover(
                    population[p1_idx][0], population[p2_idx][0], cx_prob
                )

                # Mutation
                c1 = self._mutate(c1, mut_prob)
                c2 = self._mutate(c2, mut_prob)

                # Evaluate
                l1, e1 = self.evaluate_individual(c1, val_loader, fp32_accuracy)
                l2, e2 = self.evaluate_individual(c2, val_loader, fp32_accuracy)
                offspring.append((c1, l1, e1))
                offspring.append((c2, l2, e2))
                total_evals += 2

            # ── Survivor selection ──
            combined = population + offspring
            combined_obj = [(x[1], x[2]) for x in combined]
            combined_fronts = self._non_dominated_sort(combined_obj)

            # Fill next population front-by-front
            next_pop: List[Tuple[List[int], float, float]] = []
            for front in combined_fronts:
                if len(next_pop) + len(front) <= pop_size:
                    next_pop.extend(combined[i] for i in front)
                else:
                    # Partial front: sort by crowding distance (descending)
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
        final_objectives = [(x[1], x[2]) for x in population]
        final_fronts = self._non_dominated_sort(final_objectives)
        final_cd = (
            self._crowding_distance(final_objectives, final_fronts[0])
            if final_fronts
            else {}
        )

        pareto_solutions: List[ParetoSolution] = []
        pareto_indices = final_fronts[0] if final_fronts else []

        for rank_idx, pop_idx in enumerate(pareto_indices):
            individual, acc_loss, model_size_mb = population[pop_idx]
            config = self.individual_to_config(individual)
            ebops = self._compute_ebops_for_config(config)

            ebops_reduction = (
                (self._fp32_ebops - ebops) / self._fp32_ebops * 100.0
                if self._fp32_ebops > 0
                else 0.0
            )

            solution = ParetoSolution(
                solution_id=f"nsga_gen{final_gen}_r{rank_idx + 1}",
                method="PTQ",
                accuracy=fp32_accuracy - acc_loss,
                accuracy_loss=acc_loss,
                ebops=ebops,
                ebops_reduction=ebops_reduction,
                model_size_mb=model_size_mb,
                bitwidth_assignment=config,
                rank=1,
                crowding_distance=final_cd.get(pop_idx, 0.0),
                is_dominated=False,
            )
            pareto_solutions.append(solution)

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

        population: List[Tuple[List[int], float, float]] = []
        for ind in individuals:
            acc_loss, size_mb = self.evaluate_individual(ind, val_loader, fp32_accuracy)
            population.append((ind, acc_loss, size_mb))

        objectives = [(x[1], x[2]) for x in population]
        fronts = self._non_dominated_sort(objectives)
        pareto_indices = fronts[0] if fronts else []
        cd = self._crowding_distance(objectives, pareto_indices) if pareto_indices else {}

        pareto_solutions: List[ParetoSolution] = []
        for rank_idx, idx in enumerate(pareto_indices):
            individual, acc_loss, model_size_mb = population[idx]
            config = self.individual_to_config(individual)
            ebops = self._compute_ebops_for_config(config)
            ebops_reduction = (
                (self._fp32_ebops - ebops) / self._fp32_ebops * 100.0
                if self._fp32_ebops > 0
                else 0.0
            )
            pareto_solutions.append(
                ParetoSolution(
                    solution_id=f"nsga_exhaustive_r{rank_idx + 1}",
                    method="PTQ",
                    accuracy=fp32_accuracy - acc_loss,
                    accuracy_loss=acc_loss,
                    ebops=ebops,
                    ebops_reduction=ebops_reduction,
                    model_size_mb=model_size_mb,
                    bitwidth_assignment=config,
                    rank=1,
                    crowding_distance=cd.get(idx, 0.0),
                    is_dominated=False,
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
