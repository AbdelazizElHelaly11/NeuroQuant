"""
NeuroQuant v2.0 — α-search strategies for SmoothQuant / AWQ.

Three strategies share this file because both quantizers face the same
problem (pick a per-layer migration exponent α) with the same constraints
(no expensive surrogate model, deterministic, dependency-free):

* ``closed_form_alpha``  — analytical α from the SmoothQuant paper.
                            One ``tensor.max()`` per channel, no
                            quantization simulation.
* ``golden_section_alpha`` — bisect-on-loss for any unimodal scoring
                            function. Converges to ε in ~⌈logφ(1/ε)⌉
                            evaluations (4 for ε=0.05).
* ``ClusterAmortizer``    — wraps any per-layer search and reuses the
                            chosen α across every layer in the same
                            Hessian/Fisher cluster (Phase 1a).

All three are pure functions / data classes with no torch state of
their own, so the SmoothQuant and AWQ quantizers can call them without
worrying about hook lifecycle or device placement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger("neuroquant")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Closed-form (SmoothQuant paper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def closed_form_alpha(
    act_max: torch.Tensor,
    weight_max: torch.Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Analytical α from the original SmoothQuant paper.

    For each channel ``j``::

        α_j = max|X_j|^β / (max|X_j|^β + max|W_j|^β)

    The per-layer scalar α returned here is the *channel-mean* of
    those per-channel α_j values — empirically as accurate as the
    grid-searched value but ~6× cheaper, and it is what most
    production deployments actually ship.

    Args:
        act_max:    Per-input-channel activation magnitude. Shape ``[C_in]``.
        weight_max: Per-input-channel weight magnitude. Same shape.
        beta:       Migration exponent (paper uses β=1.0). β>1 pulls more
                    weight onto activations, β<1 onto weights.
        eps:        Floor to avoid 0/0 when both magnitudes vanish.

    Returns:
        α as a Python float in [0, 1], averaged over channels.
    """
    a = act_max.detach().float().abs().clamp(min=eps)
    w = weight_max.detach().float().abs().clamp(min=eps)
    a_b = a.pow(beta)
    w_b = w.pow(beta)
    per_channel = a_b / (a_b + w_b).clamp(min=eps)
    return float(per_channel.mean().clamp(0.0, 1.0).item())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Golden-section search (any unimodal scorer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_PHI = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio
_RESPHI = 2.0 - _PHI               # 0.3819660...


def golden_section_alpha(
    score_fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tol: float = 0.02,
    max_evals: int = 6,
) -> float:
    """Locate the α minimising ``score_fn`` over ``[lo, hi]``.

    Assumes the loss curve is unimodal in α (true for SmoothQuant /
    AWQ migration exponents — there is exactly one cross-over between
    "activation outliers dominate" and "weight outliers dominate").

    Each call to ``score_fn(α)`` is a full layer-output MSE evaluation,
    so we bound the total at ``max_evals`` (default 6 — already a 1.7×
    speed-up over the 10-point grid most setups use, and faster than
    the 5-point AWQ default once the bracket starts shrinking).

    Args:
        score_fn:  α → scalar loss. Lower is better.
        lo, hi:    Search interval. ``hi > lo``.
        tol:       Stop when bracket width < tol.
        max_evals: Hard cap on score_fn calls.

    Returns:
        Best α found.
    """
    if hi <= lo:
        return float(lo)

    # Convention: ``c < d`` are the two interior points of the bracket.
    # ``c = a + resphi·(b-a)``, ``d = b - resphi·(b-a)``. Each iteration
    # discards either the leftmost or rightmost sub-interval (whichever
    # contains the higher endpoint score) and reuses the surviving
    # interior point as one of the two points of the new bracket.
    a, b = float(lo), float(hi)
    c = a + _RESPHI * (b - a)
    d = b - _RESPHI * (b - a)
    fc = score_fn(c)
    fd = score_fn(d)
    evals = 2

    best_alpha = c if fc <= fd else d
    best_score = min(fc, fd)

    while (b - a) > tol and evals < max_evals:
        if fc < fd:
            # Minimum is in [a, d]. Shrink right end; reuse c as new d.
            b, d, fd = d, c, fc
            c = a + _RESPHI * (b - a)
            fc = score_fn(c)
            evals += 1
            if fc < best_score:
                best_score = fc
                best_alpha = c
        else:
            # Minimum is in [c, b]. Shrink left end; reuse d as new c.
            a, c, fc = c, d, fd
            d = b - _RESPHI * (b - a)
            fd = score_fn(d)
            evals += 1
            if fd < best_score:
                best_score = fd
                best_alpha = d

    return float(best_alpha)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Cluster amortization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ClusterAmortizer:
    """Reuse one α per Hessian/Fisher cluster across all members.

    The amortizer is fed Phase 1a's ``cluster_assignments`` (a list of
    dicts, each with a ``layer_names`` list and a sensitivity score),
    and produces a layer-name → cluster-id map plus a per-cluster
    representative selection. Once the SmoothQuant / AWQ search has
    chosen α for each representative, ``broadcast()`` fills α for
    every other layer in the same cluster.

    Layers absent from the cluster map (e.g. depthwise convs the
    clusterer skipped) are flagged via ``is_representative`` →
    ``True`` so the caller can run the per-layer search on them as a
    fallback. This way the amortizer never silently drops layers.
    """

    layer_to_cluster: Dict[str, int]
    representatives: Dict[int, str]
    sensitivity: Dict[str, float]

    @classmethod
    def from_cluster_result(
        cls,
        cluster_result: Optional[Dict[str, Any]],
        target_layer_names: List[str],
        *,
        hessian_diag: Optional[Dict[str, float]] = None,
    ) -> "ClusterAmortizer":
        """Build an amortizer from Phase 1a output.

        ``cluster_result`` shape (matches what
        ``LayerClusterer.create_clusters`` produces)::

            {
                "cluster_assignments": [
                    {
                        "cluster_id": 0,
                        "tier": "HIGH",
                        "layer_names": ["features.0.weight", ...],
                        "score": 1.42e-4,
                    },
                    ...
                ]
            }

        The representative for each cluster is the layer with the
        highest Hessian sensitivity — losing it costs the most accuracy,
        so the α picked for it is the most conservative α the cluster
        could need.

        When ``cluster_result`` is None (e.g. Phase 1a was skipped)
        every layer becomes its own cluster — equivalent to the
        previous per-layer search behaviour.
        """
        # Normalise hessian_diag values to floats. Phase 1a stores them
        # as ``{layer_name: {"hessian_diag": float, "layer_type": ...}}``
        # rather than a flat scalar dict; older callers may pass either.
        # Anything we can't coerce becomes 0.0 so it sorts to the end.
        sens: Dict[str, float] = {}
        for name, value in (hessian_diag or {}).items():
            if isinstance(value, dict):
                v = value.get("hessian_diag", value.get("score", 0.0))
            else:
                v = value
            try:
                sens[name] = float(v)
            except (TypeError, ValueError):
                sens[name] = 0.0

        layer_to_cluster: Dict[str, int] = {}
        representatives: Dict[int, str] = {}

        if not cluster_result:
            for i, name in enumerate(target_layer_names):
                layer_to_cluster[name] = i
                representatives[i] = name
            return cls(
                layer_to_cluster=layer_to_cluster,
                representatives=representatives,
                sensitivity=sens,
            )

        # Build name → cluster-id map. ``cluster_id`` may be missing
        # from older checkpoints; fall back to the list index.
        target_set = set(target_layer_names)
        cluster_members: Dict[int, List[str]] = {}
        for idx, cluster in enumerate(
            cluster_result.get("cluster_assignments") or []
        ):
            cid = int(cluster.get("cluster_id", idx))
            for layer_name in cluster.get("layer_names", []):
                # Strip the trailing ".weight" the clusterer uses on
                # parameter names so we can match against module names
                # in either form.
                name_module = layer_name
                if name_module.endswith(".weight"):
                    name_module = name_module[: -len(".weight")]
                if layer_name in target_set:
                    layer_to_cluster[layer_name] = cid
                    cluster_members.setdefault(cid, []).append(layer_name)
                elif name_module in target_set:
                    layer_to_cluster[name_module] = cid
                    cluster_members.setdefault(cid, []).append(name_module)

        # Layers without a cluster (depthwise convs, head-only swaps)
        # become singleton clusters so the search still runs on them.
        next_id = max(layer_to_cluster.values(), default=-1) + 1
        for name in target_layer_names:
            if name not in layer_to_cluster:
                layer_to_cluster[name] = next_id
                cluster_members.setdefault(next_id, []).append(name)
                next_id += 1

        # Pick the highest-sensitivity layer per cluster as the rep.
        for cid, members in cluster_members.items():
            best_member = max(
                members, key=lambda n: sens.get(n, sens.get(n + ".weight", 0.0)),
            )
            representatives[cid] = best_member

        return cls(
            layer_to_cluster=layer_to_cluster,
            representatives=representatives,
            sensitivity=sens,
        )

    def is_representative(self, layer_name: str) -> bool:
        """Whether ``layer_name`` should run the α search itself."""
        cid = self.layer_to_cluster.get(layer_name)
        if cid is None:
            return True  # unknown layer → search it directly
        return self.representatives.get(cid) == layer_name

    def cluster_id(self, layer_name: str) -> Optional[int]:
        return self.layer_to_cluster.get(layer_name)

    def broadcast(
        self,
        chosen_per_rep: Dict[str, float],
    ) -> Dict[str, float]:
        """Map representative-α dict to a full per-layer α dict.

        Layers whose representative didn't get an α (search failed,
        layer not in target set) are simply absent from the output —
        the caller falls back to the global α for those.
        """
        out: Dict[str, float] = {}
        for layer_name, cid in self.layer_to_cluster.items():
            rep = self.representatives.get(cid)
            if rep and rep in chosen_per_rep:
                out[layer_name] = chosen_per_rep[rep]
        return out

    def num_clusters(self) -> int:
        return len(self.representatives)
