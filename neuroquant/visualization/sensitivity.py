"""
NeuroQuant v2.0 — Per-Layer Sensitivity Visualization.

Generates publication-quality visualisations of the per-layer Hessian /
Fisher sensitivity scores computed in Phase 1a. The primary plot is a
horizontal bar chart sorted by sensitivity magnitude, colour-coded by
the tier assignment (HIGH → red, MEDIUM → amber, LOW → green).

Usage (automatic — called from ``phase_1a_hessian_clustering``)::

    from neuroquant.visualization.sensitivity import plot_sensitivity_heatmap
    plot_sensitivity_heatmap(hessian_diag, cluster_result, output_dir)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("neuroquant")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Tier colours — same palette as the rest of NeuroQuant plots.
TIER_COLORS = {
    "HIGH":   "#d62728",   # red — sensitive, force INT8
    "MEDIUM": "#ff7f0e",   # amber — NSGA decides
    "LOW":    "#2ca02c",   # green — safe to compress (INT4)
}
DEFAULT_COLOR = "#7f7f7f"


def plot_sensitivity_heatmap(
    hessian_diag: Dict[str, float],
    cluster_result: Dict[str, Any],
    output_dir: str,
    *,
    top_n: int = 40,
    model_name: str = "Generic CNN",
) -> Optional[str]:
    """Horizontal bar chart of per-layer Hessian/Fisher sensitivity.

    Args:
        hessian_diag:    ``{layer_name: score}`` from Phase 1a.
        cluster_result:  Full cluster result dict containing
                         ``cluster_assignments`` with tier info.
        output_dir:      Directory to write the PNG plot.
        top_n:           Show at most this many layers (sorted by score).
        model_name:      Model name for the plot title.

    Returns:
        Path to the saved PNG, or ``None`` if matplotlib is missing.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available; skipping sensitivity plot.")
        return None

    if not hessian_diag:
        logger.warning("Empty hessian_diag; skipping sensitivity plot.")
        return None

    from neuroquant.visualization.style import apply_publication_style
    apply_publication_style()

    # Normalise hessian_diag values to scalar floats. Phase 1a stores
    # them as ``{layer_name: {"hessian_diag": float, "layer_type": ...}}``;
    # callers may also pass already-flat ``{layer_name: float}``. Both
    # forms must work, otherwise sorting fails with
    # ``'<' not supported between instances of 'dict' and 'dict'``.
    flat_hessian: Dict[str, float] = {}
    for name, value in hessian_diag.items():
        if isinstance(value, dict):
            v = value.get("hessian_diag", value.get("score", 0.0))
        else:
            v = value
        try:
            flat_hessian[name] = float(v)
        except (TypeError, ValueError):
            flat_hessian[name] = 0.0

    # Build tier lookup: layer_name → tier string
    tier_map: Dict[str, str] = {}
    assignments = cluster_result.get("cluster_assignments", [])
    for cluster in assignments:
        tier = cluster.get("tier", "MEDIUM")
        for layer_name in cluster.get("layer_names", []):
            tier_map[layer_name] = tier

    # Only show layers that are in the cluster assignments (i.e.
    # Conv/Linear weights that NSGA actually searches over). BN/bias
    # parameters have Hessian scores but no tier assignment — showing
    # them with a "MEDIUM" default creates a visual contradiction with
    # the tier-distribution pie chart.
    clustered_hessian = {
        k: v for k, v in flat_hessian.items() if k in tier_map
    }
    if not clustered_hessian:
        # Fall back to all layers if no cluster assignments found
        clustered_hessian = flat_hessian

    # Sort layers by sensitivity (descending), take top_n
    sorted_layers = sorted(
        clustered_hessian.items(), key=lambda kv: kv[1], reverse=True,
    )[:top_n]

    if not sorted_layers:
        return None

    # Reverse for horizontal bar (highest at top)
    sorted_layers = list(reversed(sorted_layers))
    names = [_short_name(n) for n, _ in sorted_layers]
    scores = [s for _, s in sorted_layers]
    full_names = [n for n, _ in sorted_layers]
    colors = [TIER_COLORS.get(tier_map.get(n, "MEDIUM"), DEFAULT_COLOR)
              for n in full_names]

    fig, ax = plt.subplots(
        figsize=(10, max(5, len(names) * 0.32 + 1.5)),
    )

    bars = ax.barh(
        range(len(names)), scores,
        color=colors, edgecolor="white", linewidth=0.5, alpha=0.85,
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Sensitivity Score (Fisher / Hessian diagonal)")
    ax.set_title(
        f"Per-Layer Quantization Sensitivity — {model_name}\n"
        f"Top {len(names)} layers (sorted by score)",
        pad=12,
    )

    # Add tier legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=TIER_COLORS["HIGH"], label="HIGH — force INT8"),
        Patch(facecolor=TIER_COLORS["MEDIUM"], label="MEDIUM — NSGA decides"),
        Patch(facecolor=TIER_COLORS["LOW"], label="LOW — safe INT4"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", title="Tier",
              fontsize=9)

    # Score annotations on bars
    max_score = max(scores) if scores else 1.0
    for i, (bar, score) in enumerate(zip(bars, scores)):
        if score > max_score * 0.05:  # only annotate visible bars
            ax.text(
                bar.get_width() + max_score * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=7, color="#555555",
            )

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sensitivity_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved: %s", path.name)
    return str(path)


def plot_tier_distribution(
    cluster_result: Dict[str, Any],
    output_dir: str,
    *,
    model_name: str = "Generic CNN",
) -> Optional[str]:
    """Pie chart of layer count per sensitivity tier.

    A quick-glance summary showing what fraction of the model is in
    each quantization tier.
    """
    if not HAS_MATPLOTLIB:
        return None

    from neuroquant.visualization.style import apply_publication_style
    apply_publication_style()

    assignments = cluster_result.get("cluster_assignments", [])
    tier_counts: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for cluster in assignments:
        tier = cluster.get("tier", "MEDIUM")
        n_layers = len(cluster.get("layer_names", []))
        tier_counts[tier] = tier_counts.get(tier, 0) + n_layers

    if sum(tier_counts.values()) == 0:
        return None

    labels = list(tier_counts.keys())
    sizes = [tier_counts[t] for t in labels]
    colors = [TIER_COLORS.get(t, DEFAULT_COLOR) for t in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title(
        f"Layer Sensitivity Tier Distribution — {model_name}",
        pad=14,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tier_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved: %s", path.name)
    return str(path)


def _short_name(layer_name: str, max_len: int = 32) -> str:
    """Shorten a fully-qualified parameter name for display.

    Model-agnostic: trims trailing ``.weight`` / ``.bias`` (uninformative
    for axis labels — every entry has them) and elides the leading
    namespace with an ellipsis when the result still exceeds ``max_len``.
    Works for any architecture (ResNet ``layer1.0.conv1``, MobileNet
    ``features.18.conv.2``, transformer ``encoder.blocks.5.attn.q_proj``,
    detection ``backbone.body.layer3.0.bn2``, etc.).
    """
    name = layer_name
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    elif name.endswith(".bias"):
        name = name[: -len(".bias")]
    if len(name) <= max_len:
        return name
    # Keep the discriminative tail; layers usually share a long prefix.
    return "..." + name[-(max_len - 3):]
