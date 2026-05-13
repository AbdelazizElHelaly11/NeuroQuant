"""
NeuroQuant v2.0 — Per-Layer Quantization Error Attribution.

Bridges quantization analysis and XAI by answering: **where in the
network is accuracy lost?**  For each quantized method, this module
runs a forward pass through both the FP32 and quantized models,
captures per-layer activations, and computes error metrics:

  - **Layer MSE**: mean-squared error between FP32 and quantized
    activations at each layer output.
  - **Cosine similarity**: how well the quantized activation direction
    is preserved (1.0 = perfect, 0.0 = orthogonal).
  - **Relative error**: ``||q - fp32|| / ||fp32||`` — normalised
    magnitude of distortion.

The output is a stacked horizontal bar chart showing per-layer error
contribution, making it immediately clear which layers degrade most
from quantization.

Usage (called from main pipeline after Phase 1f)::

    from neuroquant.visualization.error_attribution import (
        compute_layer_errors, plot_error_attribution,
    )
    errors = compute_layer_errors(fp32_model, quant_model, data_loader, device)
    plot_error_attribution(errors, output_dir)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("neuroquant")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Error Computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_layer_errors(
    fp32_model: nn.Module,
    quant_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    num_batches: int = 5,
    layer_types: Tuple[type, ...] = (nn.Conv2d, nn.Linear),
) -> List[Dict[str, Any]]:
    """Compute per-layer activation errors between FP32 and quantized models.

    Hooks into all layers of type ``layer_types`` and captures their
    output activations during a forward pass. Then computes MSE, cosine
    similarity, and relative error for each.

    Args:
        fp32_model:   The original FP32 model.
        quant_model:  The quantized model to compare against.
        data_loader:  Calibration/validation data.
        device:       Compute device.
        num_batches:  How many batches to average over.
        layer_types:  Which layer types to hook.

    Returns:
        List of dicts, one per layer, sorted by MSE (highest first):
        ``[{name, layer_type, mse, cosine_sim, relative_error}, ...]``
    """
    fp32_model = fp32_model.to(device).eval()
    quant_model = quant_model.to(device).eval()

    # Identify matching layers by name
    fp32_layers: Dict[str, nn.Module] = {}
    quant_layers: Dict[str, nn.Module] = {}

    for name, m in fp32_model.named_modules():
        if isinstance(m, layer_types):
            fp32_layers[name] = m

    for name, m in quant_model.named_modules():
        if isinstance(m, layer_types):
            quant_layers[name] = m

    # Find common layers
    common_names = sorted(set(fp32_layers.keys()) & set(quant_layers.keys()))
    if not common_names:
        # Try fuzzy matching by stripping wrapper prefixes
        common_names = _fuzzy_match_layers(fp32_layers, quant_layers)

    if not common_names:
        logger.warning("No common layers found for error attribution.")
        return []

    # Register hooks
    fp32_acts: Dict[str, List[torch.Tensor]] = {n: [] for n in common_names}
    quant_acts: Dict[str, List[torch.Tensor]] = {n: [] for n in common_names}

    fp32_hooks = []
    quant_hooks = []

    for name in common_names:
        if name in fp32_layers:
            h = fp32_layers[name].register_forward_hook(
                _make_hook(fp32_acts, name)
            )
            fp32_hooks.append(h)
        if name in quant_layers:
            h = quant_layers[name].register_forward_hook(
                _make_hook(quant_acts, name)
            )
            quant_hooks.append(h)

    # Forward pass
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            fp32_model(x)
            quant_model(x)

    # Remove hooks
    for h in fp32_hooks + quant_hooks:
        h.remove()

    # Compute errors
    results: List[Dict[str, Any]] = []
    for name in common_names:
        if not fp32_acts[name] or not quant_acts[name]:
            continue

        fp32_cat = torch.cat([a.detach().cpu().float() for a in fp32_acts[name]])
        quant_cat = torch.cat([a.detach().cpu().float() for a in quant_acts[name]])

        # Truncate to same batch count
        n = min(fp32_cat.shape[0], quant_cat.shape[0])
        fp32_cat = fp32_cat[:n]
        quant_cat = quant_cat[:n]

        # Flatten
        fp32_flat = fp32_cat.reshape(n, -1)
        quant_flat = quant_cat.reshape(n, -1)

        # MSE
        mse = float(torch.mean((fp32_flat - quant_flat) ** 2))

        # Cosine similarity (average over batch)
        cos = torch.nn.functional.cosine_similarity(fp32_flat, quant_flat, dim=1)
        cosine_sim = float(torch.mean(cos))

        # Relative error
        fp32_norm = float(torch.norm(fp32_flat).item())
        diff_norm = float(torch.norm(fp32_flat - quant_flat).item())
        rel_error = diff_norm / max(fp32_norm, 1e-8)

        layer_type = type(fp32_layers.get(name, quant_layers.get(name))).__name__

        results.append({
            "name": name,
            "layer_type": layer_type,
            "mse": mse,
            "cosine_similarity": cosine_sim,
            "relative_error": rel_error,
        })

    # Sort by MSE descending
    results.sort(key=lambda r: r["mse"], reverse=True)
    return results


def _make_hook(storage: Dict[str, List], name: str):
    """Create a forward hook that stores the output activation."""
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            storage[name].append(output.detach())
        elif isinstance(output, (tuple, list)) and output:
            storage[name].append(output[0].detach())
    return hook


def _fuzzy_match_layers(
    fp32_layers: Dict[str, nn.Module],
    quant_layers: Dict[str, nn.Module],
) -> List[str]:
    """Try to match layers with different naming (e.g. wrapper prefixes)."""
    common = []
    for qname in quant_layers:
        for fname in fp32_layers:
            if fname.endswith(qname) or qname.endswith(fname):
                common.append(fname)
                break
    return sorted(set(common))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def plot_error_attribution(
    errors: List[Dict[str, Any]],
    output_dir: str,
    *,
    top_n: int = 25,
    method_name: str = "",
    model_name: str = "Generic CNN",
) -> Optional[str]:
    """Plot per-layer quantization error as a horizontal bar chart.

    Layers are sorted by MSE (highest error at top). Bar color encodes
    cosine similarity (dark red = low similarity = high distortion,
    dark green = high similarity = well-preserved).

    Args:
        errors:       Output of ``compute_layer_errors``.
        output_dir:   Where to save the plot.
        top_n:        Show at most this many layers.
        method_name:  Quantization method name for the title.
        model_name:   Model name for the title.

    Returns:
        Path to the saved PNG, or None if matplotlib is missing.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available; skipping error attribution plot.")
        return None

    if not errors:
        logger.warning("No error data; skipping error attribution plot.")
        return None

    from neuroquant.visualization.style import apply_publication_style
    apply_publication_style()

    top_errors = errors[:top_n]
    # Reverse for bar chart (highest at top)
    top_errors = list(reversed(top_errors))

    names = [_short_layer_name(e["name"]) for e in top_errors]
    mses = [e["mse"] for e in top_errors]
    cosines = [e["cosine_similarity"] for e in top_errors]

    # Color based on cosine similarity (red=bad, green=good)
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn  # Red → Yellow → Green
    norm = mcolors.Normalize(vmin=0.8, vmax=1.0)
    colors = [cmap(norm(c)) for c in cosines]

    fig, ax = plt.subplots(
        figsize=(11, max(5, len(names) * 0.35 + 2)),
    )

    bars = ax.barh(
        range(len(names)), mses,
        color=colors, edgecolor="white", linewidth=0.5, alpha=0.9,
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Squared Error (MSE)")

    title = f"Quantization Error Attribution — {model_name}"
    if method_name:
        title += f"\n{method_name}"
    ax.set_title(title, pad=12)

    # Annotations: relative error %
    max_mse = max(mses) if mses else 1.0
    for i, (bar, err) in enumerate(zip(bars, top_errors)):
        rel = err["relative_error"] * 100
        cos = err["cosine_similarity"]
        if err["mse"] > max_mse * 0.03:
            ax.text(
                bar.get_width() + max_mse * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"rel={rel:.1f}%  cos={cos:.4f}",
                va="center", fontsize=7, color="#555555",
            )

    # Colorbar for cosine similarity
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{method_name.lower().replace(' ', '_')}" if method_name else ""
    path = out / f"error_attribution{suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved: %s", path.name)
    return str(path)


def plot_error_comparison(
    all_errors: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
    *,
    top_n: int = 15,
    model_name: str = "Generic CNN",
) -> Optional[str]:
    """Compare per-layer errors across multiple quantization methods.

    Generates a grouped bar chart where each group is a layer and each
    bar within the group is a different method's MSE.

    Args:
        all_errors:  ``{method_name: [error_dicts]}``.
        output_dir:  Where to save the plot.
        top_n:       Show at most this many layers (by max MSE across methods).
        model_name:  Model name for the title.

    Returns:
        Path to saved PNG, or None.
    """
    if not HAS_MATPLOTLIB or not all_errors:
        return None

    from neuroquant.visualization.style import apply_publication_style, style_for
    apply_publication_style()

    # Collect all unique layer names, sort by max MSE across all methods
    layer_mse: Dict[str, float] = {}
    for method_name, errors in all_errors.items():
        for e in errors:
            name = e["name"]
            layer_mse[name] = max(layer_mse.get(name, 0), e["mse"])

    top_layers = sorted(layer_mse.keys(), key=lambda n: layer_mse[n], reverse=True)[:top_n]
    top_layers = list(reversed(top_layers))  # highest at top

    methods = list(all_errors.keys())
    n_methods = len(methods)
    n_layers = len(top_layers)

    if n_layers == 0 or n_methods == 0:
        return None

    fig, ax = plt.subplots(
        figsize=(12, max(5, n_layers * 0.4 + 2)),
    )

    bar_height = 0.8 / n_methods
    for j, method in enumerate(methods):
        error_map = {e["name"]: e["mse"] for e in all_errors[method]}
        mses = [error_map.get(ln, 0.0) for ln in top_layers]
        positions = [i + j * bar_height for i in range(n_layers)]
        color, _ = style_for(method)
        ax.barh(
            positions, mses, height=bar_height,
            color=color, alpha=0.8, label=method,
            edgecolor="white", linewidth=0.3,
        )

    ax.set_yticks([i + bar_height * (n_methods - 1) / 2 for i in range(n_layers)])
    ax.set_yticklabels([_short_layer_name(n) for n in top_layers], fontsize=8)
    ax.set_xlabel("Mean Squared Error (MSE)")
    ax.set_title(
        f"Cross-Method Error Comparison — {model_name}\n"
        f"Top {n_layers} most-affected layers",
        pad=12,
    )
    ax.legend(loc="lower right", title="Method", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "error_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved: %s", path.name)
    return str(path)


def _short_layer_name(name: str, max_len: int = 35) -> str:
    """Shorten a layer name for readability in plots.

    Model-agnostic: keeps the full module path but elides the leading
    namespace with an ellipsis when it exceeds ``max_len``. Works for
    any architecture without baking in MobileNet / ResNet / transformer
    naming conventions.
    """
    if len(name) <= max_len:
        return name
    return "..." + name[-(max_len - 3):]
