"""
NeuroQuant v2.0 — Visualization package public API.

Flat re-exports for the user-facing plotters and analyzers so callers
can write::

    from neuroquant.visualization import (
        ParetoAnalyzer,
        ParetoVisualizer,
        XAIGenerator,
        plot_error_attribution,
        plot_error_comparison,
        plot_sensitivity_heatmap,
        plot_tier_distribution,
        generate_html_report,
    )

``XAIGenerator`` is logically a "visualization" concept (Grad-CAM /
SHAP heatmaps) even though its implementation lives in the ``xai``
package — re-exporting it here gives consumers a single import point
for every plot-producing class without having to know the internal
layout.
"""

from __future__ import annotations

from neuroquant.visualization.pareto_analysis import (
    ParetoAnalyzer,
    ParetoVisualizer,
)
from neuroquant.visualization.error_attribution import (
    compute_layer_errors,
    plot_error_attribution,
    plot_error_comparison,
)
from neuroquant.visualization.sensitivity import (
    plot_sensitivity_heatmap,
    plot_tier_distribution,
)
from neuroquant.visualization.report import generate_html_report

# XAI lives in its own top-level package but conceptually belongs to
# the visualization surface area. Import it under a try/except so that
# users who skipped the optional ``xai`` extras (shap, captum) don't
# crash at ``import visualization``.
try:  # pragma: no cover - optional dependency path
    from neuroquant.xai.explainability import XAIGenerator
except Exception:  # noqa: BLE001 — XAI is an optional surface
    XAIGenerator = None  # type: ignore[assignment]

__all__ = [
    "ParetoAnalyzer",
    "ParetoVisualizer",
    "XAIGenerator",
    "compute_layer_errors",
    "plot_error_attribution",
    "plot_error_comparison",
    "plot_sensitivity_heatmap",
    "plot_tier_distribution",
    "generate_html_report",
]
