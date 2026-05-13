"""
NeuroQuant — production-grade neural-network quantization framework.

Public API for library users::

    from neuroquant import (
        # Quantizers (notebook / library use)
        PTQQuantizer, AWQQuantizer, GPTQQuantizer,
        SmoothQuantQuantizer, SmoothQuantGPTQQuantizer,
        QATTrainer, AdaroundOptimizer,
        # Multi-objective search + clustering + surrogate
        NSGAIIClusterSearch, LayerClusterer, AccuracySurrogate,
        # Configuration object (every quantizer accepts ``config=None``
        # and falls back to ``QuantizationConfig()`` defaults)
        QuantizationConfig,
        # Explainability + Pareto visualization
        XAIGenerator, ParetoAnalyzer, ParetoVisualizer,
    )

The ``neuroquant`` command-line entry point lives in
:mod:`neuroquant.cli` and is exposed via ``[project.scripts]`` in
``pyproject.toml``. Library users normally do not need to import it
directly — instantiate :class:`PTQQuantizer` (etc.) and drive the
pipeline themselves.
"""

from __future__ import annotations

__version__ = "2.0.0"

# Re-export the configuration dataclass first because every other
# public symbol depends on it (directly or transitively).
from neuroquant.config import QuantizationConfig

# Quantizers and the search / clustering / surrogate trio. The
# subpackage __init__ already curates these — re-export from there so
# any future additions land in one place.
from neuroquant.quantization import (
    BaseQuantizer,
    PTQQuantizer,
    AWQQuantizer,
    GPTQQuantizer,
    SmoothQuantQuantizer,
    SmoothQuantGPTQQuantizer,
    AdaroundOptimizer,
    QATTrainer,
    NSGAIIClusterSearch,
    LayerClusterer,
    AccuracySurrogate,
)

# Visualization surface (Pareto + plot helpers). ``XAIGenerator`` is
# re-exported from the visualization package via a guarded import so
# users who skipped the optional ``xai`` extras (shap / captum) don't
# crash at ``import neuroquant``.
from neuroquant.visualization import (
    ParetoAnalyzer,
    ParetoVisualizer,
    XAIGenerator,
    compute_layer_errors,
    plot_error_attribution,
    plot_error_comparison,
    plot_sensitivity_heatmap,
    plot_tier_distribution,
    generate_html_report,
)

__all__ = [
    "__version__",
    # Configuration
    "QuantizationConfig",
    # Quantizers
    "BaseQuantizer",
    "PTQQuantizer",
    "AWQQuantizer",
    "GPTQQuantizer",
    "SmoothQuantQuantizer",
    "SmoothQuantGPTQQuantizer",
    "AdaroundOptimizer",
    "QATTrainer",
    # Search / clustering / surrogate
    "NSGAIIClusterSearch",
    "LayerClusterer",
    "AccuracySurrogate",
    # Visualization
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
