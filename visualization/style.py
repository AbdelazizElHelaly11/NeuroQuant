"""
NeuroQuant v2.0 — Shared publication plotting style.

Both the Pareto analysis and the XAI explainability modules import from
this file so a single edit changes the look of every figure the pipeline
emits. The style is deliberately light, high-contrast, and accessible:
white background, black text, light dashed grid, larger fonts, and a
consistent per-method colour/marker pair driven by ``METHOD_STYLE``.

Usage
-----

>>> from visualization.style import apply_publication_style, style_for
>>> apply_publication_style()
>>> color, marker = style_for("GPTQ_INT8")
"""

from __future__ import annotations

from typing import Tuple


# Method-family → (hex colour, matplotlib marker)
# Colours are colour-blind-safe (Tableau 10 / ColorBrewer-derived).
# Markers are picked so every family is distinguishable in monochrome.
METHOD_STYLE = {
    "FP32":        ("#000000", "D"),
    "PTQ":         ("#1f77b4", "o"),
    "QAT":         ("#ff7f0e", "s"),
    "GPTQ":        ("#2ca02c", "^"),
    "AWQ":         ("#9467bd", "P"),
    "SMOOTHQUANT": ("#d62728", "X"),
    "NSGA":        ("#1f77b4", "o"),   # NSGA solutions are tagged "PTQ"
}

DEFAULT_COLOR = "#7f7f7f"
DEFAULT_MARKER = "o"


def apply_publication_style() -> None:
    """Set matplotlib rcParams for a clean light publication theme.

    Idempotent: safe to call multiple times. Plot-emitting code should
    invoke this once before creating its figure.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe on servers
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 110,
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",

        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,

        # Text
        "text.color": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Grid
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.alpha": 0.6,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,

        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans", "Helvetica", "Arial", "sans-serif",
        ],
        "font.size": 11,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 10,
        "legend.borderpad": 0.6,

        # Lines / markers
        "lines.linewidth": 1.6,
        "lines.markersize": 7,
        "lines.markeredgewidth": 0.8,
    })


def _tokenize(name: str) -> Tuple[str, ...]:
    """Tokenize a method tag into uppercase words.

    Splits on common separators (``_``, ``-``, whitespace) so substring
    collisions like "PTQ" inside "GPTQ" never produce a false match.
    """
    if not name:
        return ()
    upper = str(name).upper()
    for sep in ("-", " ", "/"):
        upper = upper.replace(sep, "_")
    return tuple(t for t in upper.split("_") if t)


def style_for(method_name: str) -> Tuple[str, str]:
    """Return ``(hex_color, matplotlib_marker)`` for a given method tag.

    Token-aware: ``"GPTQ_INT8"`` resolves to GPTQ rather than PTQ even
    though "PTQ" is a substring of "GPTQ". Unknown families fall back
    to a neutral grey circle.
    """
    if not method_name:
        return (DEFAULT_COLOR, DEFAULT_MARKER)
    tokens = _tokenize(method_name)
    for tok in tokens:
        if tok in METHOD_STYLE:
            return METHOD_STYLE[tok]
    return (DEFAULT_COLOR, DEFAULT_MARKER)


def family_of(method_name: str) -> str:
    """Return the canonical family name for a method tag (e.g. ``"GPTQ"``)."""
    if not method_name:
        return "OTHER"
    tokens = _tokenize(method_name)
    for tok in tokens:
        if tok in METHOD_STYLE:
            return tok
    return "OTHER"
