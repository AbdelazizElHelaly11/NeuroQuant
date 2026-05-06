"""
NeuroQuant v2.0 — Centralized numerical-stability constants.

A single home for the small floor values scattered across the framework.
Pinning them in one place makes ports to lower-precision baselines (FP16,
BF16) tractable: the framework's behaviour is determined by these eight
constants and nothing else, so a downstream consumer can override them
once at process start instead of patching every call site.

Naming convention:
    EPS_*    — tiny floor values used as additive ε to prevent log/div by 0.
    MIN_*    — clamp floors used to bound a value away from zero.
    MAX_*    — clamp ceilings used to bound a value away from infinity.

Conventions for choosing a constant:
    * Histogram / probability operations → EPS_PROB (1e-12). FP32-safe.
    * Quantizer scales / amax floors      → MIN_SCALE (1e-8). Matches
      the resolution of an INT8 step in normalised space.
    * Hessian damping / Cholesky          → MIN_DAMP (1e-6). Empirically
      stable for GPTQ-style inverse-Hessian solves.
    * SmoothQuant migration scales        → MIN_SCALE / MAX_SCALE pair.

Override at process start:
    >>> from utils import numerics
    >>> numerics.MIN_SCALE = 1e-5  # if you port to FP16 weights

Or via env var:
    NEUROQUANT_MIN_SCALE=1e-5
which is read once at import time below.
"""
from __future__ import annotations

import os
from typing import Final


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment with a typed default fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ── Probability / log-domain ε ──────────────────────────────────────────────
# Used as an additive ε on histograms, KL terms, log-probabilities, and any
# other value that goes through ``log()`` or appears in a denominator that
# could be exactly zero. FP32 safe; do NOT lower below 1e-15.
EPS_PROB: Final[float] = _env_float("NEUROQUANT_EPS_PROB", 1e-12)

# ── Quantizer scale floor ───────────────────────────────────────────────────
# Floor for symmetric scales ``s = amax / qmax`` and for ``amax`` itself
# before division. Below this, weight tensors are treated as zero (no
# quantization error to recover). Tracks the smallest meaningful FP32 value
# at INT8 granularity.
MIN_SCALE: Final[float] = _env_float("NEUROQUANT_MIN_SCALE", 1e-8)

# ── Hessian / second-order damping ──────────────────────────────────────────
# Floor for diagonal damping in GPTQ-style inverse-Hessian solves.
# 1e-6 is the value cited in the GPTQ paper; below this Cholesky becomes
# unstable on ill-conditioned layers.
MIN_DAMP: Final[float] = _env_float("NEUROQUANT_MIN_DAMP", 1e-6)

# ── SmoothQuant migration-scale clamp ──────────────────────────────────────
# Two-sided floor/ceiling on the per-channel migration scale ``s_j``. Wider
# than the quantizer floor on purpose: SmoothQuant scales typically span
# 0.01–100, and clamping too aggressively neuters the migration.
MIN_MIGRATION: Final[float] = _env_float("NEUROQUANT_MIN_MIGRATION", 1e-4)
MAX_MIGRATION: Final[float] = _env_float("NEUROQUANT_MAX_MIGRATION", 1e4)

# ── Geometric / plotting tolerances ─────────────────────────────────────────
# Used by visualization helpers when normalising distances or detecting
# degenerate (zero-length) lines. Not numerically critical — only affects
# plot positioning.
EPS_GEOMETRIC: Final[float] = _env_float("NEUROQUANT_EPS_GEOMETRIC", 1e-8)


__all__ = (
    "EPS_PROB",
    "MIN_SCALE",
    "MIN_DAMP",
    "MIN_MIGRATION",
    "MAX_MIGRATION",
    "EPS_GEOMETRIC",
)
