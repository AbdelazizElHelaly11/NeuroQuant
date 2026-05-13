"""
NeuroQuant v2.0 — Quantization package public API.

This module re-exports the user-facing quantizer classes and the
multi-objective search so callers can write the librosa-style flat
import path:

    from neuroquant.quantization import (
        PTQQuantizer,
        AWQQuantizer,
        GPTQQuantizer,
        SmoothQuantQuantizer,
        SmoothQuantGPTQQuantizer,
        NSGAIIClusterSearch,
    )

The pipeline modules (``main.py``) still import from the submodules
directly when they need internal helpers. The flat API is intended for
notebooks, library consumers, and external integrators.

Every quantizer accepts an optional :class:`QuantizationConfig`. When
omitted it falls back to ``QuantizationConfig()``'s defaults so a user
can run, e.g.::

    quantizer = PTQQuantizer(my_model)
    quantized = quantizer.quantize(calib_loader, bitwidth=4)

without ever touching ``config.yaml``.
"""

from __future__ import annotations

from neuroquant.quantization.base import BaseQuantizer
from neuroquant.quantization.ptq import PTQQuantizer
from neuroquant.quantization.awq import AWQQuantizer
from neuroquant.quantization.gptq import GPTQQuantizer
from neuroquant.quantization.smoothquant import SmoothQuantQuantizer
from neuroquant.quantization.smoothquant_gptq import SmoothQuantGPTQQuantizer
from neuroquant.quantization.adaround import AdaroundOptimizer
from neuroquant.quantization.qat import QATTrainer
from neuroquant.quantization.nsga_ii_search import NSGAIIClusterSearch
from neuroquant.quantization.hessian_clustering import LayerClusterer
from neuroquant.quantization.surrogate import AccuracySurrogate

__all__ = [
    "BaseQuantizer",
    "PTQQuantizer",
    "AWQQuantizer",
    "GPTQQuantizer",
    "SmoothQuantQuantizer",
    "SmoothQuantGPTQQuantizer",
    "AdaroundOptimizer",
    "QATTrainer",
    "NSGAIIClusterSearch",
    "LayerClusterer",
    "AccuracySurrogate",
]
