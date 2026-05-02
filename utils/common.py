"""
NeuroQuant v2.0 - Common Utilities

Cross-cutting concerns: seed management, device selection,
timing, checkpoint I/O, and logging helpers.

No model-specific or dataset-specific assumptions.
"""

from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger("neuroquant")


def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug("Random seed set to %d", seed)


def get_device(preference: str = "auto") -> torch.device:
    """
    Resolve device string to torch.device.

    Args:
        preference: One of 'auto', 'cuda', 'cpu', 'mps'.
                    'auto' picks best available GPU.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


@contextmanager
def timer(label: str = ""):
    """Context manager to time a code block and log duration."""
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    if label:
        logger.info("%s: %.2fs", label, elapsed)
    else:
        logger.info("Elapsed: %.2fs", elapsed)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_model_size_mb(model: nn.Module) -> float:
    """Compute model size in MiB from the actual parameter dtypes (FP32)."""
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 * 1024)


def compute_ebops(
    model: nn.Module,
    bitwidth_assignment: Dict[str, int],
) -> float:
    """
    Compute Effective Bit Operations (EBops) for a mixed-precision model.

    EBops = sum(params × bitwidth) / 8  (bytes)

    Args:
        model: The PyTorch model.
        bitwidth_assignment: {param_name → bitwidth}.
    """
    total = 0.0
    for name, param in model.named_parameters():
        bw = bitwidth_assignment.get(name, 32)  # Default FP32
        total += param.numel() * bw / 8.0
    return total


def compute_quantized_size_mb(
    model: nn.Module,
    bitwidth_assignment: Dict[str, int],
) -> float:
    """Compute the on-disk model size in MiB under a bitwidth assignment.

    Sums ``numel × bitwidth / 8`` across all parameters (defaulting to
    32 bits for params not in the assignment) and converts to MiB
    (binary megabytes, ``1024 * 1024``). This is the canonical model
    size used as a Pareto objective and reported in the public outputs.
    """
    total_bytes = compute_ebops(model, bitwidth_assignment)
    return total_bytes / (1024 * 1024)


def model_size_mb_from_bytes(total_bytes: float) -> float:
    """Convenience wrapper: convert a byte count to MiB consistently."""
    return float(total_bytes) / (1024 * 1024)


def save_checkpoint(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint with optional metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if metadata:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, p)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint and return metadata."""
    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        metadata = checkpoint.get("metadata", {})
    else:
        # Bare state dict
        model.load_state_dict(checkpoint, strict=False)
        metadata = {}

    logger.info("Checkpoint loaded: %s", path)
    return metadata


def setup_logging(
    log_dir: str = "./logs",
    level: int = logging.INFO,
) -> None:
    """Configure logging for the framework."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    # File handler
    file_handler = logging.FileHandler(log_path / "neuroquant.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root_logger = logging.getLogger("neuroquant")
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
