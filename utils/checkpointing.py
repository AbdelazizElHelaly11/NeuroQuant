"""
NeuroQuant v2.0 - Phase Checkpointing & Reproducibility

Provides:
    - Phase-level checkpoint save/load (resume after crash)
    - Reproducibility manifest (environment + config + results snapshot)

Checkpoint format:
    - .pth for model weights (torch.save)
    - .json for metadata/configs/results

All files go to: ./artifacts/checkpoints/
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("neuroquant")

CHECKPOINT_DIR = "checkpoints"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase Checkpoint Manager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CheckpointManager:
    """
    Manages per-phase checkpoints for pipeline resume support.

    Usage:
        mgr = CheckpointManager("./artifacts")
        if mgr.phase_exists("phase_1a"):
            data = mgr.load_phase("phase_1a")
        else:
            data = run_phase_1a(...)
            mgr.save_phase("phase_1a", data)
    """

    def __init__(self, output_dir: str, resume: bool = False) -> None:
        self.ckpt_dir = Path(output_dir) / CHECKPOINT_DIR
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume

    def phase_exists(self, phase_name: str) -> bool:
        """Check if a checkpoint exists for this phase."""
        json_path = self.ckpt_dir / f"{phase_name}.json"
        pth_path = self.ckpt_dir / f"{phase_name}.pth"
        return json_path.exists() or pth_path.exists()

    def should_skip(self, phase_name: str) -> bool:
        """Return True if resume is enabled AND checkpoint exists."""
        if not self.resume:
            return False
        exists = self.phase_exists(phase_name)
        if exists:
            logger.info(
                "  [RESUME] Checkpoint found for '%s' — skipping.",
                phase_name,
            )
        return exists

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_phase_json(self, phase_name: str, data: Dict[str, Any]) -> Path:
        """Save phase results as JSON (for configs, metrics, assignments)."""
        path = self.ckpt_dir / f"{phase_name}.json"
        # Convert numpy types for JSON serialization
        serializable = _make_serializable(data)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info("  [CKPT] Saved: %s", path.name)
        return path

    def save_phase_model(
        self,
        phase_name: str,
        model: nn.Module,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model weights + optional metadata as .pth checkpoint."""
        path = self.ckpt_dir / f"{phase_name}.pth"
        checkpoint = {"model_state_dict": model.state_dict()}
        if metadata:
            checkpoint["metadata"] = _make_serializable(metadata)
        torch.save(checkpoint, path)
        logger.info("  [CKPT] Saved: %s", path.name)
        return path

    def save_phase_full(
        self,
        phase_name: str,
        model: nn.Module,
        data: Dict[str, Any],
    ) -> None:
        """Save both model weights (.pth) and metadata (.json)."""
        self.save_phase_model(phase_name, model)
        self.save_phase_json(phase_name, data)

    def save_named_model(
        self,
        filename: str,
        model: nn.Module,
    ) -> Path:
        """Save a model state_dict under an arbitrary filename inside the
        checkpoint directory. Use for auxiliary per-phase artefacts (e.g.
        multiple quantized variants produced by a single phase)."""
        path = self.ckpt_dir / filename
        torch.save(model.state_dict(), path)
        logger.info("  [CKPT] Saved: %s", path.name)
        return path

    def load_named_model(
        self,
        filename: str,
        model: nn.Module,
    ) -> None:
        """Load a state_dict saved via save_named_model into the given model."""
        path = self.ckpt_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        logger.info("  [CKPT] Loaded model: %s", path.name)

    def file_exists(self, filename: str) -> bool:
        """Check whether an auxiliary checkpoint file exists."""
        return (self.ckpt_dir / filename).exists()

    def save_full_module(self, filename: str, module: nn.Module) -> Path:
        """Pickle an entire ``nn.Module`` (architecture + weights + buffers).

        Use for models whose architecture has been modified at quantization
        time (e.g. SmoothQuant inserts per-layer input-scaling wrappers).
        State-dict-only saves lose those wrappers on reload.
        """
        path = self.ckpt_dir / filename
        torch.save(module, path)
        logger.info("  [CKPT] Saved: %s (full module)", path.name)
        return path

    def load_full_module(self, filename: str) -> nn.Module:
        """Load an entire ``nn.Module`` previously saved with save_full_module."""
        path = self.ckpt_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint: {path}")
        module = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("  [CKPT] Loaded: %s (full module)", path.name)
        return module

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_phase_json(self, phase_name: str) -> Dict[str, Any]:
        """Load phase results from JSON checkpoint."""
        path = self.ckpt_dir / f"{phase_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        logger.info("  [CKPT] Loaded: %s", path.name)
        return data

    def load_phase_model(
        self,
        phase_name: str,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """Load model weights from .pth checkpoint. Returns metadata."""
        path = self.ckpt_dir / f"{phase_name}.pth"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Support both the {"model_state_dict": ...} envelope and a bare
        # state_dict (older phase_0 checkpoints use the bare form).
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint.get("metadata", {})
        else:
            state_dict = checkpoint
            metadata = {}
        model.load_state_dict(state_dict, strict=False)
        logger.info("  [CKPT] Loaded model: %s", path.name)
        return metadata

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Remove all checkpoints (fresh run)."""
        for f in self.ckpt_dir.glob("*"):
            f.unlink()
        logger.info("  [CKPT] Cleared all checkpoints.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reproducibility Manifest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def save_reproducibility_manifest(
    output_dir: str,
    config: Any,
    results: Dict[str, Any],
) -> Path:
    """
    Save a reproducibility manifest capturing everything needed
    to reproduce the experiment.

    Includes:
        - Python/PyTorch/CUDA versions
        - OS and GPU info
        - Config hash (for exact config matching)
        - Key result metrics
        - Timestamp and seed
    """
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "os": platform.platform(),
            "cpu": platform.processor(),
        },
        "config": {
            "model_name": getattr(config, "model_name", ""),
            "model_class": getattr(config, "model_class", ""),
            "dataset_name": getattr(config, "dataset_name", ""),
            "num_classes": getattr(config, "num_classes", 0),
            "input_shape": list(getattr(config, "input_shape", [])),
            "batch_size": getattr(config, "batch_size", 0),
            "seed": getattr(config.hyperparams, "seed", 42),
            "config_hash": _hash_config(config),
        },
        "results": {
            k: v for k, v in results.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        },
    }

    # Try to get installed package versions
    try:
        import importlib.metadata as importlib_metadata
        key_packages = [
            "torch", "torchvision", "numpy", "pandas",
            "matplotlib", "mlflow", "pymoo", "pyyaml",
        ]
        manifest["packages"] = {}
        for pkg in key_packages:
            try:
                manifest["packages"][pkg] = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                manifest["packages"][pkg] = "not_installed"
    except ImportError:
        pass

    path = Path(output_dir) / "reproducibility_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Reproducibility manifest saved: %s", path)
    return path


def _hash_config(config: Any) -> str:
    """Create a SHA256 hash of the config for exact-match verification."""
    try:
        config_str = json.dumps(
            {
                "model_name": getattr(config, "model_name", ""),
                "dataset_name": getattr(config, "dataset_name", ""),
                "num_classes": getattr(config, "num_classes", 0),
                "input_shape": list(getattr(config, "input_shape", [])),
                "batch_size": getattr(config, "batch_size", 0),
                "seed": getattr(config.hyperparams, "seed", 42),
            },
            sort_keys=True,
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_serializable(obj: Any) -> Any:
    """Convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, nn.Module):
        return "<nn.Module>"  # Don't serialize models to JSON
    return obj
