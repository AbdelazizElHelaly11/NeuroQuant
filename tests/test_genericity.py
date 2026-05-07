"""
NeuroQuant v2.0 - Genericity Verification Test

Verifies that the framework works on 3 different architectures
without any hardcoded assumptions.

Test cases:
    1. MobileNetV2 + CIFAR-10 (original)
    2. ResNet18 + CIFAR-10
    3. Custom small CNN + synthetic data

Each test runs Phase 0 (model + data) → Phase 1a (Hessian) only,
verifying model loading, adaptation, and quantization compatibility.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from config import QuantizationConfig
from models.model_loader import ModelLoader, adapt_classifier, adapt_input_conv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("neuroquant.test_genericity")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom small CNN for test case 3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TinyCNN(nn.Module):
    """Minimal CNN for genericity testing. No architecture-specific names."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _verify_model(
    model: nn.Module,
    input_shape: tuple,
    num_classes: int,
    label: str,
) -> bool:
    """Verify model can do a forward pass and produces correct output shape."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dummy = torch.randn(2, *input_shape, device=device)
    with torch.no_grad():
        output = model(dummy)

    ok = True
    if output.shape != (2, num_classes):
        logger.error(
            "  [%s] FAIL: output shape %s, expected (2, %d)",
            label, output.shape, num_classes,
        )
        ok = False
    else:
        logger.info("  [%s] OK: forward pass → output %s", label, output.shape)

    # Verify quantizable layers exist
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    logger.info("  [%s] Layers: %d Conv2d, %d Linear", label, n_conv, n_linear)

    if n_conv == 0 and n_linear == 0:
        logger.error("  [%s] FAIL: no quantizable layers found", label)
        ok = False

    return ok


def _verify_phase_1a(model: nn.Module, config: QuantizationConfig, label: str) -> bool:
    """Verify Phase 1a (Hessian + clustering) works on the model."""
    from quantization.hessian_clustering import HessianComputer, LayerClusterer
    from data.data_loader import GenericDatasetLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Build calibration data
    loader = GenericDatasetLoader(config)
    calib = loader.get_calibration_loader(num_batches=1)

    # Hessian
    hessian = HessianComputer(model, config)
    diag = hessian.compute_hessian(calib, nn.CrossEntropyLoss(), num_batches=1)

    if not diag:
        logger.error("  [%s] FAIL: Hessian returned empty", label)
        return False
    logger.info("  [%s] Hessian: %d layers analysed", label, len(diag))

    # Clustering
    clusterer = LayerClusterer(model, diag, config)
    result = clusterer.create_clusters()
    n_clusters = len(result["cluster_assignments"])
    logger.info("  [%s] Clusters: %d assignments", label, n_clusters)

    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_mobilenetv2() -> bool:
    """Test 1: MobileNetV2 + CIFAR-10 (32x32) — the original use case."""
    logger.info("─" * 60)
    logger.info("TEST 1: MobileNetV2 + CIFAR-10 (32x32, 10 classes)")
    logger.info("─" * 60)

    config = QuantizationConfig()
    config.model_name = "mobilenetv2"
    config.dataset_name = "cifar10"
    config.num_classes = 10
    config.input_shape = (3, 32, 32)
    config.batch_size = 16
    config.num_workers = 0

    model = ModelLoader(config).load()
    ok = _verify_model(model, config.input_shape, config.num_classes, "MobileNetV2")
    if ok:
        ok = _verify_phase_1a(model, config, "MobileNetV2")
    return ok


def test_resnet18() -> bool:
    """Test 2: ResNet18 + CIFAR-10 (32x32) — different architecture."""
    logger.info("─" * 60)
    logger.info("TEST 2: ResNet18 + CIFAR-10 (32x32, 10 classes)")
    logger.info("─" * 60)

    config = QuantizationConfig()
    config.model_name = "resnet18"
    config.dataset_name = "cifar10"
    config.num_classes = 10
    config.input_shape = (3, 32, 32)
    config.batch_size = 16
    config.num_workers = 0

    model = ModelLoader(config).load()
    ok = _verify_model(model, config.input_shape, config.num_classes, "ResNet18")
    if ok:
        ok = _verify_phase_1a(model, config, "ResNet18")
    return ok


def test_custom_cnn() -> bool:
    """Test 3: Custom TinyCNN + synthetic data — no torchvision at all."""
    logger.info("─" * 60)
    logger.info("TEST 3: TinyCNN + Synthetic (32x32, 5 classes)")
    logger.info("─" * 60)

    config = QuantizationConfig()
    config.model_name = ""
    config.model_class = "test_genericity.TinyCNN"
    config.dataset_name = "synthetic"
    config.num_classes = 5
    config.input_shape = (3, 32, 32)
    config.batch_size = 16
    config.num_workers = 0

    model = ModelLoader(config).load()
    ok = _verify_model(model, config.input_shape, config.num_classes, "TinyCNN")
    if ok:
        ok = _verify_phase_1a(model, config, "TinyCNN")
    return ok


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    t_start = time.time()
    results = {}

    results["MobileNetV2"] = test_mobilenetv2()
    results["ResNet18"] = test_resnet18()
    results["TinyCNN"] = test_custom_cnn()

    elapsed = time.time() - t_start

    print("\n")
    print("=" * 60)
    print("  Genericity Verification Summary")
    print("=" * 60)
    for name, passed in results.items():
        marker = "[PASS]" if passed else "[FAIL]"
        print(f"  {marker} {name}")
    print(f"\n  Total time: {elapsed:.1f}s")
    all_passed = all(results.values())
    print(f"  Status: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
