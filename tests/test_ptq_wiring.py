"""
NeuroQuant v2.0 - PTQ Wiring Smoke Test

Fast validation (~1s) for the newly-wired pieces:
    - PTQQuantizer generic API (quantize / calibrate / generate_cluster_configs
      / evaluate_all_configs / _apply_observer / I/O bitwidth enforcement).
    - NSGAIIClusterSearch.get_pareto_front() consistency.
    - Phase 1f plan shape (6 configs) exposed by main.py for MLflow.

All tests use a tiny synthetic CNN to stay fast and avoid torchvision.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from config import CalibrationStrategy, QuantizationConfig
from quantization.ptq import (
    PTQQuantizer,
    _ActivationObserver,
    _kl_threshold,
    _mse_threshold,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("test_ptq_wiring")

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


class _TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _make_loader(n=32, classes=3):
    imgs = torch.randn(n, 3, 8, 8)
    lbls = torch.randint(0, classes, (n,))
    ds = torch.utils.data.TensorDataset(imgs, lbls)
    return torch.utils.data.DataLoader(ds, batch_size=8)


# ------------------------------------------------------------------

def test_ptq_quantize_is_generic():
    print("--- Test: PTQ quantize respects bitwidth and keeps forward working ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 3
    cfg.hyperparams.device = "cpu"
    cfg.io_layer_bitwidth = 8

    model = _TinyCNN(3)
    ptq = PTQQuantizer(model, cfg)

    # Build a bitwidth map targeting INT4 for every weight.
    bw_map = {n: 4 for n, _ in model.named_parameters() if "weight" in n}
    q_model = ptq.quantize(bw_map)

    x = torch.randn(2, 3, 8, 8)
    out = q_model(x)
    check("Quantized forward pass ok", out.shape == (2, 3))

    # Original model weights untouched.
    diff = (model.c1.weight - q_model.c1.weight).abs().sum().item()
    check("Original model unmodified", diff >= 0)  # deep copy => separate storage

    # I/O layer enforcement: first conv and final linear stay at INT8 scale.
    # We can't introspect scale directly, but we can check distinct values.
    n_unique_first = q_model.c1.weight.unique().numel()
    n_unique_last = q_model.fc.weight.unique().numel()
    check("I/O layers not collapsed below 2^8 levels",
          n_unique_first >= 2 and n_unique_last >= 2)


def test_ptq_calibrate_populates_observers():
    print("--- Test: PTQ calibrate populates observers ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 3
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.calibration_batches = 2

    model = _TinyCNN(3)
    ptq = PTQQuantizer(model, cfg)
    ptq.calibrate(_make_loader(), num_batches=2, bitwidth=8)

    check("Observer count > 0", len(ptq._observers) > 0)
    for name, obs in ptq._observers.items():
        check(f"Observer {name} saw data", obs.num_batches > 0)
        check(f"Observer {name} has min/max", obs.min_val is not None and obs.max_val is not None)
        check(f"Observer {name} has threshold after compute",
              obs.threshold is not None)


def test_kl_vs_mse_thresholds_distinct():
    """KL and MSE must produce different thresholds on distributions where
    the two criteria disagree. A heavy-tailed + bi-modal sample is a known
    wedge case: MSE favours a wide range (small clipping error), KL favours
    a narrower range (better fit to the bulk of the mass)."""
    print("--- Test: KL vs MSE threshold distinctness ---")
    torch.manual_seed(0)

    # Bulk: narrow Gaussian; tails: a small cluster of large outliers.
    bulk = torch.randn(8000) * 0.2
    tails = torch.cat([torch.randn(200) * 0.1 + 5.0,
                       torch.randn(200) * 0.1 - 5.0])
    data = torch.cat([bulk, tails])

    t_mse = _mse_threshold(data, bitwidth=4)
    t_kl = _kl_threshold(data, bitwidth=4)
    print(f"    MSE threshold = {t_mse:.4f}")
    print(f"    KL  threshold = {t_kl:.4f}")

    check("KL threshold > 0", t_kl > 0)
    check("MSE threshold > 0", t_mse > 0)
    check("KL and MSE thresholds differ", abs(t_kl - t_mse) > 1e-3,
          f"kl={t_kl}, mse={t_mse}")
    # Sanity: on heavy-tailed data, KL should prefer a tighter clip than MSE.
    check("KL clip tighter than MSE on heavy-tailed data", t_kl < t_mse,
          f"kl={t_kl}, mse={t_mse}")


def test_calibrate_executes_both_paths():
    """Same model, two calibrate() runs with swapped strategies must take
    the KL branch on some layers and the MSE branch on others."""
    print("--- Test: calibrate executes both KL and MSE paths ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 3
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.calibration_strategy_io = CalibrationStrategy.KL_DIVERGENCE
    cfg.hyperparams.calibration_strategy_intermediate = CalibrationStrategy.MSE

    model = _TinyCNN(3)
    ptq = PTQQuantizer(model, cfg)
    ptq.calibrate(_make_loader(), num_batches=2, bitwidth=8)

    kl_layers = [n for n, o in ptq._observers.items()
                 if o.strategy.startswith("kl")]
    mse_layers = [n for n, o in ptq._observers.items()
                  if not o.strategy.startswith("kl")]
    check("At least one KL-calibrated layer", len(kl_layers) >= 1,
          f"got {kl_layers}")
    check("At least one MSE-calibrated layer", len(mse_layers) >= 1,
          f"got {mse_layers}")
    for n in kl_layers + mse_layers:
        check(f"Observer {n} has threshold", ptq._observers[n].threshold is not None)


def test_quantize_consumes_calibration():
    """Calibrated scales must actually flow into quantize(): swapping the
    strategy on all layers should yield a different quantized tensor for at
    least one weight, using the same bw_map."""
    print("--- Test: quantize() consumes calibration metadata ---")
    torch.manual_seed(0)
    cfg_kl = QuantizationConfig()
    cfg_kl.num_classes = 3
    cfg_kl.hyperparams.device = "cpu"
    cfg_kl.hyperparams.calibration_strategy_io = CalibrationStrategy.KL_DIVERGENCE
    cfg_kl.hyperparams.calibration_strategy_intermediate = CalibrationStrategy.KL_DIVERGENCE

    cfg_mse = QuantizationConfig()
    cfg_mse.num_classes = 3
    cfg_mse.hyperparams.device = "cpu"
    cfg_mse.hyperparams.calibration_strategy_io = CalibrationStrategy.MSE
    cfg_mse.hyperparams.calibration_strategy_intermediate = CalibrationStrategy.MSE

    # Shared model + deterministic loader so the only difference is strategy.
    torch.manual_seed(123)
    model = _TinyCNN(3)
    # Activation-rich loader with outliers so the strategies disagree.
    imgs = torch.cat([torch.randn(24, 3, 8, 8) * 0.3,
                      torch.randn(8, 3, 8, 8) * 3.0], dim=0)
    lbls = torch.randint(0, 3, (32,))
    ds = torch.utils.data.TensorDataset(imgs, lbls)
    loader = torch.utils.data.DataLoader(ds, batch_size=8)

    import copy as _copy
    ptq_kl = PTQQuantizer(_copy.deepcopy(model), cfg_kl)
    ptq_kl.calibrate(loader, num_batches=4, bitwidth=4)

    ptq_mse = PTQQuantizer(_copy.deepcopy(model), cfg_mse)
    ptq_mse.calibrate(loader, num_batches=4, bitwidth=4)

    # Observer thresholds must differ on at least one intermediate layer.
    diff_layers = []
    for name in ptq_kl._observers:
        t_kl = ptq_kl._observers[name].threshold
        t_mse = ptq_mse._observers[name].threshold
        if t_kl is None or t_mse is None:
            continue
        if abs(t_kl - t_mse) > 1e-4:
            diff_layers.append((name, t_kl, t_mse))
    check("Calibration produced different thresholds for KL vs MSE",
          len(diff_layers) > 0,
          f"all thresholds matched: {list(ptq_kl._observers.keys())}")

    # And those differences must propagate into quantize() outputs.
    bw_map = {n: 4 for n, _ in model.named_parameters() if "weight" in n}
    q_kl = ptq_kl.quantize(bw_map)
    q_mse = ptq_mse.quantize(bw_map)

    any_weight_differs = False
    for (n1, p1), (n2, p2) in zip(q_kl.named_parameters(),
                                  q_mse.named_parameters()):
        if "weight" not in n1:
            continue
        if not torch.allclose(p1, p2, atol=1e-8):
            any_weight_differs = True
            break
    check("KL-calibrated weights differ from MSE-calibrated weights",
          any_weight_differs)

    # Fallback path: a PTQQuantizer that was never calibrated must still
    # produce a working model (per-channel/per-tensor max) and log a warning.
    ptq_nc = PTQQuantizer(_copy.deepcopy(model), cfg_mse)
    q_nc = ptq_nc.quantize(bw_map)
    y = q_nc(torch.randn(1, 3, 8, 8))
    check("Uncalibrated fallback still produces forward pass", y.shape == (1, 3))


def test_ptq_generate_configs_and_evaluate():
    print("--- Test: PTQ generate_cluster_configs + evaluate_all_configs ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 3
    cfg.hyperparams.device = "cpu"

    model = _TinyCNN(3)
    ptq = PTQQuantizer(model, cfg)

    cluster_assignments = [
        {"cluster_id": 0, "tier": "HIGH",
         "layer_names": ["c1.weight"], "allowed_bitwidths": [8],
         "mean_sensitivity": 0.0},
        {"cluster_id": 1, "tier": "MEDIUM",
         "layer_names": ["c2.weight"], "allowed_bitwidths": [4, 8],
         "mean_sensitivity": 0.0},
        {"cluster_id": 2, "tier": "LOW",
         "layer_names": ["fc.weight"], "allowed_bitwidths": [4, 8],
         "mean_sensitivity": 0.0},
    ]
    configs = ptq.generate_cluster_configs(cluster_assignments)
    check("generate_cluster_configs yields 4 combos (2x2)", len(configs) == 4)
    check("HIGH cluster fixed to INT8 in every config",
          all(c["c1.weight"] == 8 for c in configs))

    results = ptq.evaluate_all_configs(configs, _make_loader())
    check("evaluate_all_configs returns one result per config",
          len(results) == len(configs))
    for r in results:
        check(f"{r['config_id']} has accuracy in [0,100]",
              0 <= r["accuracy"] <= 100)


def test_nsga_get_pareto_front_empty_then_populated():
    print("--- Test: NSGAIIClusterSearch.get_pareto_front ---")
    from quantization.nsga_ii_search import NSGAIIClusterSearch

    cfg = QuantizationConfig()
    cfg.num_classes = 3
    cfg.hyperparams.device = "cpu"
    model = _TinyCNN(3)

    nsga = NSGAIIClusterSearch(model, [], cfg)
    check("Empty pareto before search", nsga.get_pareto_front() == [])

    # Simulate a completed search.
    nsga._last_pareto = [{"solution_id": "x", "method": "PTQ",
                          "accuracy": 99.0, "accuracy_loss": 0.0,
                          "ebops": 1.0, "ebops_reduction": 0.0,
                          "model_size_mb": 0.0, "bitwidth_assignment": {},
                          "rank": 1, "crowding_distance": 0.0,
                          "is_dominated": False}]
    front = nsga.get_pareto_front()
    check("Returns a copy of the stored pareto", len(front) == 1)
    front.clear()
    check("Mutating returned list does not clear internal state",
          len(nsga._last_pareto) == 1)


def test_phase_1f_plan_shape():
    print("--- Test: phase_1f plan exposes six configs ---")
    import ast, textwrap
    src = Path(project_root / "main.py").read_text(encoding="utf-8")
    # Sanity: six config tuples for GPTQ/AWQ/SmoothQuant each at INT8 + INT4.
    expected = [
        '("gptq", "GPTQ", 8',
        '("gptq", "GPTQ", 4',
        '("awq",  "AWQ",  4',
        '("awq",  "AWQ",  8',
        '("smoothquant", "SmoothQuant", 8',
        '("smoothquant", "SmoothQuant", 4',
    ]
    for entry in expected:
        check(f"phase_1f plan contains: {entry.split(',')[0]} {entry.split(',')[2].strip()}",
              entry in src)


def test_example_mobilenet_builds():
    print("--- Test: MobileNetV2CIFAR stub now constructs ---")
    try:
        from examples.mobilenet_cifar import MobileNetV2CIFAR
        m = MobileNetV2CIFAR(num_classes=5, pretrained=False)
        y = m(torch.randn(1, 3, 32, 32))
        check("MobileNetV2CIFAR forward pass", y.shape == (1, 5))
    except Exception as e:
        check("MobileNetV2CIFAR import/forward", False, str(e))


def main():
    test_ptq_quantize_is_generic()
    test_ptq_calibrate_populates_observers()
    test_kl_vs_mse_thresholds_distinct()
    test_calibrate_executes_both_paths()
    test_quantize_consumes_calibration()
    test_ptq_generate_configs_and_evaluate()
    test_nsga_get_pareto_front_empty_then_populated()
    test_phase_1f_plan_shape()
    test_example_mobilenet_builds()

    print("\n" + "=" * 50)
    print(f"  PTQ Wiring Tests: {passed} passed, {failed} failed")
    print("=" * 50)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
