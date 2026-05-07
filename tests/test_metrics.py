"""
NeuroQuant v2.0 - Metric Utilities Test

Fast validation (~2s) for:
    - Top-k accuracy (num_classes=3 and num_classes=10)
    - Latency benchmark outputs
    - Hardware report parser
    - MLflow metric key coverage
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from utils.metrics import compute_topk_accuracy, benchmark_latency, parse_hardware_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("test_metrics")

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        logger.info("  [PASS] %s", name)
    else:
        failed += 1
        logger.info("  [FAIL] %s -- %s", name, detail)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tiny model + fake dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TinyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


def make_loader(num_classes, n=100, shape=(3, 8, 8)):
    images = torch.randn(n, *shape)
    labels = torch.randint(0, num_classes, (n,))
    ds = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(ds, batch_size=16)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 1: Top-k accuracy with num_classes=10
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_topk_10classes():
    logger.info("--- Test: Top-k accuracy (10 classes) ---")
    model = TinyModel(10)
    loader = make_loader(10)
    device = torch.device("cpu")

    result = compute_topk_accuracy(model, loader, device, k=5)

    check("Returns dict", isinstance(result, dict))
    check("Has 'top1' key", "top1" in result)
    check("Has 'top5' key", "top5" in result)
    check("top1 in [0, 100]", 0 <= result["top1"] <= 100, f"got {result['top1']}")
    check("top5 in [0, 100]", 0 <= result["top5"] <= 100, f"got {result['top5']}")
    check("top5 >= top1", result["top5"] >= result["top1"],
          f"top1={result['top1']}, top5={result['top5']}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 2: Top-k accuracy with num_classes=3 (k clamped to 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_topk_3classes():
    logger.info("--- Test: Top-k accuracy (3 classes, k clamped) ---")
    model = TinyModel(3)
    loader = make_loader(3)
    device = torch.device("cpu")

    result = compute_topk_accuracy(model, loader, device, k=5)

    check("Returns dict", isinstance(result, dict))
    check("top1 in [0, 100]", 0 <= result["top1"] <= 100)
    check("top5 in [0, 100]", 0 <= result["top5"] <= 100)
    check("top5 >= top1 (clamped)", result["top5"] >= result["top1"])
    # With only 3 classes, k is clamped to 3, so top-k should be high
    check("top5 > 0 (sanity)", result["top5"] > 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 3: Latency benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_latency():
    logger.info("--- Test: Latency benchmark ---")
    model = TinyModel(10)
    device = torch.device("cpu")

    result = benchmark_latency(
        model, (3, 8, 8), device,
        batch_size=1, warmup_runs=3, measure_runs=10,
    )

    check("Returns dict", isinstance(result, dict))
    check("Has latency_mean_ms", "latency_mean_ms" in result)
    check("Has latency_p50_ms", "latency_p50_ms" in result)
    check("Has latency_p95_ms", "latency_p95_ms" in result)
    check("Has throughput_fps", "throughput_fps" in result)
    check("mean >= 0", result["latency_mean_ms"] >= 0)
    check("p50 >= 0", result["latency_p50_ms"] >= 0)
    check("p95 >= p50", result["latency_p95_ms"] >= result["latency_p50_ms"])
    check("throughput > 0", result["throughput_fps"] > 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 4: Hardware report parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_hw_parser():
    logger.info("--- Test: Hardware report parser ---")

    # Missing file → all nulls
    result = parse_hardware_report(None)
    check("None path → not_provided", result["source"] == "not_provided")
    check("None path → dsp is None", result["dsp"] is None)

    result2 = parse_hardware_report("/nonexistent/file.json")
    check("Missing file → not_provided", result2["source"] == "not_provided")

    # Valid JSON report
    report = {
        "resource": {"dsp": 42, "lut": 1500, "ff": 800},
        "timing": {"fmax_mhz": 200.5},
        "pipeline": {"ii": 1, "cycle_latency": 10},
    }
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(report))

    result3 = parse_hardware_report(str(tmp))
    check("JSON → dsp=42", result3["dsp"] == 42, f"got {result3['dsp']}")
    check("JSON → lut=1500", result3["lut"] == 1500)
    check("JSON → ff=800", result3["ff"] == 800)
    check("JSON → fmax=200.5", result3["fmax_mhz"] == 200.5)
    check("JSON → ii=1", result3["ii"] == 1)
    check("JSON → cycle_latency=10", result3["cycle_latency"] == 10)
    check("JSON → source set", result3["source"] != "not_provided")

    tmp.unlink()

    # Valid CSV report
    csv_content = "DSP,LUT,FF,fmax_mhz\n24,900,400,150.0\n"
    tmp2 = Path(tempfile.mktemp(suffix=".csv"))
    tmp2.write_text(csv_content)

    result4 = parse_hardware_report(str(tmp2))
    check("CSV → dsp=24", result4["dsp"] == 24, f"got {result4['dsp']}")
    check("CSV → lut=900", result4["lut"] == 900)
    check("CSV → fmax=150.0", result4["fmax_mhz"] == 150.0)

    tmp2.unlink()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 5: MLflow metric key coverage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_mlflow_keys():
    logger.info("--- Test: QuantizationResult has new metric fields ---")
    from config import QuantizationResult, LatencyResult, HardwareMetrics

    # Verify QuantizationResult has the new keys
    annotations = QuantizationResult.__annotations__
    check("QR has top5_accuracy", "top5_accuracy" in annotations)
    check("QR has latency field", "latency" in annotations)
    check("QR has hardware field", "hardware" in annotations)

    # Verify LatencyResult keys
    lat_keys = LatencyResult.__annotations__
    check("LatencyResult has latency_mean_ms", "latency_mean_ms" in lat_keys)
    check("LatencyResult has throughput_fps", "throughput_fps" in lat_keys)

    # Verify HardwareMetrics keys
    hw_keys = HardwareMetrics.__annotations__
    for field in ("dsp", "lut", "ff", "fmax_mhz", "ii", "cycle_latency", "source"):
        check(f"HardwareMetrics has {field}", field in hw_keys)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    test_topk_10classes()
    test_topk_3classes()
    test_latency()
    test_hw_parser()
    test_mlflow_keys()

    print("\n" + "=" * 50)
    print(f"  Metric Tests: {passed} passed, {failed} failed")
    print("=" * 50)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
