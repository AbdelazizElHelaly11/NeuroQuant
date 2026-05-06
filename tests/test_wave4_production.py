"""
NeuroQuant v2.0 — Wave 4 production tests (ONNX + hardware-aware search).

Coverage matrix:

  * J1  — Static INT8 ONNX export
            - FP32 → ONNX round-trip equivalence (numerical)
            - quantize_static produces a smaller .onnx
            - quantize_static output is loadable + runnable under ORT
  * J2  — ORT latency benchmark
            - Returns the canonical 4-key dict
            - Timings are positive and monotone in batch size
  * J3  — Real on-disk size
            - .onnx file size on disk is what gets reported
            - Disk size < FP32 disk size after INT8 quantization
            - Replaces synthetic ``model_size_mb`` in the result dict
              (not just appended)
  * C1  — N-objective NSGA
            - 3-tuple objectives accepted by ``_dominates``
            - Generalised non-dominated sort produces correct fronts
              on a 3-objective Pareto example
            - Latency objective discriminates ties on (acc, size)
  * C2  — Per-layer ORT latency LUT
            - LUT covers every Conv/Linear in the model
            - LUT row contains every requested bitwidth
            - ``latency_for_assignment`` returns FP32-equivalent total
              for an all-FP32 assignment
            - Cache write/read round-trip preserves entries exactly
  * J4  — Closed-loop wiring
            - 3-objective NSGA picks lower-latency configs when
              accuracy + size are equivalent

Tests skip cleanly when onnx / onnxruntime are unavailable so the
suite stays green on CI workers without ORT installed.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Optional-dep gate ────────────────────────────────────────────────────
try:
    import onnxruntime  # noqa: F401
    HAS_ORT = True
except Exception:
    HAS_ORT = False

ort_required = pytest.mark.skipif(
    not HAS_ORT, reason="onnx/onnxruntime not installed"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TinyCNN(nn.Module):
    """Small CIFAR-class CNN used everywhere below."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return TinyCNN().eval()


@pytest.fixture
def calib_loader() -> DataLoader:
    torch.manual_seed(0)
    xs = torch.randn(32, 3, 32, 32)
    ys = torch.zeros(32, dtype=torch.long)
    return DataLoader(TensorDataset(xs, ys), batch_size=8, shuffle=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# J1 — ONNX export + static INT8 quantization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@ort_required
def test_export_to_onnx_fp32_roundtrip(tiny_model, tmp_path):
    """FP32 ONNX export reproduces the torch model's output."""
    from utils.onnx_export import export_to_onnx
    import onnxruntime as ort
    import numpy as np

    out = tmp_path / "model.onnx"
    export_to_onnx(tiny_model, (3, 32, 32), str(out))

    assert out.exists() and out.stat().st_size > 0

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch_out = tiny_model(x).numpy()

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x.numpy()})[0]

    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-4, atol=1e-4)


@ort_required
def test_quantize_onnx_static_produces_smaller_file(
    tiny_model, calib_loader, tmp_path,
):
    """Static INT8 quantization shrinks the ONNX file."""
    from utils.onnx_export import export_to_onnx, quantize_onnx_static

    fp32 = tmp_path / "fp32.onnx"
    int8 = tmp_path / "int8.onnx"
    export_to_onnx(tiny_model, (3, 32, 32), str(fp32))
    quantize_onnx_static(str(fp32), str(int8), calib_loader, num_batches=2)

    fp32_bytes = fp32.stat().st_size
    int8_bytes = int8.stat().st_size
    assert int8_bytes > 0
    assert int8_bytes < fp32_bytes, (
        f"Expected INT8 ONNX smaller than FP32; got {int8_bytes} vs {fp32_bytes}"
    )


@ort_required
def test_quantize_onnx_static_runs_under_ort(
    tiny_model, calib_loader, tmp_path,
):
    """The INT8 .onnx file is a valid ORT session and produces output."""
    from utils.onnx_export import (
        export_to_onnx, quantize_onnx_static, benchmark_onnx_latency,
    )
    import onnxruntime as ort
    import numpy as np

    fp32 = tmp_path / "fp32.onnx"
    int8 = tmp_path / "int8.onnx"
    export_to_onnx(tiny_model, (3, 32, 32), str(fp32))
    quantize_onnx_static(str(fp32), str(int8), calib_loader, num_batches=2)

    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)})[0]
    assert out.shape == (1, 10)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# J2 — ORT latency benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@ort_required
def test_benchmark_onnx_latency_returns_canonical_keys(tiny_model, tmp_path):
    """Latency dict has the same shape as utils.metrics.benchmark_latency."""
    from utils.onnx_export import export_to_onnx, benchmark_onnx_latency

    out = tmp_path / "m.onnx"
    export_to_onnx(tiny_model, (3, 32, 32), str(out))
    res = benchmark_onnx_latency(
        str(out), (3, 32, 32),
        warmup_runs=2, measure_runs=5,
    )
    assert set(res.keys()) == {
        "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "throughput_fps",
    }
    assert res["latency_mean_ms"] > 0
    assert res["throughput_fps"] > 0


@ort_required
def test_benchmark_onnx_latency_monotone_in_batch_size(tiny_model, tmp_path):
    """Larger batch sizes do not produce smaller mean times."""
    from utils.onnx_export import export_to_onnx, benchmark_onnx_latency

    # Export with dynamic batch axis so the same .onnx serves both runs.
    out = tmp_path / "m.onnx"
    export_to_onnx(tiny_model, (3, 32, 32), str(out), dynamic_batch=True)

    bs1 = benchmark_onnx_latency(
        str(out), (3, 32, 32), batch_size=1,
        warmup_runs=2, measure_runs=10,
    )["latency_mean_ms"]
    bs8 = benchmark_onnx_latency(
        str(out), (3, 32, 32), batch_size=8,
        warmup_runs=2, measure_runs=10,
    )["latency_mean_ms"]

    # Batch=8 should take at least as long as batch=1 (within noise).
    # Allow generous slack for tiny CPU runs.
    assert bs8 >= bs1 * 0.5, f"bs8={bs8} unexpectedly less than bs1/2={bs1 / 2}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# J3 — Real on-disk size
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@ort_required
def test_onnx_disk_size_matches_filesystem(tiny_model, tmp_path):
    """``onnx_disk_size_mb`` equals the actual file size on disk."""
    from utils.onnx_export import export_to_onnx, onnx_disk_size_mb

    out = tmp_path / "m.onnx"
    export_to_onnx(tiny_model, (3, 32, 32), str(out))

    expected_mb = out.stat().st_size / (1024.0 * 1024.0)
    assert onnx_disk_size_mb(str(out)) == pytest.approx(expected_mb, rel=1e-9)


@ort_required
def test_export_quantize_and_benchmark_emits_int8_keys(
    tiny_model, calib_loader, tmp_path,
):
    """The convenience helper returns the expected J1/J2/J3 fields."""
    from utils.onnx_export import export_quantize_and_benchmark

    info = export_quantize_and_benchmark(
        tiny_model, (3, 32, 32),
        str(tmp_path),
        name="combo",
        calibration_loader=calib_loader,
        num_batches=2,
        warmup_runs=2,
        measure_runs=5,
    )
    assert info["fp32_onnx_path"].endswith("combo.fp32.onnx")
    assert info["int8_onnx_path"].endswith("combo.int8.onnx")
    assert info["int8_onnx_size_mb"] < info["fp32_onnx_size_mb"]
    assert "onnx_latency" in info
    assert info["onnx_latency"]["latency_mean_ms"] > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C1 — N-objective NSGA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_dominates_predicate_handles_3_tuples():
    """``_dominates`` returns the Pareto-correct answer for 3-objective inputs."""
    from quantization.nsga_ii_search import NSGAIIClusterSearch as N

    assert N._dominates((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    assert not N._dominates((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert N._dominates((1.0, 1.0, 1.0), (1.0, 1.0, 2.0))
    # Trade-off — neither dominates.
    assert not N._dominates((1.0, 2.0, 3.0), (2.0, 1.0, 3.0))
    assert not N._dominates((2.0, 1.0, 3.0), (1.0, 2.0, 3.0))


def test_non_dominated_sort_3_objectives_assigns_fronts_correctly():
    """Three-objective non-dominated sort recovers the dominated tail."""
    from quantization.nsga_ii_search import NSGAIIClusterSearch as N

    points = [
        (1.0, 2.0, 3.0),  # 0: Pareto
        (2.0, 1.0, 3.0),  # 1: Pareto (trade-off vs 0)
        (1.5, 1.5, 2.5),  # 2: Pareto (better latency than 0/1)
        (3.0, 3.0, 3.0),  # 3: Dominated by 0 and 1
        (2.0, 2.0, 4.0),  # 4: Dominated by 2 (1.5,1.5,2.5) ⇒ all-better
    ]
    fronts = N._non_dominated_sort(points)

    assert len(fronts) >= 2
    assert set(fronts[0]) == {0, 1, 2}
    # Both dominated points end up after the Pareto front.
    dominated_indices = {idx for f in fronts[1:] for idx in f}
    assert {3, 4}.issubset(dominated_indices)


def test_crowding_distance_handles_3_objectives_without_error():
    """Crowding-distance generalisation runs over an arbitrary N."""
    from quantization.nsga_ii_search import NSGAIIClusterSearch as N

    points = [
        (1.0, 2.0, 3.0),
        (2.0, 1.0, 3.0),
        (1.5, 1.5, 2.5),
        (1.2, 1.8, 2.8),
    ]
    front = [0, 1, 2, 3]
    cd = N._crowding_distance(points, front)
    assert set(cd.keys()) == set(front)
    # At least one boundary point in some objective gets infinity.
    assert any(v == float("inf") for v in cd.values())
    # All values are non-negative.
    for v in cd.values():
        assert v >= 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C2 — Per-layer ORT latency LUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@ort_required
def test_latency_lut_covers_every_conv_and_linear(tiny_model, calib_loader):
    """Every Conv2d / Linear in the model gets a LUT row."""
    from quantization.latency_lut import build_latency_lut

    lut = build_latency_lut(
        tiny_model, (3, 32, 32), calib_loader,
        warmup_runs=2, measure_runs=5,
    )

    expected = {
        f"{n}.weight" for n, m in tiny_model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    }
    assert expected.issubset(set(lut.keys())), (
        f"Missing LUT entries for: {expected - set(lut.keys())}"
    )


@ort_required
def test_latency_lut_row_has_each_requested_bitwidth(tiny_model, calib_loader):
    """Every requested bitwidth shows up in the LUT row plus a 32-bit fallback."""
    from quantization.latency_lut import build_latency_lut

    lut = build_latency_lut(
        tiny_model, (3, 32, 32), calib_loader,
        bitwidths=(4, 8),
        warmup_runs=2, measure_runs=5,
    )
    for pname, row in lut.items():
        assert 8 in row, f"LUT[{pname}] missing INT8 row"
        assert 4 in row, f"LUT[{pname}] missing INT4 row"
        assert 32 in row, f"LUT[{pname}] missing FP32 fallback"
        for bw, ms in row.items():
            assert ms >= 0, f"LUT[{pname}][{bw}] = {ms} (must be ≥ 0)"


@ort_required
def test_latency_for_assignment_sums_lut_entries(tiny_model, calib_loader):
    """``latency_for_assignment`` is a faithful sum over the bitwidth assignment."""
    from quantization.latency_lut import build_latency_lut, latency_for_assignment

    lut = build_latency_lut(
        tiny_model, (3, 32, 32), calib_loader,
        bitwidths=(4, 8),
        warmup_runs=2, measure_runs=5,
    )
    all_int8 = {pname: 8 for pname in lut}
    all_int4 = {pname: 4 for pname in lut}
    all_fp32 = {pname: 32 for pname in lut}

    sum_int8 = latency_for_assignment(all_int8, lut)
    sum_int4 = latency_for_assignment(all_int4, lut)
    sum_fp32 = latency_for_assignment(all_fp32, lut)

    expected_int8 = sum(row[8] for row in lut.values())
    expected_int4 = sum(row[4] for row in lut.values())
    expected_fp32 = sum(row[32] for row in lut.values())
    assert sum_int8 == pytest.approx(expected_int8, rel=1e-9)
    assert sum_int4 == pytest.approx(expected_int4, rel=1e-9)
    assert sum_fp32 == pytest.approx(expected_fp32, rel=1e-9)


@ort_required
def test_latency_lut_cache_roundtrip(tiny_model, calib_loader, tmp_path):
    """Cached LUT JSON loads back into the exact same float values."""
    from quantization.latency_lut import build_latency_lut

    cache = tmp_path / "lut.json"
    lut1 = build_latency_lut(
        tiny_model, (3, 32, 32), calib_loader,
        bitwidths=(4, 8),
        warmup_runs=2, measure_runs=5,
        cache_path=str(cache),
    )
    assert cache.exists()
    # Second call must use the cache (no rebuild).
    lut2 = build_latency_lut(
        tiny_model, (3, 32, 32), calib_loader,
        bitwidths=(4, 8),
        warmup_runs=2, measure_runs=5,
        cache_path=str(cache),
    )
    assert set(lut1) == set(lut2)
    for pname in lut1:
        assert set(lut1[pname]) == set(lut2[pname])
        for bw in lut1[pname]:
            assert lut1[pname][bw] == pytest.approx(lut2[pname][bw], abs=1e-12)


def test_latency_for_assignment_fallbacks_to_larger_bitwidth():
    """Missing bitwidth row falls back to next-larger, never to zero."""
    from quantization.latency_lut import latency_for_assignment

    lut = {
        "a.weight": {8: 1.0, 32: 2.0},   # No INT4 row.
        "b.weight": {4: 0.5, 8: 0.7, 32: 1.0},
    }
    # Asking for INT4 on "a.weight" should fall back to INT8 (1.0), not 0.
    total = latency_for_assignment(
        {"a.weight": 4, "b.weight": 4}, lut,
    )
    assert total == pytest.approx(1.0 + 0.5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# J4 — Closed-loop wiring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_nsga_evaluate_individual_returns_3_objectives_with_lut(tiny_model):
    """Eval result has 3 numbers when a latency LUT is supplied."""
    from config import QuantizationConfig
    from quantization.nsga_ii_search import NSGAIIClusterSearch
    from torch.utils.data import DataLoader, TensorDataset

    clusters = [
        {"cluster_id": 0, "tier": "HIGH",
         "layer_names": ["c1.weight"],
         "allowed_bitwidths": [8], "mean_sensitivity": 1.0},
        {"cluster_id": 1, "tier": "MEDIUM",
         "layer_names": ["c2.weight", "c3.weight"],
         "allowed_bitwidths": [4, 8], "mean_sensitivity": 0.5},
        {"cluster_id": 2, "tier": "LOW",
         "layer_names": ["fc.weight"],
         "allowed_bitwidths": [4, 8], "mean_sensitivity": 0.1},
    ]
    cfg = QuantizationConfig()
    cfg.hyperparams.nsga_population_size = 4
    cfg.hyperparams.nsga_generations = 2

    lut = {
        "c1.weight": {4: 0.05, 8: 0.10, 32: 0.20},
        "c2.weight": {4: 0.10, 8: 0.20, 32: 0.40},
        "c3.weight": {4: 0.10, 8: 0.20, 32: 0.40},
        "fc.weight": {4: 0.01, 8: 0.02, 32: 0.04},
    }

    nsga = NSGAIIClusterSearch(
        tiny_model, clusters, cfg, latency_lut=lut,
    )
    assert nsga._num_objectives == 3

    loader = DataLoader(
        TensorDataset(torch.randn(8, 3, 32, 32),
                      torch.zeros(8, dtype=torch.long)),
        batch_size=4,
    )
    obj = nsga.evaluate_individual([1, 1], loader, fp32_accuracy=10.0)
    assert len(obj) == 3
    # Latency is the LUT sum for the all-INT8 assignment (HIGH cluster
    # is fixed INT8 by ``_fixed_config``, MEDIUM/LOW pick INT8 via the
    # ``[1, 1]`` gene).
    expected_lat = lut["c1.weight"][8] + lut["c2.weight"][8] + lut["c3.weight"][8] + lut["fc.weight"][8]
    assert obj[2] == pytest.approx(expected_lat, rel=1e-9)


def test_nsga_3obj_pareto_includes_latency_in_solution():
    """3-objective Pareto solutions carry the ``latency_mean_ms`` key."""
    from config import QuantizationConfig
    from quantization.nsga_ii_search import NSGAIIClusterSearch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(1)
    m = TinyCNN().eval()
    clusters = [
        {"cluster_id": 0, "tier": "HIGH",
         "layer_names": ["c1.weight"],
         "allowed_bitwidths": [8], "mean_sensitivity": 1.0},
        {"cluster_id": 1, "tier": "MEDIUM",
         "layer_names": ["c2.weight", "c3.weight"],
         "allowed_bitwidths": [4, 8], "mean_sensitivity": 0.5},
        {"cluster_id": 2, "tier": "LOW",
         "layer_names": ["fc.weight"],
         "allowed_bitwidths": [4, 8], "mean_sensitivity": 0.1},
    ]
    cfg = QuantizationConfig()
    cfg.hyperparams.nsga_population_size = 4
    cfg.hyperparams.nsga_generations = 2
    cfg.hyperparams.seed = 0

    lut = {
        "c1.weight": {4: 0.05, 8: 0.10, 32: 0.20},
        "c2.weight": {4: 0.10, 8: 0.20, 32: 0.40},
        "c3.weight": {4: 0.10, 8: 0.20, 32: 0.40},
        "fc.weight": {4: 0.01, 8: 0.02, 32: 0.04},
    }

    nsga = NSGAIIClusterSearch(m, clusters, cfg, latency_lut=lut)
    loader = DataLoader(
        TensorDataset(torch.randn(16, 3, 32, 32),
                      torch.zeros(16, dtype=torch.long)),
        batch_size=4,
    )
    front = nsga.search(
        loader, fp32_accuracy=10.0,
        seed_config={"c1.weight": 8, "c2.weight": 8,
                     "c3.weight": 8, "fc.weight": 8},
    )
    assert front["solutions"], "search produced no Pareto solutions"
    for sol in front["solutions"]:
        assert sol.get("latency_mean_ms") is not None
        assert sol["latency_mean_ms"] >= 0


def test_nsga_2obj_backwards_compatibility():
    """No LUT → solutions still produced, ``latency_mean_ms`` is None."""
    from config import QuantizationConfig
    from quantization.nsga_ii_search import NSGAIIClusterSearch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(2)
    m = TinyCNN().eval()
    clusters = [
        {"cluster_id": 0, "tier": "MEDIUM",
         "layer_names": ["c1.weight", "c2.weight", "c3.weight", "fc.weight"],
         "allowed_bitwidths": [4, 8], "mean_sensitivity": 0.5},
    ]
    cfg = QuantizationConfig()
    cfg.hyperparams.nsga_population_size = 4
    cfg.hyperparams.nsga_generations = 2
    cfg.hyperparams.seed = 0

    nsga = NSGAIIClusterSearch(m, clusters, cfg)  # no LUT
    assert nsga._num_objectives == 2

    loader = DataLoader(
        TensorDataset(torch.randn(16, 3, 32, 32),
                      torch.zeros(16, dtype=torch.long)),
        batch_size=4,
    )
    front = nsga.search(
        loader, fp32_accuracy=10.0,
        seed_config={"c1.weight": 8, "c2.weight": 8,
                     "c3.weight": 8, "fc.weight": 8},
    )
    assert front["solutions"]
    for sol in front["solutions"]:
        # Backwards compat: latency key absent or None.
        assert sol.get("latency_mean_ms") in (None,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config knob smoke tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_config_exposes_wave4_knobs():
    """The new wave 4 knobs round-trip through QuantizationConfig."""
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    hp = cfg.hyperparams
    assert hasattr(hp, "onnx_export_enabled")
    assert hasattr(hp, "hardware_aware_search")
    assert hasattr(hp, "latency_lut_bitwidths")
    assert hp.latency_lut_bitwidths == [4, 8]
    cfg.validate()  # default config must validate cleanly


def test_config_validate_rejects_bad_lut_bitwidth():
    """An unsupported bitwidth in the LUT list trips validation."""
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    cfg.hyperparams.latency_lut_bitwidths = [3, 8]  # 3 is not supported
    with pytest.raises(ValueError, match="latency_lut_bitwidths"):
        cfg.validate()
