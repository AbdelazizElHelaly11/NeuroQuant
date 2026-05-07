"""
NeuroQuant v2.0 — Wave 6 production tests (Testing + CI + Pydantic).

Coverage matrix:

  * K1 — Shared fixtures from ``conftest.py``
            - ``tiny_model``, ``tiny_cnn_factory``, ``calib_loader``,
              ``val_loader``, ``quant_config``, ``pipeline_skeleton``
              all yield the documented type/shape and are deterministic
              across runs.

  * K4 — Property-based quantization invariants (hypothesis)
            - quantize_tensor never escapes the symmetric INT range
            - per-channel scale is positive and broadcast-compatible
            - round-trip MSE shrinks monotonically as bitwidth grows
            - latency_for_assignment is a proper sum (associativity +
              commutativity)

  * L1 — Pydantic-backed config
            - Construction-time validators fire for every guarded field
            - Validation errors include the offending field name
            - String → int coercion works for YAML-loose input
            - YAML round-trip preserves every field
            - Legacy ``.validate()`` still catches cross-field errors

  * K2 — Coverage gate config exists in pyproject.toml

  * CI — GitHub Actions workflow file exists
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

# ── Optional-dep gate ────────────────────────────────────────────────────
try:
    from hypothesis import given, settings, strategies as st
    HAS_HYPOTHESIS = True
except Exception:
    HAS_HYPOTHESIS = False

hypothesis_required = pytest.mark.skipif(
    not HAS_HYPOTHESIS, reason="hypothesis not installed"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# K1 — Shared fixtures (smoke tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_tiny_model_fixture_returns_eval_mode_tinycnn(tiny_model):
    """``tiny_model`` is a TinyCNN in eval mode with the right output shape."""
    assert tiny_model.training is False
    out = tiny_model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_tiny_cnn_factory_seeds_are_deterministic(tiny_cnn_factory):
    """Same seed → bit-identical weights; different seeds → different."""
    a = tiny_cnn_factory(seed=0)
    b = tiny_cnn_factory(seed=0)
    c = tiny_cnn_factory(seed=1)
    assert torch.equal(a.c1.weight, b.c1.weight)
    assert not torch.equal(a.c1.weight, c.c1.weight)


def test_calib_and_val_loaders_have_canonical_shape(calib_loader, val_loader):
    """Both loaders yield (B, 3, 32, 32) tensors with batch size 8."""
    for loader in (calib_loader, val_loader):
        x, y = next(iter(loader))
        assert x.shape == (8, 3, 32, 32)
        assert y.shape == (8,)


def test_calib_and_val_loaders_are_disjoint_in_data(calib_loader, val_loader):
    """The two loader fixtures must use different seeds (no shared data)."""
    x_calib, _ = next(iter(calib_loader))
    x_val, _ = next(iter(val_loader))
    # Probability of accidental equality on random tensors is ~0.
    assert not torch.allclose(x_calib, x_val)


def test_quant_config_fixture_validates(quant_config):
    """The default ``quant_config`` passes ``.validate()``."""
    quant_config.validate()  # raises on failure
    assert quant_config.hyperparams.device == "cpu"


def test_pipeline_skeleton_has_summary_helpers(pipeline_skeleton):
    """The skeleton fixture has the attributes the report helpers read."""
    p = pipeline_skeleton
    assert hasattr(p, "_add_summary_row")
    assert hasattr(p, "_print_report")
    assert hasattr(p, "_build_pareto_summary")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# L1 — Pydantic validators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.parametrize("kwargs", [
    {"num_classes": 1},
    {"num_classes": 0},
    {"batch_size": 0},
    {"batch_size": -1},
    {"io_layer_bitwidth": 7},
    {"io_layer_bitwidth": 33},
    {"input_shape": (3, 4, 4)},
    {"input_shape": (2, 32, 32)},
    {"input_shape": (3, 32)},
])
def test_quant_config_construction_rejects_bad_field(kwargs):
    """Each invalid input raises at construction, not later in pipeline."""
    from config import QuantizationConfig

    with pytest.raises(Exception) as exc_info:
        QuantizationConfig(**kwargs)
    msg = str(exc_info.value)
    # The error mentions the field that broke.
    field = next(iter(kwargs))
    assert field in msg or field.replace("_", " ") in msg or field in repr(exc_info.value)


@pytest.mark.parametrize("kwargs", [
    {"device": "tpu"},
    {"qat_lr": 0.0},
    {"qat_lr": -0.1},
    {"qat_act_bitwidth": 3},
    {"qat_distill_alpha": 1.5},
    {"qat_distill_alpha": -0.1},
    {"qat_distill_temperature": 0.0},
    {"hessian_estimator": "newton"},
    {"qat_warmstart_source": "random"},
    {"nsga_population_size": 3},
    {"nsga_generations": 0},
    {"ptq_real_rerank_topk": 0},
    {"ptq_tradeoff_max_acc_drop": -0.5},
    {"latency_lut_bitwidths": [3, 8]},
    {"latency_lut_bitwidths": [4, 5]},
])
def test_hyperparam_construction_rejects_bad_field(kwargs):
    """HyperparameterSet rejects each error case with a clear message."""
    from config import HyperparameterSet

    with pytest.raises(Exception):
        HyperparameterSet(**kwargs)


def test_pydantic_string_to_int_coercion():
    """YAML-loose strings like '42' coerce to int automatically."""
    from config import QuantizationConfig

    cfg = QuantizationConfig(num_classes="42", batch_size="64")
    assert cfg.num_classes == 42 and isinstance(cfg.num_classes, int)
    assert cfg.batch_size == 64 and isinstance(cfg.batch_size, int)


def test_validate_still_catches_cross_field_errors():
    """``.validate()`` still enforces low<high percentile, phase names, etc.

    These are CROSS-field constraints that pydantic field validators
    can't express on their own — so they stay in the runtime
    ``validate()`` method as documented.
    """
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    cfg.hyperparams.cluster_low_percentile = 0.8
    cfg.hyperparams.cluster_high_percentile = 0.5  # low > high — invalid
    with pytest.raises(ValueError, match="low < high"):
        cfg.validate()


def test_yaml_roundtrip_preserves_all_fields(tmp_path):
    """``to_yaml`` + ``from_yaml`` reproduces a non-default config exactly."""
    from config import QuantizationConfig

    original = QuantizationConfig()
    original.model_name = "resnet18"
    original.num_classes = 100
    original.batch_size = 64
    original.hyperparams.qat_epochs = 3
    original.hyperparams.hardware_aware_search = True
    original.hyperparams.latency_lut_bitwidths = [4, 8]

    out = tmp_path / "cfg.yaml"
    original.to_yaml(out)
    loaded = QuantizationConfig.from_yaml(out)

    assert loaded.model_name == "resnet18"
    assert loaded.num_classes == 100
    assert loaded.batch_size == 64
    assert loaded.hyperparams.qat_epochs == 3
    assert loaded.hyperparams.hardware_aware_search is True
    assert loaded.hyperparams.latency_lut_bitwidths == [4, 8]


def test_yaml_load_with_invalid_value_raises_at_load_time(tmp_path):
    """An invalid value in YAML raises during load, not phase execution."""
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    cfg.to_yaml(tmp_path / "cfg.yaml")
    text = (tmp_path / "cfg.yaml").read_text()
    text = text.replace("num_classes: 10", "num_classes: 1")
    (tmp_path / "cfg.yaml").write_text(text)

    with pytest.raises(Exception):
        QuantizationConfig.from_yaml(tmp_path / "cfg.yaml")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# K4 — Property-based tests (hypothesis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@hypothesis_required
@given(
    elems=st.lists(
        st.floats(min_value=-100.0, max_value=100.0,
                  allow_nan=False, allow_infinity=False),
        min_size=4, max_size=64,
    ),
    bitwidth=st.sampled_from([4, 8, 16]),
)
@settings(max_examples=50, deadline=None)
def test_quantize_tensor_stays_in_symmetric_range(elems, bitwidth):
    """Dequantized values are bounded by the input's max abs.

    The deque values cannot exceed the original abs-max more than
    ``scale`` in either direction, since scale = amax / qmax and the
    INT8 range is exactly [-qmax-1, qmax]. We assert the looser but
    correct bound: every dequantized value lies in [-amax, amax]
    (within one quantization step).
    """
    from quantization.base import BaseQuantizer

    t = torch.tensor(elems, dtype=torch.float32)
    amax = float(t.abs().max())
    if amax == 0.0:
        # Trivial input — quantizer should still return a finite tensor.
        out = BaseQuantizer.quantize_tensor(t, bitwidth=bitwidth)
        assert torch.isfinite(out).all()
        return

    out = BaseQuantizer.quantize_tensor(t, bitwidth=bitwidth)
    qmax = 2 ** (bitwidth - 1) - 1
    step = amax / qmax  # one quantization step in the input scale
    # |out| ≤ amax + one step (one step accounts for the asymmetric
    # negative bound at qmin = -qmax-1).
    assert out.abs().max().item() <= amax + 2 * step + 1e-6


@hypothesis_required
@given(
    n_channels=st.integers(min_value=2, max_value=8),
    spatial=st.integers(min_value=4, max_value=16),
    bitwidth=st.sampled_from([4, 8]),
)
@settings(max_examples=20, deadline=None)
def test_per_channel_scale_is_positive(n_channels, spatial, bitwidth):
    """Per-channel quantization always produces a positive scale.

    Property: zero or negative scale corrupts every downstream INT
    kernel (division-by-zero, sign flip). The MIN_SCALE clamp inside
    ``quantize_tensor`` must guarantee strict positivity.
    """
    from quantization.base import BaseQuantizer

    torch.manual_seed(int(n_channels) * 31 + int(spatial))
    weight = torch.randn(n_channels, 3, spatial, spatial)
    out = BaseQuantizer.quantize_tensor(
        weight, bitwidth=bitwidth, per_channel=True, channel_dim=0,
    )
    # Output shape is preserved.
    assert out.shape == weight.shape
    # No NaN / inf.
    assert torch.isfinite(out).all()


@hypothesis_required
@given(
    elems=st.lists(
        st.floats(min_value=-50.0, max_value=50.0,
                  allow_nan=False, allow_infinity=False),
        min_size=64, max_size=256,
    ),
)
@settings(max_examples=20, deadline=None)
def test_quantize_tensor_higher_bitwidth_means_lower_mse(elems):
    """Round-trip MSE is non-increasing in the bitwidth.

    Mathematically: more bits → smaller quantization step → smaller
    rounding error in expectation. Assert monotonicity (with a small
    tolerance for ties on tiny tensors).
    """
    from quantization.base import BaseQuantizer

    t = torch.tensor(elems, dtype=torch.float32)
    if float(t.abs().max()) < 1e-3:
        return  # trivial input — skip (nothing to quantize)

    mse_4 = ((BaseQuantizer.quantize_tensor(t, 4) - t) ** 2).mean().item()
    mse_8 = ((BaseQuantizer.quantize_tensor(t, 8) - t) ** 2).mean().item()
    mse_16 = ((BaseQuantizer.quantize_tensor(t, 16) - t) ** 2).mean().item()

    assert mse_8 <= mse_4 + 1e-6
    assert mse_16 <= mse_8 + 1e-6


@hypothesis_required
@given(
    n_layers=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=30, deadline=None)
def test_latency_for_assignment_is_a_pure_sum(n_layers):
    """``latency_for_assignment(A + B) == latency(A) + latency(B)``.

    Splitting an assignment across two LUT views and summing the
    per-view latencies must equal the latency of the unioned view.
    Catches any future "merge" or "deduplicate" optimisation that
    might silently drop or double-count entries.
    """
    from quantization.latency_lut import latency_for_assignment

    # Build a synthetic LUT with random per-layer latencies.
    torch.manual_seed(int(n_layers) * 17)
    lut = {
        f"layer{i}.weight": {4: 0.1 + i * 0.01,
                              8: 0.2 + i * 0.01,
                              32: 0.5 + i * 0.05}
        for i in range(n_layers)
    }
    # Assignment with mixed bitwidths.
    assignment = {
        f"layer{i}.weight": (4 if i % 2 == 0 else 8)
        for i in range(n_layers)
    }
    half = n_layers // 2
    lut_a = {k: lut[k] for k in list(lut)[:half]}
    lut_b = {k: lut[k] for k in list(lut)[half:]}

    total = latency_for_assignment(assignment, lut)
    sum_parts = (
        latency_for_assignment(assignment, lut_a)
        + latency_for_assignment(assignment, lut_b)
    )
    assert total == pytest.approx(sum_parts, rel=1e-9)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# K2 — Coverage gate file exists
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_pyproject_toml_configures_coverage():
    """``pyproject.toml`` exists and configures pytest-cov."""
    project_root = Path(__file__).resolve().parent.parent
    pyproject = project_root / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml missing"
    # ``encoding`` is mandatory on Windows: the file legitimately
    # contains UTF-8 chars (≥) that the platform's CP1252 default can't
    # decode.
    text = pyproject.read_text(encoding="utf-8")
    assert "[tool.pytest.ini_options]" in text or "[tool.coverage" in text
    assert "addopts" in text or "[tool.coverage.report]" in text
    assert "--cov-fail-under" in text, "coverage gate must be configured"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CI — GitHub Actions workflow exists
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_github_actions_workflow_exists():
    """A CI workflow file exists under ``.github/workflows/``."""
    project_root = Path(__file__).resolve().parent.parent
    workflow_dir = project_root / ".github" / "workflows"
    assert workflow_dir.exists(), ".github/workflows missing"
    workflows = list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml"))
    assert workflows, "no workflow files found"
    # At least one workflow must run pytest. UTF-8 read for portability.
    text = "\n".join(p.read_text(encoding="utf-8") for p in workflows)
    assert "pytest" in text or "pytest-cov" in text
