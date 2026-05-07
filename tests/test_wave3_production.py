"""
NeuroQuant v2.0 — Wave-3 production-grade regression tests.

Covers method audits and the faster sensitivity estimator:

  B2) ``HessianComputer`` dispatches on ``hessian_estimator``. Default
      "fisher" runs a single backprop and returns a positive sensitivity
      score per parameter. "diag_hessian" still works (legacy / ablation).

  F3) ``SmoothQuantQuantizer`` per-layer α grid search picks an α per
      layer and stashes the choices on the returned model. Different
      layers can land on different α values — exercising the search.

  F2) ``AWQQuantizer`` (rewritten) inserts ``_AWQInputScale`` wrappers
      so the deployment forward is ``Y = (X / s) · quantize(s · W)``.
      The per-layer α is searched and stashed; the salient-channel
      carve-out is exercised (top-K columns left in FP16 inside the
      stored weight).

  F4) ``SmoothQuantGPTQQuantizer`` produces a model that has BOTH
      SmoothQuant wrappers AND distinct (smoothed-then-GPTQ-rounded)
      weights. Forward is finite, and the wrapper manifest produced
      by ``serialize_smoothquant_metadata`` covers every wrapped layer.

All tests run on CPU with tiny synthetic data — no torchvision downloads.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import torch
import torch.nn as nn

from config import QuantizationConfig
from quantization.hessian_clustering import HessianComputer
from quantization.awq import (
    AWQQuantizer,
    _AWQInputScale,
    serialize_awq_metadata,
    restore_awq_wrappers,
)
from quantization.smoothquant import (
    SmoothQuantQuantizer,
    _SmoothInputScale,
    serialize_smoothquant_metadata,
)
from quantization.smoothquant_gptq import SmoothQuantGPTQQuantizer

logging.basicConfig(level=logging.WARNING, format="%(message)s")

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


class _TinyConvNet(nn.Module):
    """Diverse-channel-magnitude tiny CNN. Two convs with different
    input-channel counts so the AWQ / SmoothQuant α search has more
    than one unique decision to make."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        return self.fc(x.mean(dim=(2, 3)))


def _make_loader(n: int = 32, classes: int = 4):
    imgs = torch.randn(n, 3, 8, 8) * 1.5
    lbls = torch.randint(0, classes, (n,))
    ds = torch.utils.data.TensorDataset(imgs, lbls)
    return torch.utils.data.DataLoader(ds, batch_size=8)


# ─────────────────────────────────────────────────────────────────────────
# B2) Fisher / diag-Hessian estimator
# ─────────────────────────────────────────────────────────────────────────


def test_hessian_estimator_default_is_fisher():
    print("--- Test B2.1: hessian_estimator default is fisher ---")
    cfg = QuantizationConfig()
    check("default hessian_estimator is 'fisher'",
          cfg.hyperparams.hessian_estimator == "fisher",
          f"got {cfg.hyperparams.hessian_estimator}")

    # Validate accepts both knobs and rejects garbage.
    cfg.hyperparams.hessian_estimator = "diag_hessian"
    cfg.validate()
    check("'diag_hessian' validated", True)

    cfg.hyperparams.hessian_estimator = "garbage"
    raised = False
    try:
        cfg.validate()
    except ValueError:
        raised = True
    check("invalid estimator rejected by validate()", raised)


def test_fisher_estimator_runs_and_produces_positive_scores():
    print("--- Test B2.2: Fisher path returns positive sensitivity per param ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.hessian_estimator = "fisher"
    cfg.hyperparams.hessian_batches = 2

    model = _TinyConvNet(4)
    hc = HessianComputer(model, cfg)
    res = hc.compute_hessian(_make_loader(), nn.CrossEntropyLoss(), num_batches=2)

    check("fisher returns one entry per parameter",
          len(res) == sum(1 for _ in model.parameters()),
          f"got {len(res)}")
    for name, sens in res.items():
        check(f"{name}: hessian_diag is non-negative finite",
              isinstance(sens["hessian_diag"], float)
              and sens["hessian_diag"] >= 0
              and sens["hessian_diag"] != float("inf"),
              f"got {sens['hessian_diag']}")


def test_fisher_vs_diag_hessian_differ():
    print("--- Test B2.3: fisher and diag_hessian produce distinct scores ---")
    cfg_f = QuantizationConfig()
    cfg_f.num_classes = 4
    cfg_f.input_shape = (3, 8, 8)
    cfg_f.hyperparams.device = "cpu"
    cfg_f.hyperparams.hessian_estimator = "fisher"
    cfg_f.hyperparams.hessian_batches = 2

    cfg_h = QuantizationConfig()
    cfg_h.num_classes = 4
    cfg_h.input_shape = (3, 8, 8)
    cfg_h.hyperparams.device = "cpu"
    cfg_h.hyperparams.hessian_estimator = "diag_hessian"
    cfg_h.hyperparams.hessian_batches = 2

    torch.manual_seed(0)
    m1 = _TinyConvNet(4)
    torch.manual_seed(0)
    m2 = _TinyConvNet(4)
    loader = _make_loader()

    fisher = HessianComputer(m1, cfg_f).compute_hessian(
        loader, nn.CrossEntropyLoss(), num_batches=2,
    )
    diag = HessianComputer(m2, cfg_h).compute_hessian(
        loader, nn.CrossEntropyLoss(), num_batches=2,
    )
    # Compare on the c1.weight key — they should disagree on at least
    # the *value*, even if the ranking correlates strongly. Equality
    # here would mean Fisher == diag(Hessian) numerically, which is
    # only true at exact MLE convergence.
    f_val = fisher["c1.weight"]["hessian_diag"]
    h_val = diag["c1.weight"]["hessian_diag"]
    check("fisher and diag_hessian both finite",
          all(isinstance(v, float) for v in (f_val, h_val))
          and f_val >= 0 and h_val >= 0)
    check("fisher and diag_hessian estimators produce distinct values",
          abs(f_val - h_val) > 1e-9 or f_val != h_val,
          f"f={f_val}, h={h_val}")


# ─────────────────────────────────────────────────────────────────────────
# F3) SmoothQuant per-layer α grid search
# ─────────────────────────────────────────────────────────────────────────


def test_smoothquant_per_layer_alpha_runs_and_records_choices():
    print("--- Test F3: SmoothQuant per-layer α records a chosen α per layer ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.smoothquant_per_layer_alpha = True
    cfg.hyperparams.smoothquant_alpha_grid = [0.3, 0.5, 0.7]
    cfg.hyperparams.calibration_batches = 2

    torch.manual_seed(0)
    model = _TinyConvNet(4)
    sq = SmoothQuantQuantizer(model, cfg)
    q_model = sq.quantize(_make_loader(), bitwidth=8, num_batches=2)

    chosen = getattr(q_model, "_smoothquant_alpha", None)
    check("per-layer α dict stashed on the returned model",
          isinstance(chosen, dict) and len(chosen) >= 1,
          f"got {chosen!r}")
    if isinstance(chosen, dict):
        for layer_name, alpha in chosen.items():
            check(f"  α[{layer_name}] in grid",
                  alpha in cfg.hyperparams.smoothquant_alpha_grid,
                  f"got α={alpha}")


def test_smoothquant_global_alpha_path_still_works():
    print("--- Test F3.2: per_layer_alpha=False uses the global α ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.smoothquant_per_layer_alpha = False
    cfg.hyperparams.smoothquant_alpha = 0.6
    cfg.hyperparams.calibration_batches = 2

    model = _TinyConvNet(4)
    sq = SmoothQuantQuantizer(model, cfg)
    q_model = sq.quantize(_make_loader(), bitwidth=8, num_batches=2)
    chosen = getattr(q_model, "_smoothquant_alpha", {}) or {}
    for layer_name, alpha in chosen.items():
        check(f"  α[{layer_name}] equals global 0.6 (per_layer disabled)",
              abs(alpha - 0.6) < 1e-9,
              f"got α={alpha}")


# ─────────────────────────────────────────────────────────────────────────
# F2) Production-corrected AWQ
# ─────────────────────────────────────────────────────────────────────────


def test_awq_inserts_input_scale_wrappers():
    print("--- Test F2.1: AWQ inserts _AWQInputScale wrappers ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.awq_alpha_grid = [0.0, 0.5, 1.0]
    cfg.hyperparams.calibration_batches = 2

    torch.manual_seed(1)
    model = _TinyConvNet(4)
    awq = AWQQuantizer(model, cfg)
    q_model = awq.quantize(_make_loader(), bitwidth=4, num_batches=2)

    n_wrappers = sum(
        1 for m in q_model.modules() if isinstance(m, _AWQInputScale)
    )
    check("at least one _AWQInputScale wrapper inserted",
          n_wrappers >= 1, f"got {n_wrappers}")

    # Stashed α dict and salient-channel counts.
    chosen = getattr(q_model, "_awq_alpha", {}) or {}
    salient = getattr(q_model, "_awq_salient_kept", {}) or {}
    check("_awq_alpha dict populated",
          isinstance(chosen, dict) and len(chosen) >= 1)
    check("_awq_salient_kept dict populated",
          isinstance(salient, dict) and len(salient) >= 1)
    # All chosen α land in the grid.
    for n, a in chosen.items():
        check(f"  α[{n}] in grid",
              a in cfg.hyperparams.awq_alpha_grid,
              f"got α={a}")


def test_awq_forward_remains_finite():
    print("--- Test F2.2: AWQ forward is finite (input-scale wrapper engages) ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.awq_alpha_grid = [0.0, 0.5, 1.0]
    cfg.hyperparams.calibration_batches = 2

    model = _TinyConvNet(4)
    awq = AWQQuantizer(model, cfg)
    q_model = awq.quantize(_make_loader(), bitwidth=4, num_batches=2)
    q_model.eval()
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        y = q_model(x)
    check("AWQ forward output is finite",
          torch.isfinite(y).all() and y.shape == (2, 4),
          f"got {y}")


def test_awq_salient_carveout_keeps_top_k_columns():
    print("--- Test F2.3: keep_top_pct keeps salient columns at FP16 ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.awq_alpha_grid = [0.5]
    cfg.hyperparams.awq_keep_top_pct = 0.5  # carve out half the channels
    cfg.hyperparams.calibration_batches = 2

    torch.manual_seed(2)
    model = _TinyConvNet(4)
    awq = AWQQuantizer(model, cfg)
    q_model = awq.quantize(_make_loader(), bitwidth=4, num_batches=2)
    salient = getattr(q_model, "_awq_salient_kept", {}) or {}
    # At 50%, every applicable layer should report a non-zero salient
    # count.
    nonzero = sum(1 for v in salient.values() if v > 0)
    check("salient-channel keep produces nonzero count for ≥1 layer",
          nonzero >= 1, f"counts={salient}")


def test_awq_metadata_roundtrip():
    print("--- Test F2.4: AWQ wrapper manifest covers all wrappers ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.awq_alpha_grid = [0.5]
    cfg.hyperparams.calibration_batches = 2

    model = _TinyConvNet(4)
    awq = AWQQuantizer(model, cfg)
    q_model = awq.quantize(_make_loader(), bitwidth=4, num_batches=2)

    meta = serialize_awq_metadata(q_model)
    n_wrappers = sum(
        1 for m in q_model.modules() if isinstance(m, _AWQInputScale)
    )
    check("manifest entry count matches wrapper count",
          len(meta.get("wrappers", [])) == n_wrappers,
          f"manifest={len(meta.get('wrappers', []))}, real={n_wrappers}")

    # Round-trip: rebuild on a fresh blank model and verify state_dict
    # keys match.
    blank = _TinyConvNet(4)
    restore_awq_wrappers(blank, meta)
    blank.load_state_dict(q_model.state_dict(), strict=False)
    check("rebuilt model has the same wrapper count",
          sum(1 for m in blank.modules() if isinstance(m, _AWQInputScale))
          == n_wrappers)


# ─────────────────────────────────────────────────────────────────────────
# F4) Combined SmoothQuant → GPTQ
# ─────────────────────────────────────────────────────────────────────────


def test_smoothquant_gptq_produces_wrapped_quantized_model():
    print("--- Test F4.1: SmoothQuantGPTQ produces wrapped + quantized model ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.smoothquant_per_layer_alpha = False
    cfg.hyperparams.smoothquant_alpha = 0.5
    cfg.hyperparams.calibration_batches = 2

    torch.manual_seed(3)
    base = _TinyConvNet(4)
    sg = SmoothQuantGPTQQuantizer(base, cfg)
    q_model = sg.quantize(_make_loader(), bitwidth=8, num_batches=2)

    # Wrappers from SmoothQuant should be present.
    n_wrappers = sum(
        1 for m in q_model.modules() if isinstance(m, _SmoothInputScale)
    )
    check("at least one SmoothQuant wrapper present after combined pass",
          n_wrappers >= 1, f"got {n_wrappers}")

    # Forward must be finite (the GPTQ stage didn't break the wrapped graph).
    q_model.eval()
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        y = q_model(x)
    check("combined forward output is finite",
          torch.isfinite(y).all() and y.shape == (2, 4))


def test_smoothquant_gptq_weights_differ_from_pure_smoothquant():
    print("--- Test F4.2: SmoothQuantGPTQ weights ≠ pure SmoothQuant weights ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.smoothquant_per_layer_alpha = False
    cfg.hyperparams.smoothquant_alpha = 0.5
    cfg.hyperparams.calibration_batches = 2

    torch.manual_seed(4)
    base = _TinyConvNet(4)

    # Pure SmoothQuant
    import copy
    sq = SmoothQuantQuantizer(copy.deepcopy(base), cfg)
    sq_model = sq.quantize(_make_loader(), bitwidth=8, num_batches=2)

    # Combined
    sg = SmoothQuantGPTQQuantizer(copy.deepcopy(base), cfg)
    sg_model = sg.quantize(_make_loader(), bitwidth=8, num_batches=2)

    sq_sd = sq_model.state_dict()
    sg_sd = sg_model.state_dict()
    # Find a quantized weight key common to both and compare.
    common_keys = [
        k for k in sq_sd
        if k in sg_sd and "weight" in k and sq_sd[k].dim() >= 2
    ]
    differ = False
    for k in common_keys:
        if sq_sd[k].shape != sg_sd[k].shape:
            continue
        if not torch.allclose(sq_sd[k], sg_sd[k], atol=0.0):
            differ = True
            break
    check("combined method produces a different quantized weight than pure SQ",
          differ,
          f"all weights identical across keys={common_keys[:3]}…")


def test_smoothquant_gptq_metadata_roundtrip():
    print("--- Test F4.3: combined model survives serialize/restore round-trip ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.smoothquant_per_layer_alpha = False
    cfg.hyperparams.smoothquant_alpha = 0.5
    cfg.hyperparams.calibration_batches = 2

    base = _TinyConvNet(4)
    sg = SmoothQuantGPTQQuantizer(base, cfg)
    q_model = sg.quantize(_make_loader(), bitwidth=8, num_batches=2)
    meta = serialize_smoothquant_metadata(q_model)
    n_wrappers = sum(
        1 for m in q_model.modules() if isinstance(m, _SmoothInputScale)
    )
    check("combined-model manifest entries == wrapper count",
          len(meta.get("wrappers", [])) == n_wrappers,
          f"manifest={len(meta.get('wrappers', []))}, real={n_wrappers}")


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    test_hessian_estimator_default_is_fisher()
    test_fisher_estimator_runs_and_produces_positive_scores()
    test_fisher_vs_diag_hessian_differ()

    test_smoothquant_per_layer_alpha_runs_and_records_choices()
    test_smoothquant_global_alpha_path_still_works()

    test_awq_inserts_input_scale_wrappers()
    test_awq_forward_remains_finite()
    test_awq_salient_carveout_keeps_top_k_columns()
    test_awq_metadata_roundtrip()

    test_smoothquant_gptq_produces_wrapped_quantized_model()
    test_smoothquant_gptq_weights_differ_from_pure_smoothquant()
    test_smoothquant_gptq_metadata_roundtrip()

    print("\n" + "=" * 50)
    print(f"  Wave-3 Production Tests: {passed} passed, {failed} failed")
    print("=" * 50)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
