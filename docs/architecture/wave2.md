# Wave 2 — Real W+A QAT pipeline

## Decision matrix

| ID | Item                                       | Decision   |
| -- | ------------------------------------------ | ---------- |
| E4 | Conv-BN folding before QAT                 | Implement  |
| E1 | Real W+A quantization (INT8 activations)   | Implement  |
| E3 | Activation observer state machine          | Implement  |
| E5 | FP32 teacher KD distillation               | Implement  |
| D1 | AdaRound canonical-ordered traversal       | Implement  |

## What shipped

### E4 · Conv-BN folding
- New [`quantization/bn_folding.py`](../../quantization/bn_folding.py): `fold_conv_bn(model)` applies the analytic fusion `weight ← weight × γ/σ; bias ← (bias − μ)γ/σ + β` and replaces the BN with `nn.Identity`.
- Enabled by `qat_fold_bn: true` (production default). Disabling is for ablation only — every supported INT backend (qnnpack, fbgemm, ORT, TensorRT) requires fused conv-BN.

### E1 · Real W+A QAT
- [`quantization/qat.py`](../../quantization/qat.py) rewritten:
  - `_FakeQuantizeSTE` autograd `Function` with proper STE backward mask (was previously bypassed via `mod.weight.data = ...`).
  - `_WeightFakeQuantize` parametrization via `torch.nn.utils.parametrize` — autograd-aware, gradients flow through quantization correctly.
  - Activation INT8 always (`qat_act_bitwidth: 8`); the deployment shape every supported backend expects.

### E3 · Activation observer
- 3-phase state machine: `passthrough` (collect ranges) → `calibrating` (set scale from KL/MSE) → `quantizing` (fake-quantize forward).
- Replaces ad-hoc forward hooks with a proper `_ActivationObserver` module.
- `_QuantizationManager` orchestrates state transitions across all observed layers.

### E5 · KD distillation
- New `qat_distill_alpha` + `qat_distill_temperature` knobs.
- Loss: `α · T² · KL(student/T || teacher/T) + (1-α) · CE`.
- Teacher is `copy.deepcopy(self.model)` (FP32) frozen at QAT start.
- `α=0` disables KD; production default `α=0.5`.

### D1 · AdaRound canonical-ordered traversal
- New `optimize_ordered()` in [`quantization/adaround.py`](../../quantization/adaround.py): iterates Conv/Linear in topological order, propagates each layer's quantized output into the downstream activations the next layer sees.
- Replaces the parallel variant (which ignored upstream error and consistently underperformed on deep networks).
- New `_collect_activations_for_one_layer(pname, max_samples)` streaming collector — constant memory regardless of model depth.
- **Critical fix**: filter `_target_params` to Conv/Linear weights only via `_is_quantizable_weight` (was previously including BN weights, causing AdaRound to optimise things it couldn't quantize).

## Tests

[`test_wave2_production.py`](../../test_wave2_production.py) — 22 tests covering fold-equivalence, weight-parametrization gradient flow, observer state transitions, KD loss math, AdaRound topological order.

## Outcomes

- QAT models now match what gets deployed: INT8 weights + INT8 activations, BN folded, autograd-aware.
- Caught real bug in AdaRound (BN weights in target params) via the topological-order test.
