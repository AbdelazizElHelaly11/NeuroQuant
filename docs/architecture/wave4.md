# Wave 4 — ONNX + hardware-aware search

## Decision matrix

| ID | Item                                            | Decision   |
| -- | ----------------------------------------------- | ---------- |
| J1 | Static INT8 ONNX export                         | Implement  |
| J3 | Real on-disk model size (.onnx file size)       | Implement  |
| J2 | ORT latency benchmark per method                | Implement  |
| C2 | Per-layer ORT latency LUT                       | Implement  |
| C1 | NSGA with 3 objectives [acc_loss, size, latency]| Implement  |
| J4 | Closed-loop hardware-aware search wiring        | Implement  |

## What shipped

### J1 · Static INT8 ONNX export
- New [`utils/onnx_export.py`](../../utils/onnx_export.py):
  - `export_to_onnx(model, input_shape, path)` — torch → ONNX FP32 via the legacy TorchScript exporter (stable across torch 2.4–2.11; the new dynamo path emits opsets the ORT static quantizer cannot preprocess).
  - `quantize_onnx_static(fp32_path, dst_path, calib_loader)` — runs `onnxruntime.quantization.quantize_static` with QDQ format, per-channel weights, QInt8 activations.
  - `OnnxUnavailable` raised explicitly when onnx/ort missing — callers must handle. No silent fake fallback: a synthetic answer would be worse than no answer.

### J3 · Real on-disk size
- `onnx_disk_size_mb(path)` returns the literal filesystem size.
- Method results in [`main.py:phase_1f_gptq_smooth_awq`](../../main.py) overwrite `model_size_mb` with the on-disk INT8 ONNX size; the synthetic `numel × bw / 8` figure is preserved as `theoretical_size_mb` for ablation.

### J2 · ORT latency
- `benchmark_onnx_latency(path, input_shape, batch_size, warmup_runs, measure_runs)` returns the canonical `{latency_mean_ms, latency_p50_ms, latency_p95_ms, throughput_fps}` dict.
- CPU provider by default; pass `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` to opt into GPU inference.
- Method results carry `onnx_latency` (full dict) and `latency_ms` (mean alias).

### C2 · Per-layer ORT latency LUT
- New [`quantization/latency_lut.py`](../../quantization/latency_lut.py): `build_latency_lut(model, input_shape, calib_loader, bitwidths=(4,8))`.
- Walks every Conv2d / Linear, captures real per-layer input shape via forward hook, builds a tiny single-op micro-graph, exports + statically quantizes + benchmarks each variant under ORT.
- `latency_for_assignment(assignment, lut)` — fast O(L) sum.
- INT4 row equals INT8 row by design — no deployment backend has native INT4 kernels (INT4 weights run on INT8 kernels after unpacking).
- Cached to `output_dir/latency_lut.json`; second run is instant.

### C1 · NSGA generalised to N objectives
- [`quantization/nsga_ii_search.py`](../../quantization/nsga_ii_search.py): `NSGAIIClusterSearch.__init__` accepts `latency_lut`. With LUT → 3-obj mode `[acc_loss, size_mb, latency_ms]`; without → unchanged 2-obj behaviour.
- `_dominates(a, b)` predicate generalised to N-tuples; `_non_dominated_sort` and `_crowding_distance` iterate `range(num_obj)` instead of hard-coding 2.
- `ParetoSolution.latency_mean_ms` populated when LUT present; `None` in 2-obj mode (backwards compat).
- Note: generalised NSGA-II + crowding distance is what `pymoo` deploys for ≤3 objectives in practice; reference-direction NSGA-III adds noticeable complexity for marginal benefit at the 8–32-individual populations this framework uses.

### J4 · Closed-loop wiring
- New phase 1c branch: when `hp.hardware_aware_search=True`, builds the LUT once, passes to `NSGAIIClusterSearch`, NSGA runs in 3-obj mode.
- Graceful degradation: ONNX unavailable → 2-obj fallback with WARNING; LUT build failure → 2-obj fallback with WARNING.
- `onnx_export_enabled` (default True) controls the J1+J2+J3 hook; `hardware_aware_search` (default False, opt-in) controls the C2+C1+J4 hook.

## Tests

[`test_wave4_production.py`](../../test_wave4_production.py) — 20 tests: ONNX round-trip equivalence, INT8 .onnx smaller than FP32, per-layer LUT covers all Conv/Linear, 3-obj non-dominated sort, latency-aware NSGA selection.

## Outcomes

- Framework moved from "INT simulation in FP32 tensors" to "real INT8 ONNX model on disk, benchmarked with ONNX Runtime".
- Smoke result: tiny CIFAR-class model 0.022 MiB FP32 → 0.011 MiB INT8 (real disk reduction, not synthetic estimate).
- LUT correctly surfaces ORT's well-known "small layers can be slower under INT8 due to QDQ overhead" effect — exactly the insight a real hardware-aware search must see.
