# Wave 5 — Reporting + MLflow

## Decision matrix

| ID | Item                                                | Decision   |
| -- | --------------------------------------------------- | ---------- |
| G1 | Public report adds ONNX columns                     | Implement  |
| G2 | Pareto plot uses real ONNX size + 3-D plot          | Implement  |
| G3 | Reproducibility manifest captures ORT + LUT + ONNX  | Implement  |
| G4 | Deployment-fidelity section in final report         | Implement  |
| I1 | MLflow logs ONNX size / latency / throughput        | Implement  |
| I2 | ONNX artefacts attached to MLflow runs              | Implement  |
| I3 | Pareto comparison summary in MLflow + JSON          | Implement  |

## What shipped

### G1 · Headline summary row schema
- [`main.py:_add_summary_row`](../../main.py) accepts three new kwargs: `onnx_size_mb`, `onnx_latency_ms`, `onnx_throughput_fps`.
- Public report adds three columns (`ONNX MiB`, `ORT(ms)`, `ORT FPS`) when any row has ONNX numbers; falls back cleanly to the original layout otherwise.
- Every existing call site (Phase 0 + Phase 1c rerank + Phase 1f + every resume path) wired to forward the ONNX values.
- The `size_mb` column now reflects the *theoretical* synthetic number so the table makes the "synthetic vs real on-disk" delta legible side-by-side.

### G2 · Pareto plots
- 2-D `plot_pareto_scatter` already used `model_size_mb`, which Wave 4 overwrote with the on-disk INT8 ONNX size — verified, no change needed.
- New `plot_3d_pareto` in [`visualization/pareto_analysis.py`](../../visualization/pareto_analysis.py): Top-1 vs ONNX size vs ORT latency, using the per-method palette from `visualization.style.METHOD_STYLE`. Generated only when at least one solution carries `latency_mean_ms`.
- `compute_solution_metrics` forwards `latency_mean_ms` to the enriched solution dict so the 3-D plot has data without a separate lookup.

### G3 · Reproducibility manifest
- [`utils/checkpointing.py:save_reproducibility_manifest`](../../utils/checkpointing.py):
  - New `onnx_runtime` block: ORT version + list of compiled providers (`CPUExecutionProvider`, `CUDAExecutionProvider`, …) so reports can never claim "INT8 on CUDA" when the binary only had CPU.
  - `packages` block extended with `onnx`, `onnxruntime`, `onnxscript`.
  - New `deployment` block: `fp32_onnx_path`, `fp32_onnx_size_mb`, `fp32_onnx_latency_mean_ms`, `fp32_onnx_throughput_fps`, `latency_lut_path`, plus a `latency_lut_present_on_disk` flag.

### G4 · Deployment-fidelity section
- New `_print_deployment_fidelity_section` in [`main.py`](../../main.py): silent when no ONNX rows exist; otherwise prints
  - FP32 ONNX baseline size + ORT latency,
  - Mean (on-disk / theoretical) size ratio across quantized methods,
  - Median quantized ORT latency + speedup vs FP32 ONNX,
  - Per-method size delta and ORT speedup line for every quantized method.

### I1 + I2 · MLflow ONNX metrics + artefacts
- Phase 0 logs `fp32_onnx_size_mb`, `fp32_onnx_latency_mean_ms`, `fp32_onnx_throughput_fps`, attaches the FP32 `.onnx` to the run.
- Phase 1f logs per-method `{tag}_onnx_size_mb`, `{tag}_onnx_latency_{mean,p50,p95}_ms`, `{tag}_onnx_throughput_fps`, plus `{tag}_theoretical_size_mb` for ablation; attaches each method's INT8 `.onnx` artefact.

### I3 · Pareto comparison summary
- New `_build_pareto_summary` aggregates the public method rows into best/median/worst stats over Top-1, theoretical size, on-disk ONNX size, and ORT latency.
- Top-1 stats correctly use max-first ordering (best = highest accuracy); size/latency stats use min-first.
- Phase 4 logs `pareto_top1_best`, `pareto_onnx_size_mb_best`, `pareto_onnx_latency_ms_best`, etc., to MLflow.
- Full summary written to `pareto_summary.json` and attached to MLflow under `reports/`.

## Tests

[`test_wave5_production.py`](../../test_wave5_production.py) — 18 tests covering summary row schema, deployment fidelity printer, manifest blocks, MLflow tracker accepting ONNX keys, summary builder correctness.

## Outcomes

- Reports are now self-explanatory at the deployment level: a reader sees both the synthetic estimate and the real on-disk number, with the ORT latency speedup vs FP32 baseline computed automatically.
- Every quantized method ships with a downloadable INT8 `.onnx` attached to its MLflow run.
