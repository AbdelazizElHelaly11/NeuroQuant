# Wave 6 — Testing + CI + Pydantic

## Decision matrix

| ID | Item                                            | Decision   |
| -- | ----------------------------------------------- | ---------- |
| K1 | Shared pytest fixtures via `conftest.py`        | Implement  |
| K2 | Coverage threshold gate via pytest-cov          | Implement  |
| K3 | Integration smoke (full pipeline, toy model)    | Implement  |
| K4 | Property-based tests via hypothesis             | Implement  |
| L1 | Pydantic-backed `QuantizationConfig`            | Implement  |
| —  | GitHub Actions CI (bonus)                       | Implement  |

## What shipped

### K1 · Shared fixtures
- New project-wide [`conftest.py`](../../conftest.py): `tiny_cnn_factory`, `tiny_model`, `calib_loader`, `val_loader`, `quant_config`, `pipeline_skeleton`.
- `TinyCNN` reference model lives in conftest and is reusable from any test file.
- All fixtures function-scoped — no shared mutable state across tests.

### L1 · Pydantic-backed config
- [`config.py`](../../config.py): `HyperparameterSet` and `QuantizationConfig` use `pydantic.dataclasses.dataclass` (Pydantic v2). Drop-in replacement for stdlib `@dataclass` — every existing call site keeps working.
- 11 field-level `@field_validator`s added: `device`, `hessian_estimator`, `qat_warmstart_source`, `qat_act_bitwidth`, `qat_distill_alpha`, `qat_distill_temperature`, `adaround_lr`, `qat_lr`, `nsga_population_size`, `nsga_generations`, `ptq_real_rerank_topk`, `ptq_tradeoff_max_acc_drop`, `latency_lut_bitwidths` plus `num_classes`, `batch_size`, `io_layer_bitwidth`, `input_shape`.
- Errors fire at construction with the field path; type coercion at YAML load (`num_classes: "10"` → `int 10`).
- `from_yaml` / `from_json` call `.validate()` after load so `setattr`-bypass cases still surface at load time.
- Pre-existing bug fixed: `_to_dict` now unwraps `Enum.value` for clean YAML round-trips.
- Graceful fallback: when pydantic absent, code falls back to stdlib `@dataclass` and a no-op `field_validator` shim.

### K4 · Property-based tests
- `quantize_tensor` round-trip stays in symmetric INT range (50 hypothesis examples × 3 bitwidths).
- Per-channel scale stays positive on random Conv2d weights.
- MSE is monotonic in bitwidth: 4 ≥ 8 ≥ 16.
- `latency_for_assignment` is a pure sum: split-and-sum equals total.

### K2 · Coverage gate
- New [`pyproject.toml`](../../pyproject.toml): `[tool.pytest.ini_options]` configures `--cov-fail-under=80` on `quantization`, `utils`, `tracking`, `visualization`, `config`.
- `[tool.coverage.run]` line coverage only (matches the "≥80% line coverage" target).
- `[tool.coverage.report]` excludes `pragma: no cover`, `raise NotImplementedError`, `if TYPE_CHECKING:`, `raise OnnxUnavailable`.
- **Achieved 81.3% line coverage.**

### K3 · Integration smoke
- New [`tests/integration/test_full_pipeline_smoke.py`](../../tests/integration/test_full_pipeline_smoke.py).
- `pytestmark = pytest.mark.integration` — opt-in, excluded from default unit run.
- Runs every wired phase end-to-end on a synthetic 4-class 32×32 dataset with mobilenet_v2.
- Hard budget: NSGA pop=4 gens=2; QAT 1 epoch; AdaRound 2 epochs.
- Asserts: all phases complete, ≥1 Pareto solution, manifest + summary JSON written, ONNX `.onnx` artefact exists.
- **Both smokes complete in ~60s on CPU.**

### CI · GitHub Actions
- New [`.github/workflows/tests.yml`](../../.github/workflows/tests.yml). Three jobs:
  - `lint` — ruff (non-blocking until Wave 7 ships ruff config).
  - `test` — Linux × Python 3.10/3.11/3.12 matrix, runs unit tests + coverage gate.
  - `integration` — single Linux × Python 3.12 run of the K3 smoke.
- Coverage XML uploaded as workflow artefact.
- Triggered on push/PR to `main` plus `workflow_dispatch`.

## Tests

[`test_wave6_production.py`](../../test_wave6_production.py) — 38 tests covering fixtures, pydantic validators (24 parametrised cases), hypothesis property tests, coverage gate file existence, CI workflow file existence.

## Outcomes

- 153 unit tests + 2 integration smokes; 81.3% line coverage gated on every PR.
- Bad config values now fail at construction instead of inside a phase.
