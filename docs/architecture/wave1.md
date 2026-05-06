# Wave 1 — Foundation: security + correctness + determinism

## Decision matrix

| ID | Item                                         | Decision   |
| -- | -------------------------------------------- | ---------- |
| N1 | Replace pickle path with safe state_dict     | Implement  |
| M1 | Centralised numerical constants              | Implement  |
| M2 | Strict determinism via `set_seed`            | Implement  |
| A1 | Train/search/val/test split                  | Implement  |
| A2 | Test-set Top-1 as public headline            | Implement  |

## What shipped

### N1 · Safe checkpoints
- New `save_safe_module` / `load_safe_module` in [`utils/checkpointing.py`](../../utils/checkpointing.py): writes `{state_dict, metadata}` envelopes; loads with `weights_only=True`.
- Architectural wrappers (SmoothQuant `_SmoothInputScale`, AWQ `_AWQInputScale`) persist as JSON manifests; the loader rebuilds them before `load_state_dict`.
- All `torch.load` calls now pass `weights_only=True` — pickle RCE path closed.

### M1 · Numerical constants
- New [`utils/numerics.py`](../../utils/numerics.py) centralises `EPS_PROB`, `MIN_SCALE`, `MIN_DAMP`, `MIN_MIGRATION`, `MAX_MIGRATION`, `EPS_GEOMETRIC`.
- Each constant has an env-var override for ablation.
- Replaces 30+ scattered `1e-8` literals across the quantization modules.

### M2 · Strict determinism
- `set_seed(seed, strict=True)` in [`utils/common.py`](../../utils/common.py):
  - `PYTHONHASHSEED` for stable dict iteration.
  - `CUBLAS_WORKSPACE_CONFIG=":4096:8"` for deterministic cuBLAS GEMM.
  - `torch.use_deterministic_algorithms(True, warn_only=True)`.
  - `torch.backends.cudnn.deterministic = True`, `benchmark = False`.
- Must be called before the first CUDA context / DataLoader fork — pipeline calls it in `NeuroQuantPipeline.__init__`.

### A1 · Loader split isolation
- `data/data_loader.py:GenericDatasetLoader` carves a 10% search slice from the train set with **eval-time transforms** (no augmentation).
- `get_search_loader()` returns this slice; it is disjoint from `val` and `test` by construction.
- NSGA fitness reads `search_loader`, QAT early-stop reads `val_loader`, headline accuracy reads `test_loader`. No cross-contamination.

### A2 · Test-set headline accuracy
- New `_attach_split_metrics(result, model)` recomputes `val_top1` + `test_top1` from the model directly so the headline is independent of whichever loader the upstream evaluator happened to use.
- `result["accuracy"]` is now an alias for `test_top1` — the deployment-time estimate, not the val number used for early stopping.

## Tests

[`test_wave1_production.py`](../../test_wave1_production.py) — 30 tests covering safe pickle, deterministic seed, split disjointness, headline integrity.

## Outcomes

- Removed every pickle-based `torch.load(weights_only=False)` path.
- Eliminated NSGA over-fitting to val (was using same loader as headline).
- All 414 baseline tests pass under strict determinism.
