# Changelog

All notable changes to NeuroQuant are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [2.1.0] — 2026-06-07

Multi-task expansion: regression, HuggingFace NLP, and Vision
Transformer XAI now land alongside the existing CV task families.

### Added
- **Regression task** (`task: regression`) — `MSELoss` bridge, new
  `compute_regression_metrics` returning RMSE / MAE / R², and a
  task-agnostic `evaluate_primary_metric` dispatcher that puts
  `-RMSE` in the canonical `top1` slot so NSGA-II's `fp32 − quant`
  objective math stays untouched.
- **HuggingFace NLP task** (`task: nlp`) — loss bridge invokes
  `model(**x, labels=y).loss`; new `_load_huggingface_dataset`
  branch activated by `hf:<name>` dataset prefix (e.g. `hf:imdb`,
  `hf:glue/sst2`) with auto-tokenisation from `model_name`.
  Optional `[nlp]` extras (`transformers`, `datasets`, `tokenizers`).
- **Vision Transformer XAI** — `AttentionRolloutExplainer`
  (Abnar & Zuidema, 2020) auto-routed when `is_vision_transformer`
  detects a ViT (≥1 `MultiheadAttention` and `num_attention >=
  num_conv2d`). Handles torchvision `vit_b_16` / `vit_l_16`, Swin,
  DeiT, and timm ViTs without code changes.
- **Hessian dict-input support** — `_move_to_device` helper
  recursively shifts tensors / dicts / lists-of-dicts onto the
  estimator's device, so NLP batches `{"input_ids":, "attention_mask":}`
  work in Phase 1a alongside CV batches.
- **Smoke tests** — `tests/test_regression.py`, `tests/test_vit_xai.py`,
  `tests/test_nlp_loss.py` (NLP test uses a stub HF model so it runs
  without the `[nlp]` extras installed).
- **Docs auto-deploy** — `.github/workflows/docs.yml` rebuilds and
  publishes the MkDocs Material site to `gh-pages` on every push to
  `main`.

### Changed
- `QuantizationConfig.task` validator now accepts `regression` and
  `nlp` in addition to the existing CV trio.
- `NSGAIIClusterSearch._evaluate_accuracy` routes through
  `evaluate_primary_metric` so regression Pareto search uses RMSE
  without any new objective code.
- `XAIGenerator.run` auto-detects ViTs and dispatches to Attention
  Rollout instead of failing on missing Conv2d feature maps.
- Documentation site bumped to v2.1 with new library examples for
  regression, NLP, and ViT plus an updated task dispatch table in
  the pipeline guide.

## [2.0.0] — 2026-05-14

The first PyPI release. Ships a flat library API alongside the existing
CLI pipeline, native multi-task support (classification / detection /
segmentation), and a documentation site at `docs/`.

### Added
- **Library mode** — flat `from neuroquant import PTQQuantizer, …`
  import surface; every quantizer accepts `config=None` and falls
  back to `QuantizationConfig()` defaults, so notebooks can drive
  quantization in three lines without YAML.
- **Multi-task pipeline support** — `task: detection` and
  `task: segmentation` are first-class in `config.yaml`; the CLI
  builds the right calibration loader, loss bridge, and forward
  contract for each. Library API works for all three.
- **Surrogate-assisted NSGA-II** (Phase 1c) — GradientBoosting model
  ranks mixed-precision candidates in microseconds so a single
  generation scans hundreds of configs instead of dozens.
- **Per-layer search mode** (HAWQ-V3 / HAQ style) alongside the
  legacy cluster-level encoding, selectable via
  `hyperparams.nsga_search_mode`.
- **Hardware-aware 3-objective search** — `(acc_loss, size,
  latency)` when a per-layer ORT latency LUT is supplied.
- **Task-aware XAI** — Grad-CAM and SHAP fallbacks dispatch on
  classification logits, detection score lists, and segmentation
  `OrderedDict({"out": ...})` outputs without per-task glue.
- **MkDocs documentation site** under `docs/` with separate paths
  for researchers (CLI pipeline) and developers (library mode).
- **Smoke test suite** under `tests/` + GitHub Actions matrix
  covering Ubuntu / Windows × Python 3.10 / 3.11 / 3.12.
- **`py.typed` marker** so downstream mypy / pyright pick up the
  type hints shipped with the package.

### Changed
- Repository restructured into the standard `neuroquant/` package
  layout. `main.py` → `neuroquant/cli.py`, registered as the
  `neuroquant` console-script entry point.
- Hessian sensitivity (`compute_hessian`) now takes an optional
  `loss_fn=(model, x, y) -> scalar` bridge so detection and
  segmentation work end-to-end without hard-coding `CrossEntropyLoss`.
- `data/data_loader.py` migrated to `torchvision.transforms.v2`
  with task-specific transform builders and a `detection_collate_fn`
  that produces the `(images_tuple, targets_tuple)` shape
  torchvision detectors expect.

### Removed
- **Phase 1b FITCompress** — redundant given Hessian-tier clustering
  plus the in-loop surrogate. Phase IDs stay non-contiguous so
  legacy checkpoints still resolve.

### Fixed
- AWQ now refuses to run on `task="detection"` with a clear
  `NotImplementedError` instead of a deep `torch.cat` traceback —
  dynamic activation shapes from the RPN / RoI heads are
  incompatible with AWQ's per-input-channel α search.
- PTQ / AWQ / Hessian device-move helpers now handle both
  `(images_tensor, labels_tensor)` and `(images_list,
  targets_list_of_dicts)` batch shapes.

[2.1.0]: https://github.com/AbdelazizElHelaly11/NeuroQuant/releases/tag/v2.1.0
[2.0.0]: https://github.com/AbdelazizElHelaly11/NeuroQuant/releases/tag/v2.0.0
