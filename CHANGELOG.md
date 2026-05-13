# Changelog

All notable changes to NeuroQuant are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

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

[2.0.0]: https://github.com/AbdelazizElHelaly11/NeuroQuant/releases/tag/v2.0.0
