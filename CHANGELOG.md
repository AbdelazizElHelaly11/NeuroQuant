# Changelog

All notable changes to NeuroQuant are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [2.2.0] — 2026-06-11

End-to-end **semantic-segmentation** support and a ready-to-run Pascal
VOC / DeepLabV3 experiment (replicates the AdaRound paper's Table 9).
Previously segmentation was only partially wired (loss bridge + XAI);
the accuracy path crashed on the `OrderedDict` output and there was no
mIOU, no pretrained weights, and no VOC dataset.

### Added
- **Segmentation mIOU** — `compute_topk_accuracy` now dispatches on the
  output shape: a `[B, C, H, W]` (or `OrderedDict({"out": …})`) output is
  scored as per-class **mIOU in the `top1` slot** (+ pixel accuracy in
  `top5`), ignoring label 255. So the FP32 baseline, NSGA fitness, and
  every method headline get a real "higher is better" segmentation
  metric with no per-call-site task plumbing. Classification is
  unchanged.
- **Pretrained weights** — new `model.pretrained` flag loads
  torchvision's published `weights="DEFAULT"` (classification keeps the
  ImageNet backbone + re-adapts the head; segmentation/detection whose
  DEFAULT weights match `num_classes`, e.g. 21-class VOC DeepLabV3, use
  the full pretrained model — so the FP32 baseline is meaningful without
  training).
- **Pascal VOC adapter** — `neuroquant.data.voc_segmentation.VOCSegmentationDataset`
  reads the VOCdevkit layout directly with synchronized image+mask
  resize (image→bilinear, mask→nearest) and ImageNet normalization;
  `python -m neuroquant.data.voc_segmentation --download` fetches it on a
  login node. Maps NeuroQuant's `split="train"/"test"` to VOC
  `train`/`val` (the 1449-image val set becomes the held-out test split).
- **AdaRound is scored as a first-class method when QAT is skipped** —
  previously Phase 1d produced a refined model that only Phase 1e (QAT)
  consumed, so with QAT disabled (e.g. segmentation, which QAT does not
  support) AdaRound's accuracy was never reported. It now appears in the
  Pareto + summary. QAT-enabled (classification) runs are unchanged.
- **Task-aware pipeline — `task: segmentation` just works** — the
  pipeline auto-skips Phase 1e (QAT is classification-only and crashes on
  segmentation/detection/regression output) for non-classification tasks,
  so the *default* 9-phase config runs without any manual `phases` editing
  (the skipped QAT no longer counts as an incomplete phase). Grad-CAM /
  the XAI comparison grid now also tolerate `[H,W]` segmentation masks as
  ground-truth labels (dominant non-ignore class) instead of crashing on
  `.item()`. Classification behaviour is unchanged (QAT still runs).

## [2.1.1] — 2026-06-11

Pipeline-audit fixes. Each item was confirmed against the source in
this tree (the audit referenced a different layout, so every claim was
re-verified before changing anything). No public API changed; defaults
and behaviour are preserved except where a number was previously wrong.

### Fixed
- **FP32 baseline scored on the test split (C1)** — `phase_0` now
  reports `fp32_acc` from the same `test_loader` every quantized
  method uses for its headline, instead of the validation loader. The
  FP32-vs-quantized comparison (and `accuracy_loss`) is now
  apples-to-apples; the val number is retained as the diagnostic
  `fp32_val_acc` (persisted to the phase-0 checkpoint and restored on
  resume). Removes the systematic negative-`accuracy_loss` that
  inflated apparent quantization gains (also addresses **M1**).
- **Calibration runs on inference-time preprocessing (C3)** — PTQ /
  GPTQ / AWQ / SmoothQuant scale search, Fisher sensitivity, and
  AdaRound reconstruction now calibrate on an eval-transform view of
  the training split (`_calib_dataset`) instead of the randomly
  augmented training images. Applied to the torchvision and ImageFolder
  loaders; loaders whose training data is already eval-transformed
  (custom Dataset / synthetic / HuggingFace) are unaffected.
- **Mixed-precision size/EBops reported correctly (N4)** — the PTQ
  rerank now recomputes `ebops` from the real per-layer bitwidth
  assignment, so a `PTQ_MIXED` config's INT4 layers actually show a
  saving instead of reporting the same size as uniform INT8.
- **PTQ quantizes Linear weights per-output-channel (N2)** — Linear
  layers previously used a single per-tensor scale (much coarser than
  the per-channel scheme GPTQ/AWQ use and fragile at INT4). Now
  per-output-channel for both Conv2d and Linear; strictly lower
  reconstruction error and the standard deployable scheme.
- **QAT ONNX deployment fields survive resume (H4)** — `onnx_path`,
  `onnx_size_mb`, and `onnx_latency` for the QAT method are now
  persisted in the phase-1e checkpoint and restored on resume, so a
  resumed run no longer reports `null` size/latency for QAT. The
  resumed summary row also mirrors the original `QAT_MIXED` /
  `QAT_INT8` label rule.
- **`export_to_onnx` no longer strands the caller's model (N5)** — the
  function moved the live model to CPU/eval in place and left it there;
  it now captures and restores the original device + training mode in
  a `finally`, even on export failure.
- **In-pipeline FP32 training keeps the best-val weights (M8)** —
  `--epochs N` reported the best validation accuracy but handed
  downstream phases the last-epoch weights. The best-val `state_dict`
  is now snapshotted and restored, so the model matches the reported
  number.
- **Hypervolume is computed in normalized objective space (M2)** —
  accuracy-loss and EBops are each scaled to `[0, 1]` (consistent with
  `compute_spacing`) before the dominated-area product, replacing the
  dimensionally meaningless `points × bytes` value.
- **Config load-time validation covers the choice fields (M5)** —
  `validate()` now rejects invalid `nsga_search_mode`,
  `smoothquant_alpha_strategy`, `awq_alpha_strategy`, and out-of-set
  `supported_bitwidths` on the YAML/JSON load path (which bypasses the
  pydantic field validators via `setattr`).
- **Checkpoint loads surface skipped keys (L4)** — `ModelLoader`
  loads with `strict=False` by design (adapted head), but now logs
  missing / unexpected keys at WARNING so a genuinely mismatched
  checkpoint is visible rather than silent.
- **Model-agnostic latency note (H3, partial)** — the
  `latency_backend_note` no longer hard-codes a depthwise-conv
  rationale; it now describes the QDQ-overhead cause generically across
  CNN and transformer op families.
- **Consistent units + correct median (L1, L2)** —
  `BaseQuantizer.evaluate` reports model size in binary MiB (1024²) to
  match the rest of the pipeline; the Pareto summary and deployment
  sections now use a true median (averaging the two middle elements for
  even-length input) instead of the upper-middle element.
- **Reproducible training data order (L8)** — the train DataLoader now
  uses a seeded shuffle generator and a `worker_init_fn` that seeds each
  worker's numpy/random stream, closing a determinism gap on multi-worker
  (non-Windows) runs.

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

[2.2.0]: https://github.com/AbdelazizElHelaly11/NeuroQuant/releases/tag/v2.2.0
[2.1.1]: https://github.com/AbdelazizElHelaly11/NeuroQuant/releases/tag/v2.1.1
[2.1.0]: https://github.com/AbdelazizElHelaly11/NeuroQuant/releases/tag/v2.1.0
[2.0.0]: https://github.com/AbdelazizElHelaly11/NeuroQuant/releases/tag/v2.0.0
