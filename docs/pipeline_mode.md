# Using the CLI Pipeline

> Audience: **researchers** who want a full, reproducible quantization
> run from a single config file.

The CLI runs all 10 NeuroQuant phases end-to-end, writes every artefact
to `./artifacts/`, logs every metric to MLflow, and produces an HTML
report when it's done. You give it one YAML file. It does everything
else.

## 1 · Initialise a config

After `pip install neuroquant`, drop a fresh, fully-commented config in
your working directory:

```bash
neuroquant --init
```

This copies the bundled template (`neuroquant/quantization/_default_config.yaml`)
to `./config.yaml`. If a config already exists, the command refuses
unless you pass `--force`.

!!! tip "What's inside the template"

    Every knob ships with an inline comment explaining what it
    controls — calibration batches, NSGA population size, surrogate
    warmup, hardware-aware mode, deployment-export toggles, XAI
    samples, MLflow URI. Skimming `config.yaml` is the fastest way
    to learn the framework.

## 2 · Run the pipeline

Once the config exists, kick off the run:

```bash
neuroquant --config config.yaml
```

Add `--resume` to skip phases that already have checkpoints in
`./artifacts/checkpoints/`:

```bash
neuroquant --config config.yaml --resume
```

??? note "Selecting which phases run"

    The `phases:` block in `config.yaml` is a whitelist. Comment a
    line out — say `phase_3_xai` — and that phase is skipped. The
    pipeline still walks the registry in the fixed Phase 0 → Phase 4
    order; only the listed phases execute.

## 3 · What the YAML actually configures

The config is split into thematic blocks. Here are the highest-impact
ones:

=== "Model & task"

    ```yaml
    model:
      name: "mobilenetv2"
      num_classes: 10
      input_shape: [3, 32, 32]
      task: "classification"   # classification | detection | segmentation
    ```

    The `task` field is what makes NeuroQuant task-agnostic — the
    model loader, the XAI backward pass, and the data-loader's
    collate function all dispatch on this single value.

=== "Dataset"

    ```yaml
    dataset:
      name: "cifar10"
      path: "./data"
      batch_size: 128
      num_workers: 4
    ```

    Built-in names: `cifar10`, `cifar100`, `imagefolder`, `synthetic`.
    Anything else is treated as a `torchvision.datasets.<Name>` class
    name. Set `dataset.class` to a fully-qualified Python path if you
    need a custom `torch.utils.data.Dataset`.

=== "NSGA-II search"

    ```yaml
    hyperparams:
      nsga_population_size: 32
      nsga_generations: 30
      nsga_search_mode: "per_layer"          # HAWQ-V3 / HAQ style
      nsga_use_surrogate: true                # BRP-NAS / OFA surrogate
      nsga_surrogate_warmup_evals: 30
      nsga_surrogate_proposed_per_gen: 256
      hardware_aware_search: true             # 3-obj acc/size/latency
    ```

=== "Deployment"

    ```yaml
    hyperparams:
      onnx_export_enabled: true       # ORT INT8 static + FP32 baseline
      latency_lut_bitwidths: [4, 8]   # per-layer LUT profile bitwidths
    ```

    Set `task: detection` to also auto-export TensorRT / OpenVINO
    artefacts when the host has the backends installed.

## 4 · The 10 phases, one by one

| Phase | Name                         | What it produces                                                                          |
| ----- | ---------------------------- | ----------------------------------------------------------------------------------------- |
| 0     | Model & Data Preparation     | FP32 baseline accuracy + latency + size. Optionally trains for `--epochs N`.              |
| 1a    | Hessian / Fisher clustering  | Per-layer sensitivity + 3-tier (HIGH/MEDIUM/LOW) cluster assignment.                      |
| 1c    | Surrogate-Assisted NSGA-II   | Pareto front of mixed-precision configs. Hardware-aware when an ORT LUT is available.    |
| 1d    | AdaRound                     | Learned weight rounding refining the chosen mixed-precision config.                       |
| 1e    | QAT + Knowledge Distillation | Quantization-aware fine-tuning with the FP32 teacher.                                     |
| 1f    | GPTQ + AWQ + SmoothQuant     | Three baselines at INT4 and INT8 each, plus the combined SmoothQuant→GPTQ.               |
| 2     | Pareto analysis              | Hypervolume, knee point, bitwidth-distribution / scatter / metrics-table plots.           |
| 3     | XAI (Grad-CAM + SHAP)        | Per-technique heatmaps, predictions table, error attribution plots.                       |
| 4     | MLflow finalisation          | Summary metrics, pareto_summary.json, HTML report.                                        |

!!! info "Phase 1b is intentionally absent"

    A previous version had a `phase_1b_fitcompress` warm-start step.
    Hessian clustering plus the surrogate handle the search-space
    shaping it provided — the seed phase is now redundant and has
    been removed. The phase IDs stay non-contiguous so historical
    checkpoints still resolve.

## 5 · Output layout

After a successful run the working directory looks like:

```
artifacts/
├── checkpoints/                # one JSON per phase — drives --resume
│   ├── phase_0_preparation.json
│   ├── phase_1a_hessian_clustering.json
│   ├── ...
├── plots/                      # Pareto + sensitivity + bitwidth charts
├── error_attribution/          # per-method per-layer error PNGs
├── xai/                        # Grad-CAM heatmaps + comparison matrix
├── onnx/                       # FP32 + INT8 ONNX exports per method
│   ├── ptq_mixed.onnx
│   ├── ptq_mixed.int8.onnx
│   └── ...
├── pareto_summary.json
├── pipeline_report.txt
└── report.html                 # opens in any browser
mlruns/                         # MLflow tracking server data
```

Open `artifacts/report.html` in a browser to read the run end-to-end —
methods table, Pareto plots, sensitivity heatmap, XAI grid, error
attribution per method, deployment fidelity caveats.

## 6 · Common workflows

??? example "Resume after a Phase 1f crash"

    A pre-emptible VM kills the job during AWQ. You re-launch:

    ```bash
    neuroquant --config config.yaml --resume
    ```

    Phases 0–1e load from their JSON checkpoints (instant). Phase 1f
    re-runs from scratch. The Pareto front is preserved.

??? example "Run only Phase 2 + 3 to regenerate plots"

    Trim `phases:` in `config.yaml`:

    ```yaml
    phases:
      - phase_2_pareto
      - phase_3_xai
      - phase_4_mlflow
    ```

    Then `neuroquant --config config.yaml --resume`. Phase 0 / 1*
    are skipped; their checkpoints are read back into memory; Phase 2
    rebuilds plots from the loaded Pareto front; Phase 3 re-renders
    XAI; Phase 4 rewrites the report.

??? example "Run with a different model on the same dataset"

    Copy the config, change `model.name`, point `output.dir` at a
    fresh folder so MLflow runs stay separate, and launch:

    ```bash
    neuroquant --config config.resnet50.yaml
    ```

## 7 · MLflow integration

NeuroQuant logs every phase as its own MLflow run, parented under the
experiment named `neuroquant.<model_name>.<dataset_name>`. To browse:

```bash
mlflow ui --backend-store-uri ./mlruns
# or use the bundled wrapper:
python scripts/serve_mlflow.py
```

Each phase run carries phase-level metrics (e.g. `pareto_solutions`,
`fit_compression`, `nsga_evaluations`) and pointers to the artefacts
on disk; the `phase_4_summary` run aggregates the headline Pareto stats
so MLflow's *Compare Runs* view becomes a one-click cross-experiment
table.

[:octicons-arrow-right-24: Next: integrate the same quantizers into your own training script](library_mode.md)
