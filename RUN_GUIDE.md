# NeuroQuant v2.0 — Run Guide (Windows)

## 1. Setup

```powershell
Set-Location "C:\Users\hzezo\OneDrive\Desktop\Graduation Project\NeuroQuant"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want CUDA wheels explicitly:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Quick device check:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Pre-run sanity checks

```powershell
python test_metrics.py
python test_ptq_wiring.py
python test_genericity.py
```

## 3. Running the pipeline

### A) Full pipeline (recommended first meaningful run)

```powershell
python main.py --config config.yaml --epochs 20
```

This runs the complete flow:
`phase_0_preparation -> phase_1a_hessian_clustering -> phase_1b_fitcompress -> phase_1c_nsga_search -> phase_1d_adaround -> phase_1e_qat -> phase_1f_gptq_smooth_awq -> phase_2_pareto -> phase_3_xai -> phase_4_mlflow`

### B) Fast smoke run

```powershell
python main.py --config config.yaml --epochs 1 --device cpu --phases phase_0_preparation phase_1a_hessian_clustering phase_1b_fitcompress
```

### C) Quantize/evaluate from existing FP32 checkpoint (no training)

Set checkpoint path in `config.yaml`:

```yaml
model:
  path: ".\artifacts\checkpoint_fp32.pth"
```

Then run:

```powershell
python main.py --config config.yaml --epochs 0
```

> `--epochs 0` without a valid checkpoint means the model is untrained, so results are not meaningful.

### D) Resume after interruption

```powershell
python main.py --config config.yaml --epochs 20 --resume
```

## 4. Run only selected phases

```powershell
python main.py --config config.yaml --epochs 3 --phases phase_0_preparation phase_1a_hessian_clustering phase_1b_fitcompress
python main.py --config config.yaml --epochs 0 --phases phase_0_preparation phase_1f_gptq_smooth_awq
python main.py --config config.yaml --epochs 0 --phases phase_2_pareto phase_3_xai phase_4_mlflow
```

## 5. Edge-oriented metrics (already supported)

Use these keys in `config.yaml`:

```yaml
hyperparams:
  eval_primary_accuracy: "top5"          # report focus (top1/top5)
  nsga_accuracy_objective: "top1"        # NSGA objective (top1/top5)
  latency_warmup_runs: 10
  latency_measure_runs: 50
  latency_batch_size: 1
  hardware_report_path: ".\reports\hardware_metrics.json"
  use_latency_in_pareto: false           # set true to add latency objective
  qat_warmstart_source: "ptq_best_acc"   # ptq_best_acc | ptq_best_tradeoff
  ptq_real_rerank_topk: 3                # NSGA → real-PTQ rerank size
  ptq_tradeoff_max_acc_drop: 1.0         # max Top-1 drop (pp) for tradeoff pick
```

Hardware report parser accepts JSON or CSV and logs DSP/LUT/FF/Fmax/II/cycle latency when provided.

## 6. MLflow UI

### A) Local only (default, safest)

```powershell
python scripts\serve_mlflow.py
```

Equivalent to `mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000`. Open: `http://localhost:5000`.

If the `mlflow` shim isn't on PATH the launcher automatically falls back to `python -m mlflow`.

### B) Share on your LAN

Bind on all interfaces so collaborators on the same network can browse runs from their own machines:

```powershell
python scripts\serve_mlflow.py --host 0.0.0.0 --port 5000
```

Then share `http://<your-LAN-ip>:5000`. The launcher prints a warning when binding to `0.0.0.0`. **Do not expose this directly to the public internet without auth** — anyone on that network can read every run.

### C) Public sharing (tunnel)

For demos / supervisor reviews, tunnel the local UI through Cloudflare or ngrok rather than opening a port:

```powershell
# Cloudflare quick-tunnel (no account required, ephemeral URL)
python scripts\serve_mlflow.py --tunnel cloudflared

# ngrok (one-time `ngrok config add-authtoken <TOKEN>`)
python scripts\serve_mlflow.py --tunnel ngrok
```

Each tool prints a public `https://...` URL in its own log. The tunnel is torn down automatically when the launcher exits (Ctrl+C). For anything beyond a quick demo, put basic auth or Cloudflare Access in front.

### D) Behind a reverse proxy (production-style)

If you already run nginx/Caddy, serve the local MLflow UI as a path-prefixed virtual host:

```nginx
location /mlflow/ {
    proxy_pass http://127.0.0.1:5000/;
    auth_basic           "NeuroQuant runs";
    auth_basic_user_file /etc/nginx/.htpasswd;
}
```

Run the launcher as in (A) and let the proxy handle TLS + auth.

## 7. Reading the outputs

### Final report contract

- Public accuracy: **Top-1 only**. Top-5 is computed internally for diagnostics but never appears in the summary table, MLflow public keys, or plot annotations.
- Method names use **canonical bitwidth-tagged IDs**: `PTQ_INT8`, `PTQ_MIXED`, `QAT_INT8`, `GPTQ_INT8`, `GPTQ_INT4`, `AWQ_INT4`, `AWQ_INT8`, `SmoothQuant_INT8`, `SmoothQuant_INT4`. There is no `AWQ_AWQ_INT4` or `PTQ_PTQ_best`-style duplication anywhere.
- NSGA-II solutions (IDs prefixed `nsga_…`) are **search-internal only** — they are kept in `artifacts/checkpoints/phase_1c_nsga_search.json` for reproducibility but never appear on the public Pareto plot, ranking table, summary table, or XAI matrix.
- Optimisation objectives surfaced to the user: **Top-1 accuracy ↑** and **Model size (MiB) ↓** (with EBops kept as the equivalent low-level byte count). Model size is computed from the actual bitwidth assignment using `1024 × 1024` (MiB).

### PTQ family — multi-fidelity rerank + dual outputs

NSGA-II searches with fast fake-quant; the proxy ranking is not the same as the real PTQ ranking. Phase 1c materialises the **top-K NSGA candidates** (default `K=3`, knob: `ptq_real_rerank_topk`) through `PTQQuantizer` with **bitwidth-aware calibration** — each layer gets a KL/MSE threshold computed at *its own* target bitwidth, so INT4 layers do not inherit an INT8 clipping range.

Two PTQ models can be surfaced when the candidates differ:

- **`ptq_best_acc`** — the candidate with the highest real Top-1.
- **`ptq_best_tradeoff`** — the most compressed candidate within `ptq_tradeoff_max_acc_drop` (default `1.0` pp) of the best Top-1. If no candidate satisfies the cap, the smallest-size candidate is picked as a knee-like fallback.

Both PTQ entries appear in the public Pareto/scatter/report when distinct (canonical IDs `PTQ_INT8`, `PTQ_INT4`, `PTQ_MIXED`); duplicates are collapsed automatically. The per-layer assignment of a mixed PTQ also flows into the `bitwidth_dist.png` plot so the colour split reflects the real assignment.

### QAT warmstart policy (hybrid PTQ→QAT)

The QAT/Adaround warmstart source is **explicit** and configurable via `qat_warmstart_source`:

| Value                  | Behaviour                                              |
|------------------------|--------------------------------------------------------|
| `ptq_best_acc` (default) | QAT warmstarts from the highest-Top-1 PTQ pick.        |
| `ptq_best_tradeoff`    | QAT warmstarts from the most-compressed PTQ within the accuracy cap. |

The chosen source and the selected PTQ display name are logged at phase 1c, persisted in the phase-1c JSON checkpoint, attached to the phase-1e QAT model checkpoint metadata, and surfaced as MLflow params (`qat_warmstart_source`, `qat_warmstart_id`). Reading any of these tells you exactly which PTQ artefact was used to seed QAT.

### Pareto plot (`artifacts/pareto/pareto_scatter.png`)
- White publication theme; per-method colour and marker (FP32 ◆ black, PTQ ● blue, QAT ■ orange, GPTQ ▲ green, AWQ ✚ purple, SmoothQuant ✕ red).
- **X-axis: Model size (MiB)**, Y-axis: Top-1 accuracy.
- Each point is annotated with its bitwidth-tagged solution ID (e.g. `GPTQ_INT8`).
- Marker size scales with compression ratio.
- Highlighted markers: ▲ best accuracy, ▼ smallest size, ★ knee point.
- The dashed line is a visual frontier connector — not a regression fit.

### Adaround diagnostics

Phase 1d optimises **per-layer output reconstruction** (`||layer(X; w_q) − layer(X; w)||²`), not the trivial weight-MSE objective. The MLflow run logs:
- `adaround_mse_before` / `adaround_mse_after` / `adaround_mse_reduction` — weight-space MSE.
- `adaround_recon_before` / `adaround_recon_after` / `adaround_recon_reduction` — the *real* layer-output reconstruction (only when a `calib_loader` was supplied; this is the case in the standard pipeline).
- `objective_components` (in the phase-1d checkpoint JSON): final values of each loss component (`final_total`, `final_recon`, `final_weight_mse`, `final_reg`) and the objective tag (`layer_output_reconstruction` vs the `weight_mse_fallback` legacy path).
- `alpha_stats`: per-parameter rounding statistics (`n_near_zero`, `n_near_one`, `n_near_half`).

### Comparison matrix (`artifacts/xai/comparison_matrix.png`)
- **Columns** are sample images. The header row shows each input thumbnail with its index (`sample #i`) and ground-truth label (`GT: <class>`).
- **Rows** are quantization techniques with bitwidth-tagged labels: `FP32_baseline`, `PTQ_INT8`, `QAT_INT8`, `GPTQ_INT8`, `GPTQ_INT4`, `AWQ_INT4`, `AWQ_INT8`, `SmoothQuant_INT8`, `SmoothQuant_INT4` — whichever variants ran in the configured `methods` set. Each row header also reports per-technique accuracy on these samples (e.g. `4/5 correct`).
- Each **cell** overlays the Grad-CAM heatmap onto the input and adds a caption underneath: `pred: <class>  (confidence%)  ✓/✗`. Green ✓ = prediction matches GT, red ✗ = misprediction.

### Per-image Grad-CAM PNGs (`artifacts/xai/grad_cam/<technique>_img<i>.png`)
Same caption as in the matrix, plus a coloured ✓/✗ badge below. Useful for embedding individual heatmaps in the report.

### XAI markdown report (`artifacts/xai/<...>.report` or in MLflow)
Adds a "Predictions per Sample" table and a "Top-1 Accuracy on Explained Samples" summary so the technique vs prediction story is queryable in plain text.

## 8. Output locations

- `.\artifacts\` - pipeline outputs, plots, reports, checkpoints
- `.\artifacts\checkpoints\` - per-phase resume state
- `.\artifacts\pareto\` - Pareto scatter, bitwidth bar, ranking table, JSON
- `.\artifacts\xai\` - comparison_matrix.png + per-(technique,sample) heatmaps
- `.\mlruns\` - MLflow tracking data
