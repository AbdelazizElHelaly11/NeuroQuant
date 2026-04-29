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
```

Hardware report parser accepts JSON or CSV and logs DSP/LUT/FF/Fmax/II/cycle latency when provided.

## 6. MLflow UI

```powershell
mlflow ui --backend-store-uri .\mlruns --port 5000
```

If `mlflow` command is not recognized:

```powershell
python -m mlflow ui --backend-store-uri .\mlruns --port 5000
```

Open: `http://localhost:5000`

## 7. Output locations

- `.\artifacts\` - pipeline outputs, plots, reports, checkpoints
- `.\artifacts\checkpoints\` - per-phase resume state
- `.\mlruns\` - MLflow tracking data
