#!/bin/bash
# ===========================================================================
# NeuroQuant — Detection SMOKE TEST (SLURM batch job, ~10 min)
# ===========================================================================
# Run this BEFORE the full job to confirm the detection path end-to-end.
# Prereq: scripts/hpc_setup_detection.sh has been run on the login node.
#
#   cd ~/NeuroQuant
#   sbatch scripts/run_detection_smoke_hpc.sh
#   tail -f logs/detect_smoke_<JOBID>.log
#
# SUCCESS looks like (in the log):
#   * "FP32 baseline: test_top1=XX.XX%"   ← real mAP (NOT 0.00, NOT a crash)
#   * "[Phase 1c] NSGA-II: N Pareto solutions"
#   * "Pipeline complete" with phases_passed == phases_total
# ===========================================================================
#SBATCH --job-name=nq_detect_smoke
#SBATCH --partition=debug
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/detect_smoke_%j.log

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/neuroquant}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/NeuroQuant}"

mkdir -p "$PROJECT_DIR/logs"
echo "Smoke job $SLURM_JOB_ID started on $SLURM_NODELIST at $(date)"

source /nfs/slurm/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
echo "Python: $(which python)"

cd "$PROJECT_DIR"

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} ({p.total_memory // 1024**3} GB)")
PY

echo "== Launching detection SMOKE pipeline (PTQ-only, ~10 min) =="
neuroquant --config config_detection_smoke.yaml --epochs 0

echo "Smoke job finished at $(date)"
echo "Results in: $PROJECT_DIR/artifacts_detection_smoke/"
