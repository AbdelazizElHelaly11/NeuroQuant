#!/bin/bash
# ===========================================================================
# NeuroQuant — Image Classification Pipeline (SLURM batch job)
# ===========================================================================
# Submit from the project root:
#   cd ~/neuroquant
#   sbatch scripts/run_classification_hpc.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/classification_<JOBID>.log
# ===========================================================================
#SBATCH --job-name=nq_class
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/classification_%j.log

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/neuroquant}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/neuroquant}"

mkdir -p "$PROJECT_DIR/logs"
echo "Job $SLURM_JOB_ID started on $SLURM_NODELIST at $(date)"

# ── Conda ──────────────────────────────────────────────────────────────────
source /nfs/slurm/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
echo "Python: $(which python)"

cd "$PROJECT_DIR"

# ── GPU check (nvidia-smi is blocked; use torch) ───────────────────────────
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} ({p.total_memory // 1024**3} GB)")
print("CUDA version:", torch.version.cuda)
PY

# ── Run the full classification pipeline ───────────────────────────────────
echo "== Launching NeuroQuant classification pipeline =="
neuroquant --config config_classification.yaml

echo "Job finished at $(date)"
echo "Results in: $PROJECT_DIR/artifacts_classification/"
