#!/bin/bash
# ===========================================================================
# NeuroQuant — Object-Detection pipeline (SLURM batch job)
# ===========================================================================
# Submit from the project root AFTER running scripts/hpc_setup_detection.sh:
#   cd ~/NeuroQuant
#   sbatch scripts/run_detection_hpc.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/detection_<JOBID>.log
# ===========================================================================
#SBATCH --job-name=nq_detect
#SBATCH --partition=debug
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/detection_%j.log

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/neuroquant}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/NeuroQuant}"

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

# ── Run the full detection pipeline ────────────────────────────────────────
# --epochs 0: the model is COCO-pretrained, so we evaluate + quantize the
# real baseline rather than training from scratch. Add --resume to continue
# from checkpoints if the job is requeued.
echo "== Launching NeuroQuant detection pipeline =="
neuroquant --config config_detection.yaml --epochs 0

echo "Job finished at $(date)"
echo "Results in: $PROJECT_DIR/artifacts_detection/"
