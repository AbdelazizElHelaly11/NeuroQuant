#!/bin/bash
# ===========================================================================
# NeuroQuant — Detection experiment SETUP (run on the LOGIN NODE only)
# ===========================================================================
# The MOHESR cluster blocks `pip install` and downloads inside batch jobs,
# so everything that needs the network happens HERE, once, before sbatch.
#
# Usage (login node):
#   cd ~/NeuroQuant
#   bash scripts/hpc_setup_detection.sh
#
# Safe to re-run: conda env creation and the COCO download both skip work
# that is already present.
# ===========================================================================
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/neuroquant}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/NeuroQuant}"
COCO_ROOT="${COCO_ROOT:-$PROJECT_DIR/data/coco}"
COCO_MAX_IMAGES="${COCO_MAX_IMAGES:-800}"

echo "== NeuroQuant detection setup =="
echo "  env:        $ENV_DIR"
echo "  project:    $PROJECT_DIR"
echo "  coco root:  $COCO_ROOT  (max_images=$COCO_MAX_IMAGES)"

# ── 1. Conda ───────────────────────────────────────────────────────────────
source /nfs/slurm/conda/etc/profile.d/conda.sh

if [ ! -d "$ENV_DIR" ]; then
  echo "== Creating conda env (python 3.10) =="
  conda create -p "$ENV_DIR" python=3.10 -y
fi
conda activate "$ENV_DIR"
echo "  python: $(which python)"

# ── 2. Dependencies (login node has network; batch jobs do NOT) ────────────
cd "$PROJECT_DIR"
echo "== Installing PyTorch (CUDA 12.1) =="
pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121

echo "== Installing NeuroQuant + requirements =="
pip install -r requirements.txt
pip install -e .

# ── 3. COCO val2017 subset (dependency-free adapter, ~hundreds of MB) ──────
echo "== Downloading COCO val2017 subset =="
python -m neuroquant.data.coco_detection --download --root "$COCO_ROOT" --max-images "$COCO_MAX_IMAGES"

# ── 4. Pre-cache pretrained COCO weights (batch jobs have NO network) ──────
# `weights="DEFAULT"` downloads to ~/.cache/torch/hub/checkpoints on first
# use. That download is BLOCKED inside a compute job, so we trigger it here
# on the login node — the cache is on NFS, so the job sees it.
echo "== Pre-caching COCO-pretrained detector weights =="
COCO_ROOT="$COCO_ROOT" python - <<'PY'
import os, torch
from torchvision.models import detection as det
m = det.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
n = sum(p.numel() for p in m.parameters())
print(f"Cached fasterrcnn_mobilenet_v3_large_320_fpn weights ({n:,} params).")

# ── 5. Verify the COCO adapter on the downloaded subset ──
from neuroquant.data.coco_detection import CocoDetectionDataset
root = os.environ.get("COCO_ROOT", os.path.expanduser("~/NeuroQuant/data/coco"))
ds_tr = CocoDetectionDataset(root=root, split="train")
ds_te = CocoDetectionDataset(root=root, split="test")
import torch as _t
overlap = set(ds_tr._ids) & set(ds_te._ids)
assert not overlap, f"DATA LEAK: {len(overlap)} ids shared between train and test!"
img, tgt = ds_te[0]
print(f"COCO OK: train={len(ds_tr)} test={len(ds_te)} (disjoint) | "
      f"img{tuple(img.shape)} range[{img.min():.2f},{img.max():.2f}] | "
      f"{tgt['boxes'].shape[0]} boxes, labels={tgt['labels'][:5].tolist()}")
PY

echo ""
echo "== Setup complete. Submit the job with:  sbatch scripts/run_detection_hpc.sh =="
