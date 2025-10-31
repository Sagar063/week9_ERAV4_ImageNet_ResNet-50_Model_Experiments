#!/usr/bin/env bash
# ============================================================
# LR Finder launcher (single GPU, Albumentations loader)
# Usage:
#   bash scripts/launch_lr_finder.sh [DATA_PATH] [BATCH] [ITER] [OUTDIR]
# Example:
#   bash scripts/launch_lr_finder.sh /mnt/imagenet 256 150 lr_finder_plots_imagenet1k
# ============================================================

set -euo pipefail
source /opt/dlami/nvme/envs/imagenet1k_venv/bin/activate

DATA=${1:-/mnt/imagenet1k}
BATCH=${2:-256}
ITERS=${3:-150}
OUTDIR=${4:-lr_finder_plots_imagenet1k}

python3 lr_finder.py find_lr \
  --data_root "$DATA" \
  --batch_size "$BATCH" --workers 8 \
  --start_lr 1e-7 --end_lr 1.0 \
  --num_iter "$ITERS" --img_size 224 \
  --use_class_style_aug=False \
  --output_dir "$OUTDIR"

echo "------------------------------------------------------------"
if [[ -f "$OUTDIR/lr_suggestion.txt" ]]; then
  SUG=$(cat "$OUTDIR/lr_suggestion.txt")
  echo "Suggested LR: $SUG"
else
  echo "No suggestion file found."
fi
