# verified with small iamegnette
# #!/usr/bin/env bash
# set -euo pipefail
# source ~/venv/bin/activate
# CUDA_VISIBLE_DEVICES=0 python -u train_full_ImageNet_AWS.py \
#   --data ${1:-/mnt/imagenette} \
#   --epochs ${2:-50} --batch-size ${3:-256} \
#   --amp --channels-last --workers ${4:-8} \
#   --lr ${5:-0.1} --weight-decay ${6:-0.05} \
#   --log-dir ${7:-./logs/g5x_1gpu_run}


#!/usr/bin/env bash
# ============================================================
# Single-GPU launch script for ImageNet / Imagenette training
# Usage:
#   bash scripts/launch_single_gpu.sh [DATA_PATH] [EPOCHS] [BATCH] [WORKERS]
# Examples:
#   🔹 Full ImageNet run (no progress bars)
#      bash scripts/launch_single_gpu.sh /mnt/imagenet1k 90 256 8 \
#        --max-lr 0.0156 --stats-file data_stats/imagenet1k_aws_stats.json \
#        --out-dir imagenet1k_g5x_1gpu_dali_full --wandb --wandb-project imagenet1k_runs
#
#   🔹 Debug run on small subset *with tqdm progress bars*
#      bash scripts/launch_single_gpu.sh /mnt/debug_data 1 32 4 \
#        --show-progress --out-dir debug_7class_tqdm_test
#
# Tip:
#   --show-progress  → enables tqdm per-batch progress bars (use only for short/debug runs)
#   (no flag)        → default silent mode (faster, cleaner logs)
# ============================================================

set -euo pipefail
# Activate your venv (adjust path if different)
source /opt/dlami/nvme/envs/imagenet1k_venv/bin/activate

# Default values
DATA=${1:-/mnt/imagenet1k}
EPOCHS=${2:-90}
BATCH=${3:-256}
WORKERS=${4:-8}
EXTRA_ARGS=( "${@:5}" )

# ------------------------------------------------------------
# ✅ Run with DALI loader (default fast path)
# ------------------------------------------------------------
## Wih pretrained mdoel of resnet50
# CUDA_VISIBLE_DEVICES=0 python3 -u train_full_ImageNet_AWS.py \
#   --data "$DATA" \
#   --out-dir imagenet1k_g5x_1gpu_dali \
#   --epochs "$EPOCHS" --batch-size "$BATCH" --eval-batch-size "$BATCH" \
#   --amp --channels-last --do-report --pretrained \
#   --workers "$WORKERS"\
#   "${EXTRA_ARGS[@]}"

## without pretrained model of resnet50
CUDA_VISIBLE_DEVICES=0 python3 -u train_full_ImageNet_AWS.py \
  --data "$DATA" \
  --out-dir imagenet1k_g5x_1gpu_dali \
  --epochs "$EPOCHS" --batch-size "$BATCH" --eval-batch-size "$BATCH" \
  --amp --channels-last --do-report \
  --workers "$WORKERS"\
  "${EXTRA_ARGS[@]}"
# ------------------------------------------------------------
# 🧪 Run with Albumentations loader (to test augmentations)
# Uncomment below block if you want Albumentations path instead.
# ------------------------------------------------------------
## with pretrained model of resnet50
# CUDA_VISIBLE_DEVICES=0 python3 -u train_full_ImageNet_AWS.py \
#   --data "$DATA" \
#   --out-dir logs/imagenet1k_g5x_1gpu_run_albu \
#   --loader albumentations --use-class-style \
#   --epochs "$EPOCHS" --batch-size "$BATCH" --eval-batch-size "$BATCH" \
#   --amp --channels-last --do-report --pretrained \
#   --workers "$WORKERS" \
#   "${EXTRA_ARGS[@]}"

## without pretrained model of resnet50
# CUDA_VISIBLE_DEVICES=0 python3 -u train_full_ImageNet_AWS.py \
#   --data "$DATA" \
#   --out-dir logs/imagenet1k_g5x_1gpu_run_albu \
#   --loader albumentations --use-class-style \
#   --epochs "$EPOCHS" --batch-size "$BATCH" --eval-batch-size "$BATCH" \
#   --amp --channels-last --do-report \
#   --workers "$WORKERS" \
#  "${EXTRA_ARGS[@]}"
