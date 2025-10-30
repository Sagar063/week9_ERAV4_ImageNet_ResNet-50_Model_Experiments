# When i verified with imagenet
# #!/usr/bin/env bash
# set -euo pipefail
# source ~/venv/bin/activate
# NGPUS=${1:-4}
# DATA=${2:-/mnt/imagenet_full/imagenet}
# torchrun --nproc_per_node=$NGPUS --standalone train_full_ImageNet_AWS.py \
#   --data $DATA \
#   --epochs ${3:-90} --batch-size ${4:-512} \
#   --amp --channels-last --workers ${5:-8} \
#   --lr ${6:-0.2} --weight-decay ${7:-0.05} \
#   --log-dir ${8:-./logs/g5_multi}


#!/usr/bin/env bash
# ============================================================
# Multi-GPU (DDP) launch script for AWS g5.12xlarge etc.
# Usage:
#   bash scripts/launch_multi_gpu.sh [NUM_GPUS] [DATA_PATH] [EPOCHS] [BATCH]
# Example:
#   bash scripts/launch_multi_gpu.sh 4 /mnt/imagenet 90 256
# ============================================================

set -euo pipefail
source /opt/dlami/nvme/envs/train312/bin/activate

NGPUS=${1:-4}
DATA=${2:-/mnt/imagenet}
EPOCHS=${3:-90}
BATCH=${4:-256}
WORKERS=${4:-8}
EXTRA_ARGS=( "${@:5}" )

# ------------------------------------------------------------
# ✅ Default: DALI + DDP (fastest configuration)
# ------------------------------------------------------------
# torchrun --nproc_per_node=$NGPUS --standalone train_full_ImageNet_AWS.py \
#   --data "$DATA" \
#   --out-dir logs/g5_multi_dali \
#   --epochs "$EPOCHS" --batch-size "$BATCH" --eval-batch-size "$BATCH" \
#   --amp --channels-last --do-report --pretrained \
#   --workers "$WORKERS" \
#   "${EXTRA_ARGS[@]}"

torchrun --nproc_per_node=$NGPUS --standalone train_full_ImageNet_AWS.py \
  --data "$DATA" \
  --out-dir logs/g5_multi_dali \
  --epochs "$EPOCHS" --batch-size "$BATCH" --eval-batch-size "$BATCH" \
  --amp --channels-last --do-report  \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}"

# ------------------------------------------------------------
# 🧪 Alternative: Albumentations loader (to test augmentations)
# Uncomment below block if you want Albumentations + DDP path.
# ------------------------------------------------------------
## with pretrained model of resnet50
# torchrun --nproc_per_node=$NGPUS --standalone train_full_ImageNet_AWS.py \
#   --data "$DATA" \
#   --out-dir logs/g5_multi_albu \
#   --loader albumentations --use-class-style \
#   --epochs "$EPOCHS" --batch-size "$BATCH" --eval-batch-size "$BATCH" \
#   --amp --channels-last --do-report --pretrained \
#   --workers "$WORKERS" \
#   "${EXTRA_ARGS[@]}"

## without pretrained model of resnet50
# torchrun --nproc_per_node=$NGPUS --standalone train_full_ImageNet_AWS.py \
#   --data "$DATA" \
#   --out-dir logs/g5_multi_albu \
#   --loader albumentations --use-class-style \
#   --epochs "$EPOCHS" --batch-size "$BATCH" --eval-batch-size "$BATCH" \
#   --amp --channels-last --do-report \
#   --workers "$WORKERS" \
#   "${EXTRA_ARGS[@]}"

