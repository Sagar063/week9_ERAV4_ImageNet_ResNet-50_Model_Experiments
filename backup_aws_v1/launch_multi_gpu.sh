#!/usr/bin/env bash
set -euo pipefail
source /opt/dlami/nvme/envs/train312/bin/activate
NGPUS=${1:-4}
DATA=${2:-/mnt/imagenet}
torchrun --nproc_per_node=$NGPUS --standalone train_full_ImageNet_AWS.py \
  --data $DATA \
  --epochs ${3:-90} --batch-size ${4:-512} \
  --amp --channels-last --workers ${5:-8} \
  --lr ${6:-0.2} --weight-decay ${7:-0.05} \
  --crop-size 224 \
  --log-dir ${8:-./logs/g5_multi}
