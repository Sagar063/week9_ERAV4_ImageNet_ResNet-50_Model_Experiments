#!/usr/bin/env bash
set -euo pipefail
source /opt/dlami/nvme/envs/train312/bin/activate
CUDA_VISIBLE_DEVICES=0 python -u train_full_ImageNet_AWS.py \
  --data ${1:-/mnt/imagenette} \
  --epochs ${2:-20} --batch-size ${3:-256} \
  --amp --channels-last --workers ${4:-8} \
  --lr ${5:-0.1} --weight-decay ${6:-0.05} \
  --crop-size 224 \
  --log-dir ${7:-./logs/g5x_1gpu_run}
