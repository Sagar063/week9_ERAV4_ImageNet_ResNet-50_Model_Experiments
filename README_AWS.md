
# AWS Starter Kit for ImageNet / ResNet-50 (AMP + DDP)

## 1) Mount your EBS dataset
Assuming:
- ImageNet or Imagenette mounted at `/mnt/imagenet_full` or `/mnt/imagenette`
- Inside it: `train/` and `val/` class subfolders.

If your ImageNet **val** is flat, restructure it:
```bash
python scripts/fix_imagenet_val.py --val-dir /mnt/imagenet_full/imagenet/val --val-map /path/to/val_map.txt
```

## 2) Create env
```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements_aws.txt
```

## 3) Single-GPU (g5.xlarge)
```bash
source ~/venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -u train_full_ImageNet_AWS.py \
  --data /mnt/imagenette \
  --epochs 50 --batch-size 256 \
  --amp --channels-last --workers 8 \
  --lr 0.1 --weight-decay 0.05 \
  --log-dir ./logs/g5x_1gpu_imagenette
```

## 4) Multi-GPU (later)
```bash
source ~/venv/bin/activate
torchrun --nproc_per_node=4 --standalone train_full_ImageNet_AWS.py \
  --data /mnt/imagenet_full/imagenet \
  --epochs 90 --batch-size 512 \
  --amp --channels-last --workers 8 \
  --lr 0.2 --weight-decay 0.05 \
  --log-dir ./logs/g5_multi
```

Tips:
- `--eval-on-single-rank` to reduce validation duplication in DDP.
- Use `persistent_workers`, `pin_memory`, `prefetch_factor=4` for speed.
- Keep instance **Stopped** when idle to save costs.
