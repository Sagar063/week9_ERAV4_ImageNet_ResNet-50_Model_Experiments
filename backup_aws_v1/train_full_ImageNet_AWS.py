#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_ImageNet_AWS.py
- Single-GPU or Multi-GPU (DDP via torchrun)
- AMP + channels_last
- Fast DataLoader defaults
- Resume checkpoints
- Rank-0 logging only
- Compatible with standard ImageNet/Imagenette folder layout
- Lightweight CSV logging to --log-dir/train_log.csv
"""

import os, time, argparse, random
from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision import transforms

from utils.dist_utils import (
    setup_ddp, cleanup_ddp, is_dist, get_rank, get_world_size, is_main_process
)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # performance-friendly defaults
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

def create_model(num_classes=1000, pretrained=False):
    # ResNet-50 (torchvision)
    m = torchvision.models.resnet50(
        weights=None if not pretrained else torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )
    if num_classes != 1000:
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def build_transforms(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.crop_size * 1.14)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf

def build_datasets(args):
    data_root = Path(args.data)
    # Expect standard structure: data/train/<class> , data/val/<class>
    train_dir = data_root / "train"
    val_dir   = data_root / "val"
    train_tf, val_tf = build_transforms(args)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Val folder not found: {val_dir}\n"
            "If your ImageNet val is flat, run scripts/fix_imagenet_val.py first."
        )

    train_ds = torchvision.datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = torchvision.datasets.ImageFolder(str(val_dir),   transform=val_tf)
    return train_ds, val_ds

def save_checkpoint(state, out_dir, is_best=False):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / "checkpoint.pth")
    if is_best:
        torch.save(state, out / "best.pth")

def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, criterion):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    total = 0
    correct = 0
    for i, (images, targets) in enumerate(loader):
        if args.channels_last:
            images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    if is_main_process():
        print(f"[Train] Epoch {epoch} | loss {epoch_loss:.4f} | acc {epoch_acc:.2f}% | time {time.time()-t0:.1f}s")
    return {"loss": epoch_loss, "acc": epoch_acc}

@torch.no_grad()
def validate(model, loader, device, args, criterion):
    model.eval()
    t0 = time.time()
    running_loss = 0.0
    total = 0
    correct = 0
    for images, targets in loader:
        if args.channels_last:
            images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    if is_main_process():
        print(f"[Val]   loss {epoch_loss:.4f} | acc {epoch_acc:.2f}% | time {time.time()-t0:.1f}s")
    return {"loss": epoch_loss, "acc": epoch_acc}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='Path that contains train/ and val/ folders')
    p.add_argument('--epochs', type=int, default=90)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 8))
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--channels-last', action='store_true')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--log-dir', type=str, default='./logs')
    p.add_argument('--num-classes', type=int, default=1000)
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-on-single-rank', action='store_true')
    p.add_argument('--crop-size', type=int, default=224, help='Image crop size (default: 224)')
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    setup_ddp()  # initializes process group if torchrun env vars are set
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_dist() else 0
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    # --- CSV logging setup (rank-0 only) ---
    if is_main_process():
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "train_log.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "phase", "loss", "acc"])
    else:
        log_dir = None
        csv_path = None
    # ---------------------------------------

    train_ds, val_ds = build_datasets(args)
    train_sampler = DistributedSampler(train_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True) if is_dist() else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=get_world_size(), rank=get_rank(), shuffle=False) if (is_dist() and not args.eval_on_single_rank) else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=max(2, args.workers//2), pin_memory=True
    )

    model = create_model(num_classes=args.num_classes, pretrained=args.pretrained)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    # Wrap with DDP if multi-GPU
    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location='cpu')
        def maybe_sd(obj, sd): 
            try: obj.load_state_dict(sd, strict=True)
            except Exception: pass
        if 'model' in ckpt:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                maybe_sd(model.module, ckpt['model'])
            else:
                maybe_sd(model, ckpt['model'])
        if 'opt' in ckpt: optimizer.load_state_dict(ckpt['opt'])
        if 'scaler' in ckpt and args.amp: scaler.load_state_dict(ckpt['scaler'])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        if is_main_process():
            print(f"=> Resumed from {args.resume} @ epoch {start_epoch}")

    best_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None: train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args, criterion)
        val_stats = validate(model, val_loader, device, args, criterion) if ((not args.eval_on_single_rank) or is_main_process()) else {"acc": 0.0, "loss": 0.0}

        # --- append CSV (rank-0 only) ---
        if is_main_process():
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, "train", f"{train_stats['loss']:.6f}", f"{train_stats['acc']:.4f}"])
                w.writerow([epoch, "val",   f"{val_stats['loss']:.6f}",   f"{val_stats['acc']:.4f}"])
        # ---------------------------------

        if is_main_process():
            is_best = val_stats.get("acc", 0.0) > best_acc
            best_acc = max(best_acc, val_stats.get("acc", 0.0))
            save_checkpoint({
                "epoch": epoch,
                "model": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict() if args.amp else {},
                "args": vars(args),
                "best_acc": best_acc
            }, args.log_dir, is_best=is_best)

    cleanup_ddp()

if __name__ == "__main__":
    main()
