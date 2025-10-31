#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_ImageNet_AWS_ddp.py
Merged trainer:  single-GPU "good hygiene" + AWS speed (DDP/DALI).
- Switchable loader: --loader dali|albumentations (default: dali)
- SGD + OneCycleLR with global-batch LR scaling, label smoothing
- AMP + channels_last + cudnn.benchmark + matmul precision
- DDP with rank-0 logging/checkpointing
- Metrics: Top-1, Top-5, imgs/sec
- Logging: out/train_log.csv, out/logs.md
- Reports: reports/accuracy_curve.png, reports/loss_curve.png, reports/classification_report.txt,
           reports/confusion_matrix.csv, reports/model_summary.txt
- Checkpoints: checkpoints/last_epoch.pth, checkpoints/best_acc_epochXXX.pth
- Optional TensorBoard (--use-tb), optional classification report (--do-report)
- W&B: enable with --wandb (+ optional project/entity/tags/offline). Per-epoch logs, artifacts for ckpts & reports.
"""

import os, sys, time, math, argparse, random, csv, json
from pathlib import Path
from typing import Tuple, List
import numpy as np
import json, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from model import ResNet50
import matplotlib.pyplot as plt
from datetime import timedelta
# optional: pretty model table
try:
    from torchinfo import summary
except Exception:
    summary = None

# --- W&B: optional import (safe if not used) ---
try:
    import wandb  # only used when --wandb is set
except Exception:
    wandb = None

# cache hooks visible to helper fns
CACHED_MEAN = None
CACHED_STD  = None



# ---------------------- DDP helpers ----------------------
def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def is_main_process():
    return get_rank() == 0


def setup_ddp(backend="nccl", timeout_seconds=36000):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
        except Exception:
            dist.init_process_group(backend=backend)
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def cleanup_ddp():
    if is_dist():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

# ---------------------- Utility ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def append_csv(path: Path, row: List):
    first = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["epoch","phase","loss","top1","top5","lr","imgs_per_sec"])
        w.writerow(row)

def append_md(path: Path, line: str):
    with open(path, "a") as f:
        f.write(line + "\n")

# ---------------------- Data loaders ----------------------
def build_torchvision_datasets(args):
    mean = CACHED_MEAN if CACHED_MEAN is not None else [0.485, 0.456, 0.406]
    std  = CACHED_STD  if CACHED_STD  is not None else [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.crop_size * 1.14)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    root = Path(args.data)
    train_dir, val_dir = root / "train", root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expect {root}/train and {root}/val")
    train_ds = torchvision.datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = torchvision.datasets.ImageFolder(str(val_dir),   transform=val_tf)
    classes = train_ds.classes
    return train_ds, val_ds, classes,mean, std

def count_images(root: Path) -> int:
    exts = (".jpg",".jpeg",".png",".bmp")
    n = 0
    for p, _, files in os.walk(root):
        n += sum(1 for f in files if f.lower().endswith(exts))
    return n

def build_dali_iterators(args, device_id, world_size, rank):
    from dataset.imagenet_dali import dali_loader
    train_iter = dali_loader(str(Path(args.data)/"train"),
                            args.batch_size, args.workers,
                            device_id, world_size, rank,
                            train=True, mean=CACHED_MEAN, std=CACHED_STD)

    # For validation use torchvision loader for consistent metrics/reporting
    _, val_ds, _, _, _ = build_torchvision_datasets(args)
    return train_iter, val_ds

def build_albumentations_loaders(args, world_size, rank):
    from dataset.imagenet import make_loaders
    train_loader, val_loader, classes, mean, std = make_loaders(
        data_root=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.crop_size,
        sample_limit_for_stats=args.stats_samples,
        use_class_style_aug=args.use_class_style,
        distributed=(world_size > 1),
        seed=args.seed
    )
    return train_loader, val_loader, classes, mean, std

# ---------------------- Model / Metrics ----------------------
@torch.no_grad()
def accuracy_topk(output, target, topk=(1,5)):
    maxk = max(topk)
    B = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.item() * 100.0) / B)
    return res  # [top1, top5]

def all_reduce_sum(value: float, device):
    if get_world_size() == 1:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def progress_enumerator(loader, desc, args):
    """Return tqdm(enumerate(loader)) when --show-progress on (rank-0), else plain enumerate."""
    if is_main_process() and getattr(args, 'show_progress', False):
        try:
            from tqdm import tqdm
            return tqdm(enumerate(loader), total=len(loader), ncols=90, desc=desc)
        except Exception:
            pass
    return enumerate(loader)

# ---------------------- Training/Eval ----------------------
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, criterion, use_mixup, use_cutmix):
    model.train()
    running_loss = 0.0
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    imgs_sum = 0.0

    def mixup_data(x, y, alpha=0.2):
        if alpha <= 0: return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(crit, pred, y_a, y_b, lam):
        return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)

    def cutmix_data(x, y, alpha=1.0):
        if alpha <= 0: return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha)
        B, C, H, W = x.size()
        cx = np.random.randint(W); cy = np.random.randint(H)
        rw = int(W * np.sqrt(1 - lam)); rh = int(H * np.sqrt(1 - lam))
        x1 = max(cx - rw // 2, 0); x2 = min(cx + rw // 2, W)
        y1 = max(cy - rh // 2, 0); y2 = min(cy + rh // 2, H)
        index = torch.randperm(B, device=x.device)
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    for i, batch in progress_enumerator(loader, f"Train {epoch:03d}", args):
        # Support DALIGenericIterator vs PyTorch loader
        if isinstance(batch, list) or (isinstance(batch, tuple) and len(batch) == 1):
            data = batch[0]
            images = data["images"]
            targets = data["labels"].squeeze(-1).long()
        else:
            images, targets = batch
                    
        if args.channels_last:
            images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        # Optional MixUp/CutMix
        targets_a = targets_b = None
        lam = 1.0
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=0.2)
        elif use_cutmix:
            images, targets_a, targets_b, lam = cutmix_data(images, targets, alpha=1.0)

        optimizer.zero_grad(set_to_none=True)
        t_iter0 = time.time()
        #with torch.cuda.amp.autocast(enabled=args.amp):
        with torch.amp.autocast('cuda', enabled=args.amp):
            outputs = model(images)
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        args.scheduler.step()

        with torch.no_grad():
            t1, t5 = accuracy_topk(outputs, targets if not (use_mixup or use_cutmix) else targets_a, topk=(1,5))
        B = images.size(0)
        iter_time = max(time.time() - t_iter0, 1e-6)
        imgs_per_sec = (B * get_world_size()) / iter_time

        running_loss += loss.item() * B
        total += B
        top1_sum += t1 * B
        top5_sum += t5 * B

        # tqdm postfix (only when enabled)
        if is_main_process() and getattr(args, 'show_progress', False):
            try:
                _set = getattr(locals().get('loop', None), 'set_postfix_str', None)
                if callable(_set):
                    _set(f"loss={loss.item():.3f} top1={t1:.2f}")
            except Exception:
                pass
        imgs_sum += imgs_per_sec

        # tqdm postfix (only when enabled)
        if is_main_process() and getattr(args, 'show_progress', False):
            try:
                loop = locals().get('loop')  # if present
            except Exception:
                loop = None
            try:
                # best-effort: if current iterator is a tqdm instance
                _set = getattr(loop if 'loop' in locals() else None, 'set_postfix_str', None)
                if callable(_set):
                    _set(f"loss={loss.item():.3f} top1={t1:.2f}")
            except Exception:
                pass

    device = next(model.parameters()).device
    total_global = all_reduce_sum(float(total), device)
    loss_global  = all_reduce_sum(float(running_loss), device) / max(total_global, 1.0)
    top1_global  = all_reduce_sum(float(top1_sum), device) / max(total_global, 1.0)
    top5_global  = all_reduce_sum(float(top5_sum), device) / max(total_global, 1.0)

    if is_main_process():
        print(f"[Train] Epoch {epoch} | loss {loss_global:.4f} | top1 {top1_global:.2f}% | top5 {top5_global:.2f}% | imgs/s ~{imgs_sum/max(len(loader),1):.0f}")
    return {"loss": loss_global, "top1": top1_global, "top5": top5_global, "imgs_per_sec": imgs_sum/max(len(loader),1)}

@torch.no_grad()
def validate(model, loader, device, args, criterion, epoch):
    model.eval()
    running_loss = 0.0
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    loop = progress_enumerator(loader, f"Val {epoch:03d}", args)
    for i, batch in loop:
        images, targets = batch
        if args.channels_last:
            images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        #with torch.cuda.amp.autocast(enabled=args.amp):
        with torch.amp.autocast('cuda', enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        t1, t5 = accuracy_topk(outputs, targets, topk=(1,5))
        B = images.size(0)
        running_loss += loss.item() * B
        total += B
        top1_sum += t1 * B
        top5_sum += t5 * B

        # tqdm postfix (only when enabled)
        if is_main_process() and getattr(args, 'show_progress', False):
            try:
                _set = getattr(locals().get('loop', None), 'set_postfix_str', None)
                if callable(_set):
                    _set(f"loss={loss.item():.3f} top1={t1:.2f}")
            except Exception:
                pass

    device = next(model.parameters()).device
    total_global = all_reduce_sum(float(total), device)
    loss_global  = all_reduce_sum(float(running_loss), device) / max(total_global, 1.0)
    top1_global  = all_reduce_sum(float(top1_sum), device) / max(total_global, 1.0)
    top5_global  = all_reduce_sum(float(top5_sum), device) / max(total_global, 1.0)

    if is_main_process():
        print(f"[Val]   Epoch {epoch} | loss {loss_global:.4f} | top1 {top1_global:.2f}% | top5 {top5_global:.2f}%")
    return {"loss": loss_global, "top1": top1_global, "top5": top5_global}

# ---------------------- Args ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='Path that contains train/ and val/ folders')
    p.add_argument('--epochs', type=int, default=90)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 8))
    p.add_argument('--amp', action='store_true')
    p.add_argument('--channels-last', action='store_true')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--out-dir', type=str, default='./')
    p.add_argument('--num-classes', type=int, default=1000)
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--crop-size', type=int, default=224, help='Image crop size (default: 224)')
    p.add_argument('--loader', choices=['dali','albumentations'], default='dali', help='Data pipeline')
    p.add_argument('--stats-samples', type=int, default=50000, help='Samples to compute mean/std for albumentations')
    p.add_argument('--use-class-style', action='store_true', help='Use class-style aug (albumentations path only)')
    p.add_argument('--use-tb', action='store_true', help='Enable TensorBoard (rank-0 only)')
    p.add_argument('--stats-file', type=str, default=None,
               help='Path to cached mean/std JSON (from lr_finder). If unset, defaults are used.')

    # OneCycleLR knobs
    p.add_argument('--max-lr', type=float, default=None, help='If None use linear scaling: 0.1*(global_bsz/256)')
    p.add_argument('--pct-start', type=float, default=0.3)
    p.add_argument('--div-factor', type=float, default=25.0)
    p.add_argument('--final-div-factor', type=float, default=1e4)

    # Classification report / confusion matrix
    p.add_argument('--do-report', action='store_true', help='Generate classification_report and confusion_matrix at end')

    # --- W&B: CLI flags (all optional) ---
    p.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging (rank-0 only).')
    p.add_argument('--wandb-project', type=str, default='imagenet1k_runs', help='wandb project name.')
    p.add_argument('--wandb-entity', type=str, default=None, help='wandb entity/org (optional).')
    p.add_argument('--wandb-tags', type=str, default='', help='comma-separated list of tags (optional).')
    p.add_argument('--wandb-offline', action='store_true', help='WANDB_MODE=offline (sync later).')

        # Debug / progress control
    p.add_argument('--show-progress', action='store_true', help='Show per-batch tqdm progress during train/val (debug mode)')

    return p.parse_args()

# ---------------------- Main ----------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_dist() else 0
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    world_size = get_world_size()
    rank = get_rank()
    # console header like single-gpu script
    print(f"=> device: {'cuda' if torch.cuda.is_available() else 'cpu'} | AMP: {bool(args.amp)}")
    print(f"=> DDP: {is_dist()} | world_size: {world_size} | rank: {rank} | local_rank: {local_rank}")

    cached_mean, cached_std = None, None
    if args.stats_file and os.path.isfile(args.stats_file):
        import json
        with open(args.stats_file) as f:
            s = json.load(f)
            cached_mean, cached_std = s.get("mean"), s.get("std")
    global CACHED_MEAN, CACHED_STD
    CACHED_MEAN, CACHED_STD = cached_mean, cached_std

    # Interpret --out-dir as the RUN NAME (e.g., "g5x_1gpu_run")
    run_name   = Path(args.out_dir).name
    base       = Path(".")
    out_dir    = base / "out" / run_name
    ckpt_dir   = base / "checkpoints" / run_name
    reports_dir= base / "reports" / run_name
    runs_dir   = base / "runs" / run_name

    for d in (out_dir, ckpt_dir, reports_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "train_log.csv"
    md_path  = out_dir / "logs.md"

    if is_main_process():
        append_md(md_path, "# Training Log")
        append_md(md_path, "| epoch | phase | loss | top1 | top5 | lr | imgs/s |")
        append_md(md_path, "|---:|---|---:|---:|---:|---:|---:|")

    # --- W&B: init (rank-0 only) ---
    wb_run = None
    if args.wandb and is_main_process() and wandb is not None:
        if args.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'
        _tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
        try:
            wb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                tags=_tags or None,
                config=vars(args),
                save_code=True
            )
        except Exception as e:
            print(f"[W&B] init failed: {e}")
            wb_run = None

    # ---------- Data ----------
    if args.loader == "dali":
        train_iter, val_ds = build_dali_iterators(args, device_id=local_rank, world_size=world_size, rank=rank)
        n_train = count_images(Path(args.data)/"train")
        steps_per_epoch = math.ceil((n_train / max(world_size,1)) / args.batch_size)
        if val_ds is None:
            _, val_ds, _, _, _ = build_torchvision_datasets(args)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_dist() else None
        val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size or args.batch_size, shuffle=False, sampler=val_sampler,
                                num_workers=max(2, args.workers//2), pin_memory=True)
        train_loader = train_iter
        num_classes = len(getattr(val_ds, "classes", list(range(args.num_classes))))
        mean = CACHED_MEAN if CACHED_MEAN is not None else [0.485, 0.456, 0.406]
        std  = CACHED_STD  if CACHED_STD  is not None else [0.229, 0.224, 0.225]
    else:
        train_loader, val_loader, classes, mean, std = build_albumentations_loaders(args, world_size, rank)
        steps_per_epoch = len(train_loader)
        num_classes = len(classes)
    if is_main_process():
        print(f"=> classes: {num_classes} | mean={mean} | std={std}")

    # ---------- Model ----------
    model = ResNet50(num_classes=num_classes, pretrained=args.pretrained)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ---------- Optimizer / Scheduler / Loss ----------
    global_bsz = args.batch_size * max(1, world_size)
    base_bsz = 256
    scaled_max_lr = args.max_lr if args.max_lr is not None else 0.1 * (global_bsz / base_bsz)
    if is_main_process():
        print(f"=> using max_lr: {scaled_max_lr:.6f} (arg --max-lr {'set' if args.max_lr is not None else 'auto-scale'})")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=scaled_max_lr / args.div_factor,
                          momentum=0.9, weight_decay=1e-4, nesterov=False)
    #scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=scaled_max_lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor
    )
    args.scheduler = scheduler

    # TensorBoard (optional)
    tb_writer = None
    if args.use_tb and is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(runs_dir))
        except Exception as e:
            print(f("[warn] TensorBoard disabled: {e}"))

    # ---------- Resume ----------
    start_epoch = 0
    best_top1 = 0.0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location='cpu')
        sd = ckpt.get("model", ckpt)
        if isinstance(model, DDP):
            model.module.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(sd, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and args.amp:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            try: scheduler.load_state_dict(ckpt["scheduler"])
            except Exception: pass
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_top1 = float(ckpt.get("best_top1", 0.0))
        if is_main_process():
            print(f"=> Resumed from {args.resume} @ epoch {start_epoch} (best_top1={best_top1:.2f})")

    # ---------- Initial sanity val ----------
    if val_loader is not None:
        if is_main_process():
            print("=> initial evaluation")
        init_val = validate(model, val_loader, device, args, criterion, epoch=-1)
        if is_main_process():
            lr_now = optimizer.param_groups[0]['lr']
            append_csv(csv_path, [-1,"val", f"{init_val['loss']:.6f}", f"{init_val['top1']:.4f}", f"{init_val['top5']:.4f}", f"{lr_now:.6f}", ""])
            append_md(md_path, f"| -1 | val | {init_val['loss']:.4f} | {init_val['top1']:.2f} | {init_val['top5']:.2f} | {lr_now:.5f} | |")
            # --- W&B: log initial val ---
            if wb_run is not None:
                try:
                    wandb.log({
                        'epoch': -1,
                        'val/loss': init_val['loss'],
                        'val/top1': init_val['top1'],
                        'val/top5': init_val['top5'],
                        'lr': lr_now
                    }, step=-1)
                except Exception as e:
                    print(f"[W&B] initial log failed: {e}")

    # ---------- Train ----------
    use_mixup  = True   # <-- comment to disable MixUp
    use_cutmix = False  # <-- set True to try CutMix (don't use both)

    for epoch in range(start_epoch, args.epochs):
        if isinstance(train_loader, DataLoader):
            sampler = train_loader.sampler if hasattr(train_loader, "sampler") else None
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args, criterion, use_mixup, use_cutmix)
        val_stats = validate(model, val_loader, device, args, criterion, epoch) if val_loader is not None else {"loss":0.0,"top1":0.0,"top5":0.0}

        if is_main_process():
            lr_now = optimizer.param_groups[0]['lr']
            append_csv(csv_path, [epoch,"train", f"{train_stats['loss']:.6f}", f"{train_stats['top1']:.4f}", f"{train_stats['top5']:.4f}", f"{lr_now:.6f}", f"{train_stats['imgs_per_sec']:.0f}"])
            append_csv(csv_path, [epoch,"val",   f"{val_stats['loss']:.6f}",   f"{val_stats['top1']:.4f}",   f"{val_stats['top5']:.4f}",   f"{lr_now:.6f}", ""])
            append_md(md_path, f"| {epoch} | train | {train_stats['loss']:.4f} | {train_stats['top1']:.2f} | {train_stats['top5']:.2f} | {lr_now:.5f} | {train_stats['imgs_per_sec']:.0f} |")
            append_md(md_path, f"| {epoch} | val | {val_stats['loss']:.4f} | {val_stats['top1']:.2f} | {val_stats['top5']:.2f} | {lr_now:.5f} | |")
            if tb_writer:
                tb_writer.add_scalar("train/loss", train_stats["loss"], epoch)
                tb_writer.add_scalar("train/top1", train_stats["top1"], epoch)
                tb_writer.add_scalar("train/top5", train_stats["top5"], epoch)
                tb_writer.add_scalar("val/loss", val_stats["loss"], epoch)
                tb_writer.add_scalar("val/top1", val_stats["top1"], epoch)
                tb_writer.add_scalar("val/top5", val_stats["top5"], epoch)

            # --- W&B: per-epoch logging ---
            if wb_run is not None:
                try:
                    wandb.log({
                        'epoch': epoch,
                        'lr': lr_now,
                        'train/loss': train_stats['loss'],
                        'train/top1': train_stats['top1'],
                        'train/top5': train_stats['top5'],
                        'train/imgs_per_sec': train_stats['imgs_per_sec'],
                        'val/loss': val_stats['loss'],
                        'val/top1': val_stats['top1'],
                        'val/top5': val_stats['top5']
                    }, step=epoch)
                except Exception as e:
                    print(f"[W&B] epoch log failed: {e}")

            state = {
                "epoch": epoch,
                "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if args.amp else {},
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "best_top1": float(best_top1)
            }
            torch.save(state, ckpt_dir / "last_epoch.pth")
            is_best = val_stats["top1"] > best_top1
            if is_best:
                best_top1 = val_stats["top1"]
                torch.save(state, ckpt_dir / f"best_acc_epoch{epoch:03d}.pth")

                # --- W&B: upload best checkpoint as artifact (optional) ---
                if wb_run is not None:
                    try:
                        art = wandb.Artifact(f'{run_name}-ckpts', type='model')
                        art.add_file(str(ckpt_dir / f"best_acc_epoch{epoch:03d}.pth"))
                        wandb.log_artifact(art)
                    except Exception as e:
                        print(f"[W&B] artifact upload skipped: {e}")

    if is_main_process():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            plt.figure()
            df_train = df[df.phase=="train"]; df_val = df[df.phase=="val"]
            plt.plot(df_train.epoch, df_train.top1, label="train_top1")
            plt.plot(df_val.epoch, df_val.top1, label="val_top1")
            plt.xlabel("epoch"); plt.ylabel("Top-1 (%)"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(reports_dir / "accuracy_curve.png", bbox_inches="tight"); plt.close()

            plt.figure()
            plt.plot(df_train.epoch, df_train.loss, label="train_loss")
            plt.plot(df_val.epoch, df_val.loss, label="val_loss")
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(reports_dir / "loss_curve.png", bbox_inches="tight"); plt.close()
        except Exception as e:
            print(f"[warn] plotting failed: {e}")

        # ---- Model summary (console + file) ----
        try:
            if summary is None:
                print("[warn] torchinfo not installed; run: pip install \"torchinfo>=1.8.0\"")
            else:
                bs = max(1, args.batch_size)
                chw = (3, args.crop_size, args.crop_size)
                mdl = model.module if isinstance(model, DDP) else model
                sm = summary(
                    mdl,
                    input_size=(bs, *chw),
                    verbose=1,
                    col_names=("input_size","output_size","num_params","kernel_size"),
                    depth=5
                )
                with open(reports_dir / "model_summary.txt","w") as f:
                    f.write(str(sm))
                print(f"=> model summary saved to {reports_dir}/model_summary.txt")
        except Exception as e:
            print(f"[warn] model summary failed: {e}")

        # ===== BEGIN: Human-readable reporting for ImageNet =====
        if getattr(args, "do_report", False) and 'val_loader' in locals() and val_loader is not None:
            try:
                

                # (1) collect predictions / targets
                y_true_local, y_pred_local = [], []
                with torch.no_grad():
                    mdl = model.module if isinstance(model, DDP) else model
                    mdl.eval()
                    for images, targets in val_loader:
                        images = images.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        logits = mdl(images)
                        pred = torch.argmax(logits, dim=1)
                        y_true_local.extend(targets.cpu().tolist())
                        y_pred_local.extend(pred.cpu().tolist())

                # (2) gather across ranks if DDP
                def _gather_all(x):
                    if dist.is_available() and dist.is_initialized():
                        t = torch.tensor(x, dtype=torch.int64, device=device)
                        world = dist.get_world_size()
                        sizes = [torch.tensor(0, device=device) for _ in range(world)]
                        dist.all_gather(sizes, torch.tensor([t.numel()], device=device))
                        maxlen = int(max(s.item() for s in sizes))
                        if t.numel() < maxlen:
                            pad = torch.full((maxlen - t.numel(),), -1, dtype=torch.int64, device=device)
                            t = torch.cat([t, pad], dim=0)
                        outs = [torch.empty_like(t) for _ in range(world)]
                        dist.all_gather(outs, t)
                        res = []
                        for i, out in enumerate(outs):
                            valid = int(sizes[i].item())
                            res.extend(out[:valid].cpu().tolist())
                        return res
                    return x

                y_true = _gather_all(y_true_local)
                y_pred = _gather_all(y_pred_local)

                # (3) only rank-0 writes
                is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                if is_main:
                    # (4) load mapping
                    synset2name = {}
                    try:
                        with open("utils/imagenet_class_index.json") as f:
                            idx2pair = json.load(f)
                        for _, (syn, human) in idx2pair.items():
                            synset2name[syn] = human
                    except Exception as e:
                        print(f"[warn] could not load utils/imagenet_class_index.json: {e}")

                    # (5) convert class folders → readable names
                    classes = getattr(val_loader.dataset, "classes",
                                      [str(i) for i in range(getattr(args, "num_classes", 1000))])
                    target_names = [synset2name.get(s, s) for s in classes]

                    # (6) save confusion matrix + report
                    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
                    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)

                    reports_dir.mkdir(parents=True, exist_ok=True)
                    df_cm.to_csv(reports_dir / "confusion_matrix.csv", index=True)
                    with open(reports_dir / "classification_report.txt", "w") as f:
                        f.write(classification_report(y_true, y_pred,
                                                      target_names=target_names,
                                                      zero_division=0))

                    print(f"[report] saved readable reports to: {reports_dir}")

            except Exception as e:
                print(f"[warn] report failed: {e}")
        # ===== END: Human-readable reporting for ImageNet =====

        # if args.do_report and 'val_loader' in locals() and val_loader is not None:
        #     try:
        #         from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
        #         y_true, y_pred = [], []
        #         with torch.no_grad():
        #             mdl = model.module if isinstance(model, DDP) else model
        #             mdl.eval()
        #             for images, targets in val_loader:
        #                 images = images.to(device, non_blocking=True)
        #                 targets = targets.to(device, non_blocking=True)
        #                 logits = mdl(images)
        #                 pred = torch.argmax(logits, dim=1)
        #                 y_true.extend(targets.cpu().numpy().tolist())
        #                 y_pred.extend(pred.cpu().numpy().tolist())
        #         import pandas as pd
        #         cm = confusion_matrix(y_true, y_pred)
        #         pd.DataFrame(cm).to_csv(reports_dir / "confusion_matrix.csv", index=False)
        #         classes = getattr(val_loader.dataset, "classes", [str(i) for i in range(args.num_classes)])
        #         with open(reports_dir / "classification_report.txt","w") as f:
        #             f.write(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

        #         # --- W&B: macro P/R/F1 per-epoch final snapshot (logged once here as summary) ---
        #         if wb_run is not None:
        #             try:
        #                 prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        #                 rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
        #                 f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
        #                 wandb.summary['val/macro_precision'] = prec
        #                 wandb.summary['val/macro_recall'] = rec
        #                 wandb.summary['val/macro_f1'] = f1
        #                 # also save files for convenience
        #                 wandb.save(str(reports_dir / "confusion_matrix.csv"))
        #                 wandb.save(str(reports_dir / "classification_report.txt"))
        #             except Exception as e:
        #                 print(f"[W&B] macro metrics upload skipped: {e}")
        #     except Exception as e:
        #         print(f"[warn] report failed: {e}")

        # --- W&B: upload CSV + reports as artifact at the end (optional) ---
        if wb_run is not None:
            try:
                art = wandb.Artifact(f'{run_name}-reports', type='report')
                csv_p = str(csv_path)
                if os.path.isfile(csv_p):
                    art.add_file(csv_p)
                if os.path.isdir(reports_dir):
                    art.add_dir(str(reports_dir))
                wandb.log_artifact(art)
            except Exception as e:
                print(f"[W&B] report artifact upload skipped: {e}")
            finally:
                try:
                    wb_run.finish()
                except Exception:
                    pass

    cleanup_ddp()
    if tb_writer:
        try:
            tb_writer.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
