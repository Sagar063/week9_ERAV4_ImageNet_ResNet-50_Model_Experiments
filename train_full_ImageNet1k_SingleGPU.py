#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix

#from dataset.imagenet_mini import make_loaders
from dataset.imagenet import make_loaders
from tqdm.auto import tqdm
import sys
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------
# Utilities
# ----------------------------

def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracies."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)  # [B, maxk]
        pred = pred.t()                              # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def _append_logs(md_path: Path, csv_path: Path, epoch: int, phase: str,
                 loss: float, top1: float, top5: float, lr: float, ips: float):
    """Append one line to markdown + CSV logs (CIFAR-style)."""
    line = (f"[{phase.capitalize()}] Epoch {epoch:03d} | loss {loss:.4f} | "
            f"top1 {top1:.2f}% | top5 {top5:.2f}% | lr {lr:.6f} | ips {ips:.1f}\n")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(line)
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{phase},{loss:.6f},{top1:.4f},{top5:.4f},{lr:.8f},{ips:.2f}\n")


# ----------------------------
# Training / Evaluation
# ----------------------------

def train_one_epoch(epoch, model, optimizer, scheduler, scaler, criterion,
                    loader, device, amp_enabled, writer):
    model.train()
    total = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    start = time.time()
    bar = tqdm(loader, desc=f"Train {epoch+1}", leave=False,dynamic_ncols=True, disable=not sys.stdout.isatty())
    for step, (images, targets) in enumerate(bar):
    #for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # per-batch step for OneCycleLR

        bs = images.size(0)
        total += bs
        loss_sum += loss.item() * bs
        top1, top5 = accuracy(logits, targets, topk=(1, min(5, logits.size(1))))
        top1_sum += top1.item() * bs / 100.0
        top5_sum += top5.item() * bs / 100.0

        # Update progress bar every ~10 steps
        if step % 10 == 0:
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                top1=f"{(top1.item()):.2f}%",
                lr=f"{optimizer.param_groups[0]['lr']:.5f}"
            )

        # Step-level logging
        if writer is not None and step % 50 == 0:
            global_step = epoch * len(loader) + step
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/loss_step", loss.item(), global_step)

    dur = time.time() - start
    avg_loss = loss_sum / max(1, total)
    top1_epoch = 100.0 * (top1_sum / max(1, total))
    top5_epoch = 100.0 * (top5_sum / max(1, total))
    ips = total / max(1e-9, dur)

    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/top1", top1_epoch, epoch)
        writer.add_scalar("train/top5", top5_epoch, epoch)
        writer.add_scalar("train/imgs_per_sec", ips, epoch)

    print(f"[Train] {epoch:03d} | loss {avg_loss:.4f} | top1 {top1_epoch:.2f}% | "
          f"top5 {top5_epoch:.2f}% | {ips:.1f} img/s")

    return {"loss": avg_loss, "top1": top1_epoch, "top5": top5_epoch, "ips": ips}


@torch.no_grad()
def evaluate(epoch, model, criterion, loader, device, amp_enabled, writer):
    model.eval()
    total = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    start = time.time()
    #for images, targets in loader:
    bar = tqdm(loader, desc=f"Val {epoch+1}", leave=False,dynamic_ncols=True, disable=not sys.stdout.isatty())
    for images, targets in bar:

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        bs = images.size(0)
        total += bs
        loss_sum += loss.item() * bs
        top1, top5 = accuracy(logits, targets, topk=(1, min(5, logits.size(1))))
        top1_sum += top1.item() * bs / 100.0
        top5_sum += top5.item() * bs / 100.0

        # Show running averages on the bar
        seen = max(1, total)
        if bar.n % 10 == 0:
            bar.set_postfix(
                loss=f"{(loss_sum/seen):.4f}",
                top1=f"{(100.0*top1_sum/seen):.2f}%",
                top5=f"{(100.0*top5_sum/seen):.2f}%"
            )


    dur = time.time() - start
    val_loss = loss_sum / max(1, total)
    top1 = 100.0 * (top1_sum / max(1, total))
    top5 = 100.0 * (top5_sum / max(1, total))
    ips = total / max(1e-9, dur)


    if writer:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/top1", top1, epoch)
        writer.add_scalar("val/top5", top5, epoch)
        writer.add_scalar("val/imgs_per_sec", ips, epoch)

    print(f"[Val]   {epoch:03d} | loss {val_loss:.4f} | top1 {top1:.2f}% | "
          f"top5 {top5:.2f}% | {ips:.1f} img/s")

    return {"loss": val_loss, "top1": top1, "top5": top5, "ips": ips}


# ----------------------------
# Reports (classification report + confusion matrix)
# ----------------------------

@torch.no_grad()
def generate_reports(model, val_loader, classes, device, reports_dir: Path, amp_enabled: bool):
    """Generate classification report and confusion matrix (PNG + CSV) on validation set."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds, all_labels = [], []
    for imgs, lbls in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())

    # Classification report
    rpt = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print("\n=== Classification Report (val) ===\n")
    #print(rpt)
    (reports_dir / "classification_report.txt").write_text(rpt, encoding="utf-8")

    # Confusion matrix (PNG + CSV)
    # import matplotlib.pyplot as plt
    # cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    # np.savetxt(reports_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # fig_w = min(12, max(6, len(classes) * 0.2))  # autosize a bit
    # fig_h = fig_w
    # fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # im = ax.imshow(cm, interpolation="nearest")
    # ax.figure.colorbar(im, ax=ax)
    # ax.set(
    #     xticks=np.arange(len(classes)),
    #     yticks=np.arange(len(classes)),
    #     xlabel="Predicted label",
    #     ylabel="True label",
    #     title="Confusion Matrix (val)"
    # )
    # if len(classes) <= 30:  # only label ticks if small
    #     ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    #     ax.set_yticks(np.arange(len(classes)), labels=classes)
    # plt.tight_layout()
    # fig.savefig(reports_dir / "confusion_matrix.png", dpi=300)
    # plt.close(fig)


# ----------------------------
# Argparse / Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser("ResNet50 OneCycle+AMP (ImageNet-1k)")
    #p.add_argument("--data-root", required=True, help="root that contains train/ and val/ (ImageFolder layout)")
    default_root = str(Path(__file__).resolve().parent / "data" / "imagenet")
    p.add_argument("--data-root", default=default_root, help=f"dataset root (defaults to {default_root})")
    p.add_argument("--name", default="r50_imagenet1k_onecycle_amp")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--stats-sample", type=int, default=100000,
                   help="Limit number of train images used to compute mean/std (None = use all)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    # OneCycle params
    p.add_argument("--max-lr", type=float, default=None, help="peak LR; if None, computed via 0.1*(batch/256)")
    p.add_argument("--pct-start", type=float, default=0.1)
    p.add_argument("--div-factor", type=float, default=25.0)
    p.add_argument("--final-div-factor", type=float, default=1e4)
    # Options
    p.add_argument("--no-amp", action="store_true", help="disable AMP")
    p.add_argument("--use-class-style-aug", action="store_true",
                  help="Use CIFAR-like aug (Affine+CoarseDropout) instead of standard ImageNet aug.")
    p.add_argument("--resume", action="store_true", help="resume from checkpoints/<name>/checkpoint.pth if present")
    # Reports
    p.add_argument("--reports", action="store_true", help="Generate classification report + confusion matrix at end (val set)")
    p.add_argument("--use-best-for-reports", action="store_true",
                  help="If set with --reports, load best.pth before generating reports")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    print(f"=> device: {device} | AMP: {amp_enabled}")

    # Data loaders (computes & caches mean/std from train split)
    train_loader, val_loader, classes, mean, std = make_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        sample_limit_for_stats=args.stats_sample,          # e.g., 20000 to speed up on full 1k
        use_class_style_aug=args.use_class_style_aug,
    )
    num_classes = len(classes)
    print(f"=> classes: {num_classes} | mean={mean} | std={std}")

    # Model (from scratch)
    model = resnet50(weights=None, num_classes=num_classes).to(device)

    # (Optional) save model summary like in CIFAR flow
    try:
        from torchinfo import summary
        summary_text = str(summary(
            model,
            input_size=(args.batch_size, 3, args.img_size, args.img_size),
            depth=3,
            col_names=("input_size", "output_size", "num_params", "kernel_size")
        ))
        ms_dir = Path("reports") / args.name
        ms_dir.mkdir(parents=True, exist_ok=True)
        (ms_dir / "model_summary.txt").write_text(summary_text, encoding="utf-8")
        print(f"=> model summary saved to reports/{args.name}/model_summary.txt")
    except Exception as e:
        print(f"[warn] torchinfo summary skipped: {e}")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,  # placeholder; OneCycle controls effective LR
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # LR schedule (OneCycle per-batch)
    steps_per_epoch = len(train_loader)
    if args.max_lr is None:
        # Linear scaling rule: 0.1 for total batch 256
        world_size = 1  # single-GPU script
        total_batch = args.batch_size * world_size
        max_lr = 0.1 * (total_batch / 256.0)
        print(f"=> computed max_lr: {max_lr:.6f} (total_batch={total_batch})")
    else:
        max_lr = args.max_lr
        print(f"=> using max_lr: {max_lr:.6f}")

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
    )

    # AMP scaler
    scaler = GradScaler(enabled=amp_enabled)

    # Logging / checkpoints / outputs
    run_dir = Path("runs") / args.name
    ckpt_dir = Path("checkpoints") / args.name
    out_dir = Path("out") / args.name
    reports_dir = Path("reports") / args.name
    for d in (run_dir, ckpt_dir, out_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    # CIFAR-style logs setup
    log_path = out_dir / "train_log.csv"
    md_log_path = out_dir / "logs.md"
    if not log_path.exists():
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("epoch,phase,loss,top1,top5,lr,imgs_per_sec\n")
    with open(md_log_path, "a", encoding="utf-8") as f:
        if md_log_path.stat().st_size == 0:
            f.write("# Training Logs (terminal-like)\n\n```\n")
        else:
            f.write("```\n")  # reopen fence for a new session

    writer = SummaryWriter(str(run_dir))

    # Resume
    start_epoch = 0
    best_top1 = 0.0
    ckpt_path = ckpt_dir / "checkpoint.pth"
    best_path = ckpt_dir / "best.pth"
    if args.resume and ckpt_path.exists():
        print(f"=> resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_top1 = ckpt.get("best_top1", 0.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"=> resumed at epoch {start_epoch}, best_top1 {best_top1:.2f}%")

    # Initial eval (sanity)
    print("=> initial evaluation")
    val_out = evaluate(start_epoch, model, criterion, val_loader, device, amp_enabled, writer)
    _append_logs(md_log_path, log_path, start_epoch, "val",
                 val_out["loss"], val_out["top1"], val_out["top5"],
                 optimizer.param_groups[0]["lr"], val_out["ips"])

    # Train
    print("=> starting training")
    optimizer._step_count = 0  # avoid "scheduler before optimizer" warning under AMP
    for epoch in range(start_epoch, args.epochs):
        tr_out = train_one_epoch(epoch, model, optimizer, scheduler, scaler, criterion,
                                 train_loader, device, amp_enabled, writer)
        _append_logs(md_log_path, log_path, epoch, "train",
                     tr_out["loss"], tr_out["top1"], tr_out["top5"],
                     optimizer.param_groups[0]["lr"], tr_out["ips"])

        val_out = evaluate(epoch, model, criterion, val_loader, device, amp_enabled, writer)
        _append_logs(md_log_path, log_path, epoch, "val",
                     val_out["loss"], val_out["top1"], val_out["top5"],
                     optimizer.param_groups[0]["lr"], val_out["ips"])

        # Checkpointing
        is_best = val_out["top1"] > best_top1
        best_top1 = max(best_top1, val_out["top1"])
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top1": best_top1,
            "args": vars(args),
            "mean": mean,
            "std": std,
        }
        torch.save(state, ckpt_path)
        if is_best:
            torch.save(state, best_path)

    print(f"=> done. best val top1: {best_top1:.2f}%")
    writer.close()

    # Close markdown code fence
    with open(md_log_path, "a", encoding="utf-8") as f:
        f.write("```\n")

    # Final reports on val set
    if args.reports:
        if args.use_best_for_reports and best_path.exists():
            print("=> loading best.pth for reports")
            best_ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(best_ckpt["model"])
        print("=> generating reports (classification report + confusion matrix)")
        generate_reports(model, val_loader, classes, device, reports_dir, amp_enabled)

    # ---------------------------------------------
    # Save Accuracy & Loss plots (train vs val)
    # ---------------------------------------------
    try:
        df = pd.read_csv(out_dir / "train_log.csv")

        # Split by phase and sort by epoch so x and y lengths match
        train_df = df[df["phase"] == "train"].sort_values("epoch")
        val_df   = df[df["phase"] == "val"].sort_values("epoch")

        # X axes (can differ in length because we log an initial val before training)
        x_tr = train_df["epoch"].to_list()
        x_va = val_df["epoch"].to_list()

        # Y series
        tr_loss = train_df["loss"].to_list()
        va_loss = val_df["loss"].to_list()
        tr_top1 = train_df["top1"].to_list()
        va_top1 = val_df["top1"].to_list()

        # 1) Loss
        plt.figure(figsize=(8, 6))
        plt.plot(x_tr, tr_loss, label="Train Loss", marker="o")
        plt.plot(x_va, va_loss, label="Val Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / "loss_curve.png", dpi=300)
        plt.close()

        # 2) Top-1 Accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(x_tr, tr_top1, label="Train Top-1", marker="o")
        plt.plot(x_va, va_top1, label="Val Top-1", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training vs Validation Accuracy (Top-1)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / "accuracy_curve.png", dpi=300)
        plt.close()

        print(f"=> saved plots: {reports_dir}/loss_curve.png & accuracy_curve.png")

    except Exception as e:
        print(f"[warn] Could not save train/val plots: {e}")


    print("=> all done.")


if __name__ == "__main__":
    main()
