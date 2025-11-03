#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder

# from torchvision.models import resnet50
from model import ResNet50
from dataset.imagenet import make_loaders
# from dataset.imagenet_mini import make_loaders
import fire
import os, time
from typing import Optional

def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(x)
# def robust_lr_suggestion(history, spike_ratio: float = 1.30, smooth_win: int = 31, search_start_frac: float = 0.3):
#     """
#     Pick LR at steepest descent BEFORE the first real divergence spike.
#     - Ignore the very-left tail (search_start_frac of points).
#     - Smooth more to avoid early micro-spikes.
#     - Require a bigger jump (spike_ratio) before we say it diverged.
#     """
#     import numpy as np
#     lrs = np.array(history["lr"],   dtype=np.float64)
#     losses = np.array(history["loss"], dtype=np.float64)

#     m = np.isfinite(lrs) & np.isfinite(losses)
#     lrs, losses = lrs[m], losses[m]
#     n = lrs.size
#     if n < 20:
#         return None, None

#     # smooth
#     if 1 < smooth_win < n:
#         k = np.ones(smooth_win) / smooth_win
#         losses_s = np.convolve(losses, k, mode="same")
#     else:
#         losses_s = losses

#     run_min = np.minimum.accumulate(losses_s)

#     # first meaningful spike
#     spike_idx = np.where(losses_s > run_min * spike_ratio)[0]
#     cut = int(spike_idx[0]) if spike_idx.size > 0 else n

#     # ignore the left edge entirely
#     start = max(10, int(search_start_frac * n))
#     if cut - start < max(8, smooth_win):       # window too small? widen it
#         cut = min(n, start + max(12, smooth_win * 2))

#     # defend against degenerate cases
#     if cut <= start + 3:
#         start = max(0, start - 3)
#     if cut <= start + 3:
#         return None, None

#     g = np.gradient(losses_s[start:cut], np.log10(lrs[start:cut]))
#     idx_local = int(np.argmin(g))
#     idx = start + idx_local
#     return float(lrs[idx]), cut

def robust_lr_suggestion(
    history,
    spike_ratio: float = 1.06,      # ~+6% above running min
    spike_abs: float = 0.15,        # OR +0.15 absolute bump
    smooth_win: int = 51,           # heavier smoothing for ImageNet
    pos_grad_k: int = 8,            # need k consecutive +ve gradients
    search_start_frac: float = 0.10,# skip the leftmost 10%
    lr_floor: float = 1e-6,         # ignore LRs below this
    lr_ceil: float | None = None    # (optional) ignore LRs above this
):
    """
    Return (lr, cut_idx) where lr is the LR at steepest NEGATIVE slope
    BEFORE the first meaningful divergence spike.

    Spike is declared when ANY of these holds at index i:
      - losses_s[i] > run_min[i] * spike_ratio         (relative jump)
      - losses_s[i] > run_min[i] + spike_abs           (absolute bump)
      - gradients are positive for >= pos_grad_k steps (sustained rise)
    """
    import numpy as np

    lrs    = np.array(history["lr"],   dtype=np.float64)
    losses = np.array(history["loss"], dtype=np.float64)

    m = np.isfinite(lrs) & np.isfinite(losses)
    lrs, losses = lrs[m], losses[m]
    n = lrs.size
    if n < 20:
        return None, None

    # Smooth losses (moving average)
    if 1 < smooth_win < n:
        k = np.ones(smooth_win, dtype=np.float64) / smooth_win
        losses_s = np.convolve(losses, k, mode="same")
    else:
        losses_s = losses.copy()

    # restrict LR search to [lr_floor, lr_ceil]
    mask_lr = lrs >= lr_floor
    if lr_ceil is not None:
        mask_lr &= (lrs <= lr_ceil)
    if not np.any(mask_lr):
        return None, None
    lrs    = lrs[mask_lr]
    losses_s = losses_s[mask_lr]
    n = lrs.size

    # running min of smoothed loss
    run_min = np.minimum.accumulate(losses_s)

    # 1) ratio & absolute prominence criteria
    is_ratio_spike = losses_s > (run_min * spike_ratio)
    is_abs_spike   = (losses_s - run_min) > spike_abs

    # 2) sustained positive gradient
    g = np.gradient(losses_s, np.log10(lrs))
    pos = (g > 0).astype(np.int32)
    # rolling sum over window k
    if pos_grad_k > 1 and pos_grad_k < n:
        w = np.ones(pos_grad_k, dtype=np.int32)
        pos_run = np.convolve(pos, w, mode="same")
        is_sustained_pos = pos_run >= pos_grad_k
    else:
        is_sustained_pos = pos.astype(bool)

    is_spike = is_ratio_spike | is_abs_spike | is_sustained_pos

    # where to start searching (skip left tail)
    start = max(10, int(search_start_frac * n))

    # first spike after 'start'
    spike_idx = np.where(is_spike & (np.arange(n) >= start))[0]
    cut = int(spike_idx[0]) if spike_idx.size > 0 else n

    # ensure we have a useful window; widen if too small
    if cut - start < max(12, smooth_win // 2):
        cut = min(n, start + max(24, smooth_win))

    if cut <= start + 3:
        return None, None

    g_win = np.gradient(losses_s[start:cut], np.log10(lrs[start:cut]))
    idx_local = int(np.argmin(g_win))  # steepest descent
    idx = start + idx_local
    return float(lrs[idx]), cut


# def robust_lr_suggestion(history, spike_ratio: float = 1.10, smooth_win: int = 15):
#     """
#     Return (lr, cut_index) where lr is at the steepest loss descent BEFORE the first divergence spike.
#     - spike_ratio: consider 'diverged' once smoothed loss > (spike_ratio * running_min)
#     - smooth_win  : moving-average window on loss to reduce noise
#     """
#     lrs = np.array(history["lr"], dtype=np.float64)
#     losses = np.array(history["loss"], dtype=np.float64)

#     # keep finite
#     m = np.isfinite(lrs) & np.isfinite(losses)
#     lrs, losses = lrs[m], losses[m]
#     if lrs.size < 10:
#         return None, None

#     # smooth loss (simple moving average)
#     if 1 < smooth_win < losses.size:
#         k = np.ones(smooth_win) / smooth_win
#         losses_s = np.convolve(losses, k, mode="same")
#     else:
#         losses_s = losses

#     # running minimum of smoothed loss
#     run_min = np.minimum.accumulate(losses_s)

#     # first spike (divergence) index
#     spike = np.where(losses_s > run_min * spike_ratio)[0]
#     cut = int(spike[0]) if spike.size > 0 else int(losses_s.size)

#     # guard against tiny windows
#     if cut < max(8, smooth_win):
#         cut = min(losses_s.size, max(12, smooth_win * 2))

#     # choose steepest descent BEFORE cut
#     g = np.gradient(losses_s[:cut], np.log10(lrs[:cut]))
#     idx = int(np.argmin(g))
#     lr = float(lrs[idx])
#     return lr, cut


# def find_lr(
#     #data_root="data/imagenet_mini",
#     data_root = str(Path(__file__).resolve().parent / "data" / "imagenet-mini"),
#     start_lr=1e-7,
#     end_lr=1.0,
#     num_iter=150,
#     batch_size=64,
#     workers=8,
#     img_size=224,
#     momentum=0.9,
#     weight_decay=1e-4,
#     use_class_style_aug=False,
#     output_dir="lr_finder_plots"
# ):
def find_lr(
    #data_root="data/imagenet_mini",
    data_root: str = "/mnt/imagenet1k",
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iter: int = 150,
    batch_size: int = 64,
    workers: int = 8,
    img_size: int = 224,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    use_class_style_aug: bool = False,
    output_dir: str = "lr_finder_plots_imagenet1k",
    **kwargs,
):
    """
    LR range test to pick a good --max-lr for OneCycle.
    Example:
      python lr_finder.py find_lr --data_root /mnt/imagenet1k --batch_size 128 --iters 1000 \
        --start_lr 1e-7 --end_lr 1.0
    """
    # allow --iters
    # accept --iters as alias
    if "iters" in kwargs and kwargs["iters"] is not None:
        try:
            num_iter = int(kwargs["iters"])
        except Exception:
            pass

    # Fire may pass "False"/"True" strings
    use_class_style_aug = _to_bool(use_class_style_aug)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=> device: {device}")
    print(f"=> LR Finder: start_lr={start_lr}, end_lr={end_lr}, num_iter={num_iter}, batch={batch_size}, workers={workers}")

    # ---------------- Build train loader only ----------------
    train_loader, _, classes, _, _ = make_loaders(
        data_root=data_root,
        batch_size=batch_size,
        workers=workers,
        img_size=img_size,
        # Use a cap only for mean/std during transforms construction; the loader itself iterates full data
        sample_limit_for_stats=100000,
        use_class_style_aug=use_class_style_aug,
    )

    # ---------------- Model / Opt / Crit ----------------
    model = ResNet50(num_classes=len(classes), pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=momentum, weight_decay=weight_decay)

    # ---------------- Run LR range test ----------------
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode="exp",
    )

    # ---------------- Compute suggestion BEFORE plotting ----------------
    #suggested, cut_idx = robust_lr_suggestion(lr_finder.history, spike_ratio=1.10, smooth_win=15)
    suggested, cut_idx = robust_lr_suggestion(  lr_finder.history,
                                                spike_ratio=1.06,      # ~+6% over running min
                                                spike_abs=0.15,        # OR +0.15 absolute
                                                smooth_win=51,         # heavier smoothing
                                                pos_grad_k=8,          # sustained rise
                                                search_start_frac=0.10,# skip first 10%
                                                lr_floor=1e-6,         # ignore ultra-small LRs
                                                lr_ceil=None)


    if suggested is not None:
        # clamp away from edges (ignore bottom/top 5% of LR range)
        lrs_all = np.array(lr_finder.history["lr"])
        m = np.isfinite(lrs_all)
        lrs_all = lrs_all[m]
        if lrs_all.size:
            lo = lrs_all[int(0.05 * len(lrs_all))]
            hi = lrs_all[int(0.95 * len(lrs_all)) - 1]
            suggested = float(np.clip(suggested, lo, hi))

    # Fallback #1: steepest gradient over entire curve
    def _fallback_suggestion(history):
        lrs = np.array(history["lr"], dtype=np.float64)
        losses = np.array(history["loss"], dtype=np.float64)
        m = np.isfinite(lrs) & np.isfinite(losses)
        lrs, losses = lrs[m], losses[m]
        if len(lrs) < 5:
            return None
        grads = np.gradient(losses, np.log10(lrs))
        return float(lrs[int(np.nanargmin(grads))])

    if suggested is None:
        suggested = _fallback_suggestion(lr_finder.history)

    # Fallback #2: LR at global minimum loss
    if suggested is None:
        try:
            lrs = np.array(lr_finder.history["lr"])
            losses = np.array(lr_finder.history["loss"])
            m = np.isfinite(lrs) & np.isfinite(losses)
            lrs, losses = lrs[m], losses[m]
            suggested = float(lrs[int(np.nanargmin(losses))]) if lrs.size else None
        except Exception:
            suggested = None

    # ---------------- Plot and annotate ----------------
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'lr_finder_{ts}_start{start_lr}_end{end_lr}_iter{num_iter}.png')

    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)
    plt.title(f'Learning Rate Finder (iter: {num_iter})')

    # mark suggested LR
    if suggested is not None:
        ax.axvline(suggested, linestyle="--")
        ymax = ax.get_ylim()[1]
        ax.text(suggested, ymax * 0.9, f"LR≈{suggested:.2e}", rotation=90, va="top")

    # mark the first divergence 'cut' if available
    if cut_idx is not None:
        lrs_all = np.array(lr_finder.history["lr"])
        m = np.isfinite(lrs_all)
        lrs_all = lrs_all[m]
        if lrs_all.size and 0 <= cut_idx < lrs_all.size:
            ax.axvline(lrs_all[cut_idx], color="orange", alpha=0.4)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"=> LR curve saved to: {filepath}")

    # ---------------- Persist suggestion ----------------
    out_txt = os.path.join(output_dir, "lr_suggestion.txt")
    out_json = os.path.join(output_dir, "lr_suggestion.json")
    if suggested is not None:
        with open(out_txt, "w") as f:
            f.write(f"{suggested}\n")
        with open(out_json, "w") as f:
            json.dump(
                {
                    "suggested_lr": suggested,
                    "start_lr": start_lr,
                    "end_lr": end_lr,
                    "num_iter": num_iter,
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                },
                f,
                indent=2,
            )
        print("=> Suggested LR:", f"{suggested:.2E}")
    else:
        print("=> Suggested LR: (not available)")

    # clean up model/opt state inside LR finder
    lr_finder.reset()


if __name__ == "__main__":
    # expose the find_lr function; supports both positional and named args
    fire.Fire({"find_lr": find_lr})
