#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
from torchvision.models import resnet50

from dataset.imagenet import make_loaders
# from dataset.imagenet_mini import make_loaders
import fire


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
    data_root = str(Path(__file__).resolve().parent / "data" / "imagenet"),
    start_lr=1e-7,
    end_lr=1.0,
    num_iter=150,
    batch_size=64,
    workers=8,
    img_size=224,
    momentum=0.9,
    weight_decay=1e-4,
    use_class_style_aug=False,
    output_dir="lr_finder_plots_imagenet1k"
):
    """
    LR range test to pick a good --max-lr for OneCycle.
    Example (Windows path):
      python lr_finder.py find_lr --data_root "D:/ERA/week9/ImageNet-ResNet50-CNN/data/imagenet-mini"
    """
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=> device: {device}")
    print(f"=> LR Finder: start_lr={start_lr}, end_lr={end_lr}, num_iter={num_iter}")

    # Build train loader only
    train_loader, _, classes, _, _ = make_loaders(
        data_root=data_root,
        batch_size=batch_size,
        workers=workers,
        img_size=img_size,
        sample_limit_for_stats=100000, #None
        use_class_style_aug=use_class_style_aug,
    )


    model = resnet50(weights=None, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=momentum, weight_decay=weight_decay)

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter, step_mode="exp")

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'lr_finder_{ts}_start{start_lr}_end{end_lr}_iter{num_iter}.png')

    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)
    plt.title(f'Learning Rate Finder (iter: {num_iter})')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> LR curve saved to: {filepath}")

    try:
        print("=> Suggested LR:", lr_finder.suggestion())
    except Exception:
        pass

    lr_finder.reset()


if __name__ == "__main__":
    fire.Fire({"find_lr": find_lr})
