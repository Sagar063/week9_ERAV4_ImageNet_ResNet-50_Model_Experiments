# dataset/imagenet_mini.py
import os, json, math
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
import cv2

_DEFAULT_IMG_SIZE = 224
_STATS_DIR = "data_stats"
#_STATS_FILE = "imagenet_1k_stats.json"
_STATS_FILE = "imagenet_1k_aws_stats.json"


class AlbumentationsImageFolder(Dataset):
    """
    Wraps torchvision.datasets.ImageFolder but applies Albumentations transforms.
    Albumentations expects numpy HWC images; torchvision returns PIL.
    """
    def __init__(self, root: str, transform: Optional[A.BasicTransform], class_to_idx=None):
        # 1) Build the base dataset WITHOUT passing class_to_idx
        self.base = datasets.ImageFolder(root)
        self.transform = transform

        # 2) If a desired class_to_idx mapping is provided (e.g., from train),
        #    override the base mapping and REMAP targets to match it.
        if class_to_idx is not None:
            # Build idx->class list in the desired order
            idx_to_class = [None] * len(class_to_idx)
            for cls_name, idx in class_to_idx.items():
                idx_to_class[idx] = cls_name

            # Remap samples/targets so that labels follow the provided mapping
            new_samples = []
            new_targets = []
            for (path, _) in self.base.samples:
                cls_name = Path(path).parent.name
                tgt = class_to_idx[cls_name]  # will KeyError if class missing, which is good to catch
                new_samples.append((path, tgt))
                new_targets.append(tgt)

            # Override internals
            self.base.samples = new_samples
            self.base.targets = new_targets
            self.base.class_to_idx = class_to_idx
            self.base.classes = idx_to_class

        # Expose common attributes
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx
        self.samples = self.base.samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = self.base.loader(path)   # PIL RGB
        img = np.array(img)            # HWC uint8
        if self.transform is not None:
            out = self.transform(image=img)
            img_t = out["image"]       # CHW float tensor
        else:
            img_t = ToTensorV2()(image=img)["image"]
        return img_t, target



def _compute_mean_std(
    root: str,
    split: str = "train",
    img_size: int = _DEFAULT_IMG_SIZE,
    sample_limit: Optional[int] = None,
    cache_dir: str = _STATS_DIR,
    cache_file: str = _STATS_FILE,
) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean/std over the dataset split and cache to JSON.
    Uses Resize(256)->CenterCrop(img_size)->ToTensor (no normalization, no aug).
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) / cache_file
    if cache_path.exists():
        with open(cache_path, "r") as f:
            obj = json.load(f)
        if obj.get("source") == os.path.abspath(root):
            return obj["mean"], obj["std"]

    tf = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # [0,1]
    ])

    ds = datasets.ImageFolder(os.path.join(root, split), transform=tf)
    if sample_limit is not None:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(sample_limit, len(ds)))))

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    total_pixels = 0
    channel_sum  = torch.zeros(3)
    channel_sum2 = torch.zeros(3)
    for x, _ in loader:  # x: [B,C,H,W] in [0,1]
        bs, _, H, W = x.shape
        total_pixels += bs * H * W
        channel_sum  += x.sum(dim=[0, 2, 3])
        channel_sum2 += (x ** 2).sum(dim=[0, 2, 3])

    mean = (channel_sum / total_pixels).tolist()
    var  = (channel_sum2 / total_pixels - torch.tensor(mean) ** 2).tolist()
    std  = [float(math.sqrt(v)) for v in var]

    with open(cache_path, "w") as f:
        json.dump({
            "source": os.path.abspath(root),
            "split": split,
            "img_size": img_size,
            "mean": mean,
            "std": std
        }, f, indent=2)

    return mean, std


def make_transforms_albu(
    mean: List[float],
    std: List[float],
    img_size: int = _DEFAULT_IMG_SIZE,
    use_class_style: bool = True
):
    """
    Albumentations pipelines. If use_class_style=True, mimic your CIFAR aug:
    - HFlip, moderate affine, coarse dropout.
    For ImageNet, we also prefer RandomResizedCrop for stronger invariance.
    """
    if use_class_style:
        train_tf = A.Compose([
            # A.RandomResizedCrop(img_size, img_size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=1),
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            #A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), translate_percent=(0.0, 0.02), mode=1, p=0.5),
            A.Affine(   scale=(0.9, 1.1),
                        rotate=(-10, 10),
                        translate_percent=(0.0, 0.02),
                        border_mode=cv2.BORDER_REFLECT_101,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.5,),
            # A.CoarseDropout(max_holes=1, max_height=int(0.10 * img_size), max_width=int(0.10 * img_size),
            #                 min_holes=1, min_height=int(0.05 * img_size), min_width=int(0.05 * img_size),
            #                 fill_value=(int(255*mean[0]), int(255*mean[1]), int(255*mean[2])), p=0.5),
            CoarseDropout(  holes=1,
                            hole_height_range=(0.05, 0.10),
                            hole_width_range=(0.05, 0.10),
                            fill_value=0,
                            p=0.5,),
            # A.Cutout(   num_holes=1,
            #             max_h_size=int(0.1 * img_size),
            #             max_w_size=int(0.1 * img_size),
            #             fill_value=(0, 0, 0),
            #             always_apply=False,
            #             p=0.5,),

            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        train_tf = A.Compose([
           # A.RandomResizedCrop(img_size, img_size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=1),
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    val_tf = A.Compose([
        # A.LongestMaxSize(max_size=256, interpolation=1),
        # A.CenterCrop(img_size, img_size),
        A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_LINEAR),  # ensures the shorter side >= 256
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def make_loaders(
    data_root: str,
    batch_size: int,
    workers: int,
    img_size: int = _DEFAULT_IMG_SIZE,
    sample_limit_for_stats: Optional[int] = None,
    use_class_style_aug: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str], List[float], List[float]]:

    mean, std = _compute_mean_std(
        root=data_root,
        split="train",
        img_size=img_size,
        sample_limit=sample_limit_for_stats,
    )

    train_tf, val_tf = make_transforms_albu(mean, std, img_size=img_size, use_class_style=use_class_style_aug)

    # supply same class_to_idx to val to lock label IDs
    train_ds = AlbumentationsImageFolder(os.path.join(data_root, "train"), transform=train_tf, class_to_idx=None)
    val_ds   = AlbumentationsImageFolder(os.path.join(data_root, "val"),   transform=val_tf,  class_to_idx=train_ds.class_to_idx)

    # sanity check: ensure identical class lists
    assert train_ds.classes == val_ds.classes, \
        f"Class mismatch between train ({len(train_ds.classes)}) and val ({len(val_ds.classes)})!"

    # Conditional prefetch settings
    prefetch_args = {"prefetch_factor": 2} if workers > 0 else {}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=True, persistent_workers=(workers > 0),
        **prefetch_args
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=(workers > 0),
        **prefetch_args
    )


    return train_loader, val_loader, train_ds.classes, mean, std
