#!/usr/bin/env python3
"""
make_debug_data.py
------------------
Creates a small debug subset of ImageNet-like dataset for a few selected classes.

✅ Both train/ and val/ will have the same 7 class folders.
✅ Each split gets 10 random images per class (different images in each split).

Source:  /mnt/imagenet1k/{train,val}/<class_name>/*.jpg
Target:  /mnt/debug_data/{train,val}/<class_name>/*.jpg
"""

import random, shutil
from pathlib import Path

# === CONFIG ===
SRC_ROOT = Path("/mnt/imagenet1k")
DST_ROOT = Path("/mnt/debug_data")
SPLITS = ["train", "val"]
IMGS_PER_CLASS = 10

# 👉 Choose any 7 classes you want here:
SELECTED_CLASSES = [
    "n01440764",  # tench
    "n01443537",  # goldfish
    "n01484850",  # great_white_shark
    "n01491361",  # tiger_shark
    "n01494475",  # hammerhead
    "n01496331",  # electric_ray
    "n01498041",  # stingray
]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_subset(split: str, class_names):
    src_split = SRC_ROOT / split
    dst_split = DST_ROOT / split
    ensure_dir(dst_split)

    print(f"\n🔹 Creating {split} subset...")
    for cname in class_names:
        src_cls = src_split / cname
        dst_cls = dst_split / cname
        ensure_dir(dst_cls)

        if not src_cls.exists():
            print(f"⚠️  Missing {src_cls}, skipping")
            continue

        imgs = [f for f in src_cls.iterdir() if f.is_file()]
        if not imgs:
            continue

        # pick 10 different random images
        sel = random.sample(imgs, min(IMGS_PER_CLASS, len(imgs)))
        for img in sel:
            shutil.copy2(img, dst_cls / img.name)
        print(f"[{split}] {cname}: {len(sel)} images copied")

def main():
    print("🚀 Creating debug subset from /mnt/imagenet1k → /mnt/debug_data")
    print(f"Selected classes: {SELECTED_CLASSES}")

    for split in SPLITS:
        copy_subset(split, SELECTED_CLASSES)

    print("\n✅ Debug dataset created successfully at:", DST_ROOT)

if __name__ == "__main__":
    main()
