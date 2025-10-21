#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Auto-generate README.md for ImageNet-Mini ResNet-50 run `r50_onecycle_amp`."""

from __future__ import annotations
import csv
from pathlib import Path

EXP_NAME = "r50_onecycle_amp"
ROOT = Path(__file__).resolve().parent
DIR_LR = ROOT / "lr_finder_plots"
DIR_OUT = ROOT / "out" / EXP_NAME
DIR_REPORTS = ROOT / "reports" / EXP_NAME
README_PATH = ROOT / "README.md"

def newest_file(folder: Path, pattern: str = "*"):
    if not folder.exists():
        return None
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def read_train_log(csv_path: Path):
    best_top1 = best_top5 = best_loss = best_ips = None
    final_top1 = final_top5 = final_loss = final_ips = None
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        vals = [r for r in reader if r.get("phase") == "val"]
    if not vals:
        return {}
    vals.sort(key=lambda x: int(x["epoch"]))
    final = vals[-1]
    best = max(vals, key=lambda x: float(x["top1"]))
    return {
        "best_top1": best["top1"],
        "best_top5": best["top5"],
        "best_loss": best["loss"],
        "best_ips": best["imgs_per_sec"],
        "final_top1": final["top1"],
        "final_top5": final["top5"],
        "final_loss": final["loss"],
        "final_ips": final["imgs_per_sec"],
    }

def main():
    lr_latest = newest_file(DIR_LR, "*.png")
    lr_img = lr_latest.name if lr_latest else "REPLACE_WITH_LATEST.png"
    metrics = read_train_log(DIR_OUT / "train_log.csv")
    with README_PATH.open("r+", encoding="utf-8") as f:
        content = f.read()
        for k, v in metrics.items():
            content = content.replace("…", v if v else "…", 1)
        if lr_latest:
            content = content.replace("REPLACE_WITH_LATEST.png", lr_img)
        f.seek(0)
        f.write(content)
        f.truncate()
    print("[ok] README.md updated")

if __name__ == "__main__":
    main()
