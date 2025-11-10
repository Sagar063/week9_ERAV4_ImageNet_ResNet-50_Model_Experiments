#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_readme.py (v2)
Extends the original script to fill both global and local/AWS metrics placeholders.
Preserves --exp handling and best_epoch logic.
Usage:
    python update_readme.py [--exp <RUN_NAME>]
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "README.md"

def find_latest_exp() -> str:
    out_dir = ROOT / "out"
    if not out_dir.exists():
        return ""
    cands = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        if (c / "train_log.csv").exists():
            return c.name
    return ""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="Experiment folder name under out/")
    return ap.parse_args()

def load_best_metrics(exp_name: str):
    csv_path = ROOT / "out" / exp_name / "train_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "phase" not in df.columns:
        raise ValueError(f"'phase' column missing in {csv_path}")
    df_val = df[df["phase"] == "val"].copy()
    df_val["top1"] = df_val["top1"].astype(float)
    df_val["top5"] = df_val["top5"].astype(float)
    best_idx = df_val["top1"].idxmax()
    best_row = df_val.loc[best_idx]
    best_top1 = float(best_row["top1"])
    best_top5 = float(best_row["top5"])
    best_epoch = int(best_row["epoch"])
    return best_top1, best_top5, best_epoch

def load_train_val_metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    train_df = df[df["phase"] == "train"].copy()
    val_df = df[df["phase"] == "val"].copy()
    t1, t5 = train_df["top1"].iloc[-1], train_df["top5"].iloc[-1]
    v1 = val_df["top1"].max()
    v5 = val_df.loc[val_df["top1"].idxmax(), "top5"]
    return float(t1), float(t5), float(v1), float(v5)

def fill_tokens(md: str, run_name: str, best_top1: float, best_top5: float, best_epoch: int,
                local_metrics=None, aws_metrics=None) -> str:
    tokens = {
        "{{RUN_NAME}}": run_name,
        "{{BEST_TOP1}}": f"{best_top1:.2f}%",
        "{{BEST_TOP5}}": f"{best_top5:.2f}%",
        "{{BEST_EPOCH}}": str(best_epoch),
    }
    if local_metrics:
        lt1, lt5, lv1, lv5 = local_metrics
        tokens.update({
            "{{LOCAL_TRAIN_TOP1}}": f"{lt1:.2f}%",
            "{{LOCAL_TRAIN_TOP5}}": f"{lt5:.2f}%",
            "{{LOCAL_VAL_TOP1}}": f"{lv1:.2f}%",
            "{{LOCAL_VAL_TOP5}}": f"{lv5:.2f}%",
        })
    if aws_metrics:
        at1, at5, av1, av5 = aws_metrics
        tokens.update({
            "{{AWS_TRAIN_TOP1}}": f"{at1:.2f}%",
            "{{AWS_TRAIN_TOP5}}": f"{at5:.2f}%",
            "{{AWS_VAL_TOP1}}": f"{av1:.2f}%",
            "{{AWS_VAL_TOP5}}": f"{av5:.2f}%",
        })
    for k, v in tokens.items():
        md = md.replace(k, v)
    return md

def main():
    args = parse_args()
    exp = args.exp or find_latest_exp()
    if not exp:
        raise SystemExit("No experiment found under out/* with train_log.csv. Provide --exp <RUN_NAME>.")
    best_top1, best_top5, best_epoch = load_best_metrics(exp)

    local_csv = ROOT / "out" / "r50_imagenet1k_onecycle_amp_bs64_ep150" / "train_log.csv"
    aws_csv = ROOT / "out" / "imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6" / "train_log.csv"

    local_metrics = aws_metrics = None
    if local_csv.exists():
        local_metrics = load_train_val_metrics(local_csv)
    if aws_csv.exists():
        aws_metrics = load_train_val_metrics(aws_csv)

    content = README_PATH.read_text(encoding="utf-8")
    content_new = fill_tokens(content, exp, best_top1, best_top5, best_epoch, local_metrics, aws_metrics)
    README_PATH.write_text(content_new, encoding="utf-8")
    print(f"[ok] README.md updated. Run='{exp}' top1={best_top1:.2f} top5={best_top5:.2f} epoch={best_epoch}")

if __name__ == "__main__":
    main()
