#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_readme.py
Auto-fills README.md (Sections 1 & 5) with metrics from out/<RUN>/train_log.csv
and leaves Sections 3 & 4 untouched.

Usage:
    python update_readme.py --exp <RUN_NAME>
    # if --exp is omitted, the newest out/* having train_log.csv is used
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

def load_metrics(exp_name: str):
    csv_path = ROOT / "out" / exp_name / "train_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "phase" not in df.columns:
        raise ValueError(f"'phase' column missing in {csv_path}")
    df_val = df[df["phase"] == "val"].copy()
    if df_val.empty:
        raise ValueError(f"No 'val' rows found in {csv_path}")
    # Find best top1 (max)
    df_val["top1"] = df_val["top1"].astype(float)
    df_val["top5"] = df_val["top5"].astype(float)
    best_idx = df_val["top1"].idxmax()
    best_row = df_val.loc[best_idx]
    best_top1 = float(best_row["top1"])
    best_top5 = float(best_row["top5"])
    best_epoch = int(best_row["epoch"])
    return best_top1, best_top5, best_epoch

def fill_tokens(md: str, run_name: str, best_top1: float, best_top5: float, best_epoch: int) -> str:
    tokens = {
        "{{RUN_NAME}}": run_name,
        "{{BEST_TOP1}}": f"{best_top1:.2f}%",
        "{{BEST_TOP5}}": f"{best_top5:.2f}%",
        "{{BEST_EPOCH}}": str(best_epoch),
    }
    for k, v in tokens.items():
        md = md.replace(k, v)
    return md

def main():
    args = parse_args()
    exp = args.exp or find_latest_exp()
    if not exp:
        raise SystemExit("No experiment found under out/* with train_log.csv. Provide --exp <RUN_NAME>.")
    best_top1, best_top5, best_epoch = load_metrics(exp)

    if not README_PATH.exists():
        raise FileNotFoundError(f"README not found: {README_PATH}")
    content = README_PATH.read_text(encoding="utf-8")

    content_new = fill_tokens(content, exp, best_top1, best_top5, best_epoch)

    README_PATH.write_text(content_new, encoding="utf-8")
    print(f"[ok] README.md updated with run '{exp}': best_top1={best_top1:.2f} best_top5={best_top5:.2f} epoch={best_epoch}")

if __name__ == "__main__":
    main()
