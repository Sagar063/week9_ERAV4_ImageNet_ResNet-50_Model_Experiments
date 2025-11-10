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

# ---------------------------
# Helpers (safe read & format)
# ---------------------------

def _safe_read(p: Path, max_chars=200000) -> str:
    """Read text file safely and trim very large payloads."""
    try:
        s = p.read_text(encoding="utf-8", errors="ignore")
        return (s[:max_chars] + "\n\n... [truncated]") if len(s) > max_chars else s
    except Exception as e:
        return f"[could not read {p}: {e}]"

def _col(df: pd.DataFrame, *names, default=None):
    """First existing column name among candidates (or default)."""
    for n in names:
        if n in df.columns:
            return n
    return default

def _fmt_pct(x, digits=2):
    return "—" if x is None else f"{float(x):.{digits}f}%"

def _fmt_plain(x, digits=4):
    return "—" if x is None else f"{float(x):.{digits}f}"

# ------------------------------------------------
# CSV summaries for Section 6 & Section 7 autofill
# ------------------------------------------------

def summarize_from_csv(csv_path: Path):
    """
    Returns a dict with:
      - val_best:   {epoch, top1, top5, loss, ips}  at best val Top-1
      - train_last: {top1, top5, loss, ips}        last train row
    Tolerates small variations in column names.
    """
    df = pd.read_csv(csv_path)

    col_phase = _col(df, "phase")
    col_epoch = _col(df, "epoch")
    col_top1  = _col(df, "top1", "acc1", "top1_acc")
    col_top5  = _col(df, "top5", "acc5", "top5_acc")
    col_loss  = _col(df, "loss")
    col_ips   = _col(df, "ips", "img/s", "imgs_per_sec", "throughput")

    if any(c is None for c in (col_phase, col_epoch, col_top1)):
        raise ValueError(f"Expected columns missing in {csv_path}")

    df_val   = df[df[col_phase] == "val"].copy()
    df_train = df[df[col_phase] == "train"].copy()

    # Best validation Top-1 row (+ epoch)
    best_idx = df_val[col_top1].astype(float).idxmax()
    best     = df_val.loc[best_idx]

    # Last training row
    train_last = df_train.iloc[-1] if not df_train.empty else None

    def get_num(row, name):
        try:
            return float(row[name])
        except Exception:
            return None

    out = {
        "val_best": {
            "epoch": int(best[col_epoch]),
            "top1":  get_num(best, col_top1),
            "top5":  get_num(best, col_top5),
            "loss":  get_num(best, col_loss),
            "ips":   get_num(best, col_ips),
        },
        "train_last": None
    }
    if train_last is not None:
        out["train_last"] = {
            "top1": get_num(train_last, col_top1),
            "top5": get_num(train_last, col_top5),
            "loss": get_num(train_last, col_loss),
            "ips":  get_num(train_last, col_ips),
        }
    return out

# -----------------
# Existing utilities
# -----------------

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

# --------------------------
# Token filling (extended v3)
# --------------------------

def fill_tokens(md: str, run_name: str, best_top1: float, best_top5: float, best_epoch: int,
                local_metrics=None, aws_metrics=None,
                local_summary=None, aws_summary=None) -> str:
    tokens = {
        "{{RUN_NAME}}": run_name,
        "{{BEST_TOP1}}": f"{best_top1:.2f}%",
        "{{BEST_TOP5}}": f"{best_top5:.2f}%",
        "{{BEST_EPOCH}}": str(best_epoch),
    }

    # Section 6 metrics (final train & best val basic numbers you already used)
    if local_metrics:
        lt1, lt5, lv1, lv5 = local_metrics
        tokens.update({
            "{{LOCAL_TRAIN_TOP1}}": f"{lt1:.2f}%",
            "{{LOCAL_TRAIN_TOP5}}": f"{lt5:.2f}%",
            "{{LOCAL_VAL_TOP1}}":   f"{lv1:.2f}%",
            "{{LOCAL_VAL_TOP5}}":   f"{lv5:.2f}%",
        })
    if aws_metrics:
        at1, at5, av1, av5 = aws_metrics
        tokens.update({
            "{{AWS_TRAIN_TOP1}}": f"{at1:.2f}%",
            "{{AWS_TRAIN_TOP5}}": f"{at5:.2f}%",
            "{{AWS_VAL_TOP1}}":   f"{av1:.2f}%",
            "{{AWS_VAL_TOP5}}":   f"{av5:.2f}%",
        })

    # Section 7 table tokens (best val row + final train row)
    if local_summary:
        vb = local_summary["val_best"]; tl = local_summary["train_last"] or {}
        tokens.update({
            "{{LOCAL_VAL_TOP1_BEST}}":     _fmt_pct(vb["top1"]),
            "{{LOCAL_VAL_TOP5_BEST}}":     _fmt_pct(vb["top5"]),
            "{{LOCAL_VAL_LOSS_AT_BEST}}":  _fmt_plain(vb["loss"]),
            "{{LOCAL_VAL_IPS_AT_BEST}}":   _fmt_plain(vb["ips"]),
            "{{LOCAL_VAL_EPOCH_AT_BEST}}": str(vb["epoch"]),
            "{{LOCAL_TRAIN_TOP1_FINAL}}":  _fmt_pct(tl.get("top1")),
            "{{LOCAL_TRAIN_TOP5_FINAL}}":  _fmt_pct(tl.get("top5")),
            "{{LOCAL_TRAIN_LOSS_FINAL}}":  _fmt_plain(tl.get("loss")),
            "{{LOCAL_TRAIN_IPS_FINAL}}":   _fmt_plain(tl.get("ips")),
        })

    if aws_summary:
        vb = aws_summary["val_best"]; tl = aws_summary["train_last"] or {}
        tokens.update({
            "{{AWS_VAL_TOP1_BEST}}":     _fmt_pct(vb["top1"]),
            "{{AWS_VAL_TOP5_BEST}}":     _fmt_pct(vb["top5"]),
            "{{AWS_VAL_LOSS_AT_BEST}}":  _fmt_plain(vb["loss"]),
            "{{AWS_VAL_IPS_AT_BEST}}":   _fmt_plain(vb["ips"]),
            "{{AWS_VAL_EPOCH_AT_BEST}}": str(vb["epoch"]),
            "{{AWS_TRAIN_TOP1_FINAL}}":  _fmt_pct(tl.get("top1")),
            "{{AWS_TRAIN_TOP5_FINAL}}":  _fmt_pct(tl.get("top5")),
            "{{AWS_TRAIN_LOSS_FINAL}}":  _fmt_plain(tl.get("loss")),
            "{{AWS_TRAIN_IPS_FINAL}}":   _fmt_plain(tl.get("ips")),
        })

    # Inline collapsible content (works even if old README kept include_relative)
    inline_map = {
        "{{LOCAL_LOGS_MD}}":        ROOT / "out" / "r50_imagenet1k_onecycle_amp_bs64_ep150" / "logs.md",
        "{{LOCAL_MODEL_SUMMARY}}":  ROOT / "reports" / "r50_imagenet1k_onecycle_amp_bs64_ep150" / "model_summary.txt",
        "{{AWS_LOGS_MD}}":          ROOT / "out" / "imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6" / "logs.md",
        "{{AWS_MODEL_SUMMARY}}":    ROOT / "reports" / "imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6" / "model_summary.txt",
        "{% include_relative out/r50_imagenet1k_onecycle_amp_bs64_ep150/logs.md %}":
            ROOT / "out" / "r50_imagenet1k_onecycle_amp_bs64_ep150" / "logs.md",
        "{% include_relative reports/r50_imagenet1k_onecycle_amp_bs64_ep150/model_summary.txt %}":
            ROOT / "reports" / "r50_imagenet1k_onecycle_amp_bs64_ep150" / "model_summary.txt",
        "{% include_relative out/imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/logs.md %}":
            ROOT / "out" / "imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6" / "logs.md",
        "{% include_relative reports/imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/model_summary.txt %}":
            ROOT / "reports" / "imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6" / "model_summary.txt",
    }
    for token, path in inline_map.items():
        md = md.replace(token, _safe_read(path))

    # Finally replace numeric/string tokens
    for k, v in tokens.items():
        md = md.replace(k, v)

    return md

# --------------
# Main execution
# --------------

def main():
    args = parse_args()
    exp = args.exp or find_latest_exp()
    if not exp:
        raise SystemExit("No experiment found under out/* with train_log.csv. Provide --exp <RUN_NAME>.")

    # Headline metrics (unchanged)
    best_top1, best_top5, best_epoch = load_best_metrics(exp)

    # CSVs for Sections 6 & 7
    local_csv = ROOT / "out" / "r50_imagenet1k_onecycle_amp_bs64_ep150" / "train_log.csv"
    aws_csv   = ROOT / "out" / "imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6" / "train_log.csv"

    local_metrics = aws_metrics = None
    local_summary = aws_summary = None

    if local_csv.exists():
        local_metrics = load_train_val_metrics(local_csv)     # Section 6
        local_summary = summarize_from_csv(local_csv)         # Section 7
    if aws_csv.exists():
        aws_metrics = load_train_val_metrics(aws_csv)         # Section 6
        aws_summary = summarize_from_csv(aws_csv)             # Section 7

    content = README_PATH.read_text(encoding="utf-8")
    content_new = fill_tokens(
        content, exp, best_top1, best_top5, best_epoch,
        local_metrics=local_metrics, aws_metrics=aws_metrics,
        local_summary=local_summary, aws_summary=aws_summary,
    )
    README_PATH.write_text(content_new, encoding="utf-8")
    print(f"[ok] README.md updated. Run='{exp}' top1={best_top1:.2f} top5={best_top5:.2f} epoch={best_epoch}")

if __name__ == "__main__":
    main()