#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-generate README.md for ImageNet-Mini ResNet-50 run `r50_onecycle_amp`.

What this script does:
- Pick latest LR-Finder plot from lr_finder_plots/ and swap into README
- Fill the metrics table (best/final Top-1/Top-5/Loss/IPS) from out/<exp>/train_log.csv
- Inject file contents into collapsible sections:
  * out/<exp>/logs.md
  * reports/<exp>/model_summary.txt
  * reports/<exp>/classification_report.txt  (only accuracy/macro avg/weighted avg lines)
"""

from __future__ import annotations
import csv
import re
from pathlib import Path
from typing import Optional, List

EXP_NAME = "r50_onecycle_amp"
ROOT = Path(__file__).resolve().parent

DIR_LR = ROOT / "lr_finder_plots"
DIR_OUT = ROOT / "out" / EXP_NAME
DIR_REPORTS = ROOT / "reports" / EXP_NAME
README_PATH = ROOT / "README.md"

# ---- Helpers -----------------------------------------------------------------

def newest_file(folder: Path, pattern: str = "*.png") -> Optional[Path]:
    if not folder.exists():
        return None
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def read_train_log(csv_path: Path):
    """Return dict of best/final metrics from val rows in your CSV."""
    if not csv_path.exists():
        return {}
    vals = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("phase") != "val":
                continue
            try:
                vals.append(
                    dict(
                        epoch=int(r["epoch"]),
                        loss=float(r["loss"]),
                        top1=float(r["top1"]),
                        top5=float(r["top5"]),
                        ips=float(r["imgs_per_sec"]),
                    )
                )
            except Exception:
                continue
    if not vals:
        return {}
    vals.sort(key=lambda x: x["epoch"])
    final = vals[-1]
    best = max(vals, key=lambda x: x["top1"])
    return {
        "best_top1": f'{best["top1"]:.2f}',
        "best_top5": f'{best["top5"]:.2f}',
        "best_loss": f'{best["loss"]:.4f}',
        "best_ips": f'{best["ips"]:.1f}',
        "final_top1": f'{final["top1"]:.2f}',
        "final_top5": f'{final["top5"]:.2f}',
        "final_loss": f'{final["loss"]:.4f}',
        "final_ips": f'{final["ips"]:.1f}',
    }

def read_text(p: Path, limit: int = 80000) -> Optional[str]:
    """Read text file; truncate if huge to keep README snappy."""
    if not p.exists():
        return None
    s = p.read_text(encoding="utf-8", errors="ignore")
    if limit and len(s) > limit:
        s = s[:limit].rstrip() + "\n… (truncated)\n"
    return s

def extract_cls_toplines(txt: Optional[str]) -> Optional[str]:
    """Return only accuracy/macro avg/weighted avg lines from sklearn report."""
    if not txt:
        return None
    wanted: List[str] = []
    for line in txt.splitlines():
        low = line.strip().lower()
        if low.startswith("accuracy") or low.startswith("macro avg") or low.startswith("weighted avg"):
            wanted.append(line.rstrip())
    return "\n".join(wanted) if wanted else None

# ---- Main --------------------------------------------------------------------

def main():
    # Read artifacts
    lr_latest = newest_file(DIR_LR, "*.png")
    metrics = read_train_log(DIR_OUT / "train_log.csv")
    logs_md = read_text(DIR_OUT / "logs.md")
    model_summary = read_text(DIR_REPORTS / "model_summary.txt")
    cls_report_top = extract_cls_toplines(read_text(DIR_REPORTS / "classification_report.txt", limit=200000))

    # Load README
    content = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

    # 1) Swap LR-Finder image token (if your README uses a fixed filename, keep this)
    if lr_latest:
        content = content.replace("REPLACE_WITH_LATEST.png", lr_latest.name)

    # 2) Fill the metrics table by replacing the first eight "…" in that table.
    #    This assumes the table lines in README use "…" placeholders for metrics.
    #    To avoid touching any other ellipses elsewhere, limit replacement to the Metrics block.
    def fill_metrics_block(md: str) -> str:
        pattern = r"(\| Split \| Top-1.*?\n)([\s\S]*?)(\n\n|---\n|\Z)"
        m = re.search(pattern, md)
        if not m or not metrics:
            return md
        block = m.group(0)
        # Replace in order of appearance: best_top1, best_top5, best_loss, best_ips, final_top1, final_top5, final_loss, final_ips
        seq = [
            metrics.get("best_top1", "…"),
            metrics.get("best_top5", "…"),
            metrics.get("best_loss", "…"),
            metrics.get("best_ips", "…"),
            metrics.get("final_top1", "…"),
            metrics.get("final_top5", "…"),
            metrics.get("final_loss", "…"),
            metrics.get("final_ips", "…"),
        ]
        new_block = block
        for v in seq:
            new_block = new_block.replace("…", v, 1)
        return md.replace(block, new_block)
    content = fill_metrics_block(content)

    # 3) Inject logs.md into its placeholder
    if logs_md:
        content = content.replace(
            "… (inserted from out/r50_onecycle_amp/logs.md) …",
            logs_md.strip()
        )

    # 4) Inject model_summary.txt into its placeholder
    if model_summary:
        content = content.replace(
            "… (inserted from reports/r50_onecycle_amp/model_summary.txt) …",
            model_summary.strip()
        )

    # 5) Inject classification report toplines into the three-line placeholder block
    if cls_report_top:
        # Replace the entire three-line placeholder where it appears inside a code fence
        content = content.replace(
            "accuracy …  \nmacro avg …  \nweighted avg …",
            cls_report_top
        ).replace(
            "accuracy …\nmacro avg …\nweighted avg …",
            cls_report_top
        )

    # Write README back
    README_PATH.write_text(content, encoding="utf-8")
    print("[ok] README.md updated with LR plot, metrics, and embedded logs/summary/report.")

if __name__ == "__main__":
    main()
