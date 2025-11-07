'''
python scripts/rehumanize_imagenet_reports.py \
  --root reports/r50_imagenet1k_onecycle_amp_bs64_ep150 \
  --map utils/imagenet_class_index.json
'''
#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import pandas as pd

def load_synset_map(p: Path):
    d = json.load(open(p))
    return {v[0]: v[1] for v in d.values()}  # synset -> human

def find_latest_report(root: Path):
    cands = sorted(root.rglob("classification_report.txt"))
    return cands[-1] if cands else None

def rehumanize_classification_report(cr_path: Path, syn2name: dict):
    text = cr_path.read_text()
    def repl(m):
        syn = m.group(0)
        return f"{syn} ({syn2name.get(syn, 'unknown')})"
    new_text = re.sub(r"\bn\d{8}\b", repl, text)
    cr_path.write_text(new_text)
    return cr_path

def rehumanize_confusion_matrix(cm_path: Path, syn2name: dict):
    if not cm_path.exists():
        return None
    df = pd.read_csv(cm_path, index_col=0)
    def map_lbl(x): 
        s = str(x)
        return syn2name.get(s, s)  # if headers were synsets, map; else keep
    df.index = [map_lbl(i) for i in df.index]
    df.columns = [map_lbl(c) for c in df.columns]
    df.to_csv(cm_path, index=True)
    return cm_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Directory tree to search for reports")
    ap.add_argument("--cr", default=None, help="Path to classification_report.txt")
    ap.add_argument("--cm", default=None, help="Path to confusion_matrix.csv")
    ap.add_argument("--map", default="utils/imagenet_class_index.json", help="Mapping JSON path")
    args = ap.parse_args()

    syn2name = load_synset_map(Path(args.map))
    cr_path = Path(args.cr) if args.cr else find_latest_report(Path(args.root))
    if not cr_path or not cr_path.exists():
        raise SystemExit("No classification_report.txt found. Provide --cr explicitly.")
    cm_path = Path(args.cm) if args.cm else cr_path.parent / "confusion_matrix.csv"

    out1 = rehumanize_classification_report(cr_path, syn2name)
    print(f"[ok] Rewrote: {out1}")
    out2 = rehumanize_confusion_matrix(cm_path, syn2name)
    if out2: print(f"[ok] Rewrote: {out2}")

if __name__ == "__main__":
    main()
