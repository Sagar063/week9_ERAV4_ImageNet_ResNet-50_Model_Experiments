
#!/usr/bin/env python3
# scripts/fix_imagenet_val.py
"""
Reorganize ImageNet ILSVRC2012 validation set into class subfolders.

Expected inputs:
  - --val-dir /path/to/val  (currently flat with 50k images)
  - --map-file ILSVRC2012_val_labels.txt  (1000 integers, one per image listed in val.txt order)
  - --synset-file synset_words.txt  (maps class index -> synset wnid)

This script creates /path/to/val/<wnid>/image.jpg structure.
If your dataset already has class subfolders, you don't need this.
"""
import argparse, os, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir', required=True, help='Path to val folder (flat)')
    ap.add_argument('--val-map', required=False, default='', help='Optional file with lines: <filename> <wnid>')
    ap.add_argument('--synset-file', required=False, default='', help='Optional synset mapping (idx->wnid)')
    args = ap.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise FileNotFoundError(val_dir)

    # Case 1: user provides a mapping file "<filename> <wnid>"
    mapping = {}
    if args.val_map and Path(args.val_map).exists():
        for line in open(args.val_map, 'r'):
            line=line.strip()
            if not line: continue
            fname, wnid = line.split()
            mapping[fname] = wnid
    else:
        print("No --val-map provided. If your val is already structured, skip this script.")
        return

    # Move files
    for fname, wnid in mapping.items():
        src = val_dir / fname
        if not src.exists():
            continue
        dst_dir = val_dir / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_dir / fname))
    print("Done restructuring val/ into class subfolders.")

if __name__ == "__main__":
    main()
