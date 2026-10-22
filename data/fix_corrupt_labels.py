#!/usr/bin/env python3
"""Fix corrupt YOLO label files in datasets/indicdlp_subset/labels/.

Two types of corruption found:
  1. Moderately out-of-range values (1.0 < v <= 10.0):
     These are likely caused by absolute pixel coords divided by the wrong
     image dimension. Clamp cx, cy to [0,1] and clamp w, h to (0, 1].
     If after clamping the box area > 0 and w,h > 0.001, keep it.

  2. Wildly out-of-range values (v > 10.0):
     Source annotation was in raw pixel space with no normalization at all.
     These cannot be reliably rescued — drop the annotation.

Rule 6: NEVER overwrite the original files.
  Original files are backed up to labels/train_corrupt_backup/ and
  labels/val_corrupt_backup/ before any changes are written.

Usage
-----
    python data/fix_corrupt_labels.py --dry-run   # report only
    python data/fix_corrupt_labels.py             # fix in place
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABELS_DIRS = {
    "train": ROOT / "datasets/indicdlp_subset/labels/train",
    "val":   ROOT / "datasets/indicdlp_subset/labels/val",
}

# Threshold: values above this are unrecoverable (raw pixel space)
UNRECOVERABLE_THRESHOLD = 10.0
# After clamping, drop boxes with w or h below this (degenerate)
MIN_WH = 0.001


def fix_line(line: str) -> tuple[str | None, str]:
    """
    Returns (fixed_line_or_None, status).
    status is one of: 'ok', 'clamped', 'dropped_unrecoverable', 'dropped_degenerate'
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None, "dropped_bad_format"

    try:
        cls = int(parts[0])
        vals = [float(p) for p in parts[1:]]
    except ValueError:
        return None, "dropped_parse_error"

    cx, cy, w, h = vals

    # Check if any value is in the valid range already
    if all(0.0 <= v <= 1.0 for v in [cx, cy, w, h]) and w > 0 and h > 0:
        return line.strip(), "ok"

    # Check if unrecoverable (values wildly out of range)
    if any(abs(v) > UNRECOVERABLE_THRESHOLD for v in [cx, cy, w, h]):
        return None, "dropped_unrecoverable"

    # Moderately out of range — clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    if w < MIN_WH or h < MIN_WH:
        return None, "dropped_degenerate"

    fixed = f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    return fixed, "clamped"


def process_dir(label_dir: Path, backup_dir: Path, dry_run: bool) -> dict:
    stats = {
        "files_processed": 0,
        "files_changed": 0,
        "lines_ok": 0,
        "lines_clamped": 0,
        "lines_dropped_unrecoverable": 0,
        "lines_dropped_degenerate": 0,
        "lines_dropped_other": 0,
    }

    label_files = sorted(label_dir.glob("*.txt"))
    for lf in label_files:
        lines = lf.read_text(encoding="utf-8").splitlines()
        new_lines = []
        changed = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            fixed, status = fix_line(line)
            if status == "ok":
                new_lines.append(fixed)
                stats["lines_ok"] += 1
            elif status == "clamped":
                new_lines.append(fixed)
                stats["lines_clamped"] += 1
                changed = True
            elif status == "dropped_unrecoverable":
                stats["lines_dropped_unrecoverable"] += 1
                changed = True
            elif status == "dropped_degenerate":
                stats["lines_dropped_degenerate"] += 1
                changed = True
            else:
                stats["lines_dropped_other"] += 1
                changed = True

        stats["files_processed"] += 1

        if changed:
            stats["files_changed"] += 1
            if not dry_run:
                # Backup original (Rule 6: never delete)
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / lf.name
                if not backup_path.exists():
                    shutil.copy2(lf, backup_path)
                # Write fixed file
                lf.write_text("\n".join(new_lines) + ("\n" if new_lines else ""),
                              encoding="utf-8")

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be changed without writing anything")
    args = ap.parse_args()

    mode = "DRY RUN — no files written" if args.dry_run else "FIXING FILES"
    print(f"{'='*60}")
    print(f"fix_corrupt_labels.py  [{mode}]")
    print(f"{'='*60}\n")

    total = {
        "files_processed": 0, "files_changed": 0,
        "lines_ok": 0, "lines_clamped": 0,
        "lines_dropped_unrecoverable": 0,
        "lines_dropped_degenerate": 0, "lines_dropped_other": 0,
    }

    for split, label_dir in LABELS_DIRS.items():
        backup_dir = label_dir.parent / f"{split}_corrupt_backup"
        print(f"[{split}]  {label_dir}")
        if not args.dry_run:
            print(f"  backup → {backup_dir}")

        stats = process_dir(label_dir, backup_dir, dry_run=args.dry_run)

        print(f"  files processed  : {stats['files_processed']}")
        print(f"  files changed    : {stats['files_changed']}")
        print(f"  lines ok         : {stats['lines_ok']}")
        print(f"  lines clamped    : {stats['lines_clamped']}")
        print(f"  lines dropped (unrecoverable) : {stats['lines_dropped_unrecoverable']}")
        print(f"  lines dropped (degenerate)    : {stats['lines_dropped_degenerate']}")
        print(f"  lines dropped (other)         : {stats['lines_dropped_other']}")
        print()

        for k in total:
            total[k] += stats[k]

    print(f"{'='*60}")
    print(f"TOTAL")
    print(f"  files changed    : {total['files_changed']}")
    print(f"  lines clamped    : {total['lines_clamped']}")
    print(f"  lines dropped    : {total['lines_dropped_unrecoverable'] + total['lines_dropped_degenerate'] + total['lines_dropped_other']}")
    print(f"  lines kept ok    : {total['lines_ok'] + total['lines_clamped']}")
    if args.dry_run:
        print(f"\nRun without --dry-run to apply changes.")
    else:
        print(f"\nOriginal files backed up to labels/train_corrupt_backup/ and labels/val_corrupt_backup/")
        print(f"Nothing was deleted (Rule 6).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
