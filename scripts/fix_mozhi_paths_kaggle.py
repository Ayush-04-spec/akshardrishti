#!/usr/bin/env python3
"""Fix absolute Windows paths in Mozhi index CSVs for the Kaggle environment.

The index_*.csv files were built on a Windows machine and contain paths like:
    D:\\AksharDrishti_1\\aksharDrishti\\datasets\\mozhi_raw\\hindi\\train\\images\\foo.jpeg

On Kaggle the dataset is mounted at:
    /kaggle/input/akshardrishti-mozhi-raw/

This script rewrites the image_path column so each entry resolves correctly
under that mount point, then writes the fixed CSVs into a writable output
directory (/kaggle/working/mozhi_fixed/ by default).

Usage (from the notebook cell)
-------------------------------
    !python /kaggle/working/aksharDrishti/scripts/fix_mozhi_paths_kaggle.py \\
        --input-dir  /kaggle/input/akshardrishti-mozhi-raw \\
        --output-dir /kaggle/working/mozhi_fixed

The train script then reads from /kaggle/working/mozhi_fixed/ instead of the
original /kaggle/input/ CSVs (which are read-only and have wrong paths anyway).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath


# ---------------------------------------------------------------------------
# Path rewriting
# ---------------------------------------------------------------------------

def extract_relative(raw_path: str) -> str:
    """Strip the machine-specific prefix and return the portable relative path.

    Handles both Windows (backslash) and already-posix paths.

    Examples
    --------
    Input:  D:\\AksharDrishti_1\\aksharDrishti\\datasets\\mozhi_raw\\hindi\\train\\images\\foo.jpeg
    Output: hindi/train/images/foo.jpeg

    Input:  /kaggle/input/akshardrishti-mozhi-raw/hindi/train/images/foo.jpeg
            (already fixed — no-op, just normalise slashes)
    Output: hindi/train/images/foo.jpeg
    """
    # Normalise to forward slashes for consistent processing
    normalised = raw_path.replace("\\", "/")

    # Anchor: everything from (and including) a language folder onwards.
    # Mozhi structure: <prefix>/hindi/... or <prefix>/marathi/...
    # We want the part starting at 'hindi' or 'marathi'.
    match = re.search(r"(?:^|/)((hindi|marathi)/.+)$", normalised)
    if match:
        return match.group(1)

    # Fallback: strip known mount-point prefixes if language anchor not found
    for prefix in (
        "/kaggle/input/akshardrishti-mozhi-raw/",
        "/kaggle/input/akshardrishti-mozhi-raw",
    ):
        if normalised.startswith(prefix):
            return normalised[len(prefix):].lstrip("/")

    # Last resort: return as-is (will fail os.path.exists check, making it visible)
    return normalised


def rewrite_csv(
    src: Path,
    dst: Path,
    input_dir: Path,
) -> tuple[int, int]:
    """Rewrite one CSV file. Returns (rows_written, rows_missing)."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    rows_missing = 0

    with open(src, encoding="utf-8", newline="") as fin, \
         open(dst, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        assert reader.fieldnames is not None, f"empty CSV: {src}"
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            rel = extract_relative(row["image_path"])
            new_path = str(input_dir / rel)
            row["image_path"] = new_path
            writer.writerow(row)
            rows_written += 1
            if not os.path.exists(new_path):
                rows_missing += 1

    return rows_written, rows_missing


# ---------------------------------------------------------------------------
# Spot-check
# ---------------------------------------------------------------------------

def spot_check(csv_path: Path, n: int = 5, seed: int = 42) -> list[dict]:
    """Return n random rows and whether their image files exist."""
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    results = []
    for row in sample:
        results.append({
            "image_path": row["image_path"],
            "text":       row.get("text", ""),
            "language":   row.get("language", ""),
            "exists":     os.path.exists(row["image_path"]),
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/kaggle/input/akshardrishti-mozhi-raw"),
        help="Mount point of the Kaggle dataset (read-only)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/mozhi_fixed"),
        help="Writable directory for the fixed CSVs",
    )
    p.add_argument(
        "--spot-check-n",
        type=int,
        default=5,
        help="Number of random rows to spot-check per CSV",
    )
    args = p.parse_args()

    if not args.input_dir.exists():
        sys.exit(
            f"\nERROR: input-dir does not exist: {args.input_dir}\n"
            "Did you attach 'akshardrishti-mozhi-raw' as a notebook input?"
        )

    splits = ["train", "val", "test"]
    csv_names = [f"index_{s}.csv" for s in splits]

    print(f"\n{'='*60}")
    print(" Mozhi CSV path rewriter")
    print(f"{'='*60}")
    print(f" Input dir  : {args.input_dir}")
    print(f" Output dir : {args.output_dir}")
    print()

    all_ok = True

    for csv_name in csv_names:
        src = args.input_dir / csv_name
        dst = args.output_dir / csv_name

        if not src.exists():
            print(f"  SKIP  {csv_name}  (not found in input-dir)")
            continue

        written, missing = rewrite_csv(src, dst, args.input_dir)
        status = "OK" if missing == 0 else f"WARN — {missing}/{written} images missing"
        print(f"  {csv_name:<22}  {written:>7,} rows  →  {dst}  [{status}]")
        if missing > 0:
            all_ok = False

    # ----------------------------------------------------------------- diff
    print(f"\n{'='*60}")
    print(" Sample diff (3 rows: before → after)")
    print(f"{'='*60}")

    src_train = args.input_dir / "index_train.csv"
    dst_train = args.output_dir / "index_train.csv"
    if src_train.exists() and dst_train.exists():
        with open(src_train, encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        with open(dst_train, encoding="utf-8") as f:
            new_rows = list(csv.DictReader(f))

        for i in [0, len(old_rows)//2, -1]:
            old_p = old_rows[i]["image_path"]
            new_p = new_rows[i]["image_path"]
            print(f"\n  Row {i if i >= 0 else len(old_rows)+i}:")
            print(f"    BEFORE: {old_p}")
            print(f"    AFTER : {new_p}")
            print(f"    EXISTS: {os.path.exists(new_p)}")

    # ---------------------------------------------------------------- spot
    print(f"\n{'='*60}")
    print(f" Spot-check: {args.spot_check_n} random rows from index_train.csv")
    print(f"{'='*60}")

    if dst_train.exists():
        checks = spot_check(dst_train, n=args.spot_check_n)
        all_exist = all(c["exists"] for c in checks)
        for c in checks:
            mark = "✓" if c["exists"] else "✗ MISSING"
            print(f"  [{mark}]  {c['language']:<8}  {c['text']:<20}  {c['image_path']}")
        print()
        if all_exist:
            print("  PASS — all sampled files exist on disk.")
        else:
            print("  FAIL — some files are missing. Check the dataset was uploaded correctly.")
            all_ok = False

    print(f"\n{'='*60}")
    print(f" Overall: {'PASS' if all_ok else 'FAIL — see warnings above'}")
    print(f"{'='*60}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
