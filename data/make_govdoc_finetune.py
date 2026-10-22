"""
Create a fine-tuning dataset for the CRNN from benchmark pages.

Strategy
--------
For each benchmark page that has a GT transcription:
1. Load the image and GT layout boxes
2. Crop each text-bearing region (title, text, list, header, footer)
3. Run line segmentation on each region crop
4. Assign GT text lines to line crops in reading order
5. Save (image_path, text, language, split) rows to a CSV

The resulting CSV is appended to the Mozhi training split for fine-tuning.
Line crops are saved to datasets/govdoc_finetune/images/

Output
------
  datasets/govdoc_finetune/images/     crop images
  datasets/govdoc_finetune/index.csv   (image_path, text, language, split)
"""
from __future__ import annotations

import csv, sys, unicodedata
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.layout.lines import crop, offset_box, segment_lines_projection
from akshardrishti.schema import BBox

# ── config ────────────────────────────────────────────────────────────────────

BENCHMARK_IMAGES = ROOT / "benchmark" / "images"
BENCHMARK_GT_LAYOUT = ROOT / "benchmark" / "gt_layout"
BENCHMARK_GT_TEXT   = ROOT / "benchmark" / "gt_text"
OUT_DIR  = ROOT / "datasets" / "govdoc_finetune"
OUT_IMGS = OUT_DIR / "images"
OUT_CSV  = OUT_DIR / "index.csv"

# Only text-bearing classes (skip figure, stamp_seal, signature, table)
TEXT_BEARING = {0, 1, 2, 5, 6, 7, 8}  # title, text, list, caption, header, footer, page_number

CLASS_NAMES = {0:"title",1:"text",2:"list",3:"table",4:"figure",5:"caption",
               6:"header",7:"footer",8:"page_number",9:"stamp_seal",10:"signature"}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_gt_boxes(txt_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Load YOLO normalized boxes, convert to pixel xyxy, sort top→bottom."""
    boxes = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5: continue
        cid = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - bw/2) * img_w
        y1 = (cy - bh/2) * img_h
        x2 = (cx + bw/2) * img_w
        y2 = (cy + bh/2) * img_h
        boxes.append({"class_id": cid, "class": CLASS_NAMES[cid],
                       "bbox": BBox(x1=x1, y1=y1, x2=x2, y2=y2)})
    # Sort reading order: top to bottom, left to right
    boxes.sort(key=lambda b: (b["bbox"].y1, b["bbox"].x1))
    return boxes


def clean_line(text: str) -> str:
    """NFC normalize and strip."""
    return unicodedata.normalize("NFC", text).strip()


def split_gt_into_lines(gt_text: str) -> list[str]:
    """Split page GT text into individual lines, filtered to non-empty."""
    return [clean_line(l) for l in gt_text.splitlines() if clean_line(l)]


def assign_lines_to_regions(
    gt_lines: list[str],
    regions: list[dict],
    region_imgs: list[np.ndarray],
) -> list[list[str]]:
    """
    Assign GT text lines to regions proportionally by the number of
    line segments detected in each region's crop.

    This is a heuristic: we don't know exactly which GT lines belong to
    which region, so we split the GT lines proportionally.
    """
    # Count detected lines per region
    line_counts = []
    for reg, reg_img in zip(regions, region_imgs):
        if reg_img is None or reg_img.size == 0:
            line_counts.append(0)
            continue
        gray = cv2.cvtColor(reg_img, cv2.COLOR_BGR2GRAY) if reg_img.ndim == 3 else reg_img
        lines = segment_lines_projection(gray, min_line_height=8, smoothing=3)
        line_counts.append(max(1, len(lines)))

    total_detected = sum(line_counts)
    total_gt = len(gt_lines)

    # Assign GT lines proportionally
    assignments: list[list[str]] = []
    gt_idx = 0
    for i, count in enumerate(line_counts):
        # Proportion of GT lines this region gets
        if total_detected > 0:
            n = max(1, round(count / total_detected * total_gt))
        else:
            n = max(1, total_gt // max(1, len(regions)))
        # Last region gets all remaining lines
        if i == len(line_counts) - 1:
            n = total_gt - gt_idx
        assigned = gt_lines[gt_idx: gt_idx + n]
        assignments.append(assigned)
        gt_idx = min(gt_idx + n, total_gt)

    return assignments


def extract_line_crops(
    region_img: np.ndarray,
    assigned_lines: list[str],
    out_dir: Path,
    prefix: str,
) -> list[tuple[Path, str]]:
    """
    Segment region into line crops, pair with assigned GT lines.
    Returns list of (crop_path, text).
    """
    if region_img is None or region_img.size == 0:
        return []

    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY) if region_img.ndim == 3 else region_img
    line_boxes = segment_lines_projection(gray, min_line_height=8, smoothing=3)

    if not line_boxes or not assigned_lines:
        return []

    pairs = []
    n = min(len(line_boxes), len(assigned_lines))

    for i, (lb, text) in enumerate(zip(line_boxes[:n], assigned_lines[:n])):
        if not text.strip():
            continue
        line_crop = crop(region_img, lb)
        if line_crop.size == 0:
            continue
        # Skip very narrow or very short crops
        h, w = line_crop.shape[:2]
        if w < 20 or h < 8:
            continue

        crop_path = out_dir / f"{prefix}_line{i:03d}.png"
        cv2.imwrite(str(crop_path), line_crop)
        pairs.append((crop_path, text))

    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []  # (image_path, text, language, split)

    gt_pages = sorted(BENCHMARK_GT_TEXT.glob("BM_*.txt"))
    print(f"Pages with GT text: {len(gt_pages)}")

    for gt_txt in gt_pages:
        page_id = gt_txt.stem  # e.g. BM_001
        img_path = BENCHMARK_IMAGES / f"{page_id}.png"
        layout_path = BENCHMARK_GT_LAYOUT / f"{page_id}.txt"

        if not img_path.exists():
            print(f"  {page_id}: image missing, skipping")
            continue
        if not layout_path.exists():
            print(f"  {page_id}: layout GT missing, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Processing {page_id}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read image, skipping")
            continue
        h, w = img.shape[:2]

        # Load GT text lines
        gt_text = gt_txt.read_text(encoding="utf-8")
        gt_lines = split_gt_into_lines(gt_text)
        print(f"  GT lines: {len(gt_lines)}")

        # Load GT layout boxes (text-bearing only)
        all_boxes = load_gt_boxes(layout_path, w, h)
        text_boxes = [b for b in all_boxes if b["class_id"] in TEXT_BEARING]
        print(f"  Text-bearing regions: {len(text_boxes)} of {len(all_boxes)}")

        if not text_boxes:
            print(f"  No text regions, skipping")
            continue

        # Crop each region
        region_crops = []
        for b in text_boxes:
            reg_crop = crop(img, b["bbox"])
            region_crops.append(reg_crop)

        # Assign GT lines to regions proportionally
        assignments = assign_lines_to_regions(gt_lines, text_boxes, region_crops)

        # Extract line crops from each region
        page_pairs = 0
        for reg_idx, (box, reg_crop, assigned) in enumerate(
            zip(text_boxes, region_crops, assignments)
        ):
            prefix = f"{page_id}_r{reg_idx:02d}_{box['class']}"
            pairs = extract_line_crops(reg_crop, assigned, OUT_IMGS, prefix)

            for crop_path, text in pairs:
                all_rows.append({
                    "image_path": str(crop_path),
                    "text": text,
                    "language": "marathi",  # all benchmark pages are Marathi govt docs
                    "split": "train",
                })
                page_pairs += 1

        print(f"  Line crop pairs created: {page_pairs}")

    # Write CSV
    print(f"\n{'='*50}")
    print(f"Total line crop pairs: {len(all_rows)}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path","text","language","split"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"CSV written: {OUT_CSV}")
    print(f"Images in:   {OUT_IMGS}")

    # Show sample rows
    print(f"\nSample rows:")
    for row in all_rows[:5]:
        print(f"  {Path(row['image_path']).name}  ->  {row['text'][:50]!r}")

    # Quality check
    empty = sum(1 for r in all_rows if not r["text"].strip())
    short = sum(1 for r in all_rows if len(r["text"].strip()) < 3)
    print(f"\nQuality check:")
    print(f"  Empty text: {empty}")
    print(f"  Very short (<3 chars): {short}")
    print(f"  Usable pairs: {len(all_rows) - empty}")


if __name__ == "__main__":
    main()
