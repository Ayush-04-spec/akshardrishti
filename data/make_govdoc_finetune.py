"""
Create a fine-tuning dataset for the CRNN from benchmark pages.
Now includes data augmentation to multiply 500 crops → ~2500 effective samples.

Output:
  datasets/govdoc_finetune/images/     crop images (original + augmented)
  datasets/govdoc_finetune/index.csv   (image_path, text, language, split)
"""
from __future__ import annotations
import csv, sys, unicodedata, random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.layout.lines import crop, segment_lines_projection
from akshardrishti.schema import BBox

BENCHMARK_IMAGES    = ROOT / "benchmark" / "images"
BENCHMARK_GT_LAYOUT = ROOT / "benchmark" / "gt_layout"
BENCHMARK_GT_TEXT   = ROOT / "benchmark" / "gt_text"
OUT_DIR  = ROOT / "datasets" / "govdoc_finetune"
OUT_IMGS = OUT_DIR / "images"
OUT_CSV  = OUT_DIR / "index.csv"

TEXT_BEARING = {0, 1, 2, 5, 6, 7, 8}
CLASS_NAMES  = {0:"title",1:"text",2:"list",3:"table",4:"figure",5:"caption",
                6:"header",7:"footer",8:"page_number",9:"stamp_seal",10:"signature"}

random.seed(42)

# ── augmentation ──────────────────────────────────────────────────────────────

def augment(img: np.ndarray, n: int = 4) -> list[np.ndarray]:
    """Return n augmented versions of a line crop."""
    results = []
    h, w = img.shape[:2]
    for i in range(n):
        aug = img.copy()
        # 1. Brightness/contrast jitter
        alpha = random.uniform(0.8, 1.2)
        beta  = random.randint(-15, 15)
        aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        # 2. Small rotation (±2 degrees)
        angle = random.uniform(-2, 2)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        # 3. Slight gaussian blur (every other aug)
        if i % 2 == 0:
            aug = cv2.GaussianBlur(aug, (3, 3), 0.5)
        # 4. Salt & pepper noise (small amount)
        noise_mask = np.random.random(aug.shape) < 0.01
        aug[noise_mask] = np.random.choice([0, 255], size=noise_mask.sum())
        results.append(aug)
    return results


# ── helpers ───────────────────────────────────────────────────────────────────

def load_gt_boxes(txt_path, img_w, img_h):
    boxes = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5: continue
        cid = int(parts[0])
        cx,cy,bw,bh = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
        x1=(cx-bw/2)*img_w; y1=(cy-bh/2)*img_h
        x2=(cx+bw/2)*img_w; y2=(cy+bh/2)*img_h
        boxes.append({"class_id":cid,"class":CLASS_NAMES[cid],
                      "bbox":BBox(x1=x1,y1=y1,x2=x2,y2=y2)})
    boxes.sort(key=lambda b:(b["bbox"].y1,b["bbox"].x1))
    return boxes

def split_gt_into_lines(gt_text):
    return [unicodedata.normalize("NFC",l).strip()
            for l in gt_text.splitlines()
            if unicodedata.normalize("NFC",l).strip()]

def assign_lines(gt_lines, regions, region_imgs):
    line_counts = []
    for reg, reg_img in zip(regions, region_imgs):
        if reg_img is None or reg_img.size == 0:
            line_counts.append(0); continue
        gray = cv2.cvtColor(reg_img,cv2.COLOR_BGR2GRAY) if reg_img.ndim==3 else reg_img
        lines = segment_lines_projection(gray, min_line_height=8, smoothing=3)
        line_counts.append(max(1, len(lines)))
    total_det = sum(line_counts); total_gt = len(gt_lines)
    assignments = []; gt_idx = 0
    for i, count in enumerate(line_counts):
        n = max(1, round(count/total_det*total_gt)) if total_det>0 else max(1,total_gt//max(1,len(regions)))
        if i == len(line_counts)-1: n = total_gt - gt_idx
        assignments.append(gt_lines[gt_idx:gt_idx+n])
        gt_idx = min(gt_idx+n, total_gt)
    return assignments

def extract_line_crops(region_img, assigned_lines, out_dir, prefix):
    if region_img is None or region_img.size == 0: return []
    gray = cv2.cvtColor(region_img,cv2.COLOR_BGR2GRAY) if region_img.ndim==3 else region_img
    line_boxes = segment_lines_projection(gray, min_line_height=8, smoothing=3)
    if not line_boxes or not assigned_lines: return []
    pairs = []
    for i,(lb,text) in enumerate(zip(line_boxes[:len(assigned_lines)], assigned_lines)):
        if not text.strip(): continue
        lc = crop(region_img, lb)
        if lc.size == 0: continue
        h,w = lc.shape[:2]
        if w<20 or h<8: continue
        # Save original
        orig_path = out_dir / f"{prefix}_line{i:03d}.png"
        cv2.imwrite(str(orig_path), lc)
        pairs.append((orig_path, text))
        # Save augmented versions
        for j, aug_img in enumerate(augment(lc, n=4)):
            aug_path = out_dir / f"{prefix}_line{i:03d}_aug{j}.png"
            cv2.imwrite(str(aug_path), aug_img)
            pairs.append((aug_path, text))
    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    all_rows = []
    gt_pages = sorted(BENCHMARK_GT_TEXT.glob("BM_*.txt"))
    print(f"Pages with GT text: {len(gt_pages)}")

    for gt_txt in gt_pages:
        page_id = gt_txt.stem
        img_path    = BENCHMARK_IMAGES    / f"{page_id}.png"
        layout_path = BENCHMARK_GT_LAYOUT / f"{page_id}.txt"
        if not img_path.exists() or not layout_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        gt_lines  = split_gt_into_lines(gt_txt.read_text(encoding="utf-8"))
        all_boxes = load_gt_boxes(layout_path, w, h)
        text_boxes = [b for b in all_boxes if b["class_id"] in TEXT_BEARING]
        if not text_boxes: continue

        region_crops = [crop(img, b["bbox"]) for b in text_boxes]
        assignments  = assign_lines(gt_lines, text_boxes, region_crops)

        page_pairs = 0
        for reg_idx,(box,reg_crop,assigned) in enumerate(zip(text_boxes,region_crops,assignments)):
            prefix = f"{page_id}_r{reg_idx:02d}_{box['class']}"
            pairs  = extract_line_crops(reg_crop, assigned, OUT_IMGS, prefix)
            for crop_path, text in pairs:
                all_rows.append({"image_path":str(crop_path),"text":text,
                                 "language":"marathi","split":"train"})
                page_pairs += 1
        print(f"  {page_id}: {page_pairs} pairs (orig+aug)")

    print(f"\nTotal pairs (with augmentation): {len(all_rows)}")

    with open(OUT_CSV,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path","text","language","split"])
        w.writeheader(); w.writerows(all_rows)

    print(f"CSV: {OUT_CSV}")
    empty = sum(1 for r in all_rows if not r["text"].strip())
    print(f"Empty text: {empty}  Usable: {len(all_rows)-empty}")

if __name__ == "__main__":
    main()
