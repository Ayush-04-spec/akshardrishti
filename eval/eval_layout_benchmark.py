#!/usr/bin/env python3
"""Evaluate the layout detector on the 24-page annotated benchmark.

Computes mAP50, mAP50-95, precision, recall overall and per class.
Compares model predictions against benchmark/gt_layout/ ground truth.

No training. No GPU required (uses CPU if GPU unavailable).
Outputs go to results/layout_benchmark/.

Usage
-----
    python eval/eval_layout_benchmark.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH_DIR   = ROOT / "benchmark"
IMAGES_DIR  = BENCH_DIR / "images"
GT_DIR      = BENCH_DIR / "gt_layout"
OUT_DIR     = ROOT / "results" / "layout_benchmark"
WEIGHTS     = ROOT / "weights" / "layout" / "yolo11l_indicdlp.pt"
EXPECTED_SHA = "C7B4F755A132E10F66656F575C9159493F660641B096B2774A8C33CF9DB999F8"

CLASS_NAMES = ["title","text","list","table","figure","caption",
               "header","footer","page_number","stamp_seal","signature"]

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # 0.50 to 0.95


# ─────────────────────────── helpers ─────────────────────────────────────────

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_yolo_labels(txt_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Load YOLO format labels → list of {cls, x1, y1, x2, y2}."""
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """VOC 11-point interpolated AP."""
    if not precisions:
        return 0.0
    ap = 0.0
    for thr in np.arange(0.0, 1.1, 0.1):
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += max(p_at_r) if p_at_r else 0.0
    return ap / 11.0


def eval_class_at_iou(pred_boxes: list[dict], gt_boxes: list[dict],
                      cls: int, iou_thr: float) -> tuple[float, float, int, int]:
    """Returns precision, recall, n_pred, n_gt for one class at one IoU."""
    preds = sorted([b for b in pred_boxes if b["cls"] == cls],
                   key=lambda b: b.get("conf", 1.0), reverse=True)
    gts   = [b for b in gt_boxes if b["cls"] == cls]
    n_gt  = len(gts)
    n_pred = len(preds)
    if n_gt == 0:
        return 0.0, 0.0, n_pred, 0
    if n_pred == 0:
        return 0.0, 0.0, 0, n_gt

    matched = [False] * n_gt
    tp = fp = 0
    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thr:
            matched[best_j] = True
            tp += 1
        else:
            fp += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / n_gt if n_gt > 0 else 0.0
    return prec, rec, n_pred, n_gt


# ─────────────────────────────────────────── main ────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "layout_benchmark_results.json"

    # SHA-256 check
    print("Verifying weights...")
    actual_sha = sha256_of(WEIGHTS)
    assert actual_sha == EXPECTED_SHA, f"SHA mismatch: {actual_sha}"
    print(f"✅ Weights verified: {WEIGHTS.name}")

    # Find benchmark images
    images = sorted(IMAGES_DIR.glob("BM_*.png"))
    print(f"Benchmark pages: {len(images)}")

    # Check GT files
    gt_missing = [img for img in images
                  if not (GT_DIR / f"{img.stem}.txt").exists()]
    if gt_missing:
        print(f"WARNING: Missing GT for: {[f.name for f in gt_missing]}")

    # Run YOLO predictions
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    # Store per-image predictions and GT
    all_preds: dict[str, list[dict]] = {}
    all_gts:   dict[str, list[dict]] = {}

    print("\nRunning predictions...")
    for img_path in images:
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # Predict
        result = model.predict(source=str(img_path), imgsz=1024,
                               conf=0.25, iou=0.45,
                               device=device, verbose=False)[0]
        preds = []
        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds.append({"cls": cls, "conf": conf,
                          "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        all_preds[img_path.stem] = preds

        # Load GT
        gts = load_yolo_labels(GT_DIR / f"{img_path.stem}.txt", W, H)
        all_gts[img_path.stem] = gts

        print(f"  {img_path.name}: pred={len(preds)} gt={len(gts)}")

    # Compute mAP per class
    print("\nComputing metrics...")
    class_results = {}
    maps_50    = []
    maps_5095  = []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        # Gather all preds and gts for this class across all images
        all_p = []
        all_g = []
        for stem in [img.stem for img in images]:
            for b in all_preds.get(stem, []):
                if b["cls"] == cls_idx:
                    all_p.append(b)
            for b in all_gts.get(stem, []):
                if b["cls"] == cls_idx:
                    all_g.append(b)

        n_gt = len(all_g)
        if n_gt == 0:
            class_results[cls_name] = {
                "n_gt": 0, "n_pred": len(all_p),
                "AP50": None, "AP50-95": None,
                "note": "no GT instances — detector never trained on this class"
            }
            continue

        # AP at IoU=0.5
        prec50, rec50, n_pred, _ = eval_class_at_iou(
            [b for stem in all_preds for b in all_preds[stem]],
            [b for stem in all_gts   for b in all_gts[stem]],
            cls_idx, 0.5
        )
        ap50 = prec50 * rec50  # single operating point

        # AP50-95
        ap_vals = []
        for thr in IOU_THRESHOLDS:
            p, r, _, _ = eval_class_at_iou(
                [b for stem in all_preds for b in all_preds[stem]],
                [b for stem in all_gts   for b in all_gts[stem]],
                cls_idx, float(thr)
            )
            ap_vals.append(p * r)
        ap5095 = float(np.mean(ap_vals))

        class_results[cls_name] = {
            "n_gt": n_gt, "n_pred": n_pred,
            "precision_at_50": round(prec50, 4),
            "recall_at_50": round(rec50, 4),
            "AP50": round(ap50, 4),
            "AP50-95": round(ap5095, 4),
        }
        maps_50.append(ap50)
        maps_5095.append(ap5095)

    # Overall
    map50   = float(np.mean(maps_50))   if maps_50   else 0.0
    map5095 = float(np.mean(maps_5095)) if maps_5095 else 0.0

    # Overall precision and recall at IoU=0.5
    total_tp = total_fp = total_fn = 0
    for stem in [img.stem for img in images]:
        preds_img = all_preds.get(stem, [])
        gts_img   = all_gts.get(stem, [])
        matched_gt = set()
        for p in preds_img:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts_img):
                if j in matched_gt:
                    continue
                v = iou(p, g)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= 0.5:
                matched_gt.add(best_j)
                total_tp += 1
            else:
                total_fp += 1
        total_fn += len(gts_img) - len(matched_gt)

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # Print results
    print("\n" + "=" * 60)
    print("LAYOUT BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Pages evaluated : {len(images)}")
    print(f"  mAP@50          : {map50:.4f}")
    print(f"  mAP@50-95       : {map5095:.4f}")
    print(f"  Precision@50    : {overall_prec:.4f}")
    print(f"  Recall@50       : {overall_rec:.4f}")
    print()
    print(f"  {'Class':<14} {'GT':>5} {'Pred':>5} {'P@50':>6} {'R@50':>6} {'AP50':>6} {'AP50-95':>8}")
    print("  " + "-" * 60)
    for cls_name, r in class_results.items():
        if r.get("AP50") is None:
            print(f"  {cls_name:<14} {'0':>5} {r['n_pred']:>5}  {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>8}  (no GT)")
        else:
            print(f"  {cls_name:<14} {r['n_gt']:>5} {r['n_pred']:>5} "
                  f"{r['precision_at_50']:>6.4f} {r['recall_at_50']:>6.4f} "
                  f"{r['AP50']:>6.4f} {r['AP50-95']:>8.4f}")
    print("=" * 60)

    # Save JSON
    result = {
        "git_commit":   git_commit_hash(),
        "date":         str(date.today()),
        "weights_sha":  actual_sha,
        "n_pages":      len(images),
        "device":       str(device),
        "overall": {
            "mAP50":      round(map50, 4),
            "mAP50-95":   round(map5095, 4),
            "precision":  round(overall_prec, 4),
            "recall":     round(overall_rec, 4),
        },
        "per_class": class_results,
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
