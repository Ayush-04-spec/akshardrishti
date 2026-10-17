#!/usr/bin/env python3
"""Pre-label benchmark pages with the layout detector, for annotators to correct.

For every benchmark/images/*.png this writes a YOLO-format label file to
benchmark/gt_layout_prelabeled/<stem>.txt (class cx cy w h, normalized).
Annotators correct these and then place the result in benchmark/gt_layout/.
This script never writes to benchmark/gt_layout/.

Rules this script follows:
  * Rule 3: weights are only read. A missing weights file is reported and the
    script stops; nothing is created or downloaded.
  * Rule 4: the weights are loaded by ultralytics (which uses
    weights_only=False internally), so the file's SHA-256 is verified first.
  * Rule 5: inference only, one image per call (batch 1).
  * Rule 6: nothing is deleted or overwritten. A page whose label file already
    exists is skipped, so annotator corrections are never clobbered.
  * Outputs go under benchmark/ only. No metric numbers are embedded.
  * torch is not imported here; ultralytics is imported only after the
    pre-flight checks pass.

Usage (from the repo root, inside the venv):
    python eval/prelabel_benchmark.py --dry-run
    python eval/prelabel_benchmark.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmark"
IMAGES_DIR = BENCH_DIR / "images"
OUT_DIR = BENCH_DIR / "gt_layout_prelabeled"
META_DIR = BENCH_DIR / "prelabel_meta"
WEIGHTS = REPO_ROOT / "weights" / "layout" / "yolo11l_indicdlp.pt"

# SHA-256 recorded for the shipped layout checkpoint.
EXPECTED_SHA256 = (
    "5A927BAF51EE9EF3DDEA1DFFDAE954C73869A17B0614836F8856AA77B6AD4AED"
)

EXPECTED_CLASSES = [
    "title", "text", "list", "table", "figure", "caption", "header",
    "footer", "page_number", "stamp_seal", "signature",
]
# Classes the detector was never trained on; annotators must add them by hand.
MANUAL_ONLY = ("stamp_seal", "signature")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None,
                        help="passed to ultralytics (for example 0 or cpu); "
                             "default lets ultralytics choose")
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256,
                        help="override only if the weights were deliberately "
                             "replaced and re-hashed")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the checks and list pages; load no model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    images = sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        print(f"No PNG files in {IMAGES_DIR}. "
              "Run eval/select_benchmark_pages.py first.")
        return 2

    # Rule 3: report missing weights and stop.
    if not WEIGHTS.is_file():
        print(f"Layout weights not found: {WEIGHTS}\n"
              "Weights are read-only here; nothing was created. Stopping.")
        return 2

    # Rule 4: verify the hash before ultralytics unpickles the checkpoint.
    actual = sha256_of(WEIGHTS)
    if actual.upper() != args.expected_sha256.upper():
        print("SHA-256 mismatch for the layout weights; refusing to load.\n"
              f"  expected: {args.expected_sha256.upper()}\n"
              f"  actual:   {actual.upper()}")
        return 2
    print(f"Weights hash verified: {WEIGHTS.name}")

    todo = [p for p in images if not (OUT_DIR / f"{p.stem}.txt").exists()]
    skipped = [p for p in images if p not in todo]
    print(f"Pages found: {len(images)}  to pre-label: {len(todo)}  "
          f"already have a label file (skipped): {len(skipped)}")

    if args.dry_run:
        for page in todo:
            print(f"  would pre-label {page.name}")
        print("Dry run: no model loaded, nothing written.")
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    from ultralytics import YOLO  # imported late on purpose

    model = YOLO(str(WEIGHTS))
    names = {int(k): v for k, v in dict(model.names).items()}
    if [names.get(i) for i in range(len(EXPECTED_CLASSES))] != EXPECTED_CLASSES \
            or len(names) != len(EXPECTED_CLASSES):
        print("WARNING: the model's class names differ from the expected "
              f"11-class list.\n  model: {names}\n  Check before annotating.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    classes_file = OUT_DIR / "classes.txt"
    if not classes_file.exists():
        with open(classes_file, "x", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(names[i] for i in sorted(names)) + "\n")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detections_csv = META_DIR / f"detections_{stamp}.csv"

    predict_kwargs = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          verbose=False, save=False)
    if args.device is not None:
        predict_kwargs["device"] = args.device

    per_page: list[tuple[str, int, list[str]]] = []
    class_counts: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []

    with open(detections_csv, "x", newline="", encoding="utf-8") as det_fh:
        det_writer = csv.writer(det_fh)
        det_writer.writerow(["image", "class_id", "class_name", "conf",
                             "cx", "cy", "w", "h"])
        for page in todo:
            label_path = OUT_DIR / f"{page.stem}.txt"
            try:
                # One image per call: batch 1, well inside the VRAM limit.
                result = model.predict(source=str(page), **predict_kwargs)[0]
                cls_ids = result.boxes.cls.tolist()
                confs = result.boxes.conf.tolist()
                boxes = result.boxes.xywhn.tolist()
            except Exception as exc:
                failures.append((page.name, repr(exc)))
                print(f"  {page.name}: FAILED ({exc!r})")
                continue

            lines: list[str] = []
            found: list[str] = []
            for cid, conf, (cx, cy, w, h) in zip(cls_ids, confs, boxes):
                cid = int(cid)
                cx, cy, w, h = (clamp01(v) for v in (cx, cy, w, h))
                if w <= 0.0 or h <= 0.0:
                    continue  # degenerate box, never write it
                name = names.get(cid, str(cid))
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                det_writer.writerow([page.name, cid, name, f"{conf:.4f}",
                                     f"{cx:.6f}", f"{cy:.6f}",
                                     f"{w:.6f}", f"{h:.6f}"])
                found.append(name)

            with open(label_path, "x", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))

            class_counts.update(found)
            per_page.append((page.name, len(found), sorted(set(found))))
            print(f"  {page.name}: boxes={len(found):3d}  "
                  f"classes={','.join(sorted(set(found))) or '-'}")

    print("\n=== Summary ===")
    print(f"Pre-labeled: {len(per_page)}  skipped (label exists): "
          f"{len(skipped)}  failed: {len(failures)}")
    print(f"Label files: {OUT_DIR}")
    print(f"Detections with confidences: {detections_csv}")
    if per_page:
        total_boxes = sum(n for _, n, _ in per_page)
        print(f"Total boxes: {total_boxes}")
        print("Boxes per class:")
        for name in [names[i] for i in sorted(names)]:
            print(f"  {name:12s} {class_counts.get(name, 0)}")
        zero = [n for n, count, _ in per_page if count == 0]
        print("Pages with 0 detections: "
              + (", ".join(zero) if zero else "none"))
        never = [c for c in MANUAL_ONLY if class_counts.get(c, 0) == 0]
        if never:
            print("No automatic boxes for: " + ", ".join(never)
                  + ". Annotators must add these by hand.")
    if failures:
        print("Failed pages (no label file written; try --device cpu):")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
