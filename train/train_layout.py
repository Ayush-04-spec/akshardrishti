#!/usr/bin/env python3
"""Fine-tune YOLO11 (or YOLOv10 / DocLayout-YOLO) on the IndicDLP subset.

This produces model #1 -- the layout detector that the whole "layout-aware"
claim rests on.

Usage
-----
    python train/train_layout.py --data datasets/indicdlp_subset/data.yaml \
        --model yolo11l.pt --epochs 60 --batch 8 --imgsz 1024

    # baseline comparison: evaluate a released checkpoint without training
    python train/train_layout.py --eval-only --weights weights/indicdlp_yolov10x.pt \
        --data datasets/indicdlp_subset/data.yaml

Kaggle notes
------------
* 2x T4: use --device 0,1 --batch 16. Single T4: --batch 8.
* Sessions cap at 12h. 60 epochs on ~15k images at 1024px fits; 100 does not.
* Checkpoints land in runs/detect/<name>/weights/. Copy them to Drive BEFORE the
  session dies -- Kaggle output is not saved if the kernel is killed.

Licence note for the report: Ultralytics YOLO is AGPL-3.0. Fine for academic
work; mention it if the project is ever commercialised.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("datasets/indicdlp_subset/data.yaml"))
    p.add_argument("--model", default="yolo11l.pt", help="base checkpoint or .yaml")
    p.add_argument("--weights", type=Path, default=None, help="trained weights, for --eval-only")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--name", default="akshardrishti_layout")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--amp", action="store_true", default=True, help="mixed precision (keep on for T4)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--doclayout", action="store_true", help="use the doclayout_yolo package instead")
    p.add_argument("--wandb", default=None, help="W&B project name; omit to disable")
    p.add_argument("--backup-dir", type=Path, default=None, help="copy best.pt here after training (use your Drive)")
    args = p.parse_args()

    if not args.data.exists():
        raise SystemExit(f"data.yaml not found: {args.data}\nRun data/prepare_indicdlp.py first.")

    if args.wandb:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb
    else:
        import os

        os.environ.setdefault("WANDB_MODE", "disabled")

    if args.doclayout:
        from doclayout_yolo import YOLOv10 as ModelCls  # type: ignore
    else:
        from ultralytics import YOLO as ModelCls

    # ---------------------------------------------------------- eval only
    if args.eval_only:
        if not args.weights or not args.weights.exists():
            raise SystemExit("--eval-only needs --weights pointing at a checkpoint")
        model = ModelCls(str(args.weights))
        metrics = model.val(data=str(args.data), imgsz=args.imgsz, batch=args.batch, device=args.device)
        _report(metrics, args.weights.name)
        return

    # ------------------------------------------------------------- train
    model = ModelCls(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        amp=args.amp,
        resume=args.resume,
        save_period=1,          # per-epoch checkpoints -- Kaggle sessions die
        exist_ok=True,
        plots=True,
        # Conservative augmentation: documents are axis-aligned and upright.
        # Rotating or flipping a page teaches the model nothing it will ever see.
        degrees=2.0,
        translate=0.1,
        scale=0.3,
        shear=1.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.0,              # grayscale documents -- hue augmentation is noise
        hsv_s=0.2,
        hsv_v=0.3,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
    results = _report(metrics, args.name)

    out_json = Path(args.project) / args.name / "akshardrishti_metrics.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Metrics written to {out_json}")

    if args.backup_dir and best.exists():
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        dest = args.backup_dir / f"{args.name}_best.pt"
        shutil.copy2(best, dest)
        print(f"Backed up to {dest}")


def _report(metrics, label: str) -> dict:
    box = metrics.box
    results = {
        "model": label,
        "mAP50-95": float(box.map),
        "mAP50": float(box.map50),
        "mAP75": float(box.map75),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    print("\n" + "=" * 56)
    print(f"  {label}")
    print("=" * 56)
    print(f"  mAP@50-95 : {results['mAP50-95']:.4f}   <-- headline number for the paper")
    print(f"  mAP@50    : {results['mAP50']:.4f}")
    print(f"  mAP@75    : {results['mAP75']:.4f}")
    print(f"  precision : {results['precision']:.4f}")
    print(f"  recall    : {results['recall']:.4f}")

    names = getattr(metrics, "names", None)
    per_class = getattr(box, "maps", None)
    if names is not None and per_class is not None:
        print("\n  per-class mAP@50-95:")
        for idx, value in enumerate(per_class):
            name = names[idx] if isinstance(names, dict) else names[idx]
            print(f"    {str(name):<14} {float(value):.4f}")
            results.setdefault("per_class", {})[str(name)] = float(value)
    print("=" * 56)
    return results


if __name__ == "__main__":
    main()
