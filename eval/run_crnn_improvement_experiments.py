#!/usr/bin/env python3
"""Controlled, fixed-page CRNN experiments; all outputs stay under results/."""
from __future__ import annotations

import json
import logging
import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "crnn_improvement_experiments"


def _draw_segmentation(image: np.ndarray, boxes, destination: Path) -> None:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.as_xyxy())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 120, 255), 2)
        cv2.putText(canvas, str(index), (x1, max(12, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(destination), canvas)


def run() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="compare all matching BM_*.png/GT pairs")
    args = parser.parse_args()
    if args.full:
        page_ids = tuple(sorted(
            p.stem for p in (ROOT / "benchmark" / "gt_text").glob("BM_*.txt")
            if (ROOT / "benchmark" / "images" / f"{p.stem}.png").exists()
        ))
    else:
        page_ids = ("BM_001", "BM_015", "BM_021")

    import torch
    from akshardrishti.config import Config
    from akshardrishti.layout.lines import segment_lines
    from akshardrishti.postprocess.text_repair import normalize_for_scoring
    from eval.metrics import cer, wer
    from akshardrishti.pipeline import AksharDrishtiPipeline
    from akshardrishti.preprocess.enhance import preprocess_page, rotate_page_quadrants

    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load()
    cfg.set("layout.enabled", False)
    cfg.set("script_id.enabled", False)
    cfg.set("preprocess.orientation.enabled", False)
    pipeline = AksharDrishtiPipeline(cfg)
    router = pipeline.router
    process = psutil.Process()

    # A checkpoint is loaded lazily on first page; all variants reuse it.
    variants = [
        {"name": "A_legacy_capped", "mode": "legacy_cap", "segmentation": "projection", "rotation_cw": 0},
        {"name": "B_aspect_preserving", "mode": "aspect_preserving", "segmentation": "projection", "rotation_cw": 0},
        # Human-selected solely by visual inspection of the source page; GT text
        # was not consulted. This evaluates a correction, not an auto classifier.
        {"name": "C_visual_orientation_plus_aspect", "mode": "aspect_preserving", "segmentation": "projection", "rotation_cw": 90, "pages": ["BM_015"]},
        {"name": "D_columns_plus_aspect", "mode": "aspect_preserving", "segmentation": "columns_then_projection", "rotation_cw": 0},
    ]
    results: dict[str, dict] = {
        "metadata": {
            "pages": list(page_ids),
            "evaluation_scope": "all matching BM_*.png/GT pairs" if args.full else "fixed three-page pilot",
            "layout_enabled": False,
            "checkpoint": str(cfg.get("recognize.backends.crnn_mozhi.weights")),
            "device": router.deva.device,
            "orientation_osd_available": shutil.which("tesseract") is not None,
            "orientation_auto_status": (
                "available" if shutil.which("tesseract") else
                "blocked: pytesseract wrapper exists but tesseract executable/OSD data is unavailable"
            ),
            "orientation_visual_choice": "BM_015 clockwise 90 degrees, selected by visual page-image orientation only; no ground-truth transcription used",
            "text_scoring": "eval.metrics.cer and wer with their shared normalize_for_scoring path",
            "outputs": str(OUT),
        },
        "variants": {},
    }

    for variant in variants:
        if variant.get("pages") is not None:
            selected = set(variant["pages"])
        else:
            selected = set(page_ids)
        cfg.set("recognize.backends.crnn_mozhi.inference_mode", variant["mode"])
        cfg.set("recognize.line_segmentation.method", variant["segmentation"])
        pipeline.cfg = cfg
        pipeline.device = cfg.get("project.device", "auto")
        if router.deva._loaded:
            router.deva.inference_mode = variant["mode"]
        page_records = {}

        for page_id in page_ids:
            if page_id not in selected:
                continue
            image_path = ROOT / "benchmark" / "images" / f"{page_id}.png"
            gt_path = ROOT / "benchmark" / "gt_text" / f"{page_id}.txt"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or not gt_path.exists():
                page_records[page_id] = {"status": "unavailable", "reason": "page image or GT missing"}
                continue
            reference = gt_path.read_text(encoding="utf-8")
            rotation_cw = int(variant.get("rotation_cw", 0))
            work_image = image
            transform = None
            if rotation_cw:
                work_image, transform = rotate_page_quadrants(image, rotation_cw)
                cv2.imwrite(str(OUT / f"{page_id}_orientation_cw{rotation_cw}.png"), work_image)

            attempted = 0
            empty = 0
            failures = 0
            original_recognize = router.recognize

            def counted_recognize(*args, **kwargs):
                nonlocal attempted, empty, failures
                attempted += 1
                try:
                    result = original_recognize(*args, **kwargs)
                except Exception:
                    failures += 1
                    raise
                if not result.text.strip():
                    empty += 1
                return result

            router.recognize = counted_recognize
            rss_before = process.memory_info().rss
            cuda_available = torch.cuda.is_available() and str(router.deva.device).startswith("cuda")
            if cuda_available:
                torch.cuda.reset_peak_memory_stats(router.deva.device)
            started = time.perf_counter()
            try:
                page = pipeline.process_page(work_image, source_path=str(image_path), page_index=0)
                status = "ok"
                error = None
            except Exception as exc:
                page = None
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            router.recognize = original_recognize
            rss_after = process.memory_info().rss

            if page is None:
                page_records[page_id] = {
                    "status": status, "error": error, "attempted_crops": attempted,
                    "empty_crops": empty, "failed_crops": failures, "wall_time_s": elapsed,
                    "rss_before_mb": rss_before / (1024 ** 2), "rss_after_mb": rss_after / (1024 ** 2),
                }
                continue

            hypothesis = page.full_text()
            boxes = []
            preprocessed = preprocess_page(work_image, cfg.get("preprocess", {})).image
            h, w = preprocessed.shape[:2]
            if page.regions:
                region = page.regions[0]
                region_img = preprocessed[int(region.bbox.y1):int(region.bbox.y2), int(region.bbox.x1):int(region.bbox.x2)]
                ls = cfg.get("recognize.line_segmentation", {}) or {}
                boxes = segment_lines(
                    region_img, method=ls.get("method", "projection"),
                    min_line_height=int(ls.get("min_line_height", 8)),
                    smoothing=int(ls.get("smoothing", 3)),
                    min_column_gap_ratio=float(ls.get("min_column_gap_ratio", 0.035)),
                    min_column_width_ratio=float(ls.get("min_column_width_ratio", 0.20)),
                )
                _draw_segmentation(preprocessed, boxes, OUT / f"{page_id}_{variant['name']}_segmentation.png")
            cuda_peak_mb = (
                torch.cuda.max_memory_allocated(router.deva.device) / (1024 ** 2)
                if cuda_available else None
            )
            page_records[page_id] = {
                "status": status,
                "rotation_clockwise_applied": rotation_cw,
                "rotation_transform_original_pixels_to_processed": transform.tolist() if transform is not None else None,
                "page_cer": cer(reference, hypothesis),
                "page_wer": wer(reference, hypothesis),
                "reference_characters_normalized": len(normalize_for_scoring(reference)),
                "hypothesis_characters_normalized": len(normalize_for_scoring(hypothesis)),
                "line_crops": len(boxes),
                "attempted_crops": attempted,
                "empty_crops": empty,
                "failed_crops": failures,
                "wall_time_s": elapsed,
                "recognition_time_s": getattr(pipeline, "_last_page_timings", {}).get("recognition"),
                "rss_before_mb": rss_before / (1024 ** 2),
                "rss_after_mb": rss_after / (1024 ** 2),
                "cuda_peak_allocated_mb": cuda_peak_mb,
                "prediction": hypothesis,
                "ground_truth": reference,
            }
        successful = [row for row in page_records.values() if row.get("status") == "ok"]
        from eval.metrics import corpus_cer, corpus_wer
        results["variants"][variant["name"]] = {
            "pages": page_records,
            "summary": {
                "n_attempted": len(page_records),
                "n_successful": len(successful),
                "n_failed": sum(row.get("status") != "ok" for row in page_records.values()),
                "scope_complete": len(page_records) == len(selected) and len(successful) == len(selected),
                "corpus_cer_successful_pages_only": corpus_cer(
                    [row["ground_truth"] for row in successful],
                    [row["prediction"] for row in successful],
                )["cer"] if successful else None,
                "corpus_wer_successful_pages_only": corpus_wer(
                    [row["ground_truth"] for row in successful],
                    [row["prediction"] for row in successful],
                )["wer"] if successful else None,
            },
        }

    try:
        router.deva.unload()
    except Exception as exc:
        results["metadata"]["crnn_unload_warning"] = f"{type(exc).__name__}: {exc}"
    output_name = "full_layout_off_results.json" if args.full else "pilot_results.json"
    results["metadata"]["results_file"] = output_name
    (OUT / output_name).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    output = run()
    print(json.dumps({
        "metadata": output["metadata"],
        "variants": {
            name: {
                page_id: {
                    key: row.get(key) for key in (
                        "status", "page_cer", "page_wer", "line_crops", "empty_crops",
                        "failed_crops", "wall_time_s", "recognition_time_s",
                        "rss_before_mb", "rss_after_mb", "cuda_peak_allocated_mb",
                    )
                } for page_id, row in variant["pages"].items()
            } for name, variant in output["variants"].items()
        },
    }, ensure_ascii=False, indent=2))
