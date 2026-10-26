#!/usr/bin/env python3
"""Read-only CRNN crop/profile diagnostics; writes only under results/recognition_diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BUCKETS = (("short", 0, 128), ("medium", 129, 256), ("long", 257, 512), ("over_512", 513, 10**9))
SEED = 20261026
TARGET_HEIGHT = 32
MAX_WIDTH = 512


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def measure_row(row: dict[str, str], source: str, augmented: bool = False) -> dict | None:
    image_path = Path(row["image_path"])
    try:
        with Image.open(image_path) as image:
            w, h = image.size
    except (OSError, ValueError):
        return None
    scaled_w = max(8, int(round(w * TARGET_HEIGHT / max(h, 1))))
    capped_w = min(scaled_w, MAX_WIDTH)
    ratio_before = w / max(h, 1)
    ratio_after = capped_w / TARGET_HEIGHT
    text = row.get("text", "")
    return {
        "source": source,
        "split": row.get("split", "unknown"),
        "image_path": str(image_path),
        "original_width": w,
        "original_height": h,
        "original_aspect_ratio": ratio_before,
        "height32_width_before_cap": scaled_w,
        "height32_width_after_cap": capped_w,
        "aspect_ratio_after_cap": ratio_after,
        "relative_aspect_change": abs(ratio_after / max(ratio_before, 1e-12) - 1.0),
        "exceeds_512": scaled_w > MAX_WIDTH,
        "aspect_change_over_10pct": abs(ratio_after / max(ratio_before, 1e-12) - 1.0) > 0.10,
        "characters": len(text),
        "text": text,
        "augmented": augmented,
    }


def summarize(records: list[dict], requested: int, missing: int = 0) -> dict:
    if not records:
        return {"n_measured": 0, "n_requested": requested, "n_missing": missing}
    def q(key: str, p: float) -> float:
        return float(np.quantile([r[key] for r in records], p))
    return {
        "n_measured": len(records),
        "n_requested": requested,
        "n_missing_or_unreadable": missing,
        "original_width_px": {"p10": q("original_width", .1), "median": q("original_width", .5), "p90": q("original_width", .9)},
        "original_height_px": {"p10": q("original_height", .1), "median": q("original_height", .5), "p90": q("original_height", .9)},
        "height32_width_before_cap_px": {"p10": q("height32_width_before_cap", .1), "median": q("height32_width_before_cap", .5), "p90": q("height32_width_before_cap", .9)},
        "postprocess_width_px": {"p10": q("height32_width_after_cap", .1), "median": q("height32_width_after_cap", .5), "p90": q("height32_width_after_cap", .9)},
        "aspect_ratio_before": {"median": q("original_aspect_ratio", .5), "p90": q("original_aspect_ratio", .9)},
        "aspect_ratio_after_cap": {"median": q("aspect_ratio_after_cap", .5), "p90": q("aspect_ratio_after_cap", .9)},
        "characters_per_crop": {"median": q("characters", .5), "p90": q("characters", .9)},
        "percent_over_512_before_cap": 100 * sum(r["exceeds_512"] for r in records) / len(records),
        "percent_aspect_change_over_10pct": 100 * sum(r["aspect_change_over_10pct"] for r in records) / len(records),
        "width_buckets": {
            name: sum(lo <= r["height32_width_before_cap"] <= hi for r in records)
            for name, lo, hi in BUCKETS
        },
    }


def deterministic_sample(rows: list[dict[str, str]], cap: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= cap:
        return rows
    indices = sorted(random.Random(seed).sample(range(len(rows)), cap))
    return [rows[i] for i in indices]


def summarize_saved_bucket_predictions(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    from eval.metrics import corpus_cer, corpus_wer
    rows = read_csv(csv_path)
    output = {}
    for name, _, _ in BUCKETS:
        group = [r for r in rows if r.get("bucket") == name]
        if not group:
            output[name] = {"n": 0, "note": "No coverage-eligible held-out crops in this bucket."}
            continue
        refs = [r["ground_truth"] for r in group]
        current = [r["current_prediction"] for r in group]
        preserved = [r["aspect_preserved_prediction"] for r in group]
        output[name] = {
            "n": len(group),
            "current_cap_cer": corpus_cer(refs, current)["cer"],
            "aspect_preserved_cer": corpus_cer(refs, preserved)["cer"],
            "current_cap_wer": corpus_wer(refs, current)["wer"],
            "aspect_preserved_wer": corpus_wer(refs, preserved)["wer"],
            "exact_word_accuracy_current": sum(a == b for a, b in zip(refs, current)) / len(group),
            "exact_word_accuracy_preserved": sum(a == b for a, b in zip(refs, preserved)) / len(group),
        }
    return {"metrics_by_bucket": output, "recomputed_from_predictions_csv": csv_path.name}


def write_profile_plots(records_by_name: dict[str, list[dict]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Mozhi train": "#2b6cb0", "Mozhi validation": "#38a169", "Mozhi test": "#805ad5", "GovDoc originals": "#dd6b20"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.geomspace(8, max(1024, max(r["height32_width_before_cap"] for rs in records_by_name.values() for r in rs)), 36)
    for name, records in records_by_name.items():
        vals = [r["height32_width_before_cap"] for r in records]
        ax.hist(vals, bins=bins, histtype="step", linewidth=2, label=f"{name} (n={len(vals):,})", color=colors[name])
    ax.axvline(MAX_WIDTH, color="#c53030", linestyle="--", label="512 px cap")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Width after height normalization to 32 px, before cap")
    ax.set_ylabel("Crop count")
    ax.set_title("CRNN crop width distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "crop_width_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = list(records_by_name)
    before = [[r["original_aspect_ratio"] for r in records_by_name[n]] for n in names]
    after = [[r["aspect_ratio_after_cap"] for r in records_by_name[n]] for n in names]
    offsets = np.arange(len(names))
    ax.boxplot(before, positions=offsets - .16, widths=.25, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#90cdf4"}, medianprops={"color": "#1a365d"})
    ax.boxplot(after, positions=offsets + .16, widths=.25, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#fbd38d"}, medianprops={"color": "#7b341e"})
    ax.set_xticks(offsets, names, rotation=15, ha="right")
    ax.set_yscale("log", base=2)
    ax.set_ylabel("Aspect ratio (width / height), log scale")
    ax.set_title("Aspect ratio before and after the 512 px width cap")
    ax.plot([], [], color="#90cdf4", linewidth=8, label="Original crop")
    ax.plot([], [], color="#fbd38d", linewidth=8, label="After target height + cap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "aspect_ratio_distribution.png", dpi=160)
    plt.close(fig)


def save_page_diagnostics(cfg, page_ids: list[str], out_dir: Path) -> dict:
    import torch
    from akshardrishti.layout.lines import crop, offset_box, segment_lines
    from akshardrishti.postprocess.text_repair import clean_text
    from akshardrishti.pipeline import AksharDrishtiPipeline
    from akshardrishti.preprocess.enhance import preprocess_page, preprocess_line_crop
    from eval.metrics import cer, wer
    from akshardrishti.recognize.crnn_model import beam_decode

    cfg._data["layout"]["enabled"] = False
    pipeline = AksharDrishtiPipeline(cfg)
    detection_path = ROOT / "benchmark" / "prelabel_meta" / "detections_20261017_203514.csv"
    cached_detections = read_csv(detection_path) if detection_path.exists() else []
    pages_out: dict[str, dict] = {}
    for page_id in page_ids:
        image_path = ROOT / "benchmark" / "images" / f"{page_id}.png"
        gt_path = ROOT / "benchmark" / "gt_text" / f"{page_id}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or not gt_path.exists():
            pages_out[page_id] = {"error": "page image or full-page GT is missing"}
            continue
        gt_text = gt_path.read_text(encoding="utf-8")

        # Capture every line crop sent to the router, including predictions
        # that the production pipeline later drops because they are empty.
        route_calls = []
        original_recognize = pipeline.router.recognize
        def capture_recognize(line_img, *args, **kwargs):
            result = original_recognize(line_img, *args, **kwargs)
            route_calls.append({
                "image": line_img.copy(),
                "script": kwargs.get("script", args[0] if args else "unknown"),
                "raw_prediction": result.text,
                "backend": result.backend,
            })
            return result
        pipeline.router.recognize = capture_recognize
        page = pipeline.process_page(image, source_path=str(image_path), page_index=0)
        pipeline.router.recognize = original_recognize
        prediction = page.full_text()
        preprocessed = preprocess_page(image, cfg.get("preprocess", {})).image
        lines = []
        crop_images = []
        region = page.regions[0]
        region_img = crop(preprocessed, region.bbox)
        ls_cfg = cfg.get("recognize.line_segmentation", {}) or {}
        line_boxes = segment_lines(region_img, method=ls_cfg.get("method", "projection"),
                                   min_line_height=int(ls_cfg.get("min_line_height", 12)),
                                   smoothing=int(ls_cfg.get("smoothing", 5)))
        current_clean = []
        preserve_clean = []
        preserve_changed = 0
        preserve_errors = []
        deva = pipeline.router.deva
        for line_index, (line_box, call) in enumerate(zip(line_boxes, route_calls)):
            line_img = crop(region_img, line_box)
            if line_img.size == 0:
                continue
            page_box = offset_box(line_box, region.bbox.x1, region.bbox.y1)
            h, w = line_img.shape[:2]
            normalized = preprocess_line_crop(line_img, target_height=TARGET_HEIGHT, binarize_crop=True)
            uncapped = normalized.shape[1]
            capped = min(uncapped, MAX_WIDTH)
            current_text = clean_text(
                call["raw_prediction"],
                unicode_form=cfg.get("postprocess.unicode_normalize", "NFC"),
                repair_shirorekha=cfg.get("postprocess.repair_shirorekha", True),
                strip_control=cfg.get("postprocess.strip_control_chars", True),
                collapse_ws=cfg.get("postprocess.collapse_whitespace", True),
            )
            if current_text:
                current_clean.append(current_text)
            preserved_text = call["raw_prediction"]
            preserved_variant_ran = False
            if uncapped > MAX_WIDTH and call["backend"] == deva.name:
                # Diagnostic-only alternate: same crop processing, checkpoint,
                # right padding and beam; omit only the horizontal cap.
                pixels = normalized
                tensor = torch.from_numpy(pixels.astype(np.float32) / 255.0)
                tensor = ((tensor - .5) / .5)[None, None].to(deva.device)
                if deva.right_pad > 0:
                    tensor = torch.nn.functional.pad(tensor, (0, deva.right_pad), value=1.0)
                with torch.no_grad():
                    log_probs = deva.model(tensor)
                seq_len = max(1, (tensor.shape[-1] // 4) - 1)
                preserved_text = beam_decode(log_probs, deva.charset, deva.beam_width, [seq_len])[0][0]
                preserved_variant_ran = True
                preserve_changed += 1
            clean_preserved = clean_text(
                preserved_text,
                unicode_form=cfg.get("postprocess.unicode_normalize", "NFC"),
                repair_shirorekha=cfg.get("postprocess.repair_shirorekha", True),
                strip_control=cfg.get("postprocess.strip_control_chars", True),
                collapse_ws=cfg.get("postprocess.collapse_whitespace", True),
            )
            if clean_preserved:
                preserve_clean.append(clean_preserved)
            record = {
                "region_id": region.region_id,
                "line_index": line_index,
                "bbox_xyxy": [round(float(v), 2) for v in page_box.as_xyxy()],
                "script": call["script"],
                "backend": call["backend"],
                "raw_prediction": call["raw_prediction"],
                "text": current_text,
                "aspect_preserved_prediction": clean_preserved if preserved_variant_ran else None,
                "aspect_preserved_ran": preserved_variant_ran,
                "original_width": w,
                "original_height": h,
                "height32_width_before_cap": uncapped,
                "height32_width_after_cap": capped,
                "aspect_ratio_before": w / max(h, 1),
                "aspect_ratio_after_cap": capped / TARGET_HEIGHT,
                "would_exceed_512": uncapped > MAX_WIDTH,
                "empty_prediction": not bool(current_text.strip()),
            }
            lines.append(record)
            crop_images.append((line_img, record))

        preserve_prediction = "\n".join(preserve_clean)
        preserve_metrics = {
            "page_cer": cer(gt_text, preserve_prediction),
            "page_wer": wer(gt_text, preserve_prediction),
            "n_over_512_devanagari_lines_compared": preserve_changed,
            "interpretation": "Diagnostic page-level comparison; line segmentation/order is held constant. Other scripts/backends retain current predictions.",
        }

        # Draw GT layout boxes and historical cached YOLO predictions. The CSV
        # has no embedded weight hash and does not match today's weight hash.
        original_annotated = image.copy()
        layout_path = ROOT / "benchmark" / "gt_layout" / f"{page_id}.txt"
        if layout_path.exists():
            ph, pw = image.shape[:2]
            for line in layout_path.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                cid, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                x1, y1 = int((cx - bw / 2) * pw), int((cy - bh / 2) * ph)
                x2, y2 = int((cx + bw / 2) * pw), int((cy + bh / 2) * ph)
                cv2.rectangle(original_annotated, (x1, y1), (x2, y2), (30, 130, 255), 2)
                cv2.putText(original_annotated, f"GT class {cid}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 90, 220), 1, cv2.LINE_AA)
        page_dets = [d for d in cached_detections if d.get("image") == f"{page_id}.png"]
        ph, pw = image.shape[:2]
        for d in page_dets:
            cx, cy, bw, bh = (float(d[k]) for k in ("cx", "cy", "w", "h"))
            x1, y1 = int((cx - bw / 2) * pw), int((cy - bh / 2) * ph)
            x2, y2 = int((cx + bw / 2) * pw), int((cy + bh / 2) * ph)
            cv2.rectangle(original_annotated, (x1, y1), (x2, y2), (255, 120, 0), 2)
            cv2.putText(original_annotated, f"YOLO {d['class_name']} {float(d['conf']):.2f}", (x1, min(ph - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, .42, (190, 70, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{page_id}_page_gt_and_historical_detections.png"), original_annotated)

        # Draw actual layout-off projection line boxes on the preprocessed page.
        lines_annotated = preprocessed.copy()
        if lines_annotated.ndim == 2:
            lines_annotated = cv2.cvtColor(lines_annotated, cv2.COLOR_GRAY2BGR)
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = map(int, line["bbox_xyxy"])
            cv2.rectangle(lines_annotated, (x1, y1), (x2, y2), (0, 160, 0), 1)
            cv2.putText(lines_annotated, str(i), (x1, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 110, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{page_id}_page_projection_lines.png"), lines_annotated)

        # Deterministically select first/longest examples per width bucket.
        chosen = []
        for bucket, lo, hi in BUCKETS:
            group = [(im, rec) for im, rec in crop_images if lo <= rec["height32_width_before_cap"] <= hi]
            if group:
                chosen.extend(sorted(group, key=lambda item: (-item[1]["height32_width_before_cap"], item[1]["line_index"]))[:3])
        if chosen:
            cols = 2
            cell_w, cell_h = 720, 100
            rows_n = (len(chosen) + cols - 1) // cols
            sheet = np.full((rows_n * cell_h, cols * cell_w, 3), 255, np.uint8)
            for j, (crop_img, rec) in enumerate(chosen):
                x, y = (j % cols) * cell_w, (j // cols) * cell_h
                scale = min((cell_w - 12) / max(crop_img.shape[1], 1), 65 / max(crop_img.shape[0], 1))
                thumb = cv2.resize(crop_img, (max(1, int(crop_img.shape[1] * scale)), max(1, int(crop_img.shape[0] * scale))))
                if thumb.ndim == 2:
                    thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                h, w = thumb.shape[:2]
                sheet[y:y + h, x:x + w] = thumb
                label = f"w={rec['original_width']} h={rec['original_height']} norm={rec['height32_width_before_cap']} text={rec['text'][:34]}"
                cv2.putText(sheet, label, (x, y + 88), cv2.FONT_HERSHEY_SIMPLEX, .42, (25, 25, 25), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"{page_id}_line_crop_samples.png"), sheet)

        pages_out[page_id] = {
            "image": str(image_path),
            "ground_truth": gt_text,
            "prediction_layout_off": prediction,
            "page_cer": cer(gt_text, prediction),
            "page_wer": wer(gt_text, prediction),
            "aspect_preserved_diagnostic": preserve_metrics,
            "n_projected_lines": len(line_boxes),
            "n_recognized_nonempty_lines": sum(not x["empty_prediction"] for x in lines),
            "empty_or_whitespace_lines": sum(x["empty_prediction"] for x in lines),
            "lines_over_512": sum(x["would_exceed_512"] for x in lines),
            "per_region_cer": None,
            "per_region_cer_note": "No reliable region-level transcription references are available; benchmark text GT is page-level.",
            "line_crops": lines,
            "cached_detection_file": str(detection_path),
            "n_cached_yolo_detections": len(page_dets),
            "cached_detection_checkpoint_sha256": "not embedded; prelabel script records expected SHA256 5A927BAF51EE9EF3DDEA1DFFDAE954C73869A17B0614836F8856AA77B6AD4AED",
            "layout_preview_is": "GT layout boxes (orange) and historical cached YOLO predictions (blue); cache is not verified against current weights and is not fresh inference",
        }
    try:
        pipeline.router.unload()
    except Exception as exc:  # TrOCR may be unavailable in a lightweight environment.
        pages_out["_cleanup_warning"] = str(exc)
    return pages_out


def evaluate_mozhi_buckets(test_rows: list[dict[str, str]], cfg, max_per_bucket: int, out_dir: Path) -> dict:
    import torch
    from akshardrishti.postprocess.text_repair import normalize_for_scoring
    from akshardrishti.recognize.base import build_recognizer
    from akshardrishti.recognize.crnn_model import beam_decode
    from akshardrishti.preprocess.enhance import preprocess_line_crop
    from eval.metrics import corpus_cer, corpus_wer

    backend = cfg._data["recognize"]["deva_backend"]
    backend_cfg = dict(cfg._data["recognize"]["backends"][backend])
    for key in ("weights", "charset"):
        val = Path(backend_cfg[key])
        backend_cfg[key] = str(val if val.is_absolute() else ROOT / val)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recognizer = build_recognizer(backend, backend_cfg, device=device)
    recognizer.ensure_loaded()
    charset = recognizer.charset
    buckets: dict[str, list[dict]] = defaultdict(list)
    missing = 0
    # Use a fixed held-out sample before stratifying; metadata reads are
    # filesystem-bound on this large crop set.
    test_sample = deterministic_sample(test_rows, min(10000, len(test_rows)), SEED + 2)
    measured_test = []
    for row in test_sample:
        rec = measure_row(row, "Mozhi test")
        if rec is None:
            missing += 1
            continue
        if charset.coverage(row["text"]) < 0.999:
            continue
        for name, lo, hi in BUCKETS:
            if lo <= rec["height32_width_before_cap"] <= hi:
                measured_test.append((name, row, rec))
                break
    by_bucket: dict[str, list[tuple[dict[str, str], dict]]] = defaultdict(list)
    for name, row, rec in measured_test:
        by_bucket[name].append((row, rec))
    rng = random.Random(SEED)
    selected = []
    for name, _, _ in BUCKETS:
        pool = by_bucket.get(name, [])
        if pool:
            chosen = rng.sample(pool, min(max_per_bucket, len(pool)))
            selected.extend((name, row, rec) for row, rec in chosen)

    all_eval_rows = []
    model = recognizer.model
    model.eval()
    right_pad = int(backend_cfg.get("right_pad", 64))
    with torch.no_grad():
        for bucket_name, row, record in selected:
            image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            line = preprocess_line_crop(image, target_height=TARGET_HEIGHT, binarize_crop=True)
            uncapped_w = line.shape[1]
            capped = line
            if uncapped_w > MAX_WIDTH:
                capped = cv2.resize(line, (MAX_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            # Current production-equivalent cap and diagnostic-only preserved width.
            variants = {"current_cap": capped, "aspect_preserved": line}
            preds = {}
            for variant, pixels in variants.items():
                tensor = torch.from_numpy(pixels.astype(np.float32) / 255.0)
                tensor = ((tensor - .5) / .5)[None, None].to(device)
                if right_pad > 0:
                    tensor = torch.nn.functional.pad(tensor, (0, right_pad), value=1.0)
                log_probs = model(tensor)
                seq_len = max(1, (tensor.shape[-1] // 4) - 1)
                preds[variant] = beam_decode(log_probs, charset, int(backend_cfg.get("beam_width", 10)), [seq_len])[0][0]
            ref = normalize_for_scoring(row["text"])
            all_eval_rows.append({
                "bucket": bucket_name,
                "split": row.get("split", "test"),
                "image_path": row["image_path"],
                "ground_truth": ref,
                "original_width": record["original_width"],
                "original_height": record["original_height"],
                "height32_width_before_cap": uncapped_w,
                "current_width": capped.shape[1],
                "current_prediction": preds["current_cap"],
                "aspect_preserved_prediction": preds["aspect_preserved"],
            })
    recognizer.unload()

    summary = {}
    for name, _, _ in BUCKETS:
        rows = [r for r in all_eval_rows if r["bucket"] == name]
        if not rows:
            summary[name] = {"n": 0, "note": "No coverage-eligible held-out crops in this bucket."}
            continue
        refs = [r["ground_truth"] for r in rows]
        current = [r["current_prediction"] for r in rows]
        preserve = [r["aspect_preserved_prediction"] for r in rows]
        summary[name] = {
            "n": len(rows),
            "current_cap_cer": corpus_cer(refs, current)["cer"],
            "aspect_preserved_cer": corpus_cer(refs, preserve)["cer"],
            "current_cap_wer": corpus_wer(refs, current)["wer"],
            "aspect_preserved_wer": corpus_wer(refs, preserve)["wer"],
            "exact_word_accuracy_current": sum(a == b for a, b in zip(refs, current)) / len(rows),
            "exact_word_accuracy_preserved": sum(a == b for a, b in zip(refs, preserve)) / len(rows),
        }

    with (out_dir / "mozhi_test_bucket_predictions.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_eval_rows[0]) if all_eval_rows else ["bucket"])
        writer.writeheader()
        writer.writerows(all_eval_rows)

    return {"device": device, "n_test_rows_total": len(test_rows), "n_test_rows_profiled": len(test_sample), "n_unreadable": missing,
            "max_per_bucket": max_per_bucket, "seed": SEED, "metrics_by_bucket": summary,
            "predictions_csv": "mozhi_test_bucket_predictions.csv"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-profile-train", type=int, default=10000)
    ap.add_argument("--max-per-bucket", type=int, default=24)
    ap.add_argument("--skip-model", action="store_true", help="profile crop sizes only; skip checkpoint inference")
    ap.add_argument("--skip-pages", action="store_true", help="skip 3-page layout-off real-document inspection")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "recognition_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_csv(ROOT / "datasets/mozhi_raw/index_train.csv")
    val_rows = read_csv(ROOT / "datasets/mozhi_raw/index_val.csv")
    test_rows = read_csv(ROOT / "datasets/mozhi_raw/index_test.csv")
    gov_rows = read_csv(ROOT / "datasets/govdoc_finetune/index.csv")
    gov_originals = [r for r in gov_rows if "_aug" not in Path(r["image_path"]).stem]
    profile_specs = {
        "Mozhi train": (deterministic_sample(train_rows, args.max_profile_train, SEED), len(train_rows)),
        "Mozhi validation": (deterministic_sample(val_rows, min(5000, len(val_rows)), SEED + 1), len(val_rows)),
        "Mozhi test": (deterministic_sample(test_rows, min(5000, len(test_rows)), SEED + 2), len(test_rows)),
        "GovDoc originals": (gov_originals, len(gov_originals)),
    }
    records_by_name: dict[str, list[dict]] = {}
    summary = {"profile_seed": SEED, "profile_method": "fixed seeded samples for Mozhi train/validation/test; all unaugmented GovDoc originals",
               "max_width": MAX_WIDTH, "target_height": TARGET_HEIGHT, "sets": {}}
    for source, (rows, full_count) in profile_specs.items():
        records = []
        missing = 0
        for row in rows:
            rec = measure_row(row, source, augmented="_aug" in Path(row["image_path"]).stem)
            if rec is None:
                missing += 1
            else:
                records.append(rec)
        records_by_name[source] = records
        summary["sets"][source] = summarize(records, len(rows), missing) | {"index_rows_total": full_count}
        print(f"{source}: measured {len(records):,}/{len(rows):,}; cap rate {summary['sets'][source].get('percent_over_512_before_cap', 0):.2f}%")

    # Summarize augmentation and label split status without treating augmented rows as independent crops.
    gov_aug = [r for r in gov_rows if "_aug" in Path(r["image_path"]).stem]
    gov_splits = defaultdict(int)
    for r in gov_rows:
        gov_splits[r.get("split", "unknown")] += 1
    summary["govdoc_dataset"] = {
        "all_rows_in_index": len(gov_rows), "original_rows": len(gov_originals), "augmented_rows": len(gov_aug),
        "split_row_counts_including_aug": dict(gov_splits),
        "label_generation_note": "data/make_govdoc_finetune.py assigns page-level GT lines to regions proportionally to detected line counts; crop-text alignment is heuristic, so these labels were not used for accuracy scoring.",
    }
    write_profile_plots(records_by_name, out_dir)
    (out_dir / "crop_profile.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    import yaml
    from akshardrishti.config import Config
    cfg = Config.load(ROOT / "configs/pipeline.yaml")
    if not args.skip_model:
        summary["mozhi_test_bucket_evaluation"] = evaluate_mozhi_buckets(test_rows, cfg, args.max_per_bucket, out_dir)
        (out_dir / "mozhi_test_bucket_evaluation.json").write_text(
            json.dumps(summary["mozhi_test_bucket_evaluation"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bucket_metrics = summary["mozhi_test_bucket_evaluation"]["metrics_by_bucket"]
        names = [name for name, _, _ in BUCKETS if bucket_metrics.get(name, {}).get("n", 0)]
        if names:
            x = np.arange(len(names)); width = .36
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.bar(x - width/2, [bucket_metrics[n]["current_cap_cer"] for n in names], width, label="Current 512 cap")
            ax.bar(x + width/2, [bucket_metrics[n]["aspect_preserved_cer"] for n in names], width, label="Diagnostic: preserve aspect")
            ax.set_xticks(x, names); ax.set_ylabel("Corpus CER"); ax.set_title("Mozhi held-out crop CER by normalized-width bucket")
            ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "mozhi_bucket_cer.png", dpi=160); plt.close(fig)
    else:
        saved_metrics = summarize_saved_bucket_predictions(out_dir / "mozhi_test_bucket_predictions.csv")
        if saved_metrics:
            summary["mozhi_test_bucket_evaluation"] = saved_metrics
            (out_dir / "mozhi_test_bucket_evaluation.json").write_text(
                json.dumps(saved_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    if not args.skip_pages:
        summary["government_pages"] = save_page_diagnostics(cfg, ["BM_001", "BM_015", "BM_021"], out_dir)
        (out_dir / "government_page_samples.json").write_text(
            json.dumps(summary["government_pages"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (out_dir / "diagnostics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
