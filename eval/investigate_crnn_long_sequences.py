#!/usr/bin/env python3
"""Capture per-line A/B CRNN diagnostics for selected benchmark pages.

This is a diagnostic-only script. It reads existing benchmark pages and the
configured checkpoint, and writes crops/manifests only under results/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "crnn_long_sequence_investigation"
PAGE_IDS = ("BM_017", "BM_010", "BM_011")


def main() -> None:
    from akshardrishti.config import Config
    from akshardrishti.layout.lines import crop, segment_lines
    from akshardrishti.pipeline import AksharDrishtiPipeline
    from akshardrishti.postprocess.text_repair import clean_text
    from akshardrishti.preprocess.enhance import preprocess_page, preprocess_line_crop
    from akshardrishti.recognize.crnn import ctc_output_length
    from akshardrishti.recognize.crnn_model import beam_decode
    from eval.metrics import cer, wer, levenshtein_ops

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    cfg.set("layout.enabled", False)
    cfg.set("script_id.enabled", False)
    cfg.set("preprocess.orientation.enabled", False)
    pipeline = AksharDrishtiPipeline(cfg)
    rec = pipeline.router.deva
    rec.ensure_loaded()
    rec.inference_mode = "aspect_preserving"
    rec.max_preserved_width = max(rec.max_preserved_width, 8192)
    rows: list[dict] = []
    page_records = {}
    for page_id in PAGE_IDS:
        image_path = ROOT / "benchmark" / "images" / f"{page_id}.png"
        gt_path = ROOT / "benchmark" / "gt_text" / f"{page_id}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or not gt_path.exists():
            page_records[page_id] = {"error": "image or page-level reference missing"}
            continue
        page = preprocess_page(image, cfg.get("preprocess", {})).image
        if page.ndim == 3:
            page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        ls = cfg.get("recognize.line_segmentation", {}) or {}
        boxes = segment_lines(page, method="projection",
                              min_line_height=int(ls.get("min_line_height", 8)),
                              smoothing=int(ls.get("smoothing", 3)))
        overlay = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        page_preds = {"legacy_cap": [], "aspect_preserving": []}
        per_page = []
        for idx, box in enumerate(boxes):
            raw = crop(page, box)
            if raw.size == 0:
                continue
            normalized = preprocess_line_crop(raw, target_height=rec.img_height, binarize_crop=True)
            content_width = int(normalized.shape[1])
            image_tensor = torch.from_numpy(normalized.astype(np.float32) / 255.0)
            image_tensor = ((image_tensor - 0.5) / 0.5)[None, None].to(rec.device)
            if content_width > rec.max_width:
                legacy_pixels = cv2.resize(normalized, (rec.max_width, rec.img_height), interpolation=cv2.INTER_AREA)
                a_tensor = torch.from_numpy(legacy_pixels.astype(np.float32) / 255.0)
                a_tensor = ((a_tensor - 0.5) / 0.5)[None, None].to(rec.device)
            else:
                a_tensor = image_tensor
            pad_a = rec.right_pad
            pad_b = rec.right_pad
            a_tensor = torch.nn.functional.pad(a_tensor, (0, pad_a), value=1.0)
            b_tensor = torch.nn.functional.pad(image_tensor, (0, pad_b), value=1.0)
            with torch.no_grad():
                a_logits = rec.model(a_tensor)
                b_logits = rec.model(b_tensor)
            a_valid = int(a_logits.shape[0])
            b_valid = ctc_output_length(content_width)
            a_raw = beam_decode(a_logits, rec.charset, rec.beam_width)[0][0]
            b_raw = beam_decode(b_logits, rec.charset, rec.beam_width, [b_valid])[0][0]
            cleaner = dict(
                unicode_form=cfg.get("postprocess.unicode_normalize", "NFC"),
                repair_shirorekha=cfg.get("postprocess.repair_shirorekha", True),
                strip_control=cfg.get("postprocess.strip_control_chars", True),
                collapse_ws=cfg.get("postprocess.collapse_whitespace", True),
            )
            a_text, b_text = clean_text(a_raw, **cleaner), clean_text(b_raw, **cleaner)
            page_preds["legacy_cap"].append(a_text)
            page_preds["aspect_preserving"].append(b_text)
            x1, y1, x2, y2 = map(int, box.as_xyxy())
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 120, 255), 2)
            cv2.putText(overlay, str(idx), (x1, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 80, 220), 1)
            recrow = {
                "page_id": page_id, "line_index": idx, "bbox_xyxy": [x1, y1, x2, y2],
                "crop_original_width": int(raw.shape[1]), "crop_original_height": int(raw.shape[0]),
                "normalized_width": content_width, "legacy_resized_width": min(content_width, rec.max_width),
                "legacy_model_tensor_width": int(a_tensor.shape[-1]),
                "aspect_model_tensor_width": int(b_tensor.shape[-1]),
                "legacy_model_output_timesteps": a_valid,
                "aspect_model_output_timesteps": int(b_logits.shape[0]),
                "aspect_valid_ctc_timesteps": b_valid,
                "aspect_valid_fraction": b_valid / max(1, b_logits.shape[0]),
                "ink_fraction_normalized_crop": float((normalized < 128).mean()),
                "ground_truth": None,
                "legacy_prediction": a_text, "aspect_prediction": b_text,
                "legacy_prediction_chars": len(a_text), "aspect_prediction_chars": len(b_text),
                "legacy_raw_decoder_chars": len(a_raw), "aspect_raw_decoder_chars": len(b_raw),
            }
            rows.append(recrow)
            per_page.append(recrow)
            cv2.imwrite(str(OUT / f"{page_id}_line_{idx:03d}.png"), raw)
        cv2.imwrite(str(OUT / f"{page_id}_segmentation.png"), overlay)
        reference = gt_path.read_text(encoding="utf-8")
        hypotheses = {k: "\n".join(v) for k, v in page_preds.items()}
        page_records[page_id] = {
            "n_lines": len(per_page),
            "reference_chars_normalized": len(reference),
            "page_level_gt_used_for_line_scoring": False,
            "legacy_page_cer": cer(reference, hypotheses["legacy_cap"]),
            "aspect_page_cer": cer(reference, hypotheses["aspect_preserving"]),
            "legacy_page_wer": wer(reference, hypotheses["legacy_cap"]),
            "aspect_page_wer": wer(reference, hypotheses["aspect_preserving"]),
            "legacy_joined_chars": len(hypotheses["legacy_cap"]),
            "aspect_joined_chars": len(hypotheses["aspect_preserving"]),
            "legacy_page_prediction": hypotheses["legacy_cap"],
            "aspect_page_prediction": hypotheses["aspect_preserving"],
            "page_ground_truth": reference,
            "line_rows": len(per_page),
        }
    (OUT / "page_line_forensics.json").write_text(json.dumps({"pages": page_records, "lines": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "line_forensics.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    rec.unload()
    print(json.dumps({"output": str(OUT), "pages": {k: {a: v for a, v in r.items() if a not in ("legacy_page_prediction", "aspect_page_prediction", "page_ground_truth")} for k, r in page_records.items()}, "n_lines": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
