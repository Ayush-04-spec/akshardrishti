#!/usr/bin/env python3
"""Export untranscribed, benchmark-test-only crop candidates for human review."""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "crnn_long_sequence_investigation" / "annotation_pack"
PAGES = ("BM_001", "BM_010", "BM_011", "BM_017", "BM_021")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    from akshardrishti.config import Config
    from akshardrishti.layout.lines import segment_lines
    from akshardrishti.preprocess.enhance import preprocess_page

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    rows = []
    for doc in PAGES:
        page_path = ROOT / "benchmark" / "images" / f"{doc}.png"
        image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        page = preprocess_page(image, cfg.get("preprocess", {})).image
        if page.ndim == 3:
            page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        ls = cfg.get("recognize.line_segmentation", {}) or {}
        boxes = segment_lines(page, "projection",
                              min_line_height=int(ls.get("min_line_height", 8)),
                              smoothing=int(ls.get("smoothing", 3)))
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.as_xyxy())
            crop = page[y1:y2, x1:x2]
            crop_name = f"{doc}_line_{idx:03d}.png"
            cv2.imwrite(str(OUT / crop_name), crop)
            normalized_w = max(8, round(crop.shape[1] * 32 / max(crop.shape[0], 1)))
            rows.append({
                "sample_id": f"{doc}_L{idx:03d}", "document_id": doc, "page_id": doc,
                "split": "benchmark_test_only", "source_image": str(page_path.relative_to(ROOT)),
                "source_sha256": digest(page_path), "crop_path": crop_name,
                "bbox_xyxy_on_preprocessed_page": f"{x1},{y1},{x2},{y2}",
                "orientation_degrees_clockwise": 0, "sequence_index": idx,
                "crop_width_px": int(crop.shape[1]), "crop_height_px": int(crop.shape[0]),
                "height32_width_before_cap_px": int(normalized_w),
                "content_hint": "degraded_scan" if doc == "BM_017" else "table_or_dense_rows" if doc == "BM_021" else "printed_text",
                "crop_review_status": "needs_human_bbox_review" if doc == "BM_021" else "needs_human_visual_review",
                "transcription_nfc": "", "transcription_status": "untranscribed",
                "transcriber_id": "", "verifier_id": "", "verification_status": "unverified",
                "notes": "Do not use page-level GT as crop transcription; benchmark pages remain test-only.",
            })
    fields = list(rows[0]) if rows else []
    with (OUT / "annotation_manifest_template.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "ANNOTATION_PROTOCOL.md").write_text(
        """# Manual CRNN crop transcription protocol

All BM_001/BM_010/BM_011/BM_017/BM_021 candidates are reserved for benchmark test/evaluation only. Do not move any benchmark document or derivative into training or validation. The generated crops are candidates, not trusted labels.

1. Review each crop image and its source page. Reject artifacts, merged rows, clipped ascenders/descenders, and wrong reading orientation. Record a corrected pixel bbox before transcription.
2. For BM_021 table content, manually define cell/row crops; projection candidates can cross cells and must not be treated as line truth. Preserve reading order in `sequence_index`.
3. Two annotators independently transcribe exactly visible text, retaining meaningful spaces, punctuation, digits, matras, conjuncts, and line content. No prediction-assisted transcription.
4. Store the agreed transcription in NFC; an independent verifier compares against the crop and marks `verified`. Keep rejected crops with a reason for auditability.
5. For future model development, collect separate document IDs absent from BM_001–BM_024. Assign whole documents (not crops) to train/validation/test before producing crops; use a small validation slice for model selection and an untouched test slice for final reporting. The current repository contains no independently verified real-document line labels suitable for training/validation.

Manifest rows require stable `document_id`, `page_id`, `split`, source SHA-256, crop path, crop bbox, orientation, sequence index, transcription, transcriber, verifier, and verification status. Current CSV has untranscribed cells and benchmark-test-only split by design. Do not score model output until `verification_status=verified`.
""",
        encoding="utf-8",
    )
    print(f"Exported {len(rows)} untranscribed candidates to {OUT}")


if __name__ == "__main__":
    main()
