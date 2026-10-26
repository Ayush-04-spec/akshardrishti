#!/usr/bin/env python3
"""Build a human-reviewable proposal manifest from page GT and crop order."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PACK = ROOT / "results" / "crnn_long_sequence_investigation" / "annotation_pack"
DEFAULT_OUT = ROOT / "results" / "bm_ground_truth_crop_alignment_20261026_v7"
VISUALLY_CONFIRMED_GT_ID_OVERRIDES = {
    "BM_010": ("BM_007", "Image title/body begins 'सर्व शासकीय पत्रव्यवहार…'; distinctive text matches BM_007 GT."),
    "BM_011": ("BM_008", "Image title and मातंस/नस्ती reference match BM_008 GT."),
    "BM_017": ("BM_018", "Image continues section ६ on एकात्मिक बाल विकास; BM_018 GT begins with that continuation."),
}
VISUALLY_CONFIRMED_SAME_ID = {
    "BM_001": "Image header and body opening agree with BM_001 GT.",
    "BM_021": "Visible table rows and names agree with BM_021 GT; table crop boundaries still need review.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_bbox(raw: str) -> tuple[int, int, int, int]:
    values = tuple(int(v) for v in raw.split(","))
    if len(values) != 4:
        raise ValueError(f"expected four bbox coordinates, got {raw!r}")
    return values


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1, area_a + area_b - intersection)


def add_overlap_flags(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row["source_page_id"]].append(idx)
    for indices in groups.values():
        for pos, left in enumerate(indices):
            for right in indices[pos + 1:]:
                if bbox_iou(tuple(rows[left]["bbox_xyxy"]), tuple(rows[right]["bbox_xyxy"])) >= 0.5:
                    for index in (left, right):
                        rows[index]["overlap_with_crop_ids"].append(rows[right if index == left else left]["crop_id"])
    for row in rows:
        if row["overlap_with_crop_ids"]:
            row["alignment_status"] = "boundary_review"
            row["manual_review_required"] = True
            row["manual_review_reason"] = ";".join(filter(None, [
                row["manual_review_reason"], "bbox_overlaps_another_candidate_crop"
            ]))
    return rows


def _render_html(rows: list[dict], output: Path) -> None:
    ordered = sorted(rows, key=lambda r: (
        {"candidate_match": 0, "ambiguous_match": 1, "boundary_review": 2, "no_match": 3}.get(r["alignment_status"], 4),
        r["source_page_id"], r["sequence_index"],
    ))
    cards = []
    for row in ordered:
        crop_src = Path("..") / "crnn_long_sequence_investigation" / "annotation_pack" / row["crop_image_path"]
        prev_src = (Path("..") / "crnn_long_sequence_investigation" / "annotation_pack" / row["previous_crop_image_path"]) if row["previous_crop_image_path"] else None
        next_src = (Path("..") / "crnn_long_sequence_investigation" / "annotation_pack" / row["next_crop_image_path"]) if row["next_crop_image_path"] else None
        status = html.escape(row["alignment_status"])
        neighbor_imgs = "".join(
            f'<div class="neighbor"><small>{html.escape(label)}</small><br><img src="{html.escape(str(src).replace(chr(92), "/"))}" loading="lazy"></div>'
            for label, src in (("Previous crop", prev_src), ("Next crop", next_src)) if src
        )
        passage = row["matching_gt_span"] or "(no proposed span)"
        cards.append(f"""<article class="card" data-status="{status}" data-page="{html.escape(row['source_page_id'])}">
<header><strong>{html.escape(row['crop_id'])}</strong> · image {html.escape(row['source_page_id'])} · GT {html.escape(row['gt_source_page_id'])} · line {row['sequence_index']} · <span class="status {status}">{status}</span></header>
<div class="body"><div><img class="crop" src="{html.escape(str(crop_src).replace(chr(92), "/"))}" loading="lazy"><div class="dims">{row['crop_width_px']}×{row['crop_height_px']} px; height-32 width {row['normalized_width_px']} px; bbox {html.escape(row['bbox_string'])}</div></div>
<section><b>Proposed transcription (unverified)</b><p class="text">{html.escape(row['candidate_transcription'] or '(none)')}</p><b>Matching page-GT span</b><p class="text">{html.escape(passage)}</p><small>GT offsets {row['gt_span_start']}–{row['gt_span_end']} · heuristic score {row['alignment_confidence']} (not a probability) · review: {html.escape(row['manual_review_reason'] or 'visual confirmation required')}</small></section></div>
<footer>{neighbor_imgs}</footer></article>""")
    page = """<!doctype html><meta charset="utf-8"><title>BM crop / page-GT alignment review</title>
<style>body{font:15px system-ui,sans-serif;margin:20px;background:#f6f7f9;color:#20242a}h1{font-size:24px}.toolbar{position:sticky;top:0;background:#fff;padding:12px;border:1px solid #ccd2dc;z-index:3}.card{background:#fff;margin:14px 0;padding:12px;border:1px solid #cbd1d8;border-left:8px solid #aab}.card[data-status="candidate_match"]{border-left-color:#29945b}.card[data-status="ambiguous_match"]{border-left-color:#dc9a20}.card[data-status="boundary_review"]{border-left-color:#d34b42}.body{display:grid;grid-template-columns:minmax(300px,42%) 1fr;gap:18px;margin-top:10px}.crop{max-width:100%;max-height:180px;object-fit:contain;background:#eee}.dims,small{color:#545d69;font-size:12px}.text{white-space:pre-wrap;border:1px solid #ddd;padding:8px;background:#fafafa;min-height:1.5em}.status{font-weight:700}.candidate_match{color:#16713e}.ambiguous_match{color:#946200}.boundary_review{color:#a22620}.no_match{color:#6c4a8e}footer{display:flex;gap:15px}.neighbor img{max-width:420px;max-height:70px;object-fit:contain;background:#eee}</style>
<h1>BM full-page GT to crop alignment proposals</h1><p>Every proposed span is unverified. Crop-level scoring is disabled until a reviewer confirms the text and boundaries.</p>
<div class="toolbar">Filter: <select id="s"><option value="all">All statuses</option><option>candidate_match</option><option>ambiguous_match</option><option>boundary_review</option><option>no_match</option></select> Page: <select id="p"><option value="all">All pages</option></select> · <span id="n"></span></div>
""" + "\n".join(cards) + """
<script>const cards=[...document.querySelectorAll('.card')],p=document.querySelector('#p');[...new Set(cards.map(x=>x.dataset.page))].sort().forEach(x=>p.add(new Option(x,x)));function f(){let s=document.querySelector('#s').value,pg=p.value,n=0;cards.forEach(c=>{let show=(s==='all'||c.dataset.status===s)&&(pg==='all'||c.dataset.page===pg);c.hidden=!show;n+=show});document.querySelector('#n').textContent=n+' crops shown'}document.querySelector('#s').onchange=f;p.onchange=f;f()</script>"""
    output.write_text(page, encoding="utf-8")


def build(output: Path = DEFAULT_OUT) -> dict:
    from eval.crop_gt_alignment import align_ordered_crops

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    source_manifest = PACK / "annotation_manifest_template.csv"
    crop_rows = read_csv(source_manifest)
    gt_files = {p.stem: p for p in (ROOT / "benchmark" / "gt_text").glob("BM_*.txt")}
    image_files = {p.stem: p for p in (ROOT / "benchmark" / "images").glob("BM_*.png")}
    if set(gt_files) != set(image_files):
        raise ValueError(f"benchmark page ID mismatch: GT only={sorted(set(gt_files)-set(image_files))}, images only={sorted(set(image_files)-set(gt_files))}")
    metadata_path = ROOT / "benchmark" / "metadata.csv"
    metadata = {r["line_id"]: r for r in read_csv(metadata_path)} if metadata_path.exists() else {}
    by_page: dict[str, list[dict[str, str]]] = defaultdict(list)
    for crop in crop_rows:
        if crop["source_image"].replace("\\", "/") != f"benchmark/images/{crop['page_id']}.png":
            raise ValueError(f"crop source mismatch for {crop['sample_id']}: {crop['source_image']}")
        if crop["page_id"] not in gt_files:
            raise ValueError(f"no page GT for crop {crop['sample_id']}")
        by_page[crop["page_id"]].append(crop)

    rows = []
    page_ids = []
    mapping_audit = {}
    for page_id, candidates in sorted(by_page.items()):
        page_ids.append(page_id)
        candidates.sort(key=lambda row: int(row["sequence_index"]))
        gt_page_id, mapping_reason = VISUALLY_CONFIRMED_GT_ID_OVERRIDES.get(
            page_id, (page_id, "same-stem ID; cross-page mapping not performed")
        )
        if gt_page_id not in gt_files:
            raise ValueError(f"mapped GT file is missing for {page_id}: {gt_page_id}")
        gt = gt_files[gt_page_id].read_text(encoding="utf-8")
        mapping_audit[page_id] = {
            "image_page_id": page_id, "gt_page_id_used_for_proposal": gt_page_id,
            "same_stem_pair_matches_visually": (
                False if page_id in VISUALLY_CONFIRMED_GT_ID_OVERRIDES else
                True if page_id in VISUALLY_CONFIRMED_SAME_ID else None
            ),
            "mapping_basis": VISUALLY_CONFIRMED_SAME_ID.get(page_id, mapping_reason),
        }
        proposed = align_ordered_crops(candidates, gt, table_page=metadata.get(page_id, {}).get("has_table", "no").lower() == "yes")
        for idx, row in enumerate(proposed):
            bbox = parse_bbox(row["bbox_xyxy_on_preprocessed_page"])
            row.update({
                "crop_id": row["sample_id"], "source_page_id": page_id,
                "crop_image_path": row["crop_path"], "crop_width_px": int(row["crop_width_px"]),
                "crop_height_px": int(row["crop_height_px"]),
                "normalized_width_px": int(row["height32_width_before_cap_px"]),
                "bbox_xyxy": list(bbox), "bbox_string": row["bbox_xyxy_on_preprocessed_page"],
                "sequence_index": int(row["sequence_index"]),
                "previous_crop_image_path": candidates[idx - 1]["crop_path"] if idx else "",
                "next_crop_image_path": candidates[idx + 1]["crop_path"] if idx + 1 < len(candidates) else "",
                "overlap_with_crop_ids": [], "page_gt_path": str(gt_files[gt_page_id].relative_to(ROOT)),
                "same_stem_gt_path": str(gt_files[page_id].relative_to(ROOT)),
                "gt_source_page_id": gt_page_id,
                "gt_source_path": str(gt_files[gt_page_id].relative_to(ROOT)),
                "gt_pair_match_status": (
                    "visually_confirmed_cross_id_match" if gt_page_id != page_id else
                    "visually_confirmed_same_id_match" if page_id in VISUALLY_CONFIRMED_SAME_ID else
                    "same_stem_not_visually_audited"
                ),
                "gt_mapping_basis": VISUALLY_CONFIRMED_SAME_ID.get(page_id, mapping_reason),
                "page_image_path": str(image_files[page_id].relative_to(ROOT)),
                "page_gt_nfc": gt == __import__("unicodedata").normalize("NFC", gt),
                "split": "benchmark_test_only", "verification_status": "unverified",
                "eligible_for_scoring": False,
            })
            reasons = [row.get("manual_review_reason", "")]
            if row["normalized_width_px"] > 1200:
                reasons.append("very_wide_crop_check_for_wrapping_or_merged_content")
            if page_id == "BM_017":
                reasons.append("degraded_document_requires_visual_confirmation")
            row["manual_review_reason"] = ";".join(dict.fromkeys(filter(None, reasons)))
            rows.append(row)
    add_overlap_flags(rows)

    # Each crop appears exactly once and ordered GT spans are non-overlapping
    # by construction. Preserve all page GT source spans verbatim in the CSV.
    output.mkdir(parents=True, exist_ok=False)
    fields = [
        "crop_id", "source_page_id", "split", "crop_image_path", "source_image", "page_image_path", "page_gt_path",
        "same_stem_gt_path", "gt_source_page_id", "gt_source_path", "gt_pair_match_status", "gt_mapping_basis",
        "bbox_xyxy_on_preprocessed_page", "crop_width_px", "crop_height_px", "normalized_width_px", "sequence_index",
        "candidate_transcription", "matching_gt_span", "gt_span_start", "gt_span_end", "alignment_method",
        "alignment_confidence", "confidence_interpretation", "ambiguity_margin", "alignment_status",
        "manual_review_required", "manual_review_reason", "overlap_with_crop_ids", "page_gt_nfc",
        "verification_status", "eligible_for_scoring",
    ]
    with (output / "alignment_manifest.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {**row, "overlap_with_crop_ids": ";".join(row["overlap_with_crop_ids"])}
            writer.writerow(serialized)
    summary = {
        "source_candidate_crops": len(crop_rows), "aligned_rows": len(rows), "source_pages": page_ids,
        "benchmark_gt_files": len(gt_files), "benchmark_images": len(image_files),
        "matching_page_image_gt_pairs": len(gt_files), "metadata_page_ids_match": not metadata or set(metadata) == set(gt_files),
        "candidate_page_gt_mapping_audit": mapping_audit,
        "status_counts": dict(Counter(r["alignment_status"] for r in rows)),
        "manual_review_required": sum(bool(r["manual_review_required"]) for r in rows),
        "eligible_for_scoring": sum(bool(r["eligible_for_scoring"]) for r in rows),
        "overlap_crop_count": sum(bool(r["overlap_with_crop_ids"]) for r in rows),
        "width_distribution_by_page": {
            page: {
                "n": sum(r["source_page_id"] == page for r in rows),
                "p50": sorted(r["normalized_width_px"] for r in rows if r["source_page_id"] == page)[len([r for r in rows if r["source_page_id"] == page]) // 2],
                "p90": sorted(r["normalized_width_px"] for r in rows if r["source_page_id"] == page)[int(0.9 * (len([r for r in rows if r["source_page_id"] == page]) - 1))],
                "bins": {
                    "up_to_128": sum(r["source_page_id"] == page and r["normalized_width_px"] <= 128 for r in rows),
                    "129_to_256": sum(r["source_page_id"] == page and 128 < r["normalized_width_px"] <= 256 for r in rows),
                    "257_to_512": sum(r["source_page_id"] == page and 256 < r["normalized_width_px"] <= 512 for r in rows),
                    "513_to_1024": sum(r["source_page_id"] == page and 512 < r["normalized_width_px"] <= 1024 for r in rows),
                    "over_1024": sum(r["source_page_id"] == page and r["normalized_width_px"] > 1024 for r in rows),
                },
                "over_512": sum(r["source_page_id"] == page and r["normalized_width_px"] > 512 for r in rows),
            }
            for page in page_ids
        },
        "alignment_confidence_is_calibrated": False,
        "method_limitations": ["uses page GT line order and approximate crop width/character count only", "does not read crop pixels or OCR predictions", "all matches are unverified and excluded from scoring"],
    }
    (output / "alignment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_html(rows, output / "review.html")
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else ROOT / args.out
    print(json.dumps({"output": str(output), **build(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
