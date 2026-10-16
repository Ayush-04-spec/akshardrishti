#!/usr/bin/env python3
"""Generate a self-contained HTML labeling pack for word-gap calibration.

IndicDLP TRAIN-split pages only. No torch, no recognizer, no model output.
Each annotator gets one HTML file with embedded base64 PNG line images.

Usage
-----
    python eval/make_line_labeling_pack.py --annotators 4
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.layout.lines import segment_lines, crop    # noqa: E402
from akshardrishti.preprocess.enhance import preprocess_line_crop  # noqa: E402
from akshardrishti.schema import BBox                             # noqa: E402


# ─────────────────────────────── helpers ─────────────────────────────────────

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


def segment_at_threshold(line_32: np.ndarray, t: int) -> list[tuple[int, int]]:
    """Cut at internal runs of empty columns >= t. Same as calibrate_word_gap.py."""
    ink_mask = (line_32 < 128).astype(np.uint8)
    ink_proj = ink_mask.sum(axis=0)
    nonempty = np.where(ink_proj > 0)[0]
    if len(nonempty) < 2:
        return []
    left = nonempty[0]
    right = nonempty[-1]
    trimmed = ink_proj[left:right + 1]
    empty = (trimmed == 0)
    segments = []
    word_start = 0
    i = 0
    W_trim = len(trimmed)
    while i < W_trim:
        if empty[i]:
            run_start = i
            while i < W_trim and empty[i]:
                i += 1
            run_len = i - run_start
            if run_len >= t:
                if run_start > word_start:
                    segments.append((left + word_start, left + run_start))
                word_start = i
        else:
            i += 1
    if word_start < W_trim:
        segments.append((left + word_start, left + W_trim))
    return [(x1, x2) for x1, x2 in segments if x2 - x1 >= 2]


def img_to_base64_png(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("ascii")


# ─────────────────────────────── main ────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotators", type=int, default=4)
    args = ap.parse_args()
    N_ANNOTATORS = args.annotators

    out_dir = ROOT / "results/line_labeling"
    if out_dir.exists():
        print(f"Output dir already exists: {out_dir}")
        print("Remove it to re-run. Aborting.")
        sys.exit(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preview").mkdir(exist_ok=True)

    # ── SHA-256s (read-only, no torch) ───────────────────────────────────────
    import yaml
    with open(ROOT / "configs/pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    weights_sha = sha256_of(ROOT / cfg["recognize"]["backends"]["crnn_mozhi"]["weights"].lstrip("./"))
    charset_sha = sha256_of(ROOT / cfg["recognize"]["backends"]["crnn_mozhi"]["charset"].lstrip("./"))
    layout_sha  = sha256_of(ROOT / cfg["layout"]["weights"].lstrip("./"))
    full_sha    = sha256_of(ROOT / "weights/recognize/crnn_mozhi_himr_full.pt")

    # ── STEP 1: Sample lines ─────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Sample lines")

    manifest_path = ROOT / "split_manifest.csv"
    all_manifest = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    train_hm = [r for r in all_manifest
                if r["split"] == "train" and r["language"] in ("hindi", "marathi")]

    cells: dict[tuple, list[str]] = defaultdict(list)
    for r in train_hm:
        cells[(r["language"], r["domain"])].append(r["filename"])

    RNG = random.Random(42)
    TEXT_CLASSES = {0, 1, 2}
    TARGET_LINES_PER_CELL = 30
    MAX_LINES_PER_PAGE = 3
    MIN_ORIG_H = 12
    MIN_RESIZED_W = 64
    MAX_BOXES_PER_PAGE = 1  # no two lines from the same box

    labels_dir = ROOT / "datasets/indicdlp_subset/labels/train"
    images_dir = ROOT / "datasets/indicdlp_subset/images/train"

    # For each cell: sample pages, extract lines
    all_lines: list[dict] = []  # each entry has all metadata
    pages_used_per_cell: dict[tuple, set] = defaultdict(set)

    for (lang, domain) in sorted(cells.keys()):
        pool = sorted(cells[(lang, domain)])
        # Shuffle pages to sample from
        page_order = list(pool)
        RNG.shuffle(page_order)

        cell_lines = []
        for fname in page_order:
            if len(cell_lines) >= TARGET_LINES_PER_CELL:
                break
            stem = Path(fname).stem
            label_file = labels_dir / f"{stem}.txt"
            image_file = images_dir / fname
            if not label_file.exists() or not image_file.exists():
                continue

            img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            H, W = img.shape[:2]

            # Parse YOLO labels
            text_boxes = []
            for idx, line in enumerate(open(label_file, encoding="utf-8")):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls not in TEXT_CLASSES:
                    continue
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = max(0, int((cx - bw / 2) * W))
                y1 = max(0, int((cy - bh / 2) * H))
                x2 = min(W, int((cx + bw / 2) * W))
                y2 = min(H, int((cy + bh / 2) * H))
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                text_boxes.append({
                    "idx": idx, "cls": cls,
                    "bbox": BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                })

            lines_from_this_page = 0
            boxes_used_this_page = set()
            for box_info in text_boxes:
                if len(cell_lines) >= TARGET_LINES_PER_CELL:
                    break
                if lines_from_this_page >= MAX_LINES_PER_PAGE:
                    break
                if box_info["idx"] in boxes_used_this_page:
                    continue

                box = box_info["bbox"]
                region_img = crop(img, box)
                if region_img.size == 0:
                    continue

                line_boxes = segment_lines(region_img, method="projection",
                                           min_line_height=12, smoothing=5)
                for lb_idx, lb in enumerate(line_boxes):
                    if len(cell_lines) >= TARGET_LINES_PER_CELL:
                        break
                    if lines_from_this_page >= MAX_LINES_PER_PAGE:
                        break
                    if lb.height < MIN_ORIG_H:
                        continue
                    line_crop_orig = crop(region_img, lb)
                    if line_crop_orig.size == 0:
                        continue
                    oh, ow = line_crop_orig.shape[:2]
                    scale = 32 / max(oh, 1)
                    resized_w = max(8, int(round(ow * scale)))
                    if resized_w < MIN_RESIZED_W:
                        continue

                    line_32 = preprocess_line_crop(line_crop_orig, target_height=32, binarize_crop=True)

                    # Page coordinates of this line
                    page_x1 = box.x1 + lb.x1
                    page_y1 = box.y1 + lb.y1
                    page_x2 = box.x1 + lb.x2
                    page_y2 = box.y1 + lb.y2

                    entry = {
                        "lang": lang, "domain": domain,
                        "image_file": fname,
                        "page_w": W, "page_h": H,
                        "box_idx": box_info["idx"],
                        "box_cls": box_info["cls"],
                        "line_idx_in_box": lb_idx,
                        "page_bbox": f"({int(page_x1)},{int(page_y1)},{int(page_x2)},{int(page_y2)})",
                        "orig_h": oh, "orig_w": ow,
                        "resized_w": resized_w,
                        "line_orig": line_crop_orig,
                        "line_32": line_32,
                    }

                    # Compute segment counts per threshold
                    THRESHOLDS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
                    for t in THRESHOLDS:
                        segs = segment_at_threshold(line_32, t)
                        entry[f"segs_t{t}"] = len(segs)

                    cell_lines.append(entry)
                    lines_from_this_page += 1
                    boxes_used_this_page.add(box_info["idx"])

            if lines_from_this_page > 0:
                pages_used_per_cell[(lang, domain)].add(fname)

        all_lines.extend(cell_lines)
        n_pages = len(pages_used_per_cell[(lang, domain)])
        med_h = float(np.median([e["orig_h"] for e in cell_lines])) if cell_lines else 0
        print(f"  ({lang:>7}, {domain:<15})  lines={len(cell_lines):3d}  pages={n_pages}  med_h={med_h:.1f}px")

    total_lines = len(all_lines)
    print(f"  Total lines: {total_lines}")

    # ── STEP 2: Assign ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Assign to annotators")

    # Split all_lines into per-cell groups
    cell_line_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, l in enumerate(all_lines):
        cell_line_groups[(l["lang"], l["domain"])].append(i)

    # Shared set: 3 per cell = 30 total
    shared_idxs = []
    remaining_per_cell: dict[tuple, list[int]] = {}
    for cell in sorted(cell_line_groups.keys()):
        idxs = cell_line_groups[cell]
        shared_idxs.extend(idxs[:3])
        remaining_per_cell[cell] = idxs[3:]

    # Transcription set: 6 per cell = 60 total, disjoint from shared
    transcription_idxs = []
    non_transcription: dict[tuple, list[int]] = {}
    for cell in sorted(remaining_per_cell.keys()):
        idxs = remaining_per_cell[cell]
        transcription_idxs.extend(idxs[:6])
        non_transcription[cell] = idxs[6:]

    # Remaining: 210 lines, split evenly across annotators
    remaining_all = []
    for cell in sorted(non_transcription.keys()):
        remaining_all.extend(non_transcription[cell])

    print(f"  Shared set       : {len(shared_idxs)} lines (all annotators)")
    print(f"  Transcription set: {len(transcription_idxs)} lines (split across annotators)")
    print(f"  Remaining        : {len(remaining_all)} lines (split across annotators)")

    # Assign
    assign_rng = random.Random(42)

    # Transcription: split evenly
    tr_shuffled = list(transcription_idxs)
    assign_rng.shuffle(tr_shuffled)
    tr_per_annotator = [[] for _ in range(N_ANNOTATORS)]
    for i, idx in enumerate(tr_shuffled):
        tr_per_annotator[i % N_ANNOTATORS].append(idx)

    # Remaining: split evenly
    rem_shuffled = list(remaining_all)
    assign_rng.shuffle(rem_shuffled)
    rem_per_annotator = [[] for _ in range(N_ANNOTATORS)]
    for i, idx in enumerate(rem_shuffled):
        rem_per_annotator[i % N_ANNOTATORS].append(idx)

    # Each annotator gets: shared (all) + their transcription + their remaining
    annotator_lines: list[list[int]] = []
    annotator_transcription_flags: list[set[int]] = []
    for k in range(N_ANNOTATORS):
        lines_k = list(shared_idxs) + tr_per_annotator[k] + rem_per_annotator[k]
        annotator_lines.append(lines_k)
        annotator_transcription_flags.append(set(tr_per_annotator[k]))
        print(f"  Annotator {k+1}: {len(lines_k)} lines "
              f"({len(shared_idxs)} shared + {len(tr_per_annotator[k])} transcription + "
              f"{len(rem_per_annotator[k])} remaining)")

    # ── STEP 3: Generate HTML files ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Generate HTML files")

    shared_ids = [f"L{i+1:03d}" for i in range(len(shared_idxs))]
    # Map line index -> opaque id
    line_id_map: dict[int, str] = {}
    for i, idx in enumerate(shared_idxs):
        line_id_map[idx] = shared_ids[i]
    next_id = len(shared_idxs) + 1
    for k in range(N_ANNOTATORS):
        for idx in annotator_lines[k]:
            if idx not in line_id_map:
                line_id_map[idx] = f"L{next_id:03d}"
                next_id += 1

    for k in range(N_ANNOTATORS):
        lines_k = annotator_lines[k]
        tr_flags_k = annotator_transcription_flags[k]

        # Shuffle order per annotator
        order_rng = random.Random(42 + k)
        order = list(lines_k)
        order_rng.shuffle(order)

        html_lines = []
        for idx in order:
            entry = all_lines[idx]
            lid = line_id_map[idx]
            is_transcription = idx in tr_flags_k
            # Upscale image 4x with INTER_CUBIC
            img_orig = entry["line_orig"]
            h, w = img_orig.shape[:2]
            big = cv2.resize(img_orig, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
            b64 = img_to_base64_png(big)
            html_lines.append({
                "id": lid,
                "b64": b64,
                "transcription": is_transcription,
            })

        html = build_html(k + 1, N_ANNOTATORS, html_lines)
        html_path = out_dir / f"labeling_member{k+1}.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"  {html_path.name}: {len(html_lines)} lines")

    # ── STEP 4: key.csv ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — key.csv")

    key_path = out_dir / "key.csv"
    THRESHOLDS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
    header = ["line_id", "language", "domain", "image_file", "page_size",
              "box_index", "box_class", "line_idx_in_box", "line_bbox_page",
              "orig_h", "orig_w", "annotators", "shared", "transcription"]
    header += [f"segs_t{t}" for t in THRESHOLDS]

    with open(key_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for idx in range(total_lines):
            entry = all_lines[idx]
            lid = line_id_map[idx]
            # Which annotators got this line
            annotators_who_got = []
            for k in range(N_ANNOTATORS):
                if idx in set(annotator_lines[k]):
                    annotators_who_got.append(str(k + 1))
            is_shared = idx in set(shared_idxs)
            is_transcription = idx in set(transcription_idxs)

            row = {
                "line_id": lid,
                "language": entry["lang"],
                "domain": entry["domain"],
                "image_file": entry["image_file"],
                "page_size": f"{entry['page_w']}x{entry['page_h']}",
                "box_index": entry["box_idx"],
                "box_class": entry["box_cls"],
                "line_idx_in_box": entry["line_idx_in_box"],
                "line_bbox_page": entry["page_bbox"],
                "orig_h": entry["orig_h"],
                "orig_w": entry["orig_w"],
                "annotators": ";".join(annotators_who_got),
                "shared": int(is_shared),
                "transcription": int(is_transcription),
            }
            for t in THRESHOLDS:
                row[f"segs_t{t}"] = entry[f"segs_t{t}"]
            writer.writerow(row)

    key_rows = sum(1 for _ in open(key_path)) - 1
    print(f"  key.csv: {key_rows} rows (expect 300)")

    # ── STEP 5: Self-checks ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Self-checks")

    # 5a: lines per HTML file — count JSON entries, not template string
    for k in range(N_ANNOTATORS):
        html_path = out_dir / f"labeling_member{k+1}.html"
        content = html_path.read_text(encoding="utf-8")
        # Count "b64" keys in the JSON array (each line has one)
        n_imgs = content.count('"b64":')
        print(f"  member{k+1}.html: {n_imgs} line images")

    # 5b: shared line ids identical across files
    shared_id_set = set(shared_ids)
    all_ok = True
    for k in range(N_ANNOTATORS):
        html_path = out_dir / f"labeling_member{k+1}.html"
        content = html_path.read_text(encoding="utf-8")
        for sid in shared_ids:
            if sid not in content:
                print(f"  MISSING shared id {sid} in member{k+1}.html")
                all_ok = False
    if all_ok:
        print(f"  All {len(shared_ids)} shared IDs present in all {N_ANNOTATORS} HTML files: OK")

    # Shared images byte-identical: check first shared line's base64 across files
    first_shared_b64 = None
    for k in range(N_ANNOTATORS):
        html_path = out_dir / f"labeling_member{k+1}.html"
        content = html_path.read_text(encoding="utf-8")
        # Find the first shared id's image
        import re as _re
        pattern = _re.escape(shared_ids[0]) + r'.*?data:image/png;base64,([A-Za-z0-9+/=]+)'
        m = _re.search(pattern, content, _re.DOTALL)
        if m:
            b64 = m.group(1)[:200]  # first 200 chars for comparison
            if first_shared_b64 is None:
                first_shared_b64 = b64
            elif b64 != first_shared_b64:
                print(f"  WARNING: shared image differs in member{k+1}.html")
                all_ok = False
    if all_ok and first_shared_b64:
        print(f"  Shared images byte-identical across files: OK")

    # 5c: grep each HTML for forbidden terms — search ONLY visible text,
    #     excluding base64 data blocks (which can coincidentally contain any letter sequence)
    forbidden = ["threshold", "cut", "hindi", "marathi", "acts_rules", "forms",
                 "newspaper", "notice", "question_paper"]
    forbidden_found = False
    for k in range(N_ANNOTATORS):
        html_path = out_dir / f"labeling_member{k+1}.html"
        content = html_path.read_text(encoding="utf-8")
        # Strip out the JSON data block (between LINES = and the first </script>)
        # The JSON contains base64 strings which can have any letter sequence
        import re as _re
        clean = _re.sub(r'const LINES = \[.*?\];', 'const LINES = [...];', content, flags=_re.DOTALL)
        clean_lower = clean.lower()
        for term in forbidden:
            if term.lower() in clean_lower:
                # Find the context to report
                pos = clean_lower.find(term.lower())
                context = clean[max(0,pos-20):pos+len(term)+20]
                print(f"  FORBIDDEN term '{term}' found in member{k+1}.html: ...{context}...")
                forbidden_found = True
    if not forbidden_found:
        print(f"  No forbidden terms (threshold/cut/language/domain) in any HTML: OK")

    # 5d: render 5 random line images to preview
    preview_rng = random.Random(42)
    preview_idxs = preview_rng.sample(range(total_lines), 5)
    for i, idx in enumerate(preview_idxs, 1):
        entry = all_lines[idx]
        lid = line_id_map[idx]
        img = entry["line_orig"]
        h, w = img.shape[:2]
        big = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(out_dir / "preview" / f"preview_{i:02d}_{lid}.png"), big)
    print(f"  Saved 5 preview images to {out_dir / 'preview'}")

    # ── Save metadata JSON ───────────────────────────────────────────────────
    meta = {
        "git_commit":    git_commit_hash(),
        "date":          str(date.today()),
        "weights_sha256": {"slim": weights_sha, "full": full_sha,
                           "charset": charset_sha, "layout": layout_sha},
        "seed":          42,
        "n_annotators":  N_ANNOTATORS,
        "total_lines":   total_lines,
        "shared_lines":  len(shared_idxs),
        "transcription_lines": len(transcription_idxs),
        "remaining_lines": len(remaining_all),
        "lines_per_annotator": [len(annotator_lines[k]) for k in range(N_ANNOTATORS)],
    }
    (out_dir / "labeling_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nMetadata written to {out_dir / 'labeling_meta.json'}")
    print("Done.")


# ──────────────────────── HTML template ──────────────────────────────────────

def build_html(member_num: int, total_members: int, lines: list[dict]) -> str:
    """Build a self-contained HTML labeling interface."""
    # Lines data as JSON
    lines_json = json.dumps([
        {"id": l["id"], "b64": l["b64"], "transcription": l["transcription"]}
        for l in lines
    ])

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Line Labeling — Member {member_num}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 1.4em; }}
  .instructions {{ background: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 0.9em; }}
  .instructions ul {{ margin: 5px 0; padding-left: 20px; }}
  .line-img {{ max-width: 100%; border: 1px solid #ccc; margin: 10px 0; image-rendering: pixelated; }}
  .controls {{ margin: 10px 0; padding: 10px; background: #fafafa; border-radius: 5px; }}
  .controls label {{ display: block; margin: 5px 0; }}
  .progress {{ font-weight: bold; color: #333; }}
  button {{ padding: 8px 20px; margin: 5px; font-size: 1em; cursor: pointer; border-radius: 4px; border: 1px solid #888; background: #fff; }}
  button:hover {{ background: #e8e8e8; }}
  textarea {{ width: 100%; min-height: 60px; font-size: 1em; padding: 5px; box-sizing: border-box; }}
  .saved {{ color: green; font-size: 0.8em; }}
</style>
</head>
<body>
<h1>Line Labeling — Member {member_num} of {total_members}</h1>

<div class="instructions">
<h3>Instructions</h3>
<ul>
  <li>A <b>word</b> is a printed token separated by spaces from neighbouring tokens.</li>
  <li>A standalone danda (।) or other punctuation counts as a word <b>only if</b> it is separated by a space from adjacent text.</li>
  <li>Numerals count as words.</li>
  <li>Hyphenated compounds count as <b>one</b> word.</li>
  <li>Enter the <b>number of words</b> you see in the line (0 if blank).</li>
  <li>Check <b>"not a text line"</b> if the image shows a rule line, border, stamp, blank area, or is unreadable.</li>
  <li>Check <b>"unsure"</b> if you cannot decide the word count.</li>
  <li>For transcription lines, type the line exactly as printed, words separated by single spaces, keeping punctuation as printed.</li>
  <li>Press <b>Enter</b> or click <b>Next</b> to advance. Use <b>Previous</b> to go back.</li>
  <li>Your work is saved automatically in your browser. Click <b>Download CSV</b> when finished.</li>
</ul>
</div>

<div class="progress" id="progress">Line 1 of {len(lines)}</div>
<div id="line-container"></div>
<div class="controls" id="controls">
  <label>Number of words: <input type="number" id="n_words" min="0" max="99" value="" style="width:60px"></label>
  <label><input type="checkbox" id="not_text"> Not a text line (rule, border, stamp, blank, unreadable)</label>
  <label><input type="checkbox" id="unsure"> Unsure</label>
  <div id="transcription-area" style="display:none">
    <label>Transcription (type the line exactly as printed):</label>
    <textarea id="transcript" placeholder="Type the line here..."></textarea>
  </div>
</div>
<div>
  <button id="prev" onclick="prev()">Previous</button>
  <button id="next" onclick="next()">Next</button>
  <button id="download" onclick="download()" style="float:right">Download CSV</button>
  <span class="saved" id="saved"></span>
</div>

<script>
const LINES = {lines_json};
const MEMBER = "{member_num}";
let current = 0;
const answers = {{}};

function loadAnswers() {{
  try {{
    const saved = localStorage.getItem("labeling_" + MEMBER);
    if (saved) Object.assign(answers, JSON.parse(saved));
  }} catch(e) {{}}
}}

function saveAnswer() {{
  const lid = LINES[current].id;
  answers[lid] = {{
    n_words: document.getElementById("n_words").value,
    not_text: document.getElementById("not_text").checked,
    unsure: document.getElementById("unsure").checked,
    transcript: document.getElementById("transcript").value
  }};
  try {{
    localStorage.setItem("labeling_" + MEMBER, JSON.stringify(answers));
    document.getElementById("saved").textContent = "Saved";
    setTimeout(() => document.getElementById("saved").textContent = "", 1000);
  }} catch(e) {{}}
}}

function showLine() {{
  const line = LINES[current];
  let html = '<img class="line-img" src="data:image/png;base64,' + line.b64 + '" alt="' + line.id + '">';
  html += '<div style="font-size:0.8em;color:#999">' + line.id + '</div>';
  document.getElementById("line-container").innerHTML = html;
  document.getElementById("progress").textContent = "Line " + (current+1) + " of " + LINES.length;
  document.getElementById("transcription-area").style.display = line.transcription ? "block" : "none";
  const a = answers[line.id] || {{}};
  document.getElementById("n_words").value = a.n_words || "";
  document.getElementById("not_text").checked = a.not_text || false;
  document.getElementById("unsure").checked = a.unsure || false;
  document.getElementById("transcript").value = a.transcript || "";
  document.getElementById("saved").textContent = "";
}}

function next() {{
  saveAnswer();
  if (current < LINES.length - 1) {{ current++; showLine(); }}
}}
function prev() {{
  saveAnswer();
  if (current > 0) {{ current--; showLine(); }}
}}

document.addEventListener("keydown", function(e) {{
  if (e.key === "Enter" && e.target.tagName !== "TEXTAREA") {{
    e.preventDefault();
    next();
  }}
}});

function download() {{
  saveAnswer();
  let csv = "line_id,annotator,n_words,not_text,unsure,transcript\\n";
  for (const line of LINES) {{
    const a = answers[line.id] || {{}};
    csv += line.id + "," + MEMBER + "," + (a.n_words||"") + "," + (a.not_text?1:0) + "," + (a.unsure?1:0) + ",\\"" + (a.transcript||"").replace(/\\"/g,'\\"\\"') + "\\"\\n";
  }}
  const blob = new Blob([csv], {{type:"text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "member_" + MEMBER + ".csv";
  a.click();
}}

loadAnswers();
showLine();
</script>
</body>
</html>'''


if __name__ == "__main__":
    main()
