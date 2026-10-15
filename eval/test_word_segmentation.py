#!/usr/bin/env python3
"""Word boundary recovery and end-to-end evaluation on stitched validation lines.

PART B only.  Validation split only.  Never opens index_test.csv.

Implements segment_words() here — does NOT modify any akshardrishti/ source file.

Usage
-----
    python eval/test_word_segmentation.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.recognize.base import build_recognizer               # noqa: E402
from akshardrishti.postprocess.text_repair import normalize_for_scoring # noqa: E402


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


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        cur = [prev[0] + 1]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def corpus_cer(preds: list[str], refs: list[str]) -> float:
    total_err = total_chars = 0
    for p, r in zip(preds, refs):
        pn = normalize_for_scoring(p)
        rn = normalize_for_scoring(r)
        total_err   += edit_distance(pn, rn)
        total_chars += len(rn)
    return total_err / total_chars if total_chars else 0.0


TARGET_H = 32


def load_and_resize(image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    scale = TARGET_H / h
    new_w = max(8, int(round(w * scale)))
    return cv2.resize(img, (new_w, TARGET_H),
                      interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


def stitch_line(crops: list[np.ndarray], gap: int, margin_l: int = 0) -> np.ndarray:
    total_w = margin_l + sum(c.shape[1] for c in crops) + gap * (len(crops) - 1)
    line = np.full((TARGET_H, total_w), 255, dtype=np.uint8)
    x = margin_l
    for i, c in enumerate(crops):
        line[:, x:x + c.shape[1]] = c
        x += c.shape[1]
        if i < len(crops) - 1:
            x += gap
    return line


# ──────────────────────────── segment_words ──────────────────────────────────

class WordBox(NamedTuple):
    x1: int
    x2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    def iou(self, other: "WordBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_x2 = min(self.x2, other.x2)
        inter = max(0, inter_x2 - inter_x1)
        union = (self.x2 - self.x1) + (other.x2 - other.x1) - inter
        return inter / union if union > 0 else 0.0


def segment_words(line_gray: np.ndarray, min_gap: int) -> list[WordBox]:
    """Recover word boxes from a stitched grayscale line image.

    Algorithm
    ---------
    1. Otsu binarize (0=ink, 255=background after inversion).
    2. Build a vertical ink projection: ink_col[x] = number of ink pixels
       in column x.
    3. A column is "empty" if ink_col[x] == 0.
    4. Runs of at least min_gap consecutive empty columns are gaps.
    5. The intervals between gaps are word boxes.

    Limitation: Mozhi crops are tight (no internal whitespace), so stitched
    gaps are entirely artificial.  This method cannot recover word boundaries
    in natural scans where inter-letter spacing approaches inter-word spacing.
    """
    if line_gray.ndim == 3:
        line_gray = cv2.cvtColor(line_gray, cv2.COLOR_BGR2GRAY)
    H, W = line_gray.shape[:2]
    if H == 0 or W == 0:
        return []

    # Binarize: ink pixels become 1
    _, bw = cv2.threshold(line_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink_col = (bw > 0).sum(axis=0)  # shape (W,)

    # Find runs of empty columns
    empty = (ink_col == 0)
    boxes: list[WordBox] = []
    word_start = 0
    i = 0
    while i < W:
        if empty[i]:
            # Measure gap run length
            gap_start = i
            while i < W and empty[i]:
                i += 1
            gap_end = i  # exclusive
            gap_len = gap_end - gap_start
            if gap_len >= min_gap:
                # Close the current word if it has content
                if gap_start > word_start:
                    boxes.append(WordBox(x1=word_start, x2=gap_start))
                word_start = gap_end
        else:
            i += 1

    # Close final word
    if word_start < W:
        boxes.append(WordBox(x1=word_start, x2=W))

    # Drop empty or zero-width boxes
    boxes = [b for b in boxes if b.width > 0]
    return boxes


# ──────────────────────── boundary metrics ───────────────────────────────────

def boundary_metrics(
    pred_boxes_list: list[list[WordBox]],
    true_boxes_list: list[list[WordBox]],
    iou_threshold: float = 0.5,
) -> dict:
    """Precision, recall, F1 of recovered word boxes over a set of lines."""
    tp = fp = fn = 0
    exact_count = 0
    for pred_boxes, true_boxes in zip(pred_boxes_list, true_boxes_list):
        matched_true = set()
        n_tp = 0
        for pb in pred_boxes:
            best_iou, best_j = 0.0, -1
            for j, tb in enumerate(true_boxes):
                if j in matched_true:
                    continue
                iou = pb.iou(tb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                matched_true.add(best_j)
                n_tp += 1
        tp += n_tp
        fp += len(pred_boxes) - n_tp
        fn += len(true_boxes) - n_tp
        if len(pred_boxes) == len(true_boxes):
            exact_count += 1

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    n    = len(pred_boxes_list)
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "exact_count_rate": exact_count / n if n else 0.0,
        "n_lines": n,
    }


def true_word_boxes(crops: list[np.ndarray], gap: int) -> list[WordBox]:
    """Ground-truth word boxes for a stitched line (margin=0)."""
    boxes = []
    x = 0
    for i, c in enumerate(crops):
        boxes.append(WordBox(x1=x, x2=x + c.shape[1]))
        x += c.shape[1]
        if i < len(crops) - 1:
            x += gap
    return boxes


# ─────────────────────────────────────────────── main ────────────────────────

def main() -> None:
    out_path = ROOT / "results/word_segmentation_val.json"
    if out_path.exists():
        print(f"Output already exists: {out_path}  — delete to re-run.")
        sys.exit(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── config + recognizer ───────────────────────────────────────────────────
    with open(ROOT / "configs/pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backend_name = cfg["recognize"]["deva_backend"]
    backend_cfg  = dict(cfg["recognize"]["backends"][backend_name])
    for key in ("weights", "charset"):
        bp = Path(backend_cfg[key])
        if not bp.is_absolute():
            backend_cfg[key] = str(ROOT / bp)

    weights_sha = sha256_of(Path(backend_cfg["weights"]))
    charset_sha = sha256_of(Path(backend_cfg["charset"]))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")

    recognizer = build_recognizer(backend_name, config=backend_cfg, device=device)
    recognizer.ensure_loaded()
    charset = recognizer.charset

    # ── reproduce groups (same as stress test) ────────────────────────────────
    val_csv = ROOT / "datasets/mozhi_raw/index_val.csv"   # ONLY val; never test
    all_rows = list(csv.DictReader(open(val_csv, encoding="utf-8")))

    FNAME_RE = re.compile(r'^(.+)_(\d+)_(\d+)_(\d+)\.(jpe?g|png)$', re.IGNORECASE)
    groups: dict[tuple, list] = defaultdict(list)
    for r in all_rows:
        fname = Path(r["image_path"]).name
        m = FNAME_RE.match(fname)
        if m:
            doc, a, b, k = m.group(1), m.group(2), m.group(3), int(m.group(4))
            groups[(doc, a, b)].append((k, r["text"], r["image_path"], r["language"]))

    consec_groups: dict[tuple, list] = {}
    for key, items in groups.items():
        sorted_items = sorted(items, key=lambda x: x[0])
        ks = [x[0] for x in sorted_items]
        if ks == list(range(1, len(ks) + 1)):
            consec_groups[key] = sorted_items

    NW_LIST   = [2, 3, 4, 6]
    GAP_LIST  = [8, 16, 24]
    MAX_LINES = 300
    SEED      = 42

    FULL_NW_FOR_RNG = [1, 2, 3, 4, 6, 8]   # replicate full stress-test RNG draw
    eligible  = {nw: [key for key, items in consec_groups.items() if len(items) >= nw]
                 for nw in FULL_NW_FOR_RNG}

    # Replay full stress-test RNG up to nw=6, then use our own RNG for nw=6
    # Actually: we draw 300 per nw here (different from stress test's 500).
    # Use a separate RNG seeded at 42 for this script to be reproducible.
    RNG = random.Random(SEED)
    sampled: dict[int, list] = {}
    for nw in NW_LIST:
        pool = eligible[nw]
        chosen_keys = RNG.sample(pool, min(MAX_LINES, len(pool)))
        entries = []
        for key in chosen_keys:
            items = consec_groups[key][:nw]
            words = [t for _, t, _, _ in items]
            paths = [p for _, _, p, _ in items]
            entries.append((words, paths))
        sampled[nw] = entries
        print(f"  n_w={nw}: {len(entries)} lines sampled")

    # ═══════════════════════════════════════════════════════════════════════════
    # B1 — Stitch lines and build ground-truth boxes
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("B1 — Stitch validation lines")

    # For each (nw, gap) pair, build (line_img, crops, true_boxes, ref) tuples
    line_data: dict[tuple, list] = {}   # (nw, gap) -> list of (line_img, crops, true_boxes, ref)

    for nw in NW_LIST:
        for gap in GAP_LIST:
            entries_data = []
            for words, paths in sampled[nw]:
                crops = [load_and_resize(p) for p in paths]
                if any(c is None for c in crops):
                    continue
                line_img  = stitch_line(crops, gap=gap, margin_l=0)
                tb        = true_word_boxes(crops, gap)
                ref       = " ".join(words)
                entries_data.append((line_img, crops, tb, ref))
            line_data[(nw, gap)] = entries_data

    total_lines = sum(len(v) for v in line_data.values())
    print(f"  Total stitched lines: {total_lines}")

    # ═══════════════════════════════════════════════════════════════════════════
    # B2 — Sweep min_gap; choose on tune half; report on check half
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("B2 — min_gap sweep (choose on tune half, report on check half)")

    MIN_GAP_VALUES = [4, 6, 8, 10, 12, 16]
    SPLIT_RNG = random.Random(SEED + 1)

    # Use gap=16, all nw for the tune/check split
    all_lines_g16: list[tuple] = []
    for nw in NW_LIST:
        all_lines_g16.extend(line_data[(nw, 16)])

    n_total = len(all_lines_g16)
    indices = list(range(n_total))
    SPLIT_RNG.shuffle(indices)
    half = n_total // 2
    tune_idxs  = indices[:half]
    check_idxs = indices[half:]
    tune_lines  = [all_lines_g16[i] for i in tune_idxs]
    check_lines = [all_lines_g16[i] for i in check_idxs]
    print(f"  tune={len(tune_lines)}  check={len(check_lines)}")

    best_gap = MIN_GAP_VALUES[0]
    best_f1  = -1.0
    tune_results: dict[int, dict] = {}

    for mg in MIN_GAP_VALUES:
        pred_boxes_list = [segment_words(l[0], mg) for l in tune_lines]
        true_boxes_list = [l[2] for l in tune_lines]
        m = boundary_metrics(pred_boxes_list, true_boxes_list)
        tune_results[mg] = m
        print(f"  tune min_gap={mg:2d}  F1={m['f1']:.4f}  prec={m['precision']:.4f}"
              f"  rec={m['recall']:.4f}  exact_count={m['exact_count_rate']:.4f}")
        if m["f1"] > best_f1:
            best_f1  = m["f1"]
            best_gap = mg

    print(f"\n  CHOSEN min_gap = {best_gap}  (best tune F1={best_f1:.4f})")

    # Report on check half
    check_pred_boxes = [segment_words(l[0], best_gap) for l in check_lines]
    check_true_boxes = [l[2] for l in check_lines]
    check_bm = boundary_metrics(check_pred_boxes, check_true_boxes)
    print(f"\n  CHECK half (min_gap={best_gap}): F1={check_bm['f1']:.4f}"
          f"  prec={check_bm['precision']:.4f}  rec={check_bm['recall']:.4f}"
          f"  exact_count_rate={check_bm['exact_count_rate']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # B3 — Boundary metrics on check half per n_w and per gap
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"B3 — Boundary metrics on check half (min_gap={best_gap})")

    # We need to know which n_w and gap each check_line came from.
    # Rebuild with metadata.
    all_lines_g16_meta = []
    for nw in NW_LIST:
        for entry in line_data[(nw, 16)]:
            all_lines_g16_meta.append((entry, nw, 16))

    # Re-apply the same shuffle
    SPLIT_RNG2 = random.Random(SEED + 1)
    idx2 = list(range(len(all_lines_g16_meta)))
    SPLIT_RNG2.shuffle(idx2)
    half2 = len(idx2) // 2
    check_meta = [all_lines_g16_meta[i] for i in idx2[half2:]]

    per_nw_bm: dict[int, dict] = {}
    for nw in NW_LIST:
        sub = [cm for cm in check_meta if cm[1] == nw]
        pbs = [segment_words(cm[0][0], best_gap) for cm in sub]
        tbs = [cm[0][2] for cm in sub]
        m   = boundary_metrics(pbs, tbs)
        per_nw_bm[nw] = m
        print(f"  n_w={nw}  F1={m['f1']:.4f}  prec={m['precision']:.4f}"
              f"  rec={m['recall']:.4f}  exact_count={m['exact_count_rate']:.4f}"
              f"  n={m['n_lines']}")

    # Per gap using all n_w, for gaps 8, 16, 24
    per_gap_bm: dict[int, dict] = {}
    for gap in GAP_LIST:
        all_for_gap = []
        for nw in NW_LIST:
            all_for_gap.extend(line_data[(nw, gap)])
        SPLIT_RNG3 = random.Random(SEED + 2 + gap)
        idx3 = list(range(len(all_for_gap)))
        SPLIT_RNG3.shuffle(idx3)
        half3 = len(idx3) // 2
        check3 = [all_for_gap[i] for i in idx3[half3:]]
        pbs3 = [segment_words(c[0], best_gap) for c in check3]
        tbs3 = [c[2] for c in check3]
        m3   = boundary_metrics(pbs3, tbs3)
        per_gap_bm[gap] = m3
        print(f"  gap={gap:2d}  F1={m3['f1']:.4f}  exact_count={m3['exact_count_rate']:.4f}"
              f"  n={m3['n_lines']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # B4 — End-to-end on check half: segment + recognize + join
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"B4 — End-to-end (segment min_gap={best_gap} -> recognize -> join)")

    def spaces_correct_frac(preds: list[str], refs: list[str]) -> float:
        n = len(preds)
        c = sum(normalize_for_scoring(p).count(" ") == normalize_for_scoring(r).count(" ")
                for p, r in zip(preds, refs))
        return c / n if n else 0.0

    def word_exact_acc(preds: list[str], refs: list[str]) -> float:
        correct = total = 0
        for p, r in zip(preds, refs):
            pw = normalize_for_scoring(p).split()
            rw = normalize_for_scoring(r).split()
            total   += len(rw)
            correct += sum(a == b for a, b in zip(pw, rw))
        return correct / total if total else 0.0

    e2e_results: dict = {}
    for nw in NW_LIST:
        sub = [cm for cm in check_meta if cm[1] == nw]
        e2e_preds = []
        e2e_refs  = []
        for cm in sub:
            line_img, crops, true_boxes, ref = cm[0]
            pred_boxes = segment_words(line_img, best_gap)
            if not pred_boxes:
                e2e_preds.append("")
                e2e_refs.append(ref)
                continue
            # Crop each predicted word box and recognize
            word_preds = []
            for pb in pred_boxes:
                word_crop = line_img[:, pb.x1:pb.x2]
                result    = recognizer.recognize(word_crop, "deva")
                word_preds.append(result.text)
            joined = " ".join(word_preds)
            e2e_preds.append(joined)
            e2e_refs.append(ref)

        cer_e2e  = corpus_cer(e2e_preds, e2e_refs)
        wacc_e2e = word_exact_acc(e2e_preds, e2e_refs)
        sc_frac  = spaces_correct_frac(e2e_preds, e2e_refs)
        e2e_results[str(nw)] = {
            "n": len(sub),
            "cer": cer_e2e,
            "word_exact_acc": wacc_e2e,
            "spaces_correct_frac": sc_frac,
        }
        print(f"  n_w={nw}  CER={cer_e2e:.4f}  word_acc={wacc_e2e:.4f}"
              f"  spaces_correct={sc_frac:.3f}  n={len(sub)}")

    # Compare to whole-line results at gap=16
    # Load from the stress test json
    stress_json = ROOT / "results/crnn_line_stress_val.json"
    whole_line_compare: dict = {}
    if stress_json.exists():
        stress = json.loads(stress_json.read_text(encoding="utf-8"))
        print("\n  Comparison to whole-line (gap=16) from stress test:")
        print(f"  {'n_w':>4}  {'e2e_CER':>10}  {'whole_CER':>10}")
        for nw in NW_LIST:
            key = f"nw{nw}_g16_m0"
            wl_cer = stress["step2_metrics"].get(key, {}).get("cer", None)
            e2e_cer = e2e_results[str(nw)]["cer"]
            print(f"  {nw:>4}  {e2e_cer:>10.4f}  {str(wl_cer)[:10]:>10}")
            whole_line_compare[str(nw)] = {
                "e2e_segment_cer": e2e_cer,
                "whole_line_cer": wl_cer,
            }

    # ═══════════════════════════════════════════════════════════════════════════
    # B5 — Limitation note
    # ═══════════════════════════════════════════════════════════════════════════
    limitation_note = (
        "Mozhi crops are tightly bounded single-word images with no internal "
        "whitespace. The inter-word gaps in stitched lines are entirely artificial "
        "(inserted by the stitching code) and do not reflect the spacing in natural "
        "document scans. The segment_words() algorithm exploits these clean artificial "
        "gaps; it would perform much worse on real document line images where "
        "inter-letter spacing within a conjunct cluster can approach or exceed "
        "the inter-word spacing. Results here measure the ceiling for gap-based "
        "segmentation on clean Mozhi crops, not real-world document segmentation."
    )
    print(f"\n  Limitation: {limitation_note}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Save JSON
    # ═══════════════════════════════════════════════════════════════════════════
    result = {
        "git_commit":    git_commit_hash(),
        "date":          str(date.today()),
        "weights_sha256": weights_sha,
        "charset_sha256": charset_sha,
        "device":        device,
        "seed":          SEED,
        "split":         "validation",
        "chosen_min_gap": best_gap,
        "B1": {
            "n_w_list": NW_LIST,
            "gap_list": GAP_LIST,
            "lines_per_nw_gap": {f"nw{nw}_g{g}": len(line_data[(nw, g)])
                                  for nw in NW_LIST for g in GAP_LIST},
        },
        "B2_tune": {str(mg): m for mg, m in tune_results.items()},
        "B2_check": check_bm,
        "B3_per_nw": {str(nw): m for nw, m in per_nw_bm.items()},
        "B3_per_gap": {str(g): m for g, m in per_gap_bm.items()},
        "B4_e2e": e2e_results,
        "B4_whole_line_compare": whole_line_compare,
        "B5_limitation": limitation_note,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
