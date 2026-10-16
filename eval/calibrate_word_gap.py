#!/usr/bin/env python3
"""Calibrate word-gap threshold using a vocabulary proxy on IndicDLP train pages.

IndicDLP TRAIN-split pages only. Never opens images/val, index_test.csv,
or any Mozhi test file.

Usage
-----
    python eval/calibrate_word_gap.py
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

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.layout.lines import segment_lines, crop            # noqa: E402
from akshardrishti.preprocess.enhance import preprocess_line_crop      # noqa: E402
from akshardrishti.schema import BBox                                 # noqa: E402
from akshardrishti.recognize.base import build_recognizer             # noqa: E402
from akshardrishti.recognize.crnn_model import Charset, greedy_decode # noqa: E402
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


# Punctuation to strip from segment text for lexical check
STRIP_CHARS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~।॥\u2018\u2019\u201c\u201d\u2013\u2014")
DEVA_RANGE = re.compile(r'[\u0900-\u097F]')


def strip_punct(text: str) -> str:
    """Remove leading/trailing punctuation (ASCII + Devanagari danda + curly quotes)."""
    return text.strip(''.join(STRIP_CHARS))


def is_lexical(text: str) -> bool:
    """A segment is LEXICAL if it contains at least one Devanagari letter after strip()."""
    stripped = strip_punct(text)
    return bool(DEVA_RANGE.search(stripped))


def deva_char_count(text: str) -> int:
    return len(DEVA_RANGE.findall(text))


# ─────────────────────────────── main ────────────────────────────────────────

def main() -> None:
    out_path = ROOT / "results/word_gap_calibration.json"
    if out_path.exists():
        print(f"Output already exists: {out_path}  — delete to re-run.")
        sys.exit(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── config + recognizer ──────────────────────────────────────────────────
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
    layout_sha   = sha256_of(ROOT / cfg["layout"]["weights"].lstrip("./"))
    full_sha     = sha256_of(ROOT / "weights/recognize/crnn_mozhi_himr_full.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")

    recognizer = build_recognizer(backend_name, config=backend_cfg, device=device)
    recognizer.ensure_loaded()
    charset: Charset = recognizer.charset
    model = recognizer.model

    # ═══════════════════════════════════════════════════════════════════════════
    # PART A — Report only
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART A — Report only")

    # A1: DIP code description
    print("\nA1 — DIP detection code from measure_real_word_gaps.py:")
    print("  short_peak = max(hist[:6])  # peak count in bins 0-10px")
    print("  Scans bins 2..9 (4px..20px):")
    print("  dip_idx = first bin where count < 30% of short_peak AND count > 0")
    print("  This looks for the first valley (low point) after the short-run hump.")
    print("  It does NOT explicitly look for a second hump — it finds the first")
    print("  bin that drops below 30% of the short-run peak, which is the valley")
    print("  between the two humps if a bimodal distribution exists.")

    # A2: mode1, mode2, valley from JSON
    print("\nA2 — Mode1/Mode2/Valley analysis from real_gap_measurement.json")
    gap_json = json.loads((ROOT / "results/real_gap_measurement.json").read_text(encoding="utf-8"))

    # Parse histogram bins
    def analyze_hist(histogram: dict) -> dict:
        bins = sorted(histogram.keys(), key=lambda k: int(k.split("-")[0]))
        counts = [histogram[b] for b in bins]
        # mode1 = highest bin below 6px (bins 0-2, 2-4, 4-6)
        mode1_idx = max(range(min(3, len(counts))), key=lambda i: counts[i])
        mode1_bin = bins[mode1_idx]
        mode1_count = counts[mode1_idx]
        # mode2 = highest bin at or above 8px (bins 8-10 onward)
        long_bins_start = 4  # bin index for 8-10px
        if long_bins_start >= len(counts):
            return {"mode1_bin": mode1_bin, "mode2": "no second hump", "valley": "n/a", "valley_ratio": "n/a"}
        mode2_idx = max(range(long_bins_start, len(counts)), key=lambda i: counts[i])
        mode2_bin = bins[mode2_idx]
        mode2_count = counts[mode2_idx]
        # valley = lowest bin between mode1 and mode2
        lo = min(mode1_idx, mode2_idx) + 1
        hi = max(mode1_idx, mode2_idx)
        if lo >= hi:
            valley_bin = "n/a"
            valley_count = 0
        else:
            valley_idx = min(range(lo, hi), key=lambda i: counts[i])
            valley_bin = bins[valley_idx]
            valley_count = counts[valley_idx]
        valley_ratio = valley_count / mode2_count if mode2_count > 0 else float("inf")
        has_hump = mode2_count >= 1.2 * valley_count if valley_count > 0 else True
        return {
            "mode1_bin": mode1_bin, "mode1_count": mode1_count,
            "mode2_bin": mode2_bin if has_hump else "no second hump",
            "mode2_count": mode2_count,
            "valley_bin": valley_bin, "valley_count": valley_count,
            "valley_ratio": round(valley_ratio, 4),
            "has_second_hump": has_hump,
        }

    print(f"\n  {'level':<25}  {'mode1':>8}  {'mode2':>8}  {'valley':>8}  {'v/m2':>6}  {'2nd hump?':>10}")
    print(f"  {'-'*75}")

    a2_results = {}
    for level_key in ["B4_per_language", "B4_per_domain", "B4_per_lang_domain"]:
        for name, data in sorted(gap_json[level_key].items()):
            h = data["histogram"]
            a = analyze_hist(h)
            a2_results[f"{level_key}/{name}"] = a
            m2 = a["mode2_bin"] if isinstance(a["mode2_bin"], str) else f"{a['mode2_bin']}({a['mode2_count']})"
            vl = a["valley_bin"] if isinstance(a["valley_bin"], str) else f"{a['valley_bin']}({a['valley_count']})"
            hump = "YES" if a["has_second_hump"] else "NO"
            print(f"  {name:<25}  {a['mode1_bin']:>8}  {m2:>8}  {vl:>8}  {a['valley_ratio']:>6}  {hump:>10}")

    # A3: contact sheet
    cs_path = ROOT / "results/real_gap_contact_sheet.png"
    cs_size = cs_path.stat().st_size if cs_path.exists() else 0
    print(f"\nA3 — Contact sheet: {cs_path.name}  exists={cs_path.exists()}  size={cs_size/1024:.1f} KB")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART B — Proxy calibration
    # ═══════════════════════════════════════════════════════════════════════════

    # B1: Vocabulary
    print("\n" + "=" * 60)
    print("B1 — Vocabulary")

    def load_vocab(path: Path) -> set[str]:
        words = set()
        for line in open(path, encoding="utf-8"):
            w = unicodedata.normalize("NFC", line.strip())
            if w:
                words.add(w)
        return words

    hindi_vocab = load_vocab(ROOT / "datasets/mozhi_raw/hindi/train/vocabulary.txt")
    marathi_vocab = load_vocab(ROOT / "datasets/mozhi_raw/marathi/train/vocabulary.txt")
    union_vocab = hindi_vocab | marathi_vocab
    print(f"  Hindi vocab size   : {len(hindi_vocab)}")
    print(f"  Marathi vocab size : {len(marathi_vocab)}")
    print(f"  Union vocab size   : {len(union_vocab)}")

    # Ceiling: fraction of lexical Mozhi val words in vocabulary
    # Uses index_val.csv (validation, NOT test)
    val_csv = ROOT / "datasets/mozhi_raw/index_val.csv"
    val_rows = list(csv.DictReader(open(val_csv, encoding="utf-8")))
    kept_val = [r for r in val_rows if charset.coverage(r["text"]) >= 0.999]

    ceiling = {}
    for lang in ["hindi", "marathi"]:
        lang_rows = [r for r in kept_val if r["language"] == lang]
        lexical_words = []
        for r in lang_rows:
            w = unicodedata.normalize("NFC", r["text"])
            sw = strip_punct(w)
            if DEVA_RANGE.search(sw):
                lexical_words.append(sw)
        in_vocab = sum(1 for w in lexical_words if w in union_vocab)
        rate = in_vocab / len(lexical_words) if lexical_words else 0.0
        ceiling[lang] = {"n_lexical": len(lexical_words), "n_in_vocab": in_vocab, "rate": rate}
        print(f"  Ceiling {lang:>7}: {in_vocab}/{len(lexical_words)} = {rate:.4f}")

    # B2: Sample pages and segment lines (40 pages per cell)
    print("\n" + "=" * 60)
    print("B2 — Sample pages and segment lines (40 per cell)")

    manifest_path = ROOT / "split_manifest.csv"
    all_manifest = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    train_hm = [r for r in all_manifest
                if r["split"] == "train" and r["language"] in ("hindi", "marathi")]

    cells: dict[tuple, list[str]] = defaultdict(list)
    for r in train_hm:
        cells[(r["language"], r["domain"])].append(r["filename"])

    RNG = random.Random(42)
    MAX_PAGES = 40
    MAX_LINES_PER_IMAGE = 40
    MIN_ORIG_H = 12
    MIN_RESIZED_W = 64
    TEXT_CLASSES = {0, 1, 2}

    labels_dir = ROOT / "datasets/indicdlp_subset/labels/train"
    images_dir = ROOT / "datasets/indicdlp_subset/images/train"

    # Sample pages per cell
    cell_pages: dict[tuple, list[str]] = {}
    for (lang, domain) in sorted(cells.keys()):
        pool = sorted(cells[(lang, domain)])
        chosen = RNG.sample(pool, min(MAX_PAGES, len(pool)))
        cell_pages[(lang, domain)] = chosen

    # Split pages 50/50 into tune and check
    all_sampled_pages = []
    page_to_cell = {}
    for (lang, domain), pages in sorted(cell_pages.items()):
        for p in pages:
            all_sampled_pages.append(p)
            page_to_cell[p] = (lang, domain)

    SPLIT_RNG = random.Random(42)
    page_indices = list(range(len(all_sampled_pages)))
    SPLIT_RNG.shuffle(page_indices)
    half = len(page_indices) // 2
    tune_pages = set(all_sampled_pages[i] for i in page_indices[:half])
    check_pages = set(all_sampled_pages[i] for i in page_indices[half:])

    print(f"  Total sampled pages: {len(all_sampled_pages)}  tune={len(tune_pages)}  check={len(check_pages)}")

    # Build lines for each half
    def build_lines(page_set: set[str]) -> dict[tuple, list[dict]]:
        """Return {(lang, domain): [line_dict, ...]}"""
        result: dict[tuple, list[dict]] = defaultdict(list)
        for (lang, domain), pages in sorted(cell_pages.items()):
            for fname in pages:
                if fname not in page_set:
                    continue
                stem = Path(fname).stem
                label_file = labels_dir / f"{stem}.txt"
                image_file = images_dir / fname
                if not label_file.exists() or not image_file.exists():
                    continue
                img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                H, W = img.shape[:2]
                text_boxes = []
                for line in open(label_file, encoding="utf-8"):
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
                    text_boxes.append(BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))

                lines_this_image = 0
                for box in text_boxes:
                    if lines_this_image >= MAX_LINES_PER_IMAGE:
                        break
                    region_img = crop(img, box)
                    if region_img.size == 0:
                        continue
                    line_boxes = segment_lines(region_img, method="projection",
                                               min_line_height=12, smoothing=5)
                    for lb in line_boxes[:MAX_LINES_PER_IMAGE]:
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
                        result[(lang, domain)].append({
                            "line_32": line_32,
                            "line_orig": line_crop_orig,
                            "resized_w": resized_w,
                            "image": fname,
                            "lang": lang,
                            "domain": domain,
                        })
                        lines_this_image += 1
        return result

    tune_lines = build_lines(tune_pages)
    check_lines = build_lines(check_pages)

    tune_total = sum(len(v) for v in tune_lines.values())
    check_total = sum(len(v) for v in check_lines.values())
    print(f"  Tune lines : {tune_total}  Check lines: {check_total}")
    for (lang, domain) in sorted(tune_lines.keys()):
        tl = len(tune_lines[(lang, domain)])
        cl = len(check_lines.get((lang, domain), []))
        print(f"    ({lang:>7}, {domain:<15})  tune={tl:4d}  check={cl:4d}")

    # B3: Threshold sweep
    print("\n" + "=" * 60)
    print("B3 — Threshold sweep")

    THRESHOLDS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]

    def segment_at_threshold(line_32: np.ndarray, t: int) -> list[tuple[int, int]]:
        """Cut at internal runs of empty columns >= t. Returns list of (x1, x2) column ranges."""
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

    def run_threshold(lines_dict, t: int, recognizer, charset, device, model):
        """Run threshold t on all lines; return per-line segment data + predictions.
        Uses batched greedy decode for speed (beam is 10x slower and we only
        need the text for vocabulary lookup, not publication-quality output)."""
        all_segments = []
        # Collect all segment crops first, then batch-recognize
        seg_crops: list[np.ndarray] = []
        seg_meta: list[tuple] = []  # (lang, domain, x1, x2, width)

        for (lang, domain), lines in sorted(lines_dict.items()):
            for l in lines:
                line_32 = l["line_32"]
                line_orig = l["line_orig"]
                segs = segment_at_threshold(line_32, t)
                if not segs:
                    continue
                for x1, x2 in segs:
                    seg_crop = line_orig[:, x1:x2]
                    if seg_crop.size == 0 or seg_crop.shape[1] < 2:
                        seg_crops.append(np.full((32, 8), 255, np.uint8))
                        seg_meta.append((lang, domain, x1, x2, x2 - x1))
                        continue
                    seg_crops.append(seg_crop)
                    seg_meta.append((lang, domain, x1, x2, x2 - x1))

        # Batched greedy decode grouped by width
        model.eval()
        # Prepare all tensors once, group by width
        all_tensors: list[torch.Tensor] = []
        for i, c in enumerate(seg_crops):
            all_tensors.append(recognizer._prepare(c))
        width_to_idxs: dict[int, list[int]] = defaultdict(list)
        for i, tnsr in enumerate(all_tensors):
            width_to_idxs[tnsr.shape[-1]].append(i)

        preds_map: dict[int, str] = {}
        with torch.no_grad():
            for width, idxs in width_to_idxs.items():
                for batch_start in range(0, len(idxs), 64):
                    batch_idxs = idxs[batch_start:batch_start + 64]
                    batch = torch.cat([all_tensors[i] for i in batch_idxs], dim=0).to(device)
                    lp = model(batch)
                    decoded = greedy_decode(lp, charset)
                    for pos, i in enumerate(batch_idxs):
                        preds_map[i] = decoded[pos][0]

        for i, (lang, domain, x1, x2, w) in enumerate(seg_meta):
            all_segments.append((lang, domain, x1, x2, w, preds_map.get(i, "")))

        return all_segments

    def compute_metrics(segments, vocab: set[str], label: str):
        """Stub kept for compatibility."""
        pass

    def compute_stats(segs, vocab, n_lines):
        total_segs = len(segs)
        if total_segs == 0:
            return {"segments_per_line": 0, "n_lexical": 0, "in_vocab_rate": 0,
                    "char_in_vocab_rate": 0, "fragment_rate": 0, "mean_seg_w": 0}

        lexical_segs = []
        for s in segs:
            text = unicodedata.normalize("NFC", s[5])
            if is_lexical(text):
                lexical_segs.append((s, text))

        n_lex = len(lexical_segs)
        in_vocab_count = 0
        char_in_vocab = 0
        char_all_lex = 0
        frag_count = 0
        widths = []

        for s, text in lexical_segs:
            sw = strip_punct(text)
            widths.append(s[4])
            dc = deva_char_count(sw)
            char_all_lex += dc
            if dc <= 2:
                frag_count += 1
            if sw in vocab:
                in_vocab_count += 1
                char_in_vocab += dc

        in_vocab_rate = in_vocab_count / n_lex if n_lex else 0
        char_in_vocab_rate = char_in_vocab / char_all_lex if char_all_lex else 0
        frag_rate = frag_count / n_lex if n_lex else 0
        mean_w = np.mean(widths) if widths else 0
        segs_per_line = total_segs / n_lines if n_lines else 0

        return {
            "n_segments": total_segs,
            "segments_per_line": round(segs_per_line, 3),
            "n_lexical": n_lex,
            "in_vocab_rate": round(in_vocab_rate, 4),
            "char_in_vocab_rate": round(char_in_vocab_rate, 4),
            "fragment_rate": round(frag_rate, 4),
            "mean_seg_w": round(mean_w, 1),
        }

    # For each threshold, run on tune and check; STORE segments for B4 reuse
    all_results: dict[int, dict] = {}
    tune_segs_by_t: dict[int, list] = {}
    check_segs_by_t: dict[int, list] = {}

    for t in THRESHOLDS:
        print(f"\n  Threshold t={t}")
        tune_segs = run_threshold(tune_lines, t, recognizer, charset, device, model)
        check_segs = run_threshold(check_lines, t, recognizer, charset, device, model)
        tune_segs_by_t[t] = tune_segs
        check_segs_by_t[t] = check_segs
        tune_stats = compute_stats(tune_segs, union_vocab, tune_total)
        check_stats = compute_stats(check_segs, union_vocab, check_total)
        all_results[t] = {"tune": tune_stats, "check": check_stats}

        print(f"    tune : segs/line={tune_stats['segments_per_line']:.2f}"
              f"  lex={tune_stats['n_lexical']}"
              f"  inV={tune_stats['in_vocab_rate']:.4f}"
              f"  charInV={tune_stats['char_in_vocab_rate']:.4f}"
              f"  frag={tune_stats['fragment_rate']:.4f}"
              f"  meanW={tune_stats['mean_seg_w']:.1f}")
        print(f"    check: segs/line={check_stats['segments_per_line']:.2f}"
              f"  lex={check_stats['n_lexical']}"
              f"  inV={check_stats['in_vocab_rate']:.4f}"
              f"  charInV={check_stats['char_in_vocab_rate']:.4f}"
              f"  frag={check_stats['fragment_rate']:.4f}"
              f"  meanW={check_stats['mean_seg_w']:.1f}")

    # B4: per language and per domain — reuse stored segments from B3
    print("\n" + "=" * 60)
    print("B4 — Per language and per domain (tune)")

    per_lang_tune: dict[int, dict[str, dict]] = {t: {} for t in THRESHOLDS}
    per_domain_tune: dict[int, dict[str, dict]] = {t: {} for t in THRESHOLDS}

    for t in THRESHOLDS:
        tune_segs = tune_segs_by_t[t]  # reuse from B3 — no re-recognition
        by_lang: dict[str, list] = defaultdict(list)
        by_domain: dict[str, list] = defaultdict(list)
        for s in tune_segs:
            by_lang[s[0]].append(s)
            by_domain[s[1]].append(s)
        lines_per_lang = defaultdict(int)
        lines_per_domain = defaultdict(int)
        for (lang, domain), lines in tune_lines.items():
            lines_per_lang[lang] += len(lines)
            lines_per_domain[domain] += len(lines)
        for lang in ["hindi", "marathi"]:
            stats = compute_stats(by_lang[lang], union_vocab, lines_per_lang[lang])
            per_lang_tune[t][lang] = stats
        for domain in sorted(lines_per_domain.keys()):
            stats = compute_stats(by_domain[domain], union_vocab, lines_per_domain[domain])
            per_domain_tune[t][domain] = stats
        print(f"  t={t}: hindi charInV={per_lang_tune[t]['hindi']['char_in_vocab_rate']:.4f}"
              f"  marathi={per_lang_tune[t]['marathi']['char_in_vocab_rate']:.4f}")

    # Print table for char_in_vocab_rate per language at each threshold
    print(f"\n  Char-weighted in-vocab rate (tune):")
    print(f"  {'t':>4}  {'hindi':>8}  {'marathi':>8}  {'acts':>8}  {'forms':>8}  {'news':>8}  {'notice':>8}  {'qp':>8}")
    for t in THRESHOLDS:
        h = per_lang_tune[t]["hindi"]["char_in_vocab_rate"]
        m = per_lang_tune[t]["marathi"]["char_in_vocab_rate"]
        a = per_domain_tune[t].get("acts_rules", {}).get("char_in_vocab_rate", 0)
        f = per_domain_tune[t].get("forms", {}).get("char_in_vocab_rate", 0)
        n = per_domain_tune[t].get("newspaper", {}).get("char_in_vocab_rate", 0)
        no = per_domain_tune[t].get("notice", {}).get("char_in_vocab_rate", 0)
        q = per_domain_tune[t].get("question_paper", {}).get("char_in_vocab_rate", 0)
        print(f"  {t:>4}  {h:>8.4f}  {m:>8.4f}  {a:>8.4f}  {f:>8.4f}  {n:>8.4f}  {no:>8.4f}  {q:>8.4f}")

    # Also compute check per-language and per-domain for B4 completeness
    per_lang_check: dict[int, dict[str, dict]] = {t: {} for t in THRESHOLDS}
    per_domain_check: dict[int, dict[str, dict]] = {t: {} for t in THRESHOLDS}
    for t in THRESHOLDS:
        check_segs = check_segs_by_t[t]
        by_lang_c: dict[str, list] = defaultdict(list)
        by_domain_c: dict[str, list] = defaultdict(list)
        for s in check_segs:
            by_lang_c[s[0]].append(s)
            by_domain_c[s[1]].append(s)
        lines_per_lang_c = defaultdict(int)
        lines_per_domain_c = defaultdict(int)
        for (lang, domain), lines in check_lines.items():
            lines_per_lang_c[lang] += len(lines)
            lines_per_domain_c[domain] += len(lines)
        for lang in ["hindi", "marathi"]:
            per_lang_check[t][lang] = compute_stats(by_lang_c[lang], union_vocab, lines_per_lang_c[lang])
        for domain in sorted(lines_per_domain_c.keys()):
            per_domain_check[t][domain] = compute_stats(by_domain_c[domain], union_vocab, lines_per_domain_c[domain])

    # Print check table
    print(f"\n  Char-weighted in-vocab rate (check):")
    print(f"  {'t':>4}  {'hindi':>8}  {'marathi':>8}  {'acts':>8}  {'forms':>8}  {'news':>8}  {'notice':>8}  {'qp':>8}")
    for t in THRESHOLDS:
        h = per_lang_check[t]["hindi"]["char_in_vocab_rate"]
        m = per_lang_check[t]["marathi"]["char_in_vocab_rate"]
        a = per_domain_check[t].get("acts_rules", {}).get("char_in_vocab_rate", 0)
        f = per_domain_check[t].get("forms", {}).get("char_in_vocab_rate", 0)
        n = per_domain_check[t].get("newspaper", {}).get("char_in_vocab_rate", 0)
        no = per_domain_check[t].get("notice", {}).get("char_in_vocab_rate", 0)
        q = per_domain_check[t].get("question_paper", {}).get("char_in_vocab_rate", 0)
        print(f"  {t:>4}  {h:>8.4f}  {m:>8.4f}  {a:>8.4f}  {f:>8.4f}  {n:>8.4f}  {no:>8.4f}  {q:>8.4f}")

    # B5: Choose t on tune
    print("\n" + "=" * 60)
    print("B5 — Choose t on TUNE half")

    best_t = THRESHOLDS[0]
    best_rate = -1.0
    for t in THRESHOLDS:
        r = all_results[t]["tune"]["char_in_vocab_rate"]
        if r > best_rate:
            best_rate = r
            best_t = t
    # Check for ties within 0.002
    ties = [t for t in THRESHOLDS if all_results[t]["tune"]["char_in_vocab_rate"] >= best_rate - 0.002]
    if len(ties) > 1:
        best_t = min(ties, key=lambda t: all_results[t]["tune"]["fragment_rate"])
        print(f"  Tie within 0.002: {ties} -> choosing lowest fragment rate: t={best_t}")
    else:
        print(f"  Best char_in_vocab on tune: t={best_t}  rate={best_rate:.4f}")

    check_rate = all_results[best_t]["check"]["char_in_vocab_rate"]
    print(f"  Check half at t={best_t}: char_in_vocab={check_rate:.4f}")

    # Per-domain choice
    per_domain_best: dict[str, int] = {}
    for domain in sorted(per_domain_tune[THRESHOLDS[0]].keys()):
        best_dt = THRESHOLDS[0]
        best_dr = -1.0
        for t in THRESHOLDS:
            r = per_domain_tune[t][domain]["char_in_vocab_rate"]
            if r > best_dr:
                best_dr = r
                best_dt = t
        ties_d = [t for t in THRESHOLDS if per_domain_tune[t][domain]["char_in_vocab_rate"] >= best_dr - 0.002]
        if len(ties_d) > 1:
            best_dt = min(ties_d, key=lambda t: per_domain_tune[t][domain]["fragment_rate"])
        per_domain_best[domain] = best_dt
        check_dt_rate = per_domain_check[best_dt][domain]["char_in_vocab_rate"]
        global_check_rate = per_domain_check[best_t][domain]["char_in_vocab_rate"]
        beats = check_dt_rate - global_check_rate > 0.01
        print(f"  domain={domain:<15}  best_t={best_dt}  tune_rate={best_dr:.4f}"
              f"  check_dt={check_dt_rate:.4f}  check_global={global_check_rate:.4f}"
              f"  beats_global_by>0.01: {'YES' if beats else 'NO'}")

    print(f"\n  Ceiling from B1: hindi={ceiling['hindi']['rate']:.4f}  marathi={ceiling['marathi']['rate']:.4f}")
    print(f"  Best global t={best_t} check rate={check_rate:.4f}")
    print(f"  The ceiling sits above the measured rates because:")
    print(f"  (1) recognizer errors lower the rate, (2) names/numbers are OOV,")
    print(f"  (3) the ceiling uses clean Mozhi word crops, not segmented line crops.")

    # B6: Print 12 check-half lines and save visualizations
    print("\n" + "=" * 60)
    print(f"B6 — 12 check-half example lines at t={best_t}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples_dir = ROOT / "results/gap_calibration_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    hindi_shown = 0
    marathi_shown = 0
    example_lines = []
    for (lang, domain), lines in sorted(check_lines.items()):
        for l in lines:
            if lang == "hindi" and hindi_shown >= 6:
                continue
            if lang == "marathi" and marathi_shown >= 6:
                continue
            line_32 = l["line_32"]
            line_orig = l["line_orig"]
            segs = segment_at_threshold(line_32, best_t)
            if not segs:
                continue
            # Recognize segments
            seg_texts = []
            for x1, x2 in segs:
                seg_crop = line_orig[:, x1:x2]
                if seg_crop.size == 0 or seg_crop.shape[1] < 2:
                    seg_texts.append("")
                    continue
                result = recognizer.recognize(seg_crop, "deva")
                seg_texts.append(result.text)
            joined = " | ".join(seg_texts)
            example_lines.append({
                "lang": lang, "domain": domain, "image": l["image"],
                "segments": joined, "n_segs": len(segs),
                "cut_positions": [s[0] for s in segs],
            })
            if lang == "hindi":
                hindi_shown += 1
            else:
                marathi_shown += 1

            idx = len(example_lines)
            fig, ax = plt.subplots(1, 1, figsize=(12, 1.5))
            ax.imshow(line_32, cmap="gray", aspect="auto")
            for x in segs:
                ax.axvline(x[0], color="red", linewidth=0.8)
            ax.set_title(f"#{idx} {lang} {domain} t={best_t}", fontsize=7)
            ax.axis("off")
            plt.tight_layout()
            fig.savefig(examples_dir / f"line_{idx:02d}_{lang}.png", dpi=100)
            plt.close(fig)

            if hindi_shown >= 6 and marathi_shown >= 6:
                break
        if hindi_shown >= 6 and marathi_shown >= 6:
            break

    for i, el in enumerate(example_lines, 1):
        print(f"  #{i} {el['lang']:>7} {el['domain']:<15}  [{el['segments']}]")

    # Contact sheet 4x3
    fig, axes = plt.subplots(4, 3, figsize=(18, 6))
    for idx, ax in enumerate(axes.flat):
        if idx < len(example_lines):
            el = example_lines[idx]
            # Rebuild the line image with cuts
            # Find the original line from check_lines
            for (lang, domain), lines in check_lines.items():
                for l in lines:
                    if l["image"] == el["image"] and l["lang"] == el["lang"]:
                        segs = segment_at_threshold(l["line_32"], best_t)
                        ax.imshow(l["line_32"], cmap="gray", aspect="auto")
                        for x in segs:
                            ax.axvline(x[0], color="red", linewidth=0.8)
                        ax.set_title(f"#{idx+1} {el['lang']} {el['domain']}", fontsize=7)
                        break
                else:
                    continue
                break
        ax.axis("off")
    plt.suptitle(f"Word-gap calibration examples (t={best_t})", fontsize=10)
    plt.tight_layout()
    fig.savefig(ROOT / "results/gap_calibration_contact_sheet.png", dpi=100)
    plt.close(fig)
    print(f"\n  Saved {len(example_lines)} examples to {examples_dir}")
    print(f"  Saved contact sheet to results/gap_calibration_contact_sheet.png")

    # B7: Limitations
    limitations = (
        "PROXY METRIC: vocabulary is literary (Mozhi novels), so names, numbers, "
        "and non-literary terms count as out-of-vocabulary even when correctly segmented. "
        "Recognizer errors lower the in-vocabulary rate below the ceiling. "
        "Short valid words (1-2 Devanagari chars) flatter over-segmentation by inflating "
        "the fragment count without proportionally lowering the rate. "
        "Latin segments are excluded from the lexical check. "
        "Pages are IndicDLP train pages at relatively low scan resolution. "
        "Real accuracy requires the annotated 50-page benchmark with ground-truth word boundaries."
    )
    print(f"\nB7 — Limitations:\n  {limitations}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Save JSON
    # ═══════════════════════════════════════════════════════════════════════════
    result = {
        "git_commit":      git_commit_hash(),
        "date":            str(date.today()),
        "weights_sha256":  {"slim": weights_sha, "full": full_sha,
                           "charset": charset_sha, "layout": layout_sha},
        "seed":            42,
        "split":           "IndicDLP train (hindi+marathi), pages split 50/50 tune/check",
        "A1_dip_code_description": {
            "method": "short_peak = max(hist[:6]); dip = first bin where count < 0.3*short_peak",
            "looks_for_valley": True,
            "looks_for_second_hump": False,
        },
        "A2_mode_valley": a2_results,
        "A3_contact_sheet": {"exists": True, "size_kb": round(cs_size / 1024, 1)},
        "B1_vocabulary": {
            "hindi_size": len(hindi_vocab),
            "marathi_size": len(marathi_vocab),
            "union_size": len(union_vocab),
            "ceiling": ceiling,
        },
        "B2_sampling": {
            "pages_per_cell": MAX_PAGES,
            "tune_pages": len(tune_pages),
            "check_pages": len(check_pages),
            "tune_lines": tune_total,
            "check_lines": check_total,
        },
        "B4_threshold_sweep": {str(t): all_results[t] for t in THRESHOLDS},
        "B5_chosen_t": best_t,
        "B5_check_rate": check_rate,
        "B5_per_domain_best_t": per_domain_best,
        "B6_examples": [{"lang": el["lang"], "domain": el["domain"],
                         "segments": el["segments"], "n_segs": el["n_segs"]}
                        for el in example_lines],
        "B7_limitations": limitations,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
