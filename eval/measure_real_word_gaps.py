#!/usr/bin/env python3
"""Measure real inter-word gap distribution on IndicDLP TRAIN-split pages.

Uses ONLY datasets/indicdlp_subset/images/train and labels/train.
Never opens anything under images/val, images/test, or index_test.csv.

Usage
-----
    python eval/measure_real_word_gaps.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.layout.lines import segment_lines, crop            # noqa: E402
from akshardrishti.preprocess.enhance import preprocess_line_crop      # noqa: E402
from akshardrishti.schema import BBox                                 # noqa: E402


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


def quantiles(data: list[int | float], ps: list[float]) -> list[float]:
    if not data:
        return [0.0] * len(ps)
    s = sorted(data)
    return [s[min(int(p * len(s)), len(s) - 1)] for p in ps]


# ─────────────────────────────────── main ────────────────────────────────────

def main() -> None:
    out_path = ROOT / "results/real_gap_measurement.json"
    if out_path.exists():
        print(f"Output already exists: {out_path}  — delete to re-run.")
        sys.exit(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── config (for SHA-256 and right_pad reference) ──────────────────────────
    with open(ROOT / "configs/pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights_sha = sha256_of(ROOT / cfg["recognize"]["backends"]["crnn_mozhi"]["weights"].lstrip("./"))
    charset_sha = sha256_of(ROOT / cfg["recognize"]["backends"]["crnn_mozhi"]["charset"].lstrip("./"))
    layout_sha   = sha256_of(ROOT / cfg["layout"]["weights"].lstrip("./"))
    full_sha     = sha256_of(ROOT / "weights/recognize/crnn_mozhi_himr_full.pt")

    # ── load split_manifest.csv (train split only, hindi+marathi only) ────────
    # ONLY file opened for data: split_manifest.csv, labels/train/*.txt,
    # images/train/*.png — never val or test
    manifest_path = ROOT / "split_manifest.csv"
    all_manifest = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    train_hm = [r for r in all_manifest
                if r["split"] == "train"
                and r["language"] in ("hindi", "marathi")]
    print(f"Train hindi+marathi manifest entries: {len(train_hm)}")

    # per (language, domain) cells
    cells: dict[tuple, list[str]] = defaultdict(list)
    for r in train_hm:
        cells[(r["language"], r["domain"])].append(r["filename"])

    print(f"Cells: {len(cells)}")
    for k, v in sorted(cells.items()):
        print(f"  {k}: {len(v)} pages")

    # ── B1: sample up to 150 pages per cell, crop text boxes ──────────────────
    RNG = random.Random(42)
    TEXT_CLASSES = {0, 1, 2}  # title, text, list
    MAX_PAGES_PER_CELL = 150
    MAX_LINES_PER_IMAGE = 40
    MIN_ORIG_H = 12
    MIN_RESIZED_W = 64

    labels_dir = ROOT / "datasets/indicdlp_subset/labels/train"
    images_dir = ROOT / "datasets/indicdlp_subset/images/train"

    print("\n" + "=" * 60)
    print("B1 — Sample pages and crop text boxes")

    # cell_results[(lang, domain)] = {pages_sampled, boxes_total, median_page_w, median_page_h, ...}
    cell_data: dict[tuple, dict] = {}
    # all_lines[(lang, domain)] = list of (line_gray_original, line_resized_32h, original_h, original_w, image_filename)
    all_lines: dict[tuple, list] = defaultdict(list)
    # saved_examples: first 20 lines (10 hindi, 10 marathi) for B6
    saved_examples: list[dict] = []
    hindi_saved = 0
    marathi_saved = 0

    for (lang, domain) in sorted(cells.keys()):
        pool = sorted(cells[(lang, domain)])
        chosen = RNG.sample(pool, min(MAX_PAGES_PER_CELL, len(pool)))
        boxes_total = 0
        page_widths = []
        page_heights = []
        lines_this_cell = 0

        for fname in chosen:
            stem = Path(fname).stem
            label_file = labels_dir / f"{stem}.txt"
            image_file = images_dir / fname

            if not label_file.exists() or not image_file.exists():
                continue

            img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            H, W = img.shape[:2]
            page_widths.append(W)
            page_heights.append(H)

            # Parse YOLO labels: class cx cy w h (normalized)
            text_boxes = []
            for line in open(label_file, encoding="utf-8"):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls not in TEXT_CLASSES:
                    continue
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)
                # Clip to page
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W, x2); y2 = min(H, y2)
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                text_boxes.append(BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))

            boxes_total += len(text_boxes)

            # B2: segment each crop into lines
            for box in text_boxes:
                region_img = crop(img, box)
                if region_img.size == 0:
                    continue

                # Same call as the pipeline:
                # segment_lines(region_img, method="projection",
                #               min_line_height=12, smoothing=5)
                line_boxes = segment_lines(
                    region_img,
                    method="projection",
                    min_line_height=12,
                    smoothing=5,
                )

                for lb in line_boxes[:MAX_LINES_PER_IMAGE]:
                    # Filter: original height >= 12 px
                    if lb.height < MIN_ORIG_H:
                        continue
                    line_crop_orig = crop(region_img, lb)
                    if line_crop_orig.size == 0:
                        continue
                    oh, ow = line_crop_orig.shape[:2]
                    # Resized width at height 32
                    scale = 32 / max(oh, 1)
                    resized_w = max(8, int(round(ow * scale)))
                    if resized_w < MIN_RESIZED_W:
                        continue

                    # B3: preprocess_line_crop exactly as recognizer
                    line_32 = preprocess_line_crop(line_crop_orig, target_height=32, binarize_crop=True)

                    all_lines[(lang, domain)].append({
                        "orig_h": oh,
                        "orig_w": ow,
                        "resized_w": resized_w,
                        "line_32": line_32,
                        "line_orig": line_crop_orig,
                        "image": fname,
                    })
                    lines_this_cell += 1

                    # B6: save first 10 hindi, 10 marathi
                    if lang == "hindi" and hindi_saved < 10:
                        saved_examples.append({"lang": lang, "domain": domain,
                                              "line_orig": line_crop_orig,
                                              "line_32": line_32,
                                              "image": fname})
                        hindi_saved += 1
                    elif lang == "marathi" and marathi_saved < 10:
                        saved_examples.append({"lang": lang, "domain": domain,
                                              "line_orig": line_crop_orig,
                                              "line_32": line_32,
                                              "image": fname})
                        marathi_saved += 1

        med_w = int(np.median(page_widths)) if page_widths else 0
        med_h = int(np.median(page_heights)) if page_heights else 0
        cell_data[(lang, domain)] = {
            "pages_sampled": len(chosen),
            "boxes_total": boxes_total,
            "median_page_w": med_w,
            "median_page_h": med_h,
        }
        n_lines = len(all_lines[(lang, domain)])
        print(f"  ({lang:>7}, {domain:<15})  pages={len(chosen):3d}  boxes={boxes_total:5d}"
              f"  page_med={med_w}x{med_h}  lines_kept={n_lines}")

    # ── B2 summary: lines per cell, median original line height ──────────────
    print("\n" + "=" * 60)
    print("B2 — Line segmentation results")

    for (lang, domain) in sorted(all_lines.keys()):
        lines = all_lines[(lang, domain)]
        if not lines:
            continue
        heights = [l["orig_h"] for l in lines]
        med_h = float(np.median(heights))
        cell_data[(lang, domain)]["lines_kept"] = len(lines)
        cell_data[(lang, domain)]["median_line_height"] = med_h
        print(f"  ({lang:>7}, {domain:<15})  lines={len(lines):4d}  median_h={med_h:.1f}px")

    # ── B3+B4: collect internal empty-column runs ────────────────────────────
    print("\n" + "=" * 60)
    print("B3+B4 — Internal gap run lengths")

    # Per language and per domain aggregation
    # Also per (language, domain) for newspaper separately
    all_runs: dict[tuple, list[int]] = defaultdict(list)  # (lang, domain) -> runs
    all_run_widths: dict[tuple, list[int]] = defaultdict(list)  # for per-100px

    for (lang, domain), lines in all_lines.items():
        for l in lines:
            line_32 = l["line_32"]
            # Ink mask: ink pixels are black (0) after Otsu binarization (THRESH_BINARY)
            # preprocess_line_crop returns binary image: 0=ink, 255=background
            ink_mask = (line_32 < 128).astype(np.uint8)
            ink_proj = ink_mask.sum(axis=0)  # ink per column

            # Trim leading and trailing empty columns
            nonempty = np.where(ink_proj > 0)[0]
            if len(nonempty) < 2:
                continue
            left = nonempty[0]
            right = nonempty[-1]
            trimmed_proj = ink_proj[left:right + 1]

            # Find internal runs of empty columns (ink_proj == 0)
            empty = (trimmed_proj == 0)
            i = 0
            W_trim = len(trimmed_proj)
            while i < W_trim:
                if empty[i]:
                    run_start = i
                    while i < W_trim and empty[i]:
                        i += 1
                    run_len = i - run_start
                    all_runs[(lang, domain)].append(run_len)
                else:
                    i += 1

    # Aggregate per language and per domain
    per_lang_runs: dict[str, list[int]] = defaultdict(list)
    per_domain_runs: dict[str, list[int]] = defaultdict(list)
    per_lang_domain_runs: dict[tuple, list[int]] = defaultdict(list)

    for (lang, domain), runs in all_runs.items():
        per_lang_runs[lang].extend(runs)
        per_domain_runs[domain].extend(runs)
        per_lang_domain_runs[(lang, domain)].extend(runs)

    def print_run_stats(label: str, runs: list[int]):
        if not runs:
            print(f"  {label}: no runs")
            return
        qs = quantiles(runs, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        # Histogram in 2-px bins up to 40 px
        hist_bins = list(range(0, 42, 2))
        hist, _ = np.histogram(runs, bins=hist_bins)
        print(f"  {label}: lines_runs={len(runs):6d}  "
              f"p25={qs[0]:.0f} p50={qs[1]:.0f} p75={qs[2]:.0f} "
              f"p90={qs[3]:.0f} p95={qs[4]:.0f} p99={qs[5]:.0f}")
        print(f"    histogram (2-px bins):")
        for j in range(len(hist)):
            lo = hist_bins[j]; hi = hist_bins[j + 1]
            frac = hist[j] / len(runs) if runs else 0
            bar = "#" * int(frac * 50)
            print(f"      {lo:2d}-{hi:2d}px: {hist[j]:6d} ({frac:.4f}) {bar}")
        # Identify dip
        # Find the first bin where count drops below 50% of the peak in the short-run region (0-10px)
        # then rises again
        short_peak = max(hist[:6]) if len(hist) >= 6 else max(hist)
        dip_idx = -1
        for j in range(2, min(10, len(hist))):
            if hist[j] < short_peak * 0.3 and hist[j] > 0:
                dip_idx = j
                break
        if dip_idx >= 0:
            dip_px = hist_bins[dip_idx]
            print(f"    DIP at ~{dip_px}-{dip_px+2}px (short runs -> long runs boundary)")
        else:
            print(f"    No clear dip found in 0-20px range")

    print("\nPer language:")
    for lang in sorted(per_lang_runs.keys()):
        print_run_stats(f"lang={lang}", per_lang_runs[lang])

    print("\nPer domain:")
    for domain in sorted(per_domain_runs.keys()):
        print_run_stats(f"domain={domain}", per_domain_runs[domain])

    print("\nNewspaper separately:")
    for (lang, domain), runs in sorted(per_lang_domain_runs.items()):
        if domain == "newspaper":
            print_run_stats(f"({lang}, newspaper)", runs)

    print("\nPer (language, domain):")
    for (lang, domain) in sorted(per_lang_domain_runs.keys()):
        print_run_stats(f"({lang}, {domain})", per_lang_domain_runs[(lang, domain)])

    # ── B5: mean runs >= min_gap per line and per 100px ──────────────────────
    print("\n" + "=" * 60)
    print("B5 — Mean runs >= min_gap per line and per 100px")

    MIN_GAPS = [4, 6, 8, 10, 12, 16]

    # Compute per (lang, domain) and overall
    b5_results: dict = {}

    # Overall
    all_runs_flat = [r for runs in all_runs.values() for r in runs]
    all_widths_flat = []
    for (lang, domain), lines in all_lines.items():
        for l in lines:
            all_widths_flat.append(l["resized_w"])
    n_lines_total = len(all_widths_flat)
    total_width_px = sum(all_widths_flat)

    print(f"  Overall (n_lines={n_lines_total}, total_width={total_width_px}px)")
    for mg in MIN_GAPS:
        n_ge = sum(1 for r in all_runs_flat if r >= mg)
        per_line = n_ge / n_lines_total if n_lines_total else 0
        per_100px = n_ge / (total_width_px / 100) if total_width_px else 0
        print(f"    min_gap={mg:2d}: runs_ge={n_ge:6d}  per_line={per_line:.3f}  per_100px={per_100px:.4f}")
        b5_results[f"overall_mg{mg}"] = {"runs_ge": n_ge, "per_line": per_line, "per_100px": per_100px}

    # Per domain
    for domain in sorted(per_domain_runs.keys()):
        runs_d = per_domain_runs[domain]
        # Need widths per domain
        widths_d = []
        for (lang, dom), lines in all_lines.items():
            if dom == domain:
                for l in lines:
                    widths_d.append(l["resized_w"])
        n_d = len(widths_d)
        tw_d = sum(widths_d)
        print(f"\n  domain={domain} (n_lines={n_d}, total_width={tw_d}px)")
        for mg in MIN_GAPS:
            n_ge = sum(1 for r in runs_d if r >= mg)
            per_line = n_ge / n_d if n_d else 0
            per_100px = n_ge / (tw_d / 100) if tw_d else 0
            print(f"    min_gap={mg:2d}: runs_ge={n_ge:6d}  per_line={per_line:.3f}  per_100px={per_100px:.4f}")
            b5_results[f"{domain}_mg{mg}"] = {"runs_ge": n_ge, "per_line": per_line, "per_100px": per_100px}

    # ── B6: Save 20 example crops with projections ──────────────────────────
    print("\n" + "=" * 60)
    print("B6 — Save 20 example crops with projections")

    examples_dir = ROOT / "results/real_gap_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i, ex in enumerate(saved_examples):
        line_orig = ex["line_orig"]
        line_32 = ex["line_32"]
        ink_mask = (line_32 < 128).astype(np.uint8)
        ink_proj = ink_mask.sum(axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3),
                                       gridspec_kw={"height_ratios": [2, 1]})
        ax1.imshow(line_32, cmap="gray", aspect="auto")
        ax1.set_title(f"#{i+1} {ex['lang']} {ex['domain']} ({ex['image']})", fontsize=8)
        ax1.axis("off")
        ax2.bar(range(len(ink_proj)), ink_proj, width=1, color="black")
        ax2.set_xlabel("column (px at h=32)")
        ax2.set_ylabel("ink count")
        ax2.set_xlim(0, len(ink_proj))
        plt.tight_layout()
        fig.savefig(examples_dir / f"example_{i+1:02d}_{ex['lang']}.png", dpi=100)
        plt.close(fig)

    # Contact sheet 4x5
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    for idx, ax in enumerate(axes.flat):
        if idx < len(saved_examples):
            ex = saved_examples[idx]
            ax.imshow(ex["line_32"], cmap="gray", aspect="auto")
            ax.set_title(f"#{idx+1} {ex['lang']} {ex['domain']}", fontsize=7)
        ax.axis("off")
    plt.suptitle("Real word-gap examples (h=32, binarized)", fontsize=12)
    plt.tight_layout()
    fig.savefig(ROOT / "results/real_gap_contact_sheet.png", dpi=100)
    plt.close(fig)
    print(f"  Saved 20 individual examples to {examples_dir}")
    print(f"  Saved contact sheet to results/real_gap_contact_sheet.png")

    # ── Save JSON ────────────────────────────────────────────────────────────
    # Serialize run histograms and stats
    def serialize_runs(label: str, runs: list[int]) -> dict:
        if not runs:
            return {"n": 0}
        qs = quantiles(runs, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        hist_bins = list(range(0, 42, 2))
        hist, _ = np.histogram(runs, bins=hist_bins)
        return {
            "n": len(runs),
            "quantiles": {"p25": qs[0], "p50": qs[1], "p75": qs[2],
                          "p90": qs[3], "p95": qs[4], "p99": qs[5]},
            "histogram": {f"{hist_bins[j]}-{hist_bins[j+1]}px": int(hist[j])
                          for j in range(len(hist))},
        }

    result = {
        "git_commit":    git_commit_hash(),
        "date":          str(date.today()),
        "weights_sha256": {"slim": weights_sha, "full": full_sha,
                           "charset": charset_sha, "layout": layout_sha},
        "seed":          42,
        "split":         "train (hindi + marathi only)",
        "B1_cells": {f"{lang}_{domain}": v for (lang, domain), v in cell_data.items()},
        "B2_lines_per_cell": {f"{lang}_{domain}": cell_data.get((lang, domain), {}).get("lines_kept", 0)
                              for (lang, domain) in sorted(cell_data.keys())},
        "B4_per_language": {lang: serialize_runs(lang, per_lang_runs[lang])
                            for lang in sorted(per_lang_runs.keys())},
        "B4_per_domain": {domain: serialize_runs(domain, per_domain_runs[domain])
                           for domain in sorted(per_domain_runs.keys())},
        "B4_per_lang_domain": {f"{lang}_{domain}": serialize_runs(
                                    f"{lang}_{domain}", per_lang_domain_runs[(lang, domain)])
                                for (lang, domain) in sorted(per_lang_domain_runs.keys())},
        "B5_min_gap_sweep": b5_results,
        "B6_examples_saved": len(saved_examples),
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
