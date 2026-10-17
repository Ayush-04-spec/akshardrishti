#!/usr/bin/env python3
"""Evaluate Tesseract as a baseline on the same Mozhi test rows as AksharDrishti.

Uses the same 20,026 kept test rows (coverage filter >= 0.999).
Compares CER and word accuracy for Tesseract vs our CRNN.

Output: results/tesseract_baseline.json and updates results/graphs/fig_crnn_cer_wer_comparison.png

Usage
-----
    python eval/eval_tesseract_baseline.py
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path

import cv2
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from akshardrishti.recognize.crnn_model import Charset
from akshardrishti.postprocess.text_repair import normalize_for_scoring

# ─────────────────────────────── helpers ─────────────────────────────────────

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
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
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


def word_accuracy(preds: list[str], refs: list[str]) -> float:
    exact = sum(normalize_for_scoring(p) == normalize_for_scoring(r)
                for p, r in zip(preds, refs))
    return exact / len(refs) if refs else 0.0


def run_tesseract(img: cv2.Mat, lang: str) -> str:
    """Run tesseract on a single word crop."""
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
        config = f"--oem 1 --psm 7 -l {lang}"
        text = pytesseract.image_to_string(rgb, config=config).strip()
        return text
    except Exception:
        return ""


def main() -> None:
    out_path = ROOT / "results/tesseract_baseline.json"

    # Load charset for coverage filter
    charset = Charset.load(ROOT / "weights/recognize/charset_deva.txt")

    # Load test rows
    test_csv = ROOT / "datasets/mozhi_raw/index_test.csv"
    all_rows  = list(csv.DictReader(open(test_csv, encoding="utf-8")))
    kept_rows = [r for r in all_rows if charset.coverage(r["text"]) >= 0.999]
    print(f"Kept test rows: {len(kept_rows)}")

    refs  = [unicodedata.normalize("NFC", r["text"]) for r in kept_rows]
    langs = [r["language"] for r in kept_rows]

    # Run Tesseract — use hin+mar for both languages
    print("Running Tesseract (this takes ~10-15 minutes for 20k images)...")
    preds: list[str] = []
    n = len(kept_rows)

    for i, r in enumerate(kept_rows):
        if i % 2000 == 0:
            print(f"  {i}/{n}...")
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            preds.append("")
            continue
        # Use hin+mar combined for both languages
        text = run_tesseract(img, "hin+mar")
        preds.append(text)

    # Compute metrics
    overall_cer  = corpus_cer(preds, refs)
    overall_wacc = word_accuracy(preds, refs)
    wrong        = sum(normalize_for_scoring(p) != normalize_for_scoring(r)
                       for p, r in zip(preds, refs))

    print(f"\n{'='*50}")
    print(f"TESSERACT BASELINE RESULTS")
    print(f"{'='*50}")
    print(f"  n               : {len(kept_rows)}")
    print(f"  CER             : {overall_cer:.6f}")
    print(f"  Word accuracy   : {overall_wacc:.6f}")
    print(f"  Wrong words     : {wrong}")

    # Per language
    from collections import defaultdict
    by_lang: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p, r, l in zip(preds, refs, langs):
        by_lang[l][0].append(p)
        by_lang[l][1].append(r)

    print(f"\n  Per language:")
    lang_results = {}
    for lang in sorted(by_lang.keys()):
        ps, rs = by_lang[lang]
        lcer  = corpus_cer(ps, rs)
        lwacc = word_accuracy(ps, rs)
        print(f"    {lang:>10}  n={len(rs):6d}  CER={lcer:.6f}  word_acc={lwacc:.6f}")
        lang_results[lang] = {"n": len(rs), "cer": lcer, "word_accuracy": lwacc}

    # Save results
    result = {
        "n": len(kept_rows),
        "cer": overall_cer,
        "word_accuracy": overall_wacc,
        "wrong_words": wrong,
        "per_language": lang_results,
        "crnn_reference": {
            "cer": 0.004859024359056326,
            "word_accuracy": 0.9813742135224208,
            "n": 20026
        }
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")

    # Regenerate comparison graph with real Tesseract number
    print("\nRegenerating comparison graph with real Tesseract CER...")
    try:
        regenerate_graph(overall_cer, 1.0 - overall_wacc)
    except Exception as e:
        print(f"Graph generation failed: {e}")


def regenerate_graph(tess_cer: float, tess_wer: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    BLUE   = "#1f4e8c"
    ORANGE = "#c0392b"
    OUT_DIR = ROOT / "results" / "graphs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })

    # Use real Tesseract + estimated others + our measured value
    systems   = ["Tesseract\n(measured)", "PaddleOCR\n(est.)", "Bhashini\n(est.)",
                 "Surya\n(est.)", "AksharDrishti\n(ours)"]
    cer_vals  = [tess_cer, 0.12, 0.08, 0.10, 0.004859]
    wer_vals  = [tess_wer, 0.19, 0.12, 0.15, 0.0186]

    x     = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width/2, cer_vals, width, label="CER",
                   color=BLUE,   alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, wer_vals, width, label="WER",
                   color=ORANGE, alpha=0.85, edgecolor="white")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel("Error Rate (lower is better)")
    ax.set_title("OCR Baseline Comparison — CER and WER\n"
                 "(Mozhi test set, 20,026 Devanagari word crops, Hindi + Marathi)", fontsize=10)
    ax.set_ylim(0, max(max(cer_vals), max(wer_vals)) * 1.3)
    ax.legend()
    ax.get_xticklabels()[-1].set_color(BLUE)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.text(0.01, 0.97,
            "Measured: Tesseract 5.5.3 (hin+mar), AksharDrishti CRNN.\n"
            "est. = approximate range from published benchmarks on similar data.",
            transform=ax.transAxes, fontsize=7, va="top", color="gray")

    fig.tight_layout()
    out = OUT_DIR / "fig_crnn_cer_wer_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Graph updated: {out}")


if __name__ == "__main__":
    main()
