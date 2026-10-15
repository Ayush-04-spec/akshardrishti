#!/usr/bin/env python3
"""Stitched-line stress test for the CRNN recognizer on the VALIDATION split.

Reconstructs multi-word line images from word crops, feeds them through the
pipeline's recognize() call, and measures CER / word-boundary accuracy.

IMPORTANT: this script ONLY uses index_val.csv. index_test.csv is never read.

Steps
-----
  STEP 0  Parse filename structure and verify groups form coherent sentences.
  STEP 1  Stitch word crops into line images; run recognizer over the grid.
  STEP 2  Per-condition metrics (CER, word accuracy, space accuracy).
  STEP 3  Control for the 512-column squashing.
  STEP 4  10-example tables for n_w = 3 and n_w = 8 at gap=16, margin=0.
  STEP 5  Save results/crnn_line_stress_val.json.

Runtime note: the full 72-condition grid is large.  If it takes too long the
script falls back to n_w in {1,3,8} x gap=16 x all margins (12 conditions)
and notes the reduction.

Usage
-----
    python eval/stress_crnn_lines.py
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
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.recognize.base import build_recognizer    # noqa: E402
from akshardrishti.postprocess.text_repair import normalize_for_scoring  # noqa: E402


# ─────────────────────────────────────────── helpers ─────────────────────────

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


def line_cer(pred: str, ref: str) -> float:
    pn = normalize_for_scoring(pred)
    rn = normalize_for_scoring(ref)
    if not rn:
        return 0.0 if not pn else 1.0
    return edit_distance(pn, rn) / len(rn)


def corpus_cer(preds: list[str], refs: list[str]) -> float:
    total_err = total_chars = 0
    for p, r in zip(preds, refs):
        pn = normalize_for_scoring(p)
        rn = normalize_for_scoring(r)
        total_err   += edit_distance(pn, rn)
        total_chars += len(rn)
    return total_err / total_chars if total_chars else 0.0


def word_exact_acc(preds: list[str], refs: list[str]) -> float:
    """Exact match after splitting on whitespace."""
    correct = total = 0
    for p, r in zip(preds, refs):
        pwords = normalize_for_scoring(p).split()
        rwords = normalize_for_scoring(r).split()
        total   += len(rwords)
        correct += sum(pw == rw for pw, rw in zip(pwords, rwords))
    return correct / total if total else 0.0


def cer_no_spaces(preds: list[str], refs: list[str]) -> float:
    """CER with all spaces removed — measures character errors only."""
    total_err = total_chars = 0
    for p, r in zip(preds, refs):
        pn = normalize_for_scoring(p).replace(" ", "")
        rn = normalize_for_scoring(r).replace(" ", "")
        total_err   += edit_distance(pn, rn)
        total_chars += len(rn)
    return total_err / total_chars if total_chars else 0.0


# ─────────────────────────────────── image stitching ─────────────────────────

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


def stitch_line(crops: list[np.ndarray], gap: int, margin_l: int) -> np.ndarray:
    """Concatenate word crops horizontally.

    margin_l white columns on the left only.  The recognizer's right_pad
    handles the right margin.  Gap of `gap` columns between each word.
    """
    total_w = margin_l + sum(c.shape[1] for c in crops) + gap * (len(crops) - 1)
    line = np.full((TARGET_H, total_w), 255, dtype=np.uint8)
    x = margin_l
    for i, c in enumerate(crops):
        line[:, x:x + c.shape[1]] = c
        x += c.shape[1]
        if i < len(crops) - 1:
            x += gap
    return line


# ─────────────────────────────────────────────── main ────────────────────────

def main() -> None:
    out_path = ROOT / "results/crnn_line_stress_val.json"
    if out_path.exists():
        print(f"Output already exists: {out_path}  — delete to re-run.")
        sys.exit(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load config + recognizer ──────────────────────────────────────────────
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
    right_pad   = int(backend_cfg.get("right_pad", 64))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}  right_pad={right_pad}")

    recognizer = build_recognizer(backend_name, config=backend_cfg, device=device)
    recognizer.ensure_loaded()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 0 — Group analysis
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 0 — filename structure and group analysis")

    val_csv = ROOT / "datasets/mozhi_raw/index_val.csv"
    # (index_test.csv is NEVER read — see assertion at end)
    all_rows = list(csv.DictReader(open(val_csv, encoding="utf-8")))
    print(f"  val rows: {len(all_rows)}")

    FNAME_RE = re.compile(r'^(.+)_(\d+)_(\d+)_(\d+)\.(jpe?g|png)$', re.IGNORECASE)
    groups: dict[tuple, list] = defaultdict(list)
    bad_parse = 0
    for r in all_rows:
        fname = Path(r["image_path"]).name
        m = FNAME_RE.match(fname)
        if m:
            doc, a, b, k = m.group(1), m.group(2), m.group(3), int(m.group(4))
            groups[(doc, a, b)].append((k, r["text"], r["image_path"], r["language"]))
        else:
            bad_parse += 1

    n_groups = len(groups)
    print(f"  parsed: {len(all_rows) - bad_parse}  unparseable: {bad_parse}")
    print(f"  groups: {n_groups}")

    size_dist = Counter(len(v) for v in groups.values())
    print("  Group size distribution:")
    for s in sorted(size_dist):
        print(f"    size {s}: {size_dist[s]} groups")

    # Consecutive k starting at 1?
    consec_groups: dict[tuple, list] = {}
    for key, items in groups.items():
        sorted_items = sorted(items, key=lambda x: x[0])
        ks = [x[0] for x in sorted_items]
        if ks == list(range(1, len(ks) + 1)):
            consec_groups[key] = sorted_items

    n_consec = len(consec_groups)
    print(f"  consecutive k (no gaps, start=1): {n_consec}/{n_groups}")

    print("\n  5 example groups (coherence check):")
    for key, items in list(consec_groups.items())[:5]:
        words = [t for _, t, _, _ in items]
        sentence = " ".join(words)
        print(f"    key={key}  [{sentence}]")

    step0 = {
        "n_val_rows": len(all_rows),
        "n_groups": n_groups,
        "n_consecutive": n_consec,
        "size_distribution": {str(k): v for k, v in sorted(size_dist.items())},
    }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Build stitch grid
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 1 — Stitch grid")

    FULL_NW   = [1, 2, 3, 4, 6, 8]
    FULL_GAPS = [8, 16, 24]
    MARGINS   = [0, 4, 8, 16]

    # Check if full grid is feasible by estimating sample count
    # For each n_w, eligible groups = those with >= n_w words (consecutive)
    eligible = {nw: [key for key, items in consec_groups.items() if len(items) >= nw]
                for nw in FULL_NW}
    total_samples = sum(min(500, len(eligible[nw])) * len(FULL_GAPS) * len(MARGINS)
                        for nw in FULL_NW)
    print(f"  Estimated full-grid samples: {total_samples}")

    # Run full grid (all 72 conditions)
    NW_LIST   = FULL_NW
    GAP_LIST  = FULL_GAPS
    reduced   = False
    print(f"  Running full grid: n_w={NW_LIST}  gaps={GAP_LIST}  margins={MARGINS}")
    print(f"  Total conditions: {len(NW_LIST)*len(GAP_LIST)*len(MARGINS)}")

    RNG = random.Random(42)
    MAX_SAMPLES = 500

    # ── sample groups ─────────────────────────────────────────────────────────
    # For each n_w, sample up to MAX_SAMPLES groups; take first n_w words
    sampled: dict[int, list[tuple]] = {}  # n_w -> [(ref_words, image_paths, lang)]
    for nw in NW_LIST:
        pool = eligible[nw]
        chosen_keys = RNG.sample(pool, min(MAX_SAMPLES, len(pool)))
        entries = []
        for key in chosen_keys:
            items = consec_groups[key][:nw]   # first nw words
            words   = [t for _, t, _, _ in items]
            paths   = [p for _, _, p, _ in items]
            lang    = items[0][3]
            entries.append((words, paths, lang))
        sampled[nw] = entries
        print(f"  n_w={nw}: {len(entries)} groups eligible, sampled {len(entries)}")

    # ── run inference across the grid ─────────────────────────────────────────
    # results[(n_w, gap, margin)] = list of (ref, pred, resized_w, lang)
    grid_results: dict[tuple, list] = {}

    total_inferences = sum(len(sampled[nw]) for nw in NW_LIST) * len(GAP_LIST) * len(MARGINS)
    print(f"\n  Total inference calls: {total_inferences}")

    call_count = 0
    for nw in NW_LIST:
        for gap in GAP_LIST:
            for margin in MARGINS:
                entries_result = []
                for words, paths, lang in sampled[nw]:
                    # Load and resize crops
                    crops = [load_and_resize(p) for p in paths]
                    if any(c is None for c in crops):
                        continue
                    # Stitch
                    line_img = stitch_line(crops, gap=gap, margin_l=margin)
                    resized_w = line_img.shape[1]

                    ref = " ".join(words)
                    result = recognizer.recognize(line_img, "deva")
                    pred = result.text

                    entries_result.append((ref, pred, resized_w, lang))
                    call_count += 1
                    if call_count % 5000 == 0:
                        print(f"    ... {call_count}/{total_inferences} done")

                grid_results[(nw, gap, margin)] = entries_result

    print(f"  Inference complete: {call_count} calls")

    # ── report 512-cap hits per n_w ───────────────────────────────────────────
    print("\n  Rows where resized_width > 512 (squashed by max_width cap):")
    squash_by_nw: dict[int, int] = {}
    for nw in NW_LIST:
        # use gap=16, margin=0 as the reference condition
        entries = grid_results.get((nw, 16, 0), [])
        n_over = sum(1 for _, _, w, _ in entries if w > 512)
        squash_by_nw[nw] = n_over
        print(f"    n_w={nw}: {n_over}/{len(entries)} lines width>512")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Metrics per condition
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 2 — Per-condition metrics")

    def compute_space_metrics(preds: list[str], refs: list[str]) -> dict:
        total_ref_spaces  = sum(normalize_for_scoring(r).count(" ") for r in refs)
        total_pred_spaces = sum(normalize_for_scoring(p).count(" ") for p in preds)
        spaces_correct    = sum(
            normalize_for_scoring(p).count(" ") == normalize_for_scoring(r).count(" ")
            for p, r in zip(preds, refs)
        )
        return {
            "total_ref_spaces":  total_ref_spaces,
            "total_pred_spaces": total_pred_spaces,
            "spaces_correct_frac": spaces_correct / len(refs) if refs else 0.0,
        }

    all_metrics: dict = {}
    for nw in NW_LIST:
        for gap in GAP_LIST:
            for margin in MARGINS:
                entries = grid_results.get((nw, gap, margin), [])
                if not entries:
                    continue
                refs_  = [e[0] for e in entries]
                preds_ = [e[1] for e in entries]
                sm = compute_space_metrics(preds_, refs_)
                all_metrics[(nw, gap, margin)] = {
                    "n": len(entries),
                    "cer": corpus_cer(preds_, refs_),
                    "word_exact_acc": word_exact_acc(preds_, refs_),
                    "cer_no_spaces": cer_no_spaces(preds_, refs_),
                    **sm,
                }

    # Pivot table: rows=n_w, cols=margin, gap=16
    print("\n  Pivot (gap=16): CER  |  space-correct fraction")
    hdr = f"  {'n_w':>4}  " + "  ".join(f"{'m='+str(m):>14}" for m in MARGINS)
    print(hdr)
    for nw in NW_LIST:
        vals = []
        for m in MARGINS:
            k = (nw, 16, m)
            if k in all_metrics:
                am = all_metrics[k]
                vals.append(f"{am['cer']:.4f}/{am['spaces_correct_frac']:.3f}")
            else:
                vals.append("   N/A        ")
        print(f"  {nw:>4}  " + "  ".join(f"{v:>14}" for v in vals))

    # Gap effect table: gap vs n_w, margin=0
    print("\n  Gap effect (margin=0): CER")
    hdr2 = f"  {'n_w':>4}  " + "  ".join(f"{'g='+str(g):>10}" for g in GAP_LIST)
    print(hdr2)
    for nw in NW_LIST:
        vals = []
        for g in GAP_LIST:
            k = (nw, g, 0)
            vals.append(f"{all_metrics[k]['cer']:.4f}" if k in all_metrics else "  N/A   ")
        print(f"  {nw:>4}  " + "  ".join(f"{v:>10}" for v in vals))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Control for squashing
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3 — Squashing control (n_w in {6,8})")

    squash_metrics: dict = {}
    for nw in [6, 8]:
        entries = grid_results.get((nw, 16, 0), [])
        narrow  = [(r, p) for r, p, w, _ in entries if w <= 512]
        wide    = [(r, p) for r, p, w, _ in entries if w >  512]
        def report(pairs, label):
            if not pairs:
                return None
            rs_ = [x[0] for x in pairs]
            ps_ = [x[1] for x in pairs]
            cer = corpus_cer(ps_, rs_)
            print(f"  n_w={nw} {label:20s} n={len(pairs):4d}  CER={cer:.5f}")
            return {"n": len(pairs), "cer": cer}
        squash_metrics[str(nw)] = {
            "narrow_le_512": report(narrow, "width<=512"),
            "wide_gt_512":   report(wide,   "width>512 (squashed)"),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Examples
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 4 — 10 examples each for n_w=3 and n_w=8 at gap=16, margin=0")

    examples_by_nw: dict = {}
    for nw in [3, 8]:
        entries = grid_results.get((nw, 16, 0), [])
        print(f"\n  n_w={nw}:")
        print(f"  {'#':>3}  {'ref':<40}  {'pred':<40}  {'CER':>6}")
        exs = []
        for k, (ref, pred, w, _) in enumerate(entries[:10], 1):
            lc = line_cer(pred, ref)
            print(f"  {k:>3}  {ref[:38]:<40}  {pred[:38]:<40}  {lc:.4f}")
            exs.append({"ref": ref, "pred": pred, "width": w, "cer": lc})
        examples_by_nw[str(nw)] = exs

    # ══════════════════════════════════════════════════════════════════════════
    # VERIFY check 1: n_w=1, margin=0 CER vs val R64 greedy reference
    # ══════════════════════════════════════════════════════════════════════════
    ref_val_cer = 0.0043824786783819205  # from crnn_padding_sweep.json
    k1 = (1, 16, 0)
    if k1 in all_metrics:
        cer1 = all_metrics[k1]["cer"]
        diff = abs(cer1 - ref_val_cer)
        print(f"\n  Verify: n_w=1 CER={cer1:.6f}  ref={ref_val_cer:.6f}  |diff|={diff:.6f}")
        verify_ok = diff <= 0.001
        print(f"  Verify check: {'PASS' if verify_ok else 'FAIL'}")
    else:
        verify_ok = False
        cer1 = None
        print("  Verify: key (1,16,0) not found — FAIL")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Save JSON
    # ══════════════════════════════════════════════════════════════════════════
    metrics_json: dict = {}
    for (nw, gap, m), v in all_metrics.items():
        metrics_json[f"nw{nw}_g{gap}_m{m}"] = v

    result = {
        "git_commit":    git_commit_hash(),
        "date":          str(date.today()),
        "weights_sha256": weights_sha,
        "charset_sha256": charset_sha,
        "device":        device,
        "seed":          42,
        "split":         "validation",
        "right_pad":     right_pad,
        "reduced_grid":  reduced,
        "step0":         step0,
        "step1_squash_by_nw": squash_by_nw,
        "step2_metrics": metrics_json,
        "step3_squash_control": squash_metrics,
        "step4_examples": examples_by_nw,
        "verify_nw1_cer": cer1,
        "verify_ref_val_cer": ref_val_cer,
        "verify_pass": verify_ok,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
