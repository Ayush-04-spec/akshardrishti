#!/usr/bin/env python3
"""Diagnose the CER gap between the training-path and the inference-path on
the Mozhi test split.

Steps executed
--------------
  STEP 2  Width and label-length distribution of the 20,026 kept test rows.
  STEP 3  Pixel-level identity check between training-path and inference-path
          tensors for every kept row.
  STEP 4  Four greedy runs: A1 (batch-64, CSV order), A2 (batch-1),
          A3 (batch-64, width-sorted), B (inference per-sample greedy).
  STEP 5  Contingency table A1 vs B; per-length-bucket and per-language
          breakdown; 15 examples where A1 is right and B is wrong.
  STEP 6  Write results/crnn_gap_diagnosis.json.

Usage
-----
    python eval/diagnose_crnn_gap.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

from akshardrishti.recognize.base import build_recognizer                        # noqa: E402
from akshardrishti.recognize.crnn_model import Charset, greedy_decode           # noqa: E402
from akshardrishti.postprocess.text_repair import normalize_for_scoring          # noqa: E402
from train.dataset import LineImageDataset, ctc_collate, load_index              # noqa: E402


# ──────────────────────────────────────────────────── tiny helpers ────────────

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
    """Total edit-distance / total reference characters after normalize_for_scoring.
    Matches the local formula used in train_crnn.py exactly."""
    total_err = sum(
        edit_distance(normalize_for_scoring(p), normalize_for_scoring(r))
        for p, r in zip(preds, refs)
    )
    total_len = sum(len(normalize_for_scoring(r)) for r in refs)
    return total_err / total_len if total_len else 0.0


def word_acc(preds: list[str], refs: list[str]) -> float:
    exact = sum(
        normalize_for_scoring(p) == normalize_for_scoring(r)
        for p, r in zip(preds, refs)
    )
    return exact / len(refs) if refs else 0.0


def wrong_count(preds: list[str], refs: list[str]) -> int:
    return sum(
        normalize_for_scoring(p) != normalize_for_scoring(r)
        for p, r in zip(preds, refs)
    )


def run_summary(preds: list[str], refs: list[str], label: str) -> dict:
    n = len(refs)
    cer = corpus_cer(preds, refs)
    wa  = word_acc(preds, refs)
    wc  = wrong_count(preds, refs)
    print(f"  {label:<30}  n={n}  CER={cer:.6f}  word_acc={wa:.6f}  wrong={wc}")
    return {"n": n, "cer": cer, "word_accuracy": wa, "wrong_words": wc}


# ───────────────────────────────────────────────────── main ───────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",    type=Path, default=ROOT / "datasets/mozhi_raw")
    p.add_argument("--config",  type=Path, default=ROOT / "configs/pipeline.yaml")
    p.add_argument("--batch",   type=int,  default=64)
    p.add_argument("--workers", type=int,  default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    p.add_argument("--out",     type=Path,
                   default=ROOT / "results/crnn_gap_diagnosis.json")
    args = p.parse_args()

    # ── guard ─────────────────────────────────────────────────────────────────
    if args.out.exists():
        print(f"Output already exists: {args.out}  — delete it manually to re-run.")
        sys.exit(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── load config & build recognizer ───────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backend_name = cfg["recognize"]["deva_backend"]
    backend_cfg  = dict(cfg["recognize"]["backends"][backend_name])
    for key in ("weights", "charset"):
        bp = Path(backend_cfg[key])
        if not bp.is_absolute():
            backend_cfg[key] = str(ROOT / bp)

    weights_path = Path(backend_cfg["weights"])
    charset_path = Path(backend_cfg["charset"])
    img_h = int(backend_cfg.get("img_height", 32))
    max_w = int(backend_cfg.get("max_width",  512))

    weights_sha = sha256_of(weights_path)
    charset_sha = sha256_of(charset_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")
    print(f"Weights : {weights_path}  SHA={weights_sha[:16]}...")
    print(f"Charset : {charset_path}  SHA={charset_sha[:16]}...")
    print()

    recognizer = build_recognizer(backend_name, config=backend_cfg, device=device)
    recognizer.ensure_loaded()
    charset: Charset = recognizer.charset
    model          = recognizer.model

    # ── load & filter test rows (same as eval_crnn_test.py) ──────────────────
    all_rows  = load_index(args.data / "index_test.csv")
    kept_rows = [r for r in all_rows if charset.coverage(r["text"]) >= 0.999]
    dropped   = len(all_rows) - len(kept_rows)
    refs      = [unicodedata.normalize("NFC", r["text"]) for r in kept_rows]
    langs     = [r["language"]  for r in kept_rows]
    n         = len(kept_rows)
    print(f"Total test rows : {len(all_rows)}  dropped : {dropped}  kept : {n}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Width & label-length stats
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 2 — Width and label-length statistics")

    resized_widths: list[int] = []
    label_lengths:  list[int] = []

    for r in kept_rows:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            resized_widths.append(8)
        else:
            h, w = img.shape[:2]
            scale = img_h / max(h, 1)
            resized_widths.append(max(8, int(round(w * scale))))
        label_lengths.append(len(r["text"]))

    def quantiles(data: list[int], qs: list[float]) -> list[int]:
        s = sorted(data)
        out = []
        for q in qs:
            idx = min(int(q * len(s)), len(s) - 1)
            out.append(s[idx])
        return out

    rw_arr = sorted(resized_widths)
    over512 = sum(1 for w in resized_widths if w > 512)
    rw_q = quantiles(resized_widths, [0.0, 0.5, 0.90, 0.99, 1.0])
    ll_q = quantiles(label_lengths,  [0.0, 0.5, 0.90, 0.99, 1.0])

    print(f"  Resized width  min={rw_q[0]}  median={rw_q[1]}  p90={rw_q[2]}  p99={rw_q[3]}  max={rw_q[4]}  >512: {over512}")
    print(f"  Label length   min={ll_q[0]}  median={ll_q[1]}  p90={ll_q[2]}  p99={ll_q[3]}  max={ll_q[4]}")
    print()

    step2 = {
        "resized_width": {"min": rw_q[0], "median": rw_q[1], "p90": rw_q[2],
                          "p99": rw_q[3], "max": rw_q[4], "over512": over512},
        "label_length":  {"min": ll_q[0], "median": ll_q[1], "p90": ll_q[2],
                          "p99": ll_q[3], "max": ll_q[4]},
    }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Pixel identity check
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 3 — Pixel identity between training-path and inference-path tensors")

    # Training-path tensors (via LineImageDataset, augment=False)
    test_ds = LineImageDataset(kept_rows, charset, img_height=img_h,
                               max_width=max_w, augment=False, drop_uncovered=False)

    # Inference-path: cv2.imread -> _prepare, squeeze batch dim -> (1,H,W)
    n_equal   = 0
    n_differ  = 0
    max_diff  = 0.0
    differ_examples: list[dict] = []

    for idx in range(n):
        # training tensor: (1, H, W) from __getitem__ (no batch padding)
        train_tensor, _, _, _ = test_ds[idx]       # (1, H, W)

        # inference tensor
        img = cv2.imread(kept_rows[idx]["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            n_differ += 1
            continue
        inf_tensor = recognizer._prepare(img).squeeze(0)   # (1, 1, H, W) -> (1, H, W)

        # Shapes may differ if inference-path produces a different width
        if train_tensor.shape != inf_tensor.shape:
            diff = float("inf")
        else:
            diff = float((train_tensor - inf_tensor).abs().max())

        if diff == 0.0:
            n_equal += 1
        else:
            n_differ += 1
            if diff != float("inf"):
                max_diff = max(max_diff, diff)
            if len(differ_examples) < 5:
                differ_examples.append({
                    "index": idx,
                    "gt": kept_rows[idx]["text"],
                    "lang": kept_rows[idx]["language"],
                    "resized_w": resized_widths[idx],
                    "train_shape": list(train_tensor.shape),
                    "inf_shape": list(inf_tensor.shape),
                    "max_abs_diff": round(diff, 6) if diff != float("inf") else "shape_mismatch",
                })

    # Analyse: are all differing rows those with resized_width > 512?
    # In both paths the final clipping width is min(resized_w, max_w).
    # The only code-path difference for wide crops is:
    #   training: single resize to min(rw, max_w) using INTER_CUBIC/INTER_AREA
    #   inference: resize to rw (full aspect), THEN a second INTER_AREA resize to max_w
    # So pixel differences arise exactly when rw > max_w.
    shape_differ_wide   = sum(1 for rw in resized_widths if rw > max_w)

    print(f"  Equal pairs     : {n_equal}")
    print(f"  Differing pairs : {n_differ}  (max abs diff: {max_diff:.6f})")
    print(f"  Rows with rw>512: {over512}")
    print(f"  Rows where rw>512 (same count, code-path explains all differences): {shape_differ_wide}")
    if n_differ > 0:
        only_wide = (n_differ == shape_differ_wide)
        print(f"  All differences explained by width>512: {only_wide}")
    print("  5 differing examples:")
    for ex in differ_examples:
        print(f"    idx={ex['index']}  rw={ex['resized_w']}  train={ex['train_shape']}  "
              f"inf={ex['inf_shape']}  max_diff={ex['max_abs_diff']}")
    print()

    step3 = {
        "n_equal": n_equal,
        "n_differ": n_differ,
        "max_abs_diff": max_diff,
        "rows_wider_than_512": over512,
        "differ_examples": differ_examples,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Four greedy runs
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 4 — Four greedy runs")

    model.eval()

    # ── A1: batch-64, CSV order ───────────────────────────────────────────────
    test_loader_A1 = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=ctc_collate
    )
    preds_A1: list[str] = []
    with torch.no_grad():
        for images, _, _, _, texts in test_loader_A1:
            lp = model(images.to(device))
            preds_A1.extend(t for t, _ in greedy_decode(lp, charset))

    s_A1 = run_summary(preds_A1, refs, "A1 batch-64 CSV-order")

    # Verify A1 against reference before continuing
    ref_cer = 0.004859024359056326
    ref_n   = 20026
    a1_cer  = s_A1["cer"]
    a1_n    = s_A1["n"]
    if a1_n != ref_n:
        print(f"  STOP: A1 n={a1_n} != reference n={ref_n}")
        sys.exit(1)
    if abs(a1_cer - ref_cer) > 0.0002:
        print(f"  STOP: A1 CER={a1_cer:.8f}, reference={ref_cer:.8f}, |delta|={abs(a1_cer-ref_cer):.8f} > 0.0002")
        sys.exit(1)
    print(f"  A1 reference check: n={a1_n}=={ref_n} ✓  |dCER|={abs(a1_cer-ref_cer):.2e} <=0.0002 ✓")

    # ── A2: batch-1 (no padding at all) ──────────────────────────────────────
    test_loader_A2 = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=ctc_collate
    )
    preds_A2: list[str] = []
    with torch.no_grad():
        for images, _, _, _, texts in test_loader_A2:
            lp = model(images.to(device))
            preds_A2.extend(t for t, _ in greedy_decode(lp, charset))

    s_A2 = run_summary(preds_A2, refs, "A2 batch-1 (no padding)")

    # ── A3: batch-64, width-sorted ────────────────────────────────────────────
    # Sort indices by resized width so each batch has minimal right-padding.
    sorted_idxs = sorted(range(n), key=lambda i: resized_widths[i])
    sorted_ds   = torch.utils.data.Subset(test_ds, sorted_idxs)
    test_loader_A3 = torch.utils.data.DataLoader(
        sorted_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=ctc_collate
    )
    preds_A3_sorted: list[str] = []
    with torch.no_grad():
        for images, _, _, _, texts in test_loader_A3:
            lp = model(images.to(device))
            preds_A3_sorted.extend(t for t, _ in greedy_decode(lp, charset))

    # Map back to original CSV order
    preds_A3 = [""] * n
    for sorted_pos, orig_idx in enumerate(sorted_idxs):
        preds_A3[orig_idx] = preds_A3_sorted[sorted_pos]

    s_A3 = run_summary(preds_A3, refs, "A3 batch-64 width-sorted")

    # ── B: inference-greedy (per-sample, cv2 -> _prepare -> greedy) ──────────
    preds_B: list[str] = []
    with torch.no_grad():
        for r in kept_rows:
            img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                preds_B.append("")
                continue
            tensor = recognizer._prepare(img).to(device)
            lp     = model(tensor)
            text, _ = greedy_decode(lp, charset)[0]
            preds_B.append(text)

    s_B = run_summary(preds_B, refs, "B  inference-greedy")

    print()
    step4 = {"A1": s_A1, "A2": s_A2, "A3": s_A3, "B": s_B}

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Where do the extra errors come from?
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 5 — Error analysis")

    # -- 5a: Contingency table A1 vs B ----------------------------------------
    a1_right = [normalize_for_scoring(p) == normalize_for_scoring(r)
                for p, r in zip(preds_A1, refs)]
    b_right  = [normalize_for_scoring(p) == normalize_for_scoring(r)
                for p, r in zip(preds_B,  refs)]

    both_right    = sum(1 for a, b in zip(a1_right, b_right) if a and b)
    a1r_bw        = sum(1 for a, b in zip(a1_right, b_right) if a and not b)
    a1w_br        = sum(1 for a, b in zip(a1_right, b_right) if not a and b)
    both_wrong    = sum(1 for a, b in zip(a1_right, b_right) if not a and not b)

    print(f"\n  Contingency table A1 vs B (word level):")
    print(f"    both right       : {both_right:6d}")
    print(f"    A1 right B wrong : {a1r_bw:6d}")
    print(f"    A1 wrong B right : {a1w_br:6d}")
    print(f"    both wrong       : {both_wrong:6d}")
    print(f"    total            : {both_right+a1r_bw+a1w_br+both_wrong:6d}")

    contingency = {
        "both_right": both_right,
        "A1_right_B_wrong": a1r_bw,
        "A1_wrong_B_right": a1w_br,
        "both_wrong": both_wrong,
    }

    # -- 5b: Per-label-length-bucket breakdown ---------------------------------
    buckets = {"1": (1, 1), "2": (2, 2), "3": (3, 3), "4-5": (4, 5),
               "6-8": (6, 8), "9+": (9, 9999)}

    def bucket_of(length: int) -> str:
        for name, (lo, hi) in buckets.items():
            if lo <= length <= hi:
                return name
        return "9+"

    per_bucket: dict[str, dict] = {}
    print(f"\n  Per-label-length-bucket (wrong words):")
    print(f"  {'bucket':>6}  {'n':>7}  {'A1_wr':>6}  {'A2_wr':>6}  {'B_wr':>6}  {'A1_CER':>8}  {'A2_CER':>8}  {'B_CER':>8}")
    for bname in ["1", "2", "3", "4-5", "6-8", "9+"]:
        idxs = [i for i, r in enumerate(kept_rows) if bucket_of(len(r["text"])) == bname]
        if not idxs:
            continue
        b_refs  = [refs[i]     for i in idxs]
        b_A1    = [preds_A1[i] for i in idxs]
        b_A2    = [preds_A2[i] for i in idxs]
        b_B     = [preds_B[i]  for i in idxs]
        wr_A1 = wrong_count(b_A1, b_refs)
        wr_A2 = wrong_count(b_A2, b_refs)
        wr_B  = wrong_count(b_B,  b_refs)
        cer_A1 = corpus_cer(b_A1, b_refs)
        cer_A2 = corpus_cer(b_A2, b_refs)
        cer_B  = corpus_cer(b_B,  b_refs)
        print(f"  {bname:>6}  {len(idxs):>7}  {wr_A1:>6}  {wr_A2:>6}  {wr_B:>6}  {cer_A1:>8.4f}  {cer_A2:>8.4f}  {cer_B:>8.4f}")
        per_bucket[bname] = {
            "n": len(idxs),
            "A1": {"wrong": wr_A1, "cer": cer_A1},
            "A2": {"wrong": wr_A2, "cer": cer_A2},
            "B":  {"wrong": wr_B,  "cer": cer_B},
        }

    # -- 5c: Per-language breakdown --------------------------------------------
    per_lang: dict[str, dict] = {}
    print(f"\n  Per-language breakdown:")
    print(f"  {'lang':>10}  {'n':>7}  {'A1_wr':>6}  {'A2_wr':>6}  {'B_wr':>6}  {'A1_CER':>8}  {'A2_CER':>8}  {'B_CER':>8}")
    for lang in sorted(set(langs)):
        idxs = [i for i, l in enumerate(langs) if l == lang]
        b_refs = [refs[i]     for i in idxs]
        b_A1   = [preds_A1[i] for i in idxs]
        b_A2   = [preds_A2[i] for i in idxs]
        b_B    = [preds_B[i]  for i in idxs]
        wr_A1  = wrong_count(b_A1, b_refs)
        wr_A2  = wrong_count(b_A2, b_refs)
        wr_B   = wrong_count(b_B,  b_refs)
        cer_A1 = corpus_cer(b_A1, b_refs)
        cer_A2 = corpus_cer(b_A2, b_refs)
        cer_B  = corpus_cer(b_B,  b_refs)
        print(f"  {lang:>10}  {len(idxs):>7}  {wr_A1:>6}  {wr_A2:>6}  {wr_B:>6}  {cer_A1:>8.4f}  {cer_A2:>8.4f}  {cer_B:>8.4f}")
        per_lang[lang] = {
            "n": len(idxs),
            "A1": {"wrong": wr_A1, "cer": cer_A1},
            "A2": {"wrong": wr_A2, "cer": cer_A2},
            "B":  {"wrong": wr_B,  "cer": cer_B},
        }

    # -- 5d: 15 examples where A1 right and B wrong ---------------------------
    a1r_bw_examples: list[dict] = []
    for i in range(n):
        if a1_right[i] and not b_right[i]:
            a1r_bw_examples.append({
                "gt":    refs[i],
                "A1":    preds_A1[i],
                "B":     preds_B[i],
                "width": resized_widths[i],
                "lang":  langs[i],
            })
            if len(a1r_bw_examples) == 15:
                break

    print(f"\n  15 examples where A1 is right and B is wrong:")
    print(f"  {'#':>3}  {'rw':>4}  {'lang':>3}  {'ground truth':^30}  {'A1':^30}  {'B':^30}")
    for k, ex in enumerate(a1r_bw_examples, 1):
        gt_d  = ex["gt"][:28]
        a1_d  = ex["A1"][:28]
        b_d   = ex["B"][:28]
        print(f"  {k:>3}  {ex['width']:>4}  {ex['lang'][:2]:>3}  {gt_d:<30}  {a1_d:<30}  {b_d:<30}")

    print()
    step5 = {
        "contingency_A1_vs_B": contingency,
        "per_length_bucket": per_bucket,
        "per_language": per_lang,
        "a1_right_b_wrong_examples": a1r_bw_examples,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Save JSON
    # ══════════════════════════════════════════════════════════════════════════
    result = {
        "git_commit":   git_commit_hash(),
        "date":         str(date.today()),
        "weights_sha256": weights_sha,
        "charset_sha256": charset_sha,
        "device":       device,
        "n_total_rows": len(all_rows),
        "n_dropped":    dropped,
        "n_evaluated":  n,
        "step2_width_label": step2,
        "step3_pixel_identity": step3,
        "step4_four_runs": step4,
        "step5_error_analysis": step5,
    }

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
