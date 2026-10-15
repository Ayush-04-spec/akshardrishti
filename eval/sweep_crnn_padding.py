#!/usr/bin/env python3
"""CRNN right-padding sweep: find the padding rule that closes the training /
inference-path CER gap, selected on validation, evaluated on test once.

Variants
--------
  V0          no padding (current pipeline behaviour)
  R8..R64     append exactly N columns on the right
  W96..W512   right-pad to at least N columns (no-op if already wider)
  L16         CONTROL: pad left only, 16 columns
  B16         CONTROL: pad both sides, 8 columns each
  BATCH64     REFERENCE: batch-64 CSV-order with batch-max padding

All variants use greedy decode, eval mode, no_grad.

Speed trick: group rows that produce the same final padded width and run each
group as a single forward pass — equivalent to batch-1 for distinct widths.

Outputs
-------
  results/crnn_padding_sweep.json

Usage
-----
    python eval/sweep_crnn_padding.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.recognize.base import build_recognizer                        # noqa: E402
from akshardrishti.recognize.crnn_model import Charset, greedy_decode           # noqa: E402
from akshardrishti.postprocess.text_repair import normalize_for_scoring          # noqa: E402
from train.dataset import LineImageDataset, ctc_collate, load_index              # noqa: E402


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


def levenshtein_ops(a: str, b: str) -> tuple[int, int, int]:
    """Returns (substitutions, deletions, insertions) for a->b."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                           dp[i-1][j-1] + (a[i-1] != b[j-1]))
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (a[i-1] != b[j-1]):
            subs += int(a[i-1] != b[j-1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j]+1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1
    return subs, dels, ins


def corpus_metrics(preds: list[str], refs: list[str]) -> dict:
    total_err = total_chars = 0
    exact = subs = dels = ins = 0
    for p, r in zip(preds, refs):
        pn = normalize_for_scoring(p)
        rn = normalize_for_scoring(r)
        ed = edit_distance(pn, rn)
        total_err   += ed
        total_chars += len(rn)
        if pn == rn:
            exact += 1
        s, d, i = levenshtein_ops(rn, pn)   # ref -> hyp
        subs += s; dels += d; ins += i
    n = len(refs)
    cer = total_err / total_chars if total_chars else 0.0
    return {
        "n": n,
        "cer": cer,
        "word_accuracy": exact / n if n else 0.0,
        "wrong_words": n - exact,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "total_err": total_err,
        "total_chars": total_chars,
    }


def per_lang_metrics(preds: list[str], refs: list[str], langs: list[str]) -> dict:
    buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p, r, l in zip(preds, refs, langs):
        buckets[l][0].append(p)
        buckets[l][1].append(r)
    return {lang: corpus_metrics(ps, rs) for lang, (ps, rs) in sorted(buckets.items())}


# ────────────────────────────────────── padding variants ─────────────────────

def apply_padding(tensor: torch.Tensor, variant: str) -> torch.Tensor:
    """
    tensor: (1, 1, H, W) from recognizer._prepare()
    Returns: (1, 1, H, W') with padding applied per variant.
    White in [-1,1] normalised space = 1.0.
    """
    _, _, H, W = tensor.shape
    WHITE = 1.0

    if variant == "V0":
        return tensor

    if variant.startswith("R"):
        n = int(variant[1:])
        return F.pad(tensor, (0, n), value=WHITE)       # (left, right, top, bot) for F.pad 2D

    if variant.startswith("W"):
        target = int(variant[1:])
        if W >= target:
            return tensor
        return F.pad(tensor, (0, target - W), value=WHITE)

    if variant == "L16":
        return F.pad(tensor, (16, 0), value=WHITE)

    if variant == "B16":
        return F.pad(tensor, (8, 8), value=WHITE)

    raise ValueError(f"unknown variant: {variant!r}")


# ─────────────────────────── grouped inference (speed trick) ─────────────────

def run_variant(
    rows: list[dict],
    refs: list[str],
    langs: list[str],
    recognizer,
    model,
    charset: Charset,
    device: str,
    variant: str,
) -> list[str]:
    """Run a single padding variant over all rows.

    Groups rows that produce the same final padded width into mini-batches.
    Within a group every tensor has the same W so we can stack and run one
    forward pass — identical to batch-1 when all widths are unique.
    """
    # Build tensors and compute final padded width for grouping
    tensors: list[torch.Tensor] = []
    for r in rows:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img_h = int(recognizer.config.get("img_height", 32))
            img = (torch.ones(1, 1, img_h, 32) * 1.0)
            tensors.append(img)
            continue
        t = recognizer._prepare(img)               # (1, 1, H, W)
        t = apply_padding(t, variant)              # (1, 1, H, W')
        tensors.append(t)

    # Group by final width
    width_to_idxs: dict[int, list[int]] = defaultdict(list)
    for idx, t in enumerate(tensors):
        width_to_idxs[t.shape[-1]].append(idx)

    preds = [""] * len(rows)
    model.eval()
    with torch.no_grad():
        for width, idxs in width_to_idxs.items():
            # Process each width-group in sub-batches of <=64 to stay within VRAM
            for batch_start in range(0, len(idxs), 64):
                batch_idxs = idxs[batch_start:batch_start + 64]
                batch = torch.cat([tensors[i] for i in batch_idxs], dim=0).to(device)
                log_probs = model(batch)
                decoded = greedy_decode(log_probs, charset)
                for pos, idx in enumerate(batch_idxs):
                    preds[idx] = decoded[pos][0]

    return preds


def run_batch64(
    rows: list[dict],
    refs: list[str],
    langs: list[str],
    recognizer,
    model,
    charset: Charset,
    device: str,
    img_h: int,
    max_w: int,
    batch_size: int = 64,
    workers: int = 0,
) -> list[str]:
    """BATCH64 reference: LineImageDataset + ctc_collate, batch=64, CSV order."""
    charset_obj = recognizer.charset
    test_ds = LineImageDataset(rows, charset_obj, img_height=img_h,
                               max_width=max_w, augment=False, drop_uncovered=False)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=ctc_collate
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for images, _, _, _, texts in loader:
            lp = model(images.to(device))
            preds.extend(t for t, _ in greedy_decode(lp, charset))
    return preds


# ─────────────────────────────────────────────── main ────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data",    type=Path, default=ROOT / "datasets/mozhi_raw")
    ap.add_argument("--config",  type=Path, default=ROOT / "configs/pipeline.yaml")
    ap.add_argument("--workers", type=int,  default=0)
    ap.add_argument("--out",     type=Path,
                    default=ROOT / "results/crnn_padding_sweep.json")
    args = ap.parse_args()

    if args.out.exists():
        print(f"Output already exists: {args.out}  — delete to re-run.")
        sys.exit(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── load config & recognizer ──────────────────────────────────────────────
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
    img_h        = int(backend_cfg.get("img_height", 32))
    max_w        = int(backend_cfg.get("max_width",  512))

    weights_sha = sha256_of(weights_path)
    charset_sha = sha256_of(charset_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")
    print(f"Weights : {weights_path.name}  SHA={weights_sha[:16]}...")
    print()

    recognizer = build_recognizer(backend_name, config=backend_cfg, device=device)
    recognizer.ensure_loaded()
    charset: Charset = recognizer.charset
    model           = recognizer.model

    # ── load and filter validation rows ───────────────────────────────────────
    val_all  = load_index(args.data / "index_val.csv")
    val_rows = [r for r in val_all  if charset.coverage(r["text"]) >= 0.999]
    val_refs = [unicodedata.normalize("NFC", r["text"]) for r in val_rows]
    val_langs = [r["language"] for r in val_rows]
    print(f"Val  total={len(val_all)}  dropped={len(val_all)-len(val_rows)}  kept={len(val_rows)}"
          f"  (expect 20,118)")

    # ── load and filter test rows ─────────────────────────────────────────────
    test_all  = load_index(args.data / "index_test.csv")
    test_rows = [r for r in test_all  if charset.coverage(r["text"]) >= 0.999]
    test_refs = [unicodedata.normalize("NFC", r["text"]) for r in test_rows]
    test_langs = [r["language"] for r in test_rows]
    print(f"Test total={len(test_all)}  dropped={len(test_all)-len(test_rows)}  kept={len(test_rows)}"
          f"  (expect 20,026)")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1a — grouped-vs-batch-1 sanity check on 500 random val rows
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Sanity check: grouped == batch-1 on 500 random val rows for V0 and R32")
    rng = random.Random(42)
    check_idxs = rng.sample(range(len(val_rows)), 500)
    check_rows = [val_rows[i] for i in check_idxs]

    for chk_variant in ("V0", "R32"):
        # grouped method
        preds_grouped = run_variant(check_rows, None, None, recognizer, model,
                                    charset, device, chk_variant)
        # true batch-1 loop
        preds_b1: list[str] = []
        model.eval()
        with torch.no_grad():
            for r in check_rows:
                img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    preds_b1.append("")
                    continue
                t = recognizer._prepare(img)
                t = apply_padding(t, chk_variant)
                lp = model(t.to(device))
                text, _ = greedy_decode(lp, charset)[0]
                preds_b1.append(text)

        n_match = sum(a == b for a, b in zip(preds_grouped, preds_b1))
        n_diff  = 500 - n_match
        ok = n_diff == 0
        print(f"  {chk_variant}: {n_match}/500 match, {n_diff} differ — {'PASS' if ok else 'STOP'}")
        if not ok:
            print("  STOP: grouped method does not reproduce batch-1. Aborting.")
            sys.exit(1)

    print()
    sanity_passed = True

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1b — validation sweep over all variants
    # ═══════════════════════════════════════════════════════════════════════════
    ALL_VARIANTS = ["V0", "R8", "R16", "R32", "R64",
                    "W96", "W128", "W192", "W256", "W512",
                    "L16", "B16", "BATCH64"]

    print("=" * 60)
    print(f"Validation sweep ({len(val_rows)} rows)")
    val_preds:   dict[str, list[str]] = {}
    val_metrics: dict[str, dict]      = {}

    for variant in ALL_VARIANTS:
        if variant == "BATCH64":
            preds = run_batch64(val_rows, val_refs, val_langs, recognizer, model,
                                charset, device, img_h, max_w, workers=args.workers)
        else:
            preds = run_variant(val_rows, val_refs, val_langs, recognizer, model,
                                charset, device, variant)
        m = corpus_metrics(preds, val_refs)
        m["per_language"] = per_lang_metrics(preds, val_refs, val_langs)
        val_preds[variant]   = preds
        val_metrics[variant] = m
        print(f"  {variant:<10}  CER={m['cer']:.6f}  word_acc={m['word_accuracy']:.6f}"
              f"  wrong={m['wrong_words']:5d}")

    # Print sorted table
    print()
    print("Validation results sorted by CER:")
    sorted_variants = sorted(ALL_VARIANTS, key=lambda v: val_metrics[v]["cer"])
    hdr = f"  {'variant':<10}  {'CER':>10}  {'word_acc':>10}  {'wrong':>7}  {'S':>5}  {'D':>5}  {'I':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for v in sorted_variants:
        m = val_metrics[v]
        print(f"  {v:<10}  {m['cer']:>10.6f}  {m['word_accuracy']:>10.6f}"
              f"  {m['wrong_words']:>7}  {m['substitutions']:>5}  {m['deletions']:>5}"
              f"  {m['insertions']:>5}")

    print()
    # Per-language breakdown for each variant
    for v in sorted_variants:
        m = val_metrics[v]
        print(f"  {v} per-language:")
        for lang, lm in m["per_language"].items():
            print(f"    {lang:>10}  n={lm['n']:6d}  CER={lm['cer']:.6f}  wrong={lm['wrong_words']:5d}")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 — choose the rule on validation only
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 2 — Rule selection on validation")

    candidates = [v for v in ["R8","R16","R32","R64","W96","W128","W192","W256","W512"]]
    best_cand  = min(candidates, key=lambda v: val_metrics[v]["cer"])
    best_cer   = val_metrics[best_cand]["cer"]

    # If two variants are within 0.00005, take the smaller padding
    def padding_size(v: str) -> int:
        return int(v[1:])

    ties = [v for v in candidates if val_metrics[v]["cer"] <= best_cer + 0.00005]
    if len(ties) > 1:
        chosen = min(ties, key=padding_size)
        print(f"  Tie (within 0.00005 CER): {ties}  -> choosing smallest: {chosen}")
    else:
        chosen = best_cand

    chosen_cer    = val_metrics[chosen]["cer"]
    v0_cer        = val_metrics["V0"]["cer"]
    batch64_cer   = val_metrics["BATCH64"]["cer"]
    l16_cer       = val_metrics["L16"]["cer"]
    b16_cer       = val_metrics["B16"]["cer"]

    gap_closed = (v0_cer - chosen_cer) / (v0_cer - batch64_cer) if v0_cer > batch64_cer else float("nan")

    print(f"  V0       CER: {v0_cer:.6f}")
    print(f"  BATCH64  CER: {batch64_cer:.6f}  (training-path reference)")
    print(f"  Chosen [{chosen}] CER: {chosen_cer:.6f}")
    print(f"  Gap closed: {gap_closed:.1%}  "
          f"({'✓ closes gap' if gap_closed > 0.9 else '✗ does not fully close gap'})")
    print()
    print(f"  L16 (left-pad control) CER: {l16_cer:.6f}  "
          f"(vs V0: {v0_cer:.6f}) — "
          f"{'L16 helps: YES' if l16_cer < v0_cer - 0.00005 else 'L16 helps: NO (as predicted)'}")
    print(f"  B16 (both-side control) CER: {b16_cer:.6f}  "
          f"— {'helps' if b16_cer < v0_cer - 0.00005 else 'does not help'}")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 — test, once — only V0 and chosen variant
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"STEP 3 — Test set: V0  vs  {chosen}  (n={len(test_rows)})")

    test_preds_V0     = run_variant(test_rows, test_refs, test_langs, recognizer,
                                    model, charset, device, "V0")
    test_preds_chosen = run_variant(test_rows, test_refs, test_langs, recognizer,
                                    model, charset, device, chosen)

    test_m_V0     = corpus_metrics(test_preds_V0, test_refs)
    test_m_chosen = corpus_metrics(test_preds_chosen, test_refs)
    test_m_V0["per_language"]     = per_lang_metrics(test_preds_V0,     test_refs, test_langs)
    test_m_chosen["per_language"] = per_lang_metrics(test_preds_chosen, test_refs, test_langs)

    for label, tm in [("V0", test_m_V0), (chosen, test_m_chosen)]:
        print(f"  {label:<10}  CER={tm['cer']:.8f}  word_acc={tm['word_accuracy']:.6f}"
              f"  wrong={tm['wrong_words']:5d}  S={tm['substitutions']}  D={tm['deletions']}"
              f"  I={tm['insertions']}")
        for lang, lm in tm["per_language"].items():
            print(f"    {lang:>10}  n={lm['n']:6d}  CER={lm['cer']:.6f}  wrong={lm['wrong_words']:5d}")
    print()

    # Verify V0 test CER matches previous run
    ref_v0_cer = 0.0077467340110393
    if abs(test_m_V0["cer"] - ref_v0_cer) > 0.0001:
        print(f"WARNING: V0 test CER {test_m_V0['cer']:.10f} differs from reference {ref_v0_cer:.10f}")
    else:
        print(f"  V0 test CER matches previous run ✓")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5 — 15 examples each direction on test
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"STEP 5 — Test examples V0 vs {chosen}")

    # Compute resized widths for test rows (needed for display)
    test_widths = []
    for r in test_rows:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            test_widths.append(8)
        else:
            h, w = img.shape[:2]
            test_widths.append(max(8, int(round(w * img_h / max(h, 1)))))

    v0_right  = [normalize_for_scoring(p) == normalize_for_scoring(r)
                 for p, r in zip(test_preds_V0, test_refs)]
    ch_right  = [normalize_for_scoring(p) == normalize_for_scoring(r)
                 for p, r in zip(test_preds_chosen, test_refs)]

    v0w_chR: list[dict] = []
    v0R_chW: list[dict] = []
    for i in range(len(test_rows)):
        entry = {
            "gt": test_refs[i], "V0": test_preds_V0[i],
            "chosen": test_preds_chosen[i],
            "width": test_widths[i], "lang": test_langs[i],
        }
        if not v0_right[i] and ch_right[i] and len(v0w_chR) < 15:
            v0w_chR.append(entry)
        if v0_right[i] and not ch_right[i] and len(v0R_chW) < 15:
            v0R_chW.append(entry)

    def print_examples(title: str, examples: list[dict]) -> None:
        print(f"\n  {title} ({len(examples)} shown):")
        print(f"  {'#':>3}  {'rw':>4}  {'la':>2}  {'ground truth':<30}  {'V0':<30}  {chosen:<30}")
        for k, ex in enumerate(examples, 1):
            gt = ex["gt"][:28]
            v0 = ex["V0"][:28]
            ch = ex["chosen"][:28]
            print(f"  {k:>3}  {ex['width']:>4}  {ex['lang'][:2]:>2}  "
                  f"{gt:<30}  {v0:<30}  {ch:<30}")

    print_examples(f"V0 wrong, {chosen} right", v0w_chR)
    print_examples(f"V0 right, {chosen} wrong", v0R_chW)
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 — save JSON
    # ═══════════════════════════════════════════════════════════════════════════

    # Trim per_document_cer from metrics before serialising (too large)
    def trim(m: dict) -> dict:
        return {k: v for k, v in m.items() if k != "per_document_cer"}

    def trim_m(m: dict) -> dict:
        out = trim(m)
        if "per_language" in out:
            out["per_language"] = {l: trim(lm) for l, lm in out["per_language"].items()}
        return out

    result = {
        "git_commit":    git_commit_hash(),
        "date":          str(date.today()),
        "weights_sha256": weights_sha,
        "charset_sha256": charset_sha,
        "device":        device,
        "sanity_check_grouped_vs_batch1": sanity_passed,
        "chosen_rule":   chosen,
        "val_n":         len(val_rows),
        "val_metrics":   {v: trim_m(val_metrics[v]) for v in ALL_VARIANTS},
        "test_n":        len(test_rows),
        "test_V0":       trim_m(test_m_V0),
        f"test_{chosen}": trim_m(test_m_chosen),
        "step5_v0wrong_chosen_right": v0w_chR,
        "step5_v0right_chosen_wrong": v0R_chW,
    }

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
