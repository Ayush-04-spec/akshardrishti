#!/usr/bin/env python3
"""Reproduce and verify CRNN performance on the Mozhi test split.

Runs three evaluation modes against the shipped slim checkpoint:

  Mode A  training-path   LineImageDataset + ctc_collate + greedy_decode
                          (identical to how test_metrics.json was produced
                           during the Colab training run)

  Mode B  inference-greedy   per-sample cv2.imread -> recognizer._prepare
                              -> model -> greedy_decode

  Mode C  inference-beam     same as B, beam_width = 10 (from pipeline.yaml)

Scoring uses normalize_for_scoring only (encoding-level normalisation; no
post-processing repair is applied to ground truth — Rule 2).

Outputs
-------
  results/crnn_test_repro.json  (written once, never overwrites existing)

Usage
-----
    python eval/eval_crnn_test.py
    python eval/eval_crnn_test.py --data datasets/mozhi_raw --batch 64
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.recognize.base import build_recognizer                        # noqa: E402
from akshardrishti.recognize.crnn_model import Charset, beam_decode, greedy_decode  # noqa: E402
from akshardrishti.postprocess.text_repair import normalize_for_scoring          # noqa: E402
from train.dataset import LineImageDataset, ctc_collate, load_index              # noqa: E402


# ──────────────────────────────────────────────────────── helpers ────────────

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


def corpus_cer_scalar(preds: list[str], refs: list[str]) -> float:
    """Total edit distance / total reference characters, after normalize_for_scoring.
    Matches exactly the local corpus_cer used in train_crnn.py."""
    total_err = sum(
        edit_distance(normalize_for_scoring(p), normalize_for_scoring(r))
        for p, r in zip(preds, refs)
    )
    total_len = sum(len(normalize_for_scoring(r)) for r in refs)
    return total_err / total_len if total_len else 0.0


def word_accuracy(preds: list[str], refs: list[str]) -> float:
    exact = sum(
        normalize_for_scoring(p) == normalize_for_scoring(r)
        for p, r in zip(preds, refs)
    )
    return exact / len(refs) if refs else 0.0


def per_language_metrics(preds: list[str], refs: list[str],
                         langs: list[str]) -> dict[str, dict]:
    buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p, r, l in zip(preds, refs, langs):
        buckets[l][0].append(p)
        buckets[l][1].append(r)
    result = {}
    for lang, (ps, rs) in sorted(buckets.items()):
        result[lang] = {
            "n": len(ps),
            "cer": corpus_cer_scalar(ps, rs),
            "word_accuracy": word_accuracy(ps, rs),
        }
    return result


def worst_samples(preds: list[str], refs: list[str], n: int = 10) -> list[dict]:
    scored = [
        (edit_distance(normalize_for_scoring(p), normalize_for_scoring(r)),
         normalize_for_scoring(r), normalize_for_scoring(p))
        for p, r in zip(preds, refs)
    ]
    scored.sort(reverse=True)
    return [{"gt": r, "pred": p, "edit_dist": d} for d, r, p in scored[:n]]


def mode_metrics(preds: list[str], refs: list[str], langs: list[str]) -> dict:
    return {
        "n": len(refs),
        "cer": corpus_cer_scalar(preds, refs),
        "word_accuracy": word_accuracy(preds, refs),
        "per_language": per_language_metrics(preds, refs, langs),
        "worst_10": worst_samples(preds, refs),
    }


# ──────────────────────────────────────────────────────── main ───────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/mozhi_raw")
    p.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.yaml")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    p.add_argument("--out", type=Path, default=ROOT / "results/crnn_test_repro.json")
    args = p.parse_args()

    # ── guard: do not overwrite an existing result file ──────────────────────
    if args.out.exists():
        print(f"Output already exists: {args.out}")
        print("Delete it manually if you want to re-run. Exiting.")
        sys.exit(0)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── load pipeline config ──────────────────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backend_name = cfg["recognize"]["deva_backend"]
    backend_cfg  = dict(cfg["recognize"]["backends"][backend_name])

    # Resolve relative paths against repo root
    for key in ("weights", "charset"):
        bpath = Path(backend_cfg[key])
        if not bpath.is_absolute():
            backend_cfg[key] = str(ROOT / bpath)

    weights_path = Path(backend_cfg["weights"])
    charset_path = Path(backend_cfg["charset"])
    beam_width   = int(backend_cfg.get("beam_width", 10))

    print(f"Weights : {weights_path}")
    print(f"Charset : {charset_path}")
    print(f"Beam    : {beam_width}")
    print()

    # ── SHA-256 provenance ────────────────────────────────────────────────────
    weights_sha = sha256_of(weights_path)
    charset_sha = sha256_of(charset_path)
    print(f"weights SHA-256: {weights_sha}")
    print(f"charset SHA-256: {charset_sha}")
    print()

    # ── build recognizer (exactly as the pipeline does) ───────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    recognizer = build_recognizer(backend_name, config=backend_cfg, device=device)
    recognizer.ensure_loaded()
    charset: Charset = recognizer.charset
    model = recognizer.model
    print(f"len(recognizer.charset) = {len(charset)}  (expect 161 incl. CTC blank)")
    print()

    # ── load and filter test rows ─────────────────────────────────────────────
    test_csv = args.data / "index_test.csv"
    all_rows = load_index(test_csv)
    print(f"Rows in index_test.csv: {len(all_rows)}")

    kept_rows = [r for r in all_rows if charset.coverage(r["text"]) >= 0.999]
    dropped   = len(all_rows) - len(kept_rows)
    print(f"Dropped by coverage filter (< 0.999): {dropped}")

    if dropped:
        print("  Dropped ground-truth strings:")
        dropped_rows = [r for r in all_rows if charset.coverage(r["text"]) < 0.999]
        for r in dropped_rows:
            print(f"    {repr(r['text'])}  (lang={r['language']})")
    print(f"Evaluating on: {len(kept_rows)} samples")
    print()

    refs  = [unicodedata.normalize("NFC", r["text"]) for r in kept_rows]
    langs = [r["language"] for r in kept_rows]

    # ── MODE A: training-path ─────────────────────────────────────────────────
    print("=" * 56)
    print("Mode A: training-path (LineImageDataset + ctc_collate + greedy_decode)")
    t0 = time.perf_counter()

    img_h  = int(backend_cfg.get("img_height", 32))
    max_w  = int(backend_cfg.get("max_width",  512))

    test_ds = LineImageDataset(
        kept_rows, charset, img_height=img_h, max_width=max_w, augment=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=ctc_collate
    )

    preds_A: list[str] = []
    model.eval()
    with torch.no_grad():
        for images, _, _, _, texts in test_loader:
            log_probs = model(images.to(device))
            preds_A.extend(t for t, _ in greedy_decode(log_probs, charset))

    elapsed_A = time.perf_counter() - t0
    metrics_A = mode_metrics(preds_A, refs, langs)
    print(f"  n               : {metrics_A['n']}")
    print(f"  corpus CER      : {metrics_A['cer']:.6f}")
    print(f"  word accuracy   : {metrics_A['word_accuracy']:.6f}")
    print(f"  per language    :")
    for lang, m in metrics_A["per_language"].items():
        print(f"    {lang:10s}  n={m['n']:6d}  CER={m['cer']:.6f}  word_acc={m['word_accuracy']:.6f}")
    print(f"  elapsed         : {elapsed_A:.1f}s")
    print("  worst 10 samples (gt -> pred):")
    for s in metrics_A["worst_10"]:
        print(f"    [{s['edit_dist']:3d}] {s['gt']!r:40s} -> {s['pred']!r}")
    print()

    # ── MODE B: inference-greedy ──────────────────────────────────────────────
    print("=" * 56)
    print("Mode B: inference-greedy (cv2.imread -> _prepare -> greedy_decode)")
    t0 = time.perf_counter()

    preds_B: list[str] = []
    model.eval()
    with torch.no_grad():
        for r in kept_rows:
            img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                preds_B.append("")
                continue
            tensor = recognizer._prepare(img).to(device)
            log_probs = model(tensor)
            text, _ = greedy_decode(log_probs, charset)[0]
            preds_B.append(text)

    elapsed_B = time.perf_counter() - t0
    metrics_B = mode_metrics(preds_B, refs, langs)
    print(f"  n               : {metrics_B['n']}")
    print(f"  corpus CER      : {metrics_B['cer']:.6f}")
    print(f"  word accuracy   : {metrics_B['word_accuracy']:.6f}")
    print(f"  per language    :")
    for lang, m in metrics_B["per_language"].items():
        print(f"    {lang:10s}  n={m['n']:6d}  CER={m['cer']:.6f}  word_acc={m['word_accuracy']:.6f}")
    print(f"  elapsed         : {elapsed_B:.1f}s")
    print("  worst 10 samples (gt -> pred):")
    for s in metrics_B["worst_10"]:
        print(f"    [{s['edit_dist']:3d}] {s['gt']!r:40s} -> {s['pred']!r}")
    print()

    # ── MODE C: inference-beam ────────────────────────────────────────────────
    print("=" * 56)
    print(f"Mode C: inference-beam (beam_width={beam_width})")
    t0 = time.perf_counter()

    preds_C: list[str] = []
    model.eval()
    with torch.no_grad():
        for r in kept_rows:
            img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                preds_C.append("")
                continue
            tensor = recognizer._prepare(img).to(device)
            log_probs = model(tensor)
            text, _ = beam_decode(log_probs, charset, beam_width=beam_width)[0]
            preds_C.append(text)

    elapsed_C = time.perf_counter() - t0
    metrics_C = mode_metrics(preds_C, refs, langs)
    print(f"  n               : {metrics_C['n']}")
    print(f"  corpus CER      : {metrics_C['cer']:.6f}")
    print(f"  word accuracy   : {metrics_C['word_accuracy']:.6f}")
    print(f"  per language    :")
    for lang, m in metrics_C["per_language"].items():
        print(f"    {lang:10s}  n={m['n']:6d}  CER={m['cer']:.6f}  word_acc={m['word_accuracy']:.6f}")
    print(f"  elapsed         : {elapsed_C:.1f}s")
    print("  worst 10 samples (gt -> pred):")
    for s in metrics_C["worst_10"]:
        print(f"    [{s['edit_dist']:3d}] {s['gt']!r:40s} -> {s['pred']!r}")
    print()

    # ── write results JSON ────────────────────────────────────────────────────
    # Convert worst_10 dicts to lists for clean JSON
    def serialise_mode(m: dict) -> dict:
        out = {k: v for k, v in m.items() if k != "worst_10"}
        out["worst_10"] = m["worst_10"]
        return out

    result = {
        "git_commit": git_commit_hash(),
        "date": str(date.today()),
        "weights_sha256": weights_sha,
        "charset_sha256": charset_sha,
        "device": device,
        "n_total_rows": len(all_rows),
        "n_dropped_coverage": dropped,
        "n_evaluated": len(kept_rows),
        "mode_A_training_path": serialise_mode(metrics_A),
        "mode_B_inference_greedy": serialise_mode(metrics_B),
        "mode_C_inference_beam": serialise_mode(metrics_C),
    }

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
