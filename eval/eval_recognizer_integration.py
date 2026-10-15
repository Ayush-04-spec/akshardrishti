#!/usr/bin/env python3
"""Integration regression check for CRNNRecognizer after the right_pad change.

Verifies that:
  (a) right_pad=0 reproduces sweep V0 test CER to the last digit
  (b) right_pad=64 reproduces sweep R64 test CER to the last digit
  (c) right_pad=64 through the public recognize() call with beam decode
      from pipeline.yaml reports n, CER, word accuracy, S/D/I, per language

Reference values are read from results/crnn_padding_sweep.json — nothing is
hard-coded.  If (a) or (b) fails to match, the script stops and reports.

Usage
-----
    python eval/eval_recognizer_integration.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import date
from pathlib import Path

import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.recognize.base import build_recognizer                    # noqa: E402
from akshardrishti.recognize.crnn_model import Charset, greedy_decode        # noqa: E402
from akshardrishti.postprocess.text_repair import normalize_for_scoring      # noqa: E402
from train.dataset import load_index                                          # noqa: E402


# ────────────────────────────────────── helpers ───────────────────────────────

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
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(a[i-1] != b[j-1]))
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (a[i-1] != b[j-1]):
            subs += int(a[i-1] != b[j-1]); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j]+1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1
    return subs, dels, ins


def corpus_metrics(preds: list[str], refs: list[str], langs: list[str]) -> dict:
    total_err = total_chars = exact = subs = dels = ins = 0
    from collections import defaultdict
    lang_buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

    for p, r, l in zip(preds, refs, langs):
        pn = normalize_for_scoring(p)
        rn = normalize_for_scoring(r)
        total_err   += edit_distance(pn, rn)
        total_chars += len(rn)
        if pn == rn:
            exact += 1
        s, d, i = levenshtein_ops(rn, pn)
        subs += s; dels += d; ins += i
        lang_buckets[l][0].append(p)
        lang_buckets[l][1].append(r)

    n   = len(refs)
    cer = total_err / total_chars if total_chars else 0.0

    per_lang = {}
    for lang, (lp, lr) in sorted(lang_buckets.items()):
        le = sum(edit_distance(normalize_for_scoring(a), normalize_for_scoring(b))
                 for a, b in zip(lp, lr))
        lc = sum(len(normalize_for_scoring(b)) for b in lr)
        lx = sum(normalize_for_scoring(a) == normalize_for_scoring(b) for a, b in zip(lp, lr))
        per_lang[lang] = {
            "n": len(lr),
            "cer": le / lc if lc else 0.0,
            "word_accuracy": lx / len(lr) if lr else 0.0,
            "wrong_words": len(lr) - lx,
        }

    return {
        "n": n, "cer": cer,
        "word_accuracy": exact / n if n else 0.0,
        "wrong_words": n - exact,
        "substitutions": subs, "deletions": dels, "insertions": ins,
        "per_language": per_lang,
    }


# ─────────────────────────────────────────── main ────────────────────────────

def main() -> None:
    out_path = ROOT / "results/recognizer_integration.json"
    if out_path.exists():
        print(f"Output already exists: {out_path}  — delete to re-run.")
        sys.exit(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load sweep reference values ──────────────────────────────────────────
    sweep_path = ROOT / "results/crnn_padding_sweep.json"
    if not sweep_path.exists():
        print(f"Sweep results not found: {sweep_path}")
        sys.exit(1)
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    chosen_rule = sweep["chosen_rule"]

    ref_V0_cer    = sweep["test_V0"]["cer"]
    ref_V0_wrong  = sweep["test_V0"]["wrong_words"]
    ref_ch_cer    = sweep[f"test_{chosen_rule}"]["cer"]
    ref_ch_wrong  = sweep[f"test_{chosen_rule}"]["wrong_words"]

    print(f"Reference V0   CER={ref_V0_cer:.10f}  wrong={ref_V0_wrong}")
    print(f"Reference {chosen_rule}  CER={ref_ch_cer:.10f}  wrong={ref_ch_wrong}")
    print()

    # ── load pipeline config ──────────────────────────────────────────────────
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")

    # ── load & filter test rows ───────────────────────────────────────────────
    # Build a temporary recognizer to access Charset for the coverage filter
    rec_tmp = build_recognizer(backend_name, config=dict(backend_cfg), device=device)
    rec_tmp.ensure_loaded()
    charset: Charset = rec_tmp.charset

    test_all  = load_index(ROOT / "datasets/mozhi_raw/index_test.csv")
    test_rows = [r for r in test_all if charset.coverage(r["text"]) >= 0.999]
    test_refs = [unicodedata.normalize("NFC", r["text"]) for r in test_rows]
    test_langs = [r["language"] for r in test_rows]
    n = len(test_rows)
    print(f"Kept test rows: {n}  (expect 20,026)")
    del rec_tmp  # release; we'll rebuild with specific right_pad below
    print()

    results: dict = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # (a) right_pad=0 — must match sweep V0
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 56)
    print("(a) right_pad=0 — must reproduce sweep V0")
    cfg_a = dict(backend_cfg); cfg_a["right_pad"] = 0
    rec_a = build_recognizer(backend_name, config=cfg_a, device=device)
    rec_a.ensure_loaded()
    model_a = rec_a.model

    preds_a: list[str] = []
    model_a.eval()
    with torch.no_grad():
        for r in test_rows:
            img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                preds_a.append("")
                continue
            tensor = rec_a._prepare(img).to(device)
            lp = model_a(tensor)
            text, _ = greedy_decode(lp, charset)[0]
            preds_a.append(text)

    m_a = corpus_metrics(preds_a, test_refs, test_langs)
    print(f"  CER={m_a['cer']:.10f}  wrong={m_a['wrong_words']}")
    print(f"  ref CER={ref_V0_cer:.10f}  ref wrong={ref_V0_wrong}")

    cer_match_a   = abs(m_a["cer"]         - ref_V0_cer)   < 1e-10
    wrong_match_a = m_a["wrong_words"]    == ref_V0_wrong
    print(f"  CER match: {'PASS' if cer_match_a else 'FAIL'}  "
          f"wrong match: {'PASS' if wrong_match_a else 'FAIL'}")
    if not (cer_match_a and wrong_match_a):
        print(f"\n  STOP: (a) does not match sweep V0.")
        print(f"  dCER={abs(m_a['cer']-ref_V0_cer):.2e}  dwrong={m_a['wrong_words']-ref_V0_wrong}")
        sys.exit(1)
    results["a_right_pad_0"] = m_a

    # ═══════════════════════════════════════════════════════════════════════════
    # (b) right_pad=64 — must match sweep R64
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 56)
    print(f"(b) right_pad=64 — must reproduce sweep {chosen_rule}")
    cfg_b = dict(backend_cfg); cfg_b["right_pad"] = 64
    rec_b = build_recognizer(backend_name, config=cfg_b, device=device)
    rec_b.ensure_loaded()
    model_b = rec_b.model

    preds_b: list[str] = []
    model_b.eval()
    with torch.no_grad():
        for r in test_rows:
            img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                preds_b.append("")
                continue
            tensor = rec_b._prepare(img).to(device)
            lp = model_b(tensor)
            text, _ = greedy_decode(lp, charset)[0]
            preds_b.append(text)

    m_b = corpus_metrics(preds_b, test_refs, test_langs)
    print(f"  CER={m_b['cer']:.10f}  wrong={m_b['wrong_words']}")
    print(f"  ref CER={ref_ch_cer:.10f}  ref wrong={ref_ch_wrong}")

    cer_match_b   = abs(m_b["cer"]         - ref_ch_cer)  < 1e-10
    wrong_match_b = m_b["wrong_words"]    == ref_ch_wrong
    print(f"  CER match: {'PASS' if cer_match_b else 'FAIL'}  "
          f"wrong match: {'PASS' if wrong_match_b else 'FAIL'}")
    if not (cer_match_b and wrong_match_b):
        print(f"\n  STOP: (b) does not match sweep {chosen_rule}.")
        print(f"  dCER={abs(m_b['cer']-ref_ch_cer):.2e}  dwrong={m_b['wrong_words']-ref_ch_wrong}")
        sys.exit(1)
    results["b_right_pad_64_greedy"] = m_b

    # ═══════════════════════════════════════════════════════════════════════════
    # (c) right_pad=64 through the public recognize() — beam decode from config
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 56)
    beam_w = int(backend_cfg.get("beam_width", 10))
    print(f"(c) right_pad=64 via public recognize() — beam_width={beam_w}")
    # rec_b already has right_pad=64 and beam_width from config
    preds_c: list[str] = []
    for r in test_rows:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            preds_c.append("")
            continue
        result = rec_b(img, "deva")          # __call__ -> ensure_loaded -> recognize
        preds_c.append(result.text)

    m_c = corpus_metrics(preds_c, test_refs, test_langs)
    print(f"  n={m_c['n']}  CER={m_c['cer']:.8f}  word_acc={m_c['word_accuracy']:.6f}"
          f"  wrong={m_c['wrong_words']}  S={m_c['substitutions']}"
          f"  D={m_c['deletions']}  I={m_c['insertions']}")
    for lang, lm in m_c["per_language"].items():
        print(f"    {lang:>10}  n={lm['n']:6d}  CER={lm['cer']:.6f}"
              f"  wrong={lm['wrong_words']:5d}")
    results["c_right_pad_64_beam"] = m_c

    # ═══════════════════════════════════════════════════════════════════════════
    # Save JSON
    # ═══════════════════════════════════════════════════════════════════════════
    output = {
        "git_commit":    git_commit_hash(),
        "date":          str(date.today()),
        "weights_sha256": weights_sha,
        "charset_sha256": charset_sha,
        "device":        device,
        "n_test":        n,
        "chosen_rule":   chosen_rule,
        "ref_V0_cer":    ref_V0_cer,
        f"ref_{chosen_rule}_cer": ref_ch_cer,
        "a_right_pad_0":        results["a_right_pad_0"],
        "b_right_pad_64_greedy": results["b_right_pad_64_greedy"],
        "c_right_pad_64_beam":   results["c_right_pad_64_beam"],
        "checks": {
            "a_cer_matches_V0":       cer_match_a,
            "a_wrong_matches_V0":     wrong_match_a,
            "b_cer_matches_chosen":   cer_match_b,
            "b_wrong_matches_chosen": wrong_match_b,
        },
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path}")
    print("All checks PASSED.")


if __name__ == "__main__":
    main()
