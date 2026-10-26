"""
End-to-end CER comparison for Issue #2.

Runs 3 systems on the 5 pages that have gt_text transcriptions:
  A) AksharDrishti  layout ON   (full pipeline)
  B) AksharDrishti  layout OFF  (whole-page OCR, no layout detection)
  C) Tesseract      fullpage    (external baseline)

Output: results/cer_comparison.json + printed table
"""
from __future__ import annotations
import json, time, unicodedata, warnings, sys
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.config import Config
from akshardrishti.pipeline import AksharDrishtiPipeline
from akshardrishti.postprocess.text_repair import normalize_for_scoring

# ── helpers ──────────────────────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        cur = [prev[0] + 1]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca!=cb)))
        prev = cur
    return prev[-1]

def corpus_cer(preds: list[str], refs: list[str]) -> float:
    if len(preds) != len(refs):
        raise ValueError(f"length mismatch: {len(preds)} predictions vs {len(refs)} references")
    total_err = total_chars = 0
    for p, r in zip(preds, refs):
        pn = normalize_for_scoring(p)
        rn = normalize_for_scoring(r)
        total_err   += edit_distance(pn, rn)
        total_chars += len(rn)
    return total_err / total_chars if total_chars else 0.0

def word_acc(preds: list[str], refs: list[str]) -> float:
    if len(preds) != len(refs):
        raise ValueError(f"length mismatch: {len(preds)} predictions vs {len(refs)} references")
    correct = sum(
        normalize_for_scoring(p).strip() == normalize_for_scoring(r).strip()
        for p, r in zip(preds, refs)
    )
    return correct / len(refs) if refs else 0.0

def corpus_wer(preds: list[str], refs: list[str]) -> float:
    """Word error rate: edit distance on word sequences."""
    if len(preds) != len(refs):
        raise ValueError(f"length mismatch: {len(preds)} predictions vs {len(refs)} references")
    total_err = total_words = 0
    for p, r in zip(preds, refs):
        pn = normalize_for_scoring(p).split()
        rn = normalize_for_scoring(r).split()
        total_err   += edit_distance(rn, pn)
        total_words += len(rn)
    return total_err / total_words if total_words else 0.0

# ── load pages that have gt_text ─────────────────────────────────────────────

gt_dir  = ROOT / "benchmark" / "gt_text"
img_dir = ROOT / "benchmark" / "images"

pages = []
for gt_file in sorted(gt_dir.glob("BM_*.txt")):
    img = img_dir / (gt_file.stem + ".png")
    if img.exists():
        pages.append({
            "id":       gt_file.stem,
            "image":    img,
            "gt_text":  gt_file.read_text(encoding="utf-8").strip(),
        })

print(f"Pages with ground truth: {len(pages)}")
for p in pages:
    print(f"  {p['id']}  ({len(p['gt_text'])} chars GT)")
print()

refs = [p["gt_text"] for p in pages]

# ── System A: layout ON ───────────────────────────────────────────────────────
print("=" * 56)
print("System A: AksharDrishti layout ON")

cfg_on = Config.load()
# Confirm layout is enabled
cfg_on._data["layout"]["enabled"] = True

pipe_on = AksharDrishtiPipeline(cfg_on)
preds_A, times_A = [], []
for p in pages:
    t0 = time.perf_counter()
    doc = pipe_on.process(p["image"])
    preds_A.append(doc.full_text())
    times_A.append(time.perf_counter() - t0)
    print(f"  {p['id']}  {times_A[-1]:.1f}s")

cer_A  = corpus_cer(preds_A, refs)
wacc_A = word_acc(preds_A, refs)
print(f"  CER={cer_A:.4f}  word_acc={wacc_A:.4f}  avg={sum(times_A)/len(times_A):.1f}s/page")
print()

# ── System B: layout OFF ──────────────────────────────────────────────────────
print("=" * 56)
print("System B: AksharDrishti layout OFF (whole-page OCR)")

cfg_off = Config.load()
cfg_off._data["layout"]["enabled"] = False

pipe_off = AksharDrishtiPipeline(cfg_off)
preds_B, times_B = [], []
for p in pages:
    t0 = time.perf_counter()
    doc = pipe_off.process(p["image"])
    preds_B.append(doc.full_text())
    times_B.append(time.perf_counter() - t0)
    print(f"  {p['id']}  {times_B[-1]:.1f}s")

cer_B  = corpus_cer(preds_B, refs)
wacc_B = word_acc(preds_B, refs)
print(f"  CER={cer_B:.4f}  word_acc={wacc_B:.4f}  avg={sum(times_B)/len(times_B):.1f}s/page")
print()

# ── System C: Tesseract fullpage ──────────────────────────────────────────────
print("=" * 56)
print("System C: Tesseract fullpage")

try:
    import pytesseract, cv2
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    preds_C, times_C = [], []
    for p in pages:
        img = cv2.imread(str(p["image"]))
        t0 = time.perf_counter()
        text = pytesseract.image_to_string(img, lang="hin+mar", config="--psm 1 --oem 1")
        preds_C.append(text.strip())
        times_C.append(time.perf_counter() - t0)
        print(f"  {p['id']}  {times_C[-1]:.1f}s")

    cer_C  = corpus_cer(preds_C, refs)
    wacc_C = word_acc(preds_C, refs)
    tess_ok = True
    print(f"  CER={cer_C:.4f}  word_acc={wacc_C:.4f}  avg={sum(times_C)/len(times_C):.1f}s/page")
except Exception as e:
    print(f"  Tesseract unavailable: {e}")
    preds_C = [""] * len(pages)
    cer_C = wacc_C = None
    tess_ok = False
print()

# ── Results table ─────────────────────────────────────────────────────────────
print("=" * 56)
print("RESULTS")
print("=" * 56)
print(f"{'System':<35}  {'CER':>8}  {'Word Acc':>10}")
print("-" * 56)
print(f"{'AksharDrishti layout ON':<35}  {cer_A:>8.4f}  {wacc_A:>10.4f}")
print(f"{'AksharDrishti layout OFF':<35}  {cer_B:>8.4f}  {wacc_B:>10.4f}")
if tess_ok:
    print(f"{'Tesseract fullpage':<35}  {cer_C:>8.4f}  {wacc_C:>10.4f}")
else:
    print(f"{'Tesseract fullpage':<35}  {'N/A':>8}  {'N/A':>10}")
print()

if cer_B > 0:
    improvement = (cer_B - cer_A) / cer_B * 100
    print(f"Layout ON vs OFF: {improvement:+.1f}% CER reduction")
if tess_ok and cer_C and cer_C > 0:
    vs_tess = (cer_C - cer_A) / cer_C * 100
    print(f"Layout ON vs Tesseract: {vs_tess:+.1f}% CER reduction")

# ── per-page breakdown ────────────────────────────────────────────────────────
print()
print("Per-page CER:")
print(f"{'Page':<12}  {'Layout ON':>10}  {'Layout OFF':>10}  {'Tesseract':>10}")
print("-" * 48)
for i, p in enumerate(pages):
    cA = corpus_cer([preds_A[i]], [refs[i]])
    cB = corpus_cer([preds_B[i]], [refs[i]])
    cC = corpus_cer([preds_C[i]], [refs[i]]) if tess_ok else None
    tess_str = f"{cC:>10.4f}" if cC is not None else f"{'N/A':>10}"
    print(f"{p['id']:<12}  {cA:>10.4f}  {cB:>10.4f}  {tess_str}")

# ── save JSON ─────────────────────────────────────────────────────────────────
out = {
    "n_pages": len(pages),
    "pages": [p["id"] for p in pages],
    "layout_on":  {"cer": cer_A,  "word_accuracy": wacc_A},
    "layout_off": {"cer": cer_B,  "word_accuracy": wacc_B},
    "tesseract":  {"cer": cer_C,  "word_accuracy": wacc_C} if tess_ok else None,
    "layout_improvement_pct": round((cer_B - cer_A) / cer_B * 100, 2) if cer_B > 0 else None,
    "per_page": {
        p["id"]: {
            "layout_on":  corpus_cer([preds_A[i]], [refs[i]]),
            "layout_off": corpus_cer([preds_B[i]], [refs[i]]),
            "tesseract":  corpus_cer([preds_C[i]], [refs[i]]) if tess_ok else None,
        }
        for i, p in enumerate(pages)
    }
}

out_path = ROOT / "results" / "cer_comparison.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved: {out_path}")
