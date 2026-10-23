"""
End-to-end CER+WER comparison.
Now includes proper WER calculation.
"""
from __future__ import annotations
import json, time, warnings, sys
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from akshardrishti.config import Config
from akshardrishti.pipeline import AksharDrishtiPipeline
from akshardrishti.postprocess.text_repair import normalize_for_scoring

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
    total_err = total_chars = 0
    for p, r in zip(preds, refs):
        pn = normalize_for_scoring(p)
        rn = normalize_for_scoring(r)
        total_err   += edit_distance(pn, rn)
        total_chars += len(rn)
    return total_err / total_chars if total_chars else 0.0

def corpus_wer(preds: list[str], refs: list[str]) -> float:
    """Word error rate: total word-level edit distance / total reference words."""
    total_err = total_words = 0
    for p, r in zip(preds, refs):
        p_words = normalize_for_scoring(p).split()
        r_words = normalize_for_scoring(r).split()
        # Word-level edit distance
        if not r_words:
            total_err += len(p_words)
            continue
        if not p_words:
            total_err += len(r_words)
            total_words += len(r_words)
            continue
        # Levenshtein on word lists
        prev = list(range(len(r_words) + 1))
        for pw in p_words:
            cur = [prev[0] + 1]
            for j, rw in enumerate(r_words, 1):
                cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(pw!=rw)))
            prev = cur
        total_err += prev[-1]
        total_words += len(r_words)
    return total_err / total_words if total_words else 0.0

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
refs = [p["gt_text"] for p in pages]

# System A: layout ON
print("="*56)
print("System A: AksharDrishti layout ON")
cfg_on = Config.load()
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
wer_A  = corpus_wer(preds_A, refs)
print(f"  CER={cer_A:.4f}  WER={wer_A:.4f}  avg={sum(times_A)/len(times_A):.1f}s/page")
print()

# System B: layout OFF
print("="*56)
print("System B: AksharDrishti layout OFF")
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
wer_B  = corpus_wer(preds_B, refs)
print(f"  CER={cer_B:.4f}  WER={wer_B:.4f}  avg={sum(times_B)/len(times_B):.1f}s/page")
print()

# System C: Tesseract
print("="*56)
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
    wer_C  = corpus_wer(preds_C, refs)
    tess_ok = True
    print(f"  CER={cer_C:.4f}  WER={wer_C:.4f}  avg={sum(times_C)/len(times_C):.1f}s/page")
except Exception as e:
    print(f"  Tesseract unavailable: {e}")
    cer_C = wer_C = None
    tess_ok = False
print()

# Results
print("="*56)
print("RESULTS")
print("="*56)
print(f"{'System':<35}  {'CER':>8}  {'WER':>8}")
print("-"*56)
print(f"{'AksharDrishti layout ON':<35}  {cer_A:>8.4f}  {wer_A:>8.4f}")
print(f"{'AksharDrishti layout OFF':<35}  {cer_B:>8.4f}  {wer_B:>8.4f}")
if tess_ok:
    print(f"{'Tesseract fullpage':<35}  {cer_C:>8.4f}  {wer_C:>8.4f}")
print()

if cer_B > 0:
    print(f"Layout ON vs OFF: CER {(cer_B-cer_A)/cer_B*100:+.1f}% reduction")
if tess_ok and cer_C:
    print(f"Layout ON vs Tesseract: CER {(cer_C-cer_A)/cer_C*100:+.1f}% reduction")

# Save
out = {
    "n_pages": len(pages),
    "layout_on":  {"cer": cer_A, "wer": wer_A},
    "layout_off": {"cer": cer_B, "wer": wer_B},
    "tesseract":  {"cer": cer_C, "wer": wer_C} if tess_ok else None,
}
out_path = ROOT / "results" / "cer_wer_comparison.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved: {out_path}")
