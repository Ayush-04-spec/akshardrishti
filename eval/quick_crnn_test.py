"""
Quick direct CRNN comparison: base vs fine-tuned on govdoc crops.
No pipeline, no TrOCR, no TATR. Runs in ~30 seconds.
"""
import sys, csv, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2, torch
from akshardrishti.recognize.crnn_model import Charset, CRNN, greedy_decode
from akshardrishti.preprocess.enhance import preprocess_line_crop
from akshardrishti.postprocess.text_repair import normalize_for_scoring

ROOT = Path(__file__).parent.parent

def ed(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for ca in a:
        cur = [prev[0]+1]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca!=cb)))
        prev = cur
    return prev[-1]

def cer(preds, refs):
    err = chars = 0
    for p, r in zip(preds, refs):
        pn = normalize_for_scoring(p); rn = normalize_for_scoring(r)
        err += ed(pn, rn); chars += len(rn)
    return err/chars if chars else 0

def load_model(weights_path, charset):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = CRNN(num_classes=len(charset))
    key = "model" if "model" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model.eval()
    return model

def infer(model, img_path, charset):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return ""
    crop = preprocess_line_crop(img, target_height=32, binarize_crop=True)
    if crop.shape[1] > 512:
        crop = cv2.resize(crop, (512, 32))
    t = torch.from_numpy(crop.astype("float32")/255.0)
    t = ((t - 0.5)/0.5).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        text, _ = greedy_decode(model(t), charset)[0]
    return text

# Load charset
charset = Charset.load(ROOT / "weights/recognize/charset_deva.txt")

# Load both models
print("Loading base model...")
base = load_model(ROOT / "weights/recognize/crnn_mozhi_himr.pt", charset)
print("Loading fine-tuned model...")
ft   = load_model(ROOT / "weights/recognize/crnn_govdoc_ft.pt", charset)

# Load govdoc crops
rows = list(csv.DictReader(open(ROOT / "datasets/govdoc_finetune/index.csv", encoding="utf-8")))
print(f"Testing on {len(rows)} govdoc crops\n")

preds_base, preds_ft, refs = [], [], []
for r in rows:
    img_path = Path(r["image_path"])
    if not img_path.exists():
        # try local path
        img_path = ROOT / "datasets/govdoc_finetune/images" / img_path.name
    if not img_path.exists():
        continue
    preds_base.append(infer(base, img_path, charset))
    preds_ft.append(infer(ft,   img_path, charset))
    refs.append(r["text"])

cer_base = cer(preds_base, refs)
cer_ft   = cer(preds_ft,   refs)
diff     = cer_base - cer_ft

print("="*50)
print(f"{'Model':<30}  {'CER':>8}  {'Word Acc':>10}")
print("-"*50)
wacc_base = sum(normalize_for_scoring(p)==normalize_for_scoring(r) for p,r in zip(preds_base,refs))/len(refs)
wacc_ft   = sum(normalize_for_scoring(p)==normalize_for_scoring(r) for p,r in zip(preds_ft,  refs))/len(refs)
print(f"{'Base CRNN (Mozhi)':<30}  {cer_base:>8.4f}  {wacc_base:>10.4f}")
print(f"{'Fine-tuned CRNN (govdoc)':<30}  {cer_ft:>8.4f}  {wacc_ft:>10.4f}")
print()
print(f"Change: {diff:+.4f}  ({diff/cer_base*100:+.1f}% relative improvement)")
print(f"Samples tested: {len(refs)}")

# Save
out = {
    "n_crops": len(refs),
    "base_cer": cer_base, "base_word_acc": wacc_base,
    "finetuned_cer": cer_ft, "finetuned_word_acc": wacc_ft,
    "absolute_improvement": diff,
    "relative_improvement_pct": diff/cer_base*100 if cer_base > 0 else 0,
}
Path(ROOT/"results/crnn_govdoc_eval.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print(f"\nSaved: results/crnn_govdoc_eval.json")

# Show 5 sample comparisons
print("\nSample comparisons:")
print(f"{'GT':<40}  {'Base':<30}  {'FT'}")
print("-"*100)
for i in range(min(5, len(refs))):
    g = refs[i][:38]; b = preds_base[i][:28]; f = preds_ft[i][:28]
    print(f"{g:<40}  {b:<30}  {f}")
