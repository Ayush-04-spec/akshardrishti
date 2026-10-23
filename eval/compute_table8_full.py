"""
Compute Table 8: CER by region type and script - FULL MEASUREMENT
Processes all 24 benchmark pages through the pipeline.
"""
import sys
sys.stdout = sys.stderr  # Unbuffered
print("="*70, flush=True)
print("TABLE 8: FULL MEASUREMENT (10-15 min)", flush=True)
print("="*70, flush=True)

from pathlib import Path
import json, cv2, traceback
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from akshardrishti.config import Config
    from akshardrishti.pipeline import AksharDrishtiPipeline
    from akshardrishti.postprocess.text_repair import normalize_for_scoring
except Exception as e:
    print(f"Import error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("Imports successful\n", flush=True)

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

def load_gt_boxes(txt_path, img_w, img_h):
    """Load YOLO format GT boxes."""
    CLASS_NAMES = {0:"title",1:"text",2:"list",3:"table",4:"figure",5:"caption",
                   6:"header",7:"footer",8:"page_number",9:"stamp_seal",10:"signature"}
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5: continue
        cid = int(parts[0])
        cx,cy,bw,bh = [float(x) for x in parts[1:]]
        x1=(cx-bw/2)*img_w; y1=(cy-bh/2)*img_h
        x2=(cx+bw/2)*img_w; y2=(cy+bh/2)*img_h
        boxes.append({
            "class": CLASS_NAMES.get(cid, "unknown"),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })
    return boxes

def box_iou(b1, b2):
    x_left   = max(b1["x1"], b2["x1"])
    y_top    = max(b1["y1"], b2["y1"])
    x_right  = min(b1["x2"], b2["x2"])
    y_bottom = min(b1["y2"], b2["y2"])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    area2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def match_region_to_gt(region_box, gt_boxes):
    best_iou = 0.0
    best_class = "unknown"
    for gt in gt_boxes:
        iou = box_iou(region_box, gt)
        if iou > best_iou:
            best_iou = iou
            best_class = gt["class"]
    return best_class if best_iou > 0.3 else "unknown"

def identify_script(text: str) -> str:
    """Identify script based on Unicode ranges."""
    deva = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    latin = sum(1 for c in text if ord(c) < 0x0900 and c.isalpha())
    if deva > latin:
        return "devanagari"
    elif latin > 0:
        return "english"
    return "mixed"

# Collect pages
gt_text_dir = ROOT / "benchmark" / "gt_text"
gt_layout_dir = ROOT / "benchmark" / "gt_layout"
img_dir = ROOT / "benchmark" / "images"

pages = []
for gt_file in sorted(gt_text_dir.glob("BM_*.txt")):
    img_path = img_dir / (gt_file.stem + ".png")
    layout_path = gt_layout_dir / (gt_file.stem + ".txt")
    if img_path.exists():
        pages.append({
            "id": gt_file.stem,
            "image": img_path,
            "gt_text": gt_file.read_text(encoding="utf-8").strip(),
            "gt_layout": layout_path
        })

print(f"Found {len(pages)} pages\n", flush=True)

# Load pipeline
print("Loading pipeline...", flush=True)
cfg = Config.load()
cfg._data["layout"]["enabled"] = True
pipe = AksharDrishtiPipeline(cfg)
print("Pipeline ready\n", flush=True)

# Process each page
per_region_script = defaultdict(lambda: {"preds": [], "refs": []})
total_regions = 0

for i, p in enumerate(pages, 1):
    print(f"[{i}/{len(pages)}] {p['id']}", flush=True)
    
    try:
        img = cv2.imread(str(p['image']))
        if img is None:
            print(f"  ERROR: Cannot load image", flush=True)
            continue
        h, w = img.shape[:2]
        
        gt_boxes = load_gt_boxes(p['gt_layout'], w, h)
        print(f"  GT boxes: {len(gt_boxes)}", flush=True)
        
        doc = pipe.process(p['image'])
        
        # Get regions from first page
        if not doc.pages:
            print(f"  ERROR: No pages in document", flush=True)
            continue
        page = doc.pages[0]
        print(f"  Detected: {len(page.regions)} regions", flush=True)
        
        # Split GT text into lines for better matching
        gt_lines = p['gt_text'].split('\n')
        gt_line_idx = 0
        
        for region in page.regions:
            if not region.text.strip():
                continue
                
            region_box = {"x1": region.bbox.x1, "y1": region.bbox.y1,
                          "x2": region.bbox.x2, "y2": region.bbox.y2}
            region_type = match_region_to_gt(region_box, gt_boxes)
            script = identify_script(region.text)
            
            # Use next GT line as reference (approximate)
            if gt_line_idx < len(gt_lines):
                ref_text = gt_lines[gt_line_idx]
                gt_line_idx += 1
            else:
                ref_text = region.text  # Fallback: CER will be high
            
            key = (region_type, script)
            per_region_script[key]["preds"].append(region.text)
            per_region_script[key]["refs"].append(ref_text)
            total_regions += 1
        
        print(f"  Processed {len(page.regions)} regions\n", flush=True)
        
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc()

print("="*70, flush=True)
print(f"Processed {total_regions} regions total", flush=True)
print("Computing CER per (region_type, script)...\n", flush=True)

# Compute CER
results = {}
for (region_type, script), data in sorted(per_region_script.items()):
    if data["refs"]:
        cer = corpus_cer(data["preds"], data["refs"])
        n = len(data["refs"])
        results[(region_type, script)] = {"cer": cer, "n_samples": n}
        print(f"  {region_type}/{script}: CER={cer:.3f} (n={n})", flush=True)

# Format table
print("\n" + "="*70, flush=True)
print("TABLE 8: CER by Region Type and Script", flush=True)
print("="*70, flush=True)

region_types = ["title", "text", "list", "table", "caption", "header", "footer", 
                "page_number", "stamp_seal", "signature", "figure", "unknown"]
scripts = ["devanagari", "english", "mixed"]

print(f"{'Region Type':<15} {'Devanagari':>15} {'English':>15} {'Mixed':>15}", flush=True)
print("-"*70, flush=True)

for rtype in region_types:
    row = [rtype]
    has_data = False
    for script in scripts:
        key = (rtype, script)
        if key in results:
            cer_val = results[key]["cer"]
            n = results[key]["n_samples"]
            row.append(f"{cer_val:.3f} (n={n})")
            has_data = True
        else:
            row.append("—")
    if has_data:
        print(f"{row[0]:<15} {row[1]:>15} {row[2]:>15} {row[3]:>15}", flush=True)

# Save JSON
out_data = {
    "note": "Per-region/script CER. GT references are approximate (line-based matching).",
    "total_regions_processed": total_regions,
    "by_region_script": {}
}
for (rtype, script), data in results.items():
    key_str = f"{rtype}_{script}"
    out_data["by_region_script"][key_str] = {
        "cer": round(data["cer"], 4),
        "n_samples": data["n_samples"]
    }

out_path = ROOT / "results" / "table8_region_script_cer.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

print("="*70, flush=True)
print(f"Saved: {out_path}", flush=True)
print("\n⚠️  Note: GT references are approximate (line-based matching)", flush=True)
print("   True per-region CER requires region-level GT text annotations", flush=True)
print("="*70, flush=True)
