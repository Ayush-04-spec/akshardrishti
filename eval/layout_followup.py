#!/usr/bin/env python3
"""Layout follow-up: training curve, corrupt labels, document-level leakage.

No GPU usage. No train/val/predict calls.

Usage
-----
    python eval/layout_followup.py
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────── helpers ─────────────────────────────────────

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


# ─────────────────────────────────────────── main ────────────────────────────

def main() -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    out_json = ROOT / "results/layout_followup.json"
    audit: dict = {"git_commit": git_commit_hash(), "date": str(date.today())}

    # SHA-256 of all four weights (read-only)
    WEIGHT_PATHS = {
        "layout":  ROOT / "weights/layout/yolo11l_indicdlp.pt",
        "slim":    ROOT / "weights/recognize/crnn_mozhi_himr.pt",
        "full":    ROOT / "weights/recognize/crnn_mozhi_himr_full.pt",
        "charset": ROOT / "weights/recognize/charset_deva.txt",
    }
    EXPECTED = {
        "layout":  "5A927BAF51EE9EF3DDEA1DFFDAE954C73869A17B0614836F8856AA77B6AD4AED",
        "slim":    "7E338A7BB3CA482250C5A20DD7224AACF1D488CC12D5A33F6F21CA23CA3C92C9",
        "full":    "626E6B86968FA550DD72E3BDBB9E1119F0B7C0195E65A0219B687FDF1151939A",
        "charset": "E6D5EAA89BBC2D62A45E64C6E4CA0E07F69B1D37F52FFA3894D3183E66E973DF",
    }
    audit["sha256"] = {}
    for key, path in WEIGHT_PATHS.items():
        h = sha256_of(path) if path.exists() else "MISSING"
        audit["sha256"][key] = h
        ok = h == EXPECTED[key]
        if not ok:
            print(f"SHA-256 MISMATCH: {key}  got {h}")

    # ════════════════════════════════════════════════════════════════════════
    # PART A — Training curve
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PART A — Training curve")

    CURVE_CSV = ROOT / "results/layout_train_curve.csv"
    rows_raw: list[dict] = []
    if CURVE_CSV.exists():
        rows_raw = list(csv.DictReader(open(CURVE_CSV, encoding="utf-8")))
    else:
        print("  layout_train_curve.csv NOT FOUND")

    n_rows = len(rows_raw)
    print(f"\nA1 — {n_rows} epoch rows in layout_train_curve.csv")

    COLS = ["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
            "metrics/mAP50-95(B)", "val/box_loss", "val/cls_loss", "val/dfl_loss",
            "lr/pg0"]

    def print_rows(subset: list[dict]) -> None:
        hdr = " ".join(f"{c.split('/')[-1][:10]:>10}" for c in COLS)
        print(f"  {'':>6}  {hdr}")
        print("  " + "-" * (6 + 2 + len(hdr)))
        for r in subset:
            vals = " ".join(f"{float(r.get(c, 0)):>10.4f}" for c in COLS)
            print(f"  {'ep'+str(int(float(r['epoch']))):>6}  {vals}")

    if n_rows <= 30:
        print_rows(rows_raw)
        display_rows = rows_raw
    else:
        first5 = rows_raw[:5]
        every5 = [rows_raw[i] for i in range(0, n_rows, 5)]
        last5  = rows_raw[-5:]
        seen_epochs = set()
        display_rows = []
        for r in (first5 + every5 + last5):
            e = r["epoch"]
            if e not in seen_epochs:
                seen_epochs.add(e)
                display_rows.append(r)
        display_rows.sort(key=lambda r: float(r["epoch"]))
        print_rows(display_rows)

    # A2
    print(f"\nA2 — Curve analysis")
    if rows_raw:
        map_col = "metrics/mAP50-95(B)"
        epochs   = [float(r["epoch"]) for r in rows_raw]
        maps     = [float(r[map_col]) for r in rows_raw]
        best_idx = max(range(len(maps)), key=lambda i: maps[i])
        print(f"  Rows: {n_rows} (epoch 1 to {int(epochs[-1])})")
        print(f"  Best mAP50-95 at epoch {int(epochs[best_idx])}: {maps[best_idx]:.4f}")

        # Slope over last 5 epochs
        last5_epochs = epochs[-5:]
        last5_maps   = maps[-5:]
        if len(last5_epochs) >= 2:
            x = np.array(last5_epochs)
            y = np.array(last5_maps)
            slope = float(np.polyfit(x - x.mean(), y, 1)[0])
            print(f"  Slope of mAP50-95 over last 5 epochs (linear fit): {slope:+.4f} per epoch")
            # Largest epoch-to-epoch change
            changes = [maps[i] - maps[i-1] for i in range(1, len(maps))]
            max_change_idx = max(range(len(changes)), key=lambda i: abs(changes[i]))
            print(f"  Largest epoch-to-epoch change: ep{int(epochs[max_change_idx])}→ep{int(epochs[max_change_idx+1])}"
                  f"  Δ={changes[max_change_idx]:+.4f}")
            if slope > 0.002:
                print(f"  STILL RISING at epoch {int(epochs[-1])} (slope={slope:+.4f})")
            elif slope > -0.001:
                print(f"  APPROXIMATELY FLAT at the end (slope={slope:+.4f})")
            else:
                print(f"  DECLINING at the end (slope={slope:+.4f})")
        audit["A2"] = {
            "n_rows": n_rows, "best_epoch": int(epochs[best_idx]),
            "best_map5095": maps[best_idx],
            "last5_slope": slope if len(last5_epochs) >= 2 else None,
        }

    # A3 — Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if rows_raw:
            epochs_  = [float(r["epoch"]) for r in rows_raw]
            map5095  = [float(r["metrics/mAP50-95(B)"]) for r in rows_raw]
            map50    = [float(r["metrics/mAP50(B)"])    for r in rows_raw]
            box_loss = [float(r["train/box_loss"])      for r in rows_raw]
            cls_loss = [float(r["train/cls_loss"])      for r in rows_raw]
            dfl_loss = [float(r["train/dfl_loss"])      for r in rows_raw]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax1.plot(epochs_, map5095, "b-o", label="mAP50-95", markersize=4)
            ax1.plot(epochs_, map50,   "g-s", label="mAP50",    markersize=4)
            ax1.set_ylabel("mAP"); ax1.legend(); ax1.grid(True, alpha=0.3)
            ax1.set_title("YOLO11l Layout Detector — Training History (14 epochs, early stop)")
            ax2.plot(epochs_, box_loss, "r-",  label="box_loss")
            ax2.plot(epochs_, cls_loss, "m-",  label="cls_loss")
            ax2.plot(epochs_, dfl_loss, "c-",  label="dfl_loss")
            ax2.set_xlabel("epoch"); ax2.set_ylabel("loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(ROOT / "results/layout_train_curve.png", dpi=120)
            plt.close(fig)
            print(f"\nA3 — Plot saved to results/layout_train_curve.png")
    except Exception as e:
        print(f"\nA3 — Plot failed: {e}")

    # A4 — lr0 formula
    print(f"\nA4 — lr0 formula analysis")
    try:
        from ultralytics.engine.trainer import BaseTrainer
        src = inspect.getsource(BaseTrainer.build_optimizer)
        # Find the lr_fit line
        for line in src.splitlines():
            if "lr_fit" in line or "lr0 fit" in line or "0.002" in line:
                print(f"  {line.rstrip()}")
        # Evaluate formula
        nc = 11
        lr_fit = round(0.002 * 5 / (4 + nc), 6)
        print(f"\n  Formula: lr_fit = round(0.002 * 5 / (4 + nc), 6)")
        print(f"  With nc={nc}: lr_fit = round(0.002 * 5 / {4+nc}, 6) = {lr_fit}")
        print(f"  Checkpoint lr0 = 0.000667")
        print(f"  Formula result = {lr_fit}")
        match = abs(lr_fit - 0.000667) < 1e-6
        print(f"  lr0=0.000667 follows from formula: {match}")

        # Show other train_args fields
        print(f"\n  Checkpoint train_args (from A2 audit):")
        print(f"    optimizer     : AdamW  (auto selected AdamW for iterations <= 10000)")
        print(f"    lr0           : 0.000667  (= {lr_fit} from formula, nc={nc})")
        print(f"    lrf           : 0.01")
        print(f"    momentum      : 0.9")
        print(f"    warmup_epochs : (default 3.0, not overridden)")
        print(f"    weight_decay  : 0.0  (AdamW sets weight_decay=0.0 for all groups)")
        audit["A4"] = {"lr_fit_formula": lr_fit, "ckpt_lr0": 0.000667,
                       "formula_matches": match, "nc": nc}
    except Exception as e:
        print(f"  A4 error: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # PART B — Corrupt labels
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART B — Corrupt labels")

    LABELS_VAL = ROOT / "datasets/indicdlp_subset/labels/val"
    LABELS_TR  = ROOT / "datasets/indicdlp_subset/labels/train"
    CORRUPT_IMGS = ["np_mr_000104_1.png", "np_mr_000185_1.png"]

    print("\nB1 — Full contents of corrupt label files")
    for img_name in CORRUPT_IMGS:
        stem = Path(img_name).stem
        lf = LABELS_VAL / f"{stem}.txt"
        print(f"\n  {lf.name}:")
        if not lf.exists():
            print("    NOT FOUND")
            continue
        lines = lf.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) == 5:
                vals = [float(p) for p in parts[1:]]
                bad = any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0
            else:
                bad = True  # wrong field count
            flag = "  <-- OFFENDING" if bad else ""
            print(f"    {i:4d}: {line}{flag}")
        print(f"    ({len(lines)} lines total)")

    # B2 — Source annotations
    print("\nB2 — Source annotations from COCO JSON")
    ANN_DIR = Path(r"D:\AksharDrishti_1\indicdlp_annotations_only\indicdlp\annotations")
    # Load split_manifest to find original splits
    manifest_rows = list(csv.DictReader(open(ROOT / "split_manifest.csv", encoding="utf-8")))
    fname_to_orig = {r["filename"]: r["original_split"] for r in manifest_rows}

    def _to_yolo(bbox, img_w, img_h):
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        return cx, cy, nw, nh

    # load val annotations
    val_ann_path = ANN_DIR / "instances_val2017.json"
    train_ann_path = ANN_DIR / "instances_train2017.json"

    for img_name in CORRUPT_IMGS:
        print(f"\n  {img_name}:")
        orig_split = fname_to_orig.get(img_name, "unknown")
        print(f"    original_split = {orig_split}")
        # Which JSON to load?
        ann_path = train_ann_path if orig_split == "train" else val_ann_path
        if not ann_path.exists():
            print(f"    Annotation JSON not found: {ann_path}")
            continue
        with open(ann_path, encoding="utf-8") as f:
            ann_data = json.load(f)
        # Find image entry
        img_entry = next((im for im in ann_data["images"] if im["file_name"] == img_name), None)
        if not img_entry:
            print(f"    Image not found in JSON by file_name")
            # Try stem
            img_entry = next((im for im in ann_data["images"]
                              if Path(im["file_name"]).stem == Path(img_name).stem), None)
        if not img_entry:
            print(f"    Not found in {ann_path.name}")
            continue
        print(f"    Image entry: id={img_entry['id']}  w={img_entry['width']}  h={img_entry['height']}")
        # Get all annotations for this image
        img_anns = [a for a in ann_data["annotations"] if a["image_id"] == img_entry["id"]]
        cat_map = {c["id"]: c["name"] for c in ann_data.get("categories", [])}
        print(f"    {len(img_anns)} annotations:")
        for a in img_anns[:5]:
            cx, cy, nw, nh = _to_yolo(a["bbox"], img_entry["width"], img_entry["height"])
            bad = any(v < 0 or v > 1 for v in [cx, cy, nw, nh]) or nw <= 0 or nh <= 0
            flag = " <-- PRODUCES INVALID YOLO" if bad else ""
            print(f"      ann_id={a['id']}  cat={cat_map.get(a['category_id'],'?')}({a['category_id']})"
                  f"  bbox={[round(v,3) for v in a['bbox']]}  -> yolo=({cx:.4f},{cy:.4f},{nw:.4f},{nh:.4f}){flag}")
        if len(img_anns) > 5:
            print(f"      ... ({len(img_anns)-5} more annotations)")

    # B3 — Full label scan
    print("\nB3 — Full label scan")
    b3_results = {}
    for split_name, ldir in [("train", LABELS_TR), ("val", LABELS_VAL)]:
        n_files = n_total = n_class_oob = n_val_oob = n_wh_bad = n_field_bad = 0
        offenders_class: list[str] = []
        offenders_val:   list[str] = []
        offenders_wh:    list[str] = []
        offenders_field: list[str] = []

        for lf in sorted(ldir.glob("*.txt")):
            n_files += 1
            for i, line in enumerate(lf.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                parts = line.split()
                if len(parts) != 5:
                    n_field_bad += 1
                    if len(offenders_field) < 10:
                        offenders_field.append(f"{lf.name}:{i} fields={len(parts)}: {line[:60]}")
                    continue
                try:
                    cls = int(parts[0])
                    vals = [float(p) for p in parts[1:]]
                except ValueError:
                    n_field_bad += 1
                    if len(offenders_field) < 10:
                        offenders_field.append(f"{lf.name}:{i} parse_error: {line[:60]}")
                    continue
                if cls < 0 or cls > 10:
                    n_class_oob += 1
                    if len(offenders_class) < 10:
                        offenders_class.append(f"{lf.name}:{i} cls={cls}")
                if any(v < 0 or v > 1 for v in vals):
                    n_val_oob += 1
                    if len(offenders_val) < 10:
                        offenders_val.append(f"{lf.name}:{i} vals={[round(v,4) for v in vals]}")
                if vals[2] <= 0 or vals[3] <= 0:
                    n_wh_bad += 1
                    if len(offenders_wh) < 10:
                        offenders_wh.append(f"{lf.name}:{i} w={vals[2]:.4f} h={vals[3]:.4f}")

        print(f"\n  [{split_name}] {n_files} files, {n_total} annotations")
        print(f"    class OOB (not 0-10): {n_class_oob}")
        print(f"    value OOB ([0,1]):    {n_val_oob}")
        print(f"    w or h <= 0:         {n_wh_bad}")
        print(f"    wrong field count:   {n_field_bad}")
        for label, lst in [("class OOB", offenders_class), ("val OOB", offenders_val),
                            ("w/h bad", offenders_wh), ("field bad", offenders_field)]:
            if lst:
                print(f"    First offenders ({label}):")
                for o in lst[:5]:
                    print(f"      {o}")
        b3_results[split_name] = {
            "n_files": n_files, "n_total": n_total,
            "class_oob": n_class_oob, "val_oob": n_val_oob,
            "wh_bad": n_wh_bad, "field_bad": n_field_bad,
        }
    audit["B3"] = b3_results

    # ════════════════════════════════════════════════════════════════════════
    # PART C — Document-level leakage
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART C — Document-level leakage")

    # C1 — Parse filenames
    print("\nC1 — Filename parsing")
    FNAME_RE = re.compile(
        r'^([a-z]+)_([a-z]+)_(\d+)(?:_(\d+))?\.png$', re.IGNORECASE
    )
    manifest_all = list(csv.DictReader(open(ROOT / "split_manifest.csv", encoding="utf-8")))

    parsed: list[dict] = []
    n_with_suffix = 0
    suffix_dist: dict[str, int] = defaultdict(int)
    prefix_dist: dict[str, int] = defaultdict(int)
    for r in manifest_all:
        m = FNAME_RE.match(r["filename"])
        if m:
            prefix, lang, number, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
            has_suffix = suffix is not None
            if has_suffix:
                n_with_suffix += 1
                suffix_dist[suffix] += 1
            prefix_dist[prefix] += 1
            parsed.append({
                "filename": r["filename"], "prefix": prefix, "lang": lang,
                "number": number, "suffix": suffix, "split": r["split"],
            })
        else:
            print(f"  WARNING: unparseable filename: {r['filename']}")

    print(f"  Total images: {len(parsed)}")
    print(f"  With suffix:  {n_with_suffix} ({100*n_with_suffix/len(parsed):.1f}%)")
    print(f"  Suffix distribution: {dict(sorted(suffix_dist.items()))}")
    print(f"  Per prefix counts: {dict(sorted(prefix_dist.items()))}")
    audit["C1"] = {"n_with_suffix": n_with_suffix,
                   "suffix_distribution": dict(suffix_dist),
                   "prefix_distribution": dict(prefix_dist)}

    # C2 — Collapse test
    print(f"\nC2 — Suffix collapse check")
    # Does any (prefix, lang, number) appear with two different suffixes?
    key_to_suffixes: dict[tuple, set] = defaultdict(set)
    for p in parsed:
        key = (p["prefix"], p["lang"], p["number"])
        key_to_suffixes[key].add(p["suffix"])  # None or digit string
    multi = {k: v for k, v in key_to_suffixes.items() if len(v) > 1}
    print(f"  Keys with multiple suffixes: {len(multi)}")
    for k, v in list(multi.items())[:5]:
        print(f"    {k}: suffixes={v}")
    if multi:
        print(f"  YES: suffix removal can collapse two files into one key.")
        print(f"  BUT the earlier leakage check found 0 shared keys — meaning")
        print(f"  no (prefix,lang,number) appears in both train and val after collapse.")
    else:
        print(f"  NO: each (prefix, lang, number) has at most one suffix value.")
        print(f"  Suffix removal cannot make two different files share a key.")
    audit["C2"] = {"keys_with_multiple_suffixes": len(multi),
                   "collapse_possible": len(multi) > 0}

    # C3 — Adjacency check
    print(f"\nC3 — Number adjacency check")

    # Build per (prefix, lang) number lists for train and val
    pref_lang_train: dict[tuple, list[int]] = defaultdict(list)
    pref_lang_val:   dict[tuple, list[int]] = defaultdict(list)
    for p in parsed:
        key = (p["prefix"], p["lang"])
        num = int(p["number"])
        if p["split"] == "train":
            pref_lang_train[key].append(num)
        else:
            pref_lang_val[key].append(num)

    all_keys = sorted(set(pref_lang_train.keys()) | set(pref_lang_val.keys()))

    def nearest_neighbour_fraction(val_nums: list[int], train_nums: list[int], within: int) -> float:
        """Fraction of val numbers whose nearest train neighbour is <= within positions."""
        if not val_nums or not train_nums:
            return 0.0
        train_sorted = sorted(train_nums)
        count = 0
        for v in val_nums:
            # binary search for nearest
            import bisect
            pos = bisect.bisect_left(train_sorted, v)
            candidates = []
            if pos < len(train_sorted):
                candidates.append(train_sorted[pos])
            if pos > 0:
                candidates.append(train_sorted[pos - 1])
            nn_dist = min(abs(c - v) for c in candidates) if candidates else 999999
            if nn_dist <= within:
                count += 1
        return count / len(val_nums)

    def random_split_baseline(all_nums: list[int], n_val: int, within: int,
                              rng: random.Random, n_trials: int = 200) -> float:
        import bisect
        fracs = []
        for _ in range(n_trials):
            shuffled = list(all_nums)
            rng.shuffle(shuffled)
            v_set = shuffled[:n_val]
            t_set = sorted(shuffled[n_val:])
            count = 0
            for v in v_set:
                pos = bisect.bisect_left(t_set, v)
                cands = []
                if pos < len(t_set):
                    cands.append(t_set[pos])
                if pos > 0:
                    cands.append(t_set[pos-1])
                if cands and min(abs(c-v) for c in cands) <= within:
                    count += 1
            fracs.append(count / n_val if n_val else 0.0)
        return sum(fracs) / len(fracs)

    rng42 = random.Random(42)
    adj_results = {}
    print(f"  {'prefix+lang':<18}  {'n_tr':>6}  {'n_val':>5}  "
          f"{'frac1':>7}  {'rand1':>7}  {'frac2':>7}  {'frac5':>7}")
    for key in all_keys:
        tr_nums  = pref_lang_train.get(key, [])
        val_nums = pref_lang_val.get(key, [])
        if not tr_nums or not val_nums:
            continue
        all_nums = tr_nums + val_nums
        f1 = nearest_neighbour_fraction(val_nums, tr_nums, 1)
        f2 = nearest_neighbour_fraction(val_nums, tr_nums, 2)
        f5 = nearest_neighbour_fraction(val_nums, tr_nums, 5)
        r1 = random_split_baseline(all_nums, len(val_nums), 1, rng42)
        print(f"  {str(key):<18}  {len(tr_nums):>6}  {len(val_nums):>5}  "
              f"{f1:>7.3f}  {r1:>7.3f}  {f2:>7.3f}  {f5:>7.3f}")
        adj_results[str(key)] = {"n_tr": len(tr_nums), "n_val": len(val_nums),
                                  "frac_within1": f1, "rand_within1": r1,
                                  "frac_within2": f2, "frac_within5": f5}

    # Overall comparison
    all_val = []
    all_tr_pooled = []
    for key in all_keys:
        all_val.extend([(key, v) for v in pref_lang_val.get(key, [])])
        all_tr_pooled.extend(pref_lang_train.get(key, []))
    total_f1  = sum(v["frac_within1"] * v["n_val"] for v in adj_results.values())
    total_r1  = sum(v["rand_within1"] * v["n_val"] for v in adj_results.values())
    total_val = sum(v["n_val"] for v in adj_results.values())
    mean_f1 = total_f1 / total_val if total_val else 0
    mean_r1 = total_r1 / total_val if total_val else 0
    print(f"\n  Weighted mean frac_within1: actual={mean_f1:.4f}  random={mean_r1:.4f}")
    if abs(mean_f1 - mean_r1) < 0.02:
        print(f"  CONCLUSION: no sign of structured leakage (actual ≈ random baseline)")
    elif mean_f1 > mean_r1:
        print(f"  CONCLUSION: val is closer to train than random by {mean_f1-mean_r1:.4f} "
              f"({(mean_f1-mean_r1)/mean_r1*100:.1f}%) — possible structured split")
    else:
        print(f"  CONCLUSION: val is FURTHER from train than random (intentional spacing?)")
    audit["C3"] = {"adj_by_prefix_lang": adj_results,
                   "mean_frac_within1": mean_f1, "mean_rand_within1": mean_r1}

    # C4 — dHash
    print(f"\nC4 — dHash near-duplicate scan")
    IMGS_TR  = ROOT / "datasets/indicdlp_subset/images/train"
    IMGS_VAL = ROOT / "datasets/indicdlp_subset/images/val"

    def dhash(img_path: Path) -> int:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return -1
        small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        diff  = small[:, 1:] > small[:, :-1]  # shape (8,8)
        bits  = diff.flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h

    def hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    print("  Computing dHashes...")
    tr_hashes: dict[str, int]  = {}
    val_hashes: dict[str, int] = {}

    tr_files  = sorted(IMGS_TR.glob("*.png"))
    val_files = sorted(IMGS_VAL.glob("*.png"))
    total_imgs = len(tr_files) + len(val_files)
    print(f"  {total_imgs} images to hash...")

    for i, f in enumerate(tr_files):
        tr_hashes[f.name] = dhash(f)
        if (i + 1) % 2000 == 0:
            print(f"  ... {i+1}/{len(tr_files)} train")
    for i, f in enumerate(val_files):
        val_hashes[f.name] = dhash(f)
        if (i + 1) % 300 == 0:
            print(f"  ... {i+1}/{len(val_files)} val")

    # Find cross-split pairs with Hamming <= 4
    THRESHOLD = 4
    near_pairs: list[tuple[int, str, str]] = []
    for v_name, v_hash in val_hashes.items():
        if v_hash < 0:
            continue
        for t_name, t_hash in tr_hashes.items():
            if t_hash < 0:
                continue
            d = hamming(v_hash, t_hash)
            if d <= THRESHOLD:
                near_pairs.append((d, v_name, t_name))
    near_pairs.sort()
    print(f"  Cross-split pairs with Hamming <= {THRESHOLD}: {len(near_pairs)}")
    top10 = near_pairs[:10]
    for d, vn, tn in top10:
        print(f"    dist={d}  val={vn}  train={tn}")

    # Save contact sheet
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if top10:
        n_show = min(10, len(top10))
        fig, axes = plt.subplots(n_show, 2, figsize=(8, n_show * 1.8))
        if n_show == 1:
            axes = [axes]
        for row_idx, (d, vn, tn) in enumerate(top10[:n_show]):
            img_val   = cv2.imread(str(IMGS_VAL / vn), cv2.IMREAD_GRAYSCALE)
            img_train = cv2.imread(str(IMGS_TR  / tn), cv2.IMREAD_GRAYSCALE)
            axes[row_idx][0].imshow(img_val,   cmap="gray", aspect="auto")
            axes[row_idx][0].set_title(f"VAL: {vn[:30]}", fontsize=6)
            axes[row_idx][0].axis("off")
            axes[row_idx][1].imshow(img_train, cmap="gray", aspect="auto")
            axes[row_idx][1].set_title(f"TRAIN dist={d}: {tn[:30]}", fontsize=6)
            axes[row_idx][1].axis("off")
        plt.suptitle(f"Near-duplicate pairs (dHash Hamming <= {THRESHOLD})", fontsize=9)
        plt.tight_layout()
        plt.savefig(ROOT / "results/leakage_pairs.png", dpi=100)
        plt.close(fig)
        print(f"  Contact sheet saved to results/leakage_pairs.png")
    else:
        print(f"  No near-duplicate pairs found; no contact sheet.")

    audit["C4"] = {"n_near_pairs": len(near_pairs),
                   "threshold": THRESHOLD,
                   "top10": [(d, vn, tn) for d, vn, tn in top10]}

    # C5 — Summary
    print(f"\nC5 — Summary of leakage checks")
    print(f"""
  Three checks performed:
  (1) Document key (suffix removal): 0 (prefix, lang, number) keys shared between
      train and val. No document-level leakage from this method.
  (2) Number adjacency: actual fraction of val numbers with NN in train within 1
      position = {mean_f1:.4f}; random baseline = {mean_r1:.4f}. Difference = {mean_f1-mean_r1:.4f}.
      {'No sign of structured leakage.' if abs(mean_f1-mean_r1)<0.02 else 'Some adjacency detected.'}
  (3) Image dHash: {len(near_pairs)} cross-split pairs with Hamming <= 4. These are 
      likely near-identical page scans (same physical page photographed twice, 
      or two successive scans of the same document).

  What these checks CANNOT show:
  - Near-duplicate pages with different content but identical template (same
    form, letterhead, or masthead layout with different text filled in).
  - Two pages from the same government order that are numerically far apart.
  - Shared layout structure without visual similarity (same organization, 
    different years).
  - Documents where all pages are far apart numerically but conceptually linked.
""")
    audit["C5"] = {
        "doc_key_shared": 0,
        "adjacency_conclusion": "no sign of structured leakage" if abs(mean_f1-mean_r1)<0.02
                                else "some adjacency detected",
        "dhash_near_pairs": len(near_pairs),
    }

    # Save audit JSON
    out_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nAudit JSON written to: {out_json}")


if __name__ == "__main__":
    main()
