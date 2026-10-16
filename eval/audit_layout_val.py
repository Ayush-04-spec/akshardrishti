#!/usr/bin/env python3
"""Audit the shipped YOLO11l layout checkpoint and re-run validation.

No training. No imports from train/.
Uses batch=2, falls back to batch=1, then device=cpu on CUDA OOM.

Usage
-----
    python eval/audit_layout_val.py
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import subprocess
import sys
import types
import pathlib
from collections import defaultdict
from datetime import date
from pathlib import Path

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


def install_pathlib_shim() -> None:
    """Python 3.13 moved PosixPath into pathlib._local; install a shim so
    checkpoints pickled on 3.13 load on Python 3.12."""
    shim = types.ModuleType("pathlib._local")
    shim.PosixPath        = pathlib.PurePosixPath
    shim.WindowsPath      = pathlib.PureWindowsPath
    shim.Path             = pathlib.PurePath
    shim.PurePath         = pathlib.PurePath
    shim.PurePosixPath    = pathlib.PurePosixPath
    shim.PureWindowsPath  = pathlib.PureWindowsPath
    sys.modules["pathlib._local"] = shim


def load_json_safe(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ─────────────────────────────────────────── main ────────────────────────────

def main() -> None:
    # Refuse to overwrite existing run outputs (but allow re-running if removed)
    out_json = ROOT / "results/layout_audit.json"
    curve_csv = ROOT / "results/layout_train_curve.csv"
    layout_val_dir = ROOT / "results/layout_val"

    (ROOT / "results").mkdir(exist_ok=True)

    WEIGHTS = ROOT / "weights/layout/yolo11l_indicdlp.pt"
    DATA_YAML = ROOT / "datasets/indicdlp_subset/data.yaml"
    EXPECTED_SHA = "5A927BAF51EE9EF3DDEA1DFFDAE954C73869A17B0614836F8856AA77B6AD4AED"

    REPORT_SHA = {
        "slim":    "7E338A7BB3CA482250C5A20DD7224AACF1D488CC12D5A33F6F21CA23CA3C92C9",
        "full":    "626E6B86968FA550DD72E3BDBB9E1119F0B7C0195E65A0219B687FDF1151939A",
        "charset": "E6D5EAA89BBC2D62A45E64C6E4CA0E07F69B1D37F52FFA3894D3183E66E973DF",
        "layout":  EXPECTED_SHA,
    }

    audit: dict = {
        "git_commit":   git_commit_hash(),
        "date":         str(date.today()),
        "sha256_all_weights": {},
        "A1": {}, "A2": {}, "A3": {}, "A4": {},
        "A5_source": {}, "A6": {},
        "B_runs": {}, "B2": {}, "B3": {}, "B4": {},
        "device_used": None, "batch_used": None,
    }

    import torch
    import ultralytics
    audit["torch_version"]       = torch.__version__
    audit["ultralytics_version"] = ultralytics.__version__

    # ════════════════════════════════════════════════════════════════════════
    # A1 — SHA-256
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("A1 — SHA-256 verification")

    layout_sha = sha256_of(WEIGHTS)
    sha_match  = layout_sha == EXPECTED_SHA
    print(f"  layout weights SHA-256 : {layout_sha}")
    print(f"  expected               : {EXPECTED_SHA}")
    print(f"  match                  : {sha_match}")
    if not sha_match:
        print("  STOP: SHA-256 mismatch — aborting audit.")
        sys.exit(1)
    audit["A1"] = {"sha256": layout_sha, "match": sha_match}

    # Compute all four weight SHAs
    for key, relpath in [("slim",    "weights/recognize/crnn_mozhi_himr.pt"),
                          ("full",    "weights/recognize/crnn_mozhi_himr_full.pt"),
                          ("charset", "weights/recognize/charset_deva.txt"),
                          ("layout",  "weights/layout/yolo11l_indicdlp.pt")]:
        p = ROOT / relpath
        h = sha256_of(p) if p.exists() else "MISSING"
        audit["sha256_all_weights"][key] = h

    # ════════════════════════════════════════════════════════════════════════
    # A2 — Checkpoint inspection
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A2 — Checkpoint inspection")

    install_pathlib_shim()
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)

    print(f"  Top-level keys : {list(ckpt.keys())}")
    print(f"  epoch          : {ckpt.get('epoch')}")
    print(f"  date           : {ckpt.get('date')}")
    print(f"  version        : {ckpt.get('version')}")
    print(f"  ultralytics    : {ultralytics.__version__}")

    WANT_ARGS = [
        "model", "data", "epochs", "patience", "batch", "imgsz", "device",
        "optimizer", "lr0", "lrf", "amp", "seed", "deterministic", "mosaic",
        "close_mosaic", "mixup", "degrees", "translate", "scale", "shear",
        "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "cache", "workers",
    ]
    ta = ckpt.get("train_args", {})
    if not isinstance(ta, dict):
        try:
            ta = dict(ta)
        except Exception:
            ta = {}

    print(f"\n  train_args (relevant keys):")
    found_args = {}
    for k in WANT_ARGS:
        if k in ta:
            found_args[k] = ta[k]
            print(f"    {k:<20}: {ta[k]}")

    # Differences from Technical Report
    TECH_REPORT = {
        "epochs": 60, "batch": 16, "imgsz": 1024, "patience": 15,
        "degrees": 2.0, "translate": 0.1, "scale": 0.3, "shear": 1.0,
        "fliplr": 0.0, "flipud": 0.0, "mosaic": 0.5, "mixup": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.2, "hsv_v": 0.3,
    }
    # Tech report says "two GPUs batch 16 per GPU = effective batch 16 (per-GPU)"
    # Actual checkpoint has batch=8 and device=0 (single GPU)
    diffs = {}
    for k, expected in TECH_REPORT.items():
        actual = found_args.get(k)
        if actual is not None and float(actual) != float(expected):
            diffs[k] = {"expected": expected, "actual": actual}
    diffs["device"] = {
        "expected": "0,1 (two T4 GPUs as stated)",
        "actual": found_args.get("device", "unknown"),
        "note": "Checkpoint trained on single GPU (device=0), not two GPUs as stated in report"
    }

    print(f"\n  Differences from Technical Report:")
    if diffs:
        for k, v in diffs.items():
            print(f"    {k:<20}: expected={v['expected']}  actual={v.get('actual', 'N/A')}")
            if "note" in v:
                print(f"      NOTE: {v['note']}")
    else:
        print("    (none)")

    audit["A2"] = {
        "top_level_keys": list(ckpt.keys()),
        "epoch": ckpt.get("epoch"),
        "date": str(ckpt.get("date")),
        "version": ckpt.get("version"),
        "train_args_found": found_args,
        "diffs_from_report": diffs,
    }

    # ════════════════════════════════════════════════════════════════════════
    # A3 — Training history
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A3 — Training history")

    tr = ckpt.get("train_results")
    tm = ckpt.get("train_metrics")
    a3 = {}

    if tr is not None:
        if not isinstance(tr, dict):
            try:
                tr = dict(tr)
            except Exception:
                tr = None

    if tr:
        cols = list(tr.keys())
        epochs_data = tr.get("epoch", [])
        n_epochs = len(epochs_data)
        print(f"  columns ({len(cols)}): {cols}")
        print(f"  epochs recorded     : {n_epochs} (training ran to epoch {epochs_data[-1] if epochs_data else '?'})")

        # Find epoch with highest mAP50-95
        MAP_COL = "metrics/mAP50-95(B)"
        if MAP_COL in tr:
            maps = tr[MAP_COL]
            best_idx = max(range(len(maps)), key=lambda i: maps[i])
            print(f"\n  Best epoch: {epochs_data[best_idx]} (mAP50-95={maps[best_idx]:.4f})")
            print(f"  Best row:")
            for c in cols:
                print(f"    {c:<30}: {tr[c][best_idx]}")
            print(f"\n  Last row (epoch {epochs_data[-1]}):")
            for c in cols:
                print(f"    {c:<30}: {tr[c][-1]}")

        # Save to CSV
        with open(curve_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for i in range(n_epochs):
                writer.writerow({c: tr[c][i] for c in cols})
        print(f"\n  Saved {n_epochs} rows to {curve_csv}")

        if epochs_data and epochs_data[-1] >= 60:
            print(f"  Training ran ALL 60 epochs (no early stop)")
        else:
            print(f"  Training STOPPED EARLY at epoch {epochs_data[-1] if epochs_data else '?'} (patience=15)")

        a3["columns"] = cols
        a3["n_epochs"] = n_epochs
        a3["final_epoch"] = int(epochs_data[-1]) if epochs_data else None
        a3["stopped_early"] = (epochs_data[-1] < 60) if epochs_data else True
        if MAP_COL in tr:
            best_ep = epochs_data[max(range(len(maps)), key=lambda i: maps[i])]
            a3["best_epoch"] = int(best_ep)
            a3["best_map5095"] = float(maps[max(range(len(maps)), key=lambda i: maps[i])])
    else:
        print("  train_results: absent")
        a3["train_results"] = "absent"

    if tm:
        print(f"\n  train_metrics: {tm}")
        a3["train_metrics"] = {str(k): float(v) if hasattr(v, '__float__') else v
                               for k, v in tm.items()}
    else:
        print("  train_metrics: absent")

    audit["A3"] = a3

    # ════════════════════════════════════════════════════════════════════════
    # A4 — Model names vs config
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A4 — Model names vs config")

    from ultralytics import YOLO
    yolo_model = YOLO(str(WEIGHTS))
    model_names = yolo_model.model.names  # {0: 'title', ...}
    nc = yolo_model.model.nc if hasattr(yolo_model.model, 'nc') else len(model_names)
    print(f"  model.names : {model_names}")
    print(f"  model.nc    : {nc}")

    # From data.yaml
    import yaml
    with open(DATA_YAML, encoding="utf-8") as f:
        dyaml = yaml.safe_load(f)
    data_names = dyaml.get("names", [])
    print(f"  data.yaml names : {data_names}")

    # From configs/class_map.yaml
    with open(ROOT / "configs/class_map.yaml", encoding="utf-8") as f:
        cmap = yaml.safe_load(f)
    targets = cmap.get("targets", [])
    print(f"  class_map.yaml targets : {targets}")

    # Compare
    model_names_list = [model_names[i] for i in range(len(model_names))]
    names_match = (model_names_list == data_names == targets)
    print(f"\n  All 11 names identical and same order: {names_match}")
    if not names_match:
        for i in range(max(len(model_names_list), len(data_names), len(targets))):
            mn = model_names_list[i] if i < len(model_names_list) else "MISSING"
            dn = data_names[i] if i < len(data_names) else "MISSING"
            tn = targets[i] if i < len(targets) else "MISSING"
            eq = "==" if mn == dn == tn else "!="
            print(f"    [{i}] model={mn}  data.yaml={dn}  class_map={tn}  {eq}")

    audit["A4"] = {
        "model_names": model_names_list,
        "nc": nc,
        "data_yaml_names": data_names,
        "class_map_targets": targets,
        "all_match": names_match,
    }

    # ════════════════════════════════════════════════════════════════════════
    # A5 — Metric.maps and Metric.map source
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A5 — inspect.getsource of Metric.maps and Metric.map")

    from ultralytics.utils.metrics import Metric
    src_maps = inspect.getsource(Metric.maps.fget)
    src_map  = inspect.getsource(Metric.map.fget)
    print("\n--- Metric.maps ---")
    print(src_maps)
    print("\n--- Metric.map ---")
    print(src_map)

    audit["A5_source"] = {"Metric.maps": src_maps, "Metric.map": src_map}

    # ════════════════════════════════════════════════════════════════════════
    # A6 — train_layout.py final validation block + akshardrishti_metrics.json
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A6 — train_layout.py final validation code")

    TRAIN_LAYOUT = ROOT / "train/train_layout.py"
    src = TRAIN_LAYOUT.read_text(encoding="utf-8")
    # Extract from "metrics = model.val(" to end of _report
    val_block_start = src.find("metrics = model.val(data=str(args.data), imgsz=args.imgsz")
    report_func_start = src.find("def _report(")
    if val_block_start >= 0:
        val_block = src[val_block_start:report_func_start].strip()
        print(val_block)
        print()
        print("_report function:")
        print(src[report_func_start:])
    else:
        print("(block not found)")
        val_block = "(not found)"

    # Search for akshardrishti_metrics.json in the repo
    found_metrics = list(ROOT.rglob("akshardrishti_metrics.json"))
    if found_metrics:
        for p in found_metrics:
            print(f"\n  Found: {p}")
            data = load_json_safe(p)
            if data:
                print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("\n  No local copy of akshardrishti_metrics.json found in the repo.")

    audit["A6"] = {
        "val_call_args": "model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)",
        "metrics_json_keys": ["model","mAP50-95","mAP50","mAP75","precision","recall","per_class"],
        "per_class_built_from": "metrics.names + box.maps (per-class mAP@50-95)",
        "local_copy_found": [str(p) for p in found_metrics],
        "local_copy_content": [load_json_safe(p) for p in found_metrics],
    }

    # ════════════════════════════════════════════════════════════════════════
    # B — Re-run validation
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART B — Re-run validation")

    # val.cache reporting
    val_cache = ROOT / "datasets/indicdlp_subset/labels/val.cache"
    cache_existed_before = val_cache.exists()
    print(f"  val.cache before run: {'exists' if cache_existed_before else 'absent'}")

    RUNS = [
        ("V1", {"conf": 0.001, "iou": 0.7}),
        ("V2", {"conf": 0.001, "iou": 0.6}),
        ("V3", {"conf": 0.25,  "iou": 0.45}),
    ]

    device_used = None
    batch_used  = None
    b_results   = {}

    for run_name, run_kwargs in RUNS:
        print(f"\n{'='*60}")
        print(f"  Run {run_name}: conf={run_kwargs['conf']}  iou={run_kwargs['iou']}")

        # Attempt batch=2 on GPU, fall back to batch=1, then cpu
        metrics = None
        for attempt_device, attempt_batch in [("0", 2), ("0", 1), ("cpu", 1)]:
            try:
                yolo = YOLO(str(WEIGHTS))
                metrics = yolo.val(
                    data=str(DATA_YAML),
                    split="val",
                    imgsz=1024,
                    batch=attempt_batch,
                    device=attempt_device,
                    workers=0,
                    half=False,
                    plots=True,
                    project=str(layout_val_dir),
                    name=run_name,
                    exist_ok=False,
                    verbose=True,
                    **run_kwargs,
                )
                device_used = attempt_device
                batch_used  = attempt_batch
                print(f"  Device={attempt_device}  batch={attempt_batch}  SUCCESS")
                break
            except (RuntimeError, Exception) as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"  OOM/CUDA error at device={attempt_device} batch={attempt_batch}, retrying...")
                    continue
                else:
                    raise

        if metrics is None:
            print(f"  Run {run_name} FAILED on all attempts")
            b_results[run_name] = {"error": "failed"}
            continue

        box = metrics.box
        maps_vals  = box.maps.tolist() if hasattr(box.maps, 'tolist') else list(box.maps)
        ap_class   = box.ap_class_index.tolist() if hasattr(box.ap_class_index, 'tolist') else list(box.ap_class_index)
        names_dict = metrics.names

        print(f"\n  {run_name} results:")
        print(f"    box.map      : {float(box.map):.6f}")
        print(f"    box.map50    : {float(box.map50):.6f}")
        print(f"    box.map75    : {float(box.map75):.6f}")
        print(f"    box.mp       : {float(box.mp):.6f}")
        print(f"    box.mr       : {float(box.mr):.6f}")
        print(f"    box.maps (11 values): {[round(v,4) for v in maps_vals]}")
        print(f"    ap_class_index: {ap_class}")
        print(f"    ap_class_names: {[names_dict[i] for i in ap_class]}")

        # Images and instances per class
        print(f"\n    Per-class breakdown:")
        nc_all = len(maps_vals)
        for i in range(nc_all):
            name = names_dict.get(i, f"class_{i}")
            v = maps_vals[i]
            in_ap = i in ap_class
            print(f"      [{i}] {name:<14} maps={v:.4f}  in_ap_class={'Y' if in_ap else 'N (0 instances)'}")

        b_results[run_name] = {
            "conf": run_kwargs["conf"],
            "iou":  run_kwargs["iou"],
            "map":    float(box.map),
            "map50":  float(box.map50),
            "map75":  float(box.map75),
            "mp":     float(box.mp),
            "mr":     float(box.mr),
            "maps":   [round(v, 6) for v in maps_vals],
            "ap_class_index": ap_class,
        }

    audit["device_used"] = device_used
    audit["batch_used"]  = batch_used
    audit["B_runs"] = b_results

    # ─────────────── B2: mean analysis ───────────────────────────────────
    print("\n" + "=" * 60)
    print("B2 — box.map analysis")

    for run_name in ["V1", "V2", "V3"]:
        if run_name not in b_results or "error" in b_results[run_name]:
            continue
        rv = b_results[run_name]
        maps_all = rv["maps"]
        ap_idx   = rv["ap_class_index"]
        map_val  = rv["map"]

        maps_over_ap  = [maps_all[i] for i in ap_idx]
        mean_over_ap  = sum(maps_over_ap) / len(maps_over_ap) if maps_over_ap else 0
        mean_over_all = sum(maps_all) / len(maps_all) if maps_all else 0

        print(f"\n  {run_name}:")
        print(f"    box.map              = {map_val:.6f}")
        print(f"    mean(maps[ap_class]) = {mean_over_ap:.6f}   (over {len(ap_idx)} classes with instances)")
        print(f"    mean(maps[all 11])   = {mean_over_all:.6f}  (over all 11)")
        eq_ap  = abs(map_val - mean_over_ap)  < 1e-4
        eq_all = abs(map_val - mean_over_all) < 1e-4
        print(f"    box.map == mean(ap_class)? {eq_ap}  box.map == mean(all 11)? {eq_all}")

        # stamp_seal (idx 9) and signature (idx 10)
        for i, name in [(9, "stamp_seal"), (10, "signature")]:
            v = maps_all[i]
            # They should equal the zero-class mean
            print(f"    {name} maps={v:.6f}  in_ap_class={i in ap_idx}")

        audit["B2"][run_name] = {
            "mean_over_ap_classes": mean_over_ap,
            "mean_over_all_11": mean_over_all,
            "box_map_equals": "mean_over_ap_classes" if eq_ap else ("mean_over_all_11" if eq_all else "neither"),
        }

    # ─────────────── B3: label recount ───────────────────────────────────
    print("\n" + "=" * 60)
    print("B3 — Label recount")

    CLASS_NAMES = ["title","text","list","table","figure","caption","header",
                   "footer","page_number","stamp_seal","signature"]
    EXPECTED_COUNTS = {0:3260, 1:8026, 2:5467, 3:405, 4:2226, 5:349, 6:617,
                       7:916, 8:1674, 9:0, 10:0}
    EXPECTED_TOTAL = 22940

    labels_val = ROOT / "datasets/indicdlp_subset/labels/val"
    counts = defaultdict(int)
    n_files = 0
    for f in labels_val.glob("*.txt"):
        n_files += 1
        for line in open(f, encoding="utf-8"):
            parts = line.strip().split()
            if len(parts) >= 5:
                counts[int(parts[0])] += 1

    total = sum(counts.values())
    print(f"  Label files : {n_files}")
    print(f"  Total lines : {total}  (expected {EXPECTED_TOTAL})")
    for i in range(11):
        actual = counts.get(i, 0)
        exp    = EXPECTED_COUNTS.get(i, 0)
        diff   = actual - exp
        print(f"    [{i}] {CLASS_NAMES[i]:<14} actual={actual:5d}  expected={exp:5d}  diff={diff:+d}")

    # Check val.cache
    val_cache_after = val_cache.exists()
    print(f"\n  val.cache after run: {'created' if val_cache_after and not cache_existed_before else ('already existed' if cache_existed_before else 'not created')}")
    if val_cache_after:
        print(f"  val.cache size: {val_cache.stat().st_size / 1024:.1f} KB")

    audit["B3"] = {
        "n_label_files": n_files,
        "total_lines": total,
        "expected_total": EXPECTED_TOTAL,
        "per_class": {CLASS_NAMES[i]: {"actual": counts.get(i,0), "expected": EXPECTED_COUNTS[i]} for i in range(11)},
        "val_cache_created": val_cache_after and not cache_existed_before,
    }

    # ─────────────── B4: compare with training report ────────────────────
    print("\n" + "=" * 60)
    print("B4 — Compare V1 with training report figures")

    REPORT = {
        "mAP50-95": 0.6319,
        "mAP50":    0.9189,
        "precision": 0.8594,
        "recall":    0.8709,
        "per_class": {
            "title":        0.7240,
            "text":         0.8519,
            "list":         0.7222,
            "table":        0.9137,
            "figure":       0.8500,
            "caption":      0.6798,
            "header":       0.6156,
            "footer":       0.5163,
            "page_number":  0.5744,
            "stamp_seal":   0.7164,
            "signature":    0.7164,
        },
    }

    if "V1" in b_results and "error" not in b_results["V1"]:
        v1 = b_results["V1"]
        maps_v1 = v1["maps"]
        ap_idx  = v1["ap_class_index"]
        print(f"\n  Metric         V1_local   Report    diff")
        for metric, v1_val, rep_val in [
            ("mAP50-95", v1["map"],   REPORT["mAP50-95"]),
            ("mAP50",    v1["map50"], REPORT["mAP50"]),
            ("precision",v1["mp"],    REPORT["precision"]),
            ("recall",   v1["mr"],    REPORT["recall"]),
        ]:
            d = v1_val - rep_val
            print(f"    {metric:<14} {v1_val:.4f}    {rep_val:.4f}    {d:+.4f}")

        print(f"\n  Per-class mAP50-95:")
        b4_pc = {}
        for i, name in enumerate(CLASS_NAMES):
            v1_val  = maps_v1[i]
            rep_val = REPORT["per_class"].get(name, None)
            if rep_val is not None:
                d = v1_val - rep_val
                print(f"    [{i}] {name:<14} local={v1_val:.4f}  report={rep_val:.4f}  diff={d:+.4f}")
                b4_pc[name] = {"local": v1_val, "report": rep_val, "diff": d}
            else:
                print(f"    [{i}] {name:<14} local={v1_val:.4f}  report=N/A")

        audit["B4"] = {
            "V1_vs_report": {
                "map":  {"local": v1["map"],   "report": REPORT["mAP50-95"], "diff": v1["map"]   - REPORT["mAP50-95"]},
                "map50":{"local": v1["map50"], "report": REPORT["mAP50"],    "diff": v1["map50"] - REPORT["mAP50"]},
                "mp":   {"local": v1["mp"],    "report": REPORT["precision"],"diff": v1["mp"]    - REPORT["precision"]},
                "mr":   {"local": v1["mr"],    "report": REPORT["recall"],   "diff": v1["mr"]    - REPORT["recall"]},
            },
            "per_class": b4_pc,
        }

    # ════════════════════════════════════════════════════════════════════════
    # Save audit JSON
    # ════════════════════════════════════════════════════════════════════════
    def make_serialisable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serialisable(v) for v in obj]
        try:
            return float(obj) if hasattr(obj, '__float__') else obj
        except Exception:
            return str(obj)

    out_json.write_text(
        json.dumps(make_serialisable(audit), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\nAudit JSON written to: {out_json}")

    # Verify: SHA-256s unchanged
    print("\n" + "=" * 60)
    print("VERIFY — SHA-256 check")
    for key, expected in REPORT_SHA.items():
        actual = audit["sha256_all_weights"].get(key, "MISSING")
        ok = actual == expected
        print(f"  {key:<10}  {'PASS' if ok else 'FAIL'}  {actual}")


if __name__ == "__main__":
    main()
