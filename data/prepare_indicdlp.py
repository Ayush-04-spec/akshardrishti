#!/usr/bin/env python3
"""Build a training subset of IndicDLP for AksharDrishti.

IndicDLP is 119,806 images / 91 GB across 12 languages, 12 domains and 42
layout classes. We need a slice of it: Hindi/Marathi/English, government-ish
domains, remapped to our 11 classes, in YOLO format.

Usage
-----
    # 1. ALWAYS DO THIS FIRST -- see the real class and language names
    python data/prepare_indicdlp.py --inspect

    # 2. Reconcile anything the inspect step flags, then export
    python data/prepare_indicdlp.py --out datasets/indicdlp_subset --max-images 15000

Access
------
IndicDLP is gated behind a click-through terms form on Hugging Face. Accept it
at https://huggingface.co/datasets/ai4bharat/indicdlp and run `huggingface-cli
login` before this script will work.

Why filter rather than train on everything: 91 GB does not fit Kaggle's disk,
and a T4 does not have the hours. A focused 15k-image subset covering the
document types we actually target trains in one overnight run.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akshardrishti.config import ClassMap  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_indicdlp")

DEFAULT_LANGUAGES = ["hindi", "marathi", "english"]
DEFAULT_DOMAINS = ["acts_and_rules", "forms", "notices", "newspapers", "question_papers"]


# --------------------------------------------------------------------- helpers


def _norm(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("&", "and")


def normalize_category_name(name: str) -> str:
    """Normalize category names from JSON annotations to match class_map.yaml format.
    
    IndicDLP annotations use hyphens (e.g., 'chapter-title', 'figure-caption')
    but class_map.yaml uses underscores (e.g., 'chapter_title', 'figure_caption').
    This function bridges the gap without modifying the protected config file.
    """
    return name.strip().lower().replace("-", "_")


def load_hf_dataset(repo: str, split: str, streaming: bool = True):
    from datasets import load_dataset

    log.info("loading %s split=%s streaming=%s", repo, split, streaming)
    return load_dataset(repo, split=split, streaming=streaming)


def _get_field(row: dict, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in row and row[key] is not None:
            return str(row[key])
    return None


# --------------------------------------------------------------------- inspect


def inspect(repo: str, split: str, limit: int) -> None:
    """Dump the real schema so class_map.yaml can be reconciled against it.

    Do not skip this. The 42 class names in configs/class_map.yaml were
    reconstructed from the paper; if even one differs, `on_unmapped: fail` will
    stop the export -- which is the intended behaviour, but better discovered
    here than 40 minutes into a run.
    """
    ds = load_hf_dataset(repo, split, streaming=True)

    languages, domains, classes = Counter(), Counter(), Counter()
    keys_seen: set[str] = set()

    for i, row in enumerate(ds):
        if i >= limit:
            break
        keys_seen.update(row.keys())
        lang = _get_field(row, ("language", "lang", "language_name"))
        dom = _get_field(row, ("domain", "category", "doc_type", "document_type"))
        if lang:
            languages[_norm(lang)] += 1
        if dom:
            domains[_norm(dom)] += 1
        for name in _iter_class_names(row):
            classes[_norm(name)] += 1

    print("\n=== ROW KEYS ===")
    print(sorted(keys_seen))
    print(f"\n=== LANGUAGES (first {limit} rows) ===")
    for k, v in languages.most_common():
        print(f"  {k:<24} {v}")
    print(f"\n=== DOMAINS ===")
    for k, v in domains.most_common():
        print(f"  {k:<24} {v}")
    print(f"\n=== CLASSES ({len(classes)} distinct) ===")
    for k, v in classes.most_common():
        print(f"  {k:<24} {v}")

    cm = ClassMap.load()
    unmapped = cm.unmapped_report(list(classes))
    print("\n=== RECONCILIATION ===")
    if unmapped:
        print("Classes NOT in configs/class_map.yaml -- add each to `mapping` or `ignore`:")
        for name in unmapped:
            print(f"  - {name}")
        print("\nSuggested YAML to paste under `mapping:`")
        for name in unmapped:
            print(f"  {name}: text   # <-- CHECK THIS, 'text' is only a guess")
    else:
        print("All observed classes are mapped. Safe to export.")


def _iter_class_names(row: dict):
    """Yield category names from whichever annotation shape this row uses."""
    objects = row.get("objects") or row.get("annotations") or row.get("regions")
    if isinstance(objects, dict):
        for key in ("category", "category_name", "label", "class", "categories"):
            if key in objects and isinstance(objects[key], list):
                yield from (normalize_category_name(str(c)) for c in objects[key])
                return
    elif isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict):
                for key in ("category", "category_name", "label", "class"):
                    if key in obj:
                        yield normalize_category_name(str(obj[key]))
                        break


def _iter_annotations(row: dict):
    """Yield (class_name, [x, y, w, h]) pairs in COCO xywh, absolute pixels."""
    objects = row.get("objects") or row.get("annotations") or row.get("regions")

    if isinstance(objects, dict):
        names_key = next((k for k in ("category", "category_name", "label", "class", "categories")
                          if k in objects and isinstance(objects[k], list)), None)
        boxes_key = next((k for k in ("bbox", "bboxes", "boxes") if k in objects), None)
        if names_key and boxes_key:
            for name, box in zip(objects[names_key], objects[boxes_key]):
                yield normalize_category_name(str(name)), list(box)
    elif isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            name = next((str(obj[k]) for k in ("category", "category_name", "label", "class") if k in obj), None)
            box = next((obj[k] for k in ("bbox", "box", "bounding_box") if k in obj), None)
            if name and box:
                yield normalize_category_name(name), list(box)


def _to_yolo(box_xywh: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """COCO [x, y, w, h] absolute -> YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = (float(v) for v in box_xywh[:4])
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
    nw, nh = w / img_w, h / img_h
    if not (0 <= cx <= 1 and 0 <= cy <= 1) or nw <= 0 or nh <= 0:
        return None
    return cx, cy, min(nw, 1.0), min(nh, 1.0)


# ---------------------------------------------------------------------- export


def export(
    repo: str,
    out_dir: Path,
    languages: list[str],
    domains: list[str],
    max_images: int,
    splits: tuple[str, ...],
    seed: int,
    on_unmapped: str | None,
) -> None:
    cm = ClassMap.load()
    if on_unmapped:
        cm.on_unmapped = on_unmapped

    out_dir.mkdir(parents=True, exist_ok=True)
    lang_filter = {_norm(x) for x in languages} if languages else None
    dom_filter = {_norm(x) for x in domains} if domains else None
    rng = random.Random(seed)

    stats = {"seen": 0, "kept": 0, "no_annotations": 0, "dropped_boxes": 0, "by_class": Counter(),
             "by_language": Counter(), "by_domain": Counter()}

    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        try:
            ds = load_hf_dataset(repo, split, streaming=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load split %r (%s) -- skipping", split, exc)
            continue

        budget = max_images if split == "train" else max(1, max_images // 8)
        kept = 0

        for row in ds:
            if kept >= budget:
                break
            stats["seen"] += 1

            lang = _norm(_get_field(row, ("language", "lang", "language_name")) or "")
            dom = _norm(_get_field(row, ("domain", "category", "doc_type", "document_type")) or "")
            if lang_filter and lang and lang not in lang_filter:
                continue
            if dom_filter and dom and dom not in dom_filter:
                continue

            image = row.get("image")
            if image is None:
                continue
            img_w, img_h = image.size

            lines: list[str] = []
            for name, box in _iter_annotations(row):
                try:
                    resolved = cm.resolve(name)
                except KeyError:
                    log.error(
                        "Unmapped class %r. Run `--inspect` and update configs/class_map.yaml, "
                        "or re-run with --on-unmapped text", name,
                    )
                    raise
                if resolved is None:
                    continue
                target, cls_idx = resolved
                yolo = _to_yolo(box, img_w, img_h)
                if yolo is None:
                    stats["dropped_boxes"] += 1
                    continue
                lines.append(f"{cls_idx} " + " ".join(f"{v:.6f}" for v in yolo))
                stats["by_class"][target] += 1

            if not lines:
                stats["no_annotations"] += 1
                continue

            stem = _stem(row, split, kept, rng)
            image.convert("RGB").save(out_dir / "images" / split / f"{stem}.jpg", quality=92)
            (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

            kept += 1
            stats["kept"] += 1
            if lang:
                stats["by_language"][lang] += 1
            if dom:
                stats["by_domain"][dom] += 1

            if kept % 500 == 0:
                log.info("[%s] exported %d/%d", split, kept, budget)

        log.info("[%s] done: %d images", split, kept)

    _write_data_yaml(out_dir, cm.targets, splits)
    _write_stats(out_dir, stats, languages, domains)

    print(f"\nExported {stats['kept']} images to {out_dir}")
    print(f"  skipped (no usable annotations): {stats['no_annotations']}")
    print(f"  dropped boxes (degenerate):      {stats['dropped_boxes']}")
    print("\nInstances per class:")
    for name in cm.targets:
        count = stats["by_class"].get(name, 0)
        flag = "   <-- LOW, consider merging or dropping" if 0 < count < 50 else ("   <-- ABSENT" if count == 0 else "")
        print(f"  {name:<14} {count:>7}{flag}")


def _stem(row: dict, split: str, index: int, rng: random.Random) -> str:
    raw = _get_field(row, ("image_id", "id", "file_name", "filename"))
    if raw:
        stem = Path(str(raw)).stem
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
        if safe:
            return f"{split}_{safe}"
    return f"{split}_{index:07d}_{rng.randrange(16**6):06x}"


def _write_data_yaml(out_dir: Path, targets: list[str], splits: tuple[str, ...]) -> None:
    lines = [
        "# Auto-generated by data/prepare_indicdlp.py -- do not hand-edit",
        f"path: {out_dir.resolve()}",
        "train: images/train",
    ]
    if "validation" in splits or "val" in splits:
        lines.append("val: images/validation" if (out_dir / "images" / "validation").exists() else "val: images/val")
    if "test" in splits:
        lines.append("test: images/test")
    lines += [f"nc: {len(targets)}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(targets)]
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", out_dir / "data.yaml")


def _write_stats(out_dir: Path, stats: dict, languages: list[str], domains: list[str]) -> None:
    payload = {
        "filters": {"languages": languages, "domains": domains},
        "images_kept": stats["kept"],
        "images_seen": stats["seen"],
        "images_without_annotations": stats["no_annotations"],
        "boxes_dropped": stats["dropped_boxes"],
        "instances_per_class": dict(stats["by_class"]),
        "images_per_language": dict(stats["by_language"]),
        "images_per_domain": dict(stats["by_domain"]),
    }
    (out_dir / "subset_stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s -- put this table in the report's Data Set section", out_dir / "subset_stats.json")


# ------------------------------------------------------------------------ cli


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="ai4bharat/indicdlp", help="HF dataset repo id")
    p.add_argument("--inspect", action="store_true", help="dump real schema/classes and exit")
    p.add_argument("--inspect-limit", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("datasets/indicdlp_subset"))
    p.add_argument("--languages", nargs="*", default=DEFAULT_LANGUAGES)
    p.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS)
    p.add_argument("--max-images", type=int, default=15000, help="cap on TRAIN images; val/test get 1/8th")
    p.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--on-unmapped", choices=["fail", "text", "drop"], default=None)
    args = p.parse_args()

    if args.inspect:
        inspect(args.repo, "train", args.inspect_limit)
        return

    export(
        repo=args.repo, out_dir=args.out,
        languages=args.languages, domains=args.domains,
        max_images=args.max_images, splits=tuple(args.splits),
        seed=args.seed, on_unmapped=args.on_unmapped,
    )


if __name__ == "__main__":
    main()
