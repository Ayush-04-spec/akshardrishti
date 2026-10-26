#!/usr/bin/env python3
"""Run every system over AksharDrishti-Bench and emit the paper's results table.

This script produces Table 6.1 and the ablation table. Run it once, and the
numbers in the report and the paper come from the same JSON.

Benchmark layout expected
-------------------------
    benchmark/
      images/       page_001.jpg ...
      gt_text/      page_001.txt        <-- full ground-truth transcription
      gt_layout/    page_001.json       <-- optional: boxes + reading order

Usage
-----
    # full comparison
    python eval/run_benchmark.py --benchmark benchmark --out results/

    # just our system, with ablations
    python eval/run_benchmark.py --benchmark benchmark --systems akshardrishti --ablations

    # single system (e.g. while the Bhashini key is still pending)
    python eval/run_benchmark.py --systems tesseract_fullpage paddle_vl_zeroshot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akshardrishti.config import Config  # noqa: E402
from akshardrishti.pipeline import AksharDrishtiPipeline  # noqa: E402
from eval.metrics import bootstrap_ci, cer, corpus_cer, corpus_wer, summarize, wer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}


# ------------------------------------------------------------------ loading


def load_benchmark(
    root: Path,
    mapping_manifest: Path | None = None,
    *,
    exclude_unapproved: bool = False,
) -> list[dict]:
    images_dir, gt_dir = root / "images", root / "gt_text"
    if not images_dir.exists():
        raise SystemExit(f"missing {images_dir}")
    if not gt_dir.exists():
        raise SystemExit(f"missing {gt_dir} -- CER needs ground-truth transcriptions")

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    explicit_refs: dict[str, Path] | None = None
    if mapping_manifest is not None:
        explicit_refs = _load_mapping_manifest(
            root, mapping_manifest, image_paths, exclude_unapproved=exclude_unapproved,
        )

    items = []
    for img_path in image_paths:
        if explicit_refs is None:
            gt_path = gt_dir / f"{img_path.stem}.txt"
            if not gt_path.exists():
                log.warning("no ground truth for %s -- skipping", img_path.name)
                continue
        else:
            gt_path = explicit_refs.get(img_path.stem)
            if gt_path is None:
                log.info("excluding %s: mapping is not approved", img_path.name)
                continue
        items.append({
            "id": img_path.stem,
            "image_path": img_path,
            "gt_path": gt_path,
            "gt_text": gt_path.read_text(encoding="utf-8").strip(),
            "gt_layout_path": root / "gt_layout" / f"{img_path.stem}.json",
        })

    if not items:
        raise SystemExit("benchmark is empty -- no image/ground-truth pairs found")
    log.info("benchmark: %d pages", len(items))
    return items


def _load_mapping_manifest(
    root: Path,
    manifest_path: Path,
    image_paths: list[Path],
    *,
    exclude_unapproved: bool,
) -> dict[str, Path]:
    """Validate an explicit image-to-reference map; never infer cross-ID pairs."""
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        raise ValueError(f"mapping manifest does not exist: {manifest_path}")

    required = {
        "image_id", "image_path", "same_stem_gt_path", "proposed_ground_truth_id",
        "proposed_ground_truth_path", "audit_status", "evidence", "reviewer_notes",
        "approved_for_evaluation",
    }
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"mapping manifest is missing required columns: {sorted(required)}")
        rows = list(reader)

    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = row["image_id"].strip()
        if not image_id:
            raise ValueError("mapping manifest contains an empty image_id")
        if image_id in by_id:
            raise ValueError(f"duplicate mapping manifest entry for {image_id}")
        by_id[image_id] = row

    image_by_id = {p.stem: p for p in image_paths}
    missing_rows = sorted(set(image_by_id) - set(by_id))
    extra_rows = sorted(set(by_id) - set(image_by_id))
    if missing_rows or extra_rows:
        raise ValueError(f"mapping manifest coverage mismatch: missing={missing_rows}, extra={extra_rows}")

    accepted: dict[str, Path] = {}
    used_refs: set[Path] = set()
    unresolved: list[str] = []
    for image_id, row in by_id.items():
        expected_image = Path("images") / image_by_id[image_id].name
        if Path(row["image_path"]) != expected_image:
            raise ValueError(f"image path mismatch for {image_id}: {row['image_path']!r}")
        if row["audit_status"] != "confirmed" or row["approved_for_evaluation"].strip().lower() != "true":
            unresolved.append(image_id)
            continue

        ref_id = row["proposed_ground_truth_id"].strip()
        ref_rel = Path(row["proposed_ground_truth_path"])
        if not ref_id or ref_rel.is_absolute() or ref_rel.suffix.lower() != ".txt":
            raise ValueError(f"invalid proposed ground-truth path for {image_id}: {ref_rel}")
        ref_path = root / ref_rel
        if not ref_path.is_file():
            raise ValueError(f"mapped ground truth for {image_id} does not exist: {ref_path}")
        resolved_ref = ref_path.resolve()
        try:
            resolved_ref.relative_to(root.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"mapped ground truth for {image_id} is outside the repository: {ref_rel}") from exc
        if resolved_ref.stem != ref_id:
            raise ValueError(f"ground-truth ID/path mismatch for {image_id}: {ref_id!r} vs {ref_rel}")
        if resolved_ref in used_refs:
            raise ValueError(f"duplicate ground-truth mapping: {ref_rel}")
        used_refs.add(resolved_ref)
        accepted[image_id] = ref_path

    if unresolved and not exclude_unapproved:
        raise ValueError(
            "mapping manifest contains unapproved/unresolved pages: " + ", ".join(sorted(unresolved))
        )
    if unresolved:
        log.warning("excluding %d unapproved/unresolved page(s): %s", len(unresolved), ", ".join(sorted(unresolved)))
    return accepted


# ------------------------------------------------------------------ systems


def run_akshardrishti(items: list[dict], cfg: Config, label: str) -> dict:
    pipeline = AksharDrishtiPipeline(cfg)
    predictions, seconds = [], []
    failures = []

    for i, item in enumerate(items, 1):
        t0 = time.perf_counter()
        try:
            doc = pipeline.process(item["image_path"])
            predictions.append(doc.full_text())
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] failed on %s: %s", label, item["id"], exc)
            log.debug(traceback.format_exc())
            failures.append((item["id"], str(exc)))
            predictions.append("")
        seconds.append(time.perf_counter() - t0)
        if i % 10 == 0:
            log.info("[%s] %d/%d", label, i, len(items))

    if failures:
        raise RuntimeError(f"{label} failed on {len(failures)} benchmark page(s): {failures}")
    return {"predictions": predictions, "seconds_per_page": sum(seconds) / len(seconds)}


def run_external(items: list[dict], backend_name: str, cfg: Config, label: str) -> dict:
    """Whole-page baselines. These do their own layout analysis, which is the
    fair comparison against our pipeline."""
    from akshardrishti.recognize.base import build_recognizer

    backends = cfg.get("recognize.backends", {}) or {}
    recognizer = build_recognizer(backend_name, backends.get(backend_name, {}), cfg.get("project.device", "auto"))
    recognizer.ensure_loaded()

    predictions, seconds = [], []
    failures = []
    for i, item in enumerate(items, 1):
        image = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            failures.append((item["id"], f"could not read image: {item['image_path']}"))
            continue

        t0 = time.perf_counter()
        try:
            if hasattr(recognizer, "recognize_page"):
                result = recognizer.recognize_page(image)
            else:
                result = recognizer(image, "unknown")
            predictions.append(result.text)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] failed on %s: %s", label, item["id"], exc)
            failures.append((item["id"], str(exc)))
            predictions.append("")
        seconds.append(time.perf_counter() - t0)
        if i % 10 == 0:
            log.info("[%s] %d/%d", label, i, len(items))

    recognizer.unload()
    if failures:
        raise RuntimeError(f"{label} failed on {len(failures)} benchmark page(s): {failures}")
    return {"predictions": predictions, "seconds_per_page": sum(seconds) / len(seconds)}


SYSTEM_BUILDERS = {
    "tesseract_fullpage": lambda items, cfg: run_external(items, "tesseract", cfg, "tesseract_fullpage"),
    "paddle_vl_zeroshot": lambda items, cfg: run_external(items, "paddle_vl", cfg, "paddle_vl_zeroshot"),
    "bhashini_api": lambda items, cfg: run_external(items, "bhashini_api", cfg, "bhashini_api"),
    "surya": lambda items, cfg: run_external(items, "surya", cfg, "surya"),
    "akshardrishti": lambda items, cfg: run_akshardrishti(items, cfg, "akshardrishti"),
}


# ---------------------------------------------------------------- ablations

ABLATIONS = {
    "no_layout": {
        "layout.enabled": False,
        "_why": "Does layout-awareness actually help? This is the core claim.",
    },
    "no_sahi": {
        "layout.sahi.enabled": False,
        "_why": "What does sliced inference buy on dense pages, and at what cost?",
    },
    "no_script_routing": {
        "script_id.enabled": False,
        "_why": "Does routing beat sending everything to one recognizer?",
    },
    "no_postprocess": {
        "postprocess.repair_shirorekha": False,
        "postprocess.unicode_normalize": "NFC",
        "_why": "How much do Unicode normalisation and shirorekha repair buy?",
    },
}


# ------------------------------------------------------------------- scoring


def score(items: list[dict], predictions: list[str], seconds_per_page: float) -> dict:
    references = [item["gt_text"] for item in items]
    cer_stats = corpus_cer(references, predictions)
    wer_stats = corpus_wer(references, predictions)
    lo, hi = bootstrap_ci(cer_stats["per_document_cer"])

    return {
        "cer": cer_stats["cer"],
        "cer_ci95": [lo, hi],
        "accuracy_pct": cer_stats["accuracy"] * 100,
        "mean_document_cer": cer_stats["mean_document_cer"],
        "wer": wer_stats["wer"],
        "substitutions": cer_stats.get("substitutions", 0),
        "deletions": cer_stats.get("deletions", 0),
        "insertions": cer_stats.get("insertions", 0),
        "total_reference_chars": cer_stats["total_reference_chars"],
        "n_documents": cer_stats["n_documents"],
        "seconds_per_page": seconds_per_page,
        "per_document_cer": cer_stats["per_document_cer"],
        "per_document": [
            {
                "id": item["id"],
                "reference_path": str(item["gt_path"]),
                "cer": cer(item["gt_text"], prediction),
                "wer": wer(item["gt_text"], prediction),
            }
            for item, prediction in zip(items, predictions)
        ],
    }


def markdown_table(results: dict[str, dict]) -> str:
    lines = [
        "| System | CER ↓ | 95% CI | Accuracy | WER ↓ | s/page ↓ |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in sorted(results.items(), key=lambda kv: kv[1].get("cer", 1.0)):
        if "error" in r:
            lines.append(f"| {name} | — | — | — | — | *{r['error']}* |")
            continue
        lo, hi = r["cer_ci95"]
        lines.append(
            f"| {name} | {r['cer']:.4f} | [{lo:.4f}, {hi:.4f}] | "
            f"{r['accuracy_pct']:.2f}% | {r['wer']:.4f} | {r['seconds_per_page']:.2f} |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------- cli


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", type=Path, default=Path("benchmark"))
    p.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--systems", nargs="*", default=None, help="default: eval.systems from the config")
    p.add_argument("--ablations", action="store_true", help="also run the ablation suite")
    p.add_argument("--limit", type=int, default=None, help="first N pages only (for a smoke test)")
    p.add_argument("--mapping-manifest", type=Path, default=None,
                   help="explicit image-to-ground-truth CSV mapping; defaults to legacy same-stem pairing")
    p.add_argument("--exclude-unapproved-mappings", action="store_true",
                   help="with --mapping-manifest, exclude unresolved/unapproved rows and report them")
    args = p.parse_args()

    cfg = Config.load(args.config)
    items = load_benchmark(
        args.benchmark, args.mapping_manifest,
        exclude_unapproved=args.exclude_unapproved_mappings,
    )
    if args.limit:
        items = items[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    systems = args.systems or cfg.get("eval.systems", ["akshardrishti"])
    results: dict[str, dict] = {}

    for name in systems:
        if name not in SYSTEM_BUILDERS:
            log.error("unknown system %r (known: %s)", name, sorted(SYSTEM_BUILDERS))
            continue
        log.info("=== running %s ===", name)
        try:
            output = SYSTEM_BUILDERS[name](items, cfg)
            results[name] = score(items, output["predictions"], output["seconds_per_page"])
            print("  " + summarize(name, results[name]))
            (args.out / f"predictions_{name}.json").write_text(
                json.dumps(
                    {item["id"]: pred for item, pred in zip(items, output["predictions"])},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            log.error("system %s failed: %s", name, exc)
            log.debug(traceback.format_exc())
            results[name] = {"error": str(exc)}

    # ------------------------------------------------------------ ablations
    ablation_results: dict[str, dict] = {}
    if args.ablations:
        for name, overrides in ABLATIONS.items():
            log.info("=== ablation: %s ===", name)
            ablated = Config(cfg.to_dict())
            for key, value in overrides.items():
                if not key.startswith("_"):
                    ablated.set(key, value)
            try:
                output = run_akshardrishti(items, ablated, f"ablation:{name}")
                ablation_results[name] = score(items, output["predictions"], output["seconds_per_page"])
                ablation_results[name]["why"] = overrides["_why"]
                print("  " + summarize(name, ablation_results[name]))
            except Exception as exc:  # noqa: BLE001
                log.error("ablation %s failed: %s", name, exc)
                ablation_results[name] = {"error": str(exc)}

    # -------------------------------------------------------------- output
    payload = {
        "benchmark": str(args.benchmark),
        "n_pages": len(items),
        "mapping_manifest": str(args.mapping_manifest) if args.mapping_manifest else None,
        "reference_mapping": [
            {"image_id": item["id"], "reference_path": str(item["gt_path"])}
            for item in items
        ],
        "config_hash": cfg.hash(),
        "systems": results,
        "ablations": ablation_results,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = ["# AksharDrishti — Benchmark Results", "",
              f"Benchmark: `{args.benchmark}` · {len(items)} pages · config `{cfg.hash()}`", "",
              "## System comparison", "", markdown_table(results), ""]
    if ablation_results:
        report += ["## Ablations", "", markdown_table(ablation_results), "",
                   "| Ablation | Question it answers |", "|---|---|"]
        report += [f"| {k} | {v['_why']} |" for k, v in ABLATIONS.items()]
    (args.out / "results.md").write_text("\n".join(report), encoding="utf-8")

    print("\n" + markdown_table(results))
    print(f"\nWrote {args.out / 'results.json'} and {args.out / 'results.md'}")
    print("Paste results.md straight into the report -- these are the real Table 6.1 numbers.")


if __name__ == "__main__":
    main()
