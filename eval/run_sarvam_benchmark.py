"""Opt-in Sarvam Document AI benchmark (preflight by default; no implicit upload)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from akshardrishti.recognize.sarvam_doc_ai import (  # noqa: E402
    MAX_PAGES_PER_JOB,
    PRICE_INR_PER_PAGE,
    batch_items,
    run_batch,
)
from eval.run_benchmark import load_benchmark, score  # noqa: E402

DEFAULT_OUT = Path("results/sarvam_document_ai_bm23_v1")
MAPPING = Path("mappings/benchmark_image_gt_mapping_v1.csv")


def prepare_items(benchmark: Path) -> list[dict]:
    items = load_benchmark(benchmark, MAPPING, exclude_unapproved=True)
    ids = {item["id"] for item in items}
    if len(items) != 23 or "BM_018" in ids:
        raise ValueError(f"mapping v1 must resolve exactly 23 pages excluding BM_018; got {len(items)}")
    return items


def preflight(items: list[dict], out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"output path already exists; refusing to overwrite: {out}")
    oversized = [str(item["image_path"]) for item in items if item["image_path"].stat().st_size > 200 * 1024 * 1024]
    if oversized:
        raise ValueError("input file exceeds documented 200 MB limit: " + ", ".join(oversized))
    batches = batch_items(items)
    return {
        "mode": "preflight_only",
        "pages": len(items),
        "excluded": ["BM_018"],
        "mapping_manifest": str(Path("benchmark") / MAPPING),
        "batches": [len(batch) for batch in batches],
        "language": "mr-IN",
        "output_format": "json",
        "max_pages_per_job": MAX_PAGES_PER_JOB,
        "estimated_cost_inr_excluding_gst": round(len(items) * PRICE_INR_PER_PAGE, 2),
        "sarvamai_installed": __import__("importlib.util", fromlist=["find_spec"]).find_spec("sarvamai") is not None,
        "SARVAM_API_KEY_set": bool(os.getenv("SARVAM_API_KEY")),
        "output_path_available": str(out),
        "documents_submitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preflight", action="store_true", help="check inputs and configuration only (default)")
    parser.add_argument("--submit", action="store_true", help="submit documents to Sarvam; may incur charges")
    parser.add_argument("--confirm-upload-and-billing", action="store_true",
                        help="required with --submit to confirm upload and potential charges")
    args = parser.parse_args()
    if args.preflight and args.submit:
        parser.error("--preflight and --submit are mutually exclusive")
    if args.submit != args.confirm_upload_and_billing:
        parser.error("live execution requires both --submit and --confirm-upload-and-billing")
    items = prepare_items(args.benchmark)
    info = preflight(items, args.out)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not args.submit:
        return 0
    if not info["sarvamai_installed"]:
        print("BLOCKED: install optional SDK with: python -m pip install sarvamai", file=sys.stderr)
        return 2
    if not info["SARVAM_API_KEY_set"]:
        print("BLOCKED: set SARVAM_API_KEY in the environment", file=sys.stderr)
        return 2

    from akshardrishti.recognize.sarvam_doc_ai import sdk_client
    from eval.metrics import corpus_cer, corpus_wer

    args.out.mkdir(parents=True, exist_ok=False)
    raw_root = args.out / "raw"
    raw_root.mkdir()
    client = sdk_client()
    all_predictions: dict[str, str] = {}
    failures: list[dict] = []
    total_seconds = 0.0
    reference_map = []
    for batch_idx, batch in enumerate(batch_items(items), 1):
        batch_dir = raw_root / f"batch_{batch_idx:02d}"
        preds, batch_failures, elapsed = run_batch(client, batch, batch_dir)
        all_predictions.update(preds)
        failures.extend(batch_failures)
        total_seconds += elapsed
        reference_map.extend({"image_id": item["id"], "reference_path": str(item["gt_path"]),
                              "output_page": page} for page, item in enumerate(batch, 1))

    ordered = [all_predictions.get(item["id"], "") for item in items]
    normalized = {item["id"]: pred for item, pred in zip(items, ordered)}
    (args.out / "predictions_sarvam.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    page_metrics = score(items, ordered, total_seconds / len(items))
    result = {
        "system": "sarvam_document_ai", "created_at": datetime.now(timezone.utc).isoformat(),
        "fresh_inference": True, "mapping_manifest": str(Path("benchmark") / MAPPING),
        "pages": len(items), "excluded": ["BM_018"], "reference_mapping": reference_map,
        "corpus_cer": corpus_cer([i["gt_text"] for i in items], ordered),
        "corpus_wer": corpus_wer([i["gt_text"] for i in items], ordered),
        "scoring": page_metrics, "runtime_seconds_total": total_seconds,
        "runtime_seconds_per_page": total_seconds / len(items), "failures": failures,
        "estimated_cost_inr_excluding_gst": round(len(items) * PRICE_INR_PER_PAGE, 2),
        "actual_usage_status": "inspect raw/*/latest_status.json for API usage fields",
    }
    (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "run_manifest.json").write_text(json.dumps({"reference_mapping": reference_map}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
