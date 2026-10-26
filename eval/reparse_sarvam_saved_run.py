"""Offline parser repair for a saved Sarvam Document AI benchmark run."""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from akshardrishti.recognize.sarvam_doc_ai import (  # noqa: E402
    batch_items,
    parse_saved_batch_response,
    repair_verified_mojibake,
)
from eval.run_benchmark import load_benchmark, score  # noqa: E402

EXPECTED_PAGES = 23
EXCLUDED_PAGE = "BM_018"
MAPPING = Path("mappings/benchmark_image_gt_mapping_v1.csv")
DEFAULT_SOURCE = Path("results/sarvam_document_ai_bm23_v2")
DEFAULT_OUT = Path("results/sarvam_document_ai_bm23_v2_parserfix")


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _verify_run_manifest(rows: list[dict], batches: list[list[dict]]) -> list[list[dict]]:
    if len(rows) != EXPECTED_PAGES:
        raise ValueError(f"saved run manifest has {len(rows)} entries; expected {EXPECTED_PAGES}")
    ids = [str(row.get("image_id", "")) for row in rows]
    if len(set(ids)) != EXPECTED_PAGES or EXCLUDED_PAGE in ids:
        raise ValueError("saved run manifest has duplicate IDs or includes BM_018")

    grouped: list[list[dict]] = []
    cursor = 0
    for batch in batches:
        group = rows[cursor:cursor + len(batch)]
        if len(group) != len(batch):
            raise ValueError("saved run manifest does not cover every input batch")
        for local_page, (row, item) in enumerate(zip(group, batch), 1):
            if row.get("image_id") != item["id"]:
                raise ValueError(f"run manifest image order mismatch at {item['id']}")
            if not _same_path(row.get("reference_path", ""), item["gt_path"]):
                raise ValueError(f"run manifest reference mismatch for {item['id']}")
            try:
                output_page = int(row.get("output_page"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid batch-local output_page for {item['id']}") from exc
            if output_page != local_page:
                raise ValueError(f"wrong batch-local page number for {item['id']}: {output_page}")
        grouped.append(group)
        cursor += len(batch)
    if cursor != len(rows):
        raise ValueError("saved run manifest has extra entries beyond the input batches")
    return grouped


def _verify_input_zip(zip_path: Path, batch: list[dict]) -> None:
    if not zip_path.is_file():
        raise ValueError(f"saved input archive is missing: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        names = [Path(name).stem for name in archive.namelist()]
    expected = [str(item["id"]) for item in batch]
    if names != expected:
        raise ValueError(f"saved ZIP input order mismatch: expected {expected}, got {names}")


def reparse_saved_run(
    source: Path = DEFAULT_SOURCE,
    out: Path = DEFAULT_OUT,
    benchmark: Path = Path("benchmark"),
    refresh_existing_parserfix: bool = False,
) -> dict:
    if out.exists():
        marker_path = out / "results.json"
        if not refresh_existing_parserfix or not marker_path.is_file():
            raise FileExistsError(f"output path already exists; refusing to overwrite: {out}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (marker.get("offline_saved_response_rescore") is not True
                or not _same_path(marker.get("source_run", ""), source)):
            raise FileExistsError(f"existing output is not this source run's parserfix; refusing to overwrite: {out}")

    items = load_benchmark(benchmark, MAPPING, exclude_unapproved=True)
    item_ids = [item["id"] for item in items]
    if len(items) != EXPECTED_PAGES or EXCLUDED_PAGE in item_ids:
        raise ValueError(f"approved mapping must contain 23 pages excluding {EXCLUDED_PAGE}")
    if len(set(item_ids)) != EXPECTED_PAGES:
        raise ValueError("approved mapping contains duplicate image IDs")

    source_manifest_path = source / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    mapping_rows = source_manifest.get("reference_mapping")
    batches = batch_items(items)
    batch_rows = _verify_run_manifest(mapping_rows, batches)

    started = time.perf_counter()
    predictions: dict[str, str] = {}
    failures: list[dict] = []
    audit_mapping: list[dict] = []
    batch_summaries: list[dict] = []
    encoding_repair_count = 0
    for batch_index, (batch, manifest_group) in enumerate(zip(batches, batch_rows), 1):
        batch_name = f"batch_{batch_index:02d}"
        raw_dir = source / "raw" / batch_name
        _verify_input_zip(raw_dir / "input.zip", batch)
        response_path = raw_dir / "results_response.json"
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if isinstance(response, dict):
            for document in response.get("documents", []) or []:
                for page in (document.get("pages", []) or []) if isinstance(document, dict) else []:
                    for block in (page.get("blocks", []) or []) if isinstance(page, dict) else []:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            encoding_repair_count += repair_verified_mojibake(block["text"]) != block["text"]
        try:
            batch_predictions, batch_failures = parse_saved_batch_response(response, manifest_group)
        except (TypeError, ValueError, KeyError) as exc:
            batch_predictions = {str(item["id"]): "" for item in batch}
            batch_failures = [
                {"id": str(item["id"]), "batch": batch_index, "reason": f"batch parse error: {exc}"}
                for item in batch
            ]
        overlap = set(predictions) & set(batch_predictions)
        if overlap:
            raise ValueError(f"pages assigned more than once across batches: {sorted(overlap)}")
        if set(batch_predictions) != {item["id"] for item in batch}:
            raise ValueError(f"batch {batch_index} did not map each expected image exactly once")
        predictions.update(batch_predictions)
        failures.extend({**failure, "batch": batch_index} for failure in batch_failures)
        for row in manifest_group:
            audit_mapping.append({
                "image_id": row["image_id"],
                "reference_path": row["reference_path"],
                "batch": batch_index,
                "batch_page_num": int(row["output_page"]),
                "source_response": str(response_path),
            })
        response_documents = response.get("documents", []) if isinstance(response, dict) else []
        batch_summaries.append({
            "batch": batch_index,
            "image_ids": [item["id"] for item in batch],
            "pages_in_response": sum(len(doc.get("pages") or []) for doc in response_documents if isinstance(doc, dict)),
            "pages_mapped": len(batch_predictions),
            "pages_with_failures": len(batch_failures),
        })

    if list(predictions) != item_ids:
        # Reorder deterministically and reject missing/extra keys.
        if set(predictions) != set(item_ids) or len(predictions) != EXPECTED_PAGES:
            raise ValueError("final parsed predictions do not cover the approved 23-page set exactly once")
    ordered_predictions = [predictions[image_id] for image_id in item_ids]
    if len(predictions) != EXPECTED_PAGES or set(predictions) != set(item_ids):
        raise ValueError("final parsed predictions do not cover the approved 23-page set exactly once")

    elapsed = time.perf_counter() - started
    score_payload = score(items, ordered_predictions, seconds_per_page=0.0)
    # Reuse the canonical benchmark scorer's corpus metrics; avoid a second
    # expensive edit-distance pass and keep output exactly comparable.
    cer_payload = {
        "cer": score_payload["cer"],
        "accuracy": score_payload["accuracy_pct"] / 100,
        "mean_document_cer": score_payload["mean_document_cer"],
        "n_documents": score_payload["n_documents"],
        "per_document_cer": score_payload["per_document_cer"],
    }
    wer_payload = {"wer": score_payload["wer"]}
    output = {
        "system": "sarvam_document_ai_saved_response_parserfix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fresh_inference": False,
        "offline_saved_response_rescore": True,
        "source_run": str(source),
        "mapping_manifest": str(benchmark / MAPPING),
        "pages": EXPECTED_PAGES,
        "excluded": [EXCLUDED_PAGE],
        "pages_mapped_exactly_once": len(audit_mapping) == EXPECTED_PAGES,
        "corpus_cer": cer_payload,
        "corpus_wer": wer_payload,
        "scoring": score_payload,
        "offline_parse_runtime_seconds": elapsed,
        "batch_summaries": batch_summaries,
        "encoding_repair_count": encoding_repair_count,
        "failures": failures,
        "failed_page_count": len({failure["id"] for failure in failures}),
        "comparison_reference": {"tesseract_cer": 0.347054, "tesseract_wer": 0.684085},
    }

    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
    (out / "predictions_sarvam.json").write_text(
        json.dumps(dict(zip(item_ids, ordered_predictions)), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "run_manifest.json").write_text(
        json.dumps({"reference_mapping": audit_mapping}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "PASS" if not failures else "PARTIAL"
    report = [
        "# Sarvam saved-response parser repair",
        "",
        f"Status: **{status}**",
        "",
        f"Source: `{source}` (saved API responses only; no API request or upload)",
        f"Pages parsed: **{len(predictions)}/{EXPECTED_PAGES}**; BM_018 excluded.",
        f"Pages mapped exactly once: **{len(audit_mapping) == EXPECTED_PAGES}**.",
        f"Corpus CER: **{cer_payload['cer']:.6f}**; corpus WER: **{wer_payload['wer']:.6f}**.",
        f"Tesseract reference (same approved mapping/scorer): CER **0.347054**, WER **0.684085**.",
        f"Mojibake repair applied to saved responses: **{'yes' if encoding_repair_count else 'no'}** ({encoding_repair_count} block(s)). Unicode inspection found valid Devanagari code points in the raw strings; the apparent mojibake came from terminal display encoding. A guarded repair is tested for genuinely mojibaked input.",
        "Table HTML blocks were converted to text, with tabs between cells and newlines between rows, so markup tags are not scored as OCR characters. Predictions were extracted from saved responses; these are not fresh inference results.",
        f"Parse failures: **{len(failures)}**; offline parse time: **{elapsed:.3f}s**.",
        "",
        "## Batch mapping",
        "",
        "Page numbers restart within each ZIP. Each response page's `page_num` was matched to the corresponding batch-local `output_page` in the saved `run_manifest.json`; batch image IDs and reference paths were checked against mapping v1 and each saved ZIP's entry order.",
        "",
        "## Batch results",
        "",
        "| Batch | Pages mapped | Parse failures | Image IDs |",
        "|---:|---:|---:|---|",
    ]
    report.extend(
        f"| {row['batch']} | {row['pages_mapped']} | {row['pages_with_failures']} | {', '.join(row['image_ids'])} |"
        for row in batch_summaries
    )
    if failures:
        report.extend(["", "## Failures", ""])
        report.extend(f"- {failure['id']} (batch {failure['batch']}): {failure['reason']}" for failure in failures)
    report.append("")
    (out / "PARSER_FIX_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark"))
    parser.add_argument(
        "--refresh-existing-parserfix", action="store_true",
        help="refresh only an existing offline parserfix output for the same source run",
    )
    args = parser.parse_args()
    result = reparse_saved_run(
        args.source, args.out, args.benchmark, args.refresh_existing_parserfix
    )
    print(json.dumps({
        "output": str(args.out), "pages": result["pages"],
        "cer": result["corpus_cer"]["cer"], "wer": result["corpus_wer"]["wer"],
        "failures": result["failed_page_count"], "offline": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
