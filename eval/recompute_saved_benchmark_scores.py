"""Re-score saved full-page predictions against an explicit BM mapping manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import corpus_cer, corpus_wer  # noqa: E402
from eval.run_benchmark import load_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark"))
    parser.add_argument("--mapping-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="saved full_layout_off_results.json; source is read only")
    parser.add_argument("--out", type=Path, required=True,
                        help="new output JSON path; existing files are never overwritten")
    parser.add_argument("--exclude-unapproved-mappings", action="store_true")
    args = parser.parse_args()

    items = load_benchmark(
        args.benchmark, args.mapping_manifest,
        exclude_unapproved=args.exclude_unapproved_mappings,
    )
    source_path = args.source if args.source.is_absolute() else ROOT / args.source
    output_path = args.out if args.out.is_absolute() else ROOT / args.out
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_path}")

    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    references = [item["gt_text"] for item in items]
    ids = [item["id"] for item in items]
    results = {}
    for variant, record in source["variants"].items():
        per_page = record.get("pages", {})
        missing = [page_id for page_id in ids if page_id not in per_page or "prediction" not in per_page[page_id]]
        if missing:
            continue
        hypotheses = [per_page[page_id]["prediction"] for page_id in ids]
        results[variant] = {
            "cer": corpus_cer(references, hypotheses),
            "wer": corpus_wer(references, hypotheses),
            "n_pages": len(ids),
            "pages": ids,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_scope": "saved predictions rescored on approved rows from explicit mapping manifest",
        "mapping_manifest": str(args.mapping_manifest),
        "excluded_unapproved_pages": sorted(set(p.stem for p in (args.benchmark / "images").glob("BM_*.png")) - set(ids)),
        "source_predictions": str(source_path),
        "source_predictions_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "normalization": "eval.metrics corpus_cer/corpus_wer with shared normalize_for_scoring",
        "variants": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {n: v for n, v in val.items() if n in {"cer", "wer", "n_pages"}}
                      for k, val in results.items()}, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
