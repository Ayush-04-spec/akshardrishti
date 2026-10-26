# BM benchmark image-to-reference pairing audit

Audit date: 2026-10-26  
Repository: `aksharDrishti`  
Scope: all 24 `benchmark/images/BM_*.png` pages, all 24 text references, benchmark metadata, full-page evaluation loaders, and saved results. No source benchmark input or historical result was changed.

## Findings

Filename equality was not a valid pairing rule for this benchmark. Visual comparison of the page images against the complete corresponding transcription content confirmed 23 image/reference mappings. `BM_018.png` remains unresolved and must not be scored. The earlier crop-alignment report only checked five pages; its three mismatch findings were correct, but it did not cover all pairings. This audit also finds cross-ID mappings for BM_003, BM_004, BM_006 through BM_011, and BM_017.

The audit manifest is [benchmark_image_gt_mapping_v1.csv](../benchmark/mappings/benchmark_image_gt_mapping_v1.csv). It contains one row per image, same-stem reference, proposed mapped reference, evidence, reviewer note, status, and evaluation approval. `confirmed` means the visible page content was checked against the transcription; it is not a claim that the original files were changed or that a separate reviewer signed off.

| Image | Same-stem GT | Proposed GT | Status | Finding |
|---|---|---|---|---|
| BM_001.png | BM_001.txt | BM_001.txt | confirmed | Matching school education order; title, date, subject, body and signatory agree. |
| BM_002.png | BM_002.txt | BM_002.txt | confirmed | Matching school education order with table; this is a distinct version from BM_001. |
| BM_003.png | BM_003.txt | BM_006.txt | confirmed | Co-operation/marketing digitization order; same-stem BM_003 is Revenue and Forest. |
| BM_004.png | BM_004.txt | BM_003.txt | confirmed | Revenue and Forest certificates circular; same-stem BM_004 is Public Health. |
| BM_005.png | BM_005.txt | BM_005.txt | confirmed | Nashik Public Works e-tender notice; tender number and date agree. |
| BM_006.png | BM_006.txt | BM_009.txt | confirmed | Public Health essential-drug stock order; same-stem BM_006 is the co-operation order. |
| BM_007.png | BM_007.txt | BM_010.txt | confirmed | Haveli development-works monthly notice; same-stem BM_007 is the IT circular. |
| BM_008.png | BM_008.txt | BM_017.txt | confirmed | Haveli development-works progress report; same-stem BM_008 is the e-tender order. |
| BM_009.png | BM_009.txt | BM_004.txt | confirmed | Public Health summer disease-prevention circular; same-stem BM_009 concerns essential drugs. |
| BM_010.png | BM_010.txt | BM_007.txt | confirmed | Electronics/IT/AI department circular, 11 May 2026. |
| BM_011.png | BM_011.txt | BM_008.txt | confirmed | E-tender workflow order, 18 December 2014; title and decision number agree. |
| BM_012.png | BM_012.txt | BM_012.txt | confirmed | DGIPR circular about provisional seniority list, 5 January 2025. BM_023 is its separate title sheet. |
| BM_013.png | BM_013.txt | BM_013.txt | confirmed | Landscape statement A(2), Mumbai electrical inspection; headings and totals agree. |
| BM_014.png | BM_014.txt | BM_014.txt | confirmed | Landscape statement A(4), Chhatrapati Sambhajinagar; the reference is noisy but identifies the same table. |
| BM_015.png | BM_015.txt | BM_015.txt | confirmed | Rotated statement A(4); office and table headings agree. |
| BM_016.png | BM_016.txt | BM_016.txt | confirmed | First photographed handmade-paper booklet order page; Sagar G.R. 2014 and page 299 agree. |
| BM_017.png | BM_017.txt | BM_018.txt | confirmed | Photographed continuation starts with the booklet’s item 6 and matches the reference’s provisions and signatory. |
| BM_018.png | BM_018.txt | — | needs_human_review | Later photographed continuation starts mid-provision; no available transcription matches the page. The same-stem BM_018 text matches BM_017 image instead. |
| BM_019.png | BM_019.txt | BM_019.txt | confirmed | District approval tables and headings agree. |
| BM_020.png | BM_020.txt | BM_020.txt | confirmed | Chhatrapati Sambhajinagar retired staff renewal list and succeeding district tables agree. |
| BM_021.png | BM_021.txt | BM_021.txt | confirmed | Visible serial rows and Kolhapur section agree. |
| BM_022.png | BM_022.txt | BM_022.txt | confirmed | English tender corrigendum title, dates, clauses and signatory agree. |
| BM_023.png | BM_023.txt | BM_023.txt | confirmed | Rotated four-line DGIPR seniority-list title sheet agrees. |
| BM_024.png | BM_024.txt | BM_024.txt | confirmed | Appendix E service-cadre program and table headings/entries agree. |

Summary: 24 images; 23 visually confirmed mappings; 1 unresolved page; 9 confirmed cross-ID mappings; 0 original references changed. The unresolved row is not approved for scoring. It may represent a page whose transcription is absent from the current 24 files; no substitute was guessed.

## Mapping-aware evaluation behavior

`eval/run_benchmark.py::load_benchmark` retains legacy same-stem behavior when `mapping_manifest` is omitted. With `--mapping-manifest`, it checks that the CSV covers every image exactly once, checks image IDs and paths, requires each accepted row to be `confirmed` and approved, verifies the proposed reference path exists, and rejects duplicate reference assignments. By default, any unapproved or unresolved row stops the run before inference. `--exclude-unapproved-mappings` explicitly excludes such rows and logs their IDs.

Cross-ID mappings are never inferred by the evaluator. The versioned CSV is only used when passed explicitly. Other older scripts (`eval/run_cer_comparison.py`, `eval/run_cer_comparison_with_wer.py`, `eval/run_crnn_improvement_experiments.py`, and the Table 8 scripts) still use their same-stem logic; they were not silently redirected.

## Historical score validity

The available top-level full-page result files are `results/cer_comparison.json` and `results/cer_wer_comparison.json`. Their loaders iterate all 24 IDs and pair by stem. Their recorded 24-page values are:

| System/mode | Historical CER | Historical WER | Status |
|---|---:|---:|---|
| AksharDrishti layout on | 0.805827 | 0.996503 | Provisional; same-stem references include mismatches. Saved result has aggregates/per-page CER, not predictions. Fresh inference required for corrected scoring. |
| AksharDrishti layout off | 0.850797 | 0.999068 | Provisional; same-stem references include mismatches. `cer_wer_comparison.json` has no predictions; fresh inference required. |
| Tesseract full page | 0.672304 | 0.958858 | Provisional; same-stem references include mismatches. Saved result has no predictions; fresh Tesseract inference required. |

`results/crnn_improvement_experiments/full_layout_off_results.json` does retain full-page predictions for the legacy, aspect-preserving and columns-plus-aspect CRNN variants, together with the original same-stem references. Those predictions were safely rescored against the 23 approved mappings into the new file `results/bm_pairing_audit_v1/corrected_saved_prediction_scores.json`; the historical JSON was not edited.

| Saved variant, approved 23-page subset | Corrected-reference CER | Corrected-reference WER |
|---|---:|---:|
| A — legacy capped | 0.821057 | 0.999048 |
| B — aspect preserving | 0.453389 | 0.996786 |
| D — columns plus aspect preserving | 0.453389 | 0.996786 |

These are re-scores of saved predictions on 23 pages only. They are not new inference results, do not include BM_018, and should not be compared as if they were the original 24-page benchmark. The large CER change for variants B/D reflects corrected reference pairing in addition to the one-page exclusion; it is not a new recognition improvement claim. BM_018’s prediction cannot be scored until its reference is resolved.

The CER/WER comparison script normalizes both sides with `normalize_for_scoring`: NFC, nukta canonicalization, joiner removal, whitespace collapse and trim. Punctuation and meaningful characters are retained. This is internally consistent across reference and prediction, but it means line breaks and repeated spaces are not scored; full-page order and table serialization (`|` separators) may still differ from plain transcription formatting. Such format/order issues remain separate comparability limits. Table 8 outputs also use same-stem page selection, and `table8_region_script_cer.json` explicitly reports approximate line-based region references; those region scores need separate review and must not be treated as corrected exact ground truth.

## Reproduction

Run from the `aksharDrishti` repository root.

Check the audited loader and its fail-closed behavior without running OCR:

```powershell
@'
from pathlib import Path
from eval.run_benchmark import load_benchmark
root = Path("benchmark")
manifest = Path("mappings/benchmark_image_gt_mapping_v1.csv")
load_benchmark(root, manifest)  # expected to raise for unresolved BM_018
items = load_benchmark(root, manifest, exclude_unapproved=True)
print(f"approved pages: {len(items)}")  # 23
'@ | python -
```

Re-score saved CRNN predictions (no model inference). The exact command used for this audit was:

```powershell
python eval/recompute_saved_benchmark_scores.py --benchmark benchmark --mapping-manifest mappings/benchmark_image_gt_mapping_v1.csv --source results/crnn_improvement_experiments/full_layout_off_results.json --out results/bm_pairing_audit_v1/corrected_saved_prediction_scores.json --exclude-unapproved-mappings
```

The re-score tool refuses to overwrite an existing output. To repeat it later, select a new versioned output filename.

Run fresh inference on the 23 approved pages for a corrected current result, writing into a new directory:

```powershell
python eval/run_benchmark.py --benchmark benchmark --mapping-manifest mappings/benchmark_image_gt_mapping_v1.csv --exclude-unapproved-mappings --systems akshardrishti tesseract_fullpage --out results/bm_pairing_audit_v1/fresh_inference_23pages
```

Run the focused and full test suites:

```powershell
python -m pytest tests/test_benchmark_pairing_manifest.py -q --basetemp .pytest-tmp/pairing_audit_run1
python -m pytest -q --basetemp .pytest-tmp/pairing_full_run1
```

Observed results: focused pairing tests **5 passed**; complete repository suite **96 passed in 217.96s**. The initial `python -m pytest tests/test_benchmark_pairing_manifest.py -q` invocation failed before test setup with `OSError: could not create numbered dir with prefix pytest- in C:\Users\ayush\AppData\Local\Packages\sandbox.{43e9715a-665b-4b0e-9e34-667ef7719b2d}\AC\Temp\pytest-of-ayush after 10 tries`. Rerunning with repository-local basetemp succeeded.

## Changed files in this audit

- `benchmark/mappings/benchmark_image_gt_mapping_v1.csv` — new audited mapping manifest.
- `eval/run_benchmark.py` — optional explicit mapping support; default same-stem behavior unchanged. This file was already modified in the worktree before this audit; those prior benchmark-failure handling changes were preserved.
- `eval/recompute_saved_benchmark_scores.py` — read-only saved-prediction re-score to a new output path.
- `tests/test_benchmark_pairing_manifest.py` — mapping, cross-ID, missing/duplicate, unapproved/unresolved, and unchanged-legacy/reference tests.
- `results/bm_pairing_audit_v1/corrected_saved_prediction_scores.json` — corrected 23-page re-score of saved CRNN outputs.
- `docs/BM_BENCHMARK_PAIRING_AUDIT.md` — this report.

No original benchmark image, transcription, metadata row, or historical result file was modified. No training, fine-tuning, model inference, weight download, or production-default change was performed during the audit.
