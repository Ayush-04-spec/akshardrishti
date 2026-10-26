# BM ground-truth to crop alignment audit

Date: 2026-10-26

## Summary

The benchmark has 24 `BM_*.png` images, 24 `BM_*.txt` transcriptions, and 24 IDs in `benchmark/metadata.csv`. The stems match one-to-one. **Matching stems do not guarantee matching page content.** Visual inspection found three clear same-stem content mismatches among the five pages represented in the 153-crop pack:

| Crop source image | Same-stem GT | Visually matching GT | Evidence |
|---|---|---|---|
| BM_010.png | BM_010.txt | BM_007.txt | Image opens with “सर्व शासकीय पत्रव्यवहार…” and the matching BM_007 transcription has the same distinctive title/body. |
| BM_011.png | BM_011.txt | BM_008.txt | Image title (“रुपये 3 लाखापेक्षा… ई-निविदा”) and मातंस/नस्ती reference agree with BM_008. |
| BM_017.png | BM_017.txt | BM_018.txt | Image shows the continuation about एकात्मिक बाल विकास, section ६; BM_018 transcription starts with that same passage. |

BM_001 image/GT opening and BM_021 image/table rows/GT were manually checked and appear consistent. Other same-stem pairs outside these five candidate pages were inventoried by ID but not visually verified for semantic agreement. The cross-ID mappings above are review-aid mappings only; no ground-truth files were edited or renamed.

Because at least three image/reference pairs used by stem-based benchmark loading are mismatched, **historical page-level CER/WER that scored these same-stem pairs cannot be treated as valid for the intended source documents until the full benchmark pairing is audited and corrected in a separately approved benchmark manifest.** Existing scores and benchmark data were left untouched.

## Alignment method and limits

The input pack is `results/crnn_long_sequence_investigation/annotation_pack/annotation_manifest_template.csv`, with 153 crop candidates from BM_001, BM_010, BM_011, BM_017, and BM_021. Each candidate points to an image crop and a bounding box on the preprocessed page. The crop coordinates and images were not changed.

The proposal code `eval/crop_gt_alignment.py` uses NFC text, the transcription’s existing nonblank line order, candidate crop order, and a weak page-specific crop-width / ground-truth-character-count fit in a monotonic dynamic program. It can propose a span of one transcription line or two adjacent lines. It does not read crop pixels, run OCR, use recognition predictions, strip punctuation, or infer missing characters. A two-line span is marked for boundary review. This is a ranking/proposal aid: its confidence values are geometry-fit scores, **not calibrated probabilities** and not proof of crop transcription.

The review manifest explicitly records both the image page ID and the GT page ID used for a proposal. BM_010/BM_011/BM_017 use the manually visually matched alternate GT IDs listed above. All rows remain `unverified`, require manual review, and have `eligible_for_scoring=false`. The helper `scorable_pairs` admits only rows whose alignment and verification status are both explicitly `verified` and whose transcription is nonempty.

## Results for the 153 candidate crops

| Status | Count | Interpretation |
|---|---:|---|
| `candidate_match` | 4 | Best geometry/order fit; still unverified and must be checked against the crop. |
| `ambiguous_match` | 89 | A passage is proposed, but line-length/order evidence is weak or nonunique. |
| `boundary_review` | 50 | Proposed span needs boundary review; includes two-line suggestions and all BM_021 table candidates. |
| `no_match` | 10 | No monotonic span met the proposal alignment. |
| **Proposed nonempty spans** | **143** | Candidate text is available as a review suggestion, not ground truth. |
| **Manual review required** | **153** | Every crop, including all four `candidate_match` rows. |
| **Eligible for crop scoring** | **0** | No crop has independent transcription verification. |

No crop boxes exceeded the duplicate/overlap detector’s IoU threshold of 0.50 (0 overlapping pairs). This does not rule out partial text duplication or a crop that combines adjacent visual content. BM_021’s table rows were marked `boundary_review` because line projections cannot establish per-cell transcription boundaries.

### Crop width distribution

Width is the crop’s predicted width after normalization to height 32, before the CRNN 512 px cap.

| Source page | Crops | ≤128 | 129–256 | 257–512 | 513–1024 | >1024 | Median width | p90 width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM_001 | 31 | 0 | 2 | 8 | 9 | 12 | 846 | 1,447 |
| BM_010 | 24 | 0 | 0 | 1 | 9 | 14 | 1,477 | 1,483 |
| BM_011 | 26 | 0 | 0 | 1 | 3 | 22 | 1,497 | 1,555 |
| BM_017 | 42 | 0 | 0 | 0 | 3 | 39 | 1,481 | 1,906 |
| BM_021 | 30 | 0 | 0 | 1 | 1 | 28 | 1,533 | 1,959 |

Per-page status counts, widths, crop dimensions, pixel boxes, matched transcription spans, and review reasons are in `alignment_summary.json` and `alignment_manifest.csv` in the output directory.

## Human inspection sample

I compared representative source pages, crops, and transcription text:

- BM_001 crop 000 visually reads “महाराष्ट्र शासन” and agrees with the first BM_001 GT line.
- BM_010 crop 000 visibly begins “सर्व शासकीय पत्रव्यवहार…”; this does not agree with BM_010.txt but agrees with the BM_007 GT opening.
- BM_011 page heading and BM_008 GT title identify the same ई-निविदा circular; the same-stem BM_011 GT starts with a different Directorate header.
- BM_017 crop 001 and its page show the section ६ child-development passage in BM_018.txt, not the panchayat report text in BM_017.txt.
- BM_017 crop 000 is a scan-edge artifact; the aligner leaves it `no_match`.
- BM_021 crop rows contain table entries similar to BM_021.txt, but the projection crop may span multiple columns, so boundary review is mandatory.

This is a small visual sample, not certification of all 143 proposed spans. No OCR recognition output was used to create candidate transcriptions. No crop-level CER/WER/exact-word accuracy was computed.

## Review aid and annotation protocol

Open `results/bm_ground_truth_crop_alignment_20261026_v7/review.html` in a browser. The page shows each actual crop beside the proposed transcription and page-GT passage, image/crop IDs, dimensions, bbox, heuristic score, status, reason, and neighboring crop thumbnails. It defaults to high-priority `candidate_match` rows first and includes page/status filters. Green/amber/red status styles distinguish the remaining uncertain/boundary cases.

Reviewers should compare the crop itself with the displayed GT span and adjacent crops, correct or reject the bbox as needed, preserve punctuation/spacing, verify text in NFC, then explicitly set both `alignment_status=verified` and `verification_status=verified` only after independent human verification. Cross-page source mappings should be confirmed at the document level before line scoring. Keep all BM pages test-only; do not send them to train/validation.

## Tests and command results

Focused tests cover monotonic span proposal, ambiguous equal-geometry cases, NFC preservation of Marathi text/punctuation/spaces, and exclusion of unverified crops from scoring.

- `python -m pytest -p no:cacheprovider --basetemp .pytest-tmp tests/test_crop_gt_alignment.py -q` — **4 passed in 0.85s**.
- `python -m pytest -p no:cacheprovider --basetemp .pytest-tmp tests/ -q` — **91 passed in 14.44s**.
- `python eval/build_bm_crop_alignment_review.py` — completed; produced 153 rows, 143 nonempty proposal spans, and 0 score-eligible rows.
- The first builder attempt wrote its new output, then failed while printing Marathi JSON to the Windows CP1252 console with `UnicodeEncodeError: 'charmap' codec can't encode characters in position 864-867: character maps to <undefined>`. The builder was updated to set UTF-8 stdout; the final reproduction command above completes successfully. The failed attempt did not overwrite source GT or existing annotation files.

## Files and reproducibility

New code/tests/report:

- `eval/crop_gt_alignment.py`
- `eval/build_bm_crop_alignment_review.py`
- `tests/test_crop_gt_alignment.py`
- `docs/BM_GROUND_TRUTH_CROP_ALIGNMENT.md`

Final generated output (created as a new directory; prior output directories were not overwritten):

- `results/bm_ground_truth_crop_alignment_20261026_v7/alignment_manifest.csv`
- `results/bm_ground_truth_crop_alignment_20261026_v7/alignment_summary.json`
- `results/bm_ground_truth_crop_alignment_20261026_v7/review.html`

From `D:\AksharDrishti_1\aksharDrishti`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python eval/build_bm_crop_alignment_review.py --out results/bm_ground_truth_crop_alignment_REPRO_20261026
python -m pytest -p no:cacheprovider --basetemp .pytest-tmp tests/test_crop_gt_alignment.py -q
```

The builder refuses to run if its default output directory already exists. Use `--out` with a new path for a subsequent review build. It reads but does not alter the benchmark images, page transcriptions, manifest, or saved model-prediction artifacts.
