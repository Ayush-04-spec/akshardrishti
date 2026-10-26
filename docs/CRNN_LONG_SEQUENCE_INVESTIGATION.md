# CRNN long-sequence investigation

Date: 2026-10-26  
Scope: BM_017, BM_010, BM_011; no model training, weight downloads, benchmark-reference edits, or production-default changes.

## Repository and safeguards

The Git root is `D:\AksharDrishti_1\aksharDrishti` (the parent is not the application repository). Before this work, the worktree already contained user changes in source/config/evaluation files and many untracked user artifacts. Those were preserved. This work adds only the diagnostic scripts, focused tests, this report, and files under `results/crnn_long_sequence_investigation/`. The configured default remains `legacy_cap`.

The local checkpoint `weights/recognize/crnn_mozhi_himr.pt` and benchmark images/transcriptions were available. The three-page forensic rerun exactly reproduced the saved A/B page CER values and normalized predictions in `results/crnn_improvement_experiments/full_layout_off_results.json`.

## Findings

### Long-sequence mismatch is established; a CTC length/decoder defect is not

`train/dataset.py:96-119` height-normalizes Mozhi crops to 32 px and caps width at 512 before tensor conversion. Its recorded training profile (`results/recognition_diagnostics/diagnostics.json`) covers 10,000 Mozhi train, 5,000 validation, and 5,000 test examples: median normalized widths are 59/57/59 px, p90 widths 111/109/112 px, median text lengths 4 characters, p90 8, and none of the 20,000 sampled crops exceeds 512 px. By comparison, an earlier 423-crop GovDoc candidate profile has median normalized width 628 px, median text length 42, and 56.3% above 512 px. Those GovDoc labels are heuristic and are cited only as crop/length profile metadata, never as trusted recognition truth.

All BM_017 lines and all but one BM_010/BM_011 lines exceed the current cap after height normalization. The 92-crop forensic run measured:

| Page | Segmented lines | Width <=512 | Width >512 | normalized width range | aspect output T (range) | valid content T (range) |
|---|---:|---:|---:|---:|---:|---:|
| BM_017 | 42 | 0 | 42 | 565–2,690 px | 156–687 | 140–671 |
| BM_010 | 24 | 1 | 23 | 412–1,540 px | 118–400 | 102–384 |
| BM_011 | 26 | 1 | 25 | 454–1,556 px | 128–404 | 112–388 |

The model computes `T = floor(floor(W/2)/2) - 1`. For example, a BM_017 crop at normalized width 1,483 produces 385 actual logits steps after the 64-px configured right pad; the content-valid length is 369. Thus the 16 right-pad steps are excluded in aspect mode. The formula agrees with actual model output geometry at widths tested (128, 512, 800, 2,048). Padded high-probability characters are excluded by the existing sequence-length path. Newly added greedy and beam tests verify blank/repeat collapse (`क,क,blank,क` becomes `कक`), valid-length trimming, per-width batching, and empty valid sequences.

No confirmed defect was found in aspect-mode output-length geometry, right-padding polarity, or blank/repeat handling. Beam search is vanilla prefix CTC beam search without a language model (`crnn_model.py:163-220`); it consumes only supplied valid timesteps. The aspect mode raises beyond `max_preserved_width` (default 8,192) rather than silently truncating. Legacy behavior remains capped and default.

### BM_017 insertion-heavy output

Full-page layout-off results remain: legacy CER 0.86126/WER 1.00000; aspect-preserving CER 1.81937/WER 1.00000. The saved experiment document says aspect output length is 2,086 normalized characters against 1,146 reference characters, but the checked-in run artifact reports 2,531 normalized output characters (2,531 raw Python characters) and the current exact rerun reproduces that prediction and CER. Legacy artifact length is 757 normalized characters (759 raw characters). This is an artifact/documentation inconsistency: use the JSON prediction, `hypothesis_characters_normalized`, and exact rerun as the evidence, and correct the prior 2,086 figure before citing output length elsewhere. The CER and WER agree.

The page overlay shows normal horizontal body lines, one main column, and two page-edge boxes containing scan-edge marks. Inspected crops are single rows rather than multiple lines or unrelated columns. Some long enumerated/numeric rows and address/signature/footer rows are present. Therefore a wholesale multi-line merge is not the primary explanation for the insertion burst, although the two edge-artifact boxes are segmentation false positives. In the most affected row examples, widths near 1,480–1,540 normalized pixels (roughly 369–384 content timesteps) yield 88–95 decoded characters, versus 18–23 in legacy mode. Across the page, line-level output character totals are 718 legacy and 2,490 aspect (joined-page output is 759 and 2,531 including separators). This supports an out-of-training-range long-sequence recognition failure; without line transcripts it cannot distinguish hallucination caused by model generalization from visually ambiguous degraded glyphs at the character level.

### BM_010 and BM_011 regressions

| Page | Legacy page CER | Aspect page CER | Legacy joined chars | Aspect joined chars | lines >512 px |
|---|---:|---:|---:|---:|---:|
| BM_010 | 0.83415 | 0.84337 | 814 | 1,463 | 23/24 |
| BM_011 | 0.85347 | 0.93830 | 1,116 | 1,735 | 25/26 |

Their projection boxes are horizontal text rows, primarily about 1,400–1,556 normalized pixels wide. Per-line normalized widths and model/valid output lengths are in `results/crnn_long_sequence_investigation/line_forensics.csv`. The larger aspect-mode output length explains why CER regresses despite preserving glyph proportions. There is no reliable line-level transcription in the benchmark, so neither per-line CER nor error category counts (matra, conjunct, spacing, glyph) can be truthfully assigned. The full-page GT cannot be mapped to individual line crops without manual alignment.

The captured overlays and original crop images are diagnostic outputs, not labels: `results/crnn_long_sequence_investigation/BM_017_segmentation.png`, `BM_010_segmentation.png`, `BM_011_segmentation.png`, and `BM_*_line_NNN.png`.

## Code path and confirmed behavior

- `akshardrishti/preprocess/enhance.py:326-342` converts BGR to grayscale, resizes to height 32, applies Otsu binarization, and returns the entire width.
- `akshardrishti/recognize/crnn.py:17-24,80-108` defines model width geometry, aspect-preserving versus legacy-cap preparation, normalized grayscale `[-1,1]`, and white right padding (`+1`).
- `akshardrishti/recognize/crnn.py:111-169` runs one-item/batched inference and applies per-item valid CTC lengths in aspect-preserving mode.
- `akshardrishti/recognize/crnn_model.py:135-220` implements greedy collapse and prefix beam decoding. Regression coverage now checks repeats separated by blanks and truncation at valid lengths.
- `train/dataset.py:96-140` shows the short-crop training preprocessing and matching width geometry for training CTC lengths.
- `akshardrishti/layout/lines.py:36-115` performs horizontal projection segmentation; it can split rows but does not understand table cells or page-edge artifacts.
- `akshardrishti/pipeline.py:182-242` crops each line, routes it to recognition, cleans recognized text, and composes reading-order text.
- `configs/pipeline.yaml:102-106` keeps `legacy_cap` default, max width 512, and aspect preservation opt-in.

No production implementation bug was confirmed in this investigation, so no production code or defaults were changed. The verified defects covered in existing evaluation fixes remain documented in `docs/OCR_EVALUATION_AUDIT.md`; they are distinct from this long-sequence failure.

## Tests and measurements

- `python eval/investigate_crnn_long_sequences.py`: completed on BM_017/BM_010/BM_011; 92 crops, zero missing references/images, and reproduced the stored page CER values exactly.
- `python eval/prepare_crnn_annotation_manifest.py`: exported 153 untranscribed candidates from BM_001, BM_010, BM_011, BM_017, BM_021.
- `python -m py_compile eval/investigate_crnn_long_sequences.py eval/prepare_crnn_annotation_manifest.py`: passed.
- `python -m pytest -p no:cacheprovider --basetemp .pytest-tmp tests/ -q`: **87 passed in 8.61s** (the 84-test baseline plus 3 focused regression cases).
- Tesseract same-crop inference was unavailable: the Tesseract executable is not installed/on PATH. No crop-level Tesseract or CRNN accuracy is reported.
- No human-verified crop transcriptions are available yet. Crop-level CER, WER, exact-word accuracy, and error categories are therefore intentionally unreported.

## Proposed trusted dataset and annotation workflow

An annotation starter pack is in `results/crnn_long_sequence_investigation/annotation_pack/`. Its 153 candidates include clean printed rows from BM_001/BM_010/BM_011, degraded scan rows from BM_017, and dense/table candidates from BM_021. The CSV contains blank transcription fields, original-image SHA-256 values, crop paths, pixel bboxes, dimensions, normalized widths, review status, and explicit `benchmark_test_only` split. BM_021 rows require manual crop-boundary review; projection crops cannot be presumed to represent table cells.

`annotation_pack/ANNOTATION_PROTOCOL.md` requires visual crop verification, manual cell/row subdivision where needed, two independent transcriptions, NFC storage, and an independent verifier. Keep all BM_001–BM_024 documents and derivatives test-only. For a true train/validation/test dataset, acquire new Marathi government documents not used in this benchmark and assign whole document IDs to splits before generating crops. Do not use `data/make_govdoc_finetune.py` heuristic labels as trusted labels. No current independently verified real-document line-level training or validation split was found.

Once transcriptions are verified, report crop-level legacy/aspect/Tesseract metrics on the same crops separately from full-page benchmark scores: corpus and per-crop CER/WER, exact-word accuracy, width bucket, and reviewed error categories. Keep source page and crop IDs in the manifest; do not use page-level GT for line scoring.

## Recommendation

1. **Create manually verified, page-disjoint real-document line/cell labels.** This is the prerequisite for attributing visual recognition errors and deciding between the modes. The current benchmark page GT is not line aligned, and existing heuristic crop labels cannot safely fill that gap.
2. **Then run a controlled long-sequence recognition experiment.** There is strong distribution evidence: Mozhi training examples are mostly 4-character, <=112-px normalized crops, while these real lines are often 1,400–1,556 px and aspect mode produces much longer outputs. Evaluate aspect-preserving behavior by trusted width buckets and include safe line splitting/overlap controls. Do not raise the 8,192 guard or alter the production default based on page CER alone.
3. **Review segmentation/orientation for specific document classes.** BM_017 has two scan-edge false positives; table/page orientation problems elsewhere remain independent and need their own reviewed crop references. The inspected BM_017 text rows are not generally merged multi-line crops.
4. **Benchmark Tesseract and Surya only after the same-crop protocol is ready.** Tesseract executable is currently unavailable; Surya would add no trustworthy comparison without verified crop GT. Do not assume it is better.
5. Defer model fine-tuning until document-disjoint train and validation labels exist; do not tune on BM benchmark pages.

## Reproduction commands

From `D:\AksharDrishti_1\aksharDrishti`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python eval/investigate_crnn_long_sequences.py
python eval/prepare_crnn_annotation_manifest.py
python -m pytest -p no:cacheprovider --basetemp .pytest-tmp tests/ -q
```

The forensic command uses the current local CRNN checkpoint and existing BM page images only. It writes line crops, overlays, JSON, and CSV under `results/crnn_long_sequence_investigation/`; the annotation command writes untranscribed crop candidates and a blank manifest under its `annotation_pack/` subdirectory. Neither command changes benchmark ground truth or model configuration.
