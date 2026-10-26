# OCR Evaluation Pipeline Audit

Audit date: 2026-10-26  
Scope: evaluation, scoring, CRNN inference, image preprocessing, and document text assembly. No model training or weights download was run.

## Summary

The checked full-document CER and WER implementations use edit distance on normalized text and normalized word tokens, respectively. The existing 24-page result file `results/cer_wer_comparison.json` records CERs of 0.8058 (layout on), 0.8508 (layout off), and 0.6723 (Tesseract), with WERs of 0.9965, 0.9991, and 0.9589. Those values are consistent with the stored output from `eval/run_cer_comparison_with_wer.py`; its WER implementation is token-level. The audit did not rerun that model benchmark, so it makes no claim that the scores changed.

Two verified defects were fixed: batched CRNN decoding could decode padded timesteps as real characters, and benchmark failures could be scored as empty transcriptions. A separate local WER helper in `eval/run_cer_comparison.py` used character-level distance over space-joined words; it is corrected and now rejects misaligned input lists. That script reports word accuracy, not WER, so it did not produce the WER values above.

The high full-document CER is not explained by Unicode normalization or the WER implementation. The evidence points to a combination of actual recognition and layout/segmentation limitations, with a clear train/inference granularity mismatch: the CRNN dataset is described as line/word crops, while the application passes segmented full-line crops. Long crops are compressed horizontally to the configured maximum width. These are credible causes to investigate, not accuracy improvements verified in this audit.

## Evaluation architecture

1. `eval/run_benchmark.py` pairs each image with `gt_text/<image stem>.txt`, then runs the configured system and scores its transcription. The pairing is exact by stem, and missing ground truth causes that image to be skipped with a warning.
2. `AksharDrishtiPipeline.process_page()` preprocesses the page, detects and merges layout regions (or creates one full-page region for the no-layout ablation), assigns reading order, segments text-bearing regions into lines, and invokes the script router/recognizer for each line. See `akshardrishti/pipeline.py:182-245,247-293`.
3. `Document.full_text()` joins ordered page text; `Page.full_text()` follows ordered regions, and region text is composed from lines or table cells. Tables with reconstructed cells use row/column order and ` | ` separators (`akshardrishti/schema.py:170-180,204-212,257-270`).
4. `eval/metrics.py` computes corpus CER from character edit distance divided by reference characters and corpus WER from edit distance over whitespace-separated word tokens divided by reference words. It also reports per-document CER and edit-operation counts.
5. The full-document comparison is also implemented in `eval/run_cer_comparison_with_wer.py`; it directly pairs `BM_*.txt` to same-stem PNGs and uses the project's scoring normalizer. This result path is separate from the word-crop Tesseract experiment in `eval/eval_tesseract_baseline.py`.

## Verified bugs and fixes

### Fixed

- **Padded batch timesteps were decoded as content.** `ctc_collate` calculates per-image CTC input lengths, but the CRNN training-path validation and the mode-A reproduction discarded them. The batched `CRNNRecognizer.recognize_batch()` also padded crops to the batch maximum and decoded every output timestep. Greedy and beam decoders now accept optional per-item sequence lengths; the training-path validation, CRNN test mode A, and recognizer batch path pass the valid lengths. The single-crop public inference path is unchanged. Files: `akshardrishti/recognize/crnn_model.py:134-211`, `akshardrishti/recognize/crnn.py:102-133`, `train/train_crnn.py:70-89`, `eval/eval_crnn_test.py:225-249`.
- **A failed benchmark page could look like a valid empty OCR result.** `run_benchmark` logged page errors and appended `""`, then included the page in the score. It now raises a system-level error listing failed pages instead of reporting a misleading CER/WER. Unreadable external-baseline images are handled the same way. File: `eval/run_benchmark.py:82-143`.
- **The custom comparison script's unused WER helper used character distance.** It joined word tokens back into a string before edit distance, which is not word-level WER. It now computes edit distance directly on token lists, and its local CER/word-accuracy/WER helpers reject length mismatches. The script still labels and reports word accuracy, not WER. File: `eval/run_cer_comparison.py:37-66`.
- **Shared corpus WER silently truncated unequal inputs.** It used `zip()` without an alignment check (CER already checked). It now raises a `ValueError` on mismatched list lengths. File: `eval/metrics.py:90-136`.

### Not confirmed as bugs

- **Unicode:** `normalize_for_scoring()` applies NFC, canonicalizes nukta variants, removes joiners, and collapses all whitespace on both reference and hypothesis (`akshardrishti/postprocess/text_repair.py:148-160`). This makes line breaks and repeated whitespace equivalent while preserving punctuation and meaningful characters. Pipeline cleanup also normalizes prediction text. No inconsistent NFC path was found in the full-document scoring scripts.
- **Ground-truth mapping:** The 24-page full-document runner pairs by exact stem; the stored result reports 24 pages. No mismatched image/transcription pair was found from code inspection. Other evaluation loaders preserve their CSV row order, with `shuffle=False` for the test DataLoader.
- **Reading order and reconstruction:** Regions are read in computed reading order, lines follow projection segmentation order, and reconstructed table cells follow row/column order. The code does not prove these orders match each transcription's authoring order. Table ` | ` separators and list line breaks can differ from a plain prose transcription; use format-aware scoring or separate region/table analysis before treating these as recognizer character errors. No change was made because the benchmark ground truth's formatting convention was not changed or independently validated.
- **Empty or duplicate regions:** Empty OCR results are omitted when region text is composed; overlapping regions are merged before recognition. No reproducible duplicate-region error was established in this audit.

## CRNN and image-path findings

- The application segments text-bearing regions into **line crops** and calls the recognizer on each line (`akshardrishti/pipeline.py:197-242`). The CRNN data loader describes its examples as “line/word crops” (`train/dataset.py:24-29`), and its training data source is the Mozhi line/word-crop index. This is a material granularity/domain mismatch to investigate; the current evaluation cannot establish its size.
- Inference converts to grayscale, resizes to height 32, applies Otsu binary thresholding, scales intensities to `[-1, 1]`, then adds configured 64-pixel white right padding (`akshardrishti/recognize/crnn.py:65-78`; `configs/pipeline.yaml:92-98`). Training uses grayscale, height 32, Otsu, the same intensity range, and only batch-to-maximum-width white padding (`train/dataset.py:96-140`). The padding differs by design and is described as validation-selected in the recognizer; it is not a per-crop preprocessing parity test. Page-level resizing, deskew, denoising, and CLAHE happen before inference segmentation, and are not part of training-crop preprocessing (`akshardrishti/preprocess/enhance.py:185-241`).
- Both training and inference cap resized crop width at 512. For a wider-than-limit crop, the image is resized to exactly 512 pixels wide instead of preserving aspect ratio (`train/dataset.py:105-115`; `akshardrishti/recognize/crnn.py:65-78`). This compresses long lines and can distort glyph spacing. It is existing behavior on both paths, so it was not changed as part of an evaluation-only audit.
- The decoder's CTC collapse (repeat suppression and blank removal) is correct for unpadded single-sample outputs. The verified issue was failure to limit decoding to the per-item valid output length in a padded batch, now fixed. No wrong class/blank index was found: charset characters begin at index 1 and blank is index 0.
- Pipeline tables may be segmented to cells and joined with literal separators; lists are handled as line sequences. These formatting decisions can inflate whole-document CER/WER when ground truth uses different structure. The benchmark scores transcription text, not region-aligned content, so this remains a measurement caveat rather than a verified bug.

## Whether the previous numbers remain valid

- The stored full-document output `results/cer_wer_comparison.json` is internally aligned to 24 same-stem image/GT pairs and was produced using correct character-level CER and word-token WER. The single-crop full pipeline did not use padded batch decoding, so the decoder fix does not invalidate those full-document numbers. They remain valid as measurements for that exact benchmark snapshot and configuration; they have not been freshly reproduced after this audit.
- The `run_benchmark` failure-handling fix can change future benchmark behavior if a page errors: it now reports a failed run rather than silently scoring an empty string. There is no evidence that this affected the stored 24-page run.
- The separate `eval/eval_tesseract_baseline.py` is a 20,026 word-crop experiment and reports exact word accuracy, not general corpus WER. Its result and graph are not directly comparable to the 24-page full-document CER/WER result. Do not substitute one for the other.
- The real-document layout log `layout_bench_out.txt` reports 24 pages, mAP@50 0.2752, precision@50 0.5185, and recall@50 0.8462. The output indicates many predictions relative to ground truth on several classes. This is consistent with a detector precision/classification problem; no mAP implementation defect was established in the inspected OCR evaluation path. No detector or weights were changed.

## Tests and commands

Executed lightweight suite:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider --basetemp .pytest-tmp tests/ -q
```

Result: **62 passed**. The first attempt also exposed a reversed reference/hypothesis order in the new test assertion; I corrected it. Pytest's default temporary directory was outside the workspace sandbox, so the passing rerun directed temporary files into `.pytest-tmp` in the workspace. That temporary directory was removed afterward.

An AST parse passed for all eight modified Python files. `git diff --check` reports trailing whitespace in the pre-existing modified `eval/compute_table8.py` at lines 127, 136, 140, 145, 152, 155, 159, 161, and 185. That file was not changed by this audit.

The following model evaluation commands were not run: they perform tens of thousands of inferences or run the full 24-page detector/OCR benchmark, and were outside the lightweight test scope. They require the local model weights, datasets, and external tools already present/configured; this audit did not download or install them.

```powershell
# CRNN test-split reproduction; writes a JSON result under results/.
python eval/eval_crnn_test.py --data datasets/mozhi_raw --out results/crnn_test_repro_audit.json

# Full-page systems and layout ablation; writes predictions and metrics under results/audit_run/.
python eval/run_benchmark.py --benchmark benchmark --systems akshardrishti tesseract_fullpage --ablations --out results/audit_run/

# Historical full-page CER/WER comparison; writes results/cer_wer_comparison.json.
python eval/run_cer_comparison_with_wer.py

# Tesseract word-crop baseline; runs against the full test index and writes results/tesseract_baseline.json.
python eval/eval_tesseract_baseline.py
```

## Remaining suspected recognition problems and next steps

1. **Check crop granularity and width distortion first.** Measure Mozhi train/validation crop widths and characters per crop against application line crop distributions. Evaluate existing CRNN checkpoints by width buckets and long-line subset, without retraining. If the domain mismatch is confirmed, plan data/model work separately.
2. **Inspect prediction/reference pairs with geometry.** For a small, fixed sample of the 24 pages, compare detected regions, line boxes, ordered text, and per-region reference. This separates missed content, duplicates, reading-order errors, and glyph recognition errors.
3. **Report region-aware and format-aware text scores.** Score prose, lists, and tables separately; define whether table cell separators, list markers, and page furniture belong in the reference. Preserve a plain-text corpus score for direct comparisons.
4. **Audit YOLO false positives by class.** In the stored layout benchmark, predicted boxes exceed GT for several classes, while recall is comparatively high. Inspect false-positive examples and class mapping, then decide whether confidence/class-specific calibration or annotation review is appropriate. Do not infer a retraining fix from mAP alone.
5. **Run the full benchmark only after the output paths and snapshot are recorded.** Include the benchmark manifest, config, checkpoint hashes, page-level failures, per-page CER/WER, and the exact commands above in future result artifacts.

## Modified files

- `akshardrishti/recognize/crnn.py`
- `akshardrishti/recognize/crnn_model.py`
- `eval/eval_crnn_test.py`
- `eval/metrics.py`
- `eval/run_benchmark.py`
- `eval/run_cer_comparison.py` (was already modified before this audit; targeted edits preserved the existing work)
- `train/train_crnn.py`
- `tests/test_eval_metric_regressions.py` (new)
- `docs/OCR_EVALUATION_AUDIT.md` (this report)

The pre-existing change to `eval/compute_table8.py` and all pre-existing untracked data/results were left untouched.
