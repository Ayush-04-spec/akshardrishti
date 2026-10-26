# CRNN improvement experiments

Date: 2026-10-26

## Scope and safeguards

This phase added opt-in implementation paths and evaluated them against the existing checkpoint. It did not retrain a model, download/install a model or dataset, modify benchmark ground truth, or change the production defaults. Legacy CRNN inference remains the configured default; orientation correction is disabled by default; projection line segmentation remains the default.

The initial three-page pilot was followed by a full 24-page **layout-off** comparison because the aspect-preserving pilot improved CER on two upright pages. This is the full layout-off ablation only. It is not the layout-on production pipeline, does not compare Tesseract again, and does not establish generalization beyond this fixed benchmark snapshot.

## Changes implemented

### Optional orientation correction

The page preprocessing interface now supports an opt-in Tesseract OSD orientation provider. It consumes Tesseract's own reported clockwise rotation and orientation confidence, applies only right-angle rotations with confidence at or above the configured minimum, and fails closed (leaves the page unchanged) on low confidence or provider errors. The output preprocessing trace records the selected clockwise rotation; page rotation metadata records the correction. Preprocessing operates on its own grayscale copy, preserving the caller's original image.

Exact pixel-rotation and coordinate-transform helpers support 0, 90, 180, and 270 degrees. They are covered by tests. Configuration keeps the provider disabled by default. This environment has the pytesseract wrapper, but not the Tesseract executable/OSD data, so automated orientation classification was unavailable and no confidence values were fabricated.

For one diagnostic only, BM_015 was rotated clockwise 90 degrees after visually inspecting the source image. Ground-truth text was not used to choose the direction. This is a manually selected correction example, not an automatic orientation prediction.

### Conservative column segmentation

The opt-in line method columns_then_projection searches for low-ink vertical gutters that span the region, requires a minimum gutter width and minimum width for every resulting column, then segments columns top-to-bottom in left-to-right order. If those conditions do not provide a safe partition, it falls back to the existing projection segmenter. Existing pipeline layout regions are still established before line segmentation. Existing TABLE regions continue to use the existing table-cell reconstruction path when enabled; cells are recognized independently and composed in row/column order.

The strategy remains opt-in. With layout disabled on the full benchmark, the full-height gutter criterion did not accept any split. A separate box-by-box comparison confirmed the experimental and existing projection segmenters returned identical line boxes on all 24 preprocessed pages. It therefore produced exactly the same line counts, predictions, CER, and WER as aspect-preserving inference. This is a useful conservative failure mode, but it is not evidence that the strategy resolves the BM_021 table issue. A layout-on comparison with fresh detector output could not be run because ultralytics is unavailable.

### Aspect-preserving CRNN mode

CRNN config now exposes inference_mode with legacy_cap and aspect_preserving values. The default remains legacy_cap. Aspect-preserving mode resizes only by target height, preserves normalized width, and adds the existing configured white right padding. Valid CTC sequence lengths are calculated from the unpadded content width using the model geometry: floor(floor(width/2)/2) − 1, bounded below by one. The sequence lengths exclude configured right padding and additional per-batch padding so those timesteps cannot produce decoded characters. Charset and blank index are unchanged.

The experimental mode logs and raises if a crop exceeds max_preserved_width (default 8,192 normalized pixels); it never silently squeezes or truncates that crop. Batch inference keeps variable widths by padding to the batch maximum and passes the per-item valid lengths. Tests exercise single and batched decoding, including high-probability symbols placed only in padded output timesteps.

## Methods

The pilot and full comparison use the same page images, same-stem benchmark transcriptions, layout disabled, the same configured checkpoint and decoder, and eval.metrics.cer/wer with the project's shared score normalizer:

- A: existing legacy cap (512 px), projection segmentation.
- B: aspect-preserving inputs, projection segmentation.
- C: aspect-preserving inputs plus a visual-only clockwise 90-degree correction on BM_015.
- D: aspect-preserving inputs plus columns_then_projection.

The page model is loaded once and reused across variants. Every attempted page returned a score; every variant had zero empty and zero failed line crops on its evaluated pages. The harness records per-page wall time, pipeline recognition-stage time, process RSS before/after the page, CUDA peak allocated memory, line-crop count, empty/failed crop count, references, and predictions in results/crnn_improvement_experiments/. RSS is process resident memory and includes loaded runtime/model memory; it is not an isolated per-crop allocation. CUDA peaks are allocator-reported peaks per page.

## Full 24-page layout-off results

| Variant | Pages scored / failed | Corpus CER | Corpus WER | Sum page wall time | Sum recognition time | Max CUDA allocated | Max observed RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — legacy cap | 24 / 0 | 0.85084 | 0.99907 | 47.49 s | 32.01 s | 88.87 MB | 1,027.3 MB |
| B — aspect preserving | 24 / 0 | 0.67874 | 0.99825 | 90.11 s | 74.01 s | 114.09 MB | 1,099.1 MB |
| D — columns plus aspect preserving | 24 / 0 | 0.67874 | 0.99825 | 92.38 s | 76.35 s | 114.09 MB | 1,149.3 MB |

A reproduces the stored layout-off result (CER 0.85080, WER 0.99907) within the last reported precision. B's CER is lower by 0.17211 on this exact 24-page layout-off set, while WER remains 0.99825. The result is still slightly worse than the previously stored Tesseract full-page baseline on this set (CER 0.67230, WER 0.95886). Aspect-preserving inference also takes about 2.0 times the sum of baseline page wall time and 2.3 times recognition-stage time in this run.

Page-level A/B results follow. D matched B exactly for all pages and line counts.

| Page | Lines | A CER | A WER | B CER | B WER |
|---|---:|---:|---:|---:|---:|
| BM_001 | 31 | 0.50634 | 0.99519 | 0.16878 | 0.99038 |
| BM_002 | 31 | 0.53351 | 0.99632 | 0.29468 | 0.99265 |
| BM_003 | 34 | 0.84279 | 0.99705 | 0.77511 | 1.00000 |
| BM_004 | 35 | 0.90420 | 0.99631 | 0.78690 | 1.00000 |
| BM_005 | 25 | 0.74644 | 1.00000 | 0.38938 | 1.00000 |
| BM_006 | 28 | 0.91799 | 1.00000 | 0.73658 | 0.99662 |
| BM_007 | 27 | 0.82773 | 1.00000 | 0.81067 | 1.00000 |
| BM_008 | 19 | 0.86628 | 1.00000 | 0.83721 | 1.00000 |
| BM_009 | 31 | 0.76648 | 1.00000 | 0.76420 | 0.99587 |
| BM_010 | 24 | 0.83415 | 1.00000 | 0.84337 | 1.00000 |
| BM_011 | 26 | 0.85347 | 1.00000 | 0.93830 | 1.00000 |
| BM_012 | 26 | 0.66203 | 1.00000 | 0.28720 | 0.99444 |
| BM_013 | 22 | 0.97590 | 1.00000 | 0.68330 | 1.00000 |
| BM_014 | 19 | 0.93270 | 1.00000 | 0.87421 | 0.99687 |
| BM_015 | 21 | 0.98697 | 1.00000 | 0.98120 | 0.99943 |
| BM_016 | 45 | 0.87991 | 0.99539 | 0.19591 | 0.99539 |
| BM_017 | 42 | 0.86126 | 1.00000 | 1.81937 | 1.00000 |
| BM_018 | 25 | 0.92659 | 1.00000 | 0.66819 | 0.99550 |
| BM_019 | 21 | 0.85766 | 0.99355 | 0.40254 | 0.99355 |
| BM_020 | 21 | 0.81000 | 1.00000 | 0.47222 | 1.00000 |
| BM_021 | 30 | 0.85178 | 0.99412 | 0.63790 | 0.99412 |
| BM_022 | 21 | 0.82652 | 1.00000 | 0.49096 | 1.00000 |
| BM_023 | 10 | 0.88189 | 1.00000 | 0.88189 | 1.00000 |
| BM_024 | 11 | 0.82419 | 1.00000 | 0.59351 | 1.00000 |

Aspect preservation did not improve every page. CER worsened on BM_010 and BM_011, was unchanged on BM_023, and increased sharply on BM_017 from 0.86126 to 1.81937. BM_017's output grew from 757 to 2,086 normalized characters against a reference length of 1,146, indicating a serious long-sequence recognition failure. This regression and persistent near-1 WER argue against replacing the default without further investigation.

## Pilot orientation result

| Page / method | Lines | CER | WER | Wall time | Recognition time | CUDA peak |
|---|---:|---:|---:|---:|---:|---:|
| BM_015, capped, unrotated | 21 | 0.98697 | 1.00000 | 1.58 s | 0.70 s | 88.87 MB |
| BM_015, aspect preserving, unrotated | 21 | 0.98120 | 0.99943 | 2.68 s | 1.96 s | 104.80 MB |
| BM_015, manually CW 90°, aspect preserving | 19 | 0.94638 | 1.00000 | 4.52 s | 3.73 s | 111.10 MB |

The manually corrected image is visibly upright, and preprocessing coordinates are in the rotated image frame. CER improved modestly, but WER remained 1.0 and recognition remained unusable. Automated orientation was not tested because its provider was absent. This single page is not an orientation accuracy evaluation.

![BM_015 source page rotated clockwise 90 degrees for a visual-only correction example](../results/crnn_improvement_experiments/BM_015_orientation_cw90.png)

![BM_015 line boundaries after the visual-only correction](../results/crnn_improvement_experiments/BM_015_C_visual_orientation_plus_aspect_segmentation.png)

## Segmentation pilot and limitation

BM_021's line count remained 30 under B and D; all per-page scores matched. The optional column splitter found no full-height whitespace gutter meeting its conservative thresholds because table rules and content cross the page. It correctly used the fallback rather than guessing column boundaries. This means D is a negative result on these pages. The existing TABLE-region cell path remains available when the layout detector provides table regions, but this layout-off experiment deliberately did not use detector regions. No fresh layout-on test was run because ultralytics is unavailable.

![Existing projection segmentation on BM_021](../results/recognition_diagnostics/BM_021_page_projection_lines.png)

![Conservative column strategy overlay on BM_021; no safe gutter was accepted](../results/crnn_improvement_experiments/BM_021_D_columns_plus_aspect_segmentation.png)

## Tests, errors, and unavailable resources

- Focused regression tests: 22 passed (rotation and coordinate mapping; optional/fail-closed orientation; synthetic two-column segmentation and fallback; widths below/above 512; preserved aspect and right padding; CTC lengths matched to actual CRNN output geometry; padded batch decoding; empty and too-wide crops; exact capped tensor behavior).
- Complete repository suite: 84 passed.
- The three-page pilot completed all variants with zero empty/failed crops. The 24-page A/B/D comparison completed all 72 page/variant evaluations with zero empty/failed crops.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4.0 GiB. Full-run maximum CUDA allocation was 114.09 MB; no out-of-memory error occurred.
- Automated orientation provider: blocked because the pytesseract wrapper cannot find Tesseract. Exact runtime error: “tesseract is not installed or it's not in your PATH. See README file for more information.”
- TrOCR fallback warning: “No module named 'transformers'”. Devanagari line crops were handled by CRNN.
- Fresh detector/layout-on comparison: blocked because ultralytics is not installed. The historical cached YOLO detections are not a substitute.
- The full run initially raised NameError: name 'variant' is not defined in the console-summary formatter after writing the complete results JSON. The formatter was fixed; the shorter pilot was rerun successfully through the corrected entry point. The saved full results JSON was inspected and used for the table above.

No region-level CER was computed because the benchmark GT is page-level and does not provide reliable crop-to-region transcription alignment. The 24-page comparison uses layout-off and therefore cannot isolate layout detector errors. CER can exceed 1 when insertion-heavy output is longer than the reference.

## Conclusions and next steps

1. Keep aspect-preserving mode opt-in. Its 24-page layout-off result is materially better in corpus CER but slower and has severe page-specific regressions, including BM_017. Investigate sequence-length behavior, extreme crop segmentation, and line-level errors with trusted labels before changing production defaults.
2. Improve the evaluation slice with verified line/region references and page-disjoint real-document data. The current page text references cannot identify crop-level error sources; heuristic GovDoc crop labels are not trusted for accuracy scoring.
3. Add a reliable orientation provider (Tesseract OSD in an environment with executable and OSD data, or another validated classifier), then test it on a representative labeled orientation set. Keep the current stage disabled until validated.
4. Improve table-aware segmentation using actual detected table regions and cell geometry. The no-layout column heuristic conservatively fell back on the benchmark; it did not improve results.
5. Reassess detector calibration after a fresh layout-on comparison with the exact checkpoint and annotations. Do not use the mismatched historical detection cache.
6. Benchmark Surya only after the same page and region evaluation protocol is established.

## Reproduction commands

From D:\AksharDrishti_1\aksharDrishti:

    $env:PYTHONDONTWRITEBYTECODE='1'
    python eval/run_crnn_improvement_experiments.py
    python eval/run_crnn_improvement_experiments.py --full
    python -m pytest -p no:cacheprovider --basetemp .pytest-tmp tests/ -q

The pilot command evaluates BM_001, BM_015, and BM_021 under A/B/C/D. The full command evaluates A/B/D on every matching BM_*.png and BM_*.txt pair; C remains a single-page visual-only BM_015 experiment. JSON and visual overlays are written beneath results/crnn_improvement_experiments/. No benchmark source image or transcription is modified.

## Modified source and report files

- configs/pipeline.yaml
- akshardrishti/preprocess/enhance.py
- akshardrishti/layout/lines.py
- akshardrishti/pipeline.py
- akshardrishti/recognize/crnn.py
- eval/run_crnn_improvement_experiments.py
- tests/test_recognition_experiments.py
- docs/CRNN_IMPROVEMENT_EXPERIMENTS.md
