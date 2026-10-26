# Corrected BM benchmark results

Date: 2026-10-26. This report distinguishes fresh inference from saved-prediction rescoring.

## BM_018 reference resolution

The same-stem file `benchmark/gt_text/BM_018.txt` transcribes the preceding photographed page (BM_017). The repository contains `trans/transcription_10b.txt`, which was visually checked against BM_018 and matches its text. A separate v2 mapping manifest was created at `benchmark/mappings/benchmark_image_gt_mapping_v2.csv`; v1 remains untouched and was used for this 23-page run. BM_018 is not in the scored set below. No original ground truth was edited.

## Fresh inference run

Fresh inference completed for 23 approved pages with layout disabled for AksharDrishti and full-page Tesseract. BM_018 was excluded by v1 `exclude-unapproved-mappings`. Per-page inference errors: none for either system. The runner exited with code 1 only after saving result/prediction artifacts, while printing the final summary: Windows cp1252 could not encode `↓` (`UnicodeEncodeError: 'charmap' codec can't encode character '\u2193' in position 17: character maps to <undefined>`). The saved JSON files were parsed and validated.

Environment: CRNN checkpoint `weights/recognize/crnn_mozhi_himr.pt` and layout checkpoint `weights/layout/yolo11l_indicdlp.pt` exist. Tesseract 5.5.3.20260724 was available at `C:\Program Files\Tesseract-OCR\tesseract.exe`, with eng/hin/mar data. `ultralytics` and `doclayout_yolo` are missing, so layout-enabled inference could not run (`ModuleNotFoundError: No module named 'ultralytics'`). `transformers` is also missing; pipeline logs that TrOCR is unavailable and returns empty strings for Latin text. The CRNN/Tesseract evaluation completed, but Latin-routed content is a limitation of the AksharDrishti result.

| System | Fresh pages | Corpus CER | Corpus WER | Mean seconds/page | Errors |
|---|---:|---:|---:|---:|---:|
| AksharDrishti (layout OFF) | 23 | 0.821057 | 0.999048 | 2.42 | 0 |
| Tesseract full-page | 23 | 0.347054 | 0.684085 | 2.87 | 0 |

All scores below use fresh predictions generated in this run. Reference mapping is the exact approved mapping from `benchmark/mappings/benchmark_image_gt_mapping_v1.csv`; see `reference_mapping` in `results.json` for all 23 explicit image-to-file pairs. BM_018 is excluded.

### Per-page scores

#### AksharDrishti layout OFF

| Page | Reference | CER | WER |
|---|---|---:|---:|
| BM_001 | `benchmark\gt_text\BM_001.txt` | 0.506338 | 0.995192 |
| BM_002 | `benchmark\gt_text\BM_002.txt` | 0.533511 | 0.996324 |
| BM_003 | `benchmark\gt_text\BM_006.txt` | 0.783797 | 0.996622 |
| BM_004 | `benchmark\gt_text\BM_003.txt` | 0.907424 | 0.997050 |
| BM_005 | `benchmark\gt_text\BM_005.txt` | 0.746440 | 1.000000 |
| BM_006 | `benchmark\gt_text\BM_009.txt` | 0.904545 | 1.000000 |
| BM_007 | `benchmark\gt_text\BM_010.txt` | 0.721744 | 1.000000 |
| BM_008 | `benchmark\gt_text\BM_017.txt` | 0.705061 | 1.000000 |
| BM_009 | `benchmark\gt_text\BM_004.txt` | 0.600196 | 1.000000 |
| BM_010 | `benchmark\gt_text\BM_007.txt` | 0.717333 | 1.000000 |
| BM_011 | `benchmark\gt_text\BM_008.txt` | 0.787120 | 1.000000 |
| BM_012 | `benchmark\gt_text\BM_012.txt` | 0.662031 | 1.000000 |
| BM_013 | `benchmark\gt_text\BM_013.txt` | 0.975904 | 1.000000 |
| BM_014 | `benchmark\gt_text\BM_014.txt` | 0.932704 | 1.000000 |
| BM_015 | `benchmark\gt_text\BM_015.txt` | 0.986969 | 1.000000 |
| BM_016 | `benchmark\gt_text\BM_016.txt` | 0.879913 | 0.995392 |
| BM_017 | `benchmark\gt_text\BM_018.txt` | 0.908646 | 1.000000 |
| BM_019 | `benchmark\gt_text\BM_019.txt` | 0.857661 | 0.993548 |
| BM_020 | `benchmark\gt_text\BM_020.txt` | 0.810000 | 1.000000 |
| BM_021 | `benchmark\gt_text\BM_021.txt` | 0.851782 | 0.994118 |
| BM_022 | `benchmark\gt_text\BM_022.txt` | 0.826524 | 1.000000 |
| BM_023 | `benchmark\gt_text\BM_023.txt` | 0.881890 | 1.000000 |
| BM_024 | `benchmark\gt_text\BM_024.txt` | 0.824188 | 1.000000 |

#### Tesseract full-page

| Page | Reference | CER | WER |
|---|---|---:|---:|
| BM_001 | `benchmark\gt_text\BM_001.txt` | 0.019346 | 0.072115 |
| BM_002 | `benchmark\gt_text\BM_002.txt` | 0.110106 | 0.187500 |
| BM_003 | `benchmark\gt_text\BM_006.txt` | 0.085487 | 0.165541 |
| BM_004 | `benchmark\gt_text\BM_003.txt` | 0.223581 | 0.386431 |
| BM_005 | `benchmark\gt_text\BM_005.txt` | 0.279847 | 0.360714 |
| BM_006 | `benchmark\gt_text\BM_009.txt` | 0.202841 | 0.314050 |
| BM_007 | `benchmark\gt_text\BM_010.txt` | 0.179975 | 0.319149 |
| BM_008 | `benchmark\gt_text\BM_017.txt` | 0.114311 | 0.273585 |
| BM_009 | `benchmark\gt_text\BM_004.txt` | 0.134409 | 0.169742 |
| BM_010 | `benchmark\gt_text\BM_007.txt` | 0.007467 | 0.037815 |
| BM_011 | `benchmark\gt_text\BM_008.txt` | 0.013417 | 0.046358 |
| BM_012 | `benchmark\gt_text\BM_012.txt` | 0.016690 | 0.083333 |
| BM_013 | `benchmark\gt_text\BM_013.txt` | 0.622080 | 0.914420 |
| BM_014 | `benchmark\gt_text\BM_014.txt` | 1.872327 | 4.353125 |
| BM_015 | `benchmark\gt_text\BM_015.txt` | 0.946379 | 0.991448 |
| BM_016 | `benchmark\gt_text\BM_016.txt` | 0.228722 | 0.304147 |
| BM_017 | `benchmark\gt_text\BM_018.txt` | 0.094943 | 0.191441 |
| BM_019 | `benchmark\gt_text\BM_019.txt` | 0.213962 | 0.464516 |
| BM_020 | `benchmark\gt_text\BM_020.txt` | 0.225556 | 0.404762 |
| BM_021 | `benchmark\gt_text\BM_021.txt` | 0.305816 | 0.482353 |
| BM_022 | `benchmark\gt_text\BM_022.txt` | 0.326859 | 0.447826 |
| BM_023 | `benchmark\gt_text\BM_023.txt` | 1.000000 | 1.000000 |
| BM_024 | `benchmark\gt_text\BM_024.txt` | 0.354983 | 0.424658 |

## Comparison with earlier results

The fresh layout-off CRNN score (CER 0.821057, WER 0.999048) agrees with the corrected 23-page legacy saved-prediction rescoring (CER 0.821057; WER 0.999048). The agreement supports reproducibility of the legacy path, but the earlier numbers were rescoring, not inference. The aspect-preserving and columns + aspect-preserving values (CER 0.453389) remain saved-prediction rescoring only; neither was freshly inferred here. Earlier same-stem 24-page scores are not directly comparable because they used different page/reference pairing and page coverage.

Fresh Tesseract on the corrected approved 23-page mapping measured CER 0.347054 and WER 0.684085. The prior baseline (CER 0.67230, WER 0.95886) used the earlier 24-page context and is not an apples-to-apples comparison.

## Output discrepancy inspection

The CRNN sample BM_001 has CER 0.506338 and WER 0.995192 (reference/prediction lengths 1,507/1,003 characters). Its opening contains recognizable Marathi phrases but character substitutions/spacing split words, explaining how partial character overlap can coexist with almost no exact word matches. BM_002 similarly has CER 0.533511 and WER 0.996324; visible output includes glyph substitutions and spacing/line-format differences. This is evidence of recognition/formatting mismatch, but does not isolate a single cause.

Tesseract BM_014 is the strongest outlier: CER 1.872305, WER 4.353147, reference/prediction lengths 1,826/4,159 characters. The source is a dense landscape financial statement/table; output inflation and table ordering/formatting are plausible contributors, but the metric alone cannot distinguish these from repeated recognition errors. BM_013 also has lower CER than WER (0.6221 vs 0.9144), consistent with imperfect word segmentation/spacing. No automatic correction or reference changes were applied.

## Artifacts

- Results: `results/bm_corrected_benchmark_20261026_v1/layout_off/results.json`
- AksharDrishti predictions: `results/bm_corrected_benchmark_20261026_v1/layout_off/predictions_akshardrishti.json`
- Tesseract predictions: `results/bm_corrected_benchmark_20261026_v1/layout_off/predictions_tesseract_fullpage.json`
- Run config copy (layout disabled): `results/bm_corrected_benchmark_20261026_v1/pipeline_layout_off.yaml`
- New resolved 24-page manifest: `benchmark/mappings/benchmark_image_gt_mapping_v2.csv` (BM_018 points to existing `trans/transcription_10b.txt`); v1 untouched.

## Reproduction

The actual run used:

```powershell
$env:PATH = 'C:\Program Files\Tesseract-OCR;' + $env:PATH
python eval/run_benchmark.py --benchmark benchmark --config results/bm_corrected_benchmark_20261026_v1/pipeline_layout_off.yaml --mapping-manifest mappings/benchmark_image_gt_mapping_v1.csv --exclude-unapproved-mappings --systems akshardrishti tesseract_fullpage --out results/bm_corrected_benchmark_20261026_v1/layout_off
```

For a future layout-enabled run, install nothing as part of this report; first provision the project’s declared `ultralytics` dependency, then use the normal layout-enabled config and a fresh, unused output path. To avoid the summary printer’s Windows encoding issue, run with `$env:PYTHONIOENCODING = 'utf-8'` or patch the console formatting separately. Do not reuse the existing output directory.

## Tests and limitations

- Full suite: `python -m pytest -q --basetemp .pytest-tmp/bm_corrected_full_20261026` — **97 passed in 182.68s**.
- Focused mapping tests after v2 support: `python -m pytest -q tests/test_benchmark_pairing_manifest.py` — **6 passed in 1.77s**.
- Layout ON unavailable because `ultralytics` is not installed. No package installation, training, or weight download was performed.
- Tesseract did run. AksharDrishti’s Latin routed text was limited by missing `transformers`/TrOCR.
- BM_018 is resolved by an existing repository transcription in v2, but it remains excluded from the completed run because the requested run was explicitly the approved 23-page v1 subset. A separate full 24-page evaluation requires a new run using v2 and a new output directory.
