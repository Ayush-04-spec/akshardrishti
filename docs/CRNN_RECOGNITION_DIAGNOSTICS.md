# CRNN real-document recognition diagnostics

**Scope:** read-only diagnosis of current recognition behavior plus diagnostic artifacts. No model was trained, no model weights or data were downloaded, and no production defaults or benchmark ground truth were changed. The 24-page full benchmark was not rerun.

## Summary

The evidence shows two distinct problems. First, Mozhi training/test crops are overwhelmingly short word-scale crops, while real government-document lines are much wider and contain far more characters. In 423 original GovDoc crops, median width after height-32 normalization was 628 px and 56.26% exceeded the production 512 px width cap. On the three inspected pages, 63 of 82 line crops exceeded that cap. A diagnostic comparison with the same checkpoint and segmentation but without the cap reduced page CER on BM_001 from 0.5063 to 0.1688 and BM_021 from 0.8518 to 0.6379. BM_015 only moved from 0.9870 to 0.9812 because its page is sideways and the projection line segmentation is not orientation-aware. This indicates the cap is a real, measurable contributor, but not the only cause.

Second, page structure matters independently. Projection-based segmentation of a complex table merges content across columns, while no reliable line-level transcription reference exists for this benchmark to isolate that contribution numerically. BM_015's orientation failure is visually clear. The current results therefore do not support attributing the high full-document CER solely to CRNN recognition or solely to layout.

## Current diagnostic setup and findings

### Crop distributions

The profiling script uses deterministic seed 20261026: 10,000 indexed Mozhi train rows and 5,000 each from validation and test; every indexed GovDoc original line crop (423 rows; augmented duplicates excluded). It reads image dimensions from headers, uses the row text for character counts, then computes height-32 width as rounded aspect-ratio scaling, capped at 512. “Substantial aspect change” means more than 10% relative ratio change. These are profile samples, not full-set measurements for Mozhi.

| Split | Measured | Original width p10 / median / p90 | Original height median | Height-32 width before cap p10 / median / p90 | Characters median / p90 | >512 | Aspect change >10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mozhi train | 10,000 | 35 / 88 / 167 px | 48 px | 23 / 59 / 111 px | 4 / 8 | 0.00% | 2.84% |
| Mozhi validation | 5,000 | 33 / 86 / 164 px | 48 px | 22 / 57 / 109 px | 4 / 8 | 0.00% | 3.08% |
| Mozhi test | 5,000 | 36 / 89 / 169 px | 48 px | 24 / 59 / 112 px | 4 / 8 | 0.00% | 2.92% |
| GovDoc originals | 423 | 144 / 504 / 1,650.2 px | 22 px | 203 / 628 / 1,474.4 px | 42 / 94 | 56.26% | 54.14% |

Mozhi index totals are 159,913 train, 20,119 validation, and 20,028 test rows. Sampled normalized-width buckets (short ≤128, medium 129–256, long 257–512, over-limit >512) were train 9,487 / 512 / 1 / 0; validation 4,747 / 253 / 0 / 0; test 4,738 / 262 / 0 / 0. GovDoc originals were 15 / 60 / 110 / 238. The GovDoc fine-tuning index has 2,115 rows, including 1,692 augmented rows; all rows are marked train. Its crop labels were generated heuristically by distributing page-level GT lines across regions according to detected line counts in data/make_govdoc_finetune.py, so those labels are not used here for recognition accuracy.

![Normalized crop width distributions](../results/recognition_diagnostics/crop_width_distribution.png)

![Crop aspect ratio distributions](../results/recognition_diagnostics/aspect_ratio_distribution.png)

### Labeled CRNN bucket evaluation

The existing checkpoint and current preprocessing/beam decoder were run on 24 ground-truth-labeled, charset-eligible samples from each available bucket in a seeded subset of the Mozhi test index. The diagnostic aspect-preserving option only differs above 512 px, so results match exactly for short and medium buckets. No eligible held-out samples were available in the long or over-512 buckets; no result is claimed for those ranges.

| Normalized-width bucket | n | Current CER | Aspect-preserved CER | Current WER | Aspect-preserved WER | Exact word accuracy (both) |
|---|---:|---:|---:|---:|---:|---:|
| Short ≤128 | 24 | 0.00752 | 0.00752 | 0.04167 | 0.04167 | 0.95833 |
| Medium 129–256 | 24 | 0.00897 | 0.00897 | 0.04167 | 0.04167 | 0.95833 |
| Long 257–512 | 0 | unavailable | unavailable | unavailable | unavailable | unavailable |
| Over 512 | 0 | unavailable | unavailable | unavailable | unavailable | unavailable |

![Mozhi held-out crop CER by width bucket](../results/recognition_diagnostics/mozhi_bucket_cer.png)

The poor benchmark page CER is therefore not reproduced by these short/medium Mozhi crops. The absence of long labeled test crops is a material generalization gap, not evidence that the long-line model is accurate.

### Fixed real-page sample

Pages BM_001, BM_015, and BM_021 were selected before inference and processed with the existing benchmark page GT and layout disabled. For the alternate only, the script retained the same page preprocessing, line crops, order, checkpoint, height-32 grayscale/binarization, normalization, right padding, and beam; it omitted the horizontal cap on over-limit CRNN line inputs. The CRNN accepts variable input widths and its convolutional/recurrent output time dimension grows with width. The diagnostic computes that item's CTC length from its actual padded tensor width. It was feasible to compare without changing production preprocessing.

| Page | Lines | Nonempty / empty | Lines >512 | Current page CER / WER | Uncapped diagnostic CER / WER |
|---|---:|---:|---:|---:|---:|
| BM_001 | 31 | 31 / 0 | 21 | 0.50634 / 0.99519 | 0.16878 / 0.99038 |
| BM_015 | 21 | 21 / 0 | 13 | 0.98697 / 1.00000 | 0.98120 / 0.99943 |
| BM_021 | 30 | 30 / 0 | 29 | 0.85178 / 0.99412 | 0.63790 / 0.99412 |

The alternate CER values are page-level comparisons on this small, fixed sample. They do not represent a rerun of the full benchmark and should not replace the reported 24-page scores. Per-region CER is unavailable because the benchmark supplies page transcription, not reliable matching region/line transcripts. There were no empty predictions on these three pages; BM_015 produced only 159 characters against 4,688 GT characters, consistent with severe orientation/segmentation failure rather than empty-region inflation.

![BM_001 original page with GT layout and historical cached predictions](../results/recognition_diagnostics/BM_001_page_gt_and_historical_detections.png)

![BM_015 projection line segmentation](../results/recognition_diagnostics/BM_015_page_projection_lines.png)

![BM_021 projection line segmentation](../results/recognition_diagnostics/BM_021_page_projection_lines.png)

The detections in the page overlay are historical cached predictions from benchmark/prelabel_meta/detections_20261017_203514.csv, not fresh YOLO inference. The prelabel script records expected checkpoint SHA-256 5A927BAF51EE9EF3DDEA1DFFDAE954C73869A17B0614836F8856AA77B6AD4AED; the current checkpoint hashes to C7B4F755A132E10F66656F575C9159493F660641B096B277A4C33CF9DB999F8, so that cache is not checkpoint-verified. Fresh detector inference could not run because ultralytics is absent.

Representative observed errors on BM_001 include matra/conjunct/glyph substitutions and spacing loss: GT “महाराष्ट्र शासन” was read as “मब्ाराष्टशासन”; GT “शालेय शिक्षण व क्रीडा विभाग” as “शालयाशेक्षणवक्रीडाविभाग”. BM_015's page image is rotated 90 degrees; its lines are consequently not horizontal text lines for projection segmentation. BM_021 is a dense multi-column/table page, while the current projection pass operates over a full region and merges content across columns. Sampled line crops, line indices, dimensions, raw/current/alternate text, and page transcripts are in results/recognition_diagnostics/government_page_samples.json and diagnostics.json.

## Code path and verified behavior

- train/dataset.py:24-45,105-119,122-140: the CTC dataset is described and implemented as height-normalized line/word crops; it scales to configured height, caps width at 512, then right-pads in ctc_collate. Ground-truth text is NFC-normalized when loaded.
- akshardrishti/recognize/crnn.py:65-78: inference uses preprocess_line_crop, resizes an over-width crop directly to (512, 32) (distorting its aspect ratio), normalizes grayscale values, and right-pads 64 columns by default. Lines 119-132 computes per-item CTC lengths for batching; single-item inference decodes the model's full output length. The model's time dimension is width-dependent; the forward path is not intrinsically fixed to width 512.
- akshardrishti/preprocess/enhance.py:188-241: page preprocessing, small-angle deskew, and line-crop processing are separate stages. Deskew is designed around small skew angles and does not correct a 90-degree page orientation.
- akshardrishti/pipeline.py:183-242: recognized text is produced after region/line segmentation and routed per line. Page assembly depends on region ordering, line ordering, table handling, and text cleanup; transcription errors can therefore precede CRNN recognition.
- akshardrishti/layout/lines.py: projection segmentation is the configured line fallback used by the diagnostic. Its horizontal projection cannot by itself preserve table columns or recover a rotated reading direction.
- data/make_govdoc_finetune.py:79-157: constructs GovDoc crops and heuristically aligns page-level transcription text to regions; the resulting crop labels cannot serve as trusted per-line evaluation references without manual/alignment validation.
- eval/diagnose_crnn_recognition.py: diagnostic-only sampler, profile, bucket inference, fixed-page inference, outputs and figures. It does not edit production settings or ground truth.

The implementation issue is confirmed: both training and inference have a 512 px cap, but the Mozhi sample has no over-cap crop whereas more than half the original GovDoc line crops exceed it. For such inference lines, resizing to fixed 512×32 changes glyph proportions. The page-level comparisons suggest that preserving width can help some upright pages, but cannot repair orientation, poor region/line boundaries, or model/domain mismatch.

## Previous headline scores

Previously reported full-document values remain the recorded benchmark results: AksharDrishti layout-on CER 0.8058 / WER 0.997, layout-off CER 0.851, Tesseract CER 0.6723 / WER 0.959, and real-document YOLO mAP@50 0.275. The full benchmark was not rerun during this diagnostic, and these values are not invalidated by the small sample. They should be labeled as prior results pending a controlled full evaluation after choosing and validating the next engineering changes. No accuracy improvement claim is made.

The audit's earlier 62-test result was reproduced: 62 passed in 6.16s. This run used PYTHONDONTWRITEBYTECODE=1, pytest's cache plugin disabled, and .pytest-tmp as its temporary directory. Diagnostic script compilation also succeeded. The script logged that TrOCR was unavailable (No module named 'transformers'); for these Devanagari pages the routed CRNN output was used. Cleanup also reported "'TrOCRRecognizer' object has no attribute 'model'"; this affects only the diagnostic's attempted router unload and did not prevent measurements or modify production behavior.

## Recommended next engineering change, ranked

1. **Build orientation-aware, region/table-aware segmentation and a reliable per-line evaluation slice.** BM_015 directly demonstrates the 90-degree orientation failure; BM_021 demonstrates the limits of whole-region projection on tabular columns. Fixing boundaries and order makes any recognizer comparison interpretable. Capture verified line/region transcripts before using the GovDoc crop set for accuracy claims.
2. **Prototype aspect-preserving CRNN inputs with padding and correct CTC lengths.** The measured cap mismatch and the two upright-page diagnostics justify this next controlled experiment. Keep it behind an opt-in diagnostic/config path until validated on enough ground-truth-labeled, long real-document lines. Do not treat this small three-page result as a deployment decision.
3. **Create a trustworthy real-document recognition training/validation set** with manually checked crop-to-text alignment and page-disjoint split. Existing GovDoc labels are heuristic and currently all marked train, so are not a valid evaluation set.
4. **Calibrate the layout detector** only after fresh inference can run with the exact checkpoint and matching annotations. The historical cache does not match the current weight hash.
5. **Benchmark Surya** after fixing orientation/segmentation and establishing the same trusted evaluation protocol; adding another recognizer before these controls would not resolve attribution.

## Reproduction

From the application directory D:\AksharDrishti_1\aksharDrishti, run:

    $env:PYTHONDONTWRITEBYTECODE='1'
    python eval/diagnose_crnn_recognition.py --max-profile-train 10000 --max-per-bucket 24
    python -m pytest -p no:cacheprovider --basetemp .pytest-tmp tests/ -q

The diagnostic requires the existing Mozhi indexes/images, GovDoc original crops/index, benchmark page images/transcriptions, and configured CRNN checkpoint. It writes outputs beneath results/recognition_diagnostics/. No data or weights are fetched. Fresh YOLO inference was unavailable because ultralytics is not installed; TrOCR was unavailable because transformers is not installed. No reliable per-region ground truth exists in the sampled benchmark. The output plots and JSON/CSV preserve the measured crop and prediction details.
