# STATUS_PHASE1.md - Environment Setup and Baseline Test Results

## Summary

**Total tests:** 58  
**Passed:** 56  
**Failed:** 2

### Failure Categories
- **OK:** 56
- **EXPECTED-FAIL-missing-weights:** 0
- **EXPECTED-FAIL-missing-data:** 0
- **EXPECTED-FAIL-optional-backend:** 0
- **CODE-BUG:** 2

### Tests Tagged CODE-BUG

1. **test_preprocess_full_chain**
   - **Error:** `TypeError: cannot unpack non-iterable numpy.int32 object` in `enhance.py` line 75
   - **Full traceback:**
     ```
     File "D:\AksharDrishti_1\aksharDrishti\akshardrishti\preprocess\enhance.py", line 217, in preprocess_page
         out, rotation = deskew(out, max_angle=float(ds.get("max_angle", 15.0)))
     File "D:\AksharDrishti_1\aksharDrishti\akshardrishti\preprocess\enhance.py", line 110, in deskew
         angle = estimate_skew_hough(gray, max_angle=max_angle)
     File "D:\AksharDrishti_1\aksharDrishti\akshardrishti\preprocess\enhance.py", line 75, in estimate_skew_hough
         for x1, y1, x2, y2 in lines[:, 0]:
             ^^^^^^^^^^^^^^
     TypeError: cannot unpack non-iterable numpy.int32 object
     ```
   - **Analysis:** The code assumes `lines` from `cv2.HoughLinesP()` returns shape `(N, 1, 4)` but newer OpenCV versions (5.0.0.93 is installed) return shape `(N, 4)` directly. The unpacking `lines[:, 0]` produces individual numpy.int32 values instead of 4-tuples. This is a genuine code bug caused by OpenCV API changes.

2. **test_skew_estimation_recovers_known_angle**
   - **Error:** `TypeError: cannot unpack non-iterable numpy.int32 object` in `enhance.py` line 75
   - **Full traceback:**
     ```
     File "D:\AksharDrishti_1\aksharDrishti\tests\test_pipeline_units.py", line 162, in test_skew_estimation_recovers_known_angle
         estimated = estimate_skew_hough(skewed)
     File "D:\AksharDrishti_1\aksharDrishti\akshardrishti\preprocess\enhance.py", line 75, in estimate_skew_hough
         for x1, y1, x2, y2 in lines[:, 0]:
             ^^^^^^^^^^^^^^
     TypeError: cannot unpack non-iterable numpy.int32 object
     ```
   - **Analysis:** Same root cause as test 1. This is a code defect, not a missing dependency or expected failure.

### Statement of Integrity
**0 tests were modified, skipped, or deleted to make this pass.**

All 58 tests were executed exactly as written. The two failures are genuine code bugs related to OpenCV API compatibility, not environmental issues, missing weights, or missing data.

---

## Detailed Test Results

| Test name | Result | Category | Reason |
|-----------|--------|----------|--------|
| test_bbox_clip | PASS | OK | - |
| test_bbox_geometry | PASS | OK | - |
| test_bbox_iou | PASS | OK | - |
| test_bbox_rejects_degenerate | PASS | OK | - |
| test_compose_text_averages_confidence | PASS | OK | - |
| test_document_roundtrip | PASS | OK | - |
| test_region_type_flags | PASS | OK | - |
| test_cli_overrides | PASS | OK | - |
| test_dotted_access_and_override | PASS | OK | - |
| test_hash_changes_with_content | PASS | OK | - |
| test_ignore_returns_none | PASS | OK | - |
| test_resolution_is_case_and_separator_insensitive | PASS | OK | - |
| test_targets_are_eleven | PASS | OK | - |
| test_unmapped_fails_loudly | PASS | OK | - |
| test_unmapped_report | PASS | OK | - |
| test_clahe_preserves_shape | PASS | OK | - |
| test_line_crop_normalises_height | PASS | OK | - |
| test_preprocess_full_chain | FAIL | CODE-BUG | TypeError in estimate_skew_hough line 75: OpenCV 5.x returns HoughLinesP output as (N,4) not (N,1,4), breaking line unpacking |
| test_preprocess_rejects_empty | PASS | OK | - |
| test_preprocess_respects_max_side | PASS | OK | - |
| test_skew_estimation_recovers_known_angle | FAIL | CODE-BUG | TypeError in estimate_skew_hough line 75: same OpenCV compatibility issue as test_preprocess_full_chain |
| test_skew_returns_zero_on_blank | PASS | OK | - |
| test_crop_and_offset_roundtrip | PASS | OK | - |
| test_empty_input | PASS | OK | - |
| test_finds_all_lines | PASS | OK | - |
| test_ink_too_short_for_a_line_falls_back_to_whole_crop | PASS | OK | - |
| test_lines_are_vertically_ordered_and_disjoint | PASS | OK | - |
| test_truly_blank_region_yields_no_lines | PASS | OK | - |
| test_accuracy_metric | PASS | OK | - |
| test_empty_page | PASS | OK | - |
| test_furniture_always_last | PASS | OK | - |
| test_single_column_is_topdown | PASS | OK | - |
| test_two_column_order | PASS | OK | - |
| test_clean_text_is_idempotent | PASS | OK | - |
| test_empty_input | PASS | OK | - |
| test_matra_reattachment | PASS | OK | - |
| test_nukta_canonicalisation_makes_forms_equal | PASS | OK | - |
| test_orphan_marks_dropped | PASS | OK | - |
| test_scoring_normalisation_collapses_encoding_noise | PASS | OK | - |
| test_scoring_normalisation_does_not_repair_errors | PASS | OK | - |
| test_shirorekha_split_repair | PASS | OK | - |
| test_digits_and_punctuation_are_neutral | PASS | OK | - |
| test_empty | PASS | OK | - |
| test_mixed | PASS | OK | - |
| test_pure_devanagari | PASS | OK | - |
| test_pure_latin | PASS | OK | - |
| test_bootstrap_ci_brackets_the_mean | PASS | OK | - |
| test_cer_basic | PASS | OK | - |
| test_cer_empty_reference | PASS | OK | - |
| test_corpus_cer_is_length_weighted | PASS | OK | - |
| test_edit_ops_decomposition | PASS | OK | - |
| test_layout_map_no_overlap | PASS | OK | - |
| test_layout_map_perfect_prediction | PASS | OK | - |
| test_wer | PASS | OK | - |
| test_backends_register | PASS | OK | - |
| test_unknown_backend_raises | PASS | OK | - |
| test_low_confidence_tries_both | PASS | OK | - |
| test_routes_by_script | PASS | OK | - |

---

## Environment Details

### System Information
- **OS:** Windows 11 (win32)
- **Python:** 3.12.10
- **GPU:** NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
- **CUDA:** 13.1 (driver 591.86)

### Package Versions
- **torch:** 2.5.1+cu121 ✅
- **torchvision:** 0.20.1+cu121 ✅
- **transformers:** 5.17.0 ✅
- **numpy:** 2.5.2 ✅
- **opencv-python:** 5.0.0.93 ⚠️ (API change causing test failures)
- **ultralytics:** 8.4.153 ✅
- **sahi:** 0.12.6 ✅
- **datasets:** 5.0.1 ✅
- **huggingface-hub:** 1.31.0 ✅
- **pycocotools:** 2.0.11 ✅
- **streamlit:** 1.63.0 ✅
- **wandb:** 0.30.0 ✅
- **scipy:** 1.18.1 ✅
- **pandas:** 3.0.5 ✅
- **pydantic:** 2.13.5 ✅
- **jiwer:** (installed via transformers) ✅

### Changes Applied
1. **requirements.txt:** Updated `torch>=2.2` → `torch>=2.5.0` and `torchvision>=0.17` → `torchvision>=0.20.0` for transformers compatibility and NumPy 2.x support
2. **Steering file:** Added CRITICAL RULE 5 regarding 4GB VRAM hardware constraint

### Git Status
- Commits: 2 total
  1. `e21d6a5` - Initial commit: codebase + steering file, pre-training baseline
  2. `f2a11ec` - Fix torch/transformers/numpy compatibility, add hardware constraint rule

---

## Critical Findings

### Code Bugs Requiring Fixes
Two tests fail due to an OpenCV API compatibility issue in `akshardrishti/preprocess/enhance.py`:

**File:** `akshardrishti/preprocess/enhance.py`  
**Line:** 75  
**Issue:** `cv2.HoughLinesP()` return shape changed from `(N, 1, 4)` to `(N, 4)` in OpenCV 5.x

**Current code:**
```python
for x1, y1, x2, y2 in lines[:, 0]:
```

**Needed fix:**
```python
# Handle both old and new OpenCV API
if lines.ndim == 3:
    lines = lines[:, 0]  # Old API: (N, 1, 4) -> (N, 4)
for x1, y1, x2, y2 in lines:
```

This affects the deskewing functionality which is critical for the preprocessing pipeline. The fix is straightforward but must be implemented before real document processing can begin.

### No Missing Dependencies
All core dependencies are installed and functional. The environment is complete for:
- ✅ Core pipeline execution (once deskew bug is fixed)
- ✅ Model training on Kaggle/Colab
- ✅ Evaluation metrics computation
- ✅ Streamlit demo

### No Missing Weights Expected
Zero tests failed due to missing model weights. This is correct since:
- No model weights are trained yet (weights/ directory is empty)
- Tests are unit tests for individual components, not end-to-end integration tests
- Model loading and inference will be tested separately after training

### No Missing Data Expected
Zero tests failed due to missing benchmark data. This is correct since:
- Unit tests use synthetic test data generated inline
- Benchmark evaluation (which requires benchmark/images/ and benchmark/gt_text/) is not part of the unit test suite
