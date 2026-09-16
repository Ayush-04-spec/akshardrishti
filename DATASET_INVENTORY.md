# Dataset Inventory

**Date:** 2026-09-15  
**Purpose:** Read-only reconnaissance of downloaded datasets for AksharDrishti OCR pipeline

---

## IndicDLP

### Location
`d:\AksharDrishti_1\indicdlp.tar`

### Status
**NOT EXTRACTED** - The dataset exists as a compressed tar archive only.

### Size on Disk
79.95 GB (85,840,435,200 bytes)

### File Metadata
- **Created:** 2026-09-14 23:35:50
- **Last Modified:** 2026-09-15 22:48:35

### Contents
**CANNOT BE INVENTORIED** - The tar file has not been extracted. Extracting an 80GB archive is beyond the scope of read-only reconnaissance.

### Expected Structure (from paper/documentation)
IndicDLP is a layout analysis dataset containing:
- 119,806 images across 12 Indian languages
- 42 layout annotation classes (COCO-style bounding boxes)
- 12 document domains (acts_and_rules, forms, notices, newspapers, question_papers, etc.)
- Languages include: Hindi, Marathi, English, and 9 others

### Actual Structure
Unknown - requires extraction to inventory.

---

## Mozhi

### Location
Split across 6 directories at project root:
- **Hindi:** `d:\AksharDrishti_1\train\`, `d:\AksharDrishti_1\test\`, `d:\AksharDrishti_1\val\`
- **Marathi:** `d:\AksharDrishti_1\train (1)\`, `d:\AksharDrishti_1\test (1)\`, `d:\AksharDrishti_1\val (1)\`

### Size on Disk
**Total:** 0.57 GB across both languages and all splits

**Breakdown by split:**
- `train/`: 0.18 GB (Hindi)
- `test/`: 0.02 GB (Hindi)
- `val/`: 0.02 GB (Hindi)
- `train (1)/`: (Marathi, included in total)
- `test (1)/`: (Marathi, included in total)
- `val (1)/`: (Marathi, included in total)

### Directory Structure

Each split follows this nested pattern:
```
{split}/
  {split}/
    images/
      *.jpg
      *.jpeg
    {split}_gt.txt
    vocabulary.txt (train only)
```

**Example:**
```
train/
  train/
    images/
      0057-0104_2_3_1.jpeg
      0057-0104_2_3_2.jpeg
      ...
    train_gt.txt
    vocabulary.txt
```

### File Counts

| Split       | Language | Image Count | Non-Image Files |
|-------------|----------|-------------|-----------------|
| train       | Hindi    | 79,762      | 2 (gt.txt, vocabulary.txt) |
| test        | Hindi    | 10,173      | 1 (gt.txt) |
| val         | Hindi    | 10,114      | 1 (gt.txt) |
| train (1)   | Marathi  | 80,151      | 2 (gt.txt, vocabulary.txt) |
| test (1)    | Marathi  | 9,855       | 1 (gt.txt) |
| val (1)     | Marathi  | 10,005      | 1 (gt.txt) |
| **TOTAL**   | Both     | **200,060** | 8 |

### Languages Present
✅ **Both Hindi and Marathi are present**

- **Hindi:** Identified by Devanagari vocabulary in `train/train/vocabulary.txt` with words like "फल", "काटते", "ईरानियों"
- **Marathi:** Identified by Devanagari vocabulary in `train (1)/train/vocabulary.txt` with words like "विवेकानंदांचा", "परतायची", "भारतीयाच्या"

### Image Format
**Format:** JPG/JPEG  
**Structure:** Word-level crops (individual word images, not full page scans)

### Annotation Format
**Format:** Tab-separated text files (`{split}_gt.txt`)

**Schema:** `<relative_image_path>\t<transcription>`

**Sample rows from `train/train/train_gt.txt` (Hindi):**
```
images/0057-0104_2_3_1.jpeg	फल
images/0057-0104_2_3_2.jpeg	काटते
images/0057-0104_2_3_3.jpeg	हैं
```

**Sample rows from `test (1)/test/test_gt.txt` (Marathi):**
```
images/0492-0117_2_30_1.jpeg	बाल
images/0492-0117_2_30_2.jpeg	आवाजात
images/0492-0117_2_30_3.jpeg	चाललेलं
```

### Vocabulary Files
Present in train splits only (`train/train/vocabulary.txt`, `train (1)/train/vocabulary.txt`).  
Contains unique words from the training transcriptions, one per line.

---

## Compatibility with Preparation Scripts

### `data/prepare_indicdlp.py`

**Expected Input:**
- Hugging Face Datasets Hub streaming: `datasets.load_dataset("ai4bharat/indicdlp")`
- **No local file path expected** - downloads and streams from HF Hub
- Requires: `huggingface-cli login` (gated dataset)
- Script uses `streaming=True` to avoid downloading entire 91GB dataset at once

**Expected Structure:**
- Streaming HF dataset with fields: `image`, `language`, `domain`, `objects` (annotations)
- Annotations in COCO-style dict format with `category`, `bbox` fields

**Actual Dataset:**
- `indicdlp.tar` (79.95 GB compressed tar file, not extracted)

**Compatibility Status:**
❌ **MISMATCH**

**Issue:**
1. Script expects to **download from Hugging Face Hub**, NOT read from local files
2. Local tar file is **not extracted** and cannot be inventoried
3. Even if extracted, script has no code path to read local IndicDLP directories
4. Script is designed for **streaming access** to avoid disk space issues (targets Kaggle with limited disk)

**Resolution Required:**
- Either: Use the script as-is to download from HF Hub (requires HF auth + ~90GB network transfer)
- Or: Modify script to read from local extracted tar (requires extraction + code changes)

---

### `data/prepare_mozhi.py`

**Expected Input (from argparse defaults):**
- `--out datasets/mozhi` (output directory, not input)
- `--base-url https://ilocr.iiit.ac.in/public/printed/phase-0/v0.5` (downloads from IIIT-H server)
- Script **downloads** language-specific zips: `{base_url}/{language}/akshara/{split}.zip`

**Expected Structure (after download):**
```
datasets/mozhi/
  hindi/
    train/
    val/
    test/
  marathi/
    train/
    val/
    test/
```

Each split directory should contain:
- Images in a subdirectory (script searches for `images/` or at root)
- An annotation file (searches for: `gt.txt`, `annotation.txt`, `train.txt`, etc.)
- Format: `<path>\t<transcription>` or `<path> <transcription>`

**Actual Dataset:**
```
d:\AksharDrishti_1\
  train/train/images/, train/train/train_gt.txt     [Hindi]
  test/test/images/, test/test/test_gt.txt          [Hindi]
  val/val/images/, val/val/val_gt.txt               [Hindi]
  train (1)/train/images/, train (1)/train/train_gt.txt  [Marathi]
  test (1)/test/images/, test (1)/test/test_gt.txt       [Marathi]
  val (1)/val/images/, val (1)/val/val_gt.txt            [Marathi]
```

**Compatibility Status:**
⚠️ **PARTIAL MISMATCH**

**Issues:**
1. **Location:** Dataset is at **project root** (`d:\AksharDrishti_1\`), not inside `aksharDrishti/datasets/mozhi/`
2. **Directory naming:** Split directories have **Windows duplicate suffixes** (" (1)") for Marathi instead of language names
3. **Nested structure:** Each split has a redundant nested directory (`train/train/`, not `train/`)
4. **Language separation:** Hindi and Marathi are in separate parallel directory trees (train vs train (1)), not under `hindi/` and `marathi/` subdirectories

**What DOES Match:**
✅ Annotation file format: tab-separated `<path>\t<text>` matches expected format  
✅ Annotation file naming: `{split}_gt.txt` matches script's search patterns  
✅ Image format: JPG/JPEG images are supported  
✅ Content structure: images in `images/` subdirectory with gt.txt at split root  

**Resolution Options:**
1. **Use script to re-download** from IIIT-H servers into correct structure (requires network access)
2. **Reorganize manually** to match expected structure (move/rename directories)
3. **Use --skip-download** and point script to current locations (requires path adjustments)
4. **Directly use current structure** by updating training scripts to read from actual paths (bypass prep script)

---

## Summary

| Dataset  | Status | Size   | Files/Images | Location | Script Compatibility |
|----------|--------|--------|--------------|----------|----------------------|
| IndicDLP | ❌ Not extracted | 79.95 GB | Unknown (tar) | `d:\AksharDrishti_1\indicdlp.tar` | ❌ Script expects HF Hub download |
| Mozhi    | ✅ Extracted | 0.57 GB | 200,060 images + 8 txt files | `d:\AksharDrishti_1\{train,test,val,train (1),test (1),val (1)}/` | ⚠️ Structure mismatch (location, naming) |

**Key Findings:**
- **IndicDLP cannot be used** without either extracting the 80GB tar or downloading from Hugging Face Hub
- **Mozhi data is complete** (both languages, all splits) but organized differently than prep script expects
- **No data corruption detected** - all annotation files parse correctly, file counts are consistent
- **Total usable data:** ~200k word-crop images across Hindi and Marathi (Mozhi only)

---

## Tar Contents (Peeked)

**Method:** Read tar table of contents without extraction using `tar -tvf`

### Disk Space Constraint
**D: drive free space:** 17.17 GB  
**IndicDLP tar size:** 79.95 GB  
**❌ EXTRACTION IMPOSSIBLE** - Insufficient disk space (would need ~80GB free to extract safely)

### Total Entries
**118,419 entries** in tar file (directories + files)

### Top-Level Structure
The tar contains a single root directory `indicdlp/` with 4 subdirectories:
- `indicdlp/train2017/` - Training images
- `indicdlp/test2017/` - Test images  
- `indicdlp/val2017/` - Validation images
- `indicdlp/annotations/` - COCO-format JSON annotations

### File Extension Breakdown

| Extension | Count | Purpose |
|-----------|-------|---------|
| .png      | 115,717 | Document page images |
| .json     | 3 | COCO-format annotations (train/test/val) |
| .jpg      | 0 | - |
| .arrow    | 0 | - |
| .parquet  | 0 | - |
| .csv      | 0 | - |
| .txt      | 0 | - |

### First 30 Lines of Tar Listing
```
drwxrwx---  0 ocrteam llmteam     0 Mar 03  2025 indicdlp/
drwxrwx---  0 ocrteam llmteam     0 Mar 03  2025 indicdlp/train2017/
-rwxrwx---  0 ocrteam llmteam 295381 Mar 02  2025 indicdlp/train2017/ar_ta_001508_1.png
-rwxrwx---  0 ocrteam llmteam  43285 Mar 02  2025 indicdlp/train2017/sy_pa_000462_1.png
-rwxrwx---  0 ocrteam llmteam 1165876 Mar 02  2025 indicdlp/train2017/mg_ml_000911_0.png
-rwxrwx---  0 ocrteam llmteam  662085 Mar 02  2025 indicdlp/train2017/rp_as_000977_0.png
-rwxrwx---  0 ocrteam llmteam   94632 Mar 02  2025 indicdlp/train2017/mn_gu_001084_0.png
-rwxrwx---  0 ocrteam llmteam  151876 Mar 02  2025 indicdlp/train2017/mn_bn_000487_0.png
-rwxrwx---  0 ocrteam llmteam  489045 Mar 02  2025 indicdlp/train2017/rp_as_001634_0.png
-rwxrwx---  0 ocrteam llmteam  639119 Mar 02  2025 indicdlp/train2017/fm_mr_000427_0.png
-rwxrwx---  0 ocrteam llmteam   89213 Mar 02  2025 indicdlp/train2017/nv_pa_000302_0.png
-rwxrwx---  0 ocrteam llmteam  377512 Mar 02  2025 indicdlp/train2017/mg_kn_001012_1.png
-rwxrwx---  0 ocrteam llmteam  242683 Mar 02  2025 indicdlp/train2017/mg_bn_000450_0.png
-rwxrwx---  0 ocrteam llmteam  169705 Mar 02  2025 indicdlp/train2017/mg_or_000650_0.png
-rwxrwx---  0 ocrteam llmteam  767259 Mar 02  2025 indicdlp/train2017/np_kn_000422_1.png
-rwxrwx---  0 ocrteam llmteam  475931 Mar 02  2025 indicdlp/train2017/mg_hi_000981_1.png
-rwxrwx---  0 ocrteam llmteam  246846 Mar 02  2025 indicdlp/train2017/nv_as_000079_0.png
-rwxrwx---  0 ocrteam llmteam  152472 Mar 02  2025 indicdlp/train2017/nv_ta_000463_0.png
-rwxrwx---  0 ocrteam llmteam  107485 Mar 02  2025 indicdlp/train2017/qp_mr_000650_1.png
-rwxrwx---  0 ocrteam llmteam  276865 Mar 02  2025 indicdlp/train2017/tb_hi_000344_0.png
-rwxrwx---  0 ocrteam llmteam 1403899 Mar 02  2025 indicdlp/train2017/ar_bn_000404_0.png
-rwxrwx---  0 ocrteam llmteam  594305 Mar 02  2025 indicdlp/train2017/ar_gu_000341_1.png
-rwxrwx---  0 ocrteam llmteam 1796612 Mar 02  2025 indicdlp/train2017/br_gu_000182_0.png
-rwxrwx---  0 ocrteam llmteam   98248 Mar 02  2025 indicdlp/train2017/sy_gu_000253_0.png
-rwxrwx---  0 ocrteam llmteam  210765 Mar 02  2025 indicdlp/train2017/fm_te_000341_0.png
-rwxrwx---  0 ocrteam llmteam  146523 Mar 02  2025 indicdlp/train2017/nv_ta_000523_0.png
-rwxrwx---  0 ocrteam llmteam  228597 Mar 02  2025 indicdlp/train2017/tb_bn_000289_0.png
-rwxrwx---  0 ocrteam llmteam 1511023 Mar 02  2025 indicdlp/train2017/br_pa_000736_0.png
-rwxrwx---  0 ocrteam llmteam  301091 Mar 02  2025 indicdlp/train2017/qp_or_000094_0.png
-rwxrwx---  0 ocrteam llmteam  464774 Mar 02  2025 indicdlp/train2017/mg_gu_000329.png
```

### Last 30 Lines of Tar Listing
```
-rwxrwx---  0 ocrteam llmteam   796107 Mar 02  2025 indicdlp/val2017/qp_pa_000327_1.png
-rwxrwx---  0 ocrteam llmteam    97653 Mar 02  2025 indicdlp/val2017/nt_or_000447_0.png
-rwxrwx---  0 ocrteam llmteam   724822 Mar 02  2025 indicdlp/val2017/tb_mr_000281_0.png
-rwxrwx---  0 ocrteam llmteam   960416 Mar 02  2025 indicdlp/val2017/br_ml_000054_0.png
-rwxrwx---  0 ocrteam llmteam  1753083 Mar 02  2025 indicdlp/val2017/br_pa_000689_0.png
-rwxrwx---  0 ocrteam llmteam   129846 Mar 02  2025 indicdlp/val2017/nt_hi_000394_1.png
-rwxrwx---  0 ocrteam llmteam   490690 Mar 02  2025 indicdlp/val2017/sy_pa_000152_0.png
-rwxrwx---  0 ocrteam llmteam  2821311 Mar 02  2025 indicdlp/val2017/np_gu_000517_1.png
-rwxrwx---  0 ocrteam llmteam   433420 Mar 02  2025 indicdlp/val2017/rp_en_000837_0.png
-rwxrwx---  0 ocrteam llmteam   339798 Mar 02  2025 indicdlp/val2017/tb_en_001031_0.png
-rwxrwx---  0 ocrteam llmteam   252446 Mar 02  2025 indicdlp/val2017/mn_ml_000752_0.png
-rwxrwx---  0 ocrteam llmteam   814749 Mar 02  2025 indicdlp/val2017/tb_bn_000093_0.png
-rwxrwx---  0 ocrteam llmteam  1092178 Mar 02  2025 indicdlp/val2017/br_te_000050_0.png
-rwxrwx---  0 ocrteam llmteam   208368 Mar 02  2025 indicdlp/val2017/nt_mr_000244.png
-rwxrwx---  0 ocrteam llmteam    73137 Mar 02  2025 indicdlp/val2017/fm_en_000337_0.png
-rwxrwx---  0 ocrteam llmteam    36500 Mar 02  2025 indicdlp/val2017/qp_ml_000724_0.png
-rwxrwx---  0 ocrteam llmteam   379379 Mar 02  2025 indicdlp/val2017/mn_gu_000736_0.png
-rwxrwx---  0 ocrteam llmteam   532851 Mar 02  2025 indicdlp/val2017/br_as_000347_1.png
-rwxrwx---  0 ocrteam llmteam   655451 Mar 02  2025 indicdlp/val2017/mg_pa_000914_0.png
-rwxrwx---  0 ocrteam llmteam   396796 Mar 02  2025 indicdlp/val2017/rp_as_000996_0.png
-rwxrwx---  0 ocrteam llmteam   424769 Mar 02  2025 indicdlp/val2017/tb_as_000975_0.png
-rwxrwx---  0 ocrteam llmteam   121347 Mar 02  2025 indicdlp/val2017/nv_ta_000669_0.png
-rwxrwx---  0 ocrteam llmteam   969227 Mar 02  2025 indicdlp/val2017/nv_ta_000179_0.png
-rwxrwx---  0 ocrteam llmteam   510538 Mar 02  2025 indicdlp/val2017/tb_te_000716_1.png
-rwxrwx---  0 ocrteam llmteam   139622 Mar 02  2025 indicdlp/val2017/ar_en_001081.png
-rwxrwx---  0 ocrteam llmteam   565373 Mar 02  2025 indicdlp/val2017/ar_gu_000337_1.png
drwxrwx---  0 ocrteam llmteam        0 Mar 03  2025 indicdlp/annotations/
-rwxrwx---  0 ocrteam llmteam 70748995 Mar 03  2025 indicdlp/annotations/instances_test2017.json
-rwxrwx---  0 ocrteam llmteam 71468289 Mar 03  2025 indicdlp/annotations/instances_val2017.json
-rwxrwx---  0 ocrteam llmteam 573064766 Mar 03  2025 indicdlp/annotations/instances_train2017.json
```

### Annotation File Sizes
- `instances_train2017.json`: 573.06 MB (546.6 MiB)
- `instances_val2017.json`: 71.47 MB (68.2 MiB)
- `instances_test2017.json`: 70.75 MB (67.5 MiB)
- **Total annotations:** 715.28 MB

### Format Assessment
**This is a COCO-style raw image + annotation dataset**, NOT a Hugging Face arrow/parquet cache.

**Evidence:**
1. ✅ **COCO naming convention:** `{split}2017/` directories and `instances_{split}2017.json` files match standard COCO format
2. ✅ **Raw PNG images:** 115,717 PNG files, no preprocessed/cached formats (.arrow, .parquet)
3. ✅ **Three-way split:** Standard train/test/val division
4. ✅ **Large JSON annotations:** 573MB training annotations typical of COCO bbox/segmentation data
5. ✅ **Image naming pattern:** Filenames like `fm_mr_000427_0.png` encode domain prefix (fm=forms), language (mr=Marathi), document ID, page number
6. ❌ **No HF cache artifacts:** No `dataset_info.json`, `state.json`, `.arrow`, or `.parquet` files

**Conclusion:** This is the **original IndicDLP dataset** as released by AI4Bharat, stored in COCO format with PNG images and JSON annotations. It is NOT a Hugging Face Datasets cache dump. The prepare script's expectation to download from HF Hub is incompatible with this local tar file.

---

## Mozhi Duplicate Resolution

### Comparison: `train/train/` vs `train (1)/train/`

| Metric | `train/train/` (Hindi) | `train (1)/train/` (Marathi) |
|--------|------------------------|-------------------------------|
| Image count | 79,762 | 80,151 |
| Total size | 187.7 MB | 276.06 MB |
| GT file lines | 79,762 | 80,151 |
| GT file SHA256 | `FE576445...653F985` | `72D33135...F133C81` |

### File Hashes
- `train/train/train_gt.txt`: `FE5764458B4FC36919B6A5B695A30C2536A9AADAD2EB673825C97C6C9653F985`
- `train (1)/train/train_gt.txt`: `72D33135A47D934BBD2F790F4234DB2F5C4191BFC8414EF4E4F282B42F133C81`

### Verdict
**✅ GENUINELY DIFFERENT DATA** - These are NOT duplicates.

**Reasoning:**
1. **Different file counts:** 79,762 vs 80,151 images (389 image difference)
2. **Different sizes:** 187.7 MB vs 276.06 MB (88.36 MB difference, 47% larger)
3. **Different content hashes:** SHA256 hashes differ completely
4. **Different languages:** Hindi vs Marathi (confirmed by vocabulary analysis in main inventory)
5. **Windows naming artifact:** The " (1)" suffix was added by Windows when extracting/moving a second directory with the same base name

**Conclusion:** These represent the full Hindi and Marathi language splits of Mozhi. Both must be retained. The " (1)" suffix is a Windows filesystem naming collision artifact, not an indication of duplication.

### Verbatim Sample Content

**`train/train/train_gt.txt` (Hindi) - First 5 lines:**
```
images/0057-0104_2_3_1.jpeg	फल
images/0057-0104_2_3_2.jpeg	काटते
images/0057-0104_2_3_3.jpeg	हैं
images/0057-0104_2_3_4.jpeg	तो
images/0057-0104_2_3_5.jpeg	वह
```

**`train/train/vocabulary.txt` (Hindi) - First 5 lines:**
```
ईरानियों
सीताराम
खोलते
हाँडी
चौधरी
```

**`test/test/test_gt.txt` (Hindi) - First 5 lines:**
```
images/0057-0010_2_3_1.jpeg	पड़ता
images/0057-0010_2_3_2.jpeg	था
images/0057-0010_2_3_3.jpeg	।
images/0057-0095_2_3_1.jpeg	के
images/0057-0095_2_3_2.jpeg	के बीच
```

**Note:** Line 3 shows a single punctuation mark (।) as a word crop, and line 5 contains multiple words ("के बीच") in one transcription, demonstrating varied word-level segmentation.

---

## IndicDLP Annotation Schema

**Source:** Extracted 3 annotation JSON files from `indicdlp.tar` (images NOT extracted)  
**Location:** `d:\AksharDrishti_1\indicdlp_annotations_only\indicdlp\annotations/`

### Annotation Files

| File | Size (MB) | Images | Annotations | Categories |
|------|-----------|--------|-------------|------------|
| instances_train2017.json | 546.52 | 95,172 | 1,458,855 | 42 |
| instances_val2017.json | 68.16 | 11,827 | 180,758 | 42 |
| instances_test2017.json | 67.47 | 11,614 | 178,165 | 42 |
| **TOTAL** | **682.15** | **118,613** | **1,817,778** | 42 |

### Top-Level JSON Structure

**Keys:** `images`, `annotations`, `categories`

This is standard **COCO format** with:
- No `info` or `licenses` metadata sections
- Flat lists for images, annotations, and categories

### Full Categories List (42 classes)

| ID | Category Name |
|----|---------------|
| 0 | advertisement |
| 1 | answer |
| 2 | author |
| 3 | chapter-title |
| 4 | contact-info |
| 5 | dateline |
| 6 | figure |
| 7 | figure-caption |
| 8 | first-level-question |
| 9 | flag |
| 10 | folio |
| 11 | footer |
| 12 | footnote |
| 13 | formula |
| 14 | header |
| 15 | headline |
| 16 | index |
| 17 | jumpline |
| 18 | options |
| 19 | ordered-list |
| 20 | page-number |
| 21 | paragraph |
| 22 | placeholder-text |
| 23 | quote |
| 24 | reference |
| 25 | second-level-question |
| 26 | section-title |
| 27 | sidebar |
| 28 | sub-headline |
| 29 | sub-ordered-list |
| 30 | sub-section-title |
| 31 | subsub-ordered-list |
| 32 | subsub-section-title |
| 33 | sub-unordered-list |
| 34 | subsub-headline |
| 35 | subsub-unordered-list |
| 36 | table |
| 37 | table-caption |
| 38 | table-of-contents |
| 39 | third-level-question |
| 40 | unordered-list |
| 41 | website-link |

### Sample Image Entry

```json
{
  "file_name": "rp_as_000209_0.png",
  "id": 0,
  "language": "Assamese",
  "document_category": "Research papers",
  "width": 1654,
  "height": 2339
}
```

**Fields:**
- `file_name`: PNG filename (no path, images are in `{split}2017/` directories)
- `id`: Unique integer image ID
- `language`: Full language name (e.g., "Assamese", "Hindi", "Marathi", "English")
- `document_category`: Domain/type (e.g., "Research papers", "Forms", "Acts_Rules")
- `width`, `height`: Image dimensions in pixels

### Sample Annotation Entry

```json
{
  "id": 0,
  "image_id": 0,
  "category_id": 21,
  "segmentation": [
    [
      185.64244079589832,
      192.20416259765622,
      1465.6920181448022,
      192.20416259765622,
      1465.6920181448022,
      1532.065185546875,
      185.64244079589832,
      1532.065185546875
    ]
  ],
  "area": 1715088.5361324176,
  "bbox": [
    185.64244079589832,
    192.20416259765622,
    1280.049577348904,
    1339.8610229492188
  ],
  "iscrowd": 0,
  "image_width": 1654,
  "image_height": 2339
}
```

**Fields:**
- `id`: Unique annotation ID
- `image_id`: Foreign key to `images[].id`
- `category_id`: Foreign key to `categories[].id` (0-41)
- `bbox`: COCO format `[x, y, width, height]` in absolute pixels, top-left origin
- `segmentation`: Polygon vertices `[[x1, y1, x2, y2, ...]]` (usually rectangular for layout)
- `area`: Bounding box area in pixels²
- `iscrowd`: 0 (not a crowd annotation)
- `image_width`, `image_height`: Parent image dimensions (redundant with image entry)

### Language and Domain Metadata

**✅ LANGUAGE FIELD EXISTS:** `language` (string, full language name)

**All 12 Languages in Dataset:**
- Assamese
- Bengali
- English
- Gujarati
- Hindi ✓ (target language)
- Kannada
- Malayalam
- Marathi ✓ (target language)
- Odia
- Punjabi
- Tamil
- Telugu

**✅ DOMAIN FIELD EXISTS:** `document_category` (string, underscore/space-separated)

**All 12 Document Domains in Dataset:**
- Acts_Rules ✓ (target domain)
- Brochures
- Forms ✓ (target domain)
- Magazines
- Manuals
- Newspaper ✓ (target domain: newspapers)
- Notice ✓ (target domain: notices)
- Novels
- Question_paper ✓ (target domain: question_papers)
- Research papers
- Syllabus
- Text books

**Filtering Strategy:** The prepare script can filter by:
1. **Language:** Exact string match on `images[].language` field
2. **Domain:** String match on `images[].document_category` field (needs normalization for underscore/space variants)

### Comparison Against `configs/class_map.yaml`

**Total in JSON:** 42 categories  
**Total in class_map.yaml:** 65 entries (mapping + ignore)

#### ✓ MATCHED (12 categories found in both)

| JSON Category | class_map.yaml Target |
|---------------|-----------------------|
| answer | text |
| author | title |
| figure | figure |
| folio | page_number |
| footer | footer |
| footnote | footer |
| formula | text |
| header | header |
| paragraph | text |
| quote | text |
| reference | text |
| table | table |

#### ✗ MISSING FROM class_map.yaml (30 categories)

**These appear in the JSON but have NO mapping in class_map.yaml:**

1. advertisement
2. chapter-title
3. contact-info
4. dateline
5. figure-caption
6. first-level-question
7. flag
8. headline
9. index
10. jumpline
11. options
12. ordered-list
13. page-number
14. placeholder-text
15. second-level-question
16. section-title
17. sidebar
18. sub-headline
19. sub-ordered-list
20. sub-section-title
21. sub-unordered-list
22. subsub-headline
23. subsub-ordered-list
24. subsub-section-title
25. subsub-unordered-list
26. table-caption
27. table-of-contents
28. third-level-question
29. unordered-list
30. website-link

**⚠️ CRITICAL MISMATCH:** The class_map.yaml uses **underscored names** (e.g., `chapter_title`, `section_title`, `ordered_list`, `page_number`, `figure_caption`, `table_caption`, `unordered_list`) but the JSON uses **hyphenated names** (e.g., `chapter-title`, `section-title`, `ordered-list`, `page-number`, `figure-caption`, `table-caption`, `unordered-list`).

**Impact:** With `on_unmapped: fail`, the prepare script will **crash immediately** when it encounters any of these 30 categories. None of them can be mapped because of the naming mismatch.

#### ⚠ IN class_map.yaml BUT NOT IN JSON (53 entries)

**These appear in class_map.yaml but were NOT found in the actual JSON:**

Mapped entries (49):
- abstract, address, body_text, caption, chapter_title, chart, date, diagram, document_title, emblem, equation, figure_caption, footnote_caption, graphic, handwriting, handwritten, heading, image, list, list_item, logo, map, option, ordered_list, page_footer, page_header, page_no, page_number, picture, plain_text, question, running_footer, running_header, seal, section_title, signature, stamp, sub_section_title, subheading, subsection_title, table_body, table_caption, table_cell, table_column, table_row, text, title, unordered_list, watermark

Ignored entries (4):
- background, noise, other, unknown

**Analysis:** Most of these are **synonyms/variants** that the class_map anticipated but don't exist in the actual dataset (e.g., the JSON has `paragraph` not `text`, `folio` not `page_number`, hyphenated not underscored).

### Resolution Required

**Before running `prepare_indicdlp.py --export`:**

1. **Fix naming convention mismatch:** Update class_map.yaml to use **hyphens** instead of underscores for all multi-word category names, OR modify prepare script to normalize names (replace `-` with `_`) before lookup

2. **Add missing mappings:** The 30 unmapped categories need explicit decisions:
   - `chapter-title`, `headline`, `section-title`, `sub-headline`, `sub-section-title`, `subsub-headline`, `subsub-section-title` → likely `title`
   - `figure-caption`, `table-caption` → likely `caption`
   - `ordered-list`, `unordered-list`, `sub-ordered-list`, `sub-unordered-list`, `subsub-ordered-list`, `subsub-unordered-list` → likely `list`
   - `page-number` → likely `page_number`
   - `first-level-question`, `second-level-question`, `third-level-question`, `options` → likely `text`
   - `advertisement`, `contact-info`, `dateline`, `flag`, `index`, `jumpline`, `placeholder-text`, `sidebar`, `table-of-contents`, `website-link` → decide case-by-case (map to `text` or add to `ignore`)

3. **Run `--inspect` first:** As the script comments recommend, run `python data/prepare_indicdlp.py --inspect` to confirm categories, then update class_map.yaml before attempting export

**Current Status:** ❌ **INCOMPATIBLE** - The prepare script will fail immediately with 30 unmapped categories due to hyphen/underscore naming mismatch.

---

## Category Normalization Fix

**Date:** 2026-09-15  
**Issue:** IndicDLP annotation JSON uses hyphenated category names (e.g., `chapter-title`, `figure-caption`, `ordered-list`) but `configs/class_map.yaml` uses underscored names (e.g., `chapter_title`, `figure_caption`, `ordered_list`).

**Solution:** Added `normalize_category_name()` function to `data/prepare_indicdlp.py` that converts hyphens to underscores before category lookup. Applied to all 4 locations where category names are extracted from annotations (`_iter_class_names()` and `_iter_annotations()` functions).

### Before/After Mismatch Counts

| Metric | Before Normalization | After Normalization | Change |
|--------|---------------------|---------------------|--------|
| Total JSON categories | 42 | 42 | - |
| Categories in class_map.yaml | 65 | 65 | - |
| **Missing from class_map** | **30** | **22** | **✓ Fixed 8** |

**Summary:** Normalization fixed 8 of the 30 mismatches (26.7% improvement). The 8 fixed categories were those differing only in hyphen/underscore format.

### Categories Fixed by Normalization (8 total)

These now match after hyphen→underscore conversion:
1. `chapter-title` → `chapter_title`
2. `figure-caption` → `figure_caption`
3. `ordered-list` → `ordered_list`
4. `page-number` → `page_number`
5. `section-title` → `section_title`
6. `sub-section-title` → `sub_section_title`
7. `table-caption` → `table_caption`
8. `unordered-list` → `unordered_list`

### Still Unmatched Categories (22 remain)

The following 22 categories remain unmatched after normalization because they genuinely do not exist in `class_map.yaml`:

| Category Name (normalized) | Annotation Count | % of Total |
|---------------------------|------------------|------------|
| placeholder_text | 76,803 | 5.26% |
| headline | 42,900 | 2.94% |
| sidebar | 39,610 | 2.72% |
| first_level_question | 36,966 | 2.53% |
| options | 35,411 | 2.43% |
| dateline | 30,812 | 2.11% |
| advertisement | 18,791 | 1.29% |
| sub_ordered_list | 18,608 | 1.28% |
| sub_headline | 14,671 | 1.01% |
| second_level_question | 13,199 | 0.90% |
| contact_info | 11,851 | 0.81% |
| website_link | 11,461 | 0.79% |
| subsub_ordered_list | 4,429 | 0.30% |
| jumpline | 3,833 | 0.26% |
| flag | 3,273 | 0.22% |
| third_level_question | 2,669 | 0.18% |
| subsub_section_title | 2,500 | 0.17% |
| subsub_headline | 1,863 | 0.13% |
| table_of_contents | 940 | 0.06% |
| sub_unordered_list | 863 | 0.06% |
| index | 455 | 0.03% |
| subsub_unordered_list | 96 | 0.01% |

**Total annotations using unmatched categories:** 372,004 / 1,458,855 (25.5%)

### Impact Analysis

**High-priority unmapped categories (>2% of annotations each):**
- `placeholder_text` (5.26%) - Likely template/form placeholders
- `headline` (2.94%) - Newspaper/article headlines (distinct from section titles)
- `sidebar` (2.72%) - Sidebar content boxes
- `first_level_question` (2.53%) - Question paper questions
- `options` (2.43%) - Multiple choice options
- `dateline` (2.11%) - Date/location stamps in news articles

These 6 categories alone account for **18.8% of all annotations**. They represent real layout elements that don't have obvious mappings to the existing 11-class taxonomy.

### Recommended Actions

**Before running `prepare_indicdlp.py --export`:**

1. **Update `configs/class_map.yaml`** to add mappings for the 22 unmatched categories, such as:
   ```yaml
   headline: title
   placeholder_text: text
   sidebar: text
   first_level_question: text
   second_level_question: text
   third_level_question: text
   options: text
   dateline: text
   advertisement: text  # or add to `ignore` if not useful
   contact_info: text
   website_link: text
   sub_ordered_list: list
   sub_unordered_list: list
   subsub_ordered_list: list
   subsub_unordered_list: list
   sub_headline: title
   subsub_headline: title
   subsub_section_title: title
   jumpline: text
   flag: stamp_seal  # if it's an emblem/flag, or `ignore` otherwise
   table_of_contents: text
   index: text
   ```

2. **OR** change `on_unmapped: fail` to `on_unmapped: text` in the script invocation to automatically fold unmapped categories into the `text` class (loses granularity but allows export to proceed)

3. **Run `--inspect` command** to confirm all categories are now handled before attempting full export

**Current Status:** ✅ Normalization implemented and working. ❌ 22 categories still need explicit mapping decisions before export can succeed with `on_unmapped: fail`.

---

## placeholder_text Investigation

**Date:** 2026-09-15  
**Purpose:** Evidence-based investigation of the final unmatched category after normalization fix

### Annotation Counts

| Category | Annotation Count | % of Total |
|----------|------------------|------------|
| placeholder_text | 76,803 | 5.26% |
| paragraph (comparison) | 467,934 | 32.08% |

The placeholder_text category represents 5.26% of all training annotations - a significant proportion that cannot be ignored.

### Bounding Box Statistics Comparison

| Category | Mean Width (px) | Mean Height (px) | Mean Aspect Ratio (W/H) |
|----------|-----------------|------------------|-------------------------|
| **placeholder_text** | **306.8** | **41.5** | **11.45** |
| paragraph | 7690.6 | 3527.0 | 4.70 |

**Key Findings:**
- **placeholder_text regions are MUCH smaller** than paragraphs (2.5% of width, 1.2% of height)
- **placeholder_text has extreme horizontal aspect ratio** (11.45:1) compared to paragraphs (4.70:1)
- This suggests **narrow horizontal text fields**, typical of form input boxes or template blanks

### Document Category Distribution

**Images containing placeholder_text:** 6,964 images (7.3% of training set)

| Domain | Image Count | Percentage |
|--------|-------------|------------|
| **Forms** | **4,166** | **59.8%** |
| **Question_paper** | **1,217** | **17.5%** |
| Brochures | 523 | 7.5% |
| Notice | 310 | 4.5% |
| Manuals | 265 | 3.8% |
| Text books | 203 | 2.9% |
| Magazines | 85 | 1.2% |
| Syllabus | 74 | 1.1% |
| Acts_Rules | 69 | 1.0% |
| Research papers | 20 | 0.3% |
| Newspaper | 19 | 0.3% |
| Novels | 13 | 0.2% |

**Critical Finding:** placeholder_text appears overwhelmingly in **Forms (59.8%)** and **Question_paper (17.5%)** documents - together accounting for 77.3% of occurrences. These are both target domains for AksharDrishti's government document processing use case.

### Visual Evidence: 15 Cropped Samples

**Location:** `d:\AksharDrishti_1\placeholder_text_samples\`

All 15 samples extracted, cropped with 10px padding, and saved:

1. `sample_01.png` (87x32px) - from fm_hi_000279_1.png (Forms, Hindi)
2. `sample_02.png` (207x46px) - from nt_bn_000020_0.png (Notice, Bengali)
3. `sample_03.png` (377x55px) - from fm_mr_000411_0.png (Forms, Marathi)
4. `sample_04.png` (124x44px) - from fm_ml_000836_1.png (Forms, Malayalam)
5. `sample_05.png` (129x26px) - from fm_ml_000752_1.png (Forms, Malayalam)
6. `sample_06.png` (217x48px) - from qp_kn_000620_0.png (Question_paper, Kannada)
7. `sample_07.png` (162x33px) - from br_hi_000881_1.png (Brochures, Hindi)
8. `sample_08.png` (185x64px) - from fm_ta_000279_0.png (Forms, Tamil)
9. `sample_09.png` (138x57px) - from fm_hi_000269_1.png (Forms, Hindi)
10. `sample_10.png` (326x39px) - from fm_te_000569_0.png (Forms, Telugu)
11. `sample_11.png` (949x94px) - from fm_bn_000441_0.png (Forms, Bengali)
12. `sample_12.png` (577x87px) - from fm_bn_000143_0.png (Forms, Bengali)
13. `sample_13.png` (238x38px) - from fm_hi_000432_1.png (Forms, Hindi)
14. `sample_14.png` (886x60px) - from fm_ml_000232_0.png (Forms, Malayalam)
15. `sample_15.png` (103x33px) - from fm_ml_000992_1.png (Forms, Malayalam)

**Observations:**
- Sample dimensions range from 87x32px to 949x94px
- All samples show narrow, horizontally-oriented regions
- 12 of 15 samples (80%) are from Forms domain
- Visual inspection would reveal whether these are blank fields, underscores, dotted lines, or pre-filled template text

### Recommendation for placeholder_text Mapping

**Evidence-based assessment:**
1. **Size:** Very small regions (mean 307x42px) suggests these are NOT body text paragraphs
2. **Shape:** Extreme horizontal aspect ratio (11.45:1) suggests form fields or answer blanks
3. **Context:** 77.3% occur in Forms and Question_papers - document types with fill-in-the-blank regions
4. **Frequency:** 5.26% of annotations - too significant to ignore

**Suggested Mappings (in priority order):**
1. **`placeholder_text: text`** - Conservative choice; treats as generic text content
2. **`placeholder_text: footer`** - If visual inspection shows these are mostly underscores/lines (page furniture)
3. **Add to `ignore` list** - Only if visual inspection confirms these are non-content artifacts (template markup, not actual layout regions)

**Action Required:** Visual inspection of the 15 cropped samples is needed to make the final mapping decision. Without seeing the actual visual content, the safest mapping is `placeholder_text: text`.

---

## Category Mapping: COMPLETE

**Date:** 2026-09-15  
**Status:** ✅ Full coverage achieved across all splits

### Three-File Match Confirmation

All 42 IndicDLP categories successfully mapped across all three annotation files:

| Split | Total Categories | Matched | Unmatched | Status |
|-------|------------------|---------|-----------|--------|
| **train** | 42 | 42 | 0 | ✅ COMPLETE |
| **val** | 42 | 42 | 0 | ✅ COMPLETE |
| **test** | 42 | 42 | 0 | ✅ COMPLETE |

**Verification:** All three splits contain identical category sets. No split-specific categories found.

### Final 11-Class Distribution (Training Set)

**Total annotations analyzed:** 1,458,855

| Target Class | Count | Percentage | Status |
|--------------|-------|------------|--------|
| **text** | 589,785 | 40.43% | ✓ |
| **list** | 243,268 | 16.68% | ✓ |
| **title** | 187,222 | 12.83% | ✓ |
| **page_number** | 159,329 | 10.92% | ✓ |
| **figure** | 94,981 | 6.51% | ✓ |
| **footer** | 71,399 | 4.89% | ✓ |
| **header** | 61,406 | 4.21% | ✓ |
| **caption** | 32,307 | 2.21% | ✓ |
| **table** | 19,158 | 1.31% | ✓ |
| **stamp_seal** | 0 | 0.00% | ⚠ ABSENT |
| **signature** | 0 | 0.00% | ⚠ ABSENT |

### Class Imbalance Analysis

**⚠ 2 target classes have NO annotations in IndicDLP:**
- `stamp_seal` (0 annotations, 0.00%)
- `signature` (0 annotations, 0.00%)

**Explanation:** These classes were designed for government document-specific elements (official stamps, seals, handwritten signatures) that do not appear in the general IndicDLP dataset. They are expected to appear in real-world government forms processing but will need to be trained from other sources or fine-tuned separately.

**✓ All other 9 classes exceed 1% representation** - no critical imbalance for core layout elements.

**✓ All 1,458,855 annotations successfully mapped** - 100% coverage, no unmapped categories.

### Mapping Summary

**Total mappings added:** 22 categories
- **Normalization fix:** hyphen → underscore conversion in `prepare_indicdlp.py`
- **Missing mappings added to class_map.yaml:**
  - Titles/Headings: 5 (headline, sub_headline, subsub_headline, subsub_section_title, flag)
  - Body Text: 4 (sidebar, dateline, contact_info, website_link)
  - Lists: 11 (placeholder_text, first/second/third_level_question, options, sub_ordered_list, subsub_ordered_list, sub_unordered_list, subsub_unordered_list, table_of_contents, index)
  - Figures: 1 (advertisement)
  - Page Furniture: 1 (jumpline)

**Final mapping rate:** 42/42 source categories → 11 target classes (100% coverage)

### Ready for Export

With all categories mapped, `data/prepare_indicdlp.py --export` can now proceed without errors. The script will:
1. Stream from HuggingFace Hub: `ai4bharat/indicdlp`
2. Apply normalization: convert hyphenated names to underscores
3. Resolve categories: use class_map.yaml mappings
4. Filter: Hindi/Marathi/English, government-relevant domains
5. Export: YOLO format with 11-class taxonomy

**Recommended next steps:**
1. Run `python data/prepare_indicdlp.py --inspect` to confirm HF Hub access
2. Run export with `--max-images 15000` for T4-compatible training subset
3. Monitor for `stamp_seal` and `signature` - consider adding synthetic examples or separate fine-tuning phase for these government-specific classes

---

## Filter Verification (Local Loading)

**Date:** 2026-09-16  
**Purpose:** Verify local annotation loading and filtering logic with real data

### Implementation

**Functions added to `data/prepare_indicdlp.py`:**
1. `load_local_annotations()` - Load COCO JSON from disk
2. `map_category_id_to_name()` - Build category ID → normalized name mapping
3. `build_filtered_image_list()` - Filter images by language/domain with optional cap
4. `map_annotations_to_images()` - Build image ID → annotations mapping

**Fixes applied:**
1. **Domain field:** Added `"document_category"` as first candidate (was missing)
2. **Domain values:** Updated defaults to match actual JSON:
   - `"acts_and_rules"` → `"acts_rules"`
   - `"notices"` → `"notice"`
   - `"newspapers"` → `"newspaper"`
   - `"question_papers"` → `"question_paper"`

### Filter Test Results (Training Set)

**Source:** `instances_train2017.json` (95,172 total images)

**Filter criteria:**
- **Languages:** Hindi, Marathi
- **Domains:** acts_rules, forms, notice, newspaper, question_paper

#### Uncapped Results

**Total matching images:** 5,743 (6.0% of training set)

**Breakdown by language:**

| Language | Count | Percentage |
|----------|-------|------------|
| Marathi | 2,970 | 51.7% |
| Hindi | 2,773 | 48.3% |

**Breakdown by domain:**

| Domain | Count | Percentage |
|--------|-------|------------|
| acts_rules | 1,430 | 24.9% |
| question_paper | 1,279 | 22.3% |
| forms | 1,155 | 20.1% |
| notice | 1,104 | 19.2% |
| newspaper | 775 | 13.5% |

#### Capped Results (max_images=15,000)

**Total matching images:** 5,743 (no truncation)

**Analysis:** The filtered pool (5,743 images) is well below the 15,000 cap. All matching images from the training set fit within the budget.

### Consistency Verification

**Second run (identical parameters):**
- Uncapped: 5,743 ✓ (consistent)
- Capped: 5,743 ✓ (consistent)

**Result:** No flakiness detected - filtering is deterministic.

### Key Findings

1. **Small filtered pool:** Only 5,743 images match our target criteria out of 95,172 training images (6.0%)
   - This is much smaller than anticipated 15,000 target
   - Will need to include validation and test splits to reach target size
   
2. **Balanced language distribution:** Hindi (48.3%) and Marathi (51.7%) are nearly equal
   
3. **Domain distribution:** All 5 target domains are represented with reasonable balance (13-25% each)
   
4. **No truncation needed:** Current filter yields fewer images than max_images cap

### Next Steps for Full Export

To reach the target ~15,000 training images:
1. **Option A:** Include validation/test splits in filtered pool
2. **Option B:** Expand filter criteria (add English, or additional domains)
3. **Option C:** Use the 5,743 high-quality government documents as-is (may be sufficient for fine-tuning)

**Recommendation:** Start with Option C - 5,743 carefully filtered government documents may provide better results than a diluted 15k set including less relevant domains.

---

## Pooled Filter Results (Train+Val+Test)

**Date:** 2026-09-16  
**Purpose:** Determine total available filtered images across all three splits

### Per-File Results

**Filter criteria:** Hindi/Marathi + 5 government domains (acts_rules, forms, notice, newspaper, question_paper)

| Split | Total Images | Matching Images | Match Rate |
|-------|--------------|-----------------|------------|
| train | 95,172 | 5,743 | 6.0% |
| val | 11,636 | 723 | 6.2% |
| test | 11,634 | 723 | 6.2% |
| **Subtotal** | **118,442** | **7,189** | **6.1%** |

### Image ID Overlap Investigation

**Initial finding (INCORRECT):** 258 overlapping image IDs between val and test

**Investigation result:** All 258 were **false alarms** - ID namespace collision, not actual duplicates.

**Analysis of 258 overlapping IDs:**
- True duplicates (same filename): **0**
- False alarms (different filenames): **258** (100%)

**Example false alarms:**
- ID 7175: val=`qp_mr_000034_0.png` vs test=`fm_mr_000277_0.png`
- ID 6665: val=`np_mr_000136_1.png` vs test=`np_mr_000174_1.png`
- ID 3594: val=`np_hi_000388_1.png` vs test=`np_hi_000232_1.png`

**Root cause:** COCO `image_id` field is **per-split sequential**, not globally unique across files. Two different images can share the same ID if they're in different splits.

**Corrected overlap check (by filename):**

| Comparison | Result |
|------------|--------|
| train ∩ val | ✅ 0 duplicates |
| train ∩ test | ✅ 0 duplicates |
| val ∩ test | ✅ 0 duplicates |

**Conclusion:** All three splits are fully disjoint. No actual duplicate images exist across splits.

### Pooled Totals (All Three Splits)

**Combined total:** 7,189 unique images (verified by filename deduplication)

**Per-split contribution:**

| Split | Count | Percentage |
|-------|-------|------------|
| train | 5,743 | 79.9% |
| val | 723 | 10.1% |
| test | 723 | 10.1% |

**Combined language breakdown:**

| Language | Count | Percentage |
|----------|-------|------------|
| Marathi | 3,715 | 51.7% |
| Hindi | 3,474 | 48.3% |

**Combined domain breakdown:**

| Domain | Count | Percentage |
|--------|-------|------------|
| acts_rules | 1,788 | 24.9% |
| question_paper | 1,600 | 22.3% |
| forms | 1,442 | 20.1% |
| notice | 1,381 | 19.2% |
| newspaper | 978 | 13.6% |

### Comparison Against 15,000 Target

**⚠️ IMPORTANT NOTE: COCO image_id is NOT globally unique**

The COCO format's `image_id` field is per-split sequential (starts from 0 in each split file). When pooling across splits, always use `file_name` as the unique key, not `image_id`. Using `image_id` for cross-split deduplication will produce false positives.

| Metric | Value |
|--------|-------|
| **Target** | **15,000 images** |
| **Pooled total** | **7,189 images** |
| **Percentage of target** | **47.9%** |
| **Gap** | **7,811 images short** |
| **Shortfall** | **52.1% below target** |

### Analysis

**The filtered pool (7,189 images) represents less than half the original 15,000 target.**

**Key constraints limiting pool size:**
1. **Language filter:** Only Hindi/Marathi (excludes 10 other languages: Assamese, Bengali, English, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu)
2. **Domain filter:** Only 5 government-related domains (excludes 7 others: Brochures, Magazines, Manuals, Novels, Research papers, Syllabus, Text books)
3. **Dataset composition:** Government documents in Hindi/Marathi are a minority in IndicDLP

**To reach 15,000 images, options include:**
- Add English language (3rd most common in government docs)
- Add additional domains (e.g., Syllabus, Text books - academic documents)
- Use the current 7,189 high-quality filtered images (may be sufficient for specialized fine-tuning)

**Recommendation:** The 7,189 images are a carefully filtered, high-quality subset directly matching the target use case (Hindi/Marathi government documents). This focused dataset may provide better results than a diluted 15k set including less relevant content.

### Consistency Verification

**Second run (identical parameters):**
- Pooled total: 7,189 ✅ (consistent)

**Result:** Filtering is deterministic across runs.

---

## Pooled Filter Results (Train+Val+Test, Hindi+Marathi+English)

**Date:** 2026-09-16  
**Purpose:** Expand filter to include English language, assess impact on pool size

### Per-File Results (With English Added)

**Filter criteria:** Hindi/Marathi/**English** + 5 government domains (acts_rules, forms, notice, newspaper, question_paper)

| Split | Total Images | Matching Images | Match Rate | Change from Hindi+Marathi Only |
|-------|--------------|-----------------|------------|-------------------------------|
| train | 95,172 | 8,267 | 8.7% | +2,524 (+44.0%) |
| val | 11,636 | 1,040 | 8.9% | +317 (+43.8%) |
| test | 11,634 | 1,034 | 8.9% | +311 (+43.0%) |
| **Total** | **118,442** | **10,341** | **8.7%** | **+3,152 (+43.9%)** |

### Pooled Total (All Three Splits, Deduplicated by Filename)

**New pooled total:** 10,341 unique images

**Per-split contribution:**

| Split | Count | Percentage |
|-------|-------|------------|
| train | 8,267 | 79.9% |
| val | 1,040 | 10.1% |
| test | 1,034 | 10.0% |

### Combined Language Breakdown

| Language | Count | Percentage |
|----------|-------|------------|
| Hindi | 3,474 | 33.6% |
| Marathi | 3,715 | 35.9% |
| **English** | **3,152** | **30.5%** |
| **Total** | **10,341** | **100.0%** |

**Language balance:** Nearly perfect three-way balance - Marathi slightly ahead, Hindi and English close behind.

### Combined Domain Breakdown

| Domain | Count | Percentage | Change from Hindi+Marathi Only |
|--------|-------|------------|-------------------------------|
| acts_rules | 3,282 | 31.7% | +1,494 (+83.5%) |
| forms | 2,027 | 19.6% | +585 (+40.6%) |
| notice | 1,943 | 18.8% | +562 (+40.7%) |
| question_paper | 1,600 | 15.5% | 0 (0.0%) |
| newspaper | 1,489 | 14.4% | +511 (+52.3%) |

**Key finding:** English contributed significantly to all domains except **question_paper** (0 additional images), suggesting English-language question papers are rare or absent in the filtered government document subset.

### English's Standalone Contribution

**Previous total (Hindi+Marathi only):** 7,189 images  
**New total (Hindi+Marathi+English):** 10,341 images  
**English's contribution:** 3,152 images

**Analysis:**
- English added **3,152 images** (43.9% increase over Hindi+Marathi baseline)
- English represents **30.5%** of the expanded pool
- English contribution is nearly equal to Hindi (3,474) and Marathi (3,715) individually

### Comparison Against 15,000 Target

| Metric | Value |
|--------|-------|
| **Target** | **15,000 images** |
| **New pooled total** | **10,341 images** |
| **Percentage of target** | **68.9%** |
| **Remaining gap** | **4,659 images short** |
| **Shortfall** | **31.1% below target** |

**Progress:** Adding English improved coverage from 47.9% (7,189) to 68.9% (10,341) of the 15,000 target - a **21.0 percentage point gain**.

**Remaining gap:** 4,659 images (31.1%) still needed to reach 15,000.

### Summary

**Impact of adding English:**
- ✅ Added 3,152 images (+43.9%)
- ✅ Improved target coverage from 47.9% to 68.9% (+21.0 pp)
- ✅ Created balanced three-language dataset (33.6% / 35.9% / 30.5%)
- ✅ Significantly expanded acts_rules (+83.5%) and newspaper (+52.3%) domains
- ⚠️ No English question papers found in government document filter

**Pool composition:** 10,341 government documents across Hindi/Marathi/English, well-balanced by language and domain, covering 68.9% of the original 15,000 target.

### Consistency Verification

**Second run (identical parameters):**
- New pooled total: 10,341 ✅ (consistent)

**Result:** Filtering with English is deterministic across runs.

---

## Image Extraction Complete

**Date:** 2026-09-16  
**Duration:** ~2 minutes 10 seconds (single-pass tar iteration)  
**Method:** Single-pass tar member iteration using `iter_tar_images()`

### Extraction Summary

**Target:** 10,341 images (filtered by Hindi/Marathi/English + 5 government domains)  
**Extracted:** 10,341 images ✓  
**Output directory:** `datasets/indicdlp_subset/images/`

### Disk Space

**Estimated size (pre-extraction):** 7.14 GB  
**Actual consumption:** 6.90 GB (7,409,047,687 bytes)  
**Average file size:** 0.67 MB per image  
**Disk space gate:** PASSED (6.90 GB < 17.44 GB threshold, 80% of 21.8 GB free)

### Integrity Verification

**File count check:**
- Expected: 10,341 unique files
- Actual: 10,341 files on disk ✓
- **Collision check:** PASSED (no filename overwrites)

**Spot-check (10 random PNGs):**

| File | Dimensions | Status |
|------|------------|--------|
| ar_hi_000333_0.png | 1654×2339 | ✓ Valid |
| ar_en_000415.png | 596×842 | ✓ Valid |
| fm_hi_000643_1.png | 596×842 | ✓ Valid |
| fm_hi_000146_0.png | 1700×2400 | ✓ Valid |
| fm_en_000377_0.png | 595×842 | ✓ Valid |
| ar_mr_000085_0.png | 1484×2267 | ✓ Valid |
| ar_hi_000186_0.png | 1387×1806 | ✓ Valid |
| qp_hi_000197_0.png | 1595×2103 | ✓ Valid |
| ar_en_001431.png | 596×842 | ✓ Valid |
| qp_mr_000045_0.png | 1653×2339 | ✓ Valid |

**Result:** All 10 spot-checked images opened successfully with PIL.Image.open().verify() ✓

### Image Resolution Analysis (from spot-check)

**Range:** 595×842 to 1700×2400 pixels  
**Typical dimensions:**
- Standard letter size: ~596×842 (A4-like aspect ratio)
- Legal/long form: ~1654×2339 (higher resolution scans)
- Question papers: ~1595×2103 to 1653×2339

### Implementation Notes

**Function added:** `iter_tar_images(tar_path, filenames)` in `data/prepare_indicdlp.py`

**Key features:**
- Single-pass tar iteration (opens tar ONCE, not 10,341 times)
- Progress logging every 1000 images
- Extracts only files matching the filtered filename set
- Yields (filename, raw_bytes) tuples for memory-efficient processing

**Performance:** Extracted 10,341 images from 79.95 GB tar in ~130 seconds (~80 images/second)

### Next Steps

**Completed:**
- ✅ Image extraction from tar
- ✅ File integrity verification
- ✅ Disk space management

**Pending:**
- YOLO annotation conversion (separate step, NOT done yet)
- Training data split (after YOLO conversion)
- Final dataset validation

### Files Modified

**Code changes:**
- `data/prepare_indicdlp.py`: Added `iter_tar_images()` function only
- No changes to `normalize_category_name()`, `_iter_class_names()`, `_iter_annotations()`, `_to_yolo()`, or `class_map.yaml`

**Extraction artifacts (NOT in git, gitignored):**
- `datasets/indicdlp_subset/images/` (10,341 PNG files, 6.90 GB)

---

## Train/Val Split

**Date:** 2026-09-16  
**Method:** Stratified split by (language, domain) joint combination  
**Random seed:** 42 (per TECHNICAL_REPORT.md section 11.2)  
**Split ratio target:** 90% train / 10% val

### Split Summary

**Total images:** 10,341  
**Train:** 9,306 images (90.0%)  
**Val:** 1,035 images (10.0%)

**Directory structure:**
```
datasets/indicdlp_subset/images/
├── train/     (9,306 files)
└── val/       (1,035 files)
```

### Stratification Details

**Stratified by:** 14 unique (language, domain) combinations  
**Method:** scikit-learn train_test_split with stratify parameter

**Combinations covered:**
- 3 languages: Hindi, Marathi, English
- 5 domains: acts_rules, forms, notice, newspaper, question_paper
- 14 actual combinations (English × question_paper has 0 images)

### Per-Group Breakdown

| Language | Domain | Train | Val | Total | Val% |
|----------|--------|------:|----:|------:|-----:|
| English | acts_rules | 1,345 | 149 | 1,494 | 10.0% |
| English | forms | 526 | 59 | 585 | 10.1% |
| English | newspaper | 460 | 51 | 511 | 10.0% |
| English | notice | 506 | 56 | 562 | 10.0% |
| Hindi | acts_rules | 637 | 71 | 708 | 10.0% |
| Hindi | forms | 715 | 80 | 795 | 10.1% |
| Hindi | newspaper | 432 | 48 | 480 | 10.0% |
| Hindi | notice | 542 | 60 | 602 | 10.0% |
| Hindi | question_paper | 800 | 89 | 889 | 10.0% |
| Marathi | acts_rules | 972 | 108 | 1,080 | 10.0% |
| Marathi | forms | 582 | 65 | 647 | 10.0% |
| Marathi | newspaper | 448 | 50 | 498 | 10.0% |
| Marathi | notice | 701 | 78 | 779 | 10.0% |
| Marathi | question_paper | 640 | 71 | 711 | 10.0% |

**Val% range:** 10.0% - 10.1%  
**Maximum deviation from 10%:** 0.1%

### Balance Verification

✅ **ALL 14 groups have >0 images in BOTH train and val splits**  
✅ **Zero train groups:** None  
✅ **Zero val groups:** None  
✅ **Imbalanced groups (deviation >5%):** None

**Perfect stratification achieved** - every group maintains 10% ± 0.1% validation proportion.

### File Organization

**Method:** Physical file move (not copy) to conserve disk space  
**Source:** Flat `datasets/indicdlp_subset/images/*.png`  
**Destination:**
- `datasets/indicdlp_subset/images/train/<filename>`
- `datasets/indicdlp_subset/images/val/<filename>`

**Verification:**
- Train files: 9,306 ✓
- Val files: 1,035 ✓
- Sum: 10,341 ✓
- Filename overlap: 0 ✓
- Flat directory cleaned: Yes ✓

### Manifest

**Split assignments saved to:** `d:\AksharDrishti_1\split_manifest.csv`  
**Columns:** filename, language, domain, original_split, split

The manifest records which original COCO split (train2017/val2017/test2017) each image came from, alongside its new train/val assignment.

### Next Steps

**Completed:**
- ✅ Image extraction (10,341 files, 6.90 GB)
- ✅ Stratified train/val split (90/10 ratio, seed=42)
- ✅ Physical file organization into subdirectories

**Pending:**
- YOLO annotation conversion (separate step)
- Dataset configuration file generation
- Final dataset validation
