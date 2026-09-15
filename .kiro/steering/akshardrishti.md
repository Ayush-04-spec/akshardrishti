# AksharDrishti Steering File

## Project

AksharDrishti — layout-aware OCR pipeline for multilingual Indian government documents. Languages: Hindi (Devanagari), Marathi (Devanagari), English (Latin). B.Tech final year project. Core idea: detect document structure BEFORE text recognition, preserve reading order, route each region to a script-appropriate OCR engine.

## Stack

Python 3.10+, PyTorch 2.2+, Ultralytics YOLO11 (AGPL-3.0), HuggingFace Transformers 4.40+, OpenCV 4.9+, Pydantic, Streamlit, JiWER, pycocotools.

## Pipeline stages

1. **Preprocessing** - Deskewing (Hough/moments), CLAHE contrast enhancement, Denoising (FastNLMeans/bilateral/median), Optional Sauvola binarization
2. **Layout Detection** - YOLO11 + SAHI, 11-class taxonomy detection, Sliced inference for dense pages, Bounding box extraction
3. **Reading Order Assignment** - XY-cut recursive algorithm, Furniture removal (headers/footers), Multi-column handling, Region ordering
4. **Line Segmentation** - Projection profiles, Horizontal histogram analysis, Peak detection, Line-level crops
5. **Script Identification** - CNN classifier, Devanagari vs Latin routing, Confidence scoring
6. **Text Recognition** - Backend routing: CRNN-Mozhi (Devanagari), TrOCR (English/Latin), 6 pluggable backends
7. **Postprocessing** - Unicode NFC normalization, Shirorekha repair (Devanagari), Control char cleanup
8. **Output** - Structured JSON + Markdown + Text
9. **Orchestration** - Configuration-driven pipeline with SHA-256 config hashing for reproducibility

## Module map

- **configs/** - Master pipeline config (pipeline.yaml) and 42→11 class taxonomy mapping (class_map.yaml)
- **akshardrishti/** - Core library with config loading, Pydantic schemas, pipeline orchestrator
- **akshardrishti/preprocess/** - Deskew, CLAHE, denoise, binarize
- **akshardrishti/layout/** - YOLO11 + SAHI detector, XY-cut reading order algorithm, projection-based line segmentation
- **akshardrishti/script_id/** - CNN classifier for Devanagari vs Latin routing with Unicode fallback
- **akshardrishti/recognize/** - Recognizer interface + ScriptRouter, CRNN-Mozhi, TrOCR, Tesseract, Bhashini API, PaddleOCR, Surya backends
- **akshardrishti/postprocess/** - Unicode NFC normalization, shirorekha repair, control char cleanup
- **train/** - Dataset classes, YOLO11 fine-tuning, CRNN-CTC training, script classifier training
- **data/** - IndicDLP→YOLO exporter, Mozhi→index.csv + charset builder, synthetic data generator (future)
- **eval/** - CER/WER/Levenshtein metrics, layout mAP, benchmark runner that generates comparison tables
- **app/** - Streamlit demo UI (upload → visualize → download)
- **benchmark/** - AksharDrishti-Bench annotation dataset (images, ground truth text, optional layout boxes, metadata, annotation guidelines)
- **tests/** - 58 unit tests, no-pytest runner for Kaggle
- **notebooks/** - Kaggle/Colab training template
- **weights/** - Model checkpoints (layout YOLO, CRNN recognizer + charset, script classifier) — gitignored

## Current project state

- No model weights are trained. weights/ is empty.
- benchmark/images/, benchmark/gt_text/, benchmark/gt_layout/ are empty.
- Datasets are downloaded but may not yet be prepared into training format.
- Every accuracy number currently in TECHNICAL_REPORT.md and the demo summaries is a literature-based ESTIMATE, not a measured result. They must all be replaced.

## CRITICAL RULE 1 — No fabricated metrics

Never write, generate, estimate, guess, or hardcode any accuracy metric (CER, WER, mAP, confidence intervals, per-class scores). Every metric must come from actually executing eval/run_benchmark.py against real annotated ground truth. If a number cannot be measured right now, write exactly "NOT YET MEASURED". Never substitute a plausible-looking placeholder.

## CRITICAL RULE 2 — Don't grade our own homework

Never apply the pipeline's own error-repair to ground truth during scoring. Shirorekha repair and punctuation stripping are FORBIDDEN in scoring normalization. Scoring normalization is encoding-level only: Unicode NFC, ZWJ (U+200D) removal, variation selector removal.

## CRITICAL RULE 3 — Config is protected

Never modify configs/pipeline.yaml or configs/class_map.yaml without first telling me exactly what you intend to change and why, and waiting for my approval.

## CRITICAL RULE 4 — No silent weight creation

Never create, download, or stub out files in weights/. If a model weight is missing, report it as missing and stop.

## CRITICAL RULE 5 — Hardware constraints

Local machine has an NVIDIA RTX 3050 Laptop GPU with only 4GB VRAM. This is NOT enough for real training (layout training needs ~8-16GB, per TECHNICAL_REPORT.md section 8.3.2). NEVER run train_layout.py, train_crnn.py, or train_scriptid.py locally at production batch size/imgsz. Local GPU may only be used for tiny smoke tests (batch=1 or 2, a handful of steps) to verify code runs without crashing. All real training happens on Kaggle/Colab dual-T4 per TECHNICAL_REPORT.md section 9.
