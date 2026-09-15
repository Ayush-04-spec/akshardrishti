# AksharDrishti

**Layout-Aware Transformer for Multilingual Indian OCR** — Hindi · Marathi · English

A document-digitisation pipeline that detects layout structure *before* reading text, routes each
region to a script-appropriate recogniser, and emits structure-preserving JSON.

> B.Tech Computer Engineering project · K. K. Wagh Institute of Engineering Education and Research
> Prasad Tarde · Kalpesh Suryawanshi · Ayush Shirsath · Nikhil Pawar — guide: Prof. K. P. Birla

---

## Why this exists

Conventional OCR flattens a document to a text stream. On Indian government records — multi-column
Marathi gazette notices, bilingual certificates with seals and signatures, tabular 7/12 extracts —
that destroys reading order, merges columns, and feeds stamps into the recogniser as if they were
words.

AksharDrishti detects the layout first, then reads each region independently:

```
PDF/JPG → preprocess → layout detect (YOLO11 + SAHI) → reading order (XY-cut)
        → line segment → script ID → recognise (CRNN-Devanagari | TrOCR-Latin)
        → Devanagari repair → structured JSON
```

## Quickstart

```bash
git clone <your-repo> && cd aksharDrishti
pip install -r requirements.txt

# verify the install (no model weights needed)
python tests/run_tests_nopytest.py       # or: python -m pytest tests/ -v

# run the demo (needs weights, or pick a pretrained backend in the sidebar)
streamlit run app/streamlit_app.py
```

## Building the datasets

### 1. Layout training data — IndicDLP

[IndicDLP](https://huggingface.co/datasets/ai4bharat/indicdlp) (AI4Bharat / IIIT-H CVIT, MIT licence)
is 119,806 manually annotated document images across 12 languages, 12 domains and 42 layout classes.
Accept the terms on Hugging Face, then `huggingface-cli login`.

```bash
# ALWAYS inspect first -- our 42-class map was reconstructed from the paper and
# must be reconciled against the real category names
python data/prepare_indicdlp.py --inspect

# then export a filtered subset in YOLO format
python data/prepare_indicdlp.py --out datasets/indicdlp_subset --max-images 15000
```

Filters to Hindi/Marathi/English and government-ish domains (acts & rules, forms, notices,
newspapers, question papers), and remaps 42 classes → our 11. Writes `data.yaml` and
`subset_stats.json` (put that table in the report's Data Set section).

### 2. Recognition training data — Mozhi

[Mozhi](https://cvit.iiit.ac.in/usodi/tdocrmil.php) is ~1.2M annotated word images across 13 Indian
languages, released with *Towards Deployable OCR Models for Indic Languages* — the work behind
Bhashini's printed-OCR service.

```bash
python data/prepare_mozhi.py --languages hindi marathi --out datasets/mozhi
```

Builds `index_{train,val,test}.csv` and a charset from **train only**, then reports charset coverage
on val/test — that coverage figure is a hard floor under your achievable CER, so read it.

### 3. Evaluation data — AksharDrishti-Bench (you build this)

IndicDLP has boxes but **no transcriptions**, so it cannot measure OCR accuracy. Every CER number in
the report has to come from pages your team transcribed. See
[`benchmark/ANNOTATION_GUIDELINES.md`](benchmark/ANNOTATION_GUIDELINES.md).

```
benchmark/
  images/      page_001.jpg …
  gt_text/     page_001.txt      ← full ground-truth transcription
  gt_layout/   page_001.json     ← optional: boxes + reading order
```

## Training

```bash
# model #1 -- layout (Kalpesh's Kaggle account)
python train/train_layout.py --data datasets/indicdlp_subset/data.yaml \
    --model yolo11l.pt --epochs 60 --batch 8 --imgsz 1024 \
    --backup-dir /content/drive/MyDrive/akshardrishti

# baseline for comparison: a released IndicDLP checkpoint, no training
python train/train_layout.py --eval-only --weights weights/indicdlp_yolov10x.pt \
    --data datasets/indicdlp_subset/data.yaml

# model #2 -- Devanagari recognition (Prasad's account, in parallel)
python train/train_crnn.py --data datasets/mozhi --epochs 30 --batch 64 \
    --backup-dir /content/drive/MyDrive/akshardrishti

# model #3 -- script router (cheap, <1h)
python train/train_scriptid.py --train datasets/scriptid/train.csv --val datasets/scriptid/val.csv
```

**Both models checkpoint every epoch.** Kaggle sessions are killed without warning; always pass
`--backup-dir` pointing at Drive.

## Evaluation

```bash
# the paper's Table 6.1 + the ablation table, in one run
python eval/run_benchmark.py --benchmark benchmark --out results/ --ablations
```

Writes `results/results.json` and `results/results.md`. Paste the latter straight into the report.

Reported per system: CER with a bootstrap 95% CI, WER, substitution/deletion/insertion breakdown,
and seconds per page. Ablations: `no_layout`, `no_sahi`, `no_script_routing`, `no_postprocess` —
`no_layout` is the one that tests the project's central claim.

## Configuration

Everything runs off `configs/pipeline.yaml`; experiments are config diffs, not code edits.

```bash
python eval/run_benchmark.py --systems akshardrishti   # uses the config as-is
```

```python
from akshardrishti import Config
from akshardrishti.pipeline import AksharDrishtiPipeline

cfg = Config.load("configs/pipeline.yaml", overrides=["layout.sahi.enabled=false"])
doc = AksharDrishtiPipeline(cfg).process("notice.pdf")

print(doc.full_text())
print(doc.pages[0].regions[0].type, doc.pages[0].regions[0].reading_order)
doc.save_json("out.json")
```

Every config carries a hash (`cfg.hash()`) that is stamped onto results, so any number in the paper
traces back to the exact settings that produced it.

## Adding an OCR backend

1. Subclass `Recognizer`, implement `_load` and `recognize`
2. Decorate with `@register("your_name")`
3. Add a `recognize.backends.your_name` block to the config

It then becomes one row in the results table instead of a rewrite. Included backends:
`crnn_mozhi`, `trocr_printed`, `tesseract`, `bhashini_api`, `paddle_vl`, `surya`.

## Repository layout

```
configs/          pipeline.yaml · class_map.yaml (42 → 11 classes)
data/             prepare_indicdlp.py · prepare_mozhi.py · synth_devanagari.py
akshardrishti/
  preprocess/     deskew · CLAHE · denoise · Sauvola binarisation
  layout/         detector (YOLO+SAHI) · reading_order (XY-cut) · lines
  recognize/      base (interface + router) · crnn · trocr_en · tesseract
                  bhashini_api · paddle_vl · surya_backend
  script_id/      classifier · model
  postprocess/    text_repair (Unicode NFC · shirorekha · nukta)
  schema.py       Pydantic output contract
  pipeline.py     orchestrator
train/            train_layout.py · train_crnn.py · train_scriptid.py · dataset.py
eval/             metrics.py · run_benchmark.py
app/              streamlit_app.py
benchmark/        AksharDrishti-Bench + annotation guidelines
tests/            58 unit tests
```

## Design decisions worth knowing

**TrOCR is English-only here.** Its decoder is an English RoBERTa and it cannot read Devanagari.
Pairing it with a CTC model for Devanagari *is* the script-routing contribution, not a workaround.

**Binarisation is off for the page, on for line crops.** YOLO and TrOCR are trained on natural
grayscale and lose accuracy on hard-thresholded input; a CTC model on a tight Devanagari line crop
gains from it.

**Reading order pulls furniture out first.** A running header spanning both columns blocks every
vertical XY-cut and collapses the page to top-down order — so headers, footers and page numbers are
removed before the cut and appended after.

**Scoring normalisation never repairs errors.** `normalize_for_scoring` fixes encoding-level
differences (NFC, nukta forms, zero-width joiners) and nothing else. Applying the pipeline's own
`clean_text` to ground truth would let the post-processor grade its own homework.

**Corpus CER, not mean-of-documents.** Both are reported; the mean is the more flattering and less
correct number.

## Licences

| Component | Licence | Note |
|---|---|---|
| IndicDLP dataset | MIT | |
| Ultralytics YOLO11 | AGPL-3.0 | fine for academic use; matters if commercialised |
| DocLayout-YOLO | AGPL-3.0 | |
| Surya weights | modified OpenRAIL-M | free under $5M funding/revenue |
| TrOCR, PaddleOCR | MIT / Apache-2.0 | |
| Mozhi dataset | see CVIT terms | check before redistributing |

## Citations

- IndicDLP: *A Foundational Dataset for Multi-Lingual and Multi-Domain Document Layout Parsing*, arXiv:2512.20236
- Mozhi / Bhashini OCR: *Towards Deployable OCR Models for Indic Languages*, arXiv:2205.06740
- DocLayout-YOLO: arXiv:2410.12628
- TrOCR: arXiv:2109.10282
- SAHI: arXiv:2202.06934

---

## Getting this onto Kaggle / Colab

The project ships as a zip. Two ways to get it onto a GPU machine — `notebooks/00_setup_and_train.ipynb`
step 3 handles both.

**Option A — Kaggle Dataset (fastest, no GitHub).** Kaggle → Datasets → New Dataset → upload
`AksharDrishti.zip` (it unzips automatically) → name it `akshardrishti-code`, set **Private** → in the
notebook, Add Input → Datasets → select it. Note `/kaggle/input` is read-only, so step 3 copies the
tree to `/kaggle/working` before doing anything.

**Option B — GitHub (better for a team).**

```bash
cd aksharDrishti
git init && git add . && git commit -m "AksharDrishti: initial pipeline"
gh repo create akshardrishti --private --source=. --push
# without gh: create the repo on github.com, then
#   git remote add origin https://github.com/<you>/akshardrishti.git
#   git branch -M main && git push -u origin main
```

**Keep the repo private.** AksharDrishti-Bench contains real government documents;
`.gitignore` already excludes `benchmark/images/`, and the annotation guidelines require
anonymisation before anything is published.

With four people committing over two weeks, Option B pays for itself around day 3 — Option A means
re-uploading a zip every time anyone changes a line.
