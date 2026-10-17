# AksharDrishti-Bench — Annotation Guidelines

**Target:** 50 real Indian government/legal pages · layout boxes + reading order + full ground-truth text
**Team:** ~13 pages each · ~4–5 hours per person · finish by Day 7
**Why it matters:** IndicDLP has boxes but no transcriptions. Without this benchmark there is **no CER
number**, no results table, and no paper.

---

## Before anyone annotates a single page

**Run the calibration round.** All four annotate the *same 5 pages*, then compare box-by-box. You will
disagree — about what counts as a header, whether a two-line heading is one region or two, where a
table ends. Discovering that on page 5 costs an hour. Discovering it on page 48 costs the deadline.

Resolve every disagreement, write the resolution into the "Team decisions" section at the bottom of
this file, and only then start real annotation.

---

## 1. Sourcing pages

| Source | What to grab |
|---|---|
| `gr.maharashtra.gov.in` | Government Resolutions (GRs) — the core target |
| eGazette | gazette notifications |
| India Code | acts, rules, statutory notices |
| District collectorate sites | public notices, tender documents |
| Municipal corporation portals | forms, circulars |

**Aim for a spread, not 50 of the same thing:**

- **Language:** ~20 Marathi, ~15 Hindi, ~15 English/bilingual
- **Layout:** ~25 single-column, ~15 multi-column, ~10 with tables
- **Quality:** ~30 clean digital PDFs, ~20 scanned/photographed (skewed, noisy, low-DPI)
- **Furniture:** at least 15 pages with stamps, seals or signatures

That last row matters. `stamp_seal` and `signature` are the classes that distinguish this benchmark
from every general-purpose document dataset. If only three pages have seals, the class is
unmeasurable and a reviewer will say so.

Collect ~80 pages, keep the best 50.

---

## 2. Anonymisation — do this before annotating, not after

These are real government documents about real people. Before a page enters the benchmark:

- Black out personal names of private individuals (officials acting in their official capacity are fine)
- Black out Aadhaar numbers, PAN, phone numbers, bank details, residential addresses
- Black out photographs of individuals and specimen signatures of private persons

Redact the **image**, then transcribe the redacted version so the box and the text agree. Use
`[REDACTED]` in the transcription where content was removed.

Non-negotiable if the benchmark is published — and publishing it is the plan.

---

## 3. Layout annotation

Tool: [Label Studio](https://labelstud.io/) (free, local) or CVAT. Export as COCO or Label Studio JSON.

### The 11 classes

| Class | Use it for | Watch out for |
|---|---|---|
| `title` | headings at any level, document title, section headings | a multi-line heading is **one** region |
| `text` | body paragraphs — the default | one region per paragraph block, not per line |
| `list` | bulleted or numbered lists | the whole list is one region |
| `table` | the entire table including its rules | do **not** box individual cells |
| `figure` | images, charts, diagrams, maps, logos | includes the government emblem when decorative |
| `caption` | text captioning a figure or table | must be adjacent to what it captions |
| `header` | running header at the top of the page | only if it repeats across pages |
| `footer` | running footer, footnotes | |
| `page_number` | the folio number alone | tight box, digits only |
| `stamp_seal` | official stamps, round seals, embossed marks | **high priority — get these right** |
| `signature` | handwritten signatures | the signature mark, not the printed name below it |

### Boxing rules

1. **Tight but complete.** Include all ink of the region, exclude surrounding whitespace. A couple of
   pixels of margin is fine; ten is not.
2. **Do not overlap** unless the content genuinely overlaps (a seal stamped across text is the common
   real case — box both).
3. **Multi-line headings are one region.** So are multi-paragraph blocks that read as one unit.
4. **Tables are one box.** Cell structure is explicitly out of scope for this project.
5. **Skip decorative rules and borders.** They are not regions.
6. **If a region is ambiguous, flag it** in the shared sheet rather than guessing silently. Ambiguous
   cases resolved consistently are fine; resolved inconsistently they are noise in your metric.

### Reading order

Number regions in the order a human would read them. Body content first, in logical flow. **Page
furniture (header, footer, page number) goes last**, matching what the pipeline produces.

For multi-column pages: finish the entire left column before starting the right one.

---

## 4. Text transcription — the slow part, and the one that matters

Save as UTF-8 `.txt`, one file per page, same stem as the image.

### Rules

1. **Transcribe what is printed, including errors.** If the document misspells a word, so do you.
   This is ground truth, not a correction pass.
2. **Preserve reading order.** Region order in the text file must match your reading-order numbering.
3. **Blank line between regions.** This matches `Document.full_text()`, which joins regions with
   `\n\n` — so the comparison is apples to apples.
4. **Normalise to Unicode NFC.** Most editors do this; verify with the checker below.
5. **Do not transcribe:** stamps, seals, signatures, figures. They are not text-bearing regions in
   this pipeline. Do transcribe printed text *inside* a seal only if it is the sole content of the
   region — and note it in the sheet.
6. **Tables:** transcribe cell contents row by row, tab-separated, one row per line.
7. **Use `[REDACTED]`** where you anonymised content.
8. **Devanagari digits stay Devanagari.** Do not silently convert १२३ to 123 — that is a real
   recognition distinction we want measured.

### Quality check before you commit a file

```bash
python - <<'PY'
import unicodedata, pathlib
for p in pathlib.Path("benchmark/gt_text").glob("*.txt"):
    t = p.read_text(encoding="utf-8")
    if t != unicodedata.normalize("NFC", t):
        print("NOT NFC:", p.name)
    if "�" in t:
        print("REPLACEMENT CHAR (encoding broke):", p.name)
    if not t.strip():
        print("EMPTY:", p.name)
PY
```

---

## 5. Review — every page, by a second person

**This is not optional.** Ground truth with errors in it produces CER numbers that look completely
real and are completely wrong, and you will not catch it downstream because there is nothing to
catch it against.

Reviewer checks:

- [ ] Every visible region has a box, and no box is empty
- [ ] Class labels match the table above
- [ ] Reading order is what a human would actually do
- [ ] Transcription matches the image, character for character
- [ ] Region order in the text file matches the reading-order numbers
- [ ] Anonymisation is complete
- [ ] File is NFC, no replacement characters

Record reviewer initials in the tracking sheet. If a page needs rework, send it back rather than
fixing it yourself — that is how the original annotator learns the convention.

---

## 6. Tracking sheet

Shared sheet, updated daily, visible to everyone:

| page_id | source URL | language | layout | annotator | annotated | reviewer | reviewed | notes |
|---|---|---|---|---|---|---|---|---|
| page_001 | gr.maharashtra… | mr | 2-col | Kalpesh | ✓ | Ayush | ✓ | seal overlaps text |

Daily counts in standup. If the cumulative count falls behind the day number × 7, cut the target to
35 pages **immediately** rather than compressing the review step. A smaller honest benchmark beats a
larger sloppy one, and 35 well-reviewed pages will still support a defensible CER with a confidence
interval.

---

## 7. Final structure

```
benchmark/
  images/       page_001.jpg  page_002.jpg  …
  gt_text/      page_001.txt  page_002.txt  …
  gt_layout/    page_001.json page_002.json …
  metadata.csv  page_id,source_url,language,layout_type,quality,has_seal,annotator,reviewer
```

Then:

```bash
python eval/run_benchmark.py --benchmark benchmark --out results/ --ablations
```

---

## Team decisions

Record every convention you settle during calibration here, so it survives the two weeks.

- *(e.g. "A letterhead at the top of page 1 is `header` only if it repeats on later pages; otherwise `figure` + `title`.")*
- *(e.g. "Bilingual documents: box the Marathi and English blocks as separate regions.")*
-
-
