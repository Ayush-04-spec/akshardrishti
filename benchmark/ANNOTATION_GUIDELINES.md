# AksharDrishti Benchmark: Annotation Guidelines

These guidelines cover the pages in `benchmark/images/` (`BM_001.png` onward).
Per-page information is in `benchmark/metadata.csv`.

All annotators must apply the rules below the same way. If a rule is unclear,
ask the lead and add the decision to the log at the end of this file instead
of choosing privately.

## 1. Workflow

1. Open a page from `benchmark/images/` together with its pre-labels from
   `benchmark/gt_layout_prelabeled/` (created by `eval/prelabel_benchmark.py`).
2. Correct the pre-labels: remove wrong boxes, fix classes, tighten boxes and
   add missing ones. The pre-labels come from a detector that never saw a
   stamp or a signature, so add every `stamp_seal` and `signature` by hand.
3. Export in YOLO format (one `.txt` per image, same file stem as the image,
   normalized `class cx cy w h`).
4. Put the finished, reviewed file in `benchmark/gt_layout/`. Keep the
   original pre-label file where it is.
5. Never edit, rename, rotate or re-save the image files, and never delete
   anything. Deletions need explicit approval from the lead.

## 2. The 11 layout classes

| id | class | what to box |
|----|-------|-------------|
| 0 | title | The document title or subject line (for example a document-type heading such as a GR or circular heading, or the "subject" line), and section headings, including a district or division heading above a table. |
| 1 | text | Running paragraph text: body paragraphs and the typed name and designation under a signature. One box per paragraph. |
| 2 | list | A group of numbered or bulleted items, including a reference list and the forwarding or copy-to list. Box the whole list as one region. |
| 3 | table | A whole table including its header row and outer border. One box per table. |
| 4 | figure | Non-text graphics: the state emblem, logos, photos, diagrams, QR codes. The printed state emblem is a figure, not a stamp_seal. |
| 5 | caption | A label tied to one specific figure or table, such as "Table 1" or "Figure 2". A heading that merely sits above a table is a title. |
| 6 | header | The letterhead block in the top margin (government and department name, address line, reference number and date block), and running page headers. |
| 7 | footer | Other text in the bottom margin, such as a repeated contact line or website line. |
| 8 | page_number | A printed page number or a "page x of y" marker. |
| 9 | stamp_seal | Round or rectangular office seals, rubber stamps and ink stamps. |
| 10 | signature | A handwritten signature mark, or a digital-signature validity block. Box only the signature itself. |

Class ids follow the table above (0 to 10) and must be used as written.

General rules:

* Boxes are axis-aligned rectangles, as tight as possible around the content.
* Annotate every region, including small ones (page numbers, stamps).
* Do not box individual table rows or cells.
* Regions normally do not overlap. A caption may sit next to its figure or
  table, and a stamp or signature may overlap text; box the stamp or signature
  itself and leave the text box as it is.
* If you cannot decide between two classes, pick the closer one, put a note in
  `metadata.csv`, and raise it with the lead.

## 3. Metadata hints

`has_table`, `has_stamp_seal`, `has_signature` and `domain` in
`benchmark/metadata.csv` were pre-filled from a quick visual review. `check`
means the reviewer was unsure. Confirm or correct every value while annotating
and fix the row if it is wrong.

## 4. Stamps, seals and signatures

The layout detector was trained with no examples of these two classes, so the
pre-labels will not contain them. Annotate them carefully on every page that
has them, because they will be reported as a separate group.

## 5. Camera-photo pages (`scan_type=camera`)

* Annotate the page exactly as it appears. Do not deskew, crop or enhance the
  image file.
* Use axis-aligned boxes that enclose each skewed block as tightly as you can.
* Ignore the background around the page, shadows, punch holes and page edges.
  They are not figures.
* If text is cut off at the image edge, box the visible part.
* If text is too blurred to read, still box it with its normal class and
  transcribe it as `[illegible]`.

## 6. Pages rotated by 90 degrees (`scan_type=scan_rotated90`)

* Keep the image file as it is. The ground truth must match the pixels the
  pipeline sees, so do not rotate the file.
* Draw boxes on the image as displayed (axis-aligned around the sideways
  text).
* In the `notes` column of `metadata.csv`, record which way the page must be
  turned to read it upright, for example "turn 90 degrees clockwise".
* Decide classes, reading order and transcription as if the page were upright
  (top of the upright page = top of the document).
* These pages are a separate group in the evaluation, because the pipeline is
  not guaranteed to handle page rotation.

## 7. Repeated transfer-list table pages (`domain=transfer_list`)

The source folder holds many near-identical transfer or posting list pages.
Only three are included, so do not add more without the lead's approval.

* Box each table once as `table`, including its header row. A page can hold
  several tables, one per district or division. Box each separately.
* A district, division or section heading above a table is a `title` and is
  boxed on its own.
* A table that continues from the previous page and starts at the top without
  a heading is still a `table`.
* Do not box rows or cells.
* For transcription, write each table row on one line with cells separated by
  ` | `, copied exactly as printed. Agree the separator with the team before
  starting and use it on every page.

## 8. Text transcription

* Transcribe verbatim: keep the original spelling, punctuation and digits
  (Devanagari digits stay Devanagari). Do not correct or normalize anything,
  because scoring applies its own normalization to the ground truth side.
* Reading order follows the order in which a person would read the page.
* The transcription file format will be fixed together with the evaluation
  script. Until then, keep one transcription file per page with one printed
  line per text line.

## 9. Decisions log

(Add dated entries here when the team settles an unclear case.)
