#!/usr/bin/env python3
"""Select a diverse benchmark subset from the two extracted page-image folders.

Reads (never modifies) the extracted source folders, then creates:
  benchmark/images/BM_001.png ... BM_024.png
  benchmark/metadata.csv
  benchmark/ANNOTATION_GUIDELINES.md

Rules this script follows:
  * Rule 6: nothing is deleted or overwritten. If any target file already
    exists the script reports it and stops before writing anything.
  * Reads only from the source folders; writes only under benchmark/.
  * No torch / model imports; no metric numbers.

The per-page attributes in SELECTION come from a quick visual review of the
pages. They are hints for annotators, not ground truth: "check" means the
reviewer was unsure and the annotator must confirm it.

Usage (from the repo root, inside the venv):
    python eval/select_benchmark_pages.py --dry-run
    python eval/select_benchmark_pages.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmark"
IMAGES_DIR = BENCH_DIR / "images"
METADATA_CSV = BENCH_DIR / "metadata.csv"
GUIDELINES_MD = BENCH_DIR / "ANNOTATION_GUIDELINES.md"

DOC_ZIP = "doc_0001.zip"
DATASET_ZIP = "Dataset1_0.zip"

COLUMNS = [
    "line_id",
    "original_filename",
    "source_zip",
    "language",
    "domain",
    "scan_type",
    "has_table",
    "has_stamp_seal",
    "has_signature",
    "notes",
]


class Page(NamedTuple):
    group: str
    source_zip: str
    filename: str
    language: str
    domain: str
    scan_type: str
    has_table: str
    has_stamp_seal: str
    has_signature: str
    notes: str


def _ds(fn, domain, table, stamp, sig, notes):
    return Page("dataset1", DATASET_ZIP, fn, "marathi", domain, "clean",
                table, stamp, sig, notes)


# Order matters: it fixes the BM_xxx numbering.
SELECTION: list[Page] = [
    # ---- Dataset1: 9 unique pages (Doc010 is a duplicate of Doc009, excluded)
    _ds("Doc001.png", "order_circular", "no", "no", "check",
        "letter-style order without emblem; some Latin script"),
    _ds("Doc002.png", "order_circular", "yes", "yes", "yes",
        "emblem, table, round seal, signature"),
    _ds("Doc003.png", "order_circular", "yes", "yes", "yes",
        "office order with emblem, table, seal, signature"),
    _ds("Doc004.png", "order_circular", "yes", "yes", "yes",
        "circular with emblem; email/URL in Latin script"),
    _ds("Doc005.png", "tender", "yes", "check", "yes",
        "tender notice; Latin script in email/URL/IDs"),
    _ds("Doc006.png", "order_circular", "yes", "check", "yes",
        "order with emblem and boxed table"),
    _ds("Doc007.png", "notice", "yes", "no", "yes",
        "office notice with table"),
    _ds("Doc008.png", "report", "yes", "no", "yes",
        "report with table and two signature blocks"),
    _ds("Doc009.png", "order_circular", "no", "yes", "yes",
        "circular with emblem, seals, signatures; Doc010 is a byte-identical "
        "duplicate and is excluded"),
    # ---- doc_0001: 3 orders / circulars
    Page("orders", DOC_ZIP, "doc_0001.png", "marathi", "order_circular",
         "clean", "no", "no", "check",
         "circular with reference list; clean render"),
    Page("orders", DOC_ZIP, "doc_0026.png", "marathi", "order_circular",
         "clean", "no", "no", "check",
         "order with references and numbered paragraphs"),
    Page("orders", DOC_ZIP, "doc_0031.png", "marathi", "order_circular",
         "scan", "no", "check", "yes",
         "letter with emblem at top and handwritten signature"),
    # ---- doc_0001: 3 wide-table pages
    Page("wide_table", DOC_ZIP, "doc_0014.png", "marathi", "wide_table",
         "clean", "yes", "no", "no",
         "landscape page, dense multi-column table"),
    Page("wide_table", DOC_ZIP, "doc_0016.png", "marathi", "wide_table",
         "clean", "yes", "no", "no",
         "landscape table with highlighted (yellow) cells"),
    Page("wide_table", DOC_ZIP, "doc_0036.png", "marathi", "wide_table",
         "scan_rotated90", "yes", "check", "no",
         "page rotated 90 degrees, table text runs sideways; see guidelines"),
    # ---- doc_0001: 3 camera-photo pages
    Page("camera", DOC_ZIP, "doc_0004.png", "marathi", "order_circular",
         "camera", "check", "check", "check",
         "phone photo: skewed, grey background, punch holes"),
    Page("camera", DOC_ZIP, "doc_0005.png", "marathi", "order_circular",
         "camera", "check", "check", "check",
         "phone photo: skewed, grey background, signature near bottom"),
    Page("camera", DOC_ZIP, "doc_0023.png", "marathi", "order_circular",
         "camera", "check", "check", "yes",
         "photographed page with signature"),
    # ---- doc_0001: 3 transfer-list pages (sampled from a repeated template)
    Page("transfer_list", DOC_ZIP, "doc_0062.png", "marathi", "transfer_list",
         "clean", "yes", "no", "no",
         "several small tables with district headings"),
    Page("transfer_list", DOC_ZIP, "doc_0079.png", "marathi", "transfer_list",
         "clean", "yes", "no", "no",
         "title line above a table; tests heading vs table split"),
    Page("transfer_list", DOC_ZIP, "doc_0095.png", "marathi", "transfer_list",
         "clean", "yes", "no", "no",
         "table continued from previous page plus new heading and table"),
    # ---- doc_0001: 3 other diverse pages
    Page("other", DOC_ZIP, "doc_0007.png", "english", "tender",
         "clean", "yes", "no", "check",
         "English tender corrigendum with table"),
    Page("other", DOC_ZIP, "doc_0032.png", "marathi", "other",
         "scan_rotated90", "no", "no", "no",
         "rotated title-style page with large heading text; see guidelines"),
    Page("other", DOC_ZIP, "doc_0039.png", "marathi", "form_table",
         "clean", "yes", "no", "no",
         "form-style tables under a heading block"),
]

EXPECTED_GROUPS = {
    "dataset1": 9, "orders": 3, "wide_table": 3,
    "camera": 3, "transfer_list": 3, "other": 3,
}
EXCLUDED_DUPLICATE = "Doc010.png"
DUPLICATE_OF = "Doc009.png"

GUIDELINES = """\
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
"""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_source(base: Path, name: str) -> Path | None:
    """Find `name` in `base`, also accepting a nested 'Dataset' subfolder and
    different letter case."""
    for candidate in (base / name, base / "Dataset" / name):
        if candidate.is_file():
            return candidate
    if base.is_dir():
        wanted = name.lower()
        for candidate in base.rglob("*"):
            if candidate.is_file() and candidate.name.lower() == wanted:
                return candidate
    return None


def check_selection() -> None:
    if len({(p.source_zip, p.filename) for p in SELECTION}) != len(SELECTION):
        raise SystemExit("internal error: duplicate page in SELECTION")
    if any(p.filename == EXCLUDED_DUPLICATE for p in SELECTION):
        raise SystemExit("internal error: duplicate page must stay excluded")
    counts = Counter(p.group for p in SELECTION)
    if dict(counts) != EXPECTED_GROUPS:
        raise SystemExit(f"internal error: group counts {dict(counts)} do not "
                         f"match the plan {EXPECTED_GROUPS}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--doc-dir", type=Path,
                        default=REPO_ROOT.parent / "doc_0001",
                        help="folder with the extracted doc_0001 pages")
    parser.add_argument("--dataset-dir", type=Path,
                        default=REPO_ROOT.parent / "Dataset1",
                        help="folder with the extracted Dataset1 pages")
    parser.add_argument("--dry-run", action="store_true",
                        help="check sources and targets, write nothing")
    args = parser.parse_args(argv)

    check_selection()
    total = len(SELECTION)
    names = [f"BM_{i:03d}.png" for i in range(1, total + 1)]
    targets = [IMAGES_DIR / n for n in names] + [METADATA_CSV, GUIDELINES_MD]

    existing = [t for t in targets if t.exists()]
    if existing:
        print("Refusing to continue: these targets already exist and will not "
              "be overwritten:")
        for path in existing:
            print(f"  {path}")
        print("Nothing was written. Move them aside yourself if you really "
              "want to regenerate.")
        return 2

    base_for = {DOC_ZIP: args.doc_dir, DATASET_ZIP: args.dataset_dir}
    resolved: list[Path] = []
    missing: list[str] = []
    for page in SELECTION:
        found = find_source(base_for[page.source_zip], page.filename)
        if found is None:
            missing.append(f"{page.source_zip}: {page.filename} "
                           f"(looked in {base_for[page.source_zip]})")
        else:
            resolved.append(found)
    if missing:
        print("Source pages not found; nothing was written:")
        for line in missing:
            print(f"  {line}")
        return 2

    dup_a = find_source(args.dataset_dir, DUPLICATE_OF)
    dup_b = find_source(args.dataset_dir, EXCLUDED_DUPLICATE)
    if dup_a and dup_b:
        same = sha256_of(dup_a) == sha256_of(dup_b)
        print(f"{EXCLUDED_DUPLICATE} identical to {DUPLICATE_OF}: {same} "
              f"({EXCLUDED_DUPLICATE} is excluded either way)")
        if not same:
            print("  WARNING: the files differ, so check by eye whether "
                  f"{EXCLUDED_DUPLICATE} should be included.")

    print(f"\nPlan: {total} pages -> {IMAGES_DIR}")
    for name, page, src in zip(names, SELECTION, resolved):
        print(f"  {name}  <- {page.source_zip}: {src.name}  [{page.group}]")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        for name, src in zip(names, resolved):
            dst = IMAGES_DIR / name
            with open(src, "rb") as fin, open(dst, "xb") as fout:
                while True:
                    chunk = fin.read(1 << 20)
                    if not chunk:
                        break
                    fout.write(chunk)
            written.append(dst)

        # utf-8-sig so Excel on Windows shows non-ASCII notes correctly.
        with open(METADATA_CSV, "x", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(COLUMNS)
            for name, page in zip(names, SELECTION):
                writer.writerow([
                    name[:-4], page.filename, page.source_zip, page.language,
                    page.domain, page.scan_type, page.has_table,
                    page.has_stamp_seal, page.has_signature, page.notes,
                ])
        written.append(METADATA_CSV)

        with open(GUIDELINES_MD, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(GUIDELINES)
        written.append(GUIDELINES_MD)
    except Exception as exc:  # report what exists so nothing is left unknown
        print(f"\nFAILED: {exc!r}")
        print("Files created before the failure (not deleted):")
        for path in written:
            print(f"  {path}")
        return 1

    print(f"\nCopied {total} images, wrote {METADATA_CSV.name} and "
          f"{GUIDELINES_MD.name}.")
    by_group = Counter(p.group for p in SELECTION)
    print("Pages per group: " + ", ".join(f"{k}={v}" for k, v in by_group.items()))
    print("Next: python eval/prelabel_benchmark.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
