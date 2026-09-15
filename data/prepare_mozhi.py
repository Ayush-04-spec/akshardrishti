#!/usr/bin/env python3
"""Download and index the Mozhi dataset for Devanagari recognition training.

Mozhi (IIIT-H CVIT) is ~1.2M annotated word images across 13 Indian languages,
released alongside "Towards Deployable OCR Models for Indic Languages" -- the
work behind Bhashini's printed-OCR service. Training our CRNN on it is what
makes the Bhashini comparison a like-for-like one.

Usage
-----
    python data/prepare_mozhi.py --languages hindi marathi --out datasets/mozhi
    python data/prepare_mozhi.py --languages hindi marathi --out datasets/mozhi --charset-only

Layout produced
---------------
    datasets/mozhi/
      hindi/{train,val,test}/...        (as shipped in the zips)
      marathi/{train,val,test}/...
      index_train.csv                   image_path,text,language,split
      index_val.csv
      index_test.csv
      charset_deva.txt                  charset built from TRAIN ONLY
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_mozhi")

BASE_URL = "https://ilocr.iiit.ac.in/public/printed/phase-0/v0.5"
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    import requests

    if dest.exists() and dest.stat().st_size > 0:
        log.info("already downloaded: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=chunk):
                fh.write(block)
                done += len(block)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {dest.name}: {pct:5.1f}%  ({done/1e6:.0f}/{total/1e6:.0f} MB)", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


def extract(zip_path: Path, dest_dir: Path) -> Path:
    marker = dest_dir / ".extracted"
    if marker.exists():
        log.info("already extracted: %s", dest_dir)
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("extracting %s -> %s", zip_path.name, dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    marker.write_text("ok", encoding="utf-8")
    return dest_dir


# ------------------------------------------------------------------- indexing


def find_annotation_file(root: Path) -> Path | None:
    """Mozhi ships transcriptions in a plain text file whose name varies by
    release. Look for the usual suspects, then fall back to any .txt with
    tab/space separated `path text` lines."""
    for name in ("gt.txt", "annotation.txt", "annotations.txt", "labels.txt", "train.txt", "val.txt", "test.txt"):
        for hit in root.rglob(name):
            return hit
    for candidate in root.rglob("*.txt"):
        try:
            head = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
        except OSError:
            continue
        if head and all((("\t" in ln) or (" " in ln.strip())) for ln in head if ln.strip()):
            return candidate
    return None


def parse_annotations(ann_path: Path, root: Path) -> list[tuple[Path, str]]:
    """Parse `<relative_image_path><sep><transcription>` lines."""
    pairs: list[tuple[Path, str]] = []
    misses = 0

    for raw in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t", 1) if "\t" in line else line.split(" ", 1)
        if len(parts) != 2:
            continue
        rel, text = parts[0].strip(), parts[1].strip()
        if not text:
            continue

        img = (ann_path.parent / rel)
        if not img.exists():
            img = root / rel
        if not img.exists():
            stem = Path(rel).name
            found = next((p for p in root.rglob(stem)), None)
            if found is None:
                misses += 1
                continue
            img = found
        pairs.append((img, unicodedata.normalize("NFC", text)))

    if misses:
        log.warning("%d annotation rows had no matching image in %s", misses, root)
    return pairs


def index_language(lang_dir: Path, language: str, split: str) -> list[dict]:
    split_dir = lang_dir / split
    root = split_dir if split_dir.exists() else lang_dir

    ann = find_annotation_file(root)
    rows: list[dict] = []

    if ann is None:
        log.warning("no annotation file under %s -- falling back to filename-as-text", root)
        for img in sorted(root.rglob("*")):
            if img.suffix.lower() in IMAGE_EXTS:
                rows.append({"image_path": str(img.resolve()), "text": unicodedata.normalize("NFC", img.stem),
                             "language": language, "split": split})
        return rows

    log.info("[%s/%s] annotations: %s", language, split, ann.relative_to(lang_dir.parent))
    for img, text in parse_annotations(ann, root):
        rows.append({"image_path": str(img.resolve()), "text": text, "language": language, "split": split})
    return rows


def write_index(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "text", "language", "split"])
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d rows)", path, len(rows))


def build_charset(train_rows: list[dict], out_path: Path, min_count: int = 2) -> list[str]:
    """Charset from the TRAIN split only.

    Building it from all splits leaks test information and, worse, hides the
    real problem: any validation character absent from the charset is an
    unavoidable error that puts a hard floor under your CER. The coverage
    report below tells you what that floor is.
    """
    counts = Counter(ch for row in train_rows for ch in row["text"])
    chars = sorted(ch for ch, n in counts.items() if n >= min_count and ch not in ("\n", "\r"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chars), encoding="utf-8")

    rare = [ch for ch, n in counts.items() if n < min_count]
    log.info("charset: %d chars (dropped %d rare, min_count=%d) -> %s", len(chars), len(rare), min_count, out_path)
    return chars


def report_coverage(chars: list[str], rows: list[dict], label: str) -> float:
    charset = set(chars)
    total = covered = 0
    missing: Counter = Counter()
    for row in rows:
        for ch in row["text"]:
            total += 1
            if ch in charset:
                covered += 1
            else:
                missing[ch] += 1
    frac = covered / total if total else 1.0
    log.info("charset coverage on %s: %.4f%%  (hard CER floor >= %.4f)", label, frac * 100, 1 - frac)
    if missing:
        preview = ", ".join(f"{repr(c)}x{n}" for c, n in missing.most_common(12))
        log.info("  most common missing chars: %s", preview)
    return frac


# ------------------------------------------------------------------------ cli


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", nargs="+", default=["hindi", "marathi"])
    p.add_argument("--out", type=Path, default=Path("datasets/mozhi"))
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--min-char-count", type=int, default=2)
    p.add_argument("--skip-download", action="store_true", help="index already-extracted data")
    p.add_argument("--charset-only", action="store_true", help="rebuild charset from existing index_train.csv")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.charset_only:
        with open(args.out / "index_train.csv", encoding="utf-8") as fh:
            train_rows = list(csv.DictReader(fh))
        chars = build_charset(train_rows, args.out / "charset_deva.txt", args.min_char_count)
        for split in ("val", "test"):
            idx = args.out / f"index_{split}.csv"
            if idx.exists():
                with open(idx, encoding="utf-8") as fh:
                    report_coverage(chars, list(csv.DictReader(fh)), split)
        return

    per_split: dict[str, list[dict]] = {s: [] for s in args.splits}

    for language in args.languages:
        lang_dir = args.out / language
        for split in args.splits:
            if not args.skip_download:
                url = f"{args.base_url}/{language}/akshara/{split}.zip"
                try:
                    zip_path = download(url, args.out / "_zips" / f"{language}_{split}.zip")
                    extract(zip_path, lang_dir / split)
                except Exception as exc:  # noqa: BLE001
                    log.error("failed %s/%s: %s", language, split, exc)
                    log.error("  check the URL in a browser -- paths shift between Mozhi releases: %s", url)
                    continue
            per_split[split].extend(index_language(lang_dir, language, split))

    for split, rows in per_split.items():
        if rows:
            write_index(rows, args.out / f"index_{split}.csv")

    if per_split.get("train"):
        chars = build_charset(per_split["train"], args.out / "charset_deva.txt", args.min_char_count)
        for split in ("val", "test"):
            if per_split.get(split):
                report_coverage(chars, per_split[split], split)

    print("\nSummary")
    for split, rows in per_split.items():
        by_lang = Counter(r["language"] for r in rows)
        print(f"  {split:<6} {len(rows):>8} samples   {dict(by_lang)}")


if __name__ == "__main__":
    main()
