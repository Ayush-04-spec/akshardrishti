"""
Pseudo-label government document images using Tesseract.

Pipeline:
  1. Load each govt doc image
  2. Preprocess (deskew, denoise, CLAHE) using the same pipeline as AksharDrishti
  3. Run Tesseract with hin+mar+eng to get word-level bounding boxes + text
  4. Merge words into line crops using Tesseract's line block info
  5. Filter: keep only Devanagari-majority lines, drop very short/garbled text
  6. Save (crop_image, text) pairs to datasets/pseudo_govdoc/

Output dataset is compatible with the existing govdoc_finetune format
(index.csv + images/ dir), so the Colab fine-tune notebook works unchanged.

Usage:
    python scripts/pseudo_label_govdoc.py
    python scripts/pseudo_label_govdoc.py --min-conf 50 --min-chars 3
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytesseract

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point pytesseract at the installed binary
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Source image directories ─────────────────────────────────────────────────
SOURCE_DIRS = [
    Path(r"D:\AksharDrishti_1\Dataset"),                                         # 10 Maharashtra docs
    Path(r"D:\AksharDrishti_1\aksharDrishti\central_extracted\Central_Government_Dataset\images"),  # 74 central docs
]

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = ROOT / "datasets" / "pseudo_govdoc"
OUT_IMAGES = OUT_DIR / "images"
OUT_CSV = OUT_DIR / "index.csv"


def collect_images(source_dirs: list[Path]) -> list[Path]:
    images = []
    for d in source_dirs:
        if not d.exists():
            print(f"  WARNING: source dir not found: {d}")
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            images.extend(d.rglob(ext))
    return sorted(set(images))


def preprocess(img: np.ndarray) -> np.ndarray:
    """Same preprocessing as AksharDrishti pipeline."""
    # Convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Resize if too small (Tesseract struggles below 150 DPI equivalent)
    h, w = gray.shape
    if max(h, w) < 1000:
        scale = 1500 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=7)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def has_devanagari(text: str, threshold: float = 0.3) -> bool:
    """Returns True if at least `threshold` fraction of alpha chars are Devanagari."""
    deva = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return False
    return deva / alpha >= threshold


def is_valid_line(text: str, min_chars: int) -> bool:
    """Filter out garbage pseudo-labels."""
    text = text.strip()
    if len(text) < min_chars:
        return False
    # Must have at least some alphanumeric content
    alnum = sum(1 for c in text if c.isalnum())
    if alnum < 2:
        return False
    # Reject lines that are mostly punctuation/symbols
    if alnum / len(text) < 0.3:
        return False
    return True


def extract_line_crops(
    img_gray: np.ndarray,
    img_path: Path,
    min_conf: int,
    min_chars: int,
    pad: int = 4,
) -> list[tuple[np.ndarray, str]]:
    """Run Tesseract and extract line-level (crop, text) pairs."""

    # Use HOCR output to get line-level blocks with confidence
    try:
        data = pytesseract.image_to_data(
            img_gray,
            lang="hin+mar+eng",
            config="--psm 3 --oem 1",   # auto page seg, LSTM only
            output_type=pytesseract.Output.DICT,
        )
    except Exception as e:
        print(f"    Tesseract error on {img_path.name}: {e}")
        return []

    h, w = img_gray.shape

    # Group words into lines using (block_num, par_num, line_num)
    lines: dict[tuple, list[int]] = {}
    n = len(data["text"])
    for i in range(n):
        conf = int(data["conf"][i])
        text = str(data["text"][i]).strip()
        if conf < min_conf or not text:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)

    crops: list[tuple[np.ndarray, str]] = []

    for key, indices in lines.items():
        # Gather bboxes and text for this line
        texts = [str(data["text"][i]).strip() for i in indices]
        line_text = " ".join(t for t in texts if t)

        if not is_valid_line(line_text, min_chars):
            continue

        # Devanagari filter — keep if it has any Devanagari or if it's meaningful English
        if not has_devanagari(line_text, threshold=0.2):
            # Allow pure English lines too (useful for mixed-script docs)
            latin = sum(1 for c in line_text if c.isalpha() and ord(c) < 0x0900)
            if latin < 4:
                continue

        # Compute bounding box of entire line
        x1 = min(data["left"][i] for i in indices)
        y1 = min(data["top"][i] for i in indices)
        x2 = max(data["left"][i] + data["width"][i] for i in indices)
        y2 = max(data["top"][i] + data["height"][i] for i in indices)

        # Add padding, clamp to image bounds
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        if x2 - x1 < 10 or y2 - y1 < 5:
            continue

        crop = img_gray[y1:y2, x1:x2]
        crops.append((crop, line_text))

    return crops


def main(min_conf: int, min_chars: int, augment: bool) -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    images = collect_images(SOURCE_DIRS)
    print(f"Found {len(images)} source images")
    print(f"Output: {OUT_DIR}\n")

    all_rows: list[dict] = []
    total_crops = 0
    skipped_images = 0

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ERROR: cannot load image")
            skipped_images += 1
            continue

        gray = preprocess(img)
        crops = extract_line_crops(gray, img_path, min_conf, min_chars)

        if not crops:
            print(f"  No valid crops")
            skipped_images += 1
            continue

        print(f"  {len(crops)} line crops", flush=True)

        stem = img_path.stem
        for crop_idx, (crop, text) in enumerate(crops):
            # Save original crop
            fname = f"{stem}_{crop_idx:04d}.png"
            cv2.imwrite(str(OUT_IMAGES / fname), crop)
            all_rows.append({"image": fname, "text": text, "source": str(img_path), "augmented": "no"})
            total_crops += 1

            if augment:
                # Augmentation 1: slight Gaussian noise
                noisy = crop.astype(np.float32)
                noisy += np.random.normal(0, 8, noisy.shape)
                noisy = np.clip(noisy, 0, 255).astype(np.uint8)
                fname_aug1 = f"{stem}_{crop_idx:04d}_aug1.png"
                cv2.imwrite(str(OUT_IMAGES / fname_aug1), noisy)
                all_rows.append({"image": fname_aug1, "text": text, "source": str(img_path), "augmented": "yes"})
                total_crops += 1

                # Augmentation 2: slight dilation (simulates bold/heavier ink)
                kernel = np.ones((2, 2), np.uint8)
                dilated = cv2.dilate(crop, kernel, iterations=1)
                fname_aug2 = f"{stem}_{crop_idx:04d}_aug2.png"
                cv2.imwrite(str(OUT_IMAGES / fname_aug2), dilated)
                all_rows.append({"image": fname_aug2, "text": text, "source": str(img_path), "augmented": "yes"})
                total_crops += 1

    # Write index CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text", "source", "augmented"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 60)
    print(f"DONE")
    print(f"  Source images processed : {len(images) - skipped_images}/{len(images)}")
    print(f"  Total crops generated   : {total_crops}")
    print(f"  CSV written to          : {OUT_CSV}")
    print(f"  Images written to       : {OUT_IMAGES}")
    print("=" * 60)

    # Print a few sample labels so you can eyeball quality
    print("\nSample labels (first 10):")
    for row in all_rows[:10]:
        print(f"  {row['image']}: {row['text'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-label govt docs with Tesseract")
    parser.add_argument("--min-conf", type=int, default=40,
                        help="Minimum Tesseract word confidence to include (0-100)")
    parser.add_argument("--min-chars", type=int, default=3,
                        help="Minimum characters in a line to keep")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip augmentation (3x -> 1x crop count)")
    args = parser.parse_args()

    main(
        min_conf=args.min_conf,
        min_chars=args.min_chars,
        augment=not args.no_augment,
    )
