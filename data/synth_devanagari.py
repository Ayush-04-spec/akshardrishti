#!/usr/bin/env python3
"""Synthetic Devanagari / Latin line generator.

Two uses:
  1. Harden the CRNN against real scan degradation (Mozhi crops are relatively
     clean; Maharashtra GR scans are not).
  2. Manufacture balanced training data for the script-ID classifier, where
     real labelled Latin line crops are otherwise awkward to source.

Fonts
-----
Needs Devanagari-capable TTFs. On Colab/Kaggle:
    apt-get install -y fonts-indic fonts-noto-core fonts-deva
or download Noto Sans/Serif Devanagari from Google Fonts.

    python data/synth_devanagari.py --corpus corpus_hi.txt --n 20000 \
        --out datasets/synth --script deva
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("synth")

FONT_SEARCH = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/fonts-deva-extra/",
    "/usr/share/fonts/truetype/lohit-devanagari/",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/",
]


def discover_fonts(extra: list[str] | None = None) -> list[Path]:
    fonts: list[Path] = []
    for entry in (extra or []) + FONT_SEARCH:
        path = Path(entry)
        if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}:
            fonts.append(path)
        elif path.is_dir():
            fonts.extend(p for p in path.rglob("*") if p.suffix.lower() in {".ttf", ".otf"})
    unique = sorted({p.resolve() for p in fonts})
    log.info("found %d fonts", len(unique))
    return unique


def paper_background(w: int, h: int, rng: random.Random) -> np.ndarray:
    """Off-white background with subtle grain and a gradient, so the model does
    not learn 'text is black on pure white' -- which real scans never are."""
    base = rng.randint(228, 252)
    img = np.full((h, w), base, np.float32)
    img += np.random.normal(0, rng.uniform(1.5, 5.0), (h, w))

    if rng.random() < 0.6:  # illumination gradient, like a photographed page
        axis = np.linspace(rng.uniform(-14, 0), rng.uniform(0, 14), w if rng.random() < 0.5 else h)
        img += axis[None, :] if len(axis) == w else axis[:, None]

    if rng.random() < 0.25:  # speckle from old paper
        n = rng.randint(5, 40)
        ys = np.random.randint(0, h, n)
        xs = np.random.randint(0, w, n)
        img[ys, xs] = np.random.randint(90, 170, n)

    return np.clip(img, 0, 255).astype(np.uint8)


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        k = rng.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), rng.uniform(0.4, 1.1))
    if rng.random() < 0.35:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8)) if rng.random() < 0.5 \
            else cv2.erode(img, np.ones((2, 2), np.uint8))
    if rng.random() < 0.4:
        img = np.clip(img.astype(np.float32) + np.random.normal(0, rng.uniform(3, 14), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # JPEG artefacts -- most real scans have been through one
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 80)])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.3:  # slight rotation
        angle = rng.uniform(-1.2, 1.2)
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def render_line(text: str, font_path: Path, size: int, rng: random.Random) -> np.ndarray | None:
    try:
        font = ImageFont.truetype(str(font_path), size)
    except OSError:
        return None

    pad_x, pad_y = rng.randint(6, 20), rng.randint(5, 14)
    tmp = Image.new("L", (10, 10), 255)
    try:
        box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    except Exception:  # noqa: BLE001 - font lacks the glyphs
        return None

    tw, th = box[2] - box[0], box[3] - box[1]
    if tw <= 0 or th <= 0:
        return None

    w, h = tw + pad_x * 2, th + pad_y * 2
    canvas = Image.fromarray(paper_background(w, h, rng))
    ink = rng.randint(0, 70)
    ImageDraw.Draw(canvas).text((pad_x - box[0], pad_y - box[1]), text, font=font, fill=ink)

    img = np.array(canvas)
    # Sanity check: if almost no ink landed, the font lacked the glyphs and
    # rendered tofu boxes or nothing. Silently keeping these poisons training.
    if (img < 128).mean() < 0.005:
        return None
    return degrade(img, rng)


def load_corpus(path: Path, min_len: int, max_len: int) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = " ".join(raw.split())
        if min_len <= len(text) <= max_len:
            lines.append(text)
    log.info("corpus: %d usable lines from %s", len(lines), path)
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True, help="UTF-8 text file, one line per sample")
    p.add_argument("--out", type=Path, default=Path("datasets/synth"))
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--script", choices=["deva", "latin"], default="deva")
    p.add_argument("--fonts", nargs="*", default=None, help="extra font files or directories")
    p.add_argument("--min-size", type=int, default=22)
    p.add_argument("--max-size", type=int, default=52)
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--max-len", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    fonts = discover_fonts(args.fonts)
    if not fonts:
        raise SystemExit(
            "No fonts found. Install some:\n"
            "  apt-get install -y fonts-indic fonts-noto-core\n"
            "or pass --fonts /path/to/NotoSansDevanagari-Regular.ttf"
        )

    lines = load_corpus(args.corpus, args.min_len, args.max_len)
    if not lines:
        raise SystemExit("corpus produced no usable lines -- check --min-len/--max-len")

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows, made, attempts = [], 0, 0
    max_attempts = args.n * 6

    while made < args.n and attempts < max_attempts:
        attempts += 1
        text = rng.choice(lines)
        img = render_line(text, rng.choice(fonts), rng.randint(args.min_size, args.max_size), rng)
        if img is None:
            continue
        name = f"{args.script}_{made:07d}.jpg"
        cv2.imwrite(str(img_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"image_path": str((img_dir / name).resolve()), "text": text,
                     "language": "synthetic", "split": "train", "script": args.script})
        made += 1
        if made % 2000 == 0:
            log.info("generated %d/%d", made, args.n)

    index = args.out / f"index_{args.script}.csv"
    with open(index, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "text", "language", "split", "script"])
        writer.writeheader()
        writer.writerows(rows)

    reject_rate = 1 - (made / max(attempts, 1))
    log.info("wrote %d samples -> %s", made, index)
    if reject_rate > 0.4:
        log.warning("rejected %.0f%% of renders -- your fonts probably lack Devanagari glyphs", reject_rate * 100)


if __name__ == "__main__":
    main()
