"""PyTorch datasets for CRNN and script-ID training."""

from __future__ import annotations

import csv
import logging
import random
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def load_index(csv_path: str | Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class LineImageDataset(Dataset):
    """Height-normalised line/word crops with transcriptions, for CTC training.

    Images keep their aspect ratio (variable width); the collate function pads a
    batch to its widest member. Squashing every crop to a fixed width would
    destroy the aspect information Devanagari conjuncts depend on.
    """

    def __init__(
        self,
        rows: list[dict],
        charset,
        img_height: int = 32,
        max_width: int = 512,
        augment: bool = False,
        binarize: bool = True,
        drop_uncovered: bool = True,
    ) -> None:
        self.charset = charset
        self.img_height = img_height
        self.max_width = max_width
        self.augment = augment
        self.binarize = binarize

        if drop_uncovered:
            kept = [r for r in rows if charset.coverage(r["text"]) >= 0.999]
            dropped = len(rows) - len(kept)
            if dropped:
                log.info("dropped %d/%d samples with out-of-charset chars", dropped, len(rows))
            self.rows = kept
        else:
            self.rows = list(rows)

        if not self.rows:
            raise ValueError("dataset is empty after filtering")

    def __len__(self) -> int:
        return len(self.rows)

    # ------------------------------------------------------------ augment
    def _augment(self, img: np.ndarray) -> np.ndarray:
        rng = random.Random()

        if rng.random() < 0.3:  # slight rotation -- scans are never perfectly straight
            angle = rng.uniform(-1.5, 1.5)
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if rng.random() < 0.3:  # horizontal stretch -- font width variation
            factor = rng.uniform(0.9, 1.1)
            img = cv2.resize(img, None, fx=factor, fy=1.0, interpolation=cv2.INTER_LINEAR)

        if rng.random() < 0.25:  # scan noise
            noise = np.random.normal(0, rng.uniform(3, 12), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if rng.random() < 0.2:  # blur -- low-DPI scans
            k = rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        if rng.random() < 0.2:  # brightness/contrast
            alpha = rng.uniform(0.8, 1.2)
            beta = rng.uniform(-25, 25)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if rng.random() < 0.15:  # dilate/erode -- ink weight varies with print run
            k = np.ones((2, 2), np.uint8)
            img = cv2.dilate(img, k) if rng.random() < 0.5 else cv2.erode(img, k)

        return img

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((self.img_height, 32), 255, np.uint8)

        if self.augment:
            img = self._augment(img)

        h, w = img.shape[:2]
        scale = self.img_height / max(h, 1)
        new_w = max(8, min(int(round(w * scale)), self.max_width))
        img = cv2.resize(img, (new_w, self.img_height),
                         interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)

        if self.binarize:
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        tensor = ((tensor - 0.5) / 0.5).unsqueeze(0)  # (1, H, W)

        text = unicodedata.normalize("NFC", row["text"])
        target = torch.tensor(self.charset.encode(text), dtype=torch.long)
        return tensor, target, len(target), text


def ctc_collate(batch):
    """Pad a batch to its widest image; concatenate targets for CTC."""
    images, targets, target_lengths, texts = zip(*batch)
    max_w = max(img.shape[-1] for img in images)

    padded = torch.stack([
        torch.nn.functional.pad(img, (0, max_w - img.shape[-1]), value=1.0) for img in images
    ])

    # CTC input length = feature width after the CNN, which downsamples width /4.
    input_lengths = torch.full((len(images),), max_w // 4, dtype=torch.long)
    flat_targets = torch.cat([t for t in targets]) if targets else torch.tensor([], dtype=torch.long)
    return padded, flat_targets, input_lengths, torch.tensor(target_lengths, dtype=torch.long), list(texts)


class ScriptIDDataset(Dataset):
    """Line crops labelled deva=0 / latin=1, for the routing classifier."""

    LABELS = {"deva": 0, "latin": 1}

    def __init__(self, rows: list[dict], img_size: tuple[int, int] = (32, 128), augment: bool = False) -> None:
        self.rows = [r for r in rows if r.get("script") in self.LABELS]
        if not self.rows:
            raise ValueError("no rows with a valid 'script' column (deva/latin)")
        self.h, self.w = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((self.h, self.w), 255, np.uint8)

        if self.augment and random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(3, 12), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        tensor = ((tensor - 0.5) / 0.5).unsqueeze(0)
        return tensor, self.LABELS[row["script"]]
