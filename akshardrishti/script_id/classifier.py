"""Script identification -- routes each crop to the right recognizer.

Two methods:

``unicode_vote``
    Zero-cost, no training. Run a fast first-pass OCR and count Unicode ranges.
    Only usable when you already have a multilingual first pass; we use it as
    the fallback and as an ablation baseline.

``cnn``
    A small purpose-trained CNN over the line crop. Trains in ~2 hours on
    synthetic + Mozhi crops and is the method reported in the paper.

Why not just always run both recognizers and pick the best? Because that doubles
inference cost on every line. The router does exactly that, but only for the
minority of lines where script ID is unsure -- which is the point of having a
confidence floor.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEVANAGARI_RANGE = (0x0900, 0x097F)
DEVANAGARI_EXT_RANGE = (0xA8E0, 0xA8FF)
LATIN_RANGES = ((0x0041, 0x005A), (0x0061, 0x007A))


def script_of_char(ch: str) -> str:
    cp = ord(ch)
    if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1] or DEVANAGARI_EXT_RANGE[0] <= cp <= DEVANAGARI_EXT_RANGE[1]:
        return "deva"
    if any(lo <= cp <= hi for lo, hi in LATIN_RANGES):
        return "latin"
    return "neutral"  # digits, punctuation, whitespace


def script_of_text(text: str, mixed_threshold: float = 0.15) -> tuple[str, float]:
    """Classify a string. Returns (script, confidence).

    Digits and punctuation are neutral and excluded from the vote -- otherwise a
    line like "2024 / अ.क्र. 15" reads as majority-neutral and tells you nothing.
    """
    counts = {"deva": 0, "latin": 0}
    for ch in text:
        s = script_of_char(ch)
        if s in counts:
            counts[s] += 1

    total = counts["deva"] + counts["latin"]
    if total == 0:
        return "unknown", 0.0

    deva_frac = counts["deva"] / total
    if deva_frac >= 1.0 - mixed_threshold:
        return "deva", deva_frac
    if deva_frac <= mixed_threshold:
        return "latin", 1.0 - deva_frac
    return "mixed", 1.0 - abs(deva_frac - 0.5) * 2


class ScriptIdentifier:
    """Config-driven script identification for image crops."""

    def __init__(self, config: dict[str, Any] | None = None, device: str = "auto") -> None:
        self.config = config or {}
        self.method = self.config.get("method", "cnn")
        self.confidence_floor = float(self.config.get("confidence_floor", 0.6))
        self.device = device
        self._model = None
        self._torch = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch

        from .model import ScriptCNN

        weights = Path(self.config.get("weights", "./weights/script_id/scriptid_cnn.pt"))
        if not weights.exists():
            raise FileNotFoundError(
                f"script-ID weights not found at {weights}. Train with "
                f"`python train/train_scriptid.py`, or set script_id.method to 'unicode_vote'."
            )
        self._torch = torch
        self._model = ScriptCNN(num_classes=2)
        state = torch.load(weights, map_location=self.device)
        self._model.load_state_dict(state["model"] if "model" in state else state)
        self._model.to(self.device).eval()

    def identify(self, image: np.ndarray, fallback_text: str | None = None) -> tuple[str, float]:
        if self.method == "always_deva":
            return "deva", 1.0
        if self.method == "always_latin":
            return "latin", 1.0
        if self.method == "unicode_vote":
            if not fallback_text:
                return "unknown", 0.0
            return script_of_text(fallback_text)

        if self.method != "cnn":
            raise ValueError(f"unknown script_id method: {self.method!r}")

        try:
            self._ensure_model()
        except FileNotFoundError:
            log.warning("script-ID model missing; falling back to unicode_vote")
            self.method = "unicode_vote"
            return self.identify(image, fallback_text)

        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.size == 0:
            return "unknown", 0.0
        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)
        tensor = self._torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            probs = self._model(tensor).softmax(dim=1)[0]

        idx = int(probs.argmax())
        return ("deva", "latin")[idx], float(probs[idx])


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """NFC is the correct storage form for Devanagari.

    Without this, visually identical strings compare unequal and your CER is
    inflated by pure encoding noise -- a decomposed 'क़' vs a precomposed one
    scores as an error despite rendering identically.
    """
    return unicodedata.normalize(form, text)
