"""PaddleOCR-VL backend -- modern VLM comparison row.

A 0.9B-parameter document VLM covering 109 languages including Devanagari.
Strong zero-shot, fits a T4, and makes the results table current rather than
comparing only against a 2010-era baseline.

    pip install paddlepaddle-gpu paddleocr
"""

from __future__ import annotations

import logging

import numpy as np

from .base import RecognitionResult, Recognizer, register

log = logging.getLogger(__name__)


@register("paddle_vl")
class PaddleVLRecognizer(Recognizer):
    supported_scripts = ("deva", "latin", "mixed", "unknown")

    def _load(self) -> None:
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PaddleOCR-VL not installed. `pip install paddlepaddle-gpu paddleocr`, "
                "or drop 'paddle_vl' from eval.systems."
            ) from exc

        self._pipeline = PaddleOCRVL()
        log.info("PaddleOCR-VL loaded")

    def recognize(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        if image is None or image.size == 0:
            return RecognitionResult(text="", confidence=0.0)

        results = self._pipeline.predict(image)
        texts: list[str] = []
        for res in results or []:
            data = res.get("res", res) if isinstance(res, dict) else {}
            md = data.get("markdown")
            if isinstance(md, dict):
                md = md.get("text", "")
            if md:
                texts.append(str(md))
            elif data.get("rec_texts"):
                texts.extend(str(t) for t in data["rec_texts"])

        return RecognitionResult(text="\n".join(texts).strip(), confidence=None)

    def recognize_page(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        """Whole-page parse -- PaddleOCR-VL does its own layout analysis, so this
        is the fair comparison against our full pipeline."""
        self.ensure_loaded()
        r = self.recognize(image, script)
        r.backend = f"{self.name}_fullpage"
        return r
