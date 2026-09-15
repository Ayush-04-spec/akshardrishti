"""Surya backend -- multilingual comparison row.

    pip install surya-ocr

LICENSE WARNING: Surya's *weights* are under a modified AI Pubs Open RAIL-M
licence -- free for research, personal use, and organisations under $5M in
funding/revenue. Fine for this project; cite it and note the restriction if the
work is ever commercialised. The code itself is Apache-2.0.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import RecognitionResult, Recognizer, register

log = logging.getLogger(__name__)


@register("surya")
class SuryaRecognizer(Recognizer):
    supported_scripts = ("deva", "latin", "mixed", "unknown")

    def _load(self) -> None:
        try:
            from surya.foundation import FoundationPredictor
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Surya not installed. `pip install surya-ocr`, or drop 'surya' from eval.systems."
            ) from exc

        self._detection = DetectionPredictor()
        self._recognition = RecognitionPredictor(FoundationPredictor())
        self._pil = __import__("PIL.Image", fromlist=["Image"])
        log.info("Surya loaded")

    def _to_pil(self, image: np.ndarray):
        arr = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image[:, :, ::-1]
        return self._pil.Image.fromarray(arr.astype(np.uint8))

    def recognize(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        if image is None or image.size == 0:
            return RecognitionResult(text="", confidence=0.0)

        preds = self._recognition([self._to_pil(image)], det_predictor=self._detection)
        if not preds:
            return RecognitionResult(text="", confidence=0.0)

        lines = getattr(preds[0], "text_lines", []) or []
        texts = [ln.text for ln in lines if getattr(ln, "text", "").strip()]
        confs = [ln.confidence for ln in lines if getattr(ln, "confidence", None) is not None]

        return RecognitionResult(
            text="\n".join(texts).strip(),
            confidence=float(np.mean(confs)) if confs else None,
        )

    def recognize_page(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        self.ensure_loaded()
        r = self.recognize(image, script)
        r.backend = f"{self.name}_fullpage"
        return r
