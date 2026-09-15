"""Tesseract backend -- the traditional baseline the report compares against.

Install:
    apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar
    pip install pytesseract
"""

from __future__ import annotations

import logging

import numpy as np

from .base import RecognitionResult, Recognizer, register

log = logging.getLogger(__name__)

_LANG_FOR_SCRIPT = {"deva": "hin+mar", "latin": "eng", "mixed": "hin+mar+eng", "unknown": "hin+mar+eng"}


@register("tesseract")
class TesseractRecognizer(Recognizer):
    supported_scripts = ("deva", "latin", "mixed", "unknown")

    def _load(self) -> None:
        import pytesseract

        self._pt = pytesseract
        self.psm = int(self.config.get("psm", 7))  # 7 = single text line
        self.oem = int(self.config.get("oem", 3))
        try:
            langs = set(pytesseract.get_languages(config=""))
        except Exception:  # pragma: no cover - tesseract missing
            langs = set()
        for need in ("hin", "mar", "eng"):
            if langs and need not in langs:
                log.warning("tesseract language pack %r not installed -- baseline will be wrong", need)

    def recognize(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        if image is None or image.size == 0:
            return RecognitionResult(text="", confidence=0.0)

        lang = _LANG_FOR_SCRIPT.get(script, "hin+mar+eng")
        cfg = f"--oem {self.oem} --psm {self.psm}"

        data = self._pt.image_to_data(image, lang=lang, config=cfg, output_type=self._pt.Output.DICT)
        words, confs = [], []
        for text, conf in zip(data["text"], data["conf"]):
            if text and text.strip():
                words.append(text.strip())
                try:
                    c = float(conf)
                    if c >= 0:
                        confs.append(c / 100.0)
                except (TypeError, ValueError):
                    pass

        return RecognitionResult(
            text=" ".join(words),
            confidence=float(np.mean(confs)) if confs else None,
        )

    def recognize_page(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        """Full-page mode -- this is the actual `Tesseract fullpage` baseline row.

        PSM 3 lets Tesseract do its own layout analysis, which is precisely the
        thing our pipeline claims to beat. Comparing against PSM 7 on our own
        crops would be rigging the benchmark in our favour.
        """
        self.ensure_loaded()
        lang = _LANG_FOR_SCRIPT.get(script, "hin+mar+eng")
        text = self._pt.image_to_string(image, lang=lang, config=f"--oem {self.oem} --psm 3")
        return RecognitionResult(text=text.strip(), confidence=None, backend=f"{self.name}_fullpage")
