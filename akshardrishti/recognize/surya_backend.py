"""Optional Surya 2 full-page OCR backend.

Install the optional package with ``python -m pip install surya-ocr``. Surya 2
uses an inference server (vLLM on NVIDIA GPUs or llama.cpp on CPU/Apple
Silicon); its manager can fetch model weights on first use. To avoid implicit
downloads, this backend requires an existing ``SURYA_INFERENCE_URL`` by
default. Set ``allow_model_download: true`` only when model download/server
startup has been explicitly approved.
"""

from __future__ import annotations

import logging
import os
from html.parser import HTMLParser
from typing import Any

import numpy as np

from .base import RecognitionResult, Recognizer, register

log = logging.getLogger(__name__)


class _BlockHTMLText(HTMLParser):
    """Convert Surya's block HTML to readable text without retaining markup."""

    _LINE_TAGS = {"br", "p", "div", "li", "tr", "h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._LINE_TAGS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._LINE_TAGS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _html_to_text(value: str) -> str:
    parser = _BlockHTMLText()
    parser.feed(value or "")
    return "".join(parser.parts).strip()


@register("surya")
class SuryaRecognizer(Recognizer):
    """Surya 2 multilingual full-page OCR, including Hindi and Marathi."""

    supported_scripts = ("deva", "latin", "mixed", "unknown")
    supported_languages = frozenset({"hi", "mr", "en"})

    def _load(self) -> None:
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Surya 2 is unavailable: install it with `python -m pip install surya-ocr`. "
                "This backend requires the SuryaInferenceManager API."
            ) from exc

        requested = self.config.get("languages", ["hi", "mr", "en"])
        unsupported = sorted(set(requested) - self.supported_languages)
        if unsupported:
            raise ValueError(f"Surya backend language codes are unsupported here: {unsupported}")

        env_name = str(self.config.get("server_url_env", "SURYA_INFERENCE_URL"))
        server_url = os.environ.get(env_name, "").strip()
        if not server_url and not self.config.get("allow_model_download", False):
            raise RuntimeError(
                f"No existing Surya server configured: set {env_name} to a running server URL, "
                "or explicitly set recognize.backends.surya.allow_model_download=true to permit "
                "Surya to start its inference backend and obtain weights."
            )

        self._manager = SuryaInferenceManager()
        self._recognition = RecognitionPredictor(self._manager)
        from PIL import Image

        self._pil_image = Image
        log.info("Surya 2 loaded for configured languages %s", sorted(requested))

    def _to_pil(self, image: np.ndarray):
        if image.ndim == 2:
            arr = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 3:
            arr = image[:, :, ::-1]  # OpenCV BGR to RGB
        else:
            raise ValueError(f"Surya expected grayscale or BGR image, got {image.shape}")
        return self._pil_image.fromarray(np.ascontiguousarray(arr.astype(np.uint8)))

    def recognize(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        if image is None or image.size == 0:
            return RecognitionResult(text="", confidence=0.0)

        predictions = self._recognition([self._to_pil(image)])
        if not predictions:
            return RecognitionResult(text="", confidence=0.0)

        page = predictions[0]
        blocks = list(_field(page, "blocks", []) or [])
        errors = [b for b in blocks if _field(b, "error", False) and not _field(b, "skipped", False)]
        if errors:
            raise RuntimeError(f"Surya reported OCR errors in {len(errors)} page block(s)")
        blocks = [b for b in blocks if not _field(b, "skipped", False)]
        blocks.sort(key=lambda b: _field(b, "reading_order", 0) or 0)

        texts: list[str] = []
        confidences: list[float] = []
        for block in blocks:
            content = _field(block, "html", "") or _field(block, "text", "")
            text = _html_to_text(str(content)) if "<" in str(content) else str(content).strip()
            if text:
                texts.append(text)
            conf = _field(block, "confidence")
            if conf is not None:
                confidences.append(float(conf))

        return RecognitionResult(
            text="\n".join(texts).strip(),
            confidence=float(np.mean(confidences)) if confidences else None,
        )

    def recognize_page(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        """Full-page method used by the existing benchmark runner."""
        self.ensure_loaded()
        result = self.recognize(image, script)
        result.backend = f"{self.name}_fullpage"
        return result
