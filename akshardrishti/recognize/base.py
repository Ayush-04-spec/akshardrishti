"""The recognizer interface -- the most important abstraction in this project.

Every OCR backend implements ``Recognizer``. That is what turns "compare our
system against Tesseract, PaddleOCR-VL, Surya and Bhashini" from a rewrite into
four rows in a results table and one line in a config file.

Add a backend by:
  1. subclassing ``Recognizer``
  2. implementing ``_load`` and ``recognize``
  3. registering it with ``@register("your_name")``
  4. adding a ``backends.your_name`` block to configs/pipeline.yaml
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    text: str
    confidence: float | None = None
    backend: str = "unknown"
    elapsed_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class Recognizer(ABC):
    """Base class for all text recognition backends.

    Models load lazily on first use -- important on Kaggle, where importing
    four backends eagerly will exhaust GPU memory before any of them runs.
    """

    name: ClassVar[str] = "base"
    supported_scripts: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: dict[str, Any] | None = None, device: str = "auto") -> None:
        self.config = config or {}
        self.device = _resolve_device(device)
        self._loaded = False

    # -- lifecycle ------------------------------------------------------
    @abstractmethod
    def _load(self) -> None:
        """Load weights. Called once, lazily."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            t0 = time.perf_counter()
            self._load()
            self._loaded = True
            log.info("loaded recognizer %r on %s in %.1fs", self.name, self.device, time.perf_counter() - t0)

    def unload(self) -> None:
        """Free GPU memory. Call between backends when benchmarking."""
        self._loaded = False

    # -- inference ------------------------------------------------------
    @abstractmethod
    def recognize(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        """Recognise text in a single line or region crop."""

    def recognize_batch(self, images: list[np.ndarray], script: str = "unknown") -> list[RecognitionResult]:
        """Override where the backend supports real batching -- this default is
        a loop and will be slow for GPU models."""
        return [self.recognize(img, script) for img in images]

    def __call__(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        self.ensure_loaded()
        t0 = time.perf_counter()
        result = self.recognize(image, script)
        result.elapsed_s = time.perf_counter() - t0
        result.backend = self.name
        return result

    def supports(self, script: str) -> bool:
        return not self.supported_scripts or script in self.supported_scripts or script == "unknown"


# ------------------------------------------------------------------ registry

_REGISTRY: dict[str, type[Recognizer]] = {}


def register(name: str) -> Callable[[type[Recognizer]], type[Recognizer]]:
    def decorator(cls: type[Recognizer]) -> type[Recognizer]:
        if name in _REGISTRY:
            raise ValueError(f"recognizer {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def build_recognizer(name: str, config: dict[str, Any] | None = None, device: str = "auto") -> Recognizer:
    """Instantiate a registered backend by name.

    Imports the backend modules on demand so that a missing optional dependency
    (paddle, surya) only breaks the backend that needs it.
    """
    if name not in _REGISTRY:
        _import_backends()
    if name not in _REGISTRY:
        raise KeyError(f"unknown recognizer {name!r}. Available: {available_backends()}")
    return _REGISTRY[name](config=config, device=device)


def _import_backends() -> None:
    from importlib import import_module

    for mod in ("crnn", "trocr_en", "tesseract", "bhashini_api", "paddle_vl", "surya_backend"):
        try:
            import_module(f"akshardrishti.recognize.{mod}")
        except ImportError as exc:  # optional dependency missing -- fine
            log.debug("recognizer module %s unavailable: %s", mod, exc)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ------------------------------------------------------------------- routing


class ScriptRouter:
    """Sends each crop to the backend that handles its script.

    When script ID is unsure (confidence below the floor), runs *both* branches
    and keeps the higher-confidence transcription. That costs latency on a small
    fraction of regions and buys accuracy on exactly the mixed Hindi/English
    lines that Indian government documents are full of.
    """

    def __init__(
        self,
        deva: Recognizer,
        latin: Recognizer,
        confidence_floor: float = 0.6,
    ) -> None:
        self.deva = deva
        self.latin = latin
        self.confidence_floor = confidence_floor

    def recognize(
        self,
        image: np.ndarray,
        script: str = "unknown",
        script_confidence: float | None = None,
    ) -> RecognitionResult:
        unsure = script_confidence is not None and script_confidence < self.confidence_floor
        if script == "deva" and not unsure:
            return self.deva(image, "deva")
        if script == "latin" and not unsure:
            return self.latin(image, "latin")

        # Unknown, mixed, or low-confidence -> try both, keep the better one.
        results = [self.deva(image, "deva"), self.latin(image, "latin")]
        results = [r for r in results if r.text.strip()]
        if not results:
            return RecognitionResult(text="", confidence=0.0, backend="router")
        return max(results, key=lambda r: (r.confidence if r.confidence is not None else 0.0, len(r.text)))

    def unload(self) -> None:
        self.deva.unload()
        self.latin.unload()
