"""CRNN recognizer backend -- our trained Devanagari model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocess.enhance import preprocess_line_crop
from .base import RecognitionResult, Recognizer, register

log = logging.getLogger(__name__)


@register("crnn_mozhi")
class CRNNRecognizer(Recognizer):
    """Devanagari (Hindi + Marathi) line recognizer trained on Mozhi."""

    supported_scripts = ("deva",)

    def _load(self) -> None:
        import torch

        from .crnn_model import CRNN, Charset

        weights = Path(self.config.get("weights", "./weights/recognize/crnn_mozhi_himr.pt"))
        charset_path = Path(self.config.get("charset", "./weights/recognize/charset_deva.txt"))

        if not weights.exists():
            raise FileNotFoundError(
                f"CRNN weights not found at {weights}. Train first:\n"
                f"    python train/train_crnn.py --config configs/pipeline.yaml"
            )
        if not charset_path.exists():
            raise FileNotFoundError(f"charset not found at {charset_path}")

        self.charset = Charset.load(charset_path)
        self.img_height = int(self.config.get("img_height", 32))
        self.max_width = int(self.config.get("max_width", 512))
        self.beam_width = int(self.config.get("beam_width", 10))

        self.model = CRNN(num_classes=len(self.charset), img_height=self.img_height)
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.to(self.device).eval()
        self._torch = torch

        log.info("CRNN loaded: %d classes, beam_width=%d", len(self.charset), self.beam_width)

    def _prepare(self, image: np.ndarray) -> Any:
        crop = preprocess_line_crop(image, target_height=self.img_height, binarize_crop=True)
        if crop.shape[1] > self.max_width:
            import cv2

            crop = cv2.resize(crop, (self.max_width, self.img_height), interpolation=cv2.INTER_AREA)
        tensor = self._torch.from_numpy(crop.astype(np.float32) / 255.0)
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    def recognize(self, image: np.ndarray, script: str = "deva") -> RecognitionResult:
        from .crnn_model import beam_decode, greedy_decode

        if image is None or image.size == 0:
            return RecognitionResult(text="", confidence=0.0)

        try:
            tensor = self._prepare(image).to(self.device)
        except ValueError:
            return RecognitionResult(text="", confidence=0.0)

        with self._torch.no_grad():
            log_probs = self.model(tensor)

        decoded = (
            beam_decode(log_probs, self.charset, self.beam_width)
            if self.beam_width > 1
            else greedy_decode(log_probs, self.charset)
        )
        text, conf = decoded[0]
        return RecognitionResult(text=text, confidence=conf)

    def recognize_batch(self, images: list[np.ndarray], script: str = "deva") -> list[RecognitionResult]:
        """Real batching -- pads to the widest crop in the batch."""
        from .crnn_model import beam_decode, greedy_decode

        valid = [(i, im) for i, im in enumerate(images) if im is not None and im.size > 0]
        out: list[RecognitionResult] = [RecognitionResult(text="", confidence=0.0) for _ in images]
        if not valid:
            return out

        tensors = []
        for _, im in valid:
            try:
                tensors.append(self._prepare(im).squeeze(0))
            except ValueError:
                tensors.append(self._torch.zeros(1, self.img_height, 8))

        max_w = max(t.shape[-1] for t in tensors)
        padded = self._torch.stack([
            self._torch.nn.functional.pad(t, (0, max_w - t.shape[-1]), value=1.0) for t in tensors
        ]).to(self.device)

        with self._torch.no_grad():
            log_probs = self.model(padded)

        decoded = (
            beam_decode(log_probs, self.charset, self.beam_width)
            if self.beam_width > 1
            else greedy_decode(log_probs, self.charset)
        )
        for (orig_idx, _), (text, conf) in zip(valid, decoded):
            out[orig_idx] = RecognitionResult(text=text, confidence=conf, backend=self.name)
        return out

    def unload(self) -> None:
        if self._loaded:
            del self.model
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass
        super().unload()
