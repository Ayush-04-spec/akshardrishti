"""TrOCR backend -- the English branch.

Scope note for the report: TrOCR's decoder is an English RoBERTa. It cannot read
Devanagari and we do not pretend otherwise. Using it for Latin text only, and a
CTC model for Devanagari, is the "script routing" contribution -- not a
limitation we are hiding.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import RecognitionResult, Recognizer, register

log = logging.getLogger(__name__)


@register("trocr_printed")
class TrOCRRecognizer(Recognizer):
    supported_scripts = ("latin",)

    def _load(self) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        model_id = self.config.get("model_id", "microsoft/trocr-base-printed")
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device).eval()
        self.max_new_tokens = int(self.config.get("max_new_tokens", 128))
        self._torch = torch
        log.info("TrOCR loaded: %s", model_id)

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1)
        if image.ndim == 3 and image.shape[2] == 3:
            return image[:, :, ::-1]  # BGR (OpenCV) -> RGB
        raise ValueError(f"unexpected image shape {image.shape}")

    def recognize(self, image: np.ndarray, script: str = "latin") -> RecognitionResult:
        if image is None or image.size == 0:
            return RecognitionResult(text="", confidence=0.0)

        pixel_values = self.processor(images=self._to_rgb(image), return_tensors="pt").pixel_values.to(self.device)

        with self._torch.no_grad():
            generated = self.model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]

        # Mean max-softmax across generated steps as a rough confidence. Honest
        # caveat: generative confidence is poorly calibrated, so treat it as a
        # ranking signal for the router, not as a probability.
        conf = None
        if getattr(generated, "scores", None):
            probs = [self._torch.softmax(s, dim=-1).max().item() for s in generated.scores]
            conf = float(np.mean(probs)) if probs else None

        return RecognitionResult(text=text.strip(), confidence=conf)

    def recognize_batch(self, images: list[np.ndarray], script: str = "latin") -> list[RecognitionResult]:
        valid = [(i, im) for i, im in enumerate(images) if im is not None and im.size > 0]
        out = [RecognitionResult(text="", confidence=0.0) for _ in images]
        if not valid:
            return out

        pixel_values = self.processor(
            images=[self._to_rgb(im) for _, im in valid], return_tensors="pt"
        ).pixel_values.to(self.device)

        with self._torch.no_grad():
            ids = self.model.generate(pixel_values, max_new_tokens=self.max_new_tokens)

        for (orig_idx, _), text in zip(valid, self.processor.batch_decode(ids, skip_special_tokens=True)):
            out[orig_idx] = RecognitionResult(text=text.strip(), confidence=None, backend=self.name)
        return out

    def unload(self) -> None:
        if self._loaded:
            del self.model
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass
        super().unload()
