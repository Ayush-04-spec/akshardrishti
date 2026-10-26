"""CRNN recognizer backend -- our trained Devanagari model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocess.enhance import preprocess_line_crop
from .base import RecognitionResult, Recognizer, register

log = logging.getLogger(__name__)


def ctc_output_length(input_width: int) -> int:
    """Return valid CRNN timesteps for an unpadded input width.

    The model applies two width-wise 2x pools followed by a valid width-2
    convolution: floor(floor(width / 2) / 2) - 1.
    """
    return max(1, (int(input_width) // 2 // 2) - 1)


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
        self.inference_mode = str(self.config.get("inference_mode", "legacy_cap"))
        if self.inference_mode not in ("legacy_cap", "aspect_preserving"):
            raise ValueError(f"unknown CRNN inference_mode: {self.inference_mode!r}")
        self.max_preserved_width = int(self.config.get("max_preserved_width", 4096))
        if self.max_preserved_width < 8:
            raise ValueError("max_preserved_width must be at least 8")
        self.beam_width = int(self.config.get("beam_width", 10))
        # right_pad chosen on validation; see results/crnn_padding_sweep.json
        self.right_pad = int(self.config.get("right_pad", 64))

        self.model = CRNN(num_classes=len(self.charset), img_height=self.img_height)
        try:
            state = torch.load(weights, map_location=self.device, weights_only=True)
        except (Exception) as _e:
            if type(_e).__name__ in ("UnpicklingError", "ModuleNotFoundError") or \
                    "pickle" in str(_e).lower() or "pathlib" in str(_e).lower():
                raise RuntimeError(
                    "CRNN checkpoint is not in slim format; run train/export_crnn_slim.py "
                    "and point the config at the slim file."
                ) from _e
            raise
        self.model.load_state_dict(
            state.get("model") or state.get("model_state_dict") or state
        )
        self.model.to(self.device).eval()
        self._torch = torch

        log.info("CRNN loaded: %d classes, beam_width=%d", len(self.charset), self.beam_width)

    def _prepare_with_content_width(self, image: np.ndarray) -> tuple[Any, int]:
        crop = preprocess_line_crop(image, target_height=self.img_height, binarize_crop=True)
        content_width = int(crop.shape[1])
        if self.inference_mode == "legacy_cap" and content_width > self.max_width:
            import cv2

            crop = cv2.resize(crop, (self.max_width, self.img_height), interpolation=cv2.INTER_AREA)
            content_width = int(crop.shape[1])
        elif self.inference_mode == "aspect_preserving" and content_width > self.max_preserved_width:
            log.error(
                "CRNN crop width %d exceeds safe aspect-preserving limit %d; refusing to distort or truncate it",
                content_width, self.max_preserved_width,
            )
            raise ValueError(
                f"CRNN crop width {content_width} exceeds max_preserved_width "
                f"{self.max_preserved_width}; split the source region before recognition"
            )
        tensor = self._torch.from_numpy(crop.astype(np.float32) / 255.0)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        if self.right_pad > 0:
            # Append right_pad white columns (1.0 = white in normalised [-1,1] space).
            # right_pad chosen on validation; see results/crnn_padding_sweep.json
            tensor = self._torch.nn.functional.pad(tensor, (0, self.right_pad), value=1.0)
        return tensor, content_width

    def _prepare(self, image: np.ndarray) -> Any:
        """Prepare input with configured mode while retaining the historical API."""
        tensor, _ = self._prepare_with_content_width(image)
        return tensor

    def recognize(self, image: np.ndarray, script: str = "deva") -> RecognitionResult:
        from .crnn_model import beam_decode, greedy_decode

        if image is None or image.size == 0:
            return RecognitionResult(text="", confidence=0.0)

        try:
            tensor, content_width = self._prepare_with_content_width(image)
            tensor = tensor.to(self.device)
        except ValueError:
            if self.inference_mode == "aspect_preserving":
                raise
            return RecognitionResult(text="", confidence=0.0)

        with self._torch.no_grad():
            log_probs = self.model(tensor)

        lengths = [ctc_output_length(content_width)] if self.inference_mode == "aspect_preserving" else None
        decoded = (
            beam_decode(log_probs, self.charset, self.beam_width, lengths)
            if self.beam_width > 1
            else greedy_decode(log_probs, self.charset, lengths)
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
        content_widths = []
        for _, im in valid:
            try:
                tensor, content_width = self._prepare_with_content_width(im)
                tensors.append(tensor.squeeze(0))
                content_widths.append(content_width)
            except ValueError:
                if self.inference_mode == "aspect_preserving":
                    raise
                tensors.append(self._torch.zeros(1, self.img_height, 8))
                content_widths.append(8)

        max_w = max(t.shape[-1] for t in tensors)
        # Each crop has trailing configured right_pad; exclude only the
        # additional batch padding from CTC decoding.
        if self.inference_mode == "aspect_preserving":
            sequence_lengths = [ctc_output_length(w) for w in content_widths]
        else:
            sequence_lengths = [max(1, (t.shape[-1] // 4) - 1) for t in tensors]
        padded = self._torch.stack([
            self._torch.nn.functional.pad(t, (0, max_w - t.shape[-1]), value=1.0) for t in tensors
        ]).to(self.device)

        with self._torch.no_grad():
            log_probs = self.model(padded)

        decoded = (
            beam_decode(log_probs, self.charset, self.beam_width, sequence_lengths)
            if self.beam_width > 1
            else greedy_decode(log_probs, self.charset, sequence_lengths)
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
