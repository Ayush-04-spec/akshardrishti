"""Bhashini printed-OCR backend (IIIT-H `bhashini/iiith-bhasha-ocr`).

This is the Government of India's own OCR service for Indian languages, and
benchmarking against it is what makes the report's BHASHINI alignment
methodological rather than decorative.

Credentials come from the environment, never from the config file:
    export BHASHINI_API_KEY=...
    export BHASHINI_ENDPOINT=...
    export BHASHINI_USER_ID=...          # if your account issues one

Register at https://bhashini.gitbook.io/bhashini-apis -- approval takes days,
so do it early. If the key never arrives, drop this row from the results table
and say so in the paper; an unavailable baseline is a normal thing to report.
"""

from __future__ import annotations

import base64
import logging
import os
import time

import cv2
import numpy as np

from .base import RecognitionResult, Recognizer, register

log = logging.getLogger(__name__)

_LANG_FOR_SCRIPT = {"deva": "hi", "latin": "en", "mixed": "hi", "unknown": "hi"}


@register("bhashini_api")
class BhashiniRecognizer(Recognizer):
    """Hosted API backend. Rate-limited, network-dependent, not for training loops."""

    supported_scripts = ("deva", "latin", "mixed", "unknown")

    def _load(self) -> None:
        import requests

        self._requests = requests
        self.endpoint = os.environ.get(self.config.get("endpoint_env", "BHASHINI_ENDPOINT"), "")
        self.api_key = os.environ.get(self.config.get("api_key_env", "BHASHINI_API_KEY"), "")
        self.user_id = os.environ.get("BHASHINI_USER_ID", "")
        self.service_id = self.config.get("service_id", "bhashini/iiith-bhasha-ocr")
        self.timeout = float(self.config.get("timeout_s", 30))
        self.max_retries = int(self.config.get("max_retries", 3))

        if not self.endpoint or not self.api_key:
            raise RuntimeError(
                "Bhashini credentials missing. Set BHASHINI_ENDPOINT and BHASHINI_API_KEY, "
                "or remove 'bhashini_api' from eval.systems in configs/pipeline.yaml."
            )

    @staticmethod
    def _encode(image: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("failed to PNG-encode crop for Bhashini request")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def recognize(self, image: np.ndarray, script: str = "unknown") -> RecognitionResult:
        if image is None or image.size == 0:
            return RecognitionResult(text="", confidence=0.0)

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "ocr",
                    "config": {
                        "language": {"sourceLanguage": _LANG_FOR_SCRIPT.get(script, "hi")},
                        "serviceId": self.service_id,
                    },
                }
            ],
            "inputData": {"image": [{"imageContent": self._encode(image)}]},
        }
        headers = {"Content-Type": "application/json", "Authorization": self.api_key}
        if self.user_id:
            headers["userID"] = self.user_id

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 429:  # rate limited -- back off and retry
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return RecognitionResult(text=self._extract_text(resp.json()), confidence=None)
            except Exception as exc:  # noqa: BLE001 - network layer, retry everything
                last_err = exc
                time.sleep(2 ** attempt)

        log.warning("Bhashini request failed after %d attempts: %s", self.max_retries, last_err)
        return RecognitionResult(text="", confidence=None, meta={"error": str(last_err)})

    @staticmethod
    def _extract_text(body: dict) -> str:
        """Walk the response defensively -- the ULCA envelope shape has changed
        between releases, so pull any 'source'/'text' field we can find."""
        for key in ("pipelineResponse", "output"):
            node = body.get(key)
            if isinstance(node, list):
                for item in node:
                    out = item.get("output") if isinstance(item, dict) else None
                    if isinstance(out, list):
                        for o in out:
                            for field in ("source", "text", "target"):
                                if isinstance(o, dict) and o.get(field):
                                    return str(o[field]).strip()
                    for field in ("source", "text"):
                        if isinstance(item, dict) and item.get(field):
                            return str(item[field]).strip()
        log.debug("could not find text in Bhashini response: %s", str(body)[:400])
        return ""
