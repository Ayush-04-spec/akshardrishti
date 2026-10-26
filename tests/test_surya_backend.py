import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from akshardrishti.recognize.surya_backend import (
    SuryaRecognizer,
    _html_to_text,
)


def test_surya_html_output_preserves_lines_and_table_cell_separators():
    assert _html_to_text("<p>मराठी मजकूर</p><table><tr><td>अ</td><td>१२</td></tr></table>") == (
        "मराठी मजकूर\n\n अ  १२"
    )


def test_surya_v2_blocks_are_joined_in_reading_order_and_skip_nontext():
    recognizer = SuryaRecognizer({"languages": ["hi", "mr", "en"]}, device="cpu")
    recognizer._to_pil = lambda image: "pil-image"
    recognizer._recognition = lambda images: [SimpleNamespace(blocks=[
        SimpleNamespace(reading_order=2, html="<p>दुसरा</p>", confidence=0.8, skipped=False, error=False),
        SimpleNamespace(reading_order=1, html="<p>पहिला</p>", confidence=0.9, skipped=False, error=False),
        SimpleNamespace(reading_order=0, html="", confidence=None, skipped=True, error=False),
    ])]

    result = recognizer.recognize(np.zeros((3, 4), dtype=np.uint8))

    assert result.text == "पहिला\nदुसरा"
    assert result.confidence == pytest.approx(0.85)


def test_surya_reports_block_level_inference_failures():
    recognizer = SuryaRecognizer({}, device="cpu")
    recognizer._to_pil = lambda image: "pil-image"
    recognizer._recognition = lambda images: [SimpleNamespace(blocks=[
        SimpleNamespace(reading_order=0, html="", confidence=None, skipped=False, error=True),
    ])]
    with pytest.raises(RuntimeError, match="OCR errors in 1 page block"):
        recognizer.recognize(np.zeros((3, 4), dtype=np.uint8))


def test_surya_missing_package_has_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "surya", None)
    recognizer = SuryaRecognizer({}, device="cpu")
    with pytest.raises(ImportError, match="python -m pip install surya-ocr"):
        recognizer._load()


def _install_fake_surya_api(monkeypatch, manager_type):
    package = types.ModuleType("surya")
    package.__path__ = []
    inference = types.ModuleType("surya.inference")
    inference.SuryaInferenceManager = manager_type
    recognition = types.ModuleType("surya.recognition")
    recognition.RecognitionPredictor = lambda manager: object()
    monkeypatch.setitem(sys.modules, "surya", package)
    monkeypatch.setitem(sys.modules, "surya.inference", inference)
    monkeypatch.setitem(sys.modules, "surya.recognition", recognition)


def test_surya_refuses_to_spawn_or_download_without_explicit_permission(monkeypatch):
    calls = []

    class Manager:
        def __init__(self):
            calls.append("manager")

    _install_fake_surya_api(monkeypatch, Manager)
    monkeypatch.delenv("SURYA_INFERENCE_URL", raising=False)
    recognizer = SuryaRecognizer({"languages": ["hi", "mr", "en"]}, device="cpu")

    with pytest.raises(RuntimeError, match="set SURYA_INFERENCE_URL"):
        recognizer._load()
    assert calls == []


def test_surya_rejects_unconfigured_language_codes_before_loading(monkeypatch):
    calls = []

    class Manager:
        def __init__(self):
            calls.append("manager")

    _install_fake_surya_api(monkeypatch, Manager)
    monkeypatch.setenv("SURYA_INFERENCE_URL", "http://127.0.0.1:8000/v1")
    recognizer = SuryaRecognizer({"languages": ["zz"]}, device="cpu")

    with pytest.raises(ValueError, match="unsupported"):
        recognizer._load()
    assert calls == []


def test_surya_loads_against_an_existing_server(monkeypatch):
    calls = []

    class Manager:
        def __init__(self):
            calls.append("manager")

    _install_fake_surya_api(monkeypatch, Manager)
    monkeypatch.setenv("SURYA_INFERENCE_URL", "http://127.0.0.1:8000/v1")
    recognizer = SuryaRecognizer({"languages": ["hi", "mr", "en"]}, device="cpu")
    recognizer._load()

    assert calls == ["manager"]
    assert recognizer._pil_image is not None
