import logging

import cv2
import numpy as np
import pytest
import torch

from akshardrishti.layout.lines import segment_lines, segment_lines_projection
from akshardrishti.preprocess.enhance import (
    detect_orientation_tesseract_osd,
    preprocess_page,
    rotate_page_quadrants,
    transform_bbox_quadrants,
)
from akshardrishti.recognize.crnn import CRNNRecognizer, ctc_output_length
from akshardrishti.recognize.crnn_model import CRNN, Charset


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_quadrant_rotation_pixels_and_coordinates(rotation):
    image = np.arange(15, dtype=np.uint8).reshape(3, 5)
    rotated, matrix = rotate_page_quadrants(image, rotation)
    expected = {
        0: image,
        90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(image, cv2.ROTATE_180),
        270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }[rotation]
    np.testing.assert_array_equal(rotated, expected)

    h, w = image.shape
    probe = {0: (1, 1), 90: (h - 1 - 1, 1), 180: (w - 1 - 1, h - 1 - 1),
             270: (1, w - 1 - 1)}[rotation]
    x, y = probe
    assert rotated[y, x] == image[1, 1]

    expected_bbox = {
        0: (1, 0, 3, 2),
        90: (0, 1, 2, 3),
        180: (1, 0, 3, 2),
        270: (0, 1, 2, 3),
    }[rotation]
    assert transform_bbox_quadrants((1, 0, 3, 2), matrix) == pytest.approx(expected_bbox)


def test_orientation_osd_uses_reported_rotation_and_confidence(monkeypatch):
    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_osd", lambda *a, **k: {"rotate": 90, "orientation_conf": 12.5})
    assert detect_orientation_tesseract_osd(np.full((10, 20), 255, np.uint8), 8.0) == (90, 12.5)
    monkeypatch.setattr(pytesseract, "image_to_osd", lambda *a, **k: {"rotate": 270, "orientation_conf": 2.0})
    assert detect_orientation_tesseract_osd(np.full((10, 20), 255, np.uint8), 8.0) == (0, 2.0)


def test_preprocess_optional_orientation_preserves_source_and_records_rotation(monkeypatch):
    import akshardrishti.preprocess.enhance as enhance

    source = np.arange(30 * 50, dtype=np.uint8).reshape(30, 50)
    original = source.copy()
    monkeypatch.setattr(enhance, "detect_orientation_tesseract_osd", lambda *a, **k: (90, 11.0))
    config = {
        "enabled": True, "max_side": 0, "orientation": {"enabled": True},
        "deskew": {"enabled": False}, "denoise": {"enabled": False},
        "clahe": {"enabled": False}, "binarize": {"enabled": False},
    }
    result = preprocess_page(source, config)
    expected, _ = rotate_page_quadrants(source, 90)
    np.testing.assert_array_equal(result.image, expected)
    np.testing.assert_array_equal(source, original)
    assert result.orientation_rotation_cw == 90
    assert "orientation_correction(cw=90deg)" in result.applied


def test_preprocess_orientation_disabled_does_not_call_provider(monkeypatch):
    import akshardrishti.preprocess.enhance as enhance

    monkeypatch.setattr(
        enhance, "detect_orientation_tesseract_osd",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected OSD call")),
    )
    image = np.full((30, 50), 255, np.uint8)
    result = preprocess_page(image, {
        "enabled": True, "max_side": 0, "orientation": {"enabled": False},
        "deskew": {"enabled": False}, "denoise": {"enabled": False},
        "clahe": {"enabled": False}, "binarize": {"enabled": False},
    })
    np.testing.assert_array_equal(result.image, image)
    assert result.orientation_rotation_cw == 0


def test_preprocess_orientation_provider_error_fails_closed(monkeypatch, caplog):
    import akshardrishti.preprocess.enhance as enhance

    monkeypatch.setattr(
        enhance, "detect_orientation_tesseract_osd",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OSD unavailable")),
    )
    image = np.full((30, 50), 255, np.uint8)
    result = preprocess_page(image, {
        "enabled": True, "max_side": 0, "orientation": {"enabled": True},
        "deskew": {"enabled": False}, "denoise": {"enabled": False},
        "clahe": {"enabled": False}, "binarize": {"enabled": False},
    })
    np.testing.assert_array_equal(result.image, image)
    assert result.orientation_rotation_cw == 0
    assert "leaving page unchanged" in caplog.text


def _two_column_page():
    image = np.full((100, 600), 255, np.uint8)
    for y in (10, 35, 60):
        cv2.rectangle(image, (25, y), (210, y + 8), 0, -1)
        cv2.rectangle(image, (390, y), (575, y + 8), 0, -1)
    return image


def test_columns_then_projection_splits_gutter_and_orders_left_to_right():
    image = _two_column_page()
    boxes = segment_lines(
        image, method="columns_then_projection", min_line_height=5, smoothing=1,
        min_column_gap_ratio=0.035, min_column_width_ratio=0.20,
    )
    assert len(boxes) == 6
    assert all(boxes[i].y1 < boxes[i + 1].y1 for i in (0, 1))
    assert all(boxes[i].y1 < boxes[i + 1].y1 for i in (3, 4))
    assert all(boxes[i].x1 < boxes[i + 3].x1 for i in (0, 1, 2))
    assert all(box.x2 < 300 for box in boxes[:3])
    assert all(box.x1 > 300 for box in boxes[3:])


def test_column_strategy_falls_back_when_no_reliable_gutter():
    image = np.full((60, 180), 255, np.uint8)
    for y in (10, 30, 50):
        cv2.rectangle(image, (5, y - 3), (175, y + 3), 0, -1)
    expected = segment_lines_projection(image, min_line_height=4, smoothing=1)
    actual = segment_lines(image, "columns_then_projection", min_line_height=4, smoothing=1)
    assert actual == expected


class _EmitAtLastTimestep(torch.nn.Module):
    def forward(self, tensor):
        t = ctc_output_length(tensor.shape[-1] + 4)
        out = torch.full((t, tensor.shape[0], 2), -10.0, device=tensor.device)
        out[:, :, 0] = 0.0
        out[-1, :, 1] = 10.0
        return out.log_softmax(2)


def _fake_recognizer(mode, max_width=512, right_pad=64, max_preserved_width=4096):
    rec = CRNNRecognizer(
        {"inference_mode": mode, "max_width": max_width, "right_pad": right_pad,
         "max_preserved_width": max_preserved_width, "beam_width": 1},
        device="cpu",
    )
    rec._torch = torch
    rec._loaded = True
    rec.model = _EmitAtLastTimestep()
    rec.charset = Charset(["क"])
    rec.img_height = 32
    rec.max_width = max_width
    rec.inference_mode = mode
    rec.max_preserved_width = max_preserved_width
    rec.right_pad = right_pad
    rec.beam_width = 1
    return rec


def test_aspect_preserving_mode_keeps_width_and_right_pads():
    image = np.full((32, 800), 255, np.uint8)
    legacy = _fake_recognizer("legacy_cap")
    preserved = _fake_recognizer("aspect_preserving")
    old_tensor = legacy._prepare(image)
    new_tensor, content_width = preserved._prepare_with_content_width(image)
    assert old_tensor.shape[-1] == 512 + 64
    assert new_tensor.shape[-1] == 800 + 64
    assert content_width == 800
    assert torch.all(new_tensor[..., -64:] == 1.0)


@pytest.mark.parametrize("width", [128, 512, 800, 2048])
def test_ctc_length_uses_unpadded_width_and_convolution_geometry(width):
    assert ctc_output_length(width) == max(1, (width // 2 // 2) - 1)


@pytest.mark.parametrize("width", [128, 512, 800])
def test_ctc_length_matches_actual_crnn_output_geometry(width):
    model = CRNN(num_classes=2, hidden=4).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 32, width))
    assert output.shape[0] == ctc_output_length(width)


def test_aspect_preserving_single_decode_excludes_right_padding_characters():
    rec = _fake_recognizer("aspect_preserving")
    result = rec.recognize(np.full((32, 160), 255, np.uint8), script="deva")
    assert result.text == ""


def test_aspect_preserving_batch_decode_excludes_batch_and_right_padding():
    rec = _fake_recognizer("aspect_preserving", right_pad=8)
    images = [np.full((32, 64), 255, np.uint8), np.full((32, 160), 255, np.uint8)]
    results = rec.recognize_batch(images, script="deva")
    assert [r.text for r in results] == ["", ""]


def test_empty_crops_return_empty_and_extreme_width_fails_closed(caplog):
    rec = _fake_recognizer("aspect_preserving", max_preserved_width=512)
    assert rec.recognize(np.empty((0, 0), np.uint8)).text == ""
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="exceeds max_preserved_width"):
            rec.recognize(np.full((32, 900), 255, np.uint8))
    assert "refusing to distort or truncate" in caplog.text


def test_legacy_mode_keeps_capped_tensor_shape_and_legacy_decode():
    rec = _fake_recognizer("legacy_cap")
    tensor = rec._prepare(np.full((32, 900), 255, np.uint8))
    assert tensor.shape == (1, 1, 32, 576)
    result = rec.recognize(np.full((32, 160), 255, np.uint8))
    # Legacy path decodes the configured right-pad timestep as before.
    assert result.text == "क"


@pytest.mark.parametrize("decoder", ["greedy", "beam"])
def test_ctc_repeat_blank_and_valid_length_semantics(decoder):
    from akshardrishti.recognize.crnn_model import beam_decode, greedy_decode

    charset = Charset(["क", "ख"])
    # Best path: क, क (same run collapses), blank, क (new repeated symbol),
    # then padded ख. With length 4 the padded timestep must be ignored.
    path = [1, 1, 0, 1, 2]
    logits = torch.full((len(path), 1, len(charset)), -20.0)
    for t, idx in enumerate(path):
        logits[t, 0, idx] = 0.0
    log_probs = logits.log_softmax(2)
    decode = greedy_decode if decoder == "greedy" else lambda lp, cs, lengths: beam_decode(lp, cs, 3, lengths)
    assert decode(log_probs, charset, [4])[0][0] == "कक"
    assert decode(log_probs, charset, [0])[0][0] == ""


def test_aspect_preserving_batch_valid_lengths_follow_each_input_width():
    rec = _fake_recognizer("aspect_preserving", right_pad=0)
    images = [np.full((32, width), 255, np.uint8) for width in (128, 512, 800, 2048)]
    tensors = [rec._prepare_with_content_width(im) for im in images]
    assert [width for _, width in tensors] == [128, 512, 800, 2048]
    assert [ctc_output_length(width) for _, width in tensors] == [31, 127, 199, 511]
    results = rec.recognize_batch(images, script="deva")
    assert len(results) == len(images)
    assert all(isinstance(result.text, str) for result in results)
