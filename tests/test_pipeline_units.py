"""Unit tests for the parts of AksharDrishti that do not need trained weights.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akshardrishti.config import ClassMap, Config
from akshardrishti.layout.lines import crop, offset_box, segment_lines_projection
from akshardrishti.layout.reading_order import compute_reading_order, reading_order_accuracy
from akshardrishti.postprocess.text_repair import (
    canonicalize_nukta, clean_text, drop_orphan_marks,
    normalize_for_scoring, repair_shirorekha_splits,
)
from akshardrishti.preprocess.enhance import (
    apply_clahe, estimate_skew_hough, preprocess_line_crop, preprocess_page, rotate_bound,
)
from akshardrishti.schema import BBox, Document, Page, PageMeta, Region, RegionType, TextLine
from akshardrishti.script_id.classifier import script_of_text


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def text_page() -> np.ndarray:
    img = np.full((600, 800), 255, np.uint8)
    for i in range(6):
        cv2.putText(img, f"Government Notice line {i}", (40, 80 + i * 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    return img


def make_region(rid: int, rtype: RegionType, x1, y1, x2, y2, conf=0.9) -> Region:
    return Region(region_id=rid, type=rtype, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                  detection_confidence=conf)


# -------------------------------------------------------------------- schema


class TestSchema:
    def test_bbox_geometry(self):
        b = BBox(x1=10, y1=20, x2=110, y2=70)
        assert b.width == 100 and b.height == 50
        assert b.area == 5000
        assert b.cx == 60 and b.cy == 45

    def test_bbox_rejects_degenerate(self):
        with pytest.raises(ValueError):
            BBox(x1=10, y1=10, x2=10, y2=20)
        with pytest.raises(ValueError):
            BBox(x1=10, y1=10, x2=5, y2=20)

    def test_bbox_iou(self):
        a = BBox(x1=0, y1=0, x2=10, y2=10)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(BBox(x1=20, y1=20, x2=30, y2=30)) == 0.0
        # half-overlap: intersection 50, union 150
        assert a.iou(BBox(x1=5, y1=0, x2=15, y2=10)) == pytest.approx(50 / 150)

    def test_bbox_clip(self):
        clipped = BBox(x1=-5, y1=-5, x2=200, y2=200).clip(100, 100)
        assert clipped.x1 == 0 and clipped.y1 == 0
        assert clipped.x2 == 100 and clipped.y2 == 100

    def test_region_type_flags(self):
        assert RegionType.TEXT.is_text_bearing
        assert not RegionType.FIGURE.is_text_bearing
        assert not RegionType.STAMP_SEAL.is_text_bearing
        assert RegionType.HEADER.is_furniture
        assert not RegionType.TEXT.is_furniture

    def test_compose_text_averages_confidence(self):
        r = make_region(0, RegionType.TEXT, 0, 0, 100, 50)
        r.lines = [
            TextLine(text="पहिली ओळ", bbox=BBox(x1=0, y1=0, x2=100, y2=20), confidence=0.9),
            TextLine(text="second line", bbox=BBox(x1=0, y1=25, x2=100, y2=45), confidence=0.7),
        ]
        assert r.compose_text() == "पहिली ओळ\nsecond line"
        assert r.recognition_confidence == pytest.approx(0.8)

    def test_document_roundtrip(self, tmp_path):
        page = Page(meta=PageMeta(source_path="x.jpg", width=800, height=600),
                    regions=[make_region(0, RegionType.TITLE, 10, 10, 200, 60)])
        page.regions[0].text = "शासन निर्णय"
        doc = Document(source_path="x.jpg", pages=[page])

        path = doc.save_json(tmp_path / "out.json")
        loaded = Document.load_json(path)
        assert loaded.pages[0].regions[0].text == "शासन निर्णय"
        assert loaded.pages[0].regions[0].type is RegionType.TITLE


# -------------------------------------------------------------------- config


class TestConfig:
    def test_dotted_access_and_override(self):
        cfg = Config.load("configs/pipeline.yaml")
        assert cfg.get("layout.imgsz") == 1024
        assert cfg.get("nonexistent.key", "fallback") == "fallback"
        cfg.set("layout.imgsz", 640)
        assert cfg.get("layout.imgsz") == 640

    def test_hash_changes_with_content(self):
        a = Config.load("configs/pipeline.yaml")
        b = Config.load("configs/pipeline.yaml")
        assert a.hash() == b.hash()
        b.set("layout.conf", 0.99)
        assert a.hash() != b.hash()

    def test_cli_overrides(self):
        cfg = Config.load("configs/pipeline.yaml", overrides=["layout.sahi.enabled=false", "layout.conf=0.4"])
        assert cfg.get("layout.sahi.enabled") is False
        assert cfg.get("layout.conf") == 0.4


class TestClassMap:
    def test_targets_are_eleven(self):
        cm = ClassMap.load()
        assert len(cm.targets) == 11
        assert set(cm.targets) == {t.value for t in RegionType}

    def test_resolution_is_case_and_separator_insensitive(self):
        cm = ClassMap.load()
        assert cm.resolve("Section Title")[0] == "title"
        assert cm.resolve("section-title")[0] == "title"
        assert cm.resolve("SECTION_TITLE")[0] == "title"

    def test_ignore_returns_none(self):
        cm = ClassMap.load()
        assert cm.resolve("background") is None

    def test_unmapped_fails_loudly(self):
        cm = ClassMap.load()
        with pytest.raises(KeyError):
            cm.resolve("marginalia_gloss")

    def test_unmapped_report(self):
        cm = ClassMap.load()
        missing = cm.unmapped_report(["title", "text", "totally_new_class"])
        assert missing == ["totally_new_class"]


# --------------------------------------------------------------- preprocess


class TestPreprocess:
    def test_skew_estimation_recovers_known_angle(self, text_page):
        for true_angle in (-6.0, -3.0, 3.0, 6.0):
            skewed = rotate_bound(text_page, true_angle)
            estimated = estimate_skew_hough(skewed)
            # deskew applies rotate_bound(img, estimated), so it must be opposite
            assert estimated == pytest.approx(-true_angle, abs=1.2), f"failed at {true_angle}"

    def test_skew_returns_zero_on_blank(self):
        assert estimate_skew_hough(np.full((400, 400), 255, np.uint8)) == 0.0

    def test_preprocess_full_chain(self, text_page):
        result = preprocess_page(text_page, {
            "enabled": True, "max_side": 2048,
            "deskew": {"enabled": True, "max_angle": 15},
            "denoise": {"enabled": True, "method": "fastnlmeans", "strength": 7},
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid": [8, 8]},
            "binarize": {"enabled": False},
        })
        assert result.image.ndim == 2
        assert "clahe" in result.applied

    def test_preprocess_respects_max_side(self, text_page):
        big = cv2.resize(text_page, (4000, 3000))
        result = preprocess_page(big, {"enabled": True, "max_side": 1024,
                                       "deskew": {"enabled": False}, "denoise": {"enabled": False},
                                       "clahe": {"enabled": False}, "binarize": {"enabled": False}})
        assert max(result.shape) <= 1024

    def test_preprocess_rejects_empty(self):
        with pytest.raises(ValueError):
            preprocess_page(np.array([]), {})

    def test_clahe_preserves_shape(self, text_page):
        assert apply_clahe(text_page).shape == text_page.shape

    def test_line_crop_normalises_height(self):
        crop_img = np.full((57, 300), 255, np.uint8)
        cv2.putText(crop_img, "text", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        out = preprocess_line_crop(crop_img, target_height=32)
        assert out.shape[0] == 32


# ------------------------------------------------------------ line segments


class TestLineSegmentation:
    def test_finds_all_lines(self, text_page):
        boxes = segment_lines_projection(text_page, min_line_height=12)
        assert len(boxes) == 6

    def test_lines_are_vertically_ordered_and_disjoint(self, text_page):
        boxes = segment_lines_projection(text_page, min_line_height=12)
        for a, b in zip(boxes, boxes[1:]):
            assert a.y1 < b.y1
            assert a.y2 <= b.y2

    def test_truly_blank_region_yields_no_lines(self):
        # Empty paper: return nothing, so the recognizer is never invoked on it.
        assert segment_lines_projection(np.full((100, 200), 255, np.uint8)) == []

    def test_ink_too_short_for_a_line_falls_back_to_whole_crop(self):
        # There IS ink, but no run survives the height filter. Hand the whole
        # crop to the recognizer rather than silently dropping possible text.
        img = np.full((100, 200), 255, np.uint8)
        img[48:52, 20:180] = 0          # a 4px-tall mark, below min_line_height
        boxes = segment_lines_projection(img, min_line_height=20)
        assert len(boxes) == 1
        assert boxes[0].height == 100   # the whole crop

    def test_empty_input(self):
        assert segment_lines_projection(np.empty((0, 0), np.uint8)) == []

    def test_crop_and_offset_roundtrip(self, text_page):
        region = BBox(x1=20, y1=40, x2=400, y2=200)
        region_img = crop(text_page, region)
        assert region_img.shape == (160, 380)

        local = BBox(x1=5, y1=10, x2=100, y2=40)
        page_box = offset_box(local, region.x1, region.y1)
        assert page_box.x1 == 25 and page_box.y1 == 50


# -------------------------------------------------------------- reading order


class TestReadingOrder:
    def test_two_column_order(self):
        regions = [
            make_region(0, RegionType.HEADER, 50, 20, 950, 60),
            make_region(1, RegionType.TITLE, 50, 90, 950, 140),
            make_region(2, RegionType.TEXT, 50, 200, 470, 600),
            make_region(3, RegionType.TEXT, 50, 630, 470, 1000),
            make_region(4, RegionType.TEXT, 530, 200, 950, 500),
            make_region(5, RegionType.TEXT, 530, 530, 950, 1000),
            make_region(6, RegionType.FOOTER, 50, 1330, 950, 1380),
        ]
        page = Page(meta=PageMeta(source_path="t", width=1000, height=1400), regions=regions)
        compute_reading_order(page)
        ids = [r.region_id for r in page.ordered_regions()]
        # left column fully before right column; furniture last
        assert ids[:5] == [1, 2, 3, 4, 5]
        assert set(ids[5:]) == {0, 6}

    def test_single_column_is_topdown(self):
        regions = [make_region(i, RegionType.TEXT, 50, 100 + i * 200, 950, 250 + i * 200) for i in range(4)]
        page = Page(meta=PageMeta(source_path="t", width=1000, height=1400), regions=regions)
        compute_reading_order(page)
        assert [r.region_id for r in page.ordered_regions()] == [0, 1, 2, 3]

    def test_furniture_always_last(self):
        regions = [
            make_region(0, RegionType.PAGE_NUMBER, 480, 1350, 520, 1380),
            make_region(1, RegionType.TEXT, 50, 200, 950, 900),
        ]
        page = Page(meta=PageMeta(source_path="t", width=1000, height=1400), regions=regions)
        compute_reading_order(page)
        assert page.regions[0].reading_order > page.regions[1].reading_order

    def test_empty_page(self):
        page = Page(meta=PageMeta(source_path="t", width=100, height=100))
        compute_reading_order(page)
        assert page.regions == []

    def test_accuracy_metric(self):
        assert reading_order_accuracy([0, 1, 2], [0, 1, 2]) == 1.0
        assert reading_order_accuracy([2, 1, 0], [0, 1, 2]) == 0.0
        assert reading_order_accuracy([0, 2, 1], [0, 1, 2]) == pytest.approx(2 / 3)
        assert reading_order_accuracy([0], [0]) == 1.0


# ------------------------------------------------------------- postprocess


class TestTextRepair:
    def test_shirorekha_split_repair(self):
        assert repair_shirorekha_splits("क् ष") == "क्ष"
        assert repair_shirorekha_splits("प् र") == "प्र"

    def test_matra_reattachment(self):
        assert repair_shirorekha_splits("क ा") == "का"

    def test_nukta_canonicalisation_makes_forms_equal(self):
        precomposed = "क़"                 # क़ as a single code point
        decomposed = "क़"            # क + nukta
        assert canonicalize_nukta(precomposed) == canonicalize_nukta(decomposed)

    def test_orphan_marks_dropped(self):
        assert "ा" not in drop_orphan_marks(" ा text")

    def test_clean_text_is_idempotent(self):
        raw = "शासन  निर्णय ।  test"
        once = clean_text(raw)
        assert clean_text(once) == once

    def test_scoring_normalisation_does_not_repair_errors(self):
        # normalize_for_scoring must NOT fix a genuine OCR error, or the
        # post-processor would be grading its own homework.
        assert normalize_for_scoring("क् ष") != normalize_for_scoring("क्ष")

    def test_scoring_normalisation_collapses_encoding_noise(self):
        assert normalize_for_scoring("क़") == normalize_for_scoring("क़")
        assert normalize_for_scoring("a  b") == normalize_for_scoring("a b")

    def test_empty_input(self):
        assert clean_text("") == ""
        assert normalize_for_scoring("") == ""


# ------------------------------------------------------------- script id


class TestScriptIdentification:
    def test_pure_devanagari(self):
        script, conf = script_of_text("शासन निर्णय क्रमांक")
        assert script == "deva" and conf == 1.0

    def test_pure_latin(self):
        script, conf = script_of_text("Government of Maharashtra")
        assert script == "latin" and conf == 1.0

    def test_mixed(self):
        script, _ = script_of_text("शासन निर्णय Government Order क्रमांक")
        assert script == "mixed"

    def test_digits_and_punctuation_are_neutral(self):
        # A pure-numeric string carries no script evidence.
        assert script_of_text("2024 / 15 - 03")[0] == "unknown"
        # Digits must not swamp a clearly Devanagari line.
        assert script_of_text("क्रमांक 2024/15/०३")[0] == "deva"

    def test_empty(self):
        assert script_of_text("")[0] == "unknown"


# ------------------------------------------------------------------ metrics


class TestMetrics:
    def test_cer_basic(self):
        from eval.metrics import cer

        assert cer("MAHARASHTRA", "MAHARASHTRA") == 0.0
        # one deletion out of 11 characters -- matches the report's worked example
        assert cer("MAHARASHTRA", "MAHARASTRA") == pytest.approx(1 / 11)

    def test_cer_empty_reference(self):
        from eval.metrics import cer

        assert cer("", "") == 0.0
        assert cer("", "spurious") == 1.0

    def test_corpus_cer_is_length_weighted(self):
        from eval.metrics import corpus_cer

        # One long perfect doc and one short broken doc: the corpus figure must
        # be dominated by the long one, unlike the mean-of-documents figure.
        refs = ["a" * 100, "abcd"]
        hyps = ["a" * 100, "xxxx"]
        stats = corpus_cer(refs, hyps)
        assert stats["cer"] == pytest.approx(4 / 104)
        assert stats["mean_document_cer"] == pytest.approx(0.5)

    def test_wer(self):
        from eval.metrics import wer

        assert wer("the quick brown fox", "the quick brown fox") == 0.0
        assert wer("the quick brown fox", "the quick red fox") == pytest.approx(0.25)

    def test_edit_ops_decomposition(self):
        from eval.metrics import levenshtein_ops

        ops = levenshtein_ops("MAHARASHTRA", "MAHARASTRA")
        assert ops["deletions"] == 1
        assert ops["substitutions"] == 0 and ops["insertions"] == 0

    def test_layout_map_perfect_prediction(self):
        from eval.metrics import layout_map

        gt = [[("text", BBox(x1=0, y1=0, x2=100, y2=100))]]
        pred = [[("text", BBox(x1=0, y1=0, x2=100, y2=100), 0.99)]]
        assert layout_map(pred, gt)["mAP50"] == pytest.approx(1.0)

    def test_layout_map_no_overlap(self):
        from eval.metrics import layout_map

        gt = [[("text", BBox(x1=0, y1=0, x2=100, y2=100))]]
        pred = [[("text", BBox(x1=500, y1=500, x2=600, y2=600), 0.99)]]
        assert layout_map(pred, gt)["mAP50"] == 0.0

    def test_bootstrap_ci_brackets_the_mean(self):
        from eval.metrics import bootstrap_ci

        values = [0.05, 0.06, 0.07, 0.05, 0.08, 0.06, 0.05, 0.09]
        lo, hi = bootstrap_ci(values, n_resamples=500)
        mean = sum(values) / len(values)
        assert lo <= mean <= hi


# ------------------------------------------------------------ integration


class TestRecognizerRegistry:
    def test_backends_register(self):
        from akshardrishti.recognize.base import _import_backends, available_backends

        _import_backends()
        # crnn and trocr have no optional deps beyond torch/transformers
        assert "crnn_mozhi" in available_backends()

    def test_unknown_backend_raises(self):
        from akshardrishti.recognize.base import build_recognizer

        with pytest.raises(KeyError):
            build_recognizer("no_such_backend")


class TestScriptRouter:
    def test_routes_by_script(self):
        from akshardrishti.recognize.base import RecognitionResult, Recognizer, ScriptRouter

        class Stub(Recognizer):
            def __init__(self, label):
                super().__init__({}, "cpu")
                self.label = label
                self._loaded = True

            def _load(self):
                pass

            def recognize(self, image, script="unknown"):
                return RecognitionResult(text=self.label, confidence=0.9)

        router = ScriptRouter(deva=Stub("DEVA"), latin=Stub("LATIN"))
        img = np.zeros((32, 100), np.uint8)
        assert router.recognize(img, "deva", 0.95).text == "DEVA"
        assert router.recognize(img, "latin", 0.95).text == "LATIN"

    def test_low_confidence_tries_both(self):
        from akshardrishti.recognize.base import RecognitionResult, Recognizer, ScriptRouter

        class Stub(Recognizer):
            def __init__(self, label, conf):
                super().__init__({}, "cpu")
                self.label, self.conf = label, conf
                self._loaded = True

            def _load(self):
                pass

            def recognize(self, image, script="unknown"):
                return RecognitionResult(text=self.label, confidence=self.conf)

        router = ScriptRouter(deva=Stub("DEVA", 0.3), latin=Stub("LATIN", 0.95), confidence_floor=0.6)
        # script says deva, but script-ID confidence is below the floor -> both run,
        # and the higher-confidence transcription wins
        assert router.recognize(np.zeros((32, 100), np.uint8), "deva", 0.4).text == "LATIN"
