import cv2
import numpy as np

from akshardrishti.layout.long_line import split_long_line_at_word_gaps


def test_long_line_splits_at_word_gaps_and_only_overlaps_blank_columns():
    image = np.full((32, 150), 255, np.uint8)
    for left in (5, 25, 45, 65, 85, 105, 125):
        cv2.rectangle(image, (left, 8), (left + 9, 24), 0, -1)

    chunks, status = split_long_line_at_word_gaps(
        image, max_width=50, min_gap_px=3, table_gap_ratio=0.4,
    )

    assert status == "word_gap"
    assert len(chunks) > 1
    assert all(chunk.shape[1] <= 50 for chunk in chunks)
    # Overlap windows share only blank columns: no ink is duplicated or lost.
    assert sum(np.count_nonzero(chunk < 220) for chunk in chunks) == np.count_nonzero(image < 220)
    assert np.count_nonzero(image[:, 15:25] < 220) == 0


def test_long_line_with_no_reliable_gap_uses_original_crop():
    image = np.zeros((32, 120), np.uint8)
    chunks, status = split_long_line_at_word_gaps(image, max_width=50)
    assert status == "fallback_no_word_gap"
    assert chunks == [image]


def test_table_like_row_uses_unsplit_fallback():
    image = np.full((32, 180), 255, np.uint8)
    for left in (5, 20, 35, 100, 115, 130):
        cv2.rectangle(image, (left, 8), (left + 8, 24), 0, -1)
    chunks, status = split_long_line_at_word_gaps(
        image, max_width=50, table_gap_ratio=0.2,
    )
    assert status == "fallback_table_like"
    assert chunks[0] is image


def test_extremely_wide_ambiguous_row_uses_safe_fallback():
    image = np.full((16, 1600), 255, np.uint8)
    for left in range(5, 1590, 40):
        cv2.rectangle(image, (left, 4), (left + 20, 12), 0, -1)
    chunks, status = split_long_line_at_word_gaps(image, max_width=512, max_chunks=5)
    assert status == "fallback_ambiguous_very_wide"
    assert chunks[0] is image
