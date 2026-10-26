"""Conservative word-gap splitting for long CRNN line crops."""

from __future__ import annotations

import cv2
import numpy as np


def split_long_line_at_word_gaps(
    image: np.ndarray,
    *,
    max_width: int = 512,
    target_height: int = 32,
    min_gap_px: int = 3,
    table_gap_ratio: float = 0.20,
    max_chunks: int = 5,
) -> tuple[list[np.ndarray], str]:
    """Split a long crop at broad whitespace runs, overlapping only white pixels.

    ``max_width`` is expressed in the recognizer's resized coordinate system.
    If no safe word-gap cut exists, the original crop is returned unchanged.
    Rows with multiple very broad gaps are treated as table-like and left intact.
    The status lets callers log conservative fallbacks.
    """
    if image is None or image.size == 0:
        return [], "empty"
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return [image], "fallback_invalid_shape"
    source_limit = max(1, round(max_width * h / target_height))
    if w <= source_limit:
        return [image], "within_limit"
    # A line requiring many cuts is often a merged table row or multiple
    # columns. Projection alone cannot establish safe reading boundaries.
    if max_chunks > 0 and w > source_limit * max_chunks:
        return [image], "fallback_ambiguous_very_wide"

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A column is considered blank when at most 1% of its pixels are dark.
    ink = gray < 220
    blank = ink.mean(axis=0) <= 0.01
    padded = np.r_[False, blank, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_gap_px]

    broad_gap = round(source_limit * table_gap_ratio)
    if sum(b - a >= broad_gap for a, b in runs) >= 2:
        return [image], "fallback_table_like"

    chunks: list[np.ndarray] = []
    cursor = 0
    while w - cursor > source_limit:
        # Look near the far end of the window, with enough room for a useful
        # first chunk. Both neighboring crops include the whitespace run, so
        # their overlap contains no glyphs and does not require text stitching.
        candidates = [
            (a, b) for a, b in runs
            if cursor + source_limit // 2 <= a < b <= cursor + source_limit
        ]
        if not candidates:
            return [image], "fallback_no_word_gap"
        a, b = min(candidates, key=lambda pair: abs((pair[0] + pair[1]) / 2 - (cursor + source_limit)))
        chunks.append(image[:, cursor:b])
        cursor = a
    if cursor >= w:
        return [image], "fallback_invalid_cut"
    chunks.append(image[:, cursor:w])
    return chunks, "word_gap"
