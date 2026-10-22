"""Text-line segmentation within a detected region.

Line-level recognition beats region-level for CTC models: a CRNN trained on
single lines cannot consume a paragraph image, and even VLM recognizers degrade
on tall multi-line crops.

Devanagari note
---------------
The shirorekha (top horizontal bar) makes Devanagari lines unusually *dense* in
the horizontal projection profile -- the bar produces a hard spike at the top of
each line rather than the gentle hump Latin text gives. That is helpful: peaks
are crisp. But conjuncts hanging below the baseline can bridge into the next
line's ascenders, so we smooth the profile and require a minimum line height
rather than cutting at every zero crossing.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..schema import BBox

log = logging.getLogger(__name__)


def _smooth(profile: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return profile
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(profile, kernel, mode="same")


def segment_lines_projection(
    region_img: np.ndarray,
    min_line_height: int = 12,
    smoothing: int = 5,
    threshold_ratio: float = 0.10,
    pad: int = 2,
) -> list[BBox]:
    """Split a region crop into line boxes using a horizontal projection profile.

    Returns boxes in coordinates *relative to the region crop*. Caller offsets
    them to page coordinates.
    """
    if region_img is None or region_img.size == 0:
        return []

    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY) if region_img.ndim == 3 else region_img
    h, w = gray.shape[:2]
    if h < min_line_height:
        return [BBox(x1=0, y1=0, x2=float(w), y2=float(h))]

    # Ink = dark pixels. Otsu handles the varying scan exposure.
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    profile = _smooth((binary > 0).sum(axis=1).astype(np.float64), smoothing)

    # No ink at all -- a genuinely blank region. Return nothing rather than the
    # whole crop, so the recognizer is never invoked on empty paper. (Contrast
    # with the fallback at the end of this function, which fires when there IS
    # ink but no run survives the height filter -- there we do hand the whole
    # crop to the recognizer and let it try.)
    peak = profile.max()
    if peak <= 0:
        return []
    threshold = peak * threshold_ratio

    # Walk the profile collecting runs above threshold.
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(h):
        above = profile[y] > threshold
        if above and start is None:
            start = y
        elif not above and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, h))

    # Merge runs separated by less than a third of the minimum line height --
    # these are almost always a shirorekha and its hanging conjuncts.
    merged: list[list[int]] = []
    merge_gap = max(1, min_line_height // 3)
    for s, e in runs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    boxes: list[BBox] = []
    for s, e in merged:
        if e - s < min_line_height:
            continue
        y1 = max(0, s - pad)
        y2 = min(h, e + pad)
        # Trim horizontal whitespace so the crop is tight -- CTC models are
        # sensitive to long blank runs.
        band = binary[s:e, :]
        cols = np.where(band.sum(axis=0) > 0)[0]
        if len(cols) == 0:
            continue
        x1 = max(0, int(cols[0]) - pad)
        x2 = min(w, int(cols[-1]) + 1 + pad)
        if x2 - x1 < 4:
            continue
        boxes.append(BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))

    if not boxes:
        log.debug("line segmentation found nothing in a %dx%d region; using whole crop", w, h)
        return [BBox(x1=0, y1=0, x2=float(w), y2=float(h))]

    return boxes


def segment_lines(
    region_img: np.ndarray,
    method: str = "projection",
    min_line_height: int = 12,
    smoothing: int = 5,
) -> list[BBox]:
    if method == "projection":
        return segment_lines_projection(region_img, min_line_height=min_line_height, smoothing=smoothing)
    if method == "none":
        h, w = region_img.shape[:2]
        return [BBox(x1=0, y1=0, x2=float(w), y2=float(h))]
    raise ValueError(f"unknown line segmentation method: {method!r}")


def _projection_gaps(profile: np.ndarray, min_gap: int, threshold: float) -> list[tuple[int, int]]:
    """Return (start, end) run indices where profile > threshold.

    Used for both horizontal (row) and vertical (column) splitting.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(profile):
        if v > threshold and start is None:
            start = i
        elif v <= threshold and start is not None:
            if i - start >= min_gap:
                runs.append((start, i))
            start = None
    if start is not None and len(profile) - start >= min_gap:
        runs.append((start, len(profile)))
    return runs


def segment_table_cells(
    table_img: np.ndarray,
    min_row_height: int = 8,
    min_col_width: int = 8,
    smoothing_rows: int = 3,
    smoothing_cols: int = 3,
    threshold_ratio: float = 0.05,
    pad: int = 1,
) -> list[tuple[int, int, BBox]]:
    """Split a TABLE region crop into (row_idx, col_idx, BBox) cells.

    Algorithm
    ---------
    1. Binarise the crop (ink = dark pixels).
    2. Horizontal projection profile → gap detection → row bands.
    3. For each row band, vertical projection profile → gap detection → columns.
    4. The cross product (row_band × col_segment) gives cell bboxes in
       coordinates *relative to the table_img crop*. The caller translates
       them to page coordinates.

    If the image is too small or blank, returns a single (0, 0, full_bbox)
    cell so OCR still runs on the whole region.
    """
    if table_img is None or table_img.size == 0:
        return []

    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY) if table_img.ndim == 3 else table_img
    h, w = gray.shape[:2]

    if h < min_row_height or w < min_col_width:
        return [(0, 0, BBox(x1=0.0, y1=0.0, x2=float(w), y2=float(h)))]

    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # ── Step 1: find row bands via horizontal projection ──────────────────────
    h_profile = _smooth((binary > 0).sum(axis=1).astype(np.float64), smoothing_rows)
    peak_h = h_profile.max()
    if peak_h <= 0:
        # Blank region — one cell covering everything
        return [(0, 0, BBox(x1=0.0, y1=0.0, x2=float(w), y2=float(h)))]

    thresh_h = peak_h * threshold_ratio
    row_runs = _projection_gaps(h_profile, min_gap=min_row_height, threshold=thresh_h)

    if not row_runs:
        row_runs = [(0, h)]

    # ── Step 2: for each row band find column segments ────────────────────────
    cells: list[tuple[int, int, BBox]] = []
    for row_idx, (ry1, ry2) in enumerate(row_runs):
        ry1_pad = max(0, ry1 - pad)
        ry2_pad = min(h, ry2 + pad)
        row_strip = binary[ry1:ry2, :]

        v_profile = _smooth((row_strip > 0).sum(axis=0).astype(np.float64), smoothing_cols)
        peak_v = v_profile.max()

        if peak_v <= 0:
            # Empty row — one spanning cell
            cells.append((
                row_idx, 0,
                BBox(x1=0.0, y1=float(ry1_pad), x2=float(w), y2=float(ry2_pad))
            ))
            continue

        thresh_v = peak_v * threshold_ratio
        col_runs = _projection_gaps(v_profile, min_gap=min_col_width, threshold=thresh_v)

        if not col_runs:
            col_runs = [(0, w)]

        for col_idx, (cx1, cx2) in enumerate(col_runs):
            cells.append((
                row_idx,
                col_idx,
                BBox(
                    x1=float(max(0, cx1 - pad)),
                    y1=float(ry1_pad),
                    x2=float(min(w, cx2 + pad)),
                    y2=float(ry2_pad),
                )
            ))

    if not cells:
        return [(0, 0, BBox(x1=0.0, y1=0.0, x2=float(w), y2=float(h)))]

    log.debug("table segmentation: %d rows, %d cells total in %dx%d region",
              len(row_runs), len(cells), w, h)
    return cells


def crop(image: np.ndarray, box: BBox) -> np.ndarray:
    """Crop a page/region image by a BBox, clipped to bounds."""
    h, w = image.shape[:2]
    b = box.clip(float(w), float(h))
    y1, y2 = int(round(b.y1)), int(round(b.y2))
    x1, x2 = int(round(b.x1)), int(round(b.x2))
    if y2 <= y1 or x2 <= x1:
        return np.empty((0, 0), dtype=image.dtype)
    return image[y1:y2, x1:x2]


def offset_box(box: BBox, dx: float, dy: float) -> BBox:
    """Translate a region-relative box into page coordinates."""
    return BBox(x1=box.x1 + dx, y1=box.y1 + dy, x2=box.x2 + dx, y2=box.y2 + dy)
