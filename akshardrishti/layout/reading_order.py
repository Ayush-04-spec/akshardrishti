"""Reading-order recovery via recursive XY-cut.

This module is where the "layout-aware" claim actually gets cashed in. A naive
top-to-bottom sort destroys multi-column documents -- it interleaves the left
and right columns line by line, which is exactly the failure mode described in
the report's critique of Tesseract.

Algorithm
---------
Recursive XY-cut over region bounding boxes:
  1. Project regions onto the X axis; find the widest whitespace gap.
  2. Project onto the Y axis; find the widest whitespace gap.
  3. Cut along whichever gap is wider (preferring a vertical cut on ties, since
     column structure dominates Indian government layouts), recurse on each half.
  4. When no gap exceeds the threshold, order the remaining regions top-to-bottom
     then left-to-right.

Page furniture (headers, footers, page numbers) is removed before the cut and
appended afterwards -- otherwise a running header spanning both columns blocks
every vertical cut and the whole page degenerates to top-down order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..schema import BBox, Page, Region, RegionType

log = logging.getLogger(__name__)


@dataclass
class _Gap:
    position: float
    size: float
    axis: str  # "x" | "y"


def _find_widest_gap(intervals: list[tuple[float, float]], axis: str, min_gap: float) -> _Gap | None:
    """Largest whitespace gap between merged 1-D intervals."""
    if len(intervals) < 2:
        return None

    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    if len(merged) < 2:
        return None

    best: _Gap | None = None
    for i in range(len(merged) - 1):
        gap_start, gap_end = merged[i][1], merged[i + 1][0]
        size = gap_end - gap_start
        if size >= min_gap and (best is None or size > best.size):
            best = _Gap(position=(gap_start + gap_end) / 2.0, size=size, axis=axis)
    return best


def _xy_cut(
    regions: list[Region],
    min_gap_x: float,
    min_gap_y: float,
    depth: int,
    max_depth: int,
) -> list[Region]:
    if len(regions) <= 1:
        return list(regions)

    if depth >= max_depth:
        log.debug("xy_cut: max depth %d reached with %d regions", max_depth, len(regions))
        return sorted(regions, key=lambda r: (r.bbox.y1, r.bbox.x1))

    x_gap = _find_widest_gap([(r.bbox.x1, r.bbox.x2) for r in regions], "x", min_gap_x)
    y_gap = _find_widest_gap([(r.bbox.y1, r.bbox.y2) for r in regions], "y", min_gap_y)

    chosen: _Gap | None
    if x_gap and y_gap:
        # Normalise gap sizes by their thresholds so the comparison is fair;
        # ties go to the vertical cut (column structure first).
        chosen = x_gap if (x_gap.size / min_gap_x) >= (y_gap.size / min_gap_y) else y_gap
    else:
        chosen = x_gap or y_gap

    if chosen is None:
        return sorted(regions, key=lambda r: (r.bbox.y1, r.bbox.x1))

    if chosen.axis == "x":
        first = [r for r in regions if r.bbox.cx < chosen.position]
        second = [r for r in regions if r.bbox.cx >= chosen.position]
    else:
        first = [r for r in regions if r.bbox.cy < chosen.position]
        second = [r for r in regions if r.bbox.cy >= chosen.position]

    # Degenerate split (everything landed on one side) -- stop recursing.
    if not first or not second:
        return sorted(regions, key=lambda r: (r.bbox.y1, r.bbox.x1))

    return (
        _xy_cut(first, min_gap_x, min_gap_y, depth + 1, max_depth)
        + _xy_cut(second, min_gap_x, min_gap_y, depth + 1, max_depth)
    )


def compute_reading_order(
    page: Page,
    method: str = "xy_cut",
    min_gap_ratio: float = 0.02,
    max_recursion: int = 12,
    furniture_classes: list[str] | None = None,
) -> Page:
    """Assign ``reading_order`` to every region on the page, in place."""
    if not page.regions:
        return page

    furniture_types = {
        RegionType(c) for c in (furniture_classes or ["header", "footer", "page_number"])
    }
    body = [r for r in page.regions if r.type not in furniture_types]
    furniture = [r for r in page.regions if r.type in furniture_types]

    if method == "xy_cut":
        min_gap_x = max(1.0, page.meta.width * min_gap_ratio)
        min_gap_y = max(1.0, page.meta.height * min_gap_ratio)
        ordered = _xy_cut(body, min_gap_x, min_gap_y, depth=0, max_depth=max_recursion)
    elif method == "topdown":
        ordered = sorted(body, key=lambda r: (r.bbox.y1, r.bbox.x1))
    elif method == "column_aware":
        ordered = _column_aware(body, page.meta.width)
    else:
        raise ValueError(f"unknown reading-order method: {method!r}")

    # Furniture goes last, top-to-bottom, so it never interrupts the body flow.
    for idx, region in enumerate(ordered):
        region.reading_order = idx
    for offset, region in enumerate(sorted(furniture, key=lambda r: (r.bbox.y1, r.bbox.x1))):
        region.reading_order = len(ordered) + offset

    return page


def _column_aware(regions: list[Region], page_width: float, tolerance: float = 0.15) -> list[Region]:
    """Simpler fallback: cluster regions into columns by x-centre, then sort.

    Cheaper and more predictable than XY-cut on clean two-column layouts; worse
    on nested structures. Kept as an ablation option.
    """
    if not regions:
        return []
    tol = page_width * tolerance
    columns: list[list[Region]] = []
    for region in sorted(regions, key=lambda r: r.bbox.cx):
        placed = False
        for col in columns:
            if abs(col[0].bbox.cx - region.bbox.cx) <= tol:
                col.append(region)
                placed = True
                break
        if not placed:
            columns.append([region])
    columns.sort(key=lambda c: min(r.bbox.x1 for r in c))
    out: list[Region] = []
    for col in columns:
        out.extend(sorted(col, key=lambda r: r.bbox.y1))
    return out


# --------------------------------------------------------------- evaluation


def reading_order_accuracy(predicted: list[int], ground_truth: list[int]) -> float:
    """Fraction of region *pairs* whose relative order is correct.

    Pairwise rather than exact-match because a single inserted region should
    not zero the score for an otherwise correct page. This is the metric
    reported in the results table.
    """
    if len(predicted) != len(ground_truth):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(ground_truth)}")
    n = len(predicted)
    if n < 2:
        return 1.0
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            same = (predicted[i] < predicted[j]) == (ground_truth[i] < ground_truth[j])
            correct += int(same)
    return correct / total if total else 1.0
