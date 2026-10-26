"""Conservative order/geometry alignment of crop candidates to page GT spans.

This module never OCRs crops and never upgrades an automatic suggestion to a
verified label. Its scores are ranking hints for human review only.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable


STATUSES = {"candidate_match", "ambiguous_match", "boundary_review", "no_match", "verified"}


@dataclass(frozen=True)
class GTLine:
    text: str
    start: int
    end: int


def nfc(text: str) -> str:
    """Canonical Unicode normalization; does not remove text or punctuation."""
    return unicodedata.normalize("NFC", text)


def _lines_with_spans(text: str) -> list[GTLine]:
    lines: list[GTLine] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        content = raw.rstrip("\r\n")
        if content.strip():
            left = len(content) - len(content.lstrip())
            right = len(content.rstrip())
            lines.append(GTLine(content[left:right], offset + left, offset + right))
        offset += len(raw)
    return lines


def _crop_width(crop: dict) -> float:
    if crop.get("height32_width_before_cap_px") not in (None, ""):
        return float(crop["height32_width_before_cap_px"])
    width = float(crop.get("crop_width_px", 0) or 0)
    height = float(crop.get("crop_height_px", 0) or 0)
    return width * 32.0 / max(height, 1.0)


def _line_geometry_cost(width: float, text: str, scale: float) -> float:
    # A weak page-specific prior: expected normalized crop width per GT
    # character. This uses only dimensions and ground-truth line lengths.
    ratio = max(width, 1.0) / max(len(text), 1) / max(scale, 1e-9)
    return min(abs(math.log(max(ratio, 1e-9))), 3.0)


def align_ordered_crops(crops: list[dict], gt_text: str, *, table_page: bool = False) -> list[dict]:
    """Propose monotonic GT spans using line order and width/character fit.

    No pixel content or OCR output is consulted. Results remain unverified.
    """
    lines = _lines_with_spans(gt_text)
    if not crops:
        return []
    if not lines:
        return [{**crop, "candidate_transcription": "", "gt_span_start": None,
                 "gt_span_end": None, "matching_gt_span": "", "alignment_status": "no_match",
                 "alignment_method": "ordered_width_char_dp_v1", "alignment_confidence": 0.0,
                 "manual_review_required": True, "manual_review_reason": "page_ground_truth_has_no_nonblank_lines",
                 "eligible_for_scoring": False, "verification_status": "unverified"} for crop in crops]

    widths = [_crop_width(crop) for crop in crops]
    text_lengths = [max(1, len(line.text)) for line in lines]
    scale = sum(widths) / max(1, sum(text_lengths))
    n, m = len(crops), len(lines)
    # Global monotonic sequence alignment. Match one crop to one or two adjacent
    # transcription lines, or leave crop/GT line unmatched. The 2-line option
    # is only a review hypothesis, never a verified multi-line label.
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str, float] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    crop_gap, gt_gap, multi_penalty = 0.90, 0.85, 0.24
    for i in range(n + 1):
        for j in range(m + 1):
            current = dp[i][j]
            if not math.isfinite(current):
                continue
            def update(ni: int, nj: int, cost: float, prev: tuple[int, int, str, float]) -> None:
                value = current + cost
                if value < dp[ni][nj]:
                    dp[ni][nj] = value
                    back[ni][nj] = prev
            if i < n:
                update(i + 1, j, crop_gap, (i, j, "skip_crop", crop_gap))
            if j < m:
                update(i, j + 1, gt_gap, (i, j, "skip_gt_line", gt_gap))
            if i < n and j < m:
                cost = _line_geometry_cost(widths[i], lines[j].text, scale)
                update(i + 1, j + 1, cost, (i, j, "match_1", cost))
                if j + 1 < m:
                    joined = lines[j].text + " " + lines[j + 1].text
                    cost2 = _line_geometry_cost(widths[i], joined, scale) + multi_penalty
                    update(i + 1, j + 2, cost2, (i, j, "match_2", cost2))

    operations: list[tuple[int, int, str, float]] = []
    i, j = n, m
    while i or j:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj, op, cost = prev
        operations.append((pi, pj, op, cost))
        i, j = pi, pj
    operations.reverse()
    matches: dict[int, tuple[int, int, str, float]] = {}
    for ci, gj, op, cost in operations:
        if op.startswith("match"):
            consumed = 1 if op == "match_1" else 2
            matches[ci] = (gj, gj + consumed, op, cost)

    result = []
    for ci, crop in enumerate(crops):
        common = {
            **crop, "alignment_method": "ordered_width_char_dp_v1",
            "verification_status": "unverified", "eligible_for_scoring": False,
        }
        if ci not in matches:
            result.append({**common, "candidate_transcription": "", "gt_span_start": None,
                           "gt_span_end": None, "matching_gt_span": "", "alignment_status": "no_match",
                           "alignment_confidence": 0.0, "manual_review_required": True,
                           "manual_review_reason": "no_monotonic_ground_truth_span_fit"})
            continue
        gj, gj_end, op, cost = matches[ci]
        chosen = lines[gj:gj_end]
        span_start, span_end = chosen[0].start, chosen[-1].end
        passage = gt_text[span_start:span_end]
        # Score is a heuristic agreement between crop width and proposed text
        # length, not a probability that the transcription is correct.
        confidence = max(0.0, min(1.0, math.exp(-cost)))
        local_costs = sorted(_line_geometry_cost(widths[ci], line.text, scale) for line in lines)
        # Geometric tie detection is deliberately conservative. Near-equal
        # alternatives anywhere on the same page make a short text span weak.
        ambiguity_margin = (local_costs[1] - local_costs[0]) if len(local_costs) > 1 else 1.0
        reasons = []
        status = "candidate_match" if op == "match_1" and cost <= 0.45 and ambiguity_margin >= 0.08 else "ambiguous_match"
        if op == "match_2":
            status = "boundary_review"
            reasons.append("one_crop_proposed_for_two_ground_truth_lines")
        if table_page:
            status = "boundary_review"
            reasons.append("table_page_projection_crop_requires_cell_boundary_review")
        if float(crop.get("crop_height_px", 0) or 0) <= 8:
            status = "boundary_review"
            reasons.append("unusually_thin_crop")
        if widths[ci] > 1200:
            reasons.append("very_wide_crop_check_for_wrapping_or_merged_content")
        if status == "ambiguous_match":
            reasons.append("weak_or_nonunique_width_and_order_alignment")
        result.append({
            **common, "candidate_transcription": nfc(passage),
            "gt_span_start": span_start, "gt_span_end": span_end,
            "matching_gt_span": passage, "alignment_status": status,
            "alignment_confidence": round(confidence, 4),
            "confidence_interpretation": "heuristic width/order fit; not calibrated correctness probability",
            "ambiguity_margin": round(ambiguity_margin, 4),
            "manual_review_required": True,
            "manual_review_reason": ";".join(dict.fromkeys(reasons)) or "automatic_proposal_requires_visual_confirmation",
        })
    return result


def scorable_pairs(rows: Iterable[dict]) -> list[tuple[str, str]]:
    """Only explicitly verified crop transcriptions can enter crop evaluation."""
    return [
        (str(row.get("candidate_transcription", "")), str(row.get("prediction", "")))
        for row in rows
        if row.get("alignment_status") == "verified"
        and row.get("verification_status") == "verified"
        and str(row.get("candidate_transcription", "")).strip()
    ]
