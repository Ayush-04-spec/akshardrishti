"""Microsoft Table Transformer (TATR) structure-recognition backend.

Uses two HuggingFace models:
  microsoft/table-transformer-detection          (~120 MB, finds table bbox)
  microsoft/table-transformer-structure-recognition  (~120 MB, finds rows/cols)

The detection model is skipped here — we already have the table bbox from
the YOLO layout detector.  We only call the structure model on the cropped
table image.

Models are downloaded to the HuggingFace cache (~/.cache/huggingface/) on
first use and never re-downloaded.  Total download is ~240 MB.

Interface
---------
segment_table_cells_tatr(table_img, device, threshold) -> list[(row, col, BBox)]

Identical return type to segment_table_cells() in lines.py so the pipeline
can swap backends through config without changing any other code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..schema import BBox

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

# HuggingFace model IDs
_STRUCTURE_MODEL_ID = "microsoft/table-transformer-structure-recognition"

# TATR label map (from the model card)
_LABEL_MAP = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table projected row header",
    5: "table spanning cell",
    6: "no object",
}

# ─────────────────────────────────────── lazy singleton ──────────────────────

_structure_model = None
_structure_processor = None


def _load_structure_model(device: str) -> tuple:
    """Load (and cache) the TATR structure model.  Thread-unsafe but fine for
    single-threaded Streamlit + pipeline use."""
    global _structure_model, _structure_processor

    if _structure_model is not None:
        return _structure_model, _structure_processor

    log.info("Loading TATR structure model from HuggingFace hub (%s) …", _STRUCTURE_MODEL_ID)
    try:
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        import torch
    except ImportError as exc:
        raise ImportError(
            "TATR backend needs transformers and timm: "
            "pip install transformers timm"
        ) from exc

    _structure_processor = AutoImageProcessor.from_pretrained(_STRUCTURE_MODEL_ID)
    _structure_model = TableTransformerForObjectDetection.from_pretrained(_STRUCTURE_MODEL_ID)
    _structure_model.eval()
    _structure_model.to(device)

    log.info("TATR structure model loaded on %s", device)
    return _structure_model, _structure_processor


# ─────────────────────────────────────── helpers ─────────────────────────────

def _boxes_to_grid(
    row_boxes: list[tuple[float, float, float, float]],
    col_boxes: list[tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> list[tuple[int, int, BBox]]:
    """Cross row and column bboxes into a cell grid.

    Each row box is (x1,y1,x2,y2) in pixel coords relative to the table crop.
    Same for column boxes.  We clip to image bounds and skip degenerate cells.
    """
    cells: list[tuple[int, int, BBox]] = []

    for r_idx, (rx1, ry1, rx2, ry2) in enumerate(row_boxes):
        for c_idx, (cx1, cy1, cx2, cy2) in enumerate(col_boxes):
            # Cell bbox = intersection of row band × column band
            cell_x1 = max(0.0, cx1)
            cell_y1 = max(0.0, ry1)
            cell_x2 = min(float(img_w), cx2)
            cell_y2 = min(float(img_h), ry2)

            if cell_x2 - cell_x1 < 4 or cell_y2 - cell_y1 < 4:
                continue

            cells.append((r_idx, c_idx, BBox(
                x1=cell_x1, y1=cell_y1,
                x2=cell_x2, y2=cell_y2,
            )))

    return cells


def _nms_1d(intervals: list[tuple[float, float, float, float]],
            axis: str,
            iou_threshold: float = 0.5) -> list[tuple[float, float, float, float]]:
    """1-D NMS along the primary axis of rows (y) or columns (x).

    Sorts by score (descending), suppresses any box whose 1-D IoU with an
    already-kept box exceeds iou_threshold.

    Parameters
    ----------
    intervals : list of (x1, y1, x2, y2)  -- already filtered by score
    axis      : 'y' for rows, 'x' for columns
    """
    if not intervals:
        return intervals

    # sort by centre along the primary axis (best proxy for score when we
    # don't carry scores through here; stable ordering is fine for dedup)
    if axis == 'y':
        intervals_sorted = sorted(intervals, key=lambda b: b[1])
    else:
        intervals_sorted = sorted(intervals, key=lambda b: b[0])

    kept: list[tuple[float, float, float, float]] = []
    for box in intervals_sorted:
        suppress = False
        for k in kept:
            if axis == 'y':
                lo1, hi1 = box[1], box[3]
                lo2, hi2 = k[1], k[3]
            else:
                lo1, hi1 = box[0], box[2]
                lo2, hi2 = k[0], k[2]
            inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
            union = (hi1 - lo1) + (hi2 - lo2) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(box)
    return kept


def _extract_row_col_boxes(
    results,
    img_w: int,
    img_h: int,
    threshold: float,
) -> tuple[list, list]:
    """Pull row and column bboxes out of TATR's output dict.

    results["boxes"]  — (N, 4) float tensor in [x1,y1,x2,y2] pixel coords
    results["scores"] — (N,) float tensor
    results["labels"] — (N,) int tensor
    """
    import torch

    boxes  = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()

    row_boxes: list[tuple[float, float, float, float]] = []
    col_boxes: list[tuple[float, float, float, float]] = []

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        label_name = _LABEL_MAP.get(int(label), "")
        x1, y1, x2, y2 = (
            max(0.0, float(box[0])),
            max(0.0, float(box[1])),
            min(float(img_w), float(box[2])),
            min(float(img_h), float(box[3])),
        )
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        if label_name in ("table row", "table column header",
                          "table projected row header"):
            row_boxes.append((x1, y1, x2, y2))
        elif label_name == "table column":
            col_boxes.append((x1, y1, x2, y2))

    # Sort rows top→bottom, columns left→right
    row_boxes.sort(key=lambda b: b[1])
    col_boxes.sort(key=lambda b: b[0])

    # Deduplicate overlapping detections with 1-D NMS
    row_boxes = _nms_1d(row_boxes, axis='y', iou_threshold=0.5)
    col_boxes = _nms_1d(col_boxes, axis='x', iou_threshold=0.5)

    return row_boxes, col_boxes


# ─────────────────────────────────────── public API ──────────────────────────

def segment_table_cells_tatr(
    table_img: np.ndarray,
    device: str = "cpu",
    threshold: float = 0.5,
) -> list[tuple[int, int, BBox]]:
    """Segment a TABLE region crop into (row_idx, col_idx, BBox) cells using
    Microsoft's Table Transformer structure-recognition model.

    Parameters
    ----------
    table_img : np.ndarray
        BGR or grayscale crop of the table region (region-relative coords).
    device : str
        'cuda' or 'cpu'.
    threshold : float
        Confidence threshold for TATR detections (default 0.5).

    Returns
    -------
    List of (row_idx, col_idx, BBox) where BBox is in table_img pixel coords.
    Falls back to a single full-image cell if TATR finds nothing useful.
    """
    if table_img is None or table_img.size == 0:
        return []

    import torch
    from PIL import Image as PILImage

    h, w = table_img.shape[:2]

    # Convert to PIL RGB (TATR was trained on RGB)
    if table_img.ndim == 2:
        pil_img = PILImage.fromarray(table_img).convert("RGB")
    else:
        import cv2
        pil_img = PILImage.fromarray(cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB))

    model, processor = _load_structure_model(device)

    # Preprocess
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: rescale boxes back to original image pixel coords
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    row_boxes, col_boxes = _extract_row_col_boxes(results, img_w=w, img_h=h,
                                                  threshold=threshold)

    log.debug("TATR found %d rows, %d columns in %dx%d table",
              len(row_boxes), len(col_boxes), w, h)

    # Need at least 1 row and 1 column to build a grid
    if not row_boxes or not col_boxes:
        log.warning(
            "TATR found %d rows / %d cols — falling back to single cell",
            len(row_boxes), len(col_boxes),
        )
        return [(0, 0, BBox(x1=0.0, y1=0.0, x2=float(w), y2=float(h)))]

    cells = _boxes_to_grid(row_boxes, col_boxes, img_w=w, img_h=h)

    if not cells:
        return [(0, 0, BBox(x1=0.0, y1=0.0, x2=float(w), y2=float(h)))]

    return cells
