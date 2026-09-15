"""Document layout detection: YOLO11 / YOLOv10 / DocLayout-YOLO + SAHI.

SAHI (Slicing Aided Hyper Inference) is the report's answer to dense text
regions: slice the page into overlapping tiles, detect on each, then merge. It
genuinely helps on newspaper-style pages where a single 1024px resize shrinks
body text below the detector's effective resolution.

It is also ~4x slower. So we auto-trigger: run one cheap unsliced pass, and only
slice when the page actually looks dense. That heuristic is itself an ablation
row in the results table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import BBox, Page, PageMeta, Region, RegionType

log = logging.getLogger(__name__)


class LayoutDetector:
    """Wraps an Ultralytics-style detector and exposes ``detect(image) -> Page``."""

    def __init__(self, config: dict[str, Any] | None = None, class_names: list[str] | None = None,
                 device: str = "auto") -> None:
        self.config = config or {}
        self.device = device
        self.model_kind = self.config.get("model", "yolo11l")
        self.weights = Path(self.config.get("weights", "./weights/layout/yolo11l_indicdlp.pt"))
        self.imgsz = int(self.config.get("imgsz", 1024))
        self.conf = float(self.config.get("conf", 0.25))
        self.iou = float(self.config.get("iou", 0.45))
        self.max_det = int(self.config.get("max_det", 300))
        self.class_names = class_names or [t.value for t in RegionType]
        self._model = None
        self._sahi_model = None

    # ------------------------------------------------------------ loading
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.weights.exists():
            raise FileNotFoundError(
                f"layout weights not found at {self.weights}.\n"
                f"Train:    python train/train_layout.py --config configs/pipeline.yaml\n"
                f"Or use a released IndicDLP / DocLayout-YOLO checkpoint and point "
                f"layout.weights at it."
            )

        if self.model_kind == "doclayout_yolo":
            from doclayout_yolo import YOLOv10  # type: ignore

            self._model = YOLOv10(str(self.weights))
        else:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))

        names = getattr(self._model, "names", None)
        if isinstance(names, dict) and names:
            self.class_names = [names[i] for i in sorted(names)]
        log.info("layout detector loaded: %s (%d classes)", self.weights.name, len(self.class_names))

    def _ensure_sahi(self):
        if self._sahi_model is not None:
            return self._sahi_model
        from sahi import AutoDetectionModel

        self._sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(self.weights),
            confidence_threshold=self.conf,
            device="cuda" if self.device in ("auto", "cuda") else self.device,
        )
        return self._sahi_model

    # ---------------------------------------------------------- inference
    def _to_regions(self, boxes: list[tuple[BBox, int, float]], width: int, height: int) -> list[Region]:
        regions: list[Region] = []
        for i, (box, cls_idx, conf) in enumerate(boxes):
            name = self.class_names[cls_idx] if 0 <= cls_idx < len(self.class_names) else "text"
            try:
                rtype = RegionType(name)
            except ValueError:
                log.debug("detector emitted unknown class %r -> text", name)
                rtype = RegionType.TEXT
            regions.append(
                Region(
                    region_id=i,
                    type=rtype,
                    bbox=box.clip(float(width), float(height)),
                    detection_confidence=float(min(max(conf, 0.0), 1.0)),
                )
            )
        return regions

    def _detect_plain(self, image: np.ndarray) -> list[tuple[BBox, int, float]]:
        self._ensure_model()
        results = self._model.predict(
            source=image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            max_det=self.max_det, device=None if self.device == "auto" else self.device,
            verbose=False,
        )
        out: list[tuple[BBox, int, float]] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for xyxy, cls_idx, conf in zip(
                boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()
            ):
                try:
                    out.append((BBox.from_xyxy(xyxy), int(cls_idx), float(conf)))
                except ValueError:
                    continue  # degenerate box
        return out

    def _detect_sliced(self, image: np.ndarray) -> list[tuple[BBox, int, float]]:
        from sahi.predict import get_sliced_prediction

        sahi_cfg = self.config.get("sahi", {}) or {}
        result = get_sliced_prediction(
            image,
            self._ensure_sahi(),
            slice_height=int(sahi_cfg.get("slice_height", 640)),
            slice_width=int(sahi_cfg.get("slice_width", 640)),
            overlap_height_ratio=float(sahi_cfg.get("overlap_ratio", 0.2)),
            overlap_width_ratio=float(sahi_cfg.get("overlap_ratio", 0.2)),
            verbose=0,
        )
        out: list[tuple[BBox, int, float]] = []
        for pred in result.object_prediction_list:
            bx = pred.bbox
            try:
                out.append((
                    BBox(x1=float(bx.minx), y1=float(bx.miny), x2=float(bx.maxx), y2=float(bx.maxy)),
                    int(pred.category.id),
                    float(pred.score.value),
                ))
            except ValueError:
                continue
        return out

    @staticmethod
    def _looks_dense(boxes: list[tuple[BBox, int, float]], page_area: float,
                     min_regions: int, min_small_frac: float) -> bool:
        if len(boxes) < min_regions:
            return False
        small = sum(1 for b, _, _ in boxes if b.area < page_area * 0.01)
        return (small / len(boxes)) >= min_small_frac

    def detect(self, image: np.ndarray, source_path: str = "", page_index: int = 0) -> Page:
        """Run detection and return a populated (but unordered) ``Page``."""
        h, w = image.shape[:2]
        boxes = self._detect_plain(image)

        sahi_cfg = self.config.get("sahi", {}) or {}
        if sahi_cfg.get("enabled", False):
            auto = sahi_cfg.get("auto_trigger", {}) or {}
            should_slice = True
            if auto.get("enabled", True):
                should_slice = self._looks_dense(
                    boxes, float(h * w),
                    int(auto.get("min_regions", 25)),
                    float(auto.get("min_small_region_frac", 0.4)),
                )
            if should_slice:
                log.debug("running SAHI sliced inference (%d boxes in plain pass)", len(boxes))
                try:
                    sliced = self._detect_sliced(image)
                    if len(sliced) > len(boxes):
                        boxes = sliced
                except Exception as exc:  # noqa: BLE001
                    log.warning("SAHI pass failed (%s); keeping unsliced result", exc)

        return Page(
            meta=PageMeta(source_path=source_path, page_index=page_index, width=w, height=h),
            regions=self._to_regions(boxes, w, h),
        )


def merge_overlapping(regions: list[Region], iou_threshold: float = 0.75) -> list[Region]:
    """Drop near-duplicate detections, keeping the more confident one.

    SAHI's own NMS handles most of this, but tile boundaries occasionally leave
    a pair of highly-overlapping boxes with different class labels.
    """
    survivors: list[Region] = []
    for region in sorted(regions, key=lambda r: r.detection_confidence, reverse=True):
        if all(region.bbox.iou(keep.bbox) < iou_threshold for keep in survivors):
            survivors.append(region)
    for i, region in enumerate(survivors):
        region.region_id = i
    return survivors
