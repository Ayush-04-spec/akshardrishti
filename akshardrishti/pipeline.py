"""The AksharDrishti orchestrator.

    preprocess -> layout detect -> reading order -> line segment
              -> script ID -> recognize -> postprocess -> structured JSON

Every stage is toggleable from configs/pipeline.yaml, which is what makes the
ablation table (§ "with vs without layout detection", etc.) a matter of running
the same command with a different override rather than maintaining four forks.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Config
from .layout.detector import LayoutDetector, merge_overlapping
from .layout.lines import crop, offset_box, segment_lines
from .layout.reading_order import compute_reading_order
from .postprocess.text_repair import clean_text
from .preprocess.enhance import preprocess_page
from .recognize.base import ScriptRouter, build_recognizer
from .schema import BBox, Document, Page, PageMeta, Region, RegionType, TextLine
from .script_id.classifier import ScriptIdentifier

log = logging.getLogger(__name__)


class AksharDrishtiPipeline:
    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config.load()
        self.device = self.cfg.get("project.device", "auto")

        self._detector: LayoutDetector | None = None
        self._router: ScriptRouter | None = None
        self._script_id: ScriptIdentifier | None = None

    # ------------------------------------------------------- lazy components
    @property
    def detector(self) -> LayoutDetector:
        if self._detector is None:
            self._detector = LayoutDetector(self.cfg.get("layout", {}), device=self.device)
        return self._detector

    @property
    def router(self) -> ScriptRouter:
        if self._router is None:
            backends = self.cfg.get("recognize.backends", {}) or {}
            deva_name = self.cfg.get("recognize.deva_backend", "crnn_mozhi")
            latin_name = self.cfg.get("recognize.latin_backend", "trocr_printed")
            self._router = ScriptRouter(
                deva=build_recognizer(deva_name, backends.get(deva_name, {}), self.device),
                latin=build_recognizer(latin_name, backends.get(latin_name, {}), self.device),
                confidence_floor=float(self.cfg.get("script_id.confidence_floor", 0.6)),
            )
        return self._router

    @property
    def script_id(self) -> ScriptIdentifier:
        if self._script_id is None:
            self._script_id = ScriptIdentifier(self.cfg.get("script_id", {}), device=self.device)
        return self._script_id

    # -------------------------------------------------------------- loading
    @staticmethod
    def load_images(path: str | Path, dpi: int = 300) -> list[np.ndarray]:
        """Load an image or rasterise every page of a PDF."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() == ".pdf":
            try:
                import fitz  # PyMuPDF
            except ImportError as exc:
                raise ImportError("PDF input needs PyMuPDF: `pip install pymupdf`") from exc

            pages: list[np.ndarray] = []
            with fitz.open(path) as doc:
                zoom = dpi / 72.0
                matrix = fitz.Matrix(zoom, zoom)
                for page in doc:
                    pix = page.get_pixmap(matrix=matrix)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    if pix.n == 4:
                        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    elif pix.n == 3:
                        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    pages.append(arr)
            return pages

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"could not decode image: {path}")
        return [img]

    # ------------------------------------------------------------- stages
    def _recognize_region(self, page_img: np.ndarray, region: Region) -> None:
        """Segment a region into lines, identify script, recognise, fill in."""
        if not region.type.is_text_bearing:
            return

        region_img = crop(page_img, region.bbox)
        if region_img.size == 0:
            return

        ls_cfg = self.cfg.get("recognize.line_segmentation", {}) or {}
        if ls_cfg.get("enabled", True):
            line_boxes = segment_lines(
                region_img,
                method=ls_cfg.get("method", "projection"),
                min_line_height=int(ls_cfg.get("min_line_height", 12)),
                smoothing=int(ls_cfg.get("smoothing", 5)),
            )
        else:
            h, w = region_img.shape[:2]
            line_boxes = [BBox(x1=0, y1=0, x2=float(w), y2=float(h))]

        scripts: list[str] = []
        for lb in line_boxes:
            line_img = crop(region_img, lb)
            if line_img.size == 0:
                continue

            if self.cfg.get("script_id.enabled", True):
                script, script_conf = self.script_id.identify(line_img)
            else:
                script, script_conf = "unknown", None

            result = self.router.recognize(line_img, script=script, script_confidence=script_conf)
            text = clean_text(
                result.text,
                unicode_form=self.cfg.get("postprocess.unicode_normalize", "NFC"),
                repair_shirorekha=self.cfg.get("postprocess.repair_shirorekha", True),
                strip_control=self.cfg.get("postprocess.strip_control_chars", True),
                collapse_ws=self.cfg.get("postprocess.collapse_whitespace", True),
            )
            if not text:
                continue

            scripts.append(script)
            region.lines.append(
                TextLine(
                    text=text,
                    bbox=offset_box(lb, region.bbox.x1, region.bbox.y1),
                    script=script if script in ("deva", "latin", "mixed") else "unknown",
                    confidence=result.confidence,
                    backend=result.backend,
                )
            )

        region.compose_text()
        if scripts:
            uniq = set(scripts)
            region.script = scripts[0] if len(uniq) == 1 else "mixed"

    def process_page(self, image: np.ndarray, source_path: str = "", page_index: int = 0) -> Page:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        pre = preprocess_page(image, self.cfg.get("preprocess", {}))
        timings["preprocess"] = time.perf_counter() - t0
        page_img = pre.image

        t0 = time.perf_counter()
        if self.cfg.get("layout.enabled", True):
            page = self.detector.detect(page_img, source_path=source_path, page_index=page_index)
            page.regions = merge_overlapping(page.regions)
        else:
            # Ablation: no layout detection -- treat the whole page as one text block.
            h, w = page_img.shape[:2]
            page = Page(
                meta=PageMeta(source_path=source_path, page_index=page_index, width=w, height=h),
                regions=[
                    Region(
                        region_id=0, type=RegionType.TEXT,
                        bbox=BBox(x1=0, y1=0, x2=float(w), y2=float(h)),
                        detection_confidence=1.0,
                    )
                ],
            )
        timings["layout"] = time.perf_counter() - t0

        page.meta.preprocessing_applied = pre.applied
        page.meta.rotation_corrected_deg = pre.rotation_deg

        t0 = time.perf_counter()
        compute_reading_order(
            page,
            method=self.cfg.get("reading_order.method", "xy_cut"),
            min_gap_ratio=float(self.cfg.get("reading_order.min_gap_ratio", 0.02)),
            max_recursion=int(self.cfg.get("reading_order.max_recursion", 12)),
            furniture_classes=self.cfg.get("reading_order.furniture_classes"),
        )
        timings["reading_order"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        for region in page.ordered_regions():
            self._recognize_region(page_img, region)
        timings["recognition"] = time.perf_counter() - t0

        self._last_page_timings = timings
        return page

    def process(self, path: str | Path) -> Document:
        """Full document: load -> per-page pipeline -> Document."""
        path = Path(path)
        started = time.perf_counter()

        images = self.load_images(path, dpi=int(self.cfg.get("preprocess.target_dpi", 300)))
        doc = Document(source_path=str(path), pipeline_config={"config_hash": self.cfg.hash()})

        totals: dict[str, float] = {}
        for idx, image in enumerate(images):
            page = self.process_page(image, source_path=str(path), page_index=idx)
            doc.pages.append(page)
            for k, v in getattr(self, "_last_page_timings", {}).items():
                totals[k] = totals.get(k, 0.0) + v

        totals["total"] = time.perf_counter() - started
        totals["seconds_per_page"] = totals["total"] / max(len(images), 1)
        doc.timings_s = totals
        return doc

    def save_outputs(self, doc: Document, out_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(doc.source_path).stem
        formats = self.cfg.get("output.formats", ["json"])
        written: dict[str, Path] = {}

        if "json" in formats:
            p = out_dir / f"{stem}.json"
            doc.save_json(p, indent=int(self.cfg.get("output.json_indent", 2)))
            written["json"] = p
        if "markdown" in formats:
            p = out_dir / f"{stem}.md"
            p.write_text(doc.to_markdown(), encoding="utf-8")
            written["markdown"] = p
        if "text" in formats:
            p = out_dir / f"{stem}.txt"
            p.write_text(doc.full_text(), encoding="utf-8")
            written["text"] = p
        return written


def visualize_regions(image: np.ndarray, page: Page, thickness: int = 2) -> np.ndarray:
    """Draw detected regions with reading-order numbers -- for the demo and for
    the qualitative figures in the paper."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    colors = {
        RegionType.TITLE: (0, 0, 255), RegionType.TEXT: (0, 200, 0),
        RegionType.LIST: (0, 165, 255), RegionType.TABLE: (255, 0, 0),
        RegionType.FIGURE: (255, 0, 255), RegionType.CAPTION: (128, 128, 0),
        RegionType.HEADER: (128, 128, 128), RegionType.FOOTER: (128, 128, 128),
        RegionType.PAGE_NUMBER: (100, 100, 100), RegionType.STAMP_SEAL: (0, 255, 255),
        RegionType.SIGNATURE: (255, 128, 0),
    }
    for region in page.ordered_regions():
        color = colors.get(region.type, (0, 255, 0))
        x1, y1, x2, y2 = (int(v) for v in region.bbox.as_xyxy())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"{region.reading_order}:{region.type.value}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas
