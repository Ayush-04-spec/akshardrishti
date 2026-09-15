"""Structured output schema for AksharDrishti.

This is the contract between every stage of the pipeline and the thing that
matters most in the report: a JSON document that preserves *both* the text and
the layout structure it came from.

Design notes
------------
* Bounding boxes are always absolute pixel coords on the *preprocessed* page,
  in xyxy order. Page dimensions travel with the document so consumers can
  normalise if they want to.
* ``reading_order`` is an explicit integer, not the list index. Regions may be
  stored in detection order; the reading order is a separate, computed opinion.
* Every text-bearing region carries a confidence so downstream consumers can
  filter. Confidence is always in [0, 1]; backends that do not expose one
  report ``None`` rather than faking 1.0.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RegionType(str, Enum):
    """The 11-class taxonomy (see configs/class_map.yaml)."""

    TITLE = "title"
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    STAMP_SEAL = "stamp_seal"
    SIGNATURE = "signature"

    @property
    def is_text_bearing(self) -> bool:
        """Regions we actually run OCR on."""
        return self not in {RegionType.FIGURE, RegionType.STAMP_SEAL, RegionType.SIGNATURE}

    @property
    def is_furniture(self) -> bool:
        """Page furniture -- pulled out of the reading flow and appended."""
        return self in {RegionType.HEADER, RegionType.FOOTER, RegionType.PAGE_NUMBER}


Script = Literal["deva", "latin", "mixed", "unknown"]


class BBox(BaseModel):
    """Axis-aligned box in absolute pixels, xyxy."""

    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_order(self) -> BBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate bbox: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def iou(self, other: BBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, width: float, height: float) -> BBox:
        """Clamp to page bounds. Detectors occasionally emit boxes off-page."""
        return BBox(
            x1=max(0.0, min(self.x1, width - 1)),
            y1=max(0.0, min(self.y1, height - 1)),
            x2=max(1.0, min(self.x2, width)),
            y2=max(1.0, min(self.y2, height)),
        )

    @classmethod
    def from_xyxy(cls, box: tuple[float, float, float, float] | list[float]) -> BBox:
        return cls(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3]))


class TextLine(BaseModel):
    """A single recognised line within a region."""

    text: str
    bbox: BBox
    script: Script = "unknown"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    backend: str | None = Field(default=None, description="Which recognizer produced this")


class Region(BaseModel):
    """One detected layout region, optionally with recognised text."""

    region_id: int
    type: RegionType
    bbox: BBox
    detection_confidence: float = Field(ge=0.0, le=1.0)
    reading_order: int = Field(default=-1, description="-1 until reading order is computed")

    script: Script = "unknown"
    script_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    lines: list[TextLine] = Field(default_factory=list)
    text: str = ""
    recognition_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    backend: str | None = None

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    def compose_text(self, joiner: str = "\n") -> str:
        """Rebuild region text from its lines and cache it on ``text``."""
        self.text = joiner.join(ln.text for ln in self.lines if ln.text.strip()).strip()
        confs = [ln.confidence for ln in self.lines if ln.confidence is not None]
        self.recognition_confidence = (sum(confs) / len(confs)) if confs else None
        return self.text


class PageMeta(BaseModel):
    source_path: str
    page_index: int = 0
    width: int
    height: int
    dpi: int | None = None
    preprocessing_applied: list[str] = Field(default_factory=list)
    rotation_corrected_deg: float = 0.0


class Page(BaseModel):
    meta: PageMeta
    regions: list[Region] = Field(default_factory=list)

    def ordered_regions(self, include_furniture: bool = True) -> list[Region]:
        """Regions in reading order. Unordered regions (-1) sort last."""
        rs = self.regions if include_furniture else [r for r in self.regions if not r.type.is_furniture]
        return sorted(rs, key=lambda r: (r.reading_order < 0, r.reading_order, r.bbox.y1, r.bbox.x1))

    def full_text(self, include_furniture: bool = False) -> str:
        return "\n\n".join(
            r.text for r in self.ordered_regions(include_furniture) if r.text.strip()
        )

    def to_markdown(self) -> str:
        """Lossy but readable export -- useful for eyeballing and for the demo."""
        out: list[str] = []
        for r in self.ordered_regions(include_furniture=False):
            if not r.text.strip():
                continue
            if r.type is RegionType.TITLE:
                out.append(f"## {r.text}")
            elif r.type is RegionType.LIST:
                out.extend(f"- {ln.text}" for ln in r.lines if ln.text.strip())
            elif r.type is RegionType.TABLE:
                out.append(f"```\n{r.text}\n```")
            elif r.type is RegionType.CAPTION:
                out.append(f"*{r.text}*")
            else:
                out.append(r.text)
        return "\n\n".join(out)


class Document(BaseModel):
    """Top-level output. One input file -> one Document, N pages."""

    model_config = ConfigDict(validate_assignment=True)

    schema_version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_path: str
    pages: list[Page] = Field(default_factory=list)
    pipeline_config: dict[str, Any] = Field(default_factory=dict)
    timings_s: dict[str, float] = Field(default_factory=dict)

    def full_text(self) -> str:
        return "\n\n".join(p.full_text() for p in self.pages)

    def to_markdown(self) -> str:
        return "\n\n---\n\n".join(p.to_markdown() for p in self.pages)

    def save_json(self, path: str | Path, indent: int = 2) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=indent),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> Document:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
