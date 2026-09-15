"""AksharDrishti -- layout-aware transformer OCR for multilingual Indian documents."""

__version__ = "0.1.0"

from .config import ClassMap, Config
from .schema import BBox, Document, Page, PageMeta, Region, RegionType, TextLine

__all__ = [
    "Config", "ClassMap",
    "Document", "Page", "PageMeta", "Region", "RegionType", "TextLine", "BBox",
]
