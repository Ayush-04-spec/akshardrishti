"""Document export: JSON → DOCX, JSON → PDF, PDF → DOCX, DOCX → PDF.

All functions return bytes so Streamlit can offer them as download buttons
without writing temporary files.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Document


# ─────────────────────────────── JSON → DOCX ─────────────────────────────────

def document_to_docx(doc: "Document") -> bytes:
    """Convert an AksharDrishti Document to a DOCX file (bytes)."""
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    d = DocxDocument()

    # Title
    title_para = d.add_heading("AksharDrishti — Extracted Document", level=0)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Source info
    d.add_paragraph(f"Source: {Path(doc.source_path).name}")
    d.add_paragraph(f"Pages: {len(doc.pages)}")
    d.add_paragraph("")

    from .schema import RegionType
    for page_idx, page in enumerate(doc.pages):
        if len(doc.pages) > 1:
            d.add_heading(f"Page {page_idx + 1}", level=1)

        for region in page.ordered_regions(include_furniture=True):
            if not region.text.strip():
                continue

            rtype = region.type

            if rtype == RegionType.TITLE:
                d.add_heading(region.text.strip(), level=2)

            elif rtype == RegionType.HEADER:
                para = d.add_paragraph(region.text.strip())
                para.runs[0].bold = True if para.runs else None
                para.runs[0].font.color.rgb = RGBColor(0x50, 0x50, 0x50) if para.runs else None

            elif rtype == RegionType.LIST:
                for line in region.lines:
                    if line.text.strip():
                        d.add_paragraph(line.text.strip(), style="List Bullet")

            elif rtype == RegionType.TABLE:
                if region.cells:
                    # Build a proper Word table from extracted cells
                    rows: dict[int, list] = {}
                    for c in region.cells:
                        rows.setdefault(c.row, []).append(c)
                    sorted_row_keys = sorted(rows.keys())
                    max_cols = max(len(rows[i]) for i in sorted_row_keys) if sorted_row_keys else 1

                    tbl = d.add_table(rows=len(sorted_row_keys), cols=max_cols)
                    tbl.style = "Table Grid"
                    for r_i, row_key in enumerate(sorted_row_keys):
                        row_cells = sorted(rows[row_key], key=lambda c: c.col)
                        for c_i, cell in enumerate(row_cells):
                            if c_i < max_cols:
                                tbl.cell(r_i, c_i).text = cell.text.strip()
                    d.add_paragraph("")  # spacing after table
                else:
                    # Fallback: preformatted text
                    para = d.add_paragraph()
                    run = para.add_run(region.text.strip())
                    run.font.name = "Courier New"
                    run.font.size = Pt(9)

            elif rtype == RegionType.CAPTION:
                para = d.add_paragraph(region.text.strip())
                if para.runs:
                    para.runs[0].italic = True
                    para.runs[0].font.size = Pt(9)

            elif rtype == RegionType.FOOTER or rtype == RegionType.PAGE_NUMBER:
                para = d.add_paragraph(region.text.strip())
                if para.runs:
                    para.runs[0].font.size = Pt(8)
                    para.runs[0].font.color.rgb = RGBColor(0x80, 0x80, 0x80)

            else:
                d.add_paragraph(region.text.strip())

        if page_idx < len(doc.pages) - 1:
            d.add_page_break()

    buf = io.BytesIO()
    d.save(buf)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────── JSON → PDF ──────────────────────────────────

def document_to_pdf(doc: "Document") -> bytes:
    """Convert an AksharDrishti Document to a PDF file (bytes).

    Uses ReportLab with a Unicode font that supports Devanagari.
    Falls back to basic Latin rendering if font not available.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    PageBreak, HRFlowable, Table, TableStyle)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Try to register a Unicode font that handles Devanagari
    # Use system fonts on Windows
    font_name = "Helvetica"  # fallback
    for font_path in [
        "C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf",
        "C:/Windows/Fonts/mangal.ttf",
        "C:/Windows/Fonts/Arial Unicode MS.ttf",
        "C:/Windows/Fonts/arialuni.ttf",
    ]:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("DevanagariFont", font_path))
                font_name = "DevanagariFont"
                break
            except Exception:
                continue

    buf = io.BytesIO()
    doc_pdf = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    style_normal = ParagraphStyle(
        "AksharNormal", fontName=font_name, fontSize=11, leading=16,
        spaceAfter=6
    )
    style_title = ParagraphStyle(
        "AksharTitle", fontName=font_name, fontSize=16, leading=20,
        spaceAfter=10, textColor=colors.HexColor("#1a237e"), fontWeight="bold"
    )
    style_heading = ParagraphStyle(
        "AksharHeading", fontName=font_name, fontSize=13, leading=18,
        spaceAfter=8, textColor=colors.HexColor("#0d47a1")
    )
    style_small = ParagraphStyle(
        "AksharSmall", fontName=font_name, fontSize=9, leading=12,
        spaceAfter=4, textColor=colors.grey
    )
    style_caption = ParagraphStyle(
        "AksharCaption", fontName=font_name, fontSize=9, leading=12,
        spaceAfter=4, italic=True
    )

    story = []

    # Document header
    story.append(Paragraph("AksharDrishti — Extracted Document", style_title))
    story.append(Paragraph(f"Source: {Path(doc.source_path).name}", style_small))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 6*mm))

    from .schema import RegionType

    def safe_para(text: str, style) -> Paragraph:
        """Escape HTML-special chars for ReportLab Paragraph."""
        text = (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
        try:
            return Paragraph(text, style)
        except Exception:
            return Paragraph(text.encode("ascii", "replace").decode(), style)

    for page_idx, page in enumerate(doc.pages):
        if len(doc.pages) > 1:
            story.append(safe_para(f"── Page {page_idx + 1} ──", style_heading))
            story.append(Spacer(1, 4*mm))

        for region in page.ordered_regions(include_furniture=True):
            if not region.text.strip():
                continue
            rtype = region.type
            text = region.text.strip()

            if rtype == RegionType.TITLE:
                story.append(safe_para(text, style_heading))
            elif rtype in (RegionType.HEADER,):
                story.append(safe_para(text, style_small))
            elif rtype == RegionType.CAPTION:
                story.append(safe_para(text, style_caption))
            elif rtype in (RegionType.FOOTER, RegionType.PAGE_NUMBER):
                story.append(safe_para(text, style_small))
            elif rtype == RegionType.LIST:
                for line in region.lines:
                    if line.text.strip():
                        story.append(safe_para(f"• {line.text.strip()}", style_normal))
            elif rtype == RegionType.TABLE:
                if region.cells:
                    # Build a ReportLab Table from extracted cells
                    rows_dict: dict[int, list] = {}
                    for c in region.cells:
                        rows_dict.setdefault(c.row, []).append(c)
                    sorted_row_keys = sorted(rows_dict.keys())
                    max_cols = max(len(rows_dict[i]) for i in sorted_row_keys) if sorted_row_keys else 1
                    table_data = []
                    for row_key in sorted_row_keys:
                        row_cells = sorted(rows_dict[row_key], key=lambda c: c.col)
                        row_texts = [c.text.strip() for c in row_cells]
                        # Pad to max_cols
                        while len(row_texts) < max_cols:
                            row_texts.append("")
                        table_data.append(row_texts)
                    if table_data:
                        rl_table = Table(table_data, repeatRows=1)
                        rl_table.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3e8f0")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
                            ("FONTNAME", (0, 0), (-1, 0), font_name),
                            ("FONTNAME", (0, 1), (-1, -1), font_name),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                             [colors.white, colors.HexColor("#f5f5f5")]),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]))
                        story.append(rl_table)
                else:
                    story.append(safe_para(region.text.strip(), style_normal))

            story.append(Spacer(1, 2*mm))

        if page_idx < len(doc.pages) - 1:
            story.append(PageBreak())

    doc_pdf.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────── PDF → DOCX ──────────────────────────────────

def pdf_to_docx_bytes(pdf_bytes: bytes) -> bytes:
    """Extract text from a PDF and write it into a DOCX (bytes)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    from docx import Document as DocxDocument

    d = DocxDocument()
    d.add_heading("PDF → DOCX Conversion", level=0)

    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        text = page.get_text("text")
        if len(pdf_doc) > 1:
            d.add_heading(f"Page {page_num + 1}", level=1)
        for line in text.splitlines():
            if line.strip():
                d.add_paragraph(line.strip())
        if page_num < len(pdf_doc) - 1:
            d.add_page_break()

    buf = io.BytesIO()
    d.save(buf)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────── DOCX → PDF ──────────────────────────────────

def docx_to_pdf_bytes(docx_bytes: bytes) -> bytes:
    """Convert a DOCX to PDF.

    Uses LibreOffice if available, otherwise extracts text and builds PDF
    with ReportLab as a fallback.
    """
    import subprocess

    # Try LibreOffice (best quality)
    lo_paths = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        "soffice",
    ]
    lo_exe = None
    for path in lo_paths:
        try:
            result = subprocess.run([path, "--version"],
                                    capture_output=True, timeout=5)
            if result.returncode == 0:
                lo_exe = path
                break
        except Exception:
            continue

    if lo_exe:
        with tempfile.TemporaryDirectory() as tmpdir:
            docx_path = Path(tmpdir) / "input.docx"
            docx_path.write_bytes(docx_bytes)
            subprocess.run([lo_exe, "--headless", "--convert-to", "pdf",
                            "--outdir", tmpdir, str(docx_path)],
                           capture_output=True, timeout=60)
            pdf_path = Path(tmpdir) / "input.pdf"
            if pdf_path.exists():
                return pdf_path.read_bytes()

    # Fallback: extract DOCX text → ReportLab PDF
    from docx import Document as DocxDocument
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_name = "Helvetica"
    for font_path in [
        "C:/Windows/Fonts/mangal.ttf",
        "C:/Windows/Fonts/arialuni.ttf",
        "C:/Windows/Fonts/Arial Unicode MS.ttf",
    ]:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("DocxFont", font_path))
                font_name = "DocxFont"
                break
            except Exception:
                continue

    d = DocxDocument(io.BytesIO(docx_bytes))
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = font_name
    style.fontSize = 11
    style.leading = 16

    buf = io.BytesIO()
    pdf = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    story = []
    for para in d.paragraphs:
        if para.text.strip():
            text = (para.text.replace("&", "&amp;")
                             .replace("<", "&lt;")
                             .replace(">", "&gt;"))
            try:
                story.append(Paragraph(text, style))
                story.append(Spacer(1, 3*mm))
            except Exception:
                pass
    pdf.build(story)
    buf.seek(0)
    return buf.read()
