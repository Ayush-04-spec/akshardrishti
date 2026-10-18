"""File Converter page — PDF ↔ Word conversion, independent of OCR pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="File Converter — AksharDrishti", page_icon="🔄", layout="centered")

st.title("🔄 File Converter")
st.caption("Convert between PDF and Word formats — no OCR required")

# ── Conversion type ───────────────────────────────────────────────────────────
conv_type = st.radio(
    "Select conversion",
    ["PDF → Word (.docx)", "Word (.docx) → PDF"],
    horizontal=True,
)

st.divider()

# ── PDF → DOCX ────────────────────────────────────────────────────────────────
if conv_type == "PDF → Word (.docx)":
    st.markdown("### Upload a PDF file")
    conv_file = st.file_uploader(
        "Drop your PDF here", type=["pdf"],
        help="Scanned or digital PDF — text will be extracted and placed in the Word document"
    )

    if conv_file is not None:
        st.info(f"File: **{conv_file.name}**  ({conv_file.size/1024:.1f} KB)")

        # Store bytes immediately so button click doesn't lose the file
        if "pdf_bytes_stored" not in st.session_state or \
                st.session_state.get("pdf_filename") != conv_file.name:
            st.session_state["pdf_bytes_stored"] = conv_file.read()
            st.session_state["pdf_filename"] = conv_file.name
            st.session_state.pop("pdf_to_docx_result", None)  # clear old result

        if st.button("▶ Convert PDF → Word", type="primary", use_container_width=True):
            with st.spinner("Extracting text and building Word document..."):
                try:
                    from akshardrishti.export import pdf_to_docx_bytes
                    result = pdf_to_docx_bytes(st.session_state["pdf_bytes_stored"])
                    st.session_state["pdf_to_docx_result"] = result
                    st.session_state["pdf_to_docx_name"] = (
                        Path(conv_file.name).stem + ".docx"
                    )
                except ImportError:
                    st.error("PyMuPDF not installed. Run: `pip install pymupdf`")
                except Exception as exc:
                    st.error(f"Conversion failed: {exc}")

        if "pdf_to_docx_result" in st.session_state:
            st.success("✅ Conversion complete!")
            st.download_button(
                label=f"⬇️ Download {st.session_state['pdf_to_docx_name']}",
                data=st.session_state["pdf_to_docx_result"],
                file_name=st.session_state["pdf_to_docx_name"],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

# ── DOCX → PDF ────────────────────────────────────────────────────────────────
else:
    st.markdown("### Upload a Word (.docx) file")
    conv_file = st.file_uploader(
        "Drop your Word document here", type=["docx"],
        help="Word documents (.docx) — converted to PDF via LibreOffice or ReportLab fallback"
    )

    if conv_file is not None:
        st.info(f"File: **{conv_file.name}**  ({conv_file.size/1024:.1f} KB)")

        if "docx_bytes_stored" not in st.session_state or \
                st.session_state.get("docx_filename") != conv_file.name:
            st.session_state["docx_bytes_stored"] = conv_file.read()
            st.session_state["docx_filename"] = conv_file.name
            st.session_state.pop("docx_to_pdf_result", None)

        if st.button("▶ Convert Word → PDF", type="primary", use_container_width=True):
            with st.spinner("Building PDF..."):
                try:
                    from akshardrishti.export import docx_to_pdf_bytes
                    result = docx_to_pdf_bytes(st.session_state["docx_bytes_stored"])
                    st.session_state["docx_to_pdf_result"] = result
                    st.session_state["docx_to_pdf_name"] = (
                        Path(conv_file.name).stem + ".pdf"
                    )
                except Exception as exc:
                    st.error(f"Conversion failed: {exc}")

        if "docx_to_pdf_result" in st.session_state:
            st.success("✅ Conversion complete!")
            st.download_button(
                label=f"⬇️ Download {st.session_state['docx_to_pdf_name']}",
                data=st.session_state["docx_to_pdf_result"],
                file_name=st.session_state["docx_to_pdf_name"],
                mime="application/pdf",
                use_container_width=True,
            )

st.divider()
st.caption("AksharDrishti v0.1.0 — K. K. Wagh Institute of Engineering")
