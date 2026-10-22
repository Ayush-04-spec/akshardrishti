"""File Converter page — PDF ↔ Word conversion, independent of OCR pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(
    page_title="File Converter — AksharDrishti",
    page_icon="🔄",
    layout="centered",
)

st.title("🔄 File Converter")
st.caption("Convert between PDF and Word formats — no OCR required")

# ── Initialize session state keys ─────────────────────────────────────────────
for key in ["uploaded_bytes", "uploaded_name", "converted_bytes",
            "converted_name", "converted_mime", "conv_mode", "error_msg"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Conversion type ───────────────────────────────────────────────────────────
mode = st.radio(
    "Select conversion",
    ["PDF → Word (.docx)", "Word (.docx) → PDF"],
    horizontal=True,
    key="conv_mode_radio",
)

# Clear stored result when mode changes
if st.session_state["conv_mode"] != mode:
    st.session_state["conv_mode"]       = mode
    st.session_state["uploaded_bytes"]  = None
    st.session_state["uploaded_name"]   = None
    st.session_state["converted_bytes"] = None
    st.session_state["converted_name"]  = None
    st.session_state["converted_mime"]  = None
    st.session_state["error_msg"]       = None

st.divider()

# ── File uploader ─────────────────────────────────────────────────────────────
accept = ["pdf"] if mode == "PDF → Word (.docx)" else ["docx"]
label  = "Upload a PDF file" if mode == "PDF → Word (.docx)" else "Upload a Word (.docx) file"

uploaded = st.file_uploader(label, type=accept, key=f"uploader_{mode}")

# Store bytes the moment a new file arrives
if uploaded is not None:
    if st.session_state["uploaded_name"] != uploaded.name:
        st.session_state["uploaded_bytes"]  = uploaded.read()
        st.session_state["uploaded_name"]   = uploaded.name
        st.session_state["converted_bytes"] = None   # reset old result
        st.session_state["converted_name"]  = None
        st.session_state["converted_mime"]  = None
        st.session_state["error_msg"]       = None

# ── Show file info + Convert button ──────────────────────────────────────────
if st.session_state["uploaded_bytes"] is not None:
    size_kb = len(st.session_state["uploaded_bytes"]) / 1024
    st.info(f"📎 **{st.session_state['uploaded_name']}** — {size_kb:.1f} KB ready")

    btn_label = ("▶ Convert PDF → Word"
                 if mode == "PDF → Word (.docx)"
                 else "▶ Convert Word → PDF")

    if st.button(btn_label, type="primary", use_container_width=True):
        st.session_state["error_msg"]       = None
        st.session_state["converted_bytes"] = None

        with st.spinner("Converting, please wait..."):
            try:
                if mode == "PDF → Word (.docx)":
                    from akshardrishti.export import pdf_to_docx_bytes
                    result = pdf_to_docx_bytes(st.session_state["uploaded_bytes"])
                    out_name = Path(st.session_state["uploaded_name"]).stem + ".docx"
                    out_mime = ("application/vnd.openxmlformats-officedocument"
                                ".wordprocessingml.document")
                else:
                    from akshardrishti.export import docx_to_pdf_bytes
                    result = docx_to_pdf_bytes(st.session_state["uploaded_bytes"])
                    out_name = Path(st.session_state["uploaded_name"]).stem + ".pdf"
                    out_mime = "application/pdf"

                st.session_state["converted_bytes"] = result
                st.session_state["converted_name"]  = out_name
                st.session_state["converted_mime"]  = out_mime

            except ImportError as e:
                st.session_state["error_msg"] = (
                    f"Missing dependency: {e}. "
                    "For PDF→Word run: pip install pymupdf"
                )
            except Exception as e:
                st.session_state["error_msg"] = f"Conversion failed: {e}"

# ── Show result or error ──────────────────────────────────────────────────────
if st.session_state["error_msg"]:
    st.error(st.session_state["error_msg"])

if st.session_state["converted_bytes"] is not None:
    st.success("✅ Conversion complete — click below to download")
    st.download_button(
        label=f"⬇️ Download  {st.session_state['converted_name']}",
        data=st.session_state["converted_bytes"],
        file_name=st.session_state["converted_name"],
        mime=st.session_state["converted_mime"],
        use_container_width=True,
    )

st.divider()
st.caption("AksharDrishti v0.1.0 — K. K. Wagh Institute of Engineering")
