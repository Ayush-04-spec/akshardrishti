#!/usr/bin/env python3
"""AksharDrishti demo UI.

    streamlit run app/streamlit_app.py

On Colab/Kaggle, tunnel it:
    !pip install pyngrok && streamlit run app/streamlit_app.py &
    from pyngrok import ngrok; print(ngrok.connect(8501))
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akshardrishti.config import Config  # noqa: E402
from akshardrishti.pipeline import AksharDrishtiPipeline, visualize_regions  # noqa: E402
from akshardrishti.preprocess.enhance import preprocess_page  # noqa: E402

st.set_page_config(page_title="AksharDrishti", page_icon="📄", layout="wide")

REGION_COLORS = {
    "title": "#FF0000", "text": "#00C800", "list": "#FFA500", "table": "#0000FF",
    "figure": "#FF00FF", "caption": "#808000", "header": "#808080", "footer": "#808080",
    "page_number": "#646464", "stamp_seal": "#FFD700", "signature": "#FF8000",
}


@st.cache_resource(show_spinner=False)
def get_pipeline(config_path: str, overrides_json: str) -> AksharDrishtiPipeline:
    cfg = Config.load(config_path)
    for key, value in json.loads(overrides_json).items():
        cfg.set(key, value)
    return AksharDrishtiPipeline(cfg)


def main() -> None:
    st.title("AksharDrishti")
    st.caption("Layout-Aware Transformer for Multilingual Indian OCR — Hindi · Marathi · English")

    # ---------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Configuration")
        config_path = st.text_input("Config file", "configs/pipeline.yaml")

        st.subheader("Pipeline stages")
        use_layout = st.checkbox("Layout detection", True,
                                 help="Off = full-page OCR, i.e. the 'no layout' ablation")
        use_sahi = st.checkbox("SAHI sliced inference", True, help="Helps on dense pages, ~4x slower")
        use_script_id = st.checkbox("Script routing", True)
        use_postprocess = st.checkbox("Devanagari post-processing", True)

        st.subheader("Backends")
        deva_backend = st.selectbox("Devanagari", ["crnn_mozhi", "paddle_vl", "surya", "bhashini_api", "tesseract"])
        latin_backend = st.selectbox("Latin", ["trocr_printed", "paddle_vl", "tesseract"])

        st.subheader("Detection")
        conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)

        overrides = {
            "layout.enabled": use_layout,
            "layout.sahi.enabled": use_sahi,
            "layout.conf": conf,
            "script_id.enabled": use_script_id,
            "postprocess.repair_shirorekha": use_postprocess,
            "recognize.deva_backend": deva_backend,
            "recognize.latin_backend": latin_backend,
        }

        st.divider()
        st.caption("Region colours")
        for name, colour in REGION_COLORS.items():
            st.markdown(
                f"<span style='color:{colour}'>■</span> {name}",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------ input
    uploaded = st.file_uploader(
        "Upload a document", type=["jpg", "jpeg", "png", "tif", "tiff", "pdf"],
        help="Scanned or photographed Indian government / legal documents work best",
    )
    if uploaded is None:
        st.info("Upload a document to begin. Try a Maharashtra GR, a gazette notice, or a certificate.")
        return

    suffix = Path(uploaded.name).suffix
    file_key = f"{uploaded.name}_{uploaded.size}"

    # ── Run pipeline (only when button clicked, cache result in session_state) ─
    if st.button("Run AksharDrishti", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        try:
            pipeline = get_pipeline(config_path, json.dumps(overrides))
        except Exception as exc:
            st.error(f"Could not build the pipeline: {exc}")
            return

        progress = st.progress(0.0, "Loading…")
        started = time.perf_counter()
        try:
            progress.progress(0.15, "Loading document…")
            images = pipeline.load_images(tmp_path)
            progress.progress(0.35, f"Processing {len(images)} page(s)…")
            doc = pipeline.process(tmp_path)
            progress.progress(1.0, "Done")
            elapsed = time.perf_counter() - started
            # Cache everything in session state
            st.session_state["result_doc"]     = doc
            st.session_state["result_images"]  = images
            st.session_state["result_elapsed"] = elapsed
            st.session_state["result_key"]     = file_key
        except FileNotFoundError as exc:
            progress.empty()
            st.error(f"Missing model weights: {exc}")
            return
        except Exception as exc:
            progress.empty()
            st.error(f"Processing failed: {exc}")
            st.exception(exc)
            return
        finally:
            tmp_path.unlink(missing_ok=True)
        progress.empty()
    else:
        # No button clicked yet — show preview
        if st.session_state.get("result_key") != file_key:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
            return

    # ── Show results (from session_state so sliders/buttons don't rerun) ──────
    if st.session_state.get("result_key") != file_key:
        return

    doc     = st.session_state["result_doc"]
    images  = st.session_state["result_images"]
    elapsed = st.session_state["result_elapsed"]

    # ── Metrics ────────────────────────────────────────────────────────────────
    total_regions = sum(len(p.regions) for p in doc.pages)
    total_chars = len(doc.full_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages", len(doc.pages))
    c2.metric("Regions", total_regions)
    c3.metric("Characters", f"{total_chars:,}")
    c4.metric("Time", f"{elapsed:.1f}s")

    # ------------------------------------------------------------ output
    for page_idx, page in enumerate(doc.pages):
        if len(doc.pages) > 1:
            st.subheader(f"Page {page_idx + 1}")

        pre = preprocess_page(images[page_idx], get_pipeline(config_path, json.dumps(overrides)).cfg.get("preprocess", {}))
        overlay = visualize_regions(pre.image, page)

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Detected layout**")
            # Resize to fixed height so entire page fits without scrolling
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            oh, ow = overlay_rgb.shape[:2]
            target_h = st.select_slider(
                "Image height (px)", 
                options=[400, 500, 600, 700, 800, 900, 1000, "Full"],
                value=700,
                key=f"zoom_{page_idx}"
            )
            if target_h != "Full":
                scale = int(target_h) / max(oh, 1)
                if scale < 1.0:
                    new_w = max(1, int(ow * scale))
                    overlay_rgb = cv2.resize(overlay_rgb, (new_w, int(target_h)),
                                             interpolation=cv2.INTER_AREA)
            st.image(overlay_rgb, use_container_width=True)
            st.caption("Numbers show reading order")

        with right:
            st.markdown("**Extracted content**")
            tabs = st.tabs(["Text", "Markdown", "Regions", "JSON"])

            with tabs[0]:
                st.text_area("", page.full_text(), height=420, key=f"text_{page_idx}",
                             label_visibility="collapsed")
            with tabs[1]:
                st.markdown(page.to_markdown())
            with tabs[2]:
                st.dataframe(
                    [
                        {
                            "order": r.reading_order,
                            "type": r.type.value,
                            "script": r.script,
                            "det_conf": round(r.detection_confidence, 3),
                            "ocr_conf": round(r.recognition_confidence, 3) if r.recognition_confidence else None,
                            "chars": len(r.text),
                            "preview": (r.text[:60] + "…") if len(r.text) > 60 else r.text,
                        }
                        for r in page.ordered_regions()
                    ],
                    use_container_width=True, hide_index=True,
                )
            with tabs[3]:
                st.json(page.model_dump(mode="json"), expanded=False)

    # --------------------------------------------------------- downloads
    st.divider()
    st.markdown("### 📥 Download Extracted Output")
    stem = Path(uploaded.name).stem

    # Row 1: original formats
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "📄 Download JSON",
        json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2),
        f"{stem}_akshardrishti.json", "application/json", use_container_width=True,
    )
    d2.download_button("📝 Download Markdown", doc.to_markdown(),
                       f"{stem}.md", "text/markdown", use_container_width=True)
    d3.download_button("📃 Download Text", doc.full_text(),
                       f"{stem}.txt", "text/plain", use_container_width=True)

    # Row 2: DOCX and PDF export
    st.markdown("#### Export as Document")
    e1, e2 = st.columns(2)

    with e1:
        if st.button("📘 Export as Word (.docx)", use_container_width=True):
            with st.spinner("Generating Word document..."):
                try:
                    from akshardrishti.export import document_to_docx
                    docx_bytes = document_to_docx(doc)
                    st.session_state["export_docx"] = docx_bytes
                    st.session_state["export_stem"] = stem
                except Exception as exc:
                    st.error(f"DOCX export failed: {exc}")

        if "export_docx" in st.session_state and st.session_state.get("export_stem") == stem:
            st.download_button(
                "⬇️ Download .docx",
                st.session_state["export_docx"],
                f"{stem}_akshardrishti.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="docx_dl",
            )

    with e2:
        if st.button("📕 Export as PDF", use_container_width=True):
            with st.spinner("Generating PDF..."):
                try:
                    from akshardrishti.export import document_to_pdf
                    pdf_bytes = document_to_pdf(doc)
                    st.session_state["export_pdf"] = pdf_bytes
                    st.session_state["export_stem"] = stem
                except Exception as exc:
                    st.error(f"PDF export failed: {exc}")

        if "export_pdf" in st.session_state and st.session_state.get("export_stem") == stem:
            st.download_button(
                "⬇️ Download .pdf",
                st.session_state["export_pdf"],
                f"{stem}_akshardrishti.pdf",
                "application/pdf",
                use_container_width=True,
                key="pdf_dl",
            )

    # ─────────────── File Converter ───────────────────────────────────
    st.divider()
    st.info("🔄 Need to convert PDF ↔ Word without OCR? Use the **File Converter** page in the sidebar.")

    with st.expander("Timings"):
        st.json(doc.timings_s)


if __name__ == "__main__":
    main()
