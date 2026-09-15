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
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    if not st.button("Run AksharDrishti", type="primary"):
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
        return

    # ---------------------------------------------------------- process
    try:
        pipeline = get_pipeline(config_path, json.dumps(overrides))
    except Exception as exc:  # noqa: BLE001
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
    except FileNotFoundError as exc:
        progress.empty()
        st.error(f"Missing model weights: {exc}")
        st.info("Train the models first, or switch to a pretrained backend (paddle_vl / tesseract) in the sidebar.")
        return
    except Exception as exc:  # noqa: BLE001
        progress.empty()
        st.error(f"Processing failed: {exc}")
        st.exception(exc)
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started

    # ----------------------------------------------------------- metrics
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

        pre = preprocess_page(images[page_idx], pipeline.cfg.get("preprocess", {}))
        overlay = visualize_regions(pre.image, page)

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Detected layout**")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
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
    stem = Path(uploaded.name).stem
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download JSON",
        json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2),
        f"{stem}_akshardrishti.json", "application/json", use_container_width=True,
    )
    d2.download_button("Download Markdown", doc.to_markdown(), f"{stem}.md", "text/markdown",
                       use_container_width=True)
    d3.download_button("Download text", doc.full_text(), f"{stem}.txt", "text/plain",
                       use_container_width=True)

    with st.expander("Timings"):
        st.json(doc.timings_s)


if __name__ == "__main__":
    main()
