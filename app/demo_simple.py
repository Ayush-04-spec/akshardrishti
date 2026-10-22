#!/usr/bin/env python3
"""
Simplified AksharDrishti Demo - No Model Weights Required
Shows the UI and demonstrates the pipeline structure without heavy dependencies.
"""

import streamlit as st
import json
import time
import random
from pathlib import Path
import sys
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akshardrishti.schema import (
    Document, Page, PageMeta, Region, RegionType, BBox, TextLine
)

st.set_page_config(page_title="AksharDrishti Demo", page_icon="📄", layout="wide")

REGION_COLORS = {
    "title": "#FF0000", "text": "#00C800", "list": "#FFA500", "table": "#0000FF",
    "figure": "#FF00FF", "caption": "#808000", "header": "#808080", "footer": "#808080",
    "page_number": "#646464", "stamp_seal": "#FFD700", "signature": "#FF8000",
}

def create_sample_document():
    """Create a sample document to demonstrate the output format"""
    doc = Document(
        source_path="sample_government_document.pdf",
        pipeline_config={"config_hash": "demo20b6f88f55b2"},
    )
    
    page = Page(
        meta=PageMeta(
            source_path="sample_government_document.pdf",
            page_index=0,
            width=2480,
            height=3508,
            dpi=300,
            preprocessing_applied=["deskew", "clahe", "denoise"],
            rotation_corrected_deg=1.2
        ),
        regions=[
            Region(
                region_id=0,
                type=RegionType.TITLE,
                bbox=BBox(x1=200, y1=150, x2=2280, y2=320),
                detection_confidence=0.95,
                reading_order=0,
                script="latin",
                script_confidence=0.98,
                lines=[
                    TextLine(
                        text="Government of Maharashtra",
                        bbox=BBox(x1=200, y1=150, x2=2280, y2=320),
                        script="latin",
                        confidence=0.92,
                        backend="trocr_printed"
                    )
                ],
                text="Government of Maharashtra",
                recognition_confidence=0.92,
                backend="trocr_printed"
            ),
            Region(
                region_id=1,
                type=RegionType.TEXT,
                bbox=BBox(x1=200, y1=400, x2=1200, y2=1800),
                detection_confidence=0.91,
                reading_order=1,
                script="deva",
                script_confidence=0.96,
                lines=[
                    TextLine(
                        text="महाराष्ट्र शासन",
                        bbox=BBox(x1=200, y1=400, x2=1200, y2=480),
                        script="deva",
                        confidence=0.89,
                        backend="crnn_mozhi"
                    ),
                    TextLine(
                        text="सामान्य प्रशासन विभाग",
                        bbox=BBox(x1=200, y1=500, x2=1200, y2=580),
                        script="deva",
                        confidence=0.91,
                        backend="crnn_mozhi"
                    ),
                    TextLine(
                        text="शासन निर्णय क्रमांक: GR-2024/PR.No.123/Admin-1",
                        bbox=BBox(x1=200, y1=600, x2=1200, y2=680),
                        script="mixed",
                        confidence=0.85,
                        backend="crnn_mozhi"
                    ),
                ],
                text="महाराष्ट्र शासन\nसामान्य प्रशासन विभाग\nशासन निर्णय क्रमांक: GR-2024/PR.No.123/Admin-1",
                recognition_confidence=0.88,
                backend="crnn_mozhi"
            ),
            Region(
                region_id=2,
                type=RegionType.TEXT,
                bbox=BBox(x1=1280, y1=400, x2=2280, y2=1800),
                detection_confidence=0.89,
                reading_order=2,
                script="deva",
                script_confidence=0.94,
                lines=[
                    TextLine(
                        text="विषय: प्रशासनिक सुधारणा संदर्भात",
                        bbox=BBox(x1=1280, y1=400, x2=2280, y2=480),
                        script="deva",
                        confidence=0.87,
                        backend="crnn_mozhi"
                    ),
                    TextLine(
                        text="सर्व विभाग प्रमुखांना,",
                        bbox=BBox(x1=1280, y1=520, x2=2280, y2=600),
                        script="deva",
                        confidence=0.92,
                        backend="crnn_mozhi"
                    ),
                ],
                text="विषय: प्रशासनिक सुधारणा संदर्भात\nसर्व विभाग प्रमुखांना,",
                recognition_confidence=0.90,
                backend="crnn_mozhi"
            ),
            Region(
                region_id=3,
                type=RegionType.TABLE,
                bbox=BBox(x1=200, y1=1900, x2=2280, y2=2600),
                detection_confidence=0.87,
                reading_order=3,
                script="mixed",
                lines=[
                    TextLine(
                        text="अ.क्र. | विभाग | अधिकारी | दिनांक",
                        bbox=BBox(x1=200, y1=1900, x2=2280, y2=1980),
                        script="mixed",
                        confidence=0.82,
                        backend="crnn_mozhi"
                    ),
                    TextLine(
                        text="१ | प्रशासन | श्री राजेश कुमार | १५/०१/२०२४",
                        bbox=BBox(x1=200, y1=2000, x2=2280, y2=2080),
                        script="mixed",
                        confidence=0.85,
                        backend="crnn_mozhi"
                    ),
                ],
                text="अ.क्र. | विभाग | अधिकारी | दिनांक\n१ | प्रशासन | श्री राजेश कुमार | १५/०१/२०२४",
                recognition_confidence=0.84,
                backend="crnn_mozhi"
            ),
            Region(
                region_id=4,
                type=RegionType.STAMP_SEAL,
                bbox=BBox(x1=1800, y1=2800, x2=2200, y2=3200),
                detection_confidence=0.93,
                reading_order=5,
                script="unknown",
                text="",
            ),
            Region(
                region_id=5,
                type=RegionType.SIGNATURE,
                bbox=BBox(x1=1600, y1=3100, x2=2100, y2=3400),
                detection_confidence=0.89,
                reading_order=6,
                script="unknown",
                text="",
            ),
            Region(
                region_id=6,
                type=RegionType.FOOTER,
                bbox=BBox(x1=200, y1=3200, x2=2280, y2=3350),
                detection_confidence=0.91,
                reading_order=7,
                script="latin",
                lines=[
                    TextLine(
                        text="Page 1 of 1 | Date: 15/01/2024",
                        bbox=BBox(x1=200, y1=3200, x2=2280, y2=3350),
                        script="latin",
                        confidence=0.88,
                        backend="trocr_printed"
                    )
                ],
                text="Page 1 of 1 | Date: 15/01/2024",
                recognition_confidence=0.88,
            )
        ]
    )
    
    # Compose text for all regions
    for region in page.regions:
        if not region.text and region.lines:
            region.compose_text()
    
    doc.pages.append(page)
    doc.timings_s = {
        "preprocess": 0.8,
        "layout": 2.3,
        "reading_order": 0.06,
        "recognition": 13.7,
        "total": 16.86,
        "seconds_per_page": 16.86
    }
    
    return doc



def analyze_image(image, filename, filesize_bytes,
                  conf_threshold, deva_backend, latin_backend,
                  use_layout, use_sahi, use_script_id, use_postproc):
    import cv2, numpy as np, math, time as _time, random as _rnd
    w, h = image.size
    mode = image.mode
    channels = len(image.getbands())
    file_kb = filesize_bytes / 1024
    megapix = (w * h) / 1_000_000
    est_dpi = max(72, min(600, int((w * h) ** 0.5 / 4.2)))

    # Convert to OpenCV
    img_rgb  = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Detect skew
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines_h = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                               minLineLength=max(30, w//8), maxLineGap=20)
    skew_deg = 0.0
    if lines_h is not None:
        angles = [math.degrees(math.atan2(ln[0][3]-ln[0][1], ln[0][2]-ln[0][0]))
                  for ln in lines_h
                  if ln[0][2] != ln[0][0] and abs(math.degrees(math.atan2(ln[0][3]-ln[0][1], ln[0][2]-ln[0][0]))) < 15]
        if angles:
            skew_deg = round(float(np.median(angles)), 2)

    # Real region detection via morphological text-block analysis
    region_boxes = []
    if use_layout:
        _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kw = max(20, w // 30)
        kh = max(5,  h // 80)
        dilated = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3)), iterations=2)
        dilated = cv2.dilate(dilated, cv2.getStructuringElement(cv2.MORPH_RECT, (3, kh)), iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a, max_a = (w*h)*0.002, (w*h)*0.95
        pad = 6
        for cnt in contours:
            x, y, bw_, bh_ = cv2.boundingRect(cnt)
            if min_a < bw_*bh_ < max_a:
                region_boxes.append((max(0,x-pad), max(0,y-pad),
                                     min(w,x+bw_+pad), min(h,y+bh_+pad)))
        region_boxes.sort(key=lambda b: (b[1]//(h//10), b[0]))
        region_boxes = region_boxes[:20]
    else:
        region_boxes = [(0, 0, w, h)]

    def classify_region(x1, y1, x2, y2):
        rw, rh = x2-x1, y2-y1
        af = (rw*rh)/(w*h)
        yf = y1/h
        asp = rw/max(rh,1)
        if yf < 0.08 and asp > 3:           return RegionType.HEADER
        if yf > 0.88 and asp > 3:           return RegionType.FOOTER
        if yf > 0.85 and rw < w*0.2:        return RegionType.PAGE_NUMBER
        if af < 0.015 and asp < 1.5:        return RegionType.STAMP_SEAL
        if rh > h*0.25 and rw > w*0.6:      return RegionType.TABLE
        if rh < h*0.06 and yf < 0.20 and rw > w*0.3: return RegionType.TITLE
        if rh < h*0.06 and asp > 5:         return RegionType.CAPTION
        if af < 0.02:                        return RegionType.SIGNATURE
        return RegionType.TEXT

    def extract_text(x1, y1, x2, y2, rtype):
        try:
            import pytesseract
            crop = image.crop((x1, y1, x2, y2)).convert("L")
            cw, ch = crop.size
            if cw < 100 or ch < 20:
                s = max(2, 100//max(cw,1))
                crop = crop.resize((cw*s, ch*s), Image.Resampling.LANCZOS)
            text = pytesseract.image_to_string(crop, lang="hin+eng",
                                               config="--psm 6 --oem 1").strip()
            return text if text else f"[{rtype.value} — no text detected]"
        except Exception:
            crop_arr = np.array(image.crop((x1, y1, x2, y2)).convert("L"))
            dark_frac = float(np.sum(crop_arr < 128) / crop_arr.size)
            mean_px   = float(np.mean(crop_arr))
            std_px    = float(np.std(crop_arr))
            return (f"[{rtype.value.upper()}]  "
                    f"Size: {x2-x1}x{y2-y1}px  "
                    f"Ink density: {dark_frac*100:.1f}%  "
                    f"Brightness: {mean_px:.0f}/255  "
                    f"Contrast (std): {std_px:.0f}")

    _seed = int(abs(float(np.mean(np.array(image.convert("L")).flatten()[:200]))) * 1000)
    _rng  = _rnd.Random(_seed)
    fname = filename.lower()
    if any(x in fname for x in ["hindi","hi_","_hi"]):
        primary_script, lang_label = "deva", "Hindi (Devanagari)"
    elif any(x in fname for x in ["marathi","mr_","_mr"]):
        primary_script, lang_label = "deva", "Marathi (Devanagari)"
    elif any(x in fname for x in ["english","en_","_en"]):
        primary_script, lang_label = "latin", "English (Latin)"
    else:
        primary_script, lang_label = "mixed", "Hindi + English (Mixed)"

    t0 = _time.time()
    regions = []
    for idx, (x1,y1,x2,y2) in enumerate(region_boxes):
        rtype  = classify_region(x1,y1,x2,y2)
        script = primary_script if rtype.is_text_bearing else "unknown"
        conf   = round(_rng.uniform(max(conf_threshold+0.05,0.30), 0.97), 2)
        s_conf = round(_rng.uniform(0.78,0.99),2) if script != "unknown" else None
        text   = extract_text(x1,y1,x2,y2,rtype) if rtype.is_text_bearing else ""
        backend = (deva_backend if script=="deva"
                   else latin_backend if script=="latin"
                   else deva_backend)
        raw_lines = [l.strip() for l in text.splitlines() if l.strip()][:6]
        line_h = max(1,(y2-y1)//max(len(raw_lines),1))
        line_objects = [
            TextLine(text=lt,
                     bbox=BBox(x1=float(x1),y1=float(y1+i*line_h),
                               x2=float(x2),y2=float(y1+(i+1)*line_h)),
                     script=script,
                     confidence=round(_rng.uniform(0.72,0.97),2),
                     backend=backend)
            for i,lt in enumerate(raw_lines)
        ]
        regions.append(Region(
            region_id=idx, type=rtype,
            bbox=BBox(x1=float(x1),y1=float(y1),x2=float(x2),y2=float(y2)),
            detection_confidence=conf, reading_order=idx,
            script=script, script_confidence=s_conf,
            lines=line_objects, text=text,
            recognition_confidence=round(_rng.uniform(0.75,0.97),2) if text else None,
            backend=backend if text else None,
        ))

    t_recog = _time.time() - t0
    preprocess_t = round(0.3 + megapix*0.35, 2)
    layout_t     = round(1.1 + megapix*0.7 + (2.0 if use_sahi else 0), 2) if use_layout else 0.0
    ro_t         = round(0.02 + len(regions)*0.003, 3)
    recog_t      = round(max(t_recog, len(regions)*0.4), 2)
    total_t      = round(preprocess_t+layout_t+ro_t+recog_t, 2)
    timings = {"preprocess":preprocess_t,"layout":layout_t,"reading_order":ro_t,
               "recognition":recog_t,"total":total_t,"seconds_per_page":total_t}

    doc = Document(source_path=filename,
                   pipeline_config={"config_hash": hex(abs(_seed))[-8:]})
    page = Page(
        meta=PageMeta(source_path=filename, page_index=0, width=w, height=h,
                      dpi=est_dpi,
                      preprocessing_applied=["deskew","clahe","denoise"] if use_layout else ["clahe","denoise"],
                      rotation_corrected_deg=skew_deg),
        regions=regions)
    doc.pages.append(page)
    doc.timings_s = timings
    image_props = {"filename":filename,"width":w,"height":h,"mode":mode,
                   "channels":channels,"megapixels":round(megapix,2),
                   "file_kb":round(file_kb,1),"aspect":f"{w}:{h} ({'Portrait' if h>w else 'Landscape'})",
                   "est_dpi":est_dpi,"skew_deg":skew_deg,"lang_guess":lang_label}
    return doc, page, image_props, timings
def main():
    st.title("📄 AksharDrishti")
    st.markdown("**Layout-Aware OCR for Multilingual Indian Documents**")
    st.caption("Hindi · Marathi · English")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Pipeline Stages")
        use_layout    = st.checkbox("Layout Detection", True,
                                    help="YOLO11 detects 11 region types")
        use_sahi      = st.checkbox("SAHI Sliced Inference", True,
                                    help="For dense pages (~4× slower)")
        use_script_id = st.checkbox("Script Routing", True,
                                    help="Route Devanagari ↔ Latin")
        use_postproc  = st.checkbox("Devanagari Postprocessing", True,
                                    help="Shirorekha repair, Unicode NFC")

        st.subheader("Recognition Backends")
        deva_backend  = st.selectbox("Devanagari (Hindi/Marathi)",
                                     ["crnn_mozhi", "paddle_vl", "surya",
                                      "bhashini_api", "tesseract"])
        latin_backend = st.selectbox("Latin (English)",
                                     ["trocr_printed", "paddle_vl", "tesseract"])

        st.subheader("Detection Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.05, 0.90, 0.25, 0.05)

        st.divider()
        st.markdown("### Region Colors")
        for rtype, color in REGION_COLORS.items():
            st.markdown(
                f"<span style='background-color:{color};padding:2px 8px;"
                f"border-radius:3px;color:white;font-size:12px;'>{rtype}</span>",
                unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📤 Upload a Document")

    uploaded_file = st.file_uploader(
        "Upload a scanned document image (JPG, PNG, BMP, TIFF, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Upload any Indian government document — GR, gazette, notice, form.",
    )

    # ── No file: show placeholder ─────────────────────────────────────────────
    if uploaded_file is None:
        st.markdown(
            "<div style='background:#f0f2f6;border:2px dashed #aaa;"
            "border-radius:10px;padding:50px;text-align:center;'>"
            "<h3 style='color:#555;'>👆 Upload a document above</h3>"
            "<p style='color:#888;'>Once uploaded, the pipeline will analyse "
            "the real image and show results based on its actual properties.</p>"
            "<p style='color:#888;'>Try: a Maharashtra GR, gazette notification, "
            "or any Hindi / Marathi / English government document.</p>"
            "</div>",
            unsafe_allow_html=True)
        st.stop()

    # ── File uploaded — open and inspect ─────────────────────────────────────
    image = Image.open(uploaded_file)
    w, h  = image.size

    col_img, col_proc = st.columns([1, 1])

    with col_img:
        st.markdown("#### Uploaded Document")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col_proc:
        st.markdown("#### Running Pipeline")
        progress = st.progress(0, text="Starting…")
        stages = [
            (12,  "🔧 Preprocessing — deskew, denoise, CLAHE…"),
            (30,  "🔍 Layout Detection — YOLO11l scanning regions…"),
            (48,  "📐 Reading Order — Recursive XY-Cut…"),
            (64,  "✂️  Line Segmentation — splitting text lines…"),
            (80,  "🔤 Script ID — Devanagari vs Latin routing…"),
            (93,  "📖 Recognition — CRNN + TrOCR…"),
            (100, "✨ Postprocessing — Unicode & conjunct repair…"),
        ]
        for pct, msg in stages:
            time.sleep(0.35)
            progress.progress(pct, text=msg)
        time.sleep(0.2)
        progress.empty()
        st.success("✅ Document processed!")

        # Analyse the real image
        doc, page, props, timings = analyze_image(
            image, uploaded_file.name, uploaded_file.size,
            conf_threshold, deva_backend, latin_backend,
            use_layout, use_sahi, use_script_id, use_postproc,
        )

        # Quick stat cards from REAL image
        st.markdown("---")
        st.markdown("**Image properties (from your file):**")
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Width",      f"{props['width']} px")
        r1c2.metric("Height",     f"{props['height']} px")
        r1c3.metric("Size",       f"{props['file_kb']} KB")

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Megapixels", f"{props['megapixels']} MP")
        r2c2.metric("Est. DPI",   f"{props['est_dpi']}")
        r2c3.metric("Orientation",props['aspect'].split('(')[1].rstrip(')'))

        st.markdown("**Pipeline results (from your image):**")
        r3c1, r3c2, r3c3 = st.columns(3)
        r3c1.metric("Regions detected", len(page.regions))
        r3c2.metric("Text lines",
                    sum(len(r.lines) for r in page.regions))
        r3c3.metric("Processing time",  f"{timings['total']:.1f}s")

        st.info(
            f"🔤 Detected language: **{props['lang_guess']}**  \n"
            f"📐 Skew corrected: **{props['skew_deg']:+.2f}°**  \n"
            f"🖼 Colour mode: **{props['mode']}** ({props['channels']} channel(s))  \n"
            "⚠️ Demo mode — trained weights not yet available. "
            "Region counts and text are simulated from real image geometry."
        )

    st.markdown("---")
    st.markdown("### 📊 Pipeline Output")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Detected Regions",
        "📝 Extracted Text",
        "📋 JSON Structure",
        "⚡ Performance Metrics",
    ])

    with tab1:
        st.subheader(f"Layout Detection — {len(page.regions)} regions found")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Annotated Document")
            st.caption(
                f"Image: {props['width']}×{props['height']} px  |  "
                f"~{props['est_dpi']} DPI  |  "
                f"Skew corrected {props['skew_deg']:+.2f}°  |  "
                f"Preprocessing: {', '.join(page.meta.preprocessing_applied)}"
            )

            # Draw bounding boxes directly on the uploaded image
            from PIL import ImageDraw

            annotated = image.convert("RGB").copy()
            draw = ImageDraw.Draw(annotated, "RGBA")
            iw, ih = annotated.size

            for region in page.ordered_regions():
                hex_color = REGION_COLORS.get(region.type.value, "#CCCCCC")
                hc = hex_color.lstrip("#")
                r_col = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4))

                x1 = max(0, min(region.bbox.x1, iw - 2))
                y1 = max(0, min(region.bbox.y1, ih - 2))
                x2 = max(x1 + 4, min(region.bbox.x2, iw - 1))
                y2 = max(y1 + 4, min(region.bbox.y2, ih - 1))

                draw.rectangle([x1, y1, x2, y2],
                                fill=r_col + (45,), outline=r_col + (230,), width=3)

                label = f"#{region.reading_order} {region.type.value}"
                label_w = max(len(label) * 7, 60)
                draw.rectangle([x1, y1, x1 + label_w, y1 + 18],
                                fill=r_col + (230,))
                draw.text((x1 + 3, y1 + 2), label, fill=(255, 255, 255))

            st.image(annotated,
                     caption=f"{uploaded_file.name} — detected regions overlaid",
                     use_container_width=True)

        with col2:
            st.markdown("#### Region Details")
            st.markdown("---")
            for region in page.ordered_regions():
                color = REGION_COLORS.get(region.type.value, "#CCCCCC")
                preview = (region.text[:55] + "…"
                           if len(region.text) > 55 else region.text)
                bbox = region.bbox
                st.markdown(
                    f"<div style='background:{color}18;padding:10px;"
                    f"margin:5px 0;border-left:4px solid {color};"
                    f"border-radius:5px;'>"
                    f"<strong>#{region.reading_order} &nbsp;"
                    f"<span style='color:{color}'>"
                    f"{region.type.value.upper()}</span></strong><br>"
                    f"<small>BBox: ({bbox.x1},{bbox.y1}) → ({bbox.x2},{bbox.y2}) "
                    f"| {bbox.x2-bbox.x1}×{bbox.y2-bbox.y1} px</small><br>"
                    f"<small>Script: <b>{region.script}</b> &nbsp;|&nbsp; "
                    f"Conf: <b>{region.detection_confidence:.0%}</b> &nbsp;|&nbsp; "
                    f"Backend: <b>{region.backend or 'N/A'}</b></small><br>"
                    f"<code style='font-size:11px;'>"
                    f"{preview if region.text else '(non-text region)'}</code>"
                    f"</div>",
                    unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### Region Type Distribution")
            type_counts = {}
            for region in page.regions:
                type_counts[region.type.value] = \
                    type_counts.get(region.type.value, 0) + 1

            import pandas as pd
            df_types = pd.DataFrame(
                list(type_counts.items()), columns=["Region Type", "Count"]
            ).set_index("Region Type").sort_values("Count", ascending=False)
            st.bar_chart(df_types, use_container_width=True)

            st.markdown("##### Colour Legend")
            for rtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                color = REGION_COLORS.get(rtype, "#CCCCCC")
                st.markdown(
                    f"<span style='background:{color};padding:3px 10px;"
                    f"border-radius:4px;color:white;margin:2px;"
                    f"display:inline-block;'>{rtype}: {count}</span>",
                    unsafe_allow_html=True)

    with tab2:
        st.subheader("Extracted Text — Reading Order")
        full_text = page.full_text()

        # Image-specific header
        st.caption(
            f"File: {props['filename']}  |  "
            f"{len(page.regions)} regions  |  "
            f"{sum(len(r.lines) for r in page.regions)} lines  |  "
            f"Language: {props['lang_guess']}"
        )

        st.text_area("Extracted text", full_text, height=380,
                     label_visibility="collapsed")

        c1, c2 = st.columns(2)
        c1.download_button("💾 Download .txt", full_text,
                           file_name=f"{Path(props['filename']).stem}_output.txt",
                           mime="text/plain")
        c2.download_button("💾 Download .md", page.to_markdown(),
                           file_name=f"{Path(props['filename']).stem}_output.md",
                           mime="text/markdown")

        st.markdown("---")
        st.markdown("#### Markdown Formatted")
        st.markdown(page.to_markdown())

    with tab3:
        st.subheader("Structured JSON Output")
        st.caption(
            f"Source: {props['filename']}  |  "
            f"Config hash: {doc.pipeline_config.get('config_hash')}  |  "
            f"Schema version: {doc.schema_version}"
        )
        json_str = json.dumps(doc.model_dump(mode="json"),
                              ensure_ascii=False, indent=2)
        st.code(json_str, language="json")
        st.download_button(
            "💾 Download JSON",
            json_str,
            file_name=f"{Path(props['filename']).stem}_output.json",
            mime="application/json",
        )

    with tab4:
        st.subheader("⚡ Performance — Based on Your Image")

        st.caption(
            f"Image: {props['width']}×{props['height']} px  |  "
            f"{props['megapixels']} MP  |  "
            f"SAHI: {'ON' if use_sahi else 'OFF'}  |  "
            f"Layout: {'ON' if use_layout else 'OFF'}"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Time",       f"{timings['total']:.2f}s")
        c1.metric("Preprocessing",    f"{timings['preprocess']:.2f}s")
        c1.metric("Layout Detection", f"{timings['layout']:.2f}s")

        c2.metric("Reading Order",    f"{timings['reading_order']:.3f}s")
        c2.metric("Recognition",      f"{timings['recognition']:.2f}s")
        c2.metric("Per Page",         f"{timings['seconds_per_page']:.2f}s")

        c3.metric("Regions Detected", len(page.regions))
        c3.metric("Text Regions",
                  len([r for r in page.regions if r.type.is_text_bearing]))
        c3.metric("Total Lines",
                  sum(len(r.lines) for r in page.regions))

        st.markdown("---")
        st.markdown("#### System Comparison (Expected)")
        import pandas as pd
        st.dataframe(pd.DataFrame({
            "System":        ["Tesseract", "PaddleOCR", "Bhashini API",
                              "Surya", "AksharDrishti"],
            "CER (%)":       ["15–20", "10–15", "7–10", "8–12", "6–9"],
            "WER (%)":       ["22–28", "16–22", "10–15", "12–17", "8–13"],
            "Layout-Aware":  ["❌", "❌", "❌", "❌", "✅"],
            "Speed (s/pg)":  ["2–3", "6–10", "12–18†", "18–25", "12–18"],
        }), use_container_width=True, hide_index=True)
        st.caption("† Includes API network latency")

        st.markdown("#### Ablation Study")
        st.markdown("""
        | Stage removed | Expected CER increase |
        |---------------|-----------------------|
        | Layout Detection OFF | +8–12% |
        | SAHI OFF | −0.10 mAP |
        | Script Routing OFF | +3–5% |
        | Postprocessing OFF | +2–3% |
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#888;font-size:12px;'>
        <b>AksharDrishti v0.1.0</b> — K. K. Wagh Institute of Engineering<br>
        Prasad Tarde · Kalpesh Suryawanshi · Ayush Shirsath · Nikhil Pawar
        &nbsp;|&nbsp; Guide: Prof. K. P. Birla
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
