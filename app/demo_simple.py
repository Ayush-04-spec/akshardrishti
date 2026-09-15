#!/usr/bin/env python3
"""
Simplified AksharDrishti Demo - No Model Weights Required
Shows the UI and demonstrates the pipeline structure without heavy dependencies.
"""

import streamlit as st
import json
from pathlib import Path
import sys

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


def main():
    st.title("📄 AksharDrishti Demo")
    st.markdown("**Layout-Aware OCR for Multilingual Indian Documents**")
    st.caption("Hindi · Marathi · English")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Pipeline Stages")
        use_layout = st.checkbox("Layout Detection", True, 
                                help="YOLO11 detects 11 region types")
        use_sahi = st.checkbox("SAHI Sliced Inference", True,
                              help="For dense pages (~4× slower)")
        use_script_id = st.checkbox("Script Routing", True,
                                   help="Route Devanagari ↔ Latin to different recognizers")
        use_postprocess = st.checkbox("Devanagari Postprocessing", True,
                                     help="Shirorekha repair, Unicode NFC")
        
        st.subheader("Recognition Backends")
        deva_backend = st.selectbox(
            "Devanagari (Hindi/Marathi)",
            ["crnn_mozhi", "paddle_vl", "surya", "bhashini_api", "tesseract"]
        )
        latin_backend = st.selectbox(
            "Latin (English)",
            ["trocr_printed", "paddle_vl", "tesseract"]
        )
        
        st.subheader("Detection Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.05, 0.90, 0.25, 0.05)
        
        st.divider()
        
        st.markdown("### Region Colors")
        for region_type, color in REGION_COLORS.items():
            st.markdown(
                f"<span style='background-color:{color}; padding: 2px 8px; border-radius: 3px; color: white; font-size: 12px;'>{region_type}</span>",
                unsafe_allow_html=True
            )
    
    # Main content
    st.info("🎯 **Demo Mode**: This is a demonstration of the AksharDrishti output format. "
            "Upload functionality requires trained model weights (YOLO11 + CRNN).")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Sample Output",
        "📝 Extracted Text", 
        "📋 JSON Structure",
        "⚡ Performance Metrics"
    ])
    
    # Generate sample document
    doc = create_sample_document()
    page = doc.pages[0]
    
    with tab1:
        st.subheader("Sample Government Document - Layout Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Detected Regions")
            st.markdown(f"**Total regions:** {len(page.regions)}")
            st.markdown(f"**Page size:** {page.meta.width}×{page.meta.height}px @ {page.meta.dpi} DPI")
            st.markdown(f"**Preprocessing:** {', '.join(page.meta.preprocessing_applied)}")
            st.markdown(f"**Rotation corrected:** {page.meta.rotation_corrected_deg:.1f}°")
            
            st.markdown("---")
            
            for region in page.ordered_regions():
                color = REGION_COLORS.get(region.type.value, "#CCCCCC")
                text_preview = region.text[:50] + "..." if len(region.text) > 50 else region.text
                
                st.markdown(
                    f"<div style='background-color:{color}20; padding: 10px; margin: 5px 0; border-left: 4px solid {color}; border-radius: 4px;'>"
                    f"<strong>Region {region.reading_order}</strong>: "
                    f"<span style='color:{color}'>{region.type.value.upper()}</span><br>"
                    f"<small>Script: {region.script} | Confidence: {region.detection_confidence:.2%}</small><br>"
                    f"<small>Backend: {region.backend or 'N/A'}</small><br>"
                    f"<code style='font-size: 11px;'>{text_preview if region.text else '(non-text region)'}</code>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown("#### Bounding Box Visualization")
            st.info("📌 In the full version, this shows an interactive image with overlaid bounding boxes")
            
            # Show region statistics
            st.markdown("##### Region Type Distribution")
            type_counts = {}
            for region in page.regions:
                type_counts[region.type.value] = type_counts.get(region.type.value, 0) + 1
            
            for region_type, count in sorted(type_counts.items()):
                color = REGION_COLORS.get(region_type, "#CCCCCC")
                st.markdown(
                    f"<span style='background-color:{color}; padding: 2px 8px; border-radius: 3px; color: white;'>"
                    f"{region_type}: {count}</span>",
                    unsafe_allow_html=True
                )
    
    with tab2:
        st.subheader("Extracted Text (Reading Order)")
        
        st.markdown("#### Full Document Text")
        full_text = page.full_text()
        st.text_area("", full_text, height=400)
        
        st.download_button(
            "💾 Download as Text",
            full_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )
        
        st.markdown("---")
        
        st.markdown("#### Markdown Format")
        md_text = page.to_markdown()
        st.markdown(md_text)
        
        st.download_button(
            "💾 Download as Markdown",
            md_text,
            file_name="extracted_text.md",
            mime="text/markdown"
        )
    
    with tab3:
        st.subheader("Structured JSON Output")
        
        st.markdown("""
        The complete document structure with:
        - Bounding boxes (pixel coordinates)
        - Reading order
        - Confidence scores
        - Script identification
        - Backend tracking
        """)
        
        json_output = doc.model_dump(mode="json")
        json_str = json.dumps(json_output, ensure_ascii=False, indent=2)
        
        st.code(json_str, language="json")
        
        st.download_button(
            "💾 Download JSON",
            json_str,
            file_name="document_output.json",
            mime="application/json"
        )
    
    with tab4:
        st.subheader("⚡ Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Time", f"{doc.timings_s['total']:.2f}s")
            st.metric("Preprocessing", f"{doc.timings_s['preprocess']:.2f}s")
            st.metric("Layout Detection", f"{doc.timings_s['layout']:.2f}s")
        
        with col2:
            st.metric("Reading Order", f"{doc.timings_s['reading_order']:.3f}s")
            st.metric("Recognition", f"{doc.timings_s['recognition']:.2f}s")
            st.metric("Per Page", f"{doc.timings_s['seconds_per_page']:.2f}s")
        
        with col3:
            st.metric("Regions Detected", len(page.regions))
            st.metric("Text Regions", len([r for r in page.regions if r.type.is_text_bearing]))
            st.metric("Total Lines", sum(len(r.lines) for r in page.regions))
        
        st.markdown("---")
        
        st.markdown("#### Expected Performance (Full System)")
        
        metrics_data = {
            "System": ["Tesseract", "PaddleOCR", "Bhashini API", "Surya", "AksharDrishti"],
            "CER (%)": ["15-20", "10-15", "7-10", "8-12", "6-9"],
            "WER (%)": ["22-28", "16-22", "10-15", "12-17", "8-13"],
            "Layout-Aware": ["❌", "❌", "❌", "❌", "✅"],
            "Speed (s/page)": ["2-3", "6-10", "12-18†", "18-25", "12-18"]
        }
        
        st.table(metrics_data)
        st.caption("† Includes network latency")
        
        st.markdown("#### Ablation Study (Expected Impact)")
        st.markdown("""
        - **Without Layout:** +8-12% CER (columns merge, stamps in text)
        - **Without SAHI:** -0.10 layout mAP (misses small regions)
        - **Without Script Routing:** +3-5% CER (TrOCR fails on Devanagari)
        - **Without Postprocessing:** +2-3% CER (Unicode/conjunct issues)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        <p><strong>AksharDrishti v0.1.0</strong> — K. K. Wagh Institute of Engineering</p>
        <p>Prasad Tarde · Kalpesh Suryawanshi · Ayush Shirsath · Nikhil Pawar</p>
        <p>Guide: Prof. K. P. Birla</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
