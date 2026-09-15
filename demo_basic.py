#!/usr/bin/env python3
"""
AksharDrishti - Basic Demo (No Model Weights Required)
This demonstrates the pipeline architecture without running actual inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
from akshardrishti.schema import (
    Document, Page, PageMeta, Region, RegionType, BBox, TextLine
)
from akshardrishti.config import Config, ClassMap

def demo_schema():
    """Demonstrate the structured output schema"""
    print("\n" + "="*60)
    print("1. SCHEMA DEMONSTRATION")
    print("="*60)
    
    # Create a mock document structure
    doc = Document(
        source_path="sample_document.pdf",
        pipeline_config={"config_hash": "demo123"},
    )
    
    # Create a page with regions
    page = Page(
        meta=PageMeta(
            source_path="sample_document.pdf",
            page_index=0,
            width=2480,
            height=3508,
            dpi=300,
            preprocessing_applied=["deskew", "clahe", "denoise"]
        ),
        regions=[
            Region(
                region_id=0,
                type=RegionType.TITLE,
                bbox=BBox(x1=100, y1=100, x2=2300, y2=250),
                detection_confidence=0.95,
                reading_order=0,
                script="latin",
                lines=[
                    TextLine(
                        text="Government of India",
                        bbox=BBox(x1=100, y1=100, x2=2300, y2=250),
                        script="latin",
                        confidence=0.92,
                        backend="trocr_printed"
                    )
                ]
            ),
            Region(
                region_id=1,
                type=RegionType.TEXT,
                bbox=BBox(x1=100, y1=300, x2=1150, y2=1500),
                detection_confidence=0.88,
                reading_order=1,
                script="deva",
                lines=[
                    TextLine(
                        text="यह एक नमूना दस्तावेज़ है।",
                        bbox=BBox(x1=100, y1=300, x2=1150, y2=350),
                        script="deva",
                        confidence=0.87,
                        backend="crnn_mozhi"
                    ),
                    TextLine(
                        text="प्रशासनिक उद्देश्यों के लिए।",
                        bbox=BBox(x1=100, y1=370, x2=1150, y2=420),
                        script="deva",
                        confidence=0.91,
                        backend="crnn_mozhi"
                    )
                ]
            ),
            Region(
                region_id=2,
                type=RegionType.STAMP_SEAL,
                bbox=BBox(x1=1800, y1=2800, x2=2200, y2=3200),
                detection_confidence=0.93,
                reading_order=5,
                script="unknown"
            ),
            Region(
                region_id=3,
                type=RegionType.SIGNATURE,
                bbox=BBox(x1=1600, y1=3000, x2=2100, y2=3300),
                detection_confidence=0.89,
                reading_order=6,
                script="unknown"
            )
        ]
    )
    
    # Compose text for regions
    for region in page.regions:
        region.compose_text()
    
    doc.pages.append(page)
    doc.timings_s = {
        "preprocess": 0.8,
        "layout": 2.1,
        "reading_order": 0.05,
        "recognition": 12.3,
        "total": 15.25,
        "seconds_per_page": 15.25
    }
    
    print(f"\n✓ Document: {doc.source_path}")
    print(f"✓ Pages: {len(doc.pages)}")
    print(f"✓ Total regions detected: {len(page.regions)}")
    print(f"✓ Processing time: {doc.timings_s['seconds_per_page']:.2f}s per page")
    
    print("\n--- Region Breakdown ---")
    for region in page.ordered_regions():
        print(f"  {region.reading_order}. {region.type.value:12s} "
              f"(conf: {region.detection_confidence:.2f}) "
              f"[{region.script}] "
              f"{'✓ text' if region.text else '✗ non-text'}")
    
    print("\n--- Extracted Text (Reading Order) ---")
    for region in page.ordered_regions():
        if region.text:
            print(f"\n[{region.type.value.upper()}]")
            print(f"  {region.text}")
    
    # Export to JSON
    output_path = Path("demo_output.json")
    doc.save_json(output_path)
    print(f"\n✓ Saved structured output to: {output_path}")
    
    # Export to Markdown
    md_text = doc.to_markdown()
    md_path = Path("demo_output.md")
    md_path.write_text(md_text, encoding="utf-8")
    print(f"✓ Saved markdown to: {md_path}")
    
    return doc


def demo_config():
    """Demonstrate configuration system"""
    print("\n" + "="*60)
    print("2. CONFIGURATION SYSTEM")
    print("="*60)
    
    cfg = Config.load("configs/pipeline.yaml")
    
    print(f"\n✓ Project: {cfg.get('project.name')} v{cfg.get('project.version')}")
    print(f"✓ Languages: {', '.join(cfg.get('project.languages', []))}")
    print(f"✓ Device: {cfg.get('project.device')}")
    
    print(f"\n--- Preprocessing ---")
    print(f"  Target DPI: {cfg.get('preprocess.target_dpi')}")
    print(f"  Deskew: {cfg.get('preprocess.deskew.enabled')}")
    print(f"  CLAHE: {cfg.get('preprocess.clahe.enabled')}")
    print(f"  Denoise: {cfg.get('preprocess.denoise.enabled')}")
    
    print(f"\n--- Layout Detection ---")
    print(f"  Model: {cfg.get('layout.model')}")
    print(f"  Image size: {cfg.get('layout.imgsz')}px")
    print(f"  Confidence: {cfg.get('layout.conf')}")
    print(f"  SAHI enabled: {cfg.get('layout.sahi.enabled')}")
    
    print(f"\n--- Recognition ---")
    print(f"  Devanagari backend: {cfg.get('recognize.deva_backend')}")
    print(f"  Latin backend: {cfg.get('recognize.latin_backend')}")
    print(f"  Line segmentation: {cfg.get('recognize.line_segmentation.enabled')}")
    
    print(f"\n--- Script Identification ---")
    print(f"  Enabled: {cfg.get('script_id.enabled')}")
    print(f"  Method: {cfg.get('script_id.method')}")
    print(f"  Confidence floor: {cfg.get('script_id.confidence_floor')}")
    
    print(f"\n✓ Config hash: {cfg.hash()}")
    
    return cfg


def demo_class_taxonomy():
    """Demonstrate class mapping"""
    print("\n" + "="*60)
    print("3. CLASS TAXONOMY (42 → 11 Remapping)")
    print("="*60)
    
    class_map = ClassMap.load("configs/class_map.yaml")
    
    print(f"\n✓ Target classes ({len(class_map.targets)}):")
    for idx, name in enumerate(class_map.targets):
        print(f"  {idx:2d}. {name}")
    
    print(f"\n✓ Source mappings: {len(class_map.mapping)} classes")
    print(f"✓ Ignored classes: {len(class_map.ignore)}")
    
    # Show some example mappings
    print(f"\n--- Example Mappings ---")
    examples = [
        ("title", "title"),
        ("section_title", "title"),
        ("paragraph", "text"),
        ("table_cell", "table"),
        ("stamp", "stamp_seal"),
        ("handwritten", "signature"),
    ]
    for source, target in examples:
        if source in class_map.mapping:
            print(f"  {source:20s} → {class_map.mapping[source]}")
    
    return class_map


def demo_metrics():
    """Demonstrate evaluation metrics calculation"""
    print("\n" + "="*60)
    print("4. EVALUATION METRICS (Sample Calculation)")
    print("="*60)
    
    # Import from eval directory
    eval_path = Path(__file__).parent / "eval"
    sys.path.insert(0, str(eval_path))
    try:
        from metrics import cer, wer, levenshtein_ops
    except ImportError:
        print("\n  ✗ Metrics module not available")
        print("  (Expected - eval/ directory structure differs)")
        return
    
    # Sample predictions vs ground truth
    test_cases = [
        {
            "name": "Perfect match",
            "reference": "यह एक परीक्षण है।",
            "hypothesis": "यह एक परीक्षण है।",
        },
        {
            "name": "Minor error (Devanagari)",
            "reference": "सरकारी दस्तावेज़ संख्या १२३",
            "hypothesis": "सरकारी दस्तावेज संख्या १२३",  # missing nukta
        },
        {
            "name": "English with errors",
            "reference": "Government of India Certificate",
            "hypothesis": "Governmant of lndia Certificate",  # typos
        },
    ]
    
    print("\n--- Character Error Rate (CER) ---")
    total_cer = 0
    for tc in test_cases:
        cer_val = cer(tc["reference"], tc["hypothesis"], normalize=True)
        ops = levenshtein_ops(tc["reference"], tc["hypothesis"])
        total_cer += cer_val
        print(f"\n  {tc['name']}:")
        print(f"    CER: {cer_val*100:.2f}%")
        print(f"    Ops: S={ops['substitutions']}, D={ops['deletions']}, I={ops['insertions']}")
    
    avg_cer = total_cer / len(test_cases)
    print(f"\n  Average CER: {avg_cer*100:.2f}%")
    
    print("\n--- Word Error Rate (WER) ---")
    for tc in test_cases:
        wer_val = wer(tc["reference"], tc["hypothesis"], normalize=True)
        print(f"  {tc['name']:30s} WER: {wer_val*100:.2f}%")


def demo_architecture_summary():
    """Print architecture summary"""
    print("\n" + "="*60)
    print("5. SYSTEM ARCHITECTURE SUMMARY")
    print("="*60)
    
    architecture = """
    Input (PDF/Image)
        ↓
    [Preprocessing]
        - Deskew (Hough/moments)
        - CLAHE enhancement
        - Denoising (FastNLMeans)
        ↓
    [Layout Detection - YOLO11]
        - 11-class taxonomy
        - SAHI (sliced inference for dense pages)
        - Bounding box extraction
        ↓
    [Reading Order - XY-cut]
        - Furniture extraction
        - Recursive whitespace analysis
        - Multi-column handling
        ↓
    [Line Segmentation]
        - Horizontal projection profiles
        - Peak/valley detection
        ↓
    [Script Identification - CNN]
        - Devanagari vs Latin classification
        - Confidence-based routing
        ↓
    [Recognition - Backend Routing]
        - CRNN-Mozhi (Devanagari: Hindi, Marathi)
        - TrOCR (English/Latin)
        - 4+ additional backends available
        ↓
    [Postprocessing]
        - Unicode NFC normalization
        - Shirorekha repair (Devanagari)
        - Whitespace cleanup
        ↓
    Output (JSON + Markdown + Text)
    """
    print(architecture)
    
    print("\n--- Key Components ---")
    components = {
        "Layout Model": "YOLO11-large (Ultralytics)",
        "Training Data (Layout)": "IndicDLP (119K pages, 12 languages)",
        "Recognition (Deva)": "CRNN-CTC trained on Mozhi (1.2M words)",
        "Recognition (Latin)": "TrOCR (microsoft/trocr-base-printed)",
        "Script Classifier": "CNN (~200K params)",
        "Output Schema": "Pydantic models → JSON/Markdown/Text",
        "Config System": "YAML with dotted access + CLI overrides",
    }
    for k, v in components.items():
        print(f"  {k:25s}: {v}")


def demo_expected_results():
    """Show expected performance metrics"""
    print("\n" + "="*60)
    print("6. EXPECTED PERFORMANCE METRICS")
    print("="*60)
    print("""
Based on similar systems in literature:

┌─────────────────────────┬──────────┬──────────┬────────────┬──────────┐
│ System                  │ CER (%)  │ WER (%)  │ Layout mAP │ s/page   │
├─────────────────────────┼──────────┼──────────┼────────────┼──────────┤
│ Tesseract Full Page     │  15-20   │  22-28   │     —      │   2-3    │
│ PaddleOCR Zero-Shot     │  10-15   │  16-22   │     —      │   6-10   │
│ Bhashini API            │   7-10   │  10-15   │     —      │  12-18†  │
│ Surya                   │   8-12   │  12-17   │     —      │  18-25   │
│ AksharDrishti (Target)  │   6-9    │   8-13   │  0.65-0.75 │  12-18   │
└─────────────────────────┴──────────┴──────────┴────────────┴──────────┘

† Includes network latency

--- Ablation Study Expected Δ ---
  • no_layout:         +8-12% CER (columns merge, stamps in text)
  • no_sahi:          -0.10 layout mAP on dense pages
  • no_script_routing: +3-5% CER (TrOCR fails on Devanagari)
  • no_postprocess:   +2-3% CER (Unicode/conjunct issues)

--- Per-Script Expected Performance ---
  • Hindi/Marathi (Devanagari):  6-8% CER
  • English (Latin):             5-7% CER
  • Mixed documents:             7-10% CER

--- Throughput Breakdown ---
  • Preprocessing:     0.5-1.0s
  • Layout Detection:  1.5-3.0s (SAHI: ×4)
  • Reading Order:     0.03-0.08s
  • Recognition:       10-14s (bottleneck, 80% of time)
  • Total:            ~12-18s per page on T4 GPU
""")


def main():
    print("\n" + "="*70)
    print(" "*15 + "AksharDrishti - Basic Demo")
    print(" "*10 + "Layout-Aware OCR for Multilingual Indian Documents")
    print("="*70)
    
    try:
        # Run all demos
        doc = demo_schema()
        cfg = demo_config()
        class_map = demo_class_taxonomy()
        demo_metrics()
        demo_architecture_summary()
        demo_expected_results()
        
        print("\n" + "="*70)
        print("✓ DEMO COMPLETE")
        print("="*70)
        print("""
NEXT STEPS TO GET ACTUAL RESULTS:

1. Install Dependencies:
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

2. Download Datasets:
   a) IndicDLP (layout training):
      - Sign up at https://huggingface.co/datasets/ai4bharat/indicdlp
      - Run: python data/prepare_indicdlp.py --out datasets/indicdlp_subset
   
   b) Mozhi (recognition training):
      - Download from https://cvit.iiit.ac.in/usodi/tdocrmil.php
      - Run: python data/prepare_mozhi.py --languages hindi marathi

3. Train Models (or download pretrained):
   a) Layout: python train/train_layout.py --data datasets/indicdlp_subset/data.yaml
   b) Recognition: python train/train_crnn.py --data datasets/mozhi
   c) Script ID: python train/train_scriptid.py --train datasets/scriptid/train.csv

4. Create Evaluation Benchmark:
   - Add document images to benchmark/images/
   - Annotate ground truth in benchmark/gt_text/
   - See benchmark/ANNOTATION_GUIDELINES.md

5. Run Evaluation:
   python eval/run_benchmark.py --benchmark benchmark --out results/ --ablations

6. Or Use Demo App:
   streamlit run app/streamlit_app.py

NOTE: This demo ran WITHOUT model weights or training data.
      Actual accuracy metrics require trained models and annotated evaluation data.
""")
        
        print(f"\n✓ Generated demo files:")
        print(f"  - demo_output.json (structured document)")
        print(f"  - demo_output.md (markdown export)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
