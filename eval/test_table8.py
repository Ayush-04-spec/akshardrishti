import sys
sys.stdout = sys.stderr  # Force unbuffered
print("TEST: Script started", flush=True)

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
print(f"TEST: ROOT = {ROOT}", flush=True)

gt_text_dir = ROOT / "benchmark" / "gt_text"
print(f"TEST: gt_text_dir exists: {gt_text_dir.exists()}", flush=True)

if gt_text_dir.exists():
    files = list(gt_text_dir.glob("BM_*.txt"))
    print(f"TEST: Found {len(files)} GT files", flush=True)

print("TEST: Script finished", flush=True)
