"""
Create a small 2000-row Mozhi sample for Colab fine-tuning.
Copies images and writes a fixed CSV with correct paths.
Output: datasets/mozhi_sample/ (images + index_train.csv)
"""
import csv, random, shutil
from pathlib import Path

random.seed(42)
ROOT = Path(__file__).resolve().parent.parent

rows = list(csv.DictReader(open(ROOT / 'datasets/mozhi_raw/index_train.csv', encoding='utf-8')))
sample = random.sample(rows, min(2000, len(rows)))

out_dir = ROOT / 'datasets/mozhi_sample/images'
out_dir.mkdir(parents=True, exist_ok=True)

copied = 0
fixed_rows = []
for r in sample:
    src = Path(r['image_path'])
    if not src.exists():
        continue
    dst = out_dir / src.name
    shutil.copy(src, dst)
    fixed_rows.append({
        'image_path': str(dst),
        'text': r['text'],
        'language': r['language'],
        'split': 'train'
    })
    copied += 1
    if copied % 500 == 0:
        print(f'  {copied}/2000...')

out_csv = ROOT / 'datasets/mozhi_sample/index_train.csv'
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['image_path','text','language','split'])
    w.writeheader()
    w.writerows(fixed_rows)

size_mb = sum(p.stat().st_size for p in out_dir.glob('*')) / 1e6
print(f'Copied: {copied} images')
print(f'CSV: {out_csv}')
print(f'Size: {size_mb:.1f} MB')
