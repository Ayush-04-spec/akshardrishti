from pathlib import Path

root = Path("datasets/indicdlp_subset")
train_imgs = len(list((root / "images/train").glob("*")))
val_imgs = len(list((root / "images/val").glob("*")))

ann_count = 0
for f in (root / "labels/train").glob("*.txt"):
    ann_count += len([l for l in f.read_text().splitlines() if l.strip()])
for f in (root / "labels/val").glob("*.txt"):
    ann_count += len([l for l in f.read_text().splitlines() if l.strip()])

print(f"Train: {train_imgs}")
print(f"Val: {val_imgs}")
print(f"Total: {train_imgs + val_imgs}")
print(f"Annotations: {ann_count}")
