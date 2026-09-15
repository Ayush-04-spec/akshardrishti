#!/usr/bin/env python3
"""Train the Devanagari-vs-Latin script classifier that drives OCR routing.

Cheap: ~200k parameters, converges in well under an hour on a T4.

Input CSV needs columns: image_path, script  (script in {deva, latin})

Build one by combining Mozhi crops (script=deva) with any Latin line-crop source
-- English Mozhi pages, IIIT-5K, or synthetic renders from data/synth_devanagari.py
run with --script latin.

    python train/train_scriptid.py --train datasets/scriptid/train.csv \
        --val datasets/scriptid/val.csv --epochs 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akshardrishti.script_id.model import ScriptCNN  # noqa: E402
from train.dataset import ScriptIDDataset, load_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_scriptid")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = total = 0
    # confusion[true][pred]
    confusion = [[0, 0], [0, 0]]

    for images, labels in loader:
        preds = model(images.to(device)).argmax(1).cpu()
        for true, pred in zip(labels, preds):
            confusion[int(true)][int(pred)] += 1
            correct += int(true == pred)
            total += 1

    acc = correct / total if total else 0.0
    deva_recall = confusion[0][0] / max(sum(confusion[0]), 1)
    latin_recall = confusion[1][1] / max(sum(confusion[1]), 1)
    return {"accuracy": acc, "deva_recall": deva_recall, "latin_recall": latin_recall,
            "confusion": confusion, "n": total}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("weights/script_id"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = ScriptIDDataset(load_index(args.train), augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))
    val_loader = None
    if args.val and args.val.exists():
        val_ds = ScriptIDDataset(load_index(args.val), augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    log.info("train=%d  val=%s  device=%s", len(train_ds), len(val_loader.dataset) if val_loader else 0, device)

    model = ScriptCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, history = 0.0, []

    for epoch in range(args.epochs):
        model.train()
        running = seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": running / max(seen, 1)}
        if val_loader:
            metrics = evaluate(model, val_loader, device)
            row |= {k: v for k, v in metrics.items() if k != "confusion"}
            log.info("epoch %d  loss %.4f  acc %.4f  deva-recall %.4f  latin-recall %.4f",
                     epoch, row["train_loss"], metrics["accuracy"], metrics["deva_recall"], metrics["latin_recall"])
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                torch.save({"model": model.state_dict(), "accuracy": best_acc}, args.out / "scriptid_cnn.pt")
                log.info("  new best -> %s", args.out / "scriptid_cnn.pt")
        else:
            log.info("epoch %d  loss %.4f", epoch, row["train_loss"])
            torch.save({"model": model.state_dict()}, args.out / "scriptid_cnn.pt")
        history.append(row)

    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print("Report this in the paper -- the router's confidence floor assumes it is high.")


if __name__ == "__main__":
    main()
