#!/usr/bin/env python3
"""Train the CRNN-CTC Devanagari recognizer on Mozhi.

This is model #2 -- our Bhashini-lineage recognizer for Hindi and Marathi.

Usage
-----
    python train/train_crnn.py --data datasets/mozhi --epochs 30 --batch 64

    # resume after a Kaggle session dies
    python train/train_crnn.py --data datasets/mozhi --resume weights/recognize/last.pt

Kaggle notes
------------
* Batch 64 at height 32 fits a single T4 comfortably; 128 fits with AMP.
* ~30 epochs on Hindi+Marathi word crops is roughly 6-8h. Launch it overnight.
* Checkpoints save EVERY epoch to --out. Point --backup-dir at Drive.
* Validation CER is computed with greedy decode for speed; the final model uses
  beam search at inference, which typically gains another 1-3% absolute.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akshardrishti.recognize.crnn_model import CRNN, Charset, greedy_decode  # noqa: E402
from akshardrishti.postprocess.text_repair import normalize_for_scoring  # noqa: E402
from train.dataset import LineImageDataset, ctc_collate, load_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_crnn")


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Kept local so evaluation has no hard dependency
    on jiwer during training."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def corpus_cer(preds: list[str], refs: list[str]) -> float:
    total_err = sum(edit_distance(normalize_for_scoring(p), normalize_for_scoring(r)) for p, r in zip(preds, refs))
    total_len = sum(len(normalize_for_scoring(r)) for r in refs)
    return total_err / total_len if total_len else 0.0


@torch.no_grad()
def evaluate(model, loader, charset, device, max_batches: int | None = None) -> dict:
    model.eval()
    preds: list[str] = []
    refs: list[str] = []

    for i, (images, _, _, _, texts) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        log_probs = model(images.to(device))
        preds.extend(text for text, _ in greedy_decode(log_probs, charset))
        refs.extend(texts)

    exact = sum(normalize_for_scoring(p) == normalize_for_scoring(r) for p, r in zip(preds, refs))
    return {
        "cer": corpus_cer(preds, refs),
        "word_accuracy": exact / len(refs) if refs else 0.0,
        "n_samples": len(refs),
        "samples": list(zip(refs[:5], preds[:5])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("datasets/mozhi"))
    p.add_argument("--out", type=Path, default=Path("weights/recognize"))
    p.add_argument("--charset", type=Path, default=None, help="defaults to <data>/charset_deva.txt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img-height", type=int, default=32)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--val-batches", type=int, default=60, help="cap val batches per epoch for speed")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--backup-dir", type=Path, default=None)
    p.add_argument("--wandb", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device: %s", device)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    charset_path = args.charset or (args.data / "charset_deva.txt")
    if not charset_path.exists():
        raise SystemExit(f"charset not found: {charset_path}\nRun data/prepare_mozhi.py first.")
    charset = Charset.load(charset_path)
    log.info("charset: %d characters (+1 CTC blank)", len(charset) - 1)

    train_rows = load_index(args.data / "index_train.csv")
    val_csv = args.data / "index_val.csv"
    val_rows = load_index(val_csv) if val_csv.exists() else []
    log.info("train=%d  val=%d", len(train_rows), len(val_rows))

    train_ds = LineImageDataset(train_rows, charset, args.img_height, args.max_width, augment=not args.no_augment)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              collate_fn=ctc_collate, pin_memory=(device == "cuda"), drop_last=True)

    val_loader = None
    if val_rows:
        val_ds = LineImageDataset(val_rows, charset, args.img_height, args.max_width, augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                collate_fn=ctc_collate, pin_memory=(device == "cuda"))

    # --------------------------------------------------------------- model
    model = CRNN(num_classes=len(charset), img_height=args.img_height, hidden=args.hidden).to(device)
    log.info("parameters: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    # zero_infinity guards against inf loss when a target is longer than the
    # input sequence -- happens with very wide crops and kills training silently.
    criterion = nn.CTCLoss(blank=Charset.BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(len(train_loader), 1), pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    start_epoch, best_cer, bad_epochs = 0, float("inf"), 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_cer = ckpt.get("best_cer", float("inf"))
        log.info("resumed from %s at epoch %d (best CER %.4f)", args.resume, start_epoch, best_cer)

    if args.wandb:
        import wandb

        wandb.init(project=args.wandb, config=vars(args) | {"charset_size": len(charset)})

    history: list[dict] = []

    # --------------------------------------------------------------- train
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()

        for step, (images, targets, input_lengths, target_lengths, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(args.amp and device == "cuda")):
                log_probs = model(images)
                # CTC wants float32 log-probs even under AMP.
                loss = criterion(log_probs.float(), targets, input_lengths, target_lengths)

            if not torch.isfinite(loss):
                log.warning("non-finite loss at epoch %d step %d -- skipping batch", epoch, step)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * images.size(0)
            seen += images.size(0)

            if step % 100 == 0:
                log.info("epoch %d  step %d/%d  loss %.4f  lr %.2e",
                         epoch, step, len(train_loader), loss.item(), scheduler.get_last_lr()[0])

        train_loss = running / max(seen, 1)
        row = {"epoch": epoch, "train_loss": train_loss, "seconds": time.time() - t0}

        if val_loader:
            metrics = evaluate(model, val_loader, charset, device, max_batches=args.val_batches)
            row |= {"val_cer": metrics["cer"], "val_word_acc": metrics["word_accuracy"]}
            log.info("epoch %d  loss %.4f  val CER %.4f  word-acc %.4f  (%.0fs)",
                     epoch, train_loss, metrics["cer"], metrics["word_accuracy"], row["seconds"])
            for ref, pred in metrics["samples"][:3]:
                log.info("    ref=%r  pred=%r", ref, pred)
        else:
            log.info("epoch %d  loss %.4f  (%.0fs)", epoch, train_loss, row["seconds"])

        history.append(row)
        if args.wandb:
            import wandb

            wandb.log(row)

        # Save EVERY epoch -- Kaggle sessions get killed without warning.
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_cer": best_cer, "charset_size": len(charset), "args": vars(args)}
        torch.save(ckpt, args.out / "last.pt")

        current = row.get("val_cer", train_loss)
        if current < best_cer:
            best_cer, bad_epochs = current, 0
            torch.save(ckpt, args.out / "crnn_mozhi_himr.pt")
            charset.save(args.out / "charset_deva.txt")
            log.info("  new best (CER %.4f) -> %s", best_cer, args.out / "crnn_mozhi_himr.pt")
            if args.backup_dir:
                args.backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.out / "crnn_mozhi_himr.pt", args.backup_dir / "crnn_mozhi_himr.pt")
                shutil.copy2(args.out / "charset_deva.txt", args.backup_dir / "charset_deva.txt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info("early stopping: no improvement for %d epochs", args.patience)
                break

        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ----------------------------------------------------------- final test
    test_csv = args.data / "index_test.csv"
    if test_csv.exists():
        log.info("evaluating best checkpoint on the test split")
        model.load_state_dict(torch.load(args.out / "crnn_mozhi_himr.pt", map_location=device)["model"])
        test_ds = LineImageDataset(load_index(test_csv), charset, args.img_height, args.max_width, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, collate_fn=ctc_collate)
        metrics = evaluate(model, test_loader, charset, device)
        print("\n" + "=" * 56)
        print("  Mozhi test split (greedy decode)")
        print("=" * 56)
        print(f"  CER           : {metrics['cer']:.4f}")
        print(f"  word accuracy : {metrics['word_accuracy']:.4f}")
        print(f"  samples       : {metrics['n_samples']}")
        print("=" * 56)
        (args.out / "test_metrics.json").write_text(
            json.dumps({k: v for k, v in metrics.items() if k != "samples"}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
