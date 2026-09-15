"""CRNN + CTC architecture for Devanagari printed-text recognition.

This is model #2 -- the one you train on Mozhi. It is the same architecture
family that underlies IIIT-H's Bhashini printed-OCR service, which is why the
comparison against the Bhashini API is a fair one rather than an apples-to-
oranges stunt.

Architecture
------------
    input   32 x W x 1  (height-normalised grayscale line)
    CNN     7 conv layers, downsamples height 32 -> 1, width W -> W/4
    RNN     2x bidirectional LSTM, 256 hidden
    head    linear -> |charset| + 1  (the +1 is the CTC blank at index 0)

Devanagari-specific choice
--------------------------
The pooling schedule keeps *width* resolution high (only /4 total) while
collapsing height aggressively. Devanagari packs a lot of horizontal detail --
matras, conjuncts, the shirorekha -- into narrow glyph clusters, and downsampling
width to /8 (as many Latin CRNNs do) measurably hurts conjunct recognition.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class Charset:
    """Character vocabulary with CTC blank reserved at index 0."""

    BLANK = 0

    def __init__(self, chars: list[str]) -> None:
        if len(set(chars)) != len(chars):
            dupes = sorted({c for c in chars if chars.count(c) > 1})
            raise ValueError(f"charset has duplicates: {dupes[:10]}")
        self.chars = list(chars)
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}  # 0 = blank
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars) + 1  # + blank

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx_to_char.get(i, "") for i in indices if i != self.BLANK)

    def coverage(self, text: str) -> float:
        """Fraction of characters this charset can represent. Check before training --
        a charset built from the training split that misses 3% of validation
        characters puts a hard floor under your CER."""
        if not text:
            return 1.0
        return sum(c in self.char_to_idx for c in text) / len(text)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.chars), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Charset:
        raw = Path(path).read_text(encoding="utf-8")
        chars = [line for line in raw.split("\n") if line != ""]
        return cls(chars)

    @classmethod
    def from_corpus(cls, texts: list[str], min_count: int = 1) -> Charset:
        from collections import Counter

        counts = Counter(c for t in texts for c in t)
        chars = sorted(c for c, n in counts.items() if n >= min_count)
        return cls(chars)


class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_height: int = 32, num_channels: int = 1, hidden: int = 256) -> None:
        super().__init__()
        if img_height != 32:
            raise ValueError("this CNN stack assumes img_height=32")
        self.num_classes = num_classes

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_relu(i: int, batch_norm: bool = False) -> None:
            n_in = num_channels if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module(f"conv{i}", nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batch_norm:
                cnn.add_module(f"bn{i}", nn.BatchNorm2d(n_out))
            cnn.add_module(f"relu{i}", nn.ReLU(inplace=True))

        conv_relu(0)
        cnn.add_module("pool0", nn.MaxPool2d(2, 2))          # 32x W  -> 16 x W/2
        conv_relu(1)
        cnn.add_module("pool1", nn.MaxPool2d(2, 2))          # 16xW/2 ->  8 x W/4
        conv_relu(2, batch_norm=True)
        conv_relu(3)
        cnn.add_module("pool2", nn.MaxPool2d((2, 1), (2, 1)))  # 8xW/4 ->  4 x W/4
        conv_relu(4, batch_norm=True)
        conv_relu(5)
        cnn.add_module("pool3", nn.MaxPool2d((2, 1), (2, 1)))  # 4xW/4 ->  2 x W/4
        conv_relu(6, batch_norm=True)                          # 2xW/4 ->  1 x W/4

        self.cnn = cnn
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, batch_first=False, dropout=0.1)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, 32, W) -> log-probs (T, B, num_classes) for CTC."""
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        if h != 1:
            raise RuntimeError(f"expected feature height 1 after CNN, got {h}")
        conv = conv.squeeze(2).permute(2, 0, 1)  # (W, B, C)
        rec, _ = self.rnn(conv)
        return self.fc(rec).log_softmax(2)


# ------------------------------------------------------------------ decoding


def greedy_decode(log_probs: torch.Tensor, charset: Charset) -> list[tuple[str, float]]:
    """Best-path CTC decode. ``log_probs`` is (T, B, C)."""
    probs = log_probs.detach().cpu()
    best = probs.argmax(2).permute(1, 0)          # (B, T)
    max_lp = probs.max(2).values.permute(1, 0)    # (B, T)

    out: list[tuple[str, float]] = []
    for seq, lps in zip(best, max_lp):
        collapsed: list[int] = []
        kept: list[float] = []
        prev = -1
        for idx, lp in zip(seq.tolist(), lps.tolist()):
            if idx != prev and idx != Charset.BLANK:
                collapsed.append(idx)
                kept.append(lp)
            prev = idx
        text = charset.decode(collapsed)
        conf = float(torch.tensor(kept).exp().mean()) if kept else 0.0
        out.append((text, conf))
    return out


def beam_decode(log_probs: torch.Tensor, charset: Charset, beam_width: int = 10) -> list[tuple[str, float]]:
    """Vanilla CTC prefix beam search (no language model).

    Typically buys 1-3% absolute CER over greedy on Devanagari. Set
    ``beam_width: 1`` in the config to fall back to greedy for speed.
    """
    if beam_width <= 1:
        return greedy_decode(log_probs, charset)

    probs = log_probs.detach().cpu().exp().numpy()
    results: list[tuple[str, float]] = []

    for b in range(probs.shape[1]):
        mat = probs[:, b, :]
        # beam: prefix -> (p_blank, p_non_blank)
        beam: dict[tuple[int, ...], tuple[float, float]] = {(): (1.0, 0.0)}

        for t in range(mat.shape[0]):
            next_beam: dict[tuple[int, ...], tuple[float, float]] = {}
            # Only consider the top-k symbols at this timestep; the tail
            # contributes nothing and costs a lot.
            top = mat[t].argsort()[-beam_width * 2:]
            for prefix, (p_b, p_nb) in beam.items():
                p_total = p_b + p_nb
                if p_total <= 0.0:
                    continue
                for c in top:
                    p = float(mat[t, c])
                    if p <= 1e-8:
                        continue
                    if c == Charset.BLANK:
                        nb, nnb = next_beam.get(prefix, (0.0, 0.0))
                        next_beam[prefix] = (nb + p_total * p, nnb)
                    else:
                        last = prefix[-1] if prefix else -1
                        if c == last:
                            # repeat without blank extends the same symbol
                            nb, nnb = next_beam.get(prefix, (0.0, 0.0))
                            next_beam[prefix] = (nb, nnb + p_nb * p)
                            new = prefix + (int(c),)
                            nb2, nnb2 = next_beam.get(new, (0.0, 0.0))
                            next_beam[new] = (nb2, nnb2 + p_b * p)
                        else:
                            new = prefix + (int(c),)
                            nb2, nnb2 = next_beam.get(new, (0.0, 0.0))
                            next_beam[new] = (nb2, nnb2 + p_total * p)

            beam = dict(sorted(next_beam.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)[:beam_width])

        if not beam:
            results.append(("", 0.0))
            continue
        best_prefix, (pb, pnb) = max(beam.items(), key=lambda kv: kv[1][0] + kv[1][1])
        results.append((charset.decode(list(best_prefix)), float(min(pb + pnb, 1.0))))

    return results
