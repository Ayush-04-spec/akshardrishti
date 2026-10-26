"""Regression tests for scoring alignment and padded CRNN batch decoding."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akshardrishti.recognize.crnn_model import Charset, beam_decode, greedy_decode
from eval.metrics import corpus_cer, corpus_wer


def test_corpus_wer_uses_word_tokens() -> None:
    result = corpus_wer(["क ख ग"], ["क ग"])
    assert result["wer"] == pytest.approx(1 / 3)
    assert result["total_errors"] == 1


def test_corpus_metrics_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        corpus_cer(["reference"], [])
    with pytest.raises(ValueError, match="length mismatch"):
        corpus_wer(["reference"], [])


@pytest.mark.parametrize("decoder", [greedy_decode, beam_decode])
def test_ctc_decoders_ignore_batch_padding(decoder) -> None:
    charset = Charset(["क", "ख"])
    # First two steps are the real crop (क, blank); the rest represents only
    # batch padding and would otherwise append a spurious ख.
    logits = torch.tensor(
        [
            [[10.0, 0.0, 0.0]],
            [[0.0, 10.0, 0.0]],
            [[0.0, 0.0, 10.0]],
            [[0.0, 0.0, 10.0]],
        ]
    )
    log_probs = logits.log_softmax(dim=2)
    if decoder is beam_decode:
        decoded = decoder(log_probs, charset, beam_width=3, sequence_lengths=[2])
    else:
        decoded = decoder(log_probs, charset, sequence_lengths=[2])
    assert decoded[0][0] == "क"
