"""Evaluation metrics: CER, WER, layout mAP, reading-order accuracy.

The single most important rule in this file: **normalise prediction and ground
truth identically, and never apply the pipeline's own error-repair to ground
truth.** Otherwise the post-processor grades its own homework and the CER you
report is not measuring what you claim.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akshardrishti.postprocess.text_repair import normalize_for_scoring  # noqa: E402
from akshardrishti.schema import BBox  # noqa: E402


# ------------------------------------------------------------------ distance


def levenshtein(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def levenshtein_ops(a: Sequence, b: Sequence) -> dict[str, int]:
    """Substitutions / deletions / insertions, for the report's CER breakdown
    (the S, D, I terms in the report's equation 6.1)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))

    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            subs += int(a[i - 1] != b[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    # `a` is the reference, `b` the hypothesis: chars in a but not b are deletions.
    return {"substitutions": subs, "deletions": dels, "insertions": ins}


# ----------------------------------------------------------------- CER / WER


def cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    if normalize:
        reference, hypothesis = normalize_for_scoring(reference), normalize_for_scoring(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    if normalize:
        reference, hypothesis = normalize_for_scoring(reference), normalize_for_scoring(hypothesis)
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def corpus_cer(references: list[str], hypotheses: list[str], normalize: bool = True) -> dict:
    """Corpus-level CER: total errors / total reference characters.

    NOT the mean of per-document CERs -- that over-weights short documents and
    is the more flattering (and wrong) number. Both are returned so the
    difference is visible.
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"length mismatch: {len(references)} refs vs {len(hypotheses)} hyps")

    total_err = total_chars = 0
    ops = defaultdict(int)
    per_doc: list[float] = []

    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref, hyp = normalize_for_scoring(ref), normalize_for_scoring(hyp)
        total_err += levenshtein(ref, hyp)
        total_chars += len(ref)
        for k, v in levenshtein_ops(ref, hyp).items():
            ops[k] += v
        per_doc.append(cer(ref, hyp, normalize=False))

    corpus = total_err / total_chars if total_chars else 0.0
    mean_doc = sum(per_doc) / len(per_doc) if per_doc else 0.0
    return {
        "cer": corpus,
        "accuracy": 1.0 - corpus,
        "mean_document_cer": mean_doc,
        "total_errors": total_err,
        "total_reference_chars": total_chars,
        "n_documents": len(references),
        **dict(ops),
        "per_document_cer": per_doc,
    }


def corpus_wer(references: list[str], hypotheses: list[str], normalize: bool = True) -> dict:
    total_err = total_words = 0
    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref, hyp = normalize_for_scoring(ref), normalize_for_scoring(hyp)
        rw, hw = ref.split(), hyp.split()
        total_err += levenshtein(rw, hw)
        total_words += len(rw)
    value = total_err / total_words if total_words else 0.0
    return {"wer": value, "total_errors": total_err, "total_reference_words": total_words}


def bootstrap_ci(per_document: list[float], n_resamples: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """95% CI over documents by bootstrap.

    Report this. With a 50-page benchmark, a bare CER of "0.06" invites the
    question "plus or minus what?" -- and having the answer ready is the
    difference between a confident viva and an awkward one.
    """
    import random

    if not per_document:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(per_document)
    means = []
    for _ in range(n_resamples):
        sample = [per_document[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return (lo, hi)


# -------------------------------------------------------------- layout mAP


def _ap_for_class(preds: list[tuple[BBox, float]], gts: list[BBox], iou_threshold: float) -> float:
    """Average precision at one IoU threshold, VOC-style all-point interpolation."""
    if not gts:
        return float("nan")
    if not preds:
        return 0.0

    preds = sorted(preds, key=lambda p: p[1], reverse=True)
    matched = [False] * len(gts)
    tp = [0] * len(preds)
    fp = [0] * len(preds)

    for i, (box, _) in enumerate(preds):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched[j]:
                continue
            value = box.iou(gt)
            if value > best_iou:
                best_iou, best_j = value, j
        if best_iou >= iou_threshold and best_j >= 0:
            matched[best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1

    cum_tp = cum_fp = 0
    precisions, recalls = [], []
    for t, f in zip(tp, fp):
        cum_tp += t
        cum_fp += f
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / len(gts))

    # All-point interpolation
    ap, prev_recall = 0.0, 0.0
    max_precision_from = list(precisions)
    for i in range(len(max_precision_from) - 2, -1, -1):
        max_precision_from[i] = max(max_precision_from[i], max_precision_from[i + 1])
    for recall, precision in zip(recalls, max_precision_from):
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def layout_map(
    predictions: list[list[tuple[str, BBox, float]]],
    ground_truth: list[list[tuple[str, BBox]]],
    iou_thresholds: Sequence[float] = tuple(x / 100 for x in range(50, 100, 5)),
) -> dict:
    """COCO-style mAP@[.50:.95] over the whole benchmark."""
    by_class_preds: dict[str, list[tuple[BBox, float]]] = defaultdict(list)
    by_class_gts: dict[str, list[BBox]] = defaultdict(list)

    # Note: aggregating across pages ignores per-image matching, which slightly
    # differs from pycocotools. Use this for quick iteration; quote pycocotools
    # numbers (via train/train_layout.py --eval-only) in the paper.
    for page_preds in predictions:
        for name, box, score in page_preds:
            by_class_preds[name].append((box, score))
    for page_gts in ground_truth:
        for name, box in page_gts:
            by_class_gts[name].append(box)

    per_class: dict[str, float] = {}
    per_threshold: dict[str, float] = {}

    for threshold in iou_thresholds:
        aps = []
        for name, gts in by_class_gts.items():
            ap = _ap_for_class(by_class_preds.get(name, []), gts, threshold)
            if ap == ap:  # not NaN
                aps.append(ap)
                if abs(threshold - 0.5) < 1e-9:
                    per_class[name] = ap
        per_threshold[f"mAP@{threshold:.2f}"] = sum(aps) / len(aps) if aps else 0.0

    values = list(per_threshold.values())
    return {
        "mAP50-95": sum(values) / len(values) if values else 0.0,
        "mAP50": per_threshold.get("mAP@0.50", 0.0),
        "mAP75": per_threshold.get("mAP@0.75", 0.0),
        "per_class_AP50": per_class,
        "per_threshold": per_threshold,
    }


# ------------------------------------------------------------ reading order


def reading_order_accuracy(predicted: list[int], ground_truth: list[int]) -> float:
    """Pairwise ordering accuracy -- see layout/reading_order.py."""
    if len(predicted) != len(ground_truth):
        raise ValueError("length mismatch")
    n = len(predicted)
    if n < 2:
        return 1.0
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            correct += int((predicted[i] < predicted[j]) == (ground_truth[i] < ground_truth[j]))
    return correct / total if total else 1.0


def summarize(name: str, results: dict) -> str:
    lo, hi = bootstrap_ci(results.get("per_document_cer", []))
    return (
        f"{name:<24} CER={results.get('cer', 0):.4f} [{lo:.4f}, {hi:.4f}]  "
        f"acc={results.get('accuracy', 0)*100:.2f}%  n={results.get('n_documents', 0)}"
    )
