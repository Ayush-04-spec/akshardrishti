"""Devanagari-aware text post-processing.

This module exists because raw OCR output of Devanagari has failure modes that
generic text cleanup does not touch, and because a chunk of apparent "OCR error"
is really Unicode representation mismatch.

The report's §3.1.1 critique of monolithic engines calls out shirorekha
segmentation failures specifically. This is where we address them.
"""

from __future__ import annotations

import re
import unicodedata

# Devanagari code points we reason about explicitly
VIRAMA = "्"          # halant -- forms conjuncts
ZWJ = "‍"
ZWNJ = "‌"
NUKTA = "़"
DANDA = "।"
DOUBLE_DANDA = "॥"

COMBINING_MARKS = set(
    "ऀँंः"          # candrabindu, anusvara, visarga
    "ऺऻ़ाि"    # matras
    "ीुूृॄॅॆेै"
    "ॉॊोौ्ॎॏ"
    "॒॑॓॔ॕॖॗ"
    "ॢॣ"
)

# Precomposed nukta forms and their decomposed equivalents. OCR models trained on
# different corpora emit different ones; without normalisation they score as
# errors against each other despite rendering identically.
NUKTA_PAIRS = {
    "क़": "क़",  # qa
    "ख़": "ख़",  # khha
    "ग़": "ग़",  # ghha
    "ज़": "ज़",  # za
    "ड़": "ड़",  # dddha
    "ढ़": "ढ़",  # rha
    "फ़": "फ़",  # fa
    "य़": "य़",  # yya
}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:!?।॥\.])")
_ORPHAN_MARK_RE = re.compile(r"(?:(?<=^)|(?<=\s))([" + "".join(COMBINING_MARKS) + r"]+)")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Canonical normalisation. Do this on BOTH prediction and ground truth
    before computing CER, or you are measuring encoding differences."""
    return unicodedata.normalize(form, text)


def canonicalize_nukta(text: str) -> str:
    """Fold precomposed nukta letters to their decomposed form, consistently."""
    for composed, decomposed in NUKTA_PAIRS.items():
        text = text.replace(composed, decomposed)
    return text


def repair_shirorekha_splits(text: str) -> str:
    """Repair conjuncts fragmented at the top bar.

    When a recogniser splits a conjunct, the virama that joined the two
    consonants often survives with stray whitespace or a zero-width character
    wedged around it. The signature is a consonant + virama + separator +
    consonant, where the separator should not be there.
    """
    # consonant + virama + (space | ZWJ | ZWNJ)+ + consonant  ->  drop separator
    text = re.sub(
        r"([क-हक़-य़])" + VIRAMA + r"[\s" + ZWJ + ZWNJ + r"]+([क-हक़-य़])",
        r"\1" + VIRAMA + r"\2",
        text,
    )
    # A combining mark separated from its base consonant by whitespace.
    text = re.sub(
        r"([क-हक़-य़])\s+([" + "".join(COMBINING_MARKS) + r"])",
        r"\1\2",
        text,
    )
    return text


def drop_orphan_marks(text: str) -> str:
    """Remove combining marks with no base character.

    These are pure noise -- a matra recognised where its consonant was missed.
    Leaving them in renders as a dotted-circle placeholder and inflates CER.
    """
    return _ORPHAN_MARK_RE.sub("", text)


def strip_control_chars(text: str) -> str:
    return _CONTROL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def fix_punctuation_spacing(text: str) -> str:
    """No space before a danda or a comma; exactly one after."""
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"([" + DANDA + DOUBLE_DANDA + r",;:])(?=[^\s])", r"\1 ", text)
    return text


def clean_text(
    text: str,
    unicode_form: str = "NFC",
    repair_shirorekha: bool = True,
    strip_control: bool = True,
    collapse_ws: bool = True,
    fix_punctuation: bool = True,
    remove_orphan_marks: bool = True,
) -> str:
    """The full post-processing chain. Order matters.

    Normalise first (so later regexes see canonical forms), repair structure,
    then tidy whitespace last.
    """
    if not text:
        return ""
    if strip_control:
        text = strip_control_chars(text)
    text = normalize_unicode(text, unicode_form)
    text = canonicalize_nukta(text)
    if repair_shirorekha:
        text = repair_shirorekha_splits(text)
    if remove_orphan_marks:
        text = drop_orphan_marks(text)
    if fix_punctuation:
        text = fix_punctuation_spacing(text)
    if collapse_ws:
        text = collapse_whitespace(text)
    return normalize_unicode(text, unicode_form)


def normalize_for_scoring(text: str) -> str:
    """Minimal normalisation applied identically to prediction AND ground truth.

    Deliberately conservative: it fixes encoding-level differences only, and
    does NOT repair errors. Applying `clean_text` to ground truth would let the
    post-processor score its own homework.
    """
    text = strip_control_chars(text)
    text = normalize_unicode(text, "NFC")
    text = canonicalize_nukta(text)
    text = re.sub(r"[" + ZWJ + ZWNJ + r"]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
