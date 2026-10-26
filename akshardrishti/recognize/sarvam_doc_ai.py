"""Optional Sarvam Document AI helpers.

This module is deliberately importable without ``sarvamai``. The benchmark
runner is the only supported entry point because Document AI is asynchronous
and page-batched rather than a synchronous crop recognizer.
"""
from __future__ import annotations

import io
from html.parser import HTMLParser
import json
import os
import re
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

from akshardrishti.postprocess.text_repair import normalize_for_scoring

SUPPORTED_LANGUAGES = {"mr-IN", "hi-IN", "en-IN"}
MAX_PAGES_PER_JOB = 10  # Conservative cap across conflicting official pages.
PRICE_INR_PER_PAGE = 0.50
TERMINAL_STATES = {"completed", "partially_completed", "failed", "rejected"}


def build_digitise_call(doc_ai: Any, archive_file: Any, filename: str = "pages.zip",
                        language: str = "mr-IN") -> Any:
    """Construct the official SDK call; separated for offline contract tests."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported configured Sarvam language: {language}")
    return doc_ai.digitise(
        file=[(filename, archive_file, "application/zip")],
        language=language,
    )


def normalize_prediction(text: str) -> str:
    """Apply the same conservative Unicode/whitespace normalization as scoring."""
    return normalize_for_scoring(unicodedata.normalize("NFC", text))


def repair_verified_mojibake(text: str) -> str:
    """Repair only recognizable Latin-1/Windows-1252 UTF-8 mojibake.

    The input must contain common misdecoded Devanagari byte prefixes and no
    already-correct Devanagari. A candidate is accepted only if a reversible
    decode yields Devanagari and removes those mojibake markers.
    """
    markers = ("à¤", "à¥", "à¦", "à§")
    if not any(marker in text for marker in markers):
        return text
    if any(0x0900 <= ord(char) <= 0x097F for char in text):
        return text
    for encoding in ("latin-1", "cp1252"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if (any(0x0900 <= ord(char) <= 0x097F for char in candidate)
                and sum(candidate.count(marker) for marker in markers)
                < sum(text.count(marker) for marker in markers)):
            return candidate
    return text


def _block_position(block: dict, original_index: int) -> tuple[float, float, int]:
    bbox = block.get("bbox_norm")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
        try:
            return float(bbox[1]), float(bbox[0]), original_index
        except (TypeError, ValueError):
            pass
    coordinates = block.get("coordinates")
    if isinstance(coordinates, dict):
        try:
            return float(coordinates.get("y1", 0)), float(coordinates.get("x1", 0)), original_index
        except (TypeError, ValueError):
            pass
    return float("inf"), float("inf"), original_index


def _reading_order_key(entry: tuple[int, dict]) -> tuple:
    index, block = entry
    order = block.get("reading_order")
    if isinstance(order, bool):
        order = None
    try:
        order = float(order) if order is not None else None
    except (TypeError, ValueError):
        order = None
    if order is not None:
        return 0, order, index
    y, x, stable_index = _block_position(block, index)
    return 1, y, x, stable_index


def extract_block_text(blocks: Any) -> str:
    """Extract Unicode OCR text from a page's blocks in reading order."""
    if not isinstance(blocks, list):
        return ""
    ordered = sorted(
        ((index, block) for index, block in enumerate(blocks) if isinstance(block, dict)),
        key=_reading_order_key,
    )
    texts = []
    for _, block in ordered:
        value = block.get("text")
        if not isinstance(value, str) or not value.strip():
            continue
        value = repair_verified_mojibake(value)
        if (block.get("layout_tag") == "table" or "<table" in value.lower()) and "<" in value:
            value = _html_table_to_text(value)
        texts.append(unicodedata.normalize("NFC", value.strip()))
    return "\n".join(texts)


class _TableTextParser(HTMLParser):
    """Convert Sarvam's HTML table block into textual rows and cell spacing."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_row = False
        self.row_has_cell = False

    def _newline(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._newline()
            self.in_row = True
            self.row_has_cell = False
        elif tag in {"td", "th"}:
            if self.row_has_cell and self.parts and not self.parts[-1].endswith(("\t", "\n")):
                self.parts.append("\t")
            self.row_has_cell = True
        elif tag == "br":
            self._newline()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._newline()
            self.in_row = False
        elif tag == "table":
            self._newline()

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _html_table_to_text(value: str) -> str:
    parser = _TableTextParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception:  # Malformed markup should still yield recoverable text.
        return value
    return "".join(parser.parts).strip("\n")


def parse_saved_batch_response(
    response: Any, batch_mapping: list[dict],
) -> tuple[dict[str, str], list[dict]]:
    """Parse one saved JSON response and map its batch-local page numbers.

    ``batch_mapping`` is the corresponding contiguous slice from the saved
    run manifest. Page numbers restart at one for each batch, so mapping is
    scoped to this response and never performed globally.
    """
    if not isinstance(response, dict) or response.get("type") != "digitise":
        raise ValueError("saved batch response is not a Digitise result object")
    if not isinstance(batch_mapping, list) or not batch_mapping:
        raise ValueError("batch mapping is empty or malformed")

    ids: dict[int, str] = {}
    seen_ids: set[str] = set()
    for row in batch_mapping:
        try:
            image_id = str(row["image_id"])
            page_number = int(row["output_page"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("batch mapping contains a missing or invalid image/page entry") from exc
        if image_id in seen_ids or page_number in ids:
            raise ValueError("batch mapping contains duplicate image IDs or page numbers")
        seen_ids.add(image_id)
        ids[page_number] = image_id
    if sorted(ids) != list(range(1, len(batch_mapping) + 1)):
        raise ValueError("batch mapping has missing or non-contiguous local page numbers")

    documents = response.get("documents")
    if not isinstance(documents, list) or len(documents) != 1 or not isinstance(documents[0], dict):
        raise ValueError("expected exactly one ZIP document in saved batch response")
    document = documents[0]
    pages = document.get("pages")
    if not isinstance(pages, list):
        raise ValueError("saved ZIP document has no pages array")

    page_by_number: dict[int, dict] = {}
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("saved response contains a malformed page entry")
        try:
            page_number = int(page["page_num"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("saved page has no valid batch-local page_num") from exc
        if page_number in page_by_number:
            raise ValueError(f"duplicate batch-local page_num {page_number}")
        page_by_number[page_number] = page
    if set(page_by_number) != set(ids):
        missing = sorted(set(ids) - set(page_by_number))
        extra = sorted(set(page_by_number) - set(ids))
        raise ValueError(f"batch page mapping mismatch: missing={missing}, extra={extra}")

    predictions: dict[str, str] = {}
    failures: list[dict] = []
    response_status = str(response.get("status", "")).lower()
    for page_number, image_id in sorted(ids.items()):
        page = page_by_number[page_number]
        page_status = str(page.get("status", response_status)).lower()
        if page_status not in {"completed", "complete", "succeeded", "success"}:
            predictions[image_id] = ""
            failures.append({"id": image_id, "batch_page": page_number,
                             "reason": f"page status is {page_status or 'missing'}"})
            continue
        text = extract_block_text(page.get("blocks"))
        if not text:
            predictions[image_id] = ""
            failures.append({"id": image_id, "batch_page": page_number,
                             "reason": "page completed but has no non-empty OCR block text"})
        else:
            predictions[image_id] = text
    return predictions, failures


def _walk_text_blocks(value: Any) -> list[str]:
    """Read textual blocks while preserving their source order and characters."""
    found: list[str] = []
    if isinstance(value, dict):
        if isinstance(value.get("text"), str) and value["text"]:
            found.append(value["text"])
        for key, child in value.items():
            if key != "text":
                found.extend(_walk_text_blocks(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_text_blocks(child))
    return found


def parse_page_text(payload: bytes | str | dict | list) -> str:
    """Parse one page's documented structured JSON, or a text primary output."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8-sig")
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return ""
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return normalize_prediction(payload)
    parts = _walk_text_blocks(payload)
    return normalize_prediction("\n".join(parts))


def parse_output_zip(archive_bytes: bytes) -> dict[int, str]:
    """Return page-number -> extracted text from Sarvam's output ZIP."""
    pages: dict[int, str] = {}
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for member in sorted(archive.namelist()):
            match = re.search(r"(?:metadata/)?page_(\d+)\.json$", member, flags=re.I)
            if match:
                page_num = int(match.group(1))
                pages[page_num] = parse_page_text(archive.read(member))
        if pages:
            return pages
        # Fallback to documented primary formats if metadata files are absent.
        for member in sorted(archive.namelist()):
            if Path(member).suffix.lower() in {".md", ".html", ".txt", ".json"}:
                raw = archive.read(member)
                text = parse_page_text(raw)
                if text:
                    pages[1] = text
                    break
    return pages


def make_page_archive(items: list[dict]) -> bytes:
    """Build a flat ZIP in stable input order, filenames are page order."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in items:
            path: Path = item["image_path"]
            suffix = path.suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg"}:
                raise ValueError(f"Sarvam batch input must be PNG/JPG: {path}")
            archive.write(path, arcname=f"{item['id']}{suffix}")
    return stream.getvalue()


def batch_items(items: list[dict], batch_size: int = MAX_PAGES_PER_JOB) -> list[list[dict]]:
    if not 1 <= batch_size <= MAX_PAGES_PER_JOB:
        raise ValueError(f"Sarvam batch size must be between 1 and {MAX_PAGES_PER_JOB}")
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def sdk_client(api_key: str | None = None) -> Any:
    """Lazily construct the official client; never logs or persists the key."""
    key = api_key or os.getenv("SARVAM_API_KEY")
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set")
    try:
        from sarvamai import SarvamAI
    except ImportError as exc:
        raise RuntimeError("missing optional dependency 'sarvamai'; install with: python -m pip install sarvamai") from exc
    return SarvamAI(api_subscription_key=key)


def object_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {"value": str(value)}


def parse_digitise_results(response: Any, items: list[dict]) -> tuple[dict[str, str], list[str]]:
    """Map installed SDK Digitise results (documents[].pages[]) to input IDs.

    Prefer explicit file names. If the API returns a single ZIP document, only
    use page order when the complete returned page count matches the batch.
    """
    data = object_dict(response)
    if data.get("type") != "digitise":
        raise ValueError(f"expected Digitise results, received type={data.get('type')!r}")

    # Current Document AI JSON results use documents[].pages[].blocks[].text.
    # Page numbers are local to this uploaded ZIP, so build this map from the
    # current batch order rather than treating page_num as globally unique.
    documents = data.get("documents") or []
    if any(
        isinstance(document, dict)
        and any(isinstance(page, dict) and "blocks" in page for page in (document.get("pages") or []))
        for document in documents
    ):
        batch_mapping = [
            {"image_id": str(item["id"]), "output_page": page_number}
            for page_number, item in enumerate(items, 1)
        ]
        predictions, failures = parse_saved_batch_response(data, batch_mapping)
        return predictions, [failure["id"] for failure in failures]

    item_by_id = {str(item["id"]): item for item in items}
    predictions: dict[str, str] = {}
    ordered_pages: list[tuple[int, str]] = []
    for document in data.get("documents", []) or []:
        if not isinstance(document, dict):
            continue
        filename = Path(str(document.get("file_name") or "")).name
        stem = Path(filename).stem
        pages = document.get("pages", []) or []
        for index, page in enumerate(pages, 1):
            if not isinstance(page, dict):
                continue
            page_number = page.get("page_number")
            try:
                page_number = int(page_number) if page_number is not None else index
            except (TypeError, ValueError):
                page_number = index
            text = parse_page_text(page.get("content") or "")
            if stem in item_by_id and len(pages) == 1:
                predictions[stem] = text
            else:
                ordered_pages.append((page_number, text))

    if len(ordered_pages) == len(items) and len(predictions) == 0:
        ordered_pages.sort(key=lambda entry: entry[0])
        predictions = {str(item["id"]): page[1] for item, page in zip(items, ordered_pages)}

    missing = [str(item["id"]) for item in items if str(item["id"]) not in predictions]
    return predictions, missing


def redact_secret(text: str, secret: str | None) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


def run_batch(client: Any, items: list[dict], raw_dir: Path, *, language: str = "mr-IN",
              poll_seconds: float = 5.0, timeout_seconds: float = 3600.0) -> tuple[dict[str, str], list[dict], float]:
    """Submit one archive and save raw artifacts. Must only be called after consent."""
    raw_dir.mkdir(parents=True, exist_ok=False)
    input_zip = make_page_archive(items)
    (raw_dir / "input.zip").write_bytes(input_zip)
    started = time.monotonic()
    failures: list[dict] = []
    predictions: dict[str, str] = {}
    try:
        with io.BytesIO(input_zip) as stream:
            job = build_digitise_call(client.doc_ai, stream, language=language)
        job_id = str(getattr(job, "job_id"))
        initial = object_dict(job)
        (raw_dir / "job_created.json").write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = client.doc_ai.get_status(job_id=job_id)
            status_data = object_dict(status)
            (raw_dir / "latest_status.json").write_text(json.dumps(status_data, ensure_ascii=False, indent=2), encoding="utf-8")
            state = str(status_data.get("status", "")).lower()
            if state in TERMINAL_STATES:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Sarvam job timed out while waiting for a terminal state")
            time.sleep(poll_seconds)
        if state not in {"completed", "partially_completed"}:
            raise RuntimeError(f"Sarvam job ended with status {state or 'unknown'}")
        results = client.doc_ai.get_results(job_id=job_id, format="json")
        results_data = object_dict(results)
        (raw_dir / "results_response.json").write_text(
            json.dumps(results_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        predictions, missing = parse_digitise_results(results, items)
        for item_id in missing:
            failures.append({"id": item_id, "reason": "no matching page in Digitise results"})
            predictions[item_id] = ""
    except Exception as exc:  # Preserve page-level failures without key leakage.
        message = redact_secret(f"{type(exc).__name__}: {exc}", os.getenv("SARVAM_API_KEY"))
        failures.extend({"id": item["id"], "reason": message} for item in items if item["id"] not in predictions)
        for item in items:
            predictions.setdefault(item["id"], "")
    return predictions, failures, time.monotonic() - started
