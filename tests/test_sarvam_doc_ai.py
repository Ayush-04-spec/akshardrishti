from __future__ import annotations

import io
import json
import os
import zipfile

import pytest

from akshardrishti.recognize.sarvam_doc_ai import (
    batch_items,
    build_digitise_call,
    normalize_prediction,
    parse_output_zip,
    parse_page_text,
    parse_digitise_results,
    parse_saved_batch_response,
    extract_block_text,
    repair_verified_mojibake,
    run_batch,
    sdk_client,
)


class FakeDocAI:
    def __init__(self):
        self.kwargs = None

    def digitise(self, *, file, upload_ids=None, language=None, content_type=None,
                 model=None, request_options=None):
        self.kwargs = {
            "file": file, "upload_ids": upload_ids, "language": language,
            "content_type": content_type, "model": model,
            "request_options": request_options,
        }
        return {"job_id": "offline-test"}

    def get_results(self, *, job_id, format=None):
        self.result_call = {"job_id": job_id, "format": format}
        return {
            "type": "digitise", "job_id": job_id, "status": "completed",
            "usage": {"pages_total": 1, "pages_processed": 1, "pages_succeeded": 1, "pages_failed": 0},
            "documents": [{"file_name": "BM_001.png", "pages": [
                {"page_number": 1, "content": json.dumps({"blocks": [{"text": "महाराष्ट्र शासन"}]}, ensure_ascii=False)}
            ]}],
        }


def test_build_digitise_call_matches_installed_sdk_signature():
    api = FakeDocAI()
    file_obj = io.BytesIO(b"zip")
    result = build_digitise_call(api, file_obj, "pages.zip", "mr-IN")
    assert result == {"job_id": "offline-test"}
    assert api.kwargs == {
        "file": [("pages.zip", file_obj, "application/zip")],
        "upload_ids": None,
        "language": "mr-IN",
        "content_type": None,
        "model": None,
        "request_options": None,
    }


def test_rejects_unsupported_language():
    api = FakeDocAI()
    with pytest.raises(ValueError, match="unsupported configured Sarvam language"):
        build_digitise_call(api, io.BytesIO(), language="xx-IN")
    assert api.kwargs is None


def test_installed_sdk_signature_has_no_digitise_output_format_keyword():
    pytest.importorskip("sarvamai")
    import inspect
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key="offline-placeholder")
    parameters = inspect.signature(client.doc_ai.digitise).parameters
    assert "file" in parameters
    assert "language" in parameters
    assert "output_format" not in parameters
    assert "format" in inspect.signature(client.doc_ai.get_results).parameters


def test_parse_documented_blocks_and_preserve_marathi_text():
    data = {"blocks": [{"tag": "heading", "text": "महाराष्ट्र शासन"},
                       {"tag": "paragraph", "text": "कार्यालयीन आदेश।"}]}
    assert parse_page_text(data) == "महाराष्ट्र शासन कार्यालयीन आदेश।"


def test_unicode_normalization_matches_scoring_form():
    # Devanagari nukta and canonical composition are handled as by the scorer.
    assert normalize_prediction("क\u093c\u200d  शासन") == normalize_prediction("क़ शासन")


def test_parse_output_archive_by_page_number():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("metadata/page_002.json", json.dumps({"blocks": [{"text": "दुसरे पान"}]}, ensure_ascii=False))
        archive.writestr("metadata/page_001.json", json.dumps({"blocks": [{"text": "पहिले पान"}]}, ensure_ascii=False))
    assert parse_output_zip(stream.getvalue()) == {1: "पहिले पान", 2: "दुसरे पान"}


def test_parse_sdk_digitise_response_maps_document_filenames():
    response = {
        "type": "digitise", "documents": [
            {"file_name": "BM_002.png", "pages": [{"page_number": 1, "content": "दुसरा"}]},
            {"file_name": "BM_001.png", "pages": [{"page_number": 1, "content": "पहिला"}]},
        ],
    }
    items = [{"id": "BM_001"}, {"id": "BM_002"}]
    predictions, missing = parse_digitise_results(response, items)
    assert predictions == {"BM_002": "दुसरा", "BM_001": "पहिला"}
    assert missing == []


def test_legacy_parse_digitise_results_extracts_current_blocks_schema():
    response = {"type": "digitise", "status": "completed", "documents": [{"filename": "pages.zip", "pages": [
        {"page_num": 1, "status": "completed", "blocks": [
            {"reading_order": 2, "text": "जग"},
            {"reading_order": 1, "text": "नमस्कार"},
        ]},
    ]}]}
    predictions, missing = parse_digitise_results(response, [{"id": "BM_001"}])
    assert predictions == {"BM_001": "नमस्कार\nजग"}
    assert missing == []


def test_parse_sdk_digitise_zip_pages_in_page_order_only_when_complete():
    response = {
        "type": "digitise", "documents": [{"file_name": "pages.zip", "pages": [
            {"page_number": 2, "content": "दुसरे"},
            {"page_number": 1, "content": "पहिले"},
        ]}],
    }
    items = [{"id": "BM_001"}, {"id": "BM_002"}]
    predictions, missing = parse_digitise_results(response, items)
    assert predictions == {"BM_001": "पहिले", "BM_002": "दुसरे"}
    assert missing == []

    incomplete = {"type": "digitise", "documents": [{"file_name": "pages.zip", "pages": [
        {"page_number": 1, "content": "unmatched"},
    ]}]}
    predictions, missing = parse_digitise_results(incomplete, items)
    assert predictions == {}
    assert missing == ["BM_001", "BM_002"]


def test_extract_blocks_in_reading_order_and_keep_newlines():
    blocks = [
        {"reading_order": 3, "layout_tag": "paragraph", "text": "अंतिम भाग"},
        {"reading_order": 1, "layout_tag": "header", "text": "महाराष्ट्र शासन\nकार्यालय"},
        {"reading_order": 2, "layout_tag": "paragraph", "text": "पहिला परिच्छेद"},
        {"reading_order": 4, "text": "   "},
    ]
    assert extract_block_text(blocks) == "महाराष्ट्र शासन\nकार्यालय\nपहिला परिच्छेद\nअंतिम भाग"


def test_extract_table_html_as_ordered_cells_and_rows():
    blocks = [{
        "reading_order": 1,
        "layout_tag": "table",
        "text": "<table><tr><th>नाव</th><th>दिनांक</th></tr><tr><td>राम</td><td>२०२६</td></tr></table>",
    }]
    assert extract_block_text(blocks) == "नाव\tदिनांक\nराम\t२०२६"


def test_missing_reading_order_uses_spatial_then_stable_fallback():
    blocks = [
        {"text": "खाली", "bbox_norm": [0.1, 0.8, 0.9, 0.9]},
        {"text": "वर", "bbox_norm": [0.3, 0.1, 0.8, 0.2]},
        {"text": "त्याच ओळीत", "bbox_norm": [0.1, 0.1, 0.2, 0.2]},
    ]
    assert extract_block_text(blocks) == "त्याच ओळीत\nवर\nखाली"


def test_encoding_repair_repairs_reversible_mojibake_and_preserves_native_unicode():
    original = "महाराष्ट्र शासन"
    mojibake = original.encode("utf-8").decode("latin-1")
    assert repair_verified_mojibake(mojibake) == original
    assert repair_verified_mojibake(original) == original
    english = "Corrigendum – III"
    assert repair_verified_mojibake(english) == english


def test_saved_batch_mapping_uses_batch_local_page_numbers():
    mapping_a = [
        {"image_id": "BM_001", "output_page": 1},
        {"image_id": "BM_002", "output_page": 2},
    ]
    mapping_b = [
        {"image_id": "BM_019", "output_page": 1},
        {"image_id": "BM_020", "output_page": 2},
    ]
    response = {"type": "digitise", "status": "completed", "documents": [{"pages": [
        {"page_num": 2, "status": "completed", "blocks": [{"reading_order": 1, "text": "दुसरे"}]},
        {"page_num": 1, "status": "completed", "blocks": [{"reading_order": 1, "text": "पहिले"}]},
    ]}]}
    first, first_failures = parse_saved_batch_response(response, mapping_a)
    second, second_failures = parse_saved_batch_response(response, mapping_b)
    assert first == {"BM_001": "पहिले", "BM_002": "दुसरे"}
    assert second == {"BM_019": "पहिले", "BM_020": "दुसरे"}
    assert first_failures == second_failures == []


def test_saved_batch_parser_handles_empty_and_failed_pages():
    response = {"type": "digitise", "status": "partially_completed", "documents": [{"pages": [
        {"page_num": 1, "status": "completed", "blocks": []},
        {"page_num": 2, "status": "failed", "blocks": [{"text": "must not score"}]},
    ]}]}
    predictions, failures = parse_saved_batch_response(response, [
        {"image_id": "BM_001", "output_page": 1},
        {"image_id": "BM_002", "output_page": 2},
    ])
    assert predictions == {"BM_001": "", "BM_002": ""}
    assert {failure["id"] for failure in failures} == {"BM_001", "BM_002"}


@pytest.mark.parametrize("mapping", [
    [{"image_id": "BM_001", "output_page": 1}],
    [{"image_id": "BM_001", "output_page": 1}, {"image_id": "BM_002", "output_page": 1}],
    [{"image_id": "BM_001", "output_page": 1}, {"image_id": "BM_001", "output_page": 2}],
])
def test_saved_batch_parser_rejects_missing_or_duplicate_mappings(mapping):
    response = {"type": "digitise", "status": "completed", "documents": [{"pages": [
        {"page_num": 1, "status": "completed", "blocks": [{"text": "x"}]},
        {"page_num": 2, "status": "completed", "blocks": [{"text": "y"}]},
    ]}]}
    with pytest.raises(ValueError):
        parse_saved_batch_response(response, mapping)


def test_saved_batch_parser_rejects_duplicate_and_missing_response_pages():
    mapping = [{"image_id": "BM_001", "output_page": 1}, {"image_id": "BM_002", "output_page": 2}]
    duplicate = {"type": "digitise", "documents": [{"pages": [
        {"page_num": 1, "blocks": []}, {"page_num": 1, "blocks": []},
    ]}]}
    missing = {"type": "digitise", "documents": [{"pages": [
        {"page_num": 1, "blocks": []},
    ]}]}
    with pytest.raises(ValueError, match="duplicate batch-local page_num"):
        parse_saved_batch_response(duplicate, mapping)
    with pytest.raises(ValueError, match="batch page mapping mismatch"):
        parse_saved_batch_response(missing, mapping)


def test_missing_key_errors_without_constructing_client(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SARVAM_API_KEY is not set"):
        sdk_client()


def test_batch_size_enforces_conservative_documented_limit():
    rows = [{"id": str(i)} for i in range(23)]
    assert [len(x) for x in batch_items(rows)] == [10, 10, 3]
    with pytest.raises(ValueError, match="between 1 and 10"):
        batch_items(rows, 11)


def test_terminal_job_failure_is_recorded_per_page_without_live_api(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"offline fixture")

    class FailedJob:
        job_id = "offline-job"

    class FailedDocAI:
        def digitise(self, **kwargs):
            return FailedJob()

        def get_status(self, **kwargs):
            return {"status": "failed", "usage": {"pages_failed": 1}}

    class FakeClient:
        doc_ai = FailedDocAI()

    rows = [{"id": "BM_001", "image_path": image}]
    predictions, failures, elapsed = run_batch(FakeClient(), rows, tmp_path / "raw")
    assert predictions == {"BM_001": ""}
    assert failures == [{"id": "BM_001", "reason": "RuntimeError: Sarvam job ended with status failed"}]
    assert elapsed >= 0
    assert (tmp_path / "raw" / "latest_status.json").is_file()


def test_exception_message_redacts_api_key(tmp_path, monkeypatch):
    secret = "offline-secret-that-must-not-leak"
    monkeypatch.setenv("SARVAM_API_KEY", secret)
    image = tmp_path / "page.png"
    image.write_bytes(b"offline fixture")

    class FailedDocAI:
        def digitise(self, **kwargs):
            raise RuntimeError(f"response echoed {secret}")

    class FakeClient:
        doc_ai = FailedDocAI()

    rows = [{"id": "BM_001", "image_path": image}]
    predictions, failures, _ = run_batch(FakeClient(), rows, tmp_path / "raw")
    assert predictions == {"BM_001": ""}
    assert secret not in failures[0]["reason"]
    assert "[REDACTED]" in failures[0]["reason"]


def test_success_path_polls_then_fetches_json_results(tmp_path):
    image = tmp_path / "BM_001.png"
    image.write_bytes(b"offline fixture")

    class FakeJob:
        job_id = "offline-job"
        status = "pending"

    class SuccessfulDocAI(FakeDocAI):
        def digitise(self, **kwargs):
            # Strict call contract checked in FakeDocAI's dedicated test.
            self.digitise_kwargs = kwargs
            return FakeJob()

        def get_status(self, *, job_id):
            return {"job_id": job_id, "status": "completed", "usage": {"pages_processed": 1}}

    api = SuccessfulDocAI()

    class FakeClient:
        doc_ai = api

    rows = [{"id": "BM_001", "image_path": image}]
    predictions, failures, _ = run_batch(FakeClient(), rows, tmp_path / "raw")
    assert predictions == {"BM_001": "महाराष्ट्र शासन"}
    assert failures == []
    assert api.result_call == {"job_id": "offline-job", "format": "json"}
    assert (tmp_path / "raw" / "results_response.json").is_file()
