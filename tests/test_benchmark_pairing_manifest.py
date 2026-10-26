import csv
import hashlib
from pathlib import Path

import pytest

from eval.run_benchmark import load_benchmark


FIELDS = [
    "image_id", "image_path", "same_stem_gt_path", "proposed_ground_truth_id",
    "proposed_ground_truth_path", "audit_status", "evidence", "reviewer_notes",
    "approved_for_evaluation",
]


def _fixture(tmp_path: Path):
    root = tmp_path / "benchmark"
    (root / "images").mkdir(parents=True)
    (root / "gt_text").mkdir()
    (root / "images" / "BM_001.png").write_bytes(b"image1")
    (root / "images" / "BM_002.png").write_bytes(b"image2")
    (root / "gt_text" / "BM_001.txt").write_text("same", encoding="utf-8")
    (root / "gt_text" / "BM_002.txt").write_text("cross", encoding="utf-8")
    manifest = tmp_path / "mapping.csv"
    rows = [
        ["BM_001", "images/BM_001.png", "gt_text/BM_001.txt", "BM_001", "gt_text/BM_001.txt", "confirmed", "matching", "checked", "true"],
        ["BM_002", "images/BM_002.png", "gt_text/BM_002.txt", "BM_001", "gt_text/BM_001.txt", "confirmed", "cross match", "checked", "true"],
    ]
    # A duplicate GT is invalid by design. Use distinct references except in the
    # dedicated cross-ID test, where the first row is changed to BM_002.
    rows[0][3:5] = ["BM_002", "gt_text/BM_002.txt"]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return root, manifest


def _read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_explicit_manifest_resolves_cross_id_reference(tmp_path):
    root, manifest = _fixture(tmp_path)
    rows = _read_rows(manifest)
    rows[0]["proposed_ground_truth_id"] = "BM_002"
    rows[0]["proposed_ground_truth_path"] = "gt_text/BM_002.txt"
    rows[1]["proposed_ground_truth_id"] = "BM_001"
    rows[1]["proposed_ground_truth_path"] = "gt_text/BM_001.txt"
    _write_rows(manifest, rows)

    items = load_benchmark(root, manifest)
    assert [(x["id"], x["gt_text"]) for x in items] == [("BM_001", "cross"), ("BM_002", "same")]


def test_explicit_manifest_accepts_existing_repo_relative_reference(tmp_path):
    root, manifest = _fixture(tmp_path)
    alternate = root.parent / "transcription_10b.txt"
    alternate.write_text("alternate source", encoding="utf-8")
    rows = _read_rows(manifest)
    rows[1]["proposed_ground_truth_id"] = "transcription_10b"
    rows[1]["proposed_ground_truth_path"] = "../transcription_10b.txt"
    _write_rows(manifest, rows)

    items = load_benchmark(root, manifest)
    assert [(x["id"], x["gt_text"]) for x in items] == [("BM_001", "cross"), ("BM_002", "alternate source")]


def test_missing_and_duplicate_manifest_entries_fail(tmp_path):
    root, manifest = _fixture(tmp_path)
    rows = _read_rows(manifest)
    _write_rows(manifest, rows[:1])
    with pytest.raises(ValueError, match="coverage mismatch"):
        load_benchmark(root, manifest)

    _write_rows(manifest, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate mapping"):
        load_benchmark(root, manifest)


def test_unapproved_mapping_fails_or_is_explicitly_excluded(tmp_path):
    root, manifest = _fixture(tmp_path)
    rows = _read_rows(manifest)
    rows[1]["audit_status"] = "needs_human_review"
    rows[1]["approved_for_evaluation"] = "false"
    _write_rows(manifest, rows)
    with pytest.raises(ValueError, match="unapproved/unresolved"):
        load_benchmark(root, manifest)
    items = load_benchmark(root, manifest, exclude_unapproved=True)
    assert [x["id"] for x in items] == ["BM_001"]


def test_unresolved_page_fails_safely(tmp_path):
    root, manifest = _fixture(tmp_path)
    rows = _read_rows(manifest)
    rows[1]["proposed_ground_truth_id"] = ""
    rows[1]["proposed_ground_truth_path"] = ""
    rows[1]["audit_status"] = "needs_human_review"
    rows[1]["approved_for_evaluation"] = "false"
    _write_rows(manifest, rows)
    with pytest.raises(ValueError, match="BM_002"):
        load_benchmark(root, manifest)


def test_legacy_loader_stays_same_stem_and_does_not_edit_references(tmp_path):
    root, manifest = _fixture(tmp_path)
    ref_paths = [root / "gt_text" / f"BM_{i:03}.txt" for i in (1, 2)]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ref_paths}

    legacy = load_benchmark(root)
    explicit = load_benchmark(root, manifest)

    assert [(x["id"], x["gt_text"]) for x in legacy] == [("BM_001", "same"), ("BM_002", "cross")]
    assert [x["gt_text"] for x in explicit] == ["cross", "same"]
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ref_paths}
    assert before == after
