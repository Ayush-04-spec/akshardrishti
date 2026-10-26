from eval.crop_gt_alignment import align_ordered_crops, nfc, scorable_pairs


def _crop(crop_id, width, height=20, index=0):
    return {
        "sample_id": crop_id,
        "sequence_index": index,
        "height32_width_before_cap_px": str(width),
        "crop_width_px": str(width),
        "crop_height_px": str(height),
    }


def test_ordered_alignment_proposes_exact_unicode_ground_truth_spans():
    crops = [_crop("c0", 20, index=0), _crop("c1", 40, index=1)]
    result = align_ordered_crops(crops, "कक\nमहाराष्ट्र")
    assert [row["candidate_transcription"] for row in result] == ["कक", "महाराष्ट्र"]
    assert [row["matching_gt_span"] for row in result] == ["कक", "महाराष्ट्र"]
    assert all(row["alignment_status"] != "verified" for row in result)
    assert all(not row["eligible_for_scoring"] for row in result)


def test_equal_geometry_alternatives_are_marked_ambiguous():
    crops = [_crop("c0", 15, index=0), _crop("c1", 15, index=1)]
    result = align_ordered_crops(crops, "मराठी\nहिंदी")
    assert all(row["alignment_status"] == "ambiguous_match" for row in result)
    assert all(row["verification_status"] == "unverified" for row in result)
    assert all(row["eligible_for_scoring"] is False for row in result)


def test_nfc_normalization_preserves_meaningful_text_and_punctuation():
    text = "क़, महाराष्ट्र! १२३।"
    assert nfc(text) == text
    assert "," in nfc(text) and "!" in nfc(text) and " " in nfc(text) and "।" in nfc(text)


def test_only_explicitly_verified_crops_enter_scoring():
    rows = [
        {"candidate_transcription": "मजकूर", "prediction": "मजकूर",
         "alignment_status": "candidate_match", "verification_status": "unverified"},
        {"candidate_transcription": "मजकूर", "prediction": "मजकूर",
         "alignment_status": "verified", "verification_status": "unverified"},
        {"candidate_transcription": "मजकूर", "prediction": "मजकूर",
         "alignment_status": "verified", "verification_status": "verified"},
    ]
    assert scorable_pairs(rows) == [("मजकूर", "मजकूर")]
